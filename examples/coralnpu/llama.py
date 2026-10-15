"""Llama 3.2 3B operator/dataflow timing workload, without checkpoint weights.

This is a compact performance workload, not numerical inference. PE geometry
comes from MatrixSpec; operator streaming and vector costs remain assumptions
until compared with an end-to-end implementation of the same mapping.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, replace
import io
import json
from math import ceil
from pathlib import Path
import re
import subprocess
import time

from lpnlang_mlir import NetBuilder
from .matrix import MatrixCommand, MatrixSpec
from .model import MemoryPort, MemorySpec


SOURCE = "https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/sku_list.py"


@dataclass(frozen=True)
class LlamaConfig:
  hidden: int = 3072
  intermediate: int = 8192
  layers: int = 28
  heads: int = 24
  kv_heads: int = 8
  vocabulary: int = 128256
  max_context: int = 131072

  def __post_init__(self):
    if min(asdict(self).values()) < 1 or self.hidden % self.heads or self.heads % self.kv_heads:
      raise ValueError("invalid Llama dimensions or grouped-query attention geometry")

  @property
  def head_dim(self):
    return self.hidden // self.heads

  @property
  def kv_dim(self):
    return self.kv_heads * self.head_dim

  @property
  def parameters(self):
    d, f = self.hidden, self.intermediate
    # Input embedding and LM head share weights; norms have no biases.
    return self.vocabulary*d + self.layers*(2*d*d + 2*d*self.kv_dim + 3*d*f + 2*d) + d


@dataclass(frozen=True)
class Operator:
  name: str
  phase: int
  kind: str
  dependencies: tuple
  cycles: int
  read_bytes: int
  write_bytes: int
  macs: int = 0
  matrix_commands: int = 0
  repetitions: int = 1


class LlamaWorkload:
  def __init__(self, prompt_tokens=128, generate_tokens=4, config=None,
               memory=None, local_bandwidth=16, stream_bytes=4096, buffers=2,
               fold_layers=True):
    self.config = cfg = config or LlamaConfig()
    if min(prompt_tokens, generate_tokens, stream_bytes, buffers) < 1:
      raise ValueError("token counts, stream size, and buffer credits must be positive")
    if prompt_tokens + generate_tokens - 1 > cfg.max_context:
      raise ValueError("requested sequence exceeds the model context limit")
    self.local = MemorySpec(0, local_bandwidth)
    self.memory_spec = memory or MemorySpec(64, 16, 16)
    self.prompt_tokens, self.generate_tokens = prompt_tokens, generate_tokens
    self.stream_bytes, self.buffers = stream_bytes, buffers
    self.operators = []
    self.feedback = {}
    self.fold_layers = fold_layers
    self.phase_ends = []
    previous = None
    for phase in range(generate_tokens):
      # Prefill logits produce the first generated token. Subsequent decode
      # steps append one KV entry, then attend to all entries including it.
      length = prompt_tokens if phase == 0 else 1
      context = prompt_tokens + phase
      previous = self._phase(phase, length, context, previous)
      self.phase_ends.append(previous)
    self._build_net()

  def _append(self, phase, name, kind, dependencies, cycles, reads=0, writes=0,
              macs=0, commands=0):
    index = len(self.operators)
    self.operators.append(Operator(name, phase, kind,
        tuple(dep for dep in dependencies if dep is not None), max(1, int(cycles)),
        int(reads), int(writes), int(macs), int(commands)))
    return index

  def _vector(self, phase, name, dependencies, elements, passes, reads, writes,
              overhead=0):
    return self._append(phase, name, "vector", dependencies,
                        ceil(elements/8)*passes + overhead, reads, writes)

  def _gemm(self, phase, name, dependencies, m, n, k, repeats=1,
            read_bytes=None, write_bytes=None):
    spec = MatrixSpec()
    groups = ceil(k/2)  # Native BF16 weights and activations, FP32 accumulation.
    cycles = commands = 0
    for row in range(0, m, 16):
      for col in range(0, n, 16):
        command = MatrixCommand(m=min(16, m-row), n=min(16, n-col),
                                k=min(2, k), dtype="bf16")
        cycles += 4 + (groups-1)*spec.reuse_cycles(command) + spec.latency(command)
        commands += groups
    # M-major, output-stationary mapping: stream each input panel per output
    # tile. No full weight matrix is assumed to fit in the local SRAM.
    local_reads = 2*(m*k*ceil(n/16) + n*k*ceil(m/16))*repeats
    local_bytes = local_reads + m*n*6*repeats  # FP32 result plus BF16 conversion.
    compute = max(cycles*repeats + ceil(m*n*repeats/8),
                  self.local.service_cycles(local_bytes))
    return self._append(phase, name, "matrix", dependencies, compute,
        local_reads if read_bytes is None else read_bytes,
        m*n*2*repeats if write_bytes is None else write_bytes,
        m*n*k*repeats, commands*repeats)

  def _phase(self, phase, s, context, previous):
    cfg = self.config
    d, f, kv = cfg.hidden, cfg.intermediate, cfg.kv_dim
    prefix = f"p{phase}"
    hidden = self._append(phase, f"{prefix}.embedding", "vector", (previous,),
                          ceil(s*d/8), s*d*2, s*d*2)
    for layer in range(1 if self.fold_layers else cfg.layers):
      name = f"{prefix}.l{layer}"
      first_operator = len(self.operators)
      norm = self._vector(phase, name+".attention_norm", (hidden,), s*d, 6,
                          s*d*2+d*2, s*d*2, s*36)
      q = self._gemm(phase, name+".q", (norm,), s, d, d)
      k = self._gemm(phase, name+".k", (norm,), s, kv, d)
      v = self._gemm(phase, name+".v", (norm,), s, kv, d)
      q_rope = self._vector(phase, name+".q_rope", (q,), s*d, 6, s*d*2, s*d*2)
      k_rope = self._vector(phase, name+".k_rope", (k,), s*kv, 6, s*kv*2, s*kv*2)
      key_cache = self._append(phase, name+".key_append", "vector", (k_rope,),
                               ceil(s*kv/8), s*kv*2, s*kv*2)
      value_cache = self._append(phase, name+".value_append", "vector", (v,),
                                 ceil(s*kv/8), s*kv*2, s*kv*2)
      # Dense causal attention: compute the score matrix then apply a mask.
      # GQA shares K/V storage across three query heads. The mapping assumes
      # grouped head traversal can reuse a streamed K/V panel locally.
      scores = cfg.heads*s*context
      qk = self._gemm(phase, name+".qk", (q_rope,key_cache), s, context, cfg.head_dim,
          repeats=cfg.heads, read_bytes=2*(s*d+context*kv), write_bytes=scores*2)
      softmax = self._vector(phase, name+".softmax", (qk,), scores, 16,
                             scores*2, scores*2, cfg.heads*s*36)
      av = self._gemm(phase, name+".attention_value", (softmax,value_cache), s, cfg.head_dim,
          context, repeats=cfg.heads, read_bytes=2*(scores+context*kv), write_bytes=s*d*2)
      output = self._gemm(phase, name+".o", (av,), s, d, d)
      residual = self._vector(phase, name+".attention_residual", (hidden, output),
                              s*d, 1, s*d*4, s*d*2)
      norm = self._vector(phase, name+".ffn_norm", (residual,), s*d, 6,
                          s*d*2+d*2, s*d*2, s*36)
      gate = self._gemm(phase, name+".gate", (norm,), s, f, d)
      up = self._gemm(phase, name+".up", (norm,), s, f, d)
      activation = self._vector(phase, name+".silu_mul", (gate, up), s*f, 12,
                                s*f*4, s*f*2)
      down = self._gemm(phase, name+".down", (activation,), s, d, f)
      hidden = self._vector(phase, name+".ffn_residual", (residual, down),
                            s*d, 1, s*d*4, s*d*2)
      if self.fold_layers:
        self.feedback[hidden] = (first_operator,cfg.layers)
        for index in range(first_operator,hidden+1):
          self.operators[index] = replace(self.operators[index],repetitions=cfg.layers)
    norm = self._vector(phase, prefix+".final_norm", (hidden,), d, 6, d*4, d*2, 36)
    # Generation only needs logits for the final prompt position.
    logits = self._gemm(phase, prefix+".lm_head", (norm,), 1, cfg.vocabulary, d)
    return self._vector(phase, prefix+".argmax", (logits,), cfg.vocabulary, 2,
                        cfg.vocabulary*2, 4, 36)

  def _build_net(self):
    self.net = net = NetBuilder("llama_3_2_3b")
    memory = MemoryPort(net, "dram", self.memory_spec)
    staging = net.place("stream_buffer_credits", initial_tokens=self.buffers)
    units = {kind: net.place(kind+"_engine", initial_tokens=1)
             for kind in ("matrix", "vector")}
    edges = {(dep, idx): net.place(f"edge_{dep}_{idx}")
             for idx, op in enumerate(self.operators) for dep in op.dependencies}
    outputs = {idx: [place for (source, _), place in edges.items() if source == idx]
               for idx in range(len(self.operators))}
    self.finished = net.place("llama_completed", observable=True)
    self.phase_markers = [net.place(f"phase_{p}_completed", observable=True)
                          for p in range(self.generate_tokens)]
    for idx, op in enumerate(self.operators):
      pending = net.place(f"pending_{idx}", initial_tokens=1)
      loaded = net.place(f"loaded_{idx}")
      computed = net.place(f"computed_{idx}")
      inputs = [edges[(dep, idx)] for dep in op.dependencies]
      targets = outputs[idx]
      engine = units[op.kind]
      read_service = memory.spec.service_cycles(op.read_bytes) if op.read_bytes else 0
      read_delay = memory.spec.latency+read_service if op.read_bytes else 0
      first = (memory.spec.latency + memory.spec.service_cycles(
          min(op.read_bytes, self.stream_bytes))) if op.read_bytes else 0
      # Fold steady streaming into the service interval. Compute may start
      # after the first panel, but cannot finish before the stream drains.
      compute_delay = max(op.cycles, read_delay-first)
      write_service = memory.spec.service_cycles(op.write_bytes) if op.write_bytes else 0
      write_delay = memory.spec.latency+write_service if op.write_bytes else 0
      phase = op.phase if idx == self.phase_ends[op.phase] else None
      marker = self.phase_markers[op.phase]
      final = idx == self.phase_ends[-1]
      feedback = None
      if idx in self.feedback:
        first_operator,repeats = self.feedback[idx]
        first_dependency = self.operators[first_operator].dependencies[0]
        # The hidden-state skip connection and norm both consume the next
        # layer's input. Renew every fanout edge when closing the loop.
        feedback = (outputs[first_dependency],repeats)
      self._add_operator(idx, op, pending, loaded, computed, inputs, targets,
          engine, memory, staging, read_service, read_delay, first, compute_delay,
          write_service, write_delay, phase, marker, final, feedback)

  def _add_operator(self, idx, op, pending, loaded, computed, inputs, targets,
                    engine, memory, staging, read_service, read_delay, first,
                    compute_delay, write_service, write_delay, phase, marker, final,
                    feedback):
    net = self.net
    first_input = inputs[0] if inputs else None
    other_inputs = inputs[1:]
    repeat = op.repetitions > 1
    feedback_targets,iterations = feedback if feedback else (None,0)
    reset_layer = op.name.endswith(".embedding")

    @net.transition(f"load_{idx}")
    def load():
      take(pending)
      if first_input is not None:
        token = take(first_input)
      else:
        token = create()
      if reset_layer:
        token = token.set("layer",0)
      for dependency in other_inputs:
        take(dependency)
      take(staging)
      bus = take(memory.bus)
      slot = take(memory.slots)
      emit(memory.bus, bus, delay=float(read_service))
      emit(memory.slots, slot, delay=float(read_delay))
      emit(loaded, token, delay=float(first))

    @net.transition(f"compute_{idx}")
    def compute():
      token = take(loaded)
      unit = take(engine)
      emit(engine, unit, delay=float(compute_delay))
      emit(staging, create(), delay=float(compute_delay))
      emit(computed, token, delay=float(compute_delay))

    @net.transition(f"store_{idx}")
    def store():
      token = take(computed)
      bus = take(memory.bus)
      slot = take(memory.slots)
      emit(memory.bus, bus, delay=float(write_service))
      emit(memory.slots, slot, delay=float(write_delay))
      if repeat:
        emit(pending, create())
      if feedback_targets is not None:
        layer = token.get("layer") + 1
        if layer < iterations:
          for target in feedback_targets:
            emit(target, token.set("layer",layer), delay=float(write_delay))
        else:
          for target in targets:
            emit(target, token, delay=float(write_delay))
      else:
        for target in targets:
          emit(target, token, delay=float(write_delay))
      if phase is not None:
        emit(marker, token, delay=float(write_delay))
      if final:
        emit(self.finished, token, delay=float(write_delay))

    if phase is not None:
      observed = net.place(f"observed_phase_{phase}")

      @net.transition(f"phase_done_{phase}")
      def phase_done():
        emit(observed, take(marker))

  def build(self):
    return self.net.build()

  def report(self, phase_cycles):
    return {"model": "meta-llama/Llama-3.2-3B", "architecture_source": SOURCE,
        "configuration": asdict(self.config), "parameters": self.config.parameters,
        "dtype": "bf16", "prompt_tokens": self.prompt_tokens,
        "generated_tokens": self.generate_tokens,
        "operators": sum(op.repetitions for op in self.operators),
        "operator_templates": len(self.operators),
        "layers_use_feedback_tokens": self.fold_layers,
        "macs": sum(op.macs*op.repetitions for op in self.operators),
        "matrix_commands": sum(op.matrix_commands*op.repetitions for op in self.operators),
        "dram_bytes": sum((op.read_bytes+op.write_bytes)*op.repetitions for op in self.operators),
        "kv_cache_bytes": self.config.layers*2*(self.prompt_tokens+self.generate_tokens-1)*self.config.kv_dim*2,
        "dram": asdict(self.memory_spec), "local_bytes_per_cycle": self.local.bytes_per_cycle,
        "stream_panel_bytes": self.stream_bytes, "stream_buffers": self.buffers,
        "phase_completion_cycles": phase_cycles,
        "time_to_first_token_cycles": phase_cycles[0], "total_cycles": phase_cycles[-1],
        "decode_cycles": [b-a for a,b in zip(phase_cycles,phase_cycles[1:])],
        "numerical_inference": False, "accuracy_target_relative_error": 0.05,
        "end_to_end_verilator_validated": False,
        "accuracy": {"status": "unmeasured", "passed": False},
        "assumptions": ["operator streams reserve the shared DRAM bandwidth until drained",
            "finite stream credits represent in-flight panels, not whole tensors resident in SRAM",
            "M-major BF16 matrix tiling with repeated panel traffic; no full weight caching",
            "GQA shares streamed KV panels across query-head groups; attention materializes masked scores",
            "vector pass counts and whole-operator streaming overlap are not RTL-calibrated",
            "sampling uses argmax; checkpoint weights, tokenizer, and numerical arithmetic are not executed"]}

  def simulate(self):
    simulator = self.net.python_simulator()
    log = io.StringIO()
    with redirect_stdout(log):
      simulator.run(max_time=1e18, debug=True)
    if len(simulator.place_dicts(self.finished)) != 1:
      raise RuntimeError("Llama workload did not complete")
    return self.report(self.phase_times(log.getvalue()))

  def phase_times(self, trace):
    phase_cycles = [None]*self.generate_tokens
    for match in re.finditer(r"\[t=([0-9.]+)\] fired phase_done_([0-9]+)", trace):
      phase_cycles[int(match.group(2))] = float(match.group(1))
    if any(value is None for value in phase_cycles):
      raise RuntimeError("Llama workload did not observe every phase completion")
    return phase_cycles


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--prompt-tokens", type=int, default=128)
  parser.add_argument("--generate-tokens", type=int, default=4)
  parser.add_argument("--memory-latency", type=int, default=64)
  parser.add_argument("--memory-bandwidth", type=float, default=16)
  parser.add_argument("--memory-outstanding", type=int, default=16)
  parser.add_argument("--buffers", type=int, default=2)
  parser.add_argument("--emit", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--lpn-compiler", type=Path)
  args = parser.parse_args()
  workload = LlamaWorkload(args.prompt_tokens, args.generate_tokens,
      memory=MemorySpec(args.memory_latency,args.memory_bandwidth,args.memory_outstanding),
      buffers=args.buffers)
  if args.emit:
    args.emit.write_text(workload.build())
    if not args.output and not args.lpn_compiler:
      return
  if not args.emit or not args.lpn_compiler:
    parser.error("running the full workload requires --emit and --lpn-compiler")
  binary = args.emit.with_suffix(".native")
  before = time.perf_counter()
  subprocess.run([str(args.lpn_compiler.resolve()),str(args.emit),"-o",str(binary)],check=True)
  compile_seconds = time.perf_counter()-before
  before = time.perf_counter()
  result = subprocess.run([str(binary.resolve()),"--max-time=1000000000000000000","--stats","--trace"],
                          capture_output=True,text=True,check=True)
  seconds = time.perf_counter()-before
  report = workload.report(workload.phase_times(result.stdout))
  report.update(compile_wall_seconds=compile_seconds,native_wall_seconds=seconds)
  match = re.search(r"Simulation finished at t=([0-9.]+)",result.stdout)
  if not match or float(match.group(1)) != report["total_cycles"]:
    raise RuntimeError("native Llama final time disagrees with phase completion")
  if "place llama_completed: 1 token(s)" not in result.stdout:
    raise RuntimeError("native Llama workload did not complete")
  stats = re.search(r"Guard checks: (\d+); body executions: (\d+); rollbacks: (\d+)",result.stdout)
  if not stats or int(stats.group(3)):
    raise RuntimeError("native Llama workload had unexpected rollback")
  report["native_stats"] = dict(zip(("guard_checks","body_executions","rollbacks"),map(int,stats.groups())))
  text = json.dumps(report,indent=2)+"\n"
  if args.output:
    args.output.write_text(text)
  print(text,end="")


if __name__ == "__main__":
  main()
