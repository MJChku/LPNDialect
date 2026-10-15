"""Resumable trained-model generation with identical LPN/Verilator inputs.

Every token traverses all transformer layers, updates the persistent KV cache,
and computes real vocabulary logits. Numerical ISA execution is timed separately
from the native LPN timing simulation. No RTL timestamps enter the LPN model.
"""

import argparse
import fcntl
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from . import REVISION
from .benchmark import native_observation,write_inputs
from .compact import CompactOpenTPU
from .model import Timing
from .workload import execute,use_checkout


def digest(path):
  with Path(path).open("rb") as f:
    return hashlib.file_digest(f,"sha256").hexdigest()


def atomic_json(path,value):
  path = Path(path)
  pending = path.with_suffix(path.suffix+".tmp")
  with pending.open("w") as f:
    json.dump(value,f,indent=2)
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())
  pending.replace(path)


class CheckedBackend:
  """Host argument and memory interface expected by upstream Engine."""
  args = True
  generates = False
  chains = False

  def __init__(self,cfg,images,options):
    import numpy as np
    self.cfg,self.options = cfg,options
    self.drams = [np.asarray(im,np.uint8).copy() for im in images]
    self.records = []

  def write(self,s,addr,data):
    import numpy as np
    data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
    self.drams[s][addr:addr+len(data)] = data

  def read(self,s,addr,nbytes):
    return self.drams[s][addr:addr+nbytes].copy()

  def run(self,programs,args=None):
    import numpy as np
    from opentpu import isa as I,rtlsim,waits as W
    out = self.options.output
    current = out/"current"
    current.mkdir(exist_ok=True)
    # The files are written once by the same writer as RTL; retained while a
    # token runs, replaced on the next token. No per-token copies of weights.
    write_inputs(current,self.cfg,programs,self.drams,args)
    hashes = {name:digest(current/name) for s in range(self.cfg.S)
              for name in (f"prog_{s}.hex",f"dram_{s}.bin")}
    before = time.perf_counter()
    w = execute(self.cfg,programs,self.drams,args,hashes)
    front = time.perf_counter()-before
    timing = Timing(batch_rows=self.options.batch_rows,
                    dram_bytes_per_cycle=min(128,self.cfg.D))
    before = time.perf_counter()
    model = CompactOpenTPU(w,timing)
    ir,binary = current/"model.mlir",current/"model.native"
    ir.write_text(model.build())
    build = time.perf_counter()-before
    before = time.perf_counter()
    result = subprocess.run([str(self.options.lpn_compiler),str(ir),"-o",str(binary)],
                            capture_output=True,text=True)
    (current/"compile.log").write_text(result.stdout+result.stderr)
    result.check_returncode()
    compile_seconds = time.perf_counter()-before
    cycles,wall,process,log = native_observation(binary)
    (current/"lpn.log").write_text(log)
    print(json.dumps({"stage":"rtl","step":len(self.records),"predicted_cycles":cycles,
                      "frontend_seconds":front,"compile_seconds":compile_seconds}),flush=True)
    before = time.perf_counter()
    trace = len(self.records)==0
    # Build outside the wall timer used for the RTL process measurement.
    exe = rtlsim.build_top(self.cfg,dram_lat=timing.dram_latency)
    rtl_build = time.perf_counter()-before
    before = time.perf_counter()
    # Override only the process timeout, retaining upstream cleanup of child
    # process groups on failure. The event-driven LPN scheduler is untouched.
    original = rtlsim.run_sim
    captured = []
    def run_sim(command,timeout=None):
      result = original(command,timeout=self.options.rtl_timeout)
      captured.append(result.stdout+result.stderr)
      return result
    rtlsim.run_sim = run_sim
    try:
      drams,tmems,stats = rtlsim.run(self.cfg,programs,self.drams,args=args,
          dram_lat=timing.dram_latency,max_cycles=1<<40,trace=trace,
          keep=current,plusargs=["+lpn_wall"])
    finally:
      rtlsim.run_sim = original
    rtl_process = time.perf_counter()-before
    output = captured[-1]
    timer = re.search(r"LPN_WALL seconds=([0-9.]+) cycles=(\d+)",output)
    if not timer or int(timer[2])!=stats["cycles"]:
      raise RuntimeError("passive RTL wall timer missing or inconsistent; apply observer patch")
    if stats["instructions"]!=[len(c) for c in w.commands]:
      raise RuntimeError("functional and RTL instruction counts differ")
    for s,golden in enumerate(w.machine.slices):
      if not np.array_equal(drams[s],golden.dram) or not np.array_equal(tmems[s],golden.tmem):
        raise RuntimeError(f"numerical RTL mismatch in slice {s}")
      for name in (f"prog_{s}.hex",f"dram_{s}.bin"):
        if digest(current/name)!=hashes[name]:
          raise RuntimeError(f"input differs between engines: {name}")
      if trace:
        actual = [int(pc) for pc in re.findall(rf"T{s} D c=\d+ s=\d+ pc=(\d+)",output)]
        expected = [c.pc for c in w.commands[s] if c.instruction.op in W.UNIT]
        if actual!=expected:
          raise RuntimeError("observed RTL command path differs")
    error = cycles/stats["cycles"]-1
    record = dict(step=len(self.records),rtl_cycles=stats["cycles"],lpn_cycles=cycles,
        relative_cycle_error=error,accuracy_within_5_percent=abs(error)<=0.05,
        all_dram_and_tmem_bit_exact=True,identical_input_files=True,
        observed_command_path_checked=trace,zero_rollbacks=True,
        rtl_traced=trace,
        instructions=stats["instructions"],input_sha256=hashes,
        functional_frontend_seconds=front,model_build_seconds=build,
        native_compile_seconds=compile_seconds,lpn_core_seconds=wall,
        lpn_process_seconds=process,rtl_core_seconds=float(timer[1]),
        rtl_process_seconds=rtl_process,rtl_build_seconds=rtl_build,
        baseline_binary_sha256=digest(exe),timing=asdict(timing),
        numerical_validation="independent ISA versus RTL; INT8 checkpoint execution")
    record["core_speedup"] = record["rtl_core_seconds"]/wall
    record["single_use_lpn_seconds"] = front+build+compile_seconds+process
    if not record["accuracy_within_5_percent"]:
      atomic_json(out/"failed-step.json",record)
      raise RuntimeError(f"timing error {error:.2%} exceeds 5%; see failed-step.json")
    self.records.append(record)
    # Upstream images can be shorter than configured DRAM. Keep their original
    # extent; all output and KV allocations lie inside that extent.
    self.drams = [d[:len(im)].copy() for d,im in zip(drams,self.drams)]
    return record


def checkpoint(out,backend,state,logits):
  import numpy as np
  # Alternate two bounded snapshots. The old manifest remains valid until the
  # new complete snapshot is flushed and atomically selected by state.json.
  slot = state["position"]%2
  target = out/f"checkpoint-{slot}.npz"
  temporary = out/f"checkpoint-{slot}.tmp"
  with temporary.open("wb") as f:
    np.savez(f,logits=logits,**{f"dram_{s}":d for s,d in enumerate(backend.drams)})
    f.flush()
    os.fsync(f.fileno())
  temporary.replace(target)
  state.update(snapshot=target.name,snapshot_sha256=digest(target),records=backend.records)
  atomic_json(out/"state.json",state)


def summarize(identity,state,backend,tokenizer):
  records = backend.records
  report = dict(identity=identity,position=state["position"],generated_tokens=state["generated"],
      generated_text=tokenizer.decode(state["generated"]),records=records,status="running",
      rtl_core_seconds=sum(r["rtl_core_seconds"] for r in records),
      rtl_cycles=sum(r["rtl_cycles"] for r in records),
      lpn_cycles=sum(r["lpn_cycles"] for r in records),
      lpn_core_seconds=sum(r["lpn_core_seconds"] for r in records),
      single_use_lpn_seconds=sum(r["single_use_lpn_seconds"] for r in records),
      all_steps_within_5_percent=all(r["accuracy_within_5_percent"] for r in records))
  report["core_speedup"] = (report["rtl_core_seconds"]/report["lpn_core_seconds"]
                            if records else None)
  return report


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--opentpu-root",type=Path,required=True)
  p.add_argument("--model-dir",type=Path,required=True)
  p.add_argument("--lpn-compiler",type=Path,required=True)
  p.add_argument("--output",type=Path,required=True)
  p.add_argument("--prompt",default="Once upon a time, in a quiet village near the mountains,")
  p.add_argument("--tokens",type=int,default=64,help="maximum generated tokens, stopping at EOS")
  p.add_argument("--cap",type=int,default=512)
  p.add_argument("--depth",type=int,default=128)
  p.add_argument("--batch-rows",type=int,default=8)
  p.add_argument("--rtl-timeout",type=float,default=28800)
  p.add_argument("--resume",action="store_true")
  p.add_argument("--max-steps",type=int,help="checkpoint and stop after this many total input tokens")
  a = p.parse_args()
  use_checkout(a.opentpu_root)
  import numpy as np
  import torch
  from tokenizers import Tokenizer
  from opentpu import rtlsim
  from opentpu.llm import load_spec
  from opentpu.llm.qwen3 import Engine,load_weights,device_config
  torch.set_num_threads(1)
  a.output,a.model_dir,a.lpn_compiler = a.output.resolve(),a.model_dir.resolve(),a.lpn_compiler.resolve()
  a.output.mkdir(parents=True,exist_ok=True)
  lock = (a.output/"run.lock").open("a")
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if a.tokens<1 or a.cap<1 or a.depth not in (64,128):
    p.error("positive tokens/cap and depth 64 or 128 required")
  rtlsim.VFLAGS[rtlsim.VFLAGS.index("-j")+1] = "16"
  rtlsim.VFLAGS += ["-CFLAGS","-O3"]
  spec = load_spec(a.model_dir)
  if json.loads((a.model_dir/"config.json").read_text())["model_type"] not in (
      "llama","smollm3","phi3","qwen3"):
    p.error("only dense Qwen3 and Llama-family models have a supported timing backend")
  tokenizer = Tokenizer.from_file(str(a.model_dir/"tokenizer.json"))
  prompt = tokenizer.encode(a.prompt).ids
  if not prompt or len(prompt)+a.tokens>min(a.cap,getattr(spec,"ctx",None) or a.cap):
    p.error("prompt plus requested generation must fit the context capacity")
  cfg = device_config(spec,a.cap,S=1,D=a.depth,MCOLS=4,ACT_ROWS=16,
      ACT_BLOCKS=64,LANES=8,IMEM_WORDS=32768,TMEM_WORDS=65536)
  files = sorted(a.model_dir.glob("*.safetensors"))+sorted(a.model_dir.glob("*.json"))
  identity = dict(upstream_revision=REVISION,checkpoint_sha256={f.name:digest(f) for f in files},
      config=asdict(cfg),prompt=a.prompt,prompt_ids=prompt,cap=a.cap,batch_rows=a.batch_rows,
      model_directory=str(a.model_dir),verilator=rtlsim.verilator_version(),verilator_flags=rtlsim.VFLAGS)
  identity["lpn_source_sha256"] = {name:digest(Path(__file__).with_name(name))
                                  for name in ("e2e.py","compact.py","model.py","workload.py")}
  identity["lpn_compiler_sha256"] = digest(a.lpn_compiler)
  identity["lpn_runtime_sha256"] = digest(a.lpn_compiler.parent/"libLPNNativeRuntime.a")
  identity["lpn_frontend_sha256"] = digest(Path(__file__).resolve().parents[2]/"lpnlang_mlir/__init__.py")
  identity["numpy_version"] = np.__version__
  identity["torch_version"] = torch.__version__
  identity["rtl_uarch"] = rtlsim.UARCH
  identity["rtl_memory"] = rtlsim.MEMORY
  saved = a.output/"state.json"
  if saved.exists() and not a.resume:
    p.error("output has a checkpoint; use --resume or a new output directory")
  state = json.loads(saved.read_text()) if a.resume else dict(identity=identity,
      position=0,generated=[],input_tokens=[],records=[])
  if state["identity"]!=identity:
    raise ValueError("checkpoint, prompt, configuration or software differs from saved run")
  if a.resume and digest(a.output/state["snapshot"])!=state["snapshot_sha256"]:
    raise ValueError("resume snapshot hash differs")
  before = time.perf_counter()
  engine = Engine(spec,load_weights(a.model_dir),a.cap,cfg,
      backend=lambda c,im:CheckedBackend(c,im,a),pipeline=False,resident=False)
  print(json.dumps({"stage":"engine_ready","seconds":time.perf_counter()-before,
                    "config":asdict(cfg),"prompt_ids":prompt}),flush=True)
  backend,logits = engine.backend,None
  if a.resume:
    with np.load(a.output/state["snapshot"]) as snap:
      backend.drams = [snap[f"dram_{s}"].copy() for s in range(cfg.S)]
      logits = snap["logits"].copy()
    engine.pos = state["position"]
    backend.records = state["records"]
  while len(state["generated"])<a.tokens:
    if a.max_steps is not None and engine.pos>=a.max_steps:
      break
    if engine.pos<len(prompt):
      token,phase = prompt[engine.pos],"prefill"
    else:
      token,phase = int(np.argmax(logits)),"decode"
      if token in spec.eos:
        state["eos"] = token
        atomic_json(saved,state)
        break
      state["generated"].append(token)
    before = time.perf_counter()
    logits = engine.step(token)
    backend.records[-1].update(phase=phase,input_token=token,position=engine.pos-1,
        next_token=int(np.argmax(logits)),logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest(),
        total_step_seconds=time.perf_counter()-before)
    state["input_tokens"].append(token)
    state["position"] = engine.pos
    checkpoint(a.output,backend,state,logits)
    report = summarize(identity,state,backend,tokenizer)
    atomic_json(a.output/"report.json",report)
    print(json.dumps({"stage":"checkpoint","position":engine.pos,"phase":phase,
        "generated_text":report["generated_text"],"rtl_hours":report["rtl_core_seconds"]/3600,
        "cycle_error":backend.records[-1]["relative_cycle_error"]}),flush=True)
  if backend.records:
    # state.json is authoritative if a session ended after committing its
    # snapshot but before refreshing report.json. Rebuild, never reuse a
    # potentially stale report when a resumed run has already reached its end.
    report = summarize(identity,state,backend,tokenizer)
    report["status"] = "eos" if "eos" in state else (
        "complete" if len(state["generated"])>=a.tokens else "checkpointed")
    report["requested_generated_tokens"] = a.tokens
    atomic_json(a.output/"report.json",report)


if __name__=="__main__":
  main()
