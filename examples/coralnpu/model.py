"""Event-driven CoralNPU timing model, compiled using the existing LPN backend.

The input is a dynamic instruction stream, not a stream of measured timestamps.
Timing comes from resource tokens, dependencies, and configurable unit delays.
This is a performance abstraction, not an ISA functional emulator.
"""

from dataclasses import dataclass, field
from math import ceil, isfinite
from typing import Optional

from lpnlang_mlir import NetBuilder
from .matrix import MatrixCommand, MatrixSpec


@dataclass(frozen=True)
class UnitSpec:
  latency: int
  interval: int = 1
  ports: int = 1

  def __post_init__(self):
    if min(self.latency, self.interval, self.ports) < 1:
      raise ValueError("unit latency, interval, and ports must be positive")


@dataclass(frozen=True)
class MemorySpec:
  latency: int = 3
  bytes_per_cycle: float = 16
  outstanding: int = 16

  def __post_init__(self):
    if (self.latency < 0 or self.outstanding < 1 or
        not isfinite(self.bytes_per_cycle) or self.bytes_per_cycle <= 0):
      raise ValueError("invalid memory latency, bandwidth, or outstanding limit")

  def service_cycles(self, size: int) -> int:
    if size <= 0:
      raise ValueError("memory request size must be positive")
    return max(1, ceil(size / self.bytes_per_cycle))


@dataclass
class CoreConfig:
  dispatch_width: int = 4
  rob_entries: int = 16
  branch_penalty: int = 3
  vector_queue_entries: int = 24  # command queue (8) + decoded uop queue (16)
  vector_decode_latency: int = 3  # command handoff plus two decoder stages
  vector_dispatch_width: int = 2
  # Defaults reflect structural RTL parameters. Latencies are model parameters
  # and are validated per kernel; they are not a claim of universal cycle accuracy.
  units: dict = field(default_factory=lambda: {
      "alu": UnitSpec(1, ports=4), "branch": UnitSpec(1, ports=4),
      "mul": UnitSpec(2), "div": UnitSpec(34, interval=34),
      "fpu": UnitSpec(4), "lsu": UnitSpec(1),
      "vconfig": UnitSpec(1), "valu": UnitSpec(4, ports=2),
      "vmul": UnitSpec(5, ports=2), "vfpu": UnitSpec(6, ports=2),
      "vdiv": UnitSpec(36, interval=36), "vperm": UnitSpec(6),
      "matrix": UnitSpec(1), "tile": UnitSpec(4),
  })
  dtcm: MemorySpec = field(default_factory=MemorySpec)
  external: MemorySpec = field(default_factory=lambda: MemorySpec(64, 16, 16))
  matrix: MatrixSpec = field(default_factory=MatrixSpec)

  def __post_init__(self):
    if (self.dispatch_width < 1 or self.rob_entries < 1 or self.branch_penalty < 0
        or self.vector_queue_entries < 1 or self.vector_decode_latency < 0
        or self.vector_dispatch_width < 1):
      raise ValueError("invalid core configuration")


@dataclass(frozen=True)
class Instruction:
  unit: str
  reads: tuple = ()
  writes: tuple = ()
  pc: int = 0
  bits: int = 0
  bytes: int = 0
  external: bool = False
  redirect: bool = False
  # RVV LMUL/SEW expansion is supplied by the trace adapter/user workload.
  uops: int = 1
  matrix: Optional[MatrixCommand] = None

  @property
  def vector(self):
    return self.unit in ("matrix", "tile") or any(reg.startswith("v") and reg != "vtype"
               for reg in self.reads + self.writes)


class ExecutionUnit:
  def __init__(self, net, name, spec):
    self.spec = spec
    self.available = net.place(f"{name}_ports", initial_tokens=spec.ports)


class MemoryPort:
  """Latency plus bandwidth memory endpoint; usable for future DRAM requests."""
  def __init__(self, net, name, spec):
    self.spec = spec
    self.bus = net.place(f"{name}_bus", initial_tokens=1)
    self.slots = net.place(f"{name}_slots", initial_tokens=spec.outstanding)


class CoralNPU:
  def __init__(self, instructions, config: Optional[CoreConfig] = None):
    self.instructions = list(instructions)
    self.config = config or CoreConfig()
    self.net = net = NetBuilder("coralnpu")
    cfg = self.config
    self.units = {name: ExecutionUnit(net, name, spec)
                  for name, spec in cfg.units.items()}
    self.dtcm = MemoryPort(net, "dtcm", cfg.dtcm)
    self.external = MemoryPort(net, "external", cfg.external)
    self.dispatch = net.place("dispatch_slots", initial_tokens=cfg.dispatch_width)
    self.rob = net.place("rob_slots", initial_tokens=cfg.rob_entries)
    self.vector_queue = net.place("vector_queue_slots", initial_tokens=cfg.vector_queue_entries)
    self.vector_dispatch = net.place("vector_dispatch_slots", initial_tokens=cfg.vector_dispatch_width)
    self.retired = net.place("retired", observable=True)
    count = len(self.instructions)
    self.issued = [net.place(f"issued_{i}") for i in range(count)]
    self.completed = [net.place(f"completed_{i}") for i in range(count)]
    self.retire_order = [net.place(f"retire_order_{i}") for i in range(count)]
    self.result_ready = [net.place(f"result_{i}") for i in range(count)]
    self.reuse_ready = [net.place(f"reuse_{i}") for i in range(count)]
    last_writer = {}
    previous_memory = None
    for idx, instruction in enumerate(self.instructions):
      if instruction.unit not in self.units:
        raise ValueError(f"unsupported CoralNPU unit: {instruction.unit}")
      if instruction.uops < 1:
        raise ValueError("uops must be positive")
      if (instruction.unit == "matrix") != (instruction.matrix is not None):
        raise ValueError("matrix commands require explicit shape and format metadata")
      # RAW and WAW hazards. Reads are sampled at in-order dispatch, so WAR
      # needs no additional completion dependency in this abstraction.
      deps = {last_writer[reg] for reg in instruction.reads + instruction.writes
              if reg in last_writer}
      dispatch_deps = {last_writer[reg] for reg in instruction.reads + instruction.writes
                       if reg in last_writer and reg.startswith(("x", "f"))}
      reuse_deps = {last_writer[reg] for reg in instruction.reads + instruction.writes
                    if reg in last_writer and reg.startswith("mt")
                    and instruction.unit == "matrix"
                    and self.instructions[last_writer[reg]].unit == "matrix"}
      if instruction.bytes:
        # Conservative memory ordering until addresses/aliasing are provided.
        if previous_memory is not None:
          deps.add(previous_memory)
        previous_memory = idx
      for reg in instruction.writes:
        last_writer[reg] = idx
      self._add_instruction(idx, instruction, sorted(deps), sorted(dispatch_deps), reuse_deps)

  def _add_instruction(self, idx, instruction, dependencies, dispatch_dependencies, reuse_dependencies):
    net, cfg = self.net, self.config
    latch = net.place(f"pending_{idx}", initial_tokens=1)
    previous = self.issued[idx - 1] if idx else None
    deps = [self.reuse_ready[dep] if dep in reuse_dependencies else self.result_ready[dep]
            for dep in dependencies]
    issue_deps = ([self.result_ready[dep] for dep in dispatch_dependencies]
                  if instruction.vector else deps)
    unit = self.units[instruction.unit]
    memory = self.external if instruction.external else self.dtcm
    service = memory.spec.service_cycles(instruction.bytes) if instruction.bytes else 0
    latency = (memory.spec.latency + service if instruction.bytes else
               unit.spec.latency + (instruction.uops - 1) * unit.spec.interval)
    interval = instruction.uops * unit.spec.interval
    if instruction.matrix is not None:
      latency = cfg.matrix.latency(instruction.matrix)
      interval = cfg.matrix.issue_cycles(instruction.matrix)
    reuse_latency = cfg.matrix.reuse_cycles(instruction.matrix) if instruction.matrix else latency
    reuse = self.reuse_ready[idx]
    complete = self.completed[idx]
    result = self.result_ready[idx]
    issued = self.issued[idx]
    redirect_delay = cfg.branch_penalty if instruction.redirect else 0
    queued = net.place(f"vector_queued_{idx}") if instruction.vector else None

    @net.transition(f"issue_{idx}")
    def issue():
      # Explicit read-only enabling annotation. Each guard precedes side
      # effects and covers all resources this firing will consume.
      guard(count(latch) > 0)
      guard(count(self.dispatch) > 0)
      guard(count(self.rob) > 0)
      if instruction.vector:
        guard(count(self.vector_queue) > 0)
        guard(count(self.vector_dispatch) > 0)
      else:
        guard(count(unit.available) > 0)
      if previous is not None:
        guard(count(previous) > 0)
      for dep in issue_deps:
        guard(count(dep) > 0)
      if instruction.bytes and not instruction.vector:
        guard(count(memory.bus) > 0)
        guard(count(memory.slots) > 0)
      take(latch)
      slot = take(self.dispatch)
      take(self.rob)
      emit(self.dispatch, slot, delay=1.0)
      token = create().set("id", idx)
      emit(issued, token, delay=float(redirect_delay))
      if instruction.vector:
        take(self.vector_queue)
        vector_slot = take(self.vector_dispatch)
        emit(self.vector_dispatch, vector_slot, delay=1.0)
        emit(queued, token, delay=float(cfg.vector_decode_latency))
      else:
        port = take(unit.available)
        emit(unit.available, port, delay=float(interval))
        if instruction.bytes:
          bus = take(memory.bus)
          mem_slot = take(memory.slots)
          emit(memory.bus, bus, delay=float(service))
          emit(memory.slots, mem_slot, delay=float(latency))
        emit(result, token, delay=float(latency))
        emit(complete, token, delay=float(latency))

    if instruction.vector:
      @net.transition(f"vector_execute_{idx}")
      def vector_execute():
        guard(count(queued) > 0)
        guard(count(unit.available) > 0)
        for dep in deps:
          guard(count(dep) > 0)
        if instruction.bytes:
          guard(count(memory.bus) > 0)
          guard(count(memory.slots) > 0)
        token = take(queued)
        port = take(unit.available)
        emit(self.vector_queue, create())
        emit(unit.available, port, delay=float(interval))
        if instruction.bytes:
          bus = take(memory.bus)
          mem_slot = take(memory.slots)
          emit(memory.bus, bus, delay=float(service))
          emit(memory.slots, mem_slot, delay=float(latency))
        emit(result, token, delay=float(latency))
        emit(complete, token, delay=float(latency))
        emit(reuse, token, delay=float(reuse_latency))

    previous_retire = self.retire_order[idx - 1] if idx else None
    retired_order = self.retire_order[idx]

    @net.transition(f"retire_{idx}")
    def retire():
      guard(count(complete) > 0)
      if previous_retire is not None:
        guard(count(previous_retire) > 0)
      token = take(complete)
      emit(self.rob, create())
      emit(retired_order, token)
      emit(self.retired, token)

  def build(self):
    return self.net.build()

  def simulate(self):
    simulator = self.net.python_simulator()
    cycles = simulator.run(max_time=1e12)
    retired = simulator.place_dicts(self.retired)
    if len(retired) != len(self.instructions):
      raise RuntimeError(f"model deadlocked: retired {len(retired)}/{len(self.instructions)}")
    return {"cycles": cycles, "instructions": len(retired)}
