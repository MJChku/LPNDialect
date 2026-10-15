"""CoralNPU's four-block, packed-MAC matrix datapath as an LPN.

Each token describes a row group, not an individual MAC. Physical PE blocks,
their 0/1/1/2-cycle operand staggering, multiplier/FIFO/adder stages, and
accumulator reuse are explicit. The existing LPN scheduler handles all events.
"""

from dataclasses import dataclass
from math import ceil

from lpnlang_mlir import NetBuilder


@dataclass(frozen=True)
class MatrixCommand:
  tile: int = 0
  m: int = 16
  n: int = 16
  k: int = 4
  dtype: str = "int8"

  def __post_init__(self):
    limits = {"int8": 4, "uint8": 4, "bf16": 2, "fp32": 1}
    if self.dtype not in limits:
      raise ValueError(f"unsupported matrix format: {self.dtype}")
    if self.tile not in (0, 4, 8, 12):
      raise ValueError("32-bit accumulator groups must start at mt0/4/8/12")
    if not (1 <= self.m <= 16 and 1 <= self.n <= 16 and 1 <= self.k <= limits[self.dtype]):
      raise ValueError("matrix shape exceeds the selected RTL format")

  @property
  def macs(self):
    return self.m * self.n * self.k


@dataclass(frozen=True)
class MatrixSpec:
  tile_elements: int = 16
  rows_per_cycle: int = 2
  multiply_stages: int = 3
  product_fifo_stages: int = 1
  add_stages: int = 3

  def __post_init__(self):
    if self.tile_elements != 16 or self.rows_per_cycle != 2:
      raise ValueError("this model targets the pinned VLEN=128 PE geometry")
    if min(self.multiply_stages, self.product_fifo_stages, self.add_stages) < 1:
      raise ValueError("pipeline stage counts must be positive")

  @property
  def pipeline_latency(self):
    return self.multiply_stages + self.product_fifo_stages + self.add_stages

  def blocks(self, command):
    half = self.tile_elements // 2
    blocks = []
    for block in range(4):
      row_half, col_half = divmod(block, 2)
      rows = min(half, max(0, command.m - row_half * half))
      cols = min(half, max(0, command.n - col_half * half))
      if rows and cols:
        blocks.append((block, row_half + col_half, rows, cols))
    return blocks

  def issue_cycles(self, command):
    return ceil(min(command.m, 8) / self.rows_per_cycle)

  def reuse_cycles(self, command):
    # zvt_pe_array.hitRaw waits for outstanding multiplier tags in the
    # first row group (readMtId == 0). Later row groups follow the same
    # pipeline and do not require an additional whole-command drain.
    return max(self.issue_cycles(command), self.multiply_stages + 1)

  def latency(self, command):
    return max(stagger + ceil(rows / self.rows_per_cycle) - 1
               for _, stagger, rows, _ in self.blocks(command)) + self.pipeline_latency


class MatrixArray:
  def __init__(self, commands, spec=None, net=None, input_ready=None):
    self.commands = list(commands)
    self.spec = spec or MatrixSpec()
    self.net = net = net or NetBuilder("coralnpu_matrix")
    self.input_ready = input_ready
    self.ingress = net.place("matrix_ingress", initial_tokens=1)
    self.accumulator = {tile: net.place(f"accumulator_{tile}", initial_tokens=1)
                        for tile in (0, 4, 8, 12)}
    self.mul_ports = [net.place(f"block_{b}_mul_port", initial_tokens=1) for b in range(4)]
    self.add_ports = [net.place(f"block_{b}_add_port", initial_tokens=1) for b in range(4)]
    self.finished = net.place("matrix_completed", observable=True)
    self.issued = [net.place(f"matrix_issued_{i}") for i in range(len(self.commands))]
    self.completed = []
    for idx, command in enumerate(self.commands):
      self._add_command(idx, command)

  def _add_command(self, idx, command):
    net, spec = self.net, self.spec
    pending = net.place(f"matrix_pending_{idx}", initial_tokens=1)
    previous = self.issued[idx - 1] if idx else None
    issued = self.issued[idx]
    acc = self.accumulator[command.tile]
    ready = self.input_ready[idx] if self.input_ready is not None else None
    blocks = spec.blocks(command)
    done = net.place(f"matrix_done_{idx}")
    completed = net.place(f"matrix_result_{idx}", observable=True)
    self.completed.append(completed)
    batches = []
    for block, stagger, rows, cols in blocks:
      for batch in range(ceil(rows / spec.rows_per_cycle)):
        width = min(spec.rows_per_cycle, rows - batch * spec.rows_per_cycle)
        request = net.place(f"cmd_{idx}_block_{block}_batch_{batch}")
        batches.append((request, stagger + batch,
                        width * cols * command.k))
        self._add_batch(idx, block, batch, request, done,
                        batch == ceil(rows / spec.rows_per_cycle) - 1)

    @net.transition(f"matrix_launch_{idx}")
    def launch():
      guard(count(pending) > 0)
      guard(count(self.ingress) > 0)
      guard(count(acc) > 0)
      if previous is not None:
        guard(count(previous) > 0)
      if ready is not None:
        guard(count(ready) > 0)
      take(pending)
      port = take(self.ingress)
      accumulator = take(acc)
      emit(self.ingress, port, delay=float(spec.issue_cycles(command)))
      emit(acc, accumulator, delay=float(spec.reuse_cycles(command)))
      token = create().set("id", idx).set("macs", command.macs)
      emit(issued, token)
      for request, delay, macs in batches:
        emit(request, token.set("batch_macs", macs), delay=float(delay))

    @net.transition(f"matrix_finish_{idx}")
    def finish():
      guard(count(done) == len(blocks))
      for _ in range(len(blocks)):
        take(done)
      token = create().set("id", idx).set("macs", command.macs)
      emit(completed, token)
      emit(self.finished, token)

  def _add_batch(self, idx, block, batch, request, done, last):
    net, spec = self.net, self.spec
    product = net.place(f"product_{idx}_{block}_{batch}")
    mul_port, add_port = self.mul_ports[block], self.add_ports[block]

    @net.transition(f"multiply_{idx}_{block}_{batch}")
    def multiply():
      guard(count(request) > 0)
      guard(count(mul_port) > 0)
      token = take(request)
      port = take(mul_port)
      emit(mul_port, port, delay=1.0)
      emit(product, token, delay=float(spec.multiply_stages + spec.product_fifo_stages))

    @net.transition(f"accumulate_{idx}_{block}_{batch}")
    def accumulate():
      guard(count(product) > 0)
      guard(count(add_port) > 0)
      token = take(product)
      port = take(add_port)
      emit(add_port, port, delay=1.0)
      if last:
        emit(done, token, delay=float(spec.add_stages))

  def build(self):
    return self.net.build()

  def simulate(self):
    sim = self.net.python_simulator()
    cycles = sim.run(max_time=1e12)
    completed = sim.place_dicts(self.finished)
    if len(completed) != len(self.commands):
      raise RuntimeError("matrix LPN did not complete every command")
    return {"cycles": cycles, "commands": len(completed),
            "macs": sum(command.macs for command in self.commands)}


class GemmModel:
  """Tile a GEMM onto PE/accumulator resources with operand prefetch and stores.

  An operand-buffer credit is held from DMA request until the associated
  matrix operation finishes. Accumulator ownership lasts through output
  storage. This permits compute/memory overlap without unbounded buffering.
  """
  def __init__(self, m, n, k, dtype="int8", memory=None, buffers=2):
    from .model import MemoryPort, MemorySpec
    if min(m, n, k, buffers) < 1:
      raise ValueError("GEMM dimensions and operand buffer count must be positive")
    kmax = {"int8": 4, "uint8": 4, "bf16": 2, "fp32": 1}.get(dtype)
    if kmax is None:
      raise ValueError("unsupported GEMM format")
    self.net = net = NetBuilder("coralnpu_gemm")
    self.memory = MemoryPort(net, "gemm_memory", memory or MemorySpec())
    self.buffer_slots = net.place("operand_buffer_slots", initial_tokens=buffers)
    self.tile_control = net.place("tile_control_port", initial_tokens=1)
    self.owners = {tile: net.place(f"accumulator_owner_{tile}", initial_tokens=1)
                   for tile in (0, 4, 8, 12)}
    self.finished = net.place("gemm_completed", observable=True)
    self.macs = m * n * k
    commands, jobs = [], []
    for row in range(0, m, 16):
      for col in range(0, n, 16):
        tile = (len(jobs) % 4) * 4
        start = len(commands)
        for inner in range(0, k, kmax):
          commands.append(MatrixCommand(tile, min(16, m-row), min(16, n-col),
                                        min(kmax, k-inner), dtype))
        jobs.append((tile, start, len(commands)-1))
    ready = [net.place(f"operands_ready_{idx}") for idx in range(len(commands))]
    self.array = MatrixArray(commands, net=net, input_ready=ready)
    for job_id, (tile, first, last) in enumerate(jobs):
      owned = net.place(f"output_tile_owned_{job_id}")
      for idx in range(first, last + 1):
        self._prefetch(idx, commands[idx], ready[idx], owned, first == idx)
      self._store(job_id, commands[last], self.array.completed[last])
    self.jobs = len(jobs)

  def _prefetch(self, idx, command, ready, owned, first):
    net, memory = self.net, self.memory
    pending = net.place(f"operand_request_{idx}", initial_tokens=1)
    owner = self.owners[command.tile]
    size = (command.m + command.n) * command.k * {"int8": 1, "uint8": 1, "bf16": 2, "fp32": 4}[command.dtype]
    service = memory.spec.service_cycles(size)
    delay = memory.spec.latency + service
    complete = self.array.completed[idx]
    released = net.place(f"buffer_release_{idx}", initial_tokens=1)

    @net.transition(f"prefetch_{idx}")
    def prefetch():
      guard(count(pending) > 0)
      guard(count(self.buffer_slots) > 0)
      guard(count(memory.bus) > 0)
      guard(count(memory.slots) > 0)
      if first:
        guard(count(owner) > 0)
        guard(count(self.tile_control) > 0)
      else:
        guard(count(owned) > 0)
      take(pending)
      take(self.buffer_slots)
      if first:
        take(owner)
        control = take(self.tile_control)
        emit(self.tile_control, control, delay=4.0)
        emit(owned, create(), delay=4.0)
      bus = take(memory.bus)
      slot = take(memory.slots)
      emit(memory.bus, bus, delay=float(service))
      emit(memory.slots, slot, delay=float(delay))
      emit(ready, create(), delay=float(max(delay, 4) if first else delay))

    @net.transition(f"release_operands_{idx}")
    def release_operands():
      guard(count(released) > 0)
      guard(count(complete) > 0)
      take(released)
      emit(self.buffer_slots, create())

  def _store(self, idx, command, complete):
    net, memory = self.net, self.memory
    pending = net.place(f"store_request_{idx}", initial_tokens=1)
    owner = self.owners[command.tile]
    service = memory.spec.service_cycles(command.m * command.n * 4)
    delay = memory.spec.latency + service

    @net.transition(f"store_result_{idx}")
    def store_result():
      guard(count(pending) > 0)
      guard(count(complete) > 0)
      guard(count(memory.bus) > 0)
      guard(count(memory.slots) > 0)
      take(pending)
      bus = take(memory.bus)
      slot = take(memory.slots)
      emit(memory.bus, bus, delay=float(service))
      emit(memory.slots, slot, delay=float(delay))
      emit(owner, create(), delay=float(delay))
      emit(self.finished, create().set("tile", idx), delay=float(delay))

  def build(self):
    return self.net.build()

  def simulate(self):
    sim = self.net.python_simulator()
    cycles = sim.run(max_time=1e12)
    if len(sim.place_dicts(self.finished)) != self.jobs:
      raise RuntimeError("GEMM LPN did not store every output tile")
    return {"cycles": cycles, "tiles": self.jobs, "macs": self.macs,
            "matrix_commands": len(self.array.commands)}
