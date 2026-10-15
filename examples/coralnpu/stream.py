"""A compact instruction-dataflow LPN for an independently executed ELF path.

Instruction records contain ISA-derived resources/dependencies, never measured
timestamps. Reusable transitions keep long loops from expanding the net.
"""

from lpnlang_mlir import NetBuilder
from .model import CoreConfig, MemoryPort


READ,WRITE,ISSUE,STRIDE = 11,27,35,43


class InstructionStream:
  def __init__(self, workload, config=None):
    cfg = self.config = config or CoreConfig()
    self.workload = workload
    if any(instruction.external for instruction in workload.instructions):
      raise ValueError("the compact ELF scoreboard currently supports local-memory workloads only")
    self.net = net = NetBuilder("coralnpu_elf")
    self.units = list(cfg.units)
    self.unit_places = [net.place(f"{name}_ports",initial_tokens=cfg.units[name].ports)
                        for name in self.units]
    names = ([f"x{i}" for i in range(1,32)]+[f"f{i}" for i in range(32)]+
             [f"v{i}" for i in range(32)]+[f"mt{i}" for i in range(16)]+["vtype","mtype"])
    ids = {name:i+1 for i,name in enumerate(names)}
    self.vector_begin,self.vector_end = ids["v0"],ids["v31"]+1
    written_chunks = set()
    for ins,address in zip(workload.instructions,workload.addresses):
      if address is not None and ins.bits&127 in (0x23,0x27):
        written_chunks.update(range(address//16,(address+ins.bytes-1)//16+1))
    memory_ids = {chunk:len(names)+1+i for i,chunk in enumerate(sorted(written_chunks))}
    names += [f"memory_{chunk*16:x}" for chunk in sorted(written_chunks)]
    self.registers = [net.place("constant_ready",initial_tokens=1)]+[
        net.place(name+"_ready",initial_tokens=1) for name in names]
    self.records,self.program = [],[]
    unique = {}
    for ins,address in zip(workload.instructions,workload.addresses):
      unit = cfg.units[ins.unit]
      reads = {ids[name] for name in ins.reads if not(ins.vector and name.startswith("x"))}
      writes = {ids[name] for name in ins.writes}
      if ins.matrix:
        writes.clear()  # Accumulator reuse and drain are separate resources.
      if address is not None:
        chunks = set(range(address//16,(address+ins.bytes-1)//16+1))
        alias = {memory_ids[chunk] for chunk in chunks if chunk in memory_ids}
        if ins.bits&127 in (0x23,0x27): writes |= alias
        else: reads |= alias
      reads -= writes
      if ins.vector:
        issue = {ids[name] for name in ins.reads if name.startswith(("x","f"))}
      else:
        issue = reads|writes
      if max(len(reads)-16,len(writes)-8,len(issue)-8)>0:
        raise ValueError("instruction exceeds compact scoreboard capacity")
      memory = cfg.external if ins.external else cfg.dtcm
      latency = memory.latency+memory.service_cycles(ins.bytes) if ins.bytes else unit.latency+(ins.uops-1)*unit.interval
      interval = ins.uops*unit.interval
      if ins.matrix:
        latency,interval = cfg.matrix.latency(ins.matrix),cfg.matrix.issue_cycles(ins.matrix)
      record = [int(ins.vector),self.units.index(ins.unit),latency,interval,
                cfg.branch_penalty*int(ins.redirect),len(reads),len(writes),
                memory.service_cycles(ins.bytes) if ins.bytes else 0,int(ins.matrix is not None),
                ins.matrix.tile//4+1 if ins.matrix else 0,int(ins.unit=="tile")]
      for values,width in ((reads,16),(writes,8),(issue,8)):
        values = sorted(values)
        record += values+[0]*(width-len(values))
      record = tuple(record)
      if record not in unique:
        unique[record] = len(self.records)//STRIDE
        self.records.extend(record)
      self.program.append(unique[record])
    self.cursor = net.place("dispatch_cursor",initial_tokens=1)
    self.retire_cursor = net.place("retire_cursor",initial_tokens=1)
    self.dispatch = net.place("dispatch_slots",initial_tokens=cfg.dispatch_width)
    self.vector_dispatch = net.place("vector_dispatch_slots",initial_tokens=cfg.vector_dispatch_width)
    self.rob = net.place("rob_credits",initial_tokens=cfg.rob_entries)
    self.vector_credits = net.place("vector_queue_credits",initial_tokens=cfg.vector_queue_entries)
    self.queue = net.place("vector_queue")
    self.completions = [net.place(f"rob_complete_{i}") for i in range(cfg.rob_entries)]
    self.matrix_slots = net.place("matrix_inflight_credits",initial_tokens=cfg.rob_entries)
    self.accumulators = [net.place(f"matrix_acc_{i}",initial_tokens=1) for i in range(5)]
    self.memory = MemoryPort(net,"memory",cfg.dtcm)
    self.finished = net.place("program_completed",observable=True)
    self.roi_start,self.roi_end = net.place("roi_start"),net.place("roi_end")
    self._transitions()

  def _transitions(self):
    net,cfg = self.net,self.config
    program,records = self.program,self.records
    n = len(program)
    roi_first,roi_last = self.workload.roi[0],self.workload.roi[1]-1
    checks16,checks8 = tuple(range(16)),tuple(range(8))
    vbegin,vend = self.vector_begin,self.vector_end

    @net.transition
    def issue():
      guard(count(self.cursor)>0)
      cursor = peek(self.cursor)
      idx = cursor.get("idx")
      guard(idx<n)
      ids = array(program)
      data = array(records)
      regs = array(self.registers)
      ports = array(self.unit_places)
      complete = array(self.completions)
      base = muli(ids[idx],STRIDE)
      vector = data[base]
      port = ports[data[base+1]]
      latency,interval = data[base+2],data[base+3]
      redirect = data[base+4]
      write_count,service = data[base+6],data[base+7]
      guard(count(self.dispatch)>0)
      guard(count(self.rob)>0)
      guard(count(self.vector_dispatch)>=vector)
      guard(count(self.vector_credits)>=vector)
      guard(count(port)>=1-vector)
      for j in checks8:
        guard(count(regs[data[base+ISSUE+j]])>0)
      # Scalar memory has no queued execution stage.
      guard(count(self.memory.bus)>=select(service>0,1-vector,0))
      guard(count(self.memory.slots)>=select(service>0,1-vector,0))
      take(self.cursor)
      slot = take(self.dispatch)
      take(self.rob)
      emit(self.dispatch,slot,delay=1.0)
      token = cursor.set("idx",idx)
      emit(self.cursor,cursor.set("idx",idx+1),delay=sitofp(redirect))
      if vector:
        vslot = take(self.vector_dispatch)
        take(self.vector_credits)
        emit(self.vector_dispatch,vslot,delay=1.0)
        emit(self.queue,token,delay=float(cfg.vector_decode_latency))
      else:
        unit = take(port)
        emit(port,unit,delay=sitofp(interval))
        for j in range(write_count):
          target = regs[data[base+WRITE+j]]
          value = take(target)
          emit(target,value,delay=sitofp(latency))
        if service>0:
          bus = take(self.memory.bus)
          credit = take(self.memory.slots)
          emit(self.memory.bus,bus,delay=sitofp(service))
          emit(self.memory.slots,credit,delay=sitofp(latency))
        emit(complete[idx%cfg.rob_entries],token,delay=sitofp(latency))
      if idx==roi_first:
        emit(self.roi_start,create())
      if idx==roi_last:
        emit(self.roi_end,create(),delay=1.0)

    @net.transition
    def vector_execute():
      guard(count(self.queue)>0)
      token = peek(self.queue)
      idx = token.get("idx")
      data,ids = array(records),array(program)
      regs,ports = array(self.registers),array(self.unit_places)
      accumulators,complete = array(self.accumulators),array(self.completions)
      base = muli(ids[idx],STRIDE)
      port = ports[data[base+1]]
      latency,interval = data[base+2],data[base+3]
      read_count,write_count,service = data[base+5],data[base+6],data[base+7]
      matrix,acc,drain = data[base+8],accumulators[data[base+9]],data[base+10]
      guard(count(port)>0)
      guard(count(acc)>0)
      guard(count(self.matrix_slots)>=muli(drain,cfg.rob_entries)+matrix)
      for j in checks16:
        guard(count(regs[data[base+READ+j]])>0)
      for j in checks8:
        guard(count(regs[data[base+WRITE+j]])>0)
      guard(count(self.memory.bus)>=select(service>0,1,0))
      guard(count(self.memory.slots)>=select(service>0,1,0))
      token = take(self.queue)
      unit = take(port)
      emit(port,unit,delay=sitofp(interval))
      emit(self.vector_credits,create())
      for j in range(read_count):
        reg = data[base+READ+j]
        if reg>=vbegin:
          if reg<vend:
            value = take(regs[reg])
            emit(regs[reg],value,delay=sitofp(interval))
      for j in range(write_count):
        target = regs[data[base+WRITE+j]]
        value = take(target)
        emit(target,value,delay=sitofp(latency))
      if service>0:
        bus = take(self.memory.bus)
        credit = take(self.memory.slots)
        emit(self.memory.bus,bus,delay=sitofp(service))
        emit(self.memory.slots,credit,delay=sitofp(latency))
      if matrix:
        credit = take(self.matrix_slots)
        owner = take(acc)
        emit(self.matrix_slots,credit,delay=sitofp(latency))
        emit(acc,owner,delay=sitofp(select(interval>cfg.matrix.multiply_stages+1,
                                         interval,cfg.matrix.multiply_stages+1)))
      emit(complete[idx%cfg.rob_entries],token,delay=sitofp(latency))

    @net.transition
    def retire():
      guard(count(self.retire_cursor)>0)
      token = peek(self.retire_cursor)
      idx = token.get("idx")
      guard(idx<n)
      complete = array(self.completions)
      guard(count(complete[idx%cfg.rob_entries])>0)
      take(self.retire_cursor)
      take(complete[idx%cfg.rob_entries])
      emit(self.rob,create())
      emit(self.retire_cursor,token.set("idx",idx+1))
      if idx==n-1:
        emit(self.finished,token.set("instructions",n))

    seen_first,seen_last = net.place("roi_start_observed"),net.place("roi_end_observed")

    @net.transition
    def observe_roi_start():
      emit(seen_first,take(self.roi_start))

    @net.transition
    def observe_roi_end():
      emit(seen_last,take(self.roi_end))

  def build(self):
    return self.net.build()
