"""Reusable command-window and matrix-row transitions for long executable traces.

The net size grows with completion signals and immutable command metadata, rather
than transition code. The existing event-driven LPN scheduler is unchanged.
"""

from bisect import bisect_left
from math import ceil, gcd

from .model import OpenTPU, DRAM
from .components import CommandInterface, CommandTiming, ExecutionUnit, MatrixEngine


def dependencies(commands,I,W,matrix_engines=1):
  """Exact footprint conflicts, indexed by intervals; remove transitive edges.

  Completion ancestry only contains completion dependencies. Matrix prefetch/
  issue ordering is deliberately excluded: neither implies result completion.
  """
  boundaries = [set() for _ in range(3)]
  for c in commands:
    for space,lo,hi in c.footprint.rd+c.footprint.wr:
      boundaries[space].update((lo,hi))
  bounds = [sorted(b) for b in boundaries]
  reads = [[set() for _ in b] for b in bounds]
  writes = [[set() for _ in b] for b in bounds]
  ancestors,compute,dram,last_unit = [],[],[],{}
  barrier = None

  def segments(ranges):
    for space,lo,hi in ranges:
      for segment in range(bisect_left(bounds[space],lo),bisect_left(bounds[space],hi)):
        yield space,segment

  def reduce_edges(edges):
    kept,covered = [],0
    for j in sorted(edges,reverse=True):
      if not (covered>>j)&1:
        kept.append(j)
        covered |= ancestors[j]|(1<<j)
    return sorted(kept)

  for idx,c in enumerate(commands):
    unit = W.UNIT.get(c.instruction.op,-1)
    conflicts,dc = set(),set()
    if c.footprint.all or c.instruction.op==I.HALT:
      conflicts.update(range(idx))
      dc.update(range(idx))
    else:
      for space,segment in segments(c.footprint.rd):
        conflicts.update(writes[space][segment])
        if space==W.SP_DRAM:
          dc.update(writes[space][segment])
      for space,segment in segments(c.footprint.wr):
        conflicts.update(reads[space][segment]|writes[space][segment])
        if space==W.SP_DRAM:
          dc.update(reads[space][segment]|writes[space][segment])
      if barrier is not None:
        conflicts.add(barrier)
        dc.add(barrier)
    if c.instruction.op!=I.HALT:
      # A single in-order unit enforces its own hazards. Multiple MM engines
      # share ACT/TMEM, so all RAW/WAR/WAW hazards between them must be explicit.
      conflicts = {j for j in conflicts if unit!=W.UNIT.get(commands[j].instruction.op,-2)
                   or unit==W.U_VPU or (unit==W.U_MXU and matrix_engines>1)}
    if unit in last_unit and unit not in (W.U_MXU,W.U_VPU):
      conflicts.add(last_unit[unit])
    compute.append(reduce_edges(conflicts))
    dram.append(reduce_edges(dc))
    complete_deps = conflicts | (dc if unit==W.U_MXU else set())
    ancestry = 0
    for j in complete_deps:
      ancestry |= ancestors[j]|(1<<j)
    ancestors.append(ancestry)
    if unit>=0:
      last_unit[unit] = idx
    if c.footprint.all:
      barrier = idx
      reads = [[set() for _ in b] for b in bounds]
      writes = [[set() for _ in b] for b in bounds]
    for space,segment in segments(c.footprint.rd):
      reads[space][segment].add(idx)
    for space,segment in segments(c.footprint.wr):
      writes[space][segment].add(idx)
  return compute,dram


class CompactOpenTPU(OpenTPU):
  supports_multiple_matrix_engines = True

  def __init__(self,workload,timing=None):
    self.components = {}
    super().__init__(workload,timing)

  def statistics(self):
    """Counts of declared net objects; helpers and instructions are not transitions."""
    names = [p.name for p,*_ in self.net._places]
    return {"transitions":len(self.net._transitions),"places":len(names),
        "completion_flag_banks":sum("_flags_" in n for n in names),
        "command_slots":sum("_slot_" in n for n in names),
        "matrix_engines_per_slice":self.timing.matrix_engines,
        "instructions_per_slice":[len(c) for c in self.workload.commands]}

  def _slice(self,sid,commands,I,W):
    cfg,t,net = self.workload.config,self.timing,self.net
    prefix = f"s{sid}"
    ncommands = len(commands)
    if not ncommands:
      raise ValueError("empty command trace")
    memory = DRAM(net,prefix+"_dram",t)
    nsig = ncommands
    # Immutable flag banks replace thousands of individual completion places.
    # A bank latch is always present; delayed completion messages update it
    # atomically. This keeps the scheduler's place scans bounded and retains
    # out-of-order VPU completion without assuming an in-order retire counter.
    banks = [net.place(prefix+f"_flags_{b}",initial_tokens=1) for b in range(nsig//64+1)]
    signals = net.place(prefix+"_signals")
    finished = net.place(prefix+"_halted")
    @net.func(name=prefix+"_is_done",args=("i64",),results=("i1",))
    def is_done(idx):
      token = peek(array(banks)[divi(idx,64)])
      mask = shli(1,remi(idx,64))
      func_return(select(idx==nsig,True,andi(token.get("bits"),mask)!=0,typ="i1"))

    @net.func(name=prefix+"_publish",args=("i64","f64"),results=())
    def publish(idx,latency):
      emit(signals,create().set("id",idx),delay=latency)
      func_return()
    # Each slot always has a latch token; the pipeline queue updates it after
    # dispatch latency. Window credits remain reserved until result completion.
    slots = [net.place(prefix+f"_slot_{s}",initial_tokens=1) for s in range(t.window)]
    window = net.place(prefix+"_window")
    initialize = net.place(prefix+"_initialize",initial_tokens=1)
    cursor = net.place(prefix+"_cursor",initial_tokens=1)
    pipeline = net.place(prefix+"_pipeline")
    pending = [net.place(prefix+f"_pending_{u}") for u in range(6)]
    ports = [memory.b,memory.sw]
    deps,dram_deps = dependencies(commands,I,W,t.matrix_engines)
    kinds,latencies,services,port_ids = [],[],[],[]
    batches,chunkses,scale_flags,kbs,ns,row_services,tails = [],[],[],[],[],[],[]
    barriers,dispatch_delays,pipeline_delays = [],[],[]
    last_register = ncommands
    for idx,c in enumerate(commands):
      ins,w = c.instruction,c.instruction.w
      u = W.UNIT.get(ins.op,5)
      kinds.append(u)
      barriers.append(last_register)
      dispatch_delays.append(2 if ins.op==I.LOOP else 1)
      pipeline_delays.append(5 if ins.op>=I.LD else 0)
      if ins.op==I.RLD:
        last_register = idx
      latency,port,service = (0,None,0) if u in (W.U_MXU,5) else self._simple_timing(ins,memory,I)
      latencies.append(latency)
      port_ids.append(ports.index(port) if port is not None else -1)
      services.append(service)
      batch,chunks,scaled,kb,n,row_service,tail = 1,1,0,1,1,1,0
      if ins.op==I.MM:
        n,kb,m = w[3]&65535,(w[3]>>16)&255,(w[5]>>16)&255
        wf,pair = (ins.flags>>I.WF_SHIFT)&3,bool(ins.flags&I.F_PAIR)
        chunks = kb if wf==I.W8 else ceil(kb/2)
        if chunks>t.fifo_chunks:
          raise ValueError("one matrix row must fit the prefetch FIFO")
        batch = min(t.batch_rows,max(1,n//64),max(1,t.fifo_chunks//(2*chunks)))
        scaled = int(not bool(ins.flags&I.F_UNIT))
        drain = ceil(m/min(t.epilogue_lanes,cfg.LANES//gcd(w[5]&65535,cfg.LANES)))
        row_service = max((chunks if pair else kb)*ceil(m/cfg.MCOLS),drain)
        tail = min(cfg.D//8,t.chain_length)+cfg.MCOLS+5+9+(6 if ins.flags&I.F_ACC else 0)+drain
        self.matrix_batch_sizes.append(batch)
      batches.append(batch)
      chunkses.append(chunks)
      scale_flags.append(scaled)
      kbs.append(kb)
      ns.append(n)
      row_services.append(row_service)
      tails.append(tail)
    parameters = CommandTiming(latencies,services,port_ids,batches,chunkses,
        scale_flags,kbs,ns,row_services,tails)

    def csr(rows):
      offsets,flat = [0],[]
      for row in rows:
        flat.extend(row)
        offsets.append(len(flat))
      return offsets,flat or [ncommands]

    # MM starts on DRAM dependencies; TMEM/ACT dependencies gate its compute.
    # Prefetch and compute ordering belong to each MatrixEngine, not this interface.
    starts = [dram_deps[i] if kinds[i]==W.U_MXU else deps[i] for i in range(ncommands)]
    offsets,indices = csr(starts)
    compute_offsets,compute_indices = csr(deps)

    @net.func(name=prefix+"_select_ready",args=("i64",),results=("i64",))
    def select_ready(unit):
      slot_places = array(slots)
      kind,begin,end = array(kinds),array(offsets[:-1]),array(offsets[1:])
      dep_indices = array(indices)
      selected,oldest = subi(0,1),addi(ncommands,1)
      selected_enabled = const_i64(0)==0
      for s in range(t.window):
        token = peek(slot_places[s])
        idx = token.get("id")
        active = select(token.get("active")==1,kind[idx]==unit,False,typ="i1")
        enabled = active
        for d in range(select(enabled,begin[idx],0),select(enabled,end[idx],0)):
          enabled = select(enabled,is_done(dep_indices[d]),False,typ="i1")
        # Matrix admission is FIFO across its engine pool. An engine must not
        # fill its input FIFO with a younger command waiting for an older,
        # still-unadmitted command. Engines execute admitted work independently.
        candidate = select(unit==W.U_MXU,active,enabled,typ="i1")
        eligible = select(candidate,idx<oldest,False,typ="i1")
        selected = select(eligible,s,selected,typ="i64")
        oldest = select(eligible,idx,oldest)
        selected_enabled = select(eligible,enabled,selected_enabled,typ="i1")
      func_return(select(selected_enabled,selected,subi(0,1),typ="i64"))

    @net.transition(prefix+"_initialize_window")
    def initialize_window():
      take(initialize)
      for s in range(t.window):
        emit(window,create().set("slot",s))

    @net.transition(prefix+"_dispatch")
    def dispatch():
      guard(count(cursor)>0)
      guard(count(window)>0)
      idx = peek(cursor).get("id")
      guard(idx<ncommands)
      bs,ps = array(barriers),array(pipeline_delays)
      barrier = select(bs[idx]==ncommands,nsig,bs[idx])
      guard(is_done(barrier))
      take(cursor)
      token = take(window).set("id",idx).set("active",1)
      emit(pipeline,token,delay=sitofp(ps[idx]))
      emit(cursor,create().set("id",idx+1),delay=sitofp(array(dispatch_delays)[idx]))

    @net.transition(prefix+"_arrive")
    def arrive():
      guard(count(pipeline)>0)
      token = take(pipeline)
      place = array(slots)[token.get("slot")]
      take(place)
      emit(place,token)
      emit(array(pending)[array(kinds)[token.get("id")]],create())

    interface = CommandInterface(slots,pending,window,select_ready,is_done,publish,
        compute_offsets,compute_indices)
    units = [ExecutionUnit(net,prefix+f"_unit_{u}",u,interface,parameters,memory)
             for u in (W.U_DMA,W.U_Q,W.U_VPU,W.U_COLL,5)]
    matrices = [MatrixEngine(net,prefix+f"_mxu_{engine}",cfg,t,interface,
        parameters,memory,W.U_MXU) for engine in range(t.matrix_engines)]
    self.components[sid] = {"memory":memory,"units":units,"matrix_engines":matrices}

    @net.transition(prefix+"_mark_done")
    def mark_done():
      guard(count(signals)>0)
      idx = peek(signals).get("id")
      bank = array(banks)[divi(idx,64)]
      guard(count(bank)>0)
      take(signals)
      token = take(bank)
      emit(bank,token.set("bits",ori(token.get("bits"),shli(1,remi(idx,64)))))
      if idx==ncommands-1:
        emit(finished,create())

    self.slice_done.append(finished)
