"""Reusable command-window and matrix-row transitions for long executable traces.

The net size grows with completion signals and immutable command metadata, rather
than transition code. The existing event-driven LPN scheduler is unchanged.
"""

from bisect import bisect_left
from math import ceil, gcd

from .model import OpenTPU, DRAM, SystolicArray


def dependencies(commands,I,W):
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
      conflicts = {j for j in conflicts if unit!=W.UNIT.get(commands[j].instruction.op,-2)
                   or unit==W.U_VPU}
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
  def _slice(self,sid,commands,I,W):
    cfg,t,net = self.workload.config,self.timing,self.net
    prefix = f"s{sid}"
    ncommands = len(commands)
    if not ncommands:
      raise ValueError("empty command trace")
    memory = DRAM(net,prefix+"_dram",t)
    units = [net.place(prefix+f"_unit_{u}",initial_tokens=1) for u in range(5)]
    matrix_ids = [i for i,c in enumerate(commands) if c.instruction.op==I.MM]
    matrix_index = {idx:rank for rank,idx in enumerate(matrix_ids)}
    nsig = ncommands+2*len(matrix_ids)
    # Immutable flag banks replace thousands of individual completion places.
    # A bank latch is always present; delayed completion messages update it
    # atomically. This keeps the scheduler's place scans bounded and retains
    # out-of-order VPU completion without assuming an in-order retire counter.
    banks = [net.place(prefix+f"_flags_{b}",initial_tokens=1) for b in range(nsig//64+1)]
    signals = net.place(prefix+"_signals")
    finished = net.place(prefix+"_halted")
    control_resource = net.place(prefix+"_control_resource",initial_tokens=1)
    prefetched = [ncommands+r for r in range(len(matrix_ids))]+[nsig]
    issued = [ncommands+len(matrix_ids)+r for r in range(len(matrix_ids))]+[nsig]

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
    request,ready = net.place(prefix+"_request"),net.place(prefix+"_ready")
    ports = [memory.b,memory.sw]
    deps,dram_deps = dependencies(commands,I,W)
    kinds,latencies,services,port_ids = [],[],[],[]
    batches,chunkses,scale_flags,kbs,ns,row_services,tails = [],[],[],[],[],[],[]
    barriers,prev_matrices,dispatch_delays,pipeline_delays = [],[],[],[]
    last_register,last_matrix = ncommands,len(matrix_ids)
    matrix_ranks = [matrix_index.get(i,0) for i in range(ncommands)]
    for idx,c in enumerate(commands):
      ins,w = c.instruction,c.instruction.w
      u = W.UNIT.get(ins.op,5)
      kinds.append(u)
      barriers.append(last_register)
      prev_matrices.append(last_matrix)
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
        last_matrix = matrix_index[idx]
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
    array_unit = SystolicArray(net,prefix+"_mxu",t,max(b*c for b,c in zip(batches,chunkses)))

    def csr(rows):
      offsets,flat = [0],[]
      for row in rows:
        flat.extend(row)
        offsets.append(len(flat))
      return offsets,flat or [ncommands]

    # MM starts on DRAM dependencies; TMEM/ACT dependencies gate its compute.
    # Previous MM prefetch has its own signal, handled separately below.
    starts = [dram_deps[i] if kinds[i]==W.U_MXU else deps[i] for i in range(ncommands)]
    offsets,indices = csr(starts)
    compute_offsets,compute_indices = csr(deps)

    @net.func(name=prefix+"_select_ready",args=("i64",),results=("i64",))
    def select_ready(unit):
      slot_places = array(slots)
      kind,begin,end = array(kinds),array(offsets[:-1]),array(offsets[1:])
      dep_indices = array(indices)
      previous,pre = array(prev_matrices),array(prefetched)
      selected,oldest = subi(0,1),addi(ncommands,1)
      for s in range(t.window):
        token = peek(slot_places[s])
        idx = token.get("id")
        enabled = select(token.get("active")==1,kind[idx]==unit,False,typ="i1")
        for d in range(select(enabled,begin[idx],0),select(enabled,end[idx],0)):
          enabled = select(enabled,is_done(dep_indices[d]),False,typ="i1")
        matrix_ready = select(unit==W.U_MXU,is_done(pre[previous[idx]]),True,typ="i1")
        eligible = select(enabled,select(matrix_ready,idx<oldest,False,typ="i1"),False,typ="i1")
        selected = select(eligible,s,selected,typ="i64")
        oldest = select(eligible,idx,oldest)
      func_return(selected)

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

    for u in (W.U_DMA,W.U_Q,W.U_VPU,W.U_COLL,5):
      resource = units[u] if u<5 else control_resource

      @net.transition(prefix+f"_execute_{u}")
      def execute(u=u,resource=resource):
        guard(count(resource)>0)
        guard(count(pending[u])>0)
        chosen = select_ready(const_i64(u))
        guard(chosen>=0)
        slot_places = array(slots)
        observed = peek(slot_places[chosen])
        idx = observed.get("id")
        bus_id = array(port_ids)[idx]
        buses = array(ports)
        guard(select(bus_id>=0,count(buses[select(bus_id>=0,bus_id,0)])>0,True,typ="i1"))
        token = take(slot_places[chosen])
        take(pending[u])
        emit(slot_places[chosen],create())
        latency = sitofp(array(latencies)[idx])
        if u<5:
          credit = take(resource)
          emit(resource,credit,delay=latency)
        if bus_id>=0:
          bus = take(buses[bus_id])
          emit(buses[bus_id],bus,delay=sitofp(array(services)[idx]))
        publish(idx,latency)
        emit(window,token,delay=latency)

    @net.transition(prefix+"_matrix_start")
    def matrix_start():
      guard(count(array_unit.commands)>0)
      guard(count(pending[W.U_MXU])>0)
      chosen = select_ready(const_i64(W.U_MXU))
      guard(chosen>=0)
      place = array(slots)[chosen]
      token = take(place)
      take(pending[W.U_MXU])
      emit(place,create())
      take(array_unit.commands)
      emit(request,token.set("row",0))

    @net.transition(prefix+"_prefetch")
    def prefetch():
      guard(count(request)>0)
      guard(count(array_unit.buffers)>0)
      guard(count(memory.b)>0)
      idx = peek(request).get("id")
      scaled = array(scale_flags)[idx]
      guard(count(memory.a)>=scaled)
      token = take(request)
      row = token.get("row")
      n,batch,chunks = array(ns)[idx],array(batches)[idx],array(chunkses)[idx]
      width = select(row+batch<n,batch,n-row)
      take(array_unit.buffers)
      bus = take(memory.b)
      service = sitofp(divi(muli(muli(width,chunks),cfg.D)+t.dram_bytes_per_cycle-1,t.dram_bytes_per_cycle))
      emit(memory.b,bus,delay=service)
      if scaled>0:
        scales = take(memory.a)
        emit(memory.a,scales,delay=sitofp(muli(width,array(kbs)[idx])))
      emit(ready,token.set("width",width),delay=service+float(t.dram_latency+4))
      if row+batch<n:
        emit(request,token.set("row",row+batch),delay=service)
      else:
        publish(array(prefetched)[array(matrix_ranks)[idx]],service)

    @net.transition(prefix+"_compute")
    def compute():
      guard(count(ready)>0)
      guard(count(array_unit.compute)>0)
      idx = peek(ready).get("id")
      ci = array(compute_indices)
      enabled = const_i64(0)==0
      for d in range(array(compute_offsets)[idx],array(compute_offsets)[idx+1]):
        enabled = select(enabled,is_done(ci[d]),False,typ="i1")
      guard(enabled)
      guard(is_done(array(issued)[array(prev_matrices)[idx]]))
      token = take(ready)
      port = take(array_unit.compute)
      service = sitofp(muli(token.get("width"),array(row_services)[idx]))
      emit(array_unit.compute,port,delay=service)
      emit(array_unit.buffers,create())
      if token.get("row")+array(batches)[idx]>=array(ns)[idx]:
        publish(array(issued)[array(matrix_ranks)[idx]],service)
        latency = service+sitofp(array(tails)[idx])
        publish(idx,latency)
        emit(array_unit.commands,create(),delay=latency)
        emit(window,token,delay=latency)
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
