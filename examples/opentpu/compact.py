"""Reusable command-window and matrix-row transitions for long executable traces.

The net size grows with completion signals and immutable command metadata, rather
than transition code. The existing event-driven LPN scheduler is unchanged.
"""

from bisect import bisect_left
from math import ceil, gcd
from lpnlang_mlir import InputArray

from .model import OpenTPU, DRAM, bank_write_cycles, matrix_batch_rows, matrix_scale_requests, vector_rank, VECTOR_DRAIN_RANK
from .components import CommandInterface, CommandTiming, ExecutionUnit, MatrixEngine


def dependencies(commands,I,W,matrix_engines=1,streaming_producers=None):
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
  ancestors,compute,dram = [],[],[]
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
      # The RTL sequencer can send a younger ready command to a unit while an
      # older command waits for another unit. Preserve actual memory hazards,
      # without inventing completion edges between unrelated commands.
      # One matrix engine already orders its own prefetch and compute streams.
      conflicts = {j for j in conflicts if unit!=W.U_MXU
                   or W.UNIT.get(commands[j].instruction.op,-2)!=W.U_MXU
                   or matrix_engines>1}
    if streaming_producers is not None and streaming_producers[idx]>=0:
      conflicts.discard(streaming_producers[idx])
    compute.append(reduce_edges(conflicts))
    dram.append(reduce_edges(dc))
    complete_deps = conflicts | (dc if unit==W.U_MXU else set())
    ancestry = 0
    for j in complete_deps:
      ancestry |= ancestors[j]|(1<<j)
    ancestors.append(ancestry)
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

  def __init__(self,workload,timing=None,input_capacity=None,prefetch_profile=None):
    if input_capacity is not None and (not isinstance(input_capacity,int) or
        input_capacity<max(len(c) for c in workload.commands)):
      raise ValueError("input_capacity must cover every slice's command count")
    self.input_capacity = input_capacity
    self.input_tables = {}
    self.components = {}
    self.prefetch_profile=prefetch_profile
    super().__init__(workload,timing)

  def _table(self,name,values):
    if self.input_capacity is None:
      return values
    self.input_tables[name] = values
    return InputArray(name,values)

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
    cache_offsets,cache_prefix=[],[0]
    if t.pinned_cache_bytes:
      profile=self.prefetch_profile
      if (cfg.S!=1 or profile is None or profile.get('version')!=1
          or profile.get('cache_bytes')!=t.pinned_cache_bytes):
        raise ValueError('matching validated single-slice prefetch profile required')
      base=profile.get('base')
      if type(base) is not int or base<0 or base%64 or base+t.pinned_cache_bytes>cfg.DRAM_BYTES:
        raise ValueError('cache profile region is outside physical memory')
      row_hits=profile['hit_bytes_by_command_row']
      if len(row_hits)!=ncommands:raise ValueError('cache profile command count differs')
      for idx,c in enumerate(commands):
        ins,w=c.instruction,c.instruction.w
        n=w[3]&65535 if ins.op==I.MM else 0
        kb=(w[3]>>16)&255 if ins.op==I.MM else 0
        chunks=kb if ((ins.flags>>I.WF_SHIFT)&3)==I.W8 else ceil(kb/2)
        if len(row_hits[idx])!=n or any(type(v) is not int or v<0 or v>chunks*cfg.D or v%64 for v in row_hits[idx]):
          raise ValueError('invalid per-row cached byte count')
        for row,value in enumerate(row_hits[idx]):
          address=w[0]+row*w[4]
          covered=max(0,min(address+chunks*cfg.D,base+t.pinned_cache_bytes)-max(address,base))
          if value>covered:raise ValueError('cache hit exceeds the configured pinned region')
        cache_offsets.append(len(cache_prefix));cache_prefix.append(0)
        for value in row_hits[idx]:cache_prefix.append(cache_prefix[-1]+value)
    else:cache_offsets=[0]*ncommands
    memory = DRAM(net,prefix+"_dram",t)
    nsig = self.input_capacity or ncommands
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
    activation=None
    if t.activation_streaming:
      from .activation_stream import ActivationStream,stream_producers
      producers=stream_producers(commands,I,W,t)
      activation=ActivationStream(net,prefix+'_act',commands,I,t,self._table,is_done)
      producer_table=self._table(prefix+'_act_producers',producers)
    deps,dram_deps = dependencies(commands,I,W,t.matrix_engines,
                                producers if activation is not None else None)
    kinds,latencies,services,port_ids,ranks = [],[],[],[],[]
    batches,chunkses,scale_flags,kbs,ns,row_services,tails = [],[],[],[],[],[],[]
    scale_bytes = []
    retire_cycles=[]
    physical_scale_offsets,physical_scale_prefix=[],[0]
    write_full,write_tail = [],[]
    barriers,dispatch_delays,pipeline_delays = [],[],[]
    last_register = nsig
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
      if activation is not None and ins.op==I.QACT and 0<(w[1]>>16)<=t.quant_buffer_chunks and (w[1]&255)>0:
        latency+=2 # completion includes the final ACT RAM landing
      latencies.append(latency)
      rank=vector_rank(ins,I,t) if ins.op==I.VOP else 0
      ranks.append(rank)
      if ins.op==I.VOP:
        service=self._vector_interval(ins,I) if rank<VECTOR_DRAIN_RANK else latency
      port_ids.append(ports.index(port) if port is not None else -1)
      services.append(service)
      batch,chunks,scaled,kb,n,row_service,tail = 1,1,0,1,1,1,0
      retire=0
      if ins.op==I.MM:
        n,kb,m = w[3]&65535,(w[3]>>16)&255,(w[5]>>16)&255
        wf,pair = (ins.flags>>I.WF_SHIFT)&3,bool(ins.flags&I.F_PAIR)
        if activation is not None or t.physical_scale_lines: pair=False # parallel RTL consumes packed halves serially
        chunks = kb if wf==I.W8 else ceil(kb/2)
        if chunks>t.fifo_chunks:
          raise ValueError("one matrix row must fit the prefetch FIFO")
        batch = matrix_batch_rows(n,chunks,t)
        scaled = int(not bool(ins.flags&I.F_UNIT))
        drain = ceil(m/min(t.epilogue_lanes,cfg.MCOLS,cfg.LANES//gcd(w[5]&65535,cfg.LANES)))
        row_service = max((chunks if pair else kb)*ceil(m/cfg.MCOLS),drain)
        tail = min(cfg.D//8,t.chain_length)+cfg.MCOLS+5+(5+2*t.fp_mul_latency)+(
            t.fp_mul_latency+t.fp_add_latency if ins.flags&I.F_ACC else 0)+drain
        if t.matrix_retirement:
          # LLAT (registered pick, i2f48, Sw/Sa multiplies), then the
          # registered head switch. ACC also traverses rA, r0 and r1.
          extra=2+(3 if ins.flags&I.F_ACC else 0)
          tail+=extra
          retire=5+2*t.fp_mul_latency+drain+extra+(
              t.fp_mul_latency+t.fp_add_latency if ins.flags&I.F_ACC else 0)
        self.matrix_batch_sizes.append(batch)
      physical_scale_offsets.append(len(physical_scale_prefix));physical_scale_prefix.append(0)
      if ins.op==I.MM and t.physical_scale_lines:
        for row in range(n):
          physical_scale_prefix.append(physical_scale_prefix[-1]+32*matrix_scale_requests(ins,I,row,t.matrix_row_lanes))
      batches.append(batch)
      chunkses.append(chunks)
      scale_flags.append(scaled)
      kbs.append(kb)
      ns.append(n)
      row_services.append(row_service)
      tails.append(tail)
      retire_cycles.append(retire)
      scale_bytes.append(max(4,w[6]) if ins.op==I.MM and scaled else 0)
      write_full.append(bank_write_cycles(batch,m,w[5]&65535,cfg.LANES) if ins.op==I.MM else 0)
      write_tail.append(bank_write_cycles(n%batch or batch,m,w[5]&65535,cfg.LANES) if ins.op==I.MM else 0)
    parameter_values = [latencies,services,port_ids,batches,chunkses,scale_flags,kbs,ns,row_services,tails,scale_bytes,write_full,write_tail,cache_offsets,cache_prefix,retire_cycles,physical_scale_offsets,physical_scale_prefix]
    parameters = CommandTiming(*(self._table(prefix+"_"+name,values)
        for name,values in zip(CommandTiming.__dataclass_fields__,parameter_values)))
    metadata = self._table(prefix+"_metadata",[ncommands])

    def csr(rows):
      offsets,flat = [0],[]
      for row in rows:
        flat.extend(row)
        offsets.append(len(flat))
      return offsets,flat or [nsig]

    # MM starts on DRAM dependencies; TMEM/ACT dependencies gate its compute.
    # Prefetch and compute ordering belong to each MatrixEngine, not this interface.
    starts = [dram_deps[i] if kinds[i]==W.U_MXU else deps[i] for i in range(ncommands)]
    offsets,indices = csr(starts)
    compute_offsets,compute_indices = csr(deps)
    begins = self._table(prefix+"_starts_begin",offsets[:-1])
    ends = self._table(prefix+"_starts_end",offsets[1:])
    indices = self._table(prefix+"_starts_indices",indices)
    compute_offsets = self._table(prefix+"_compute_offsets",compute_offsets)
    compute_indices = self._table(prefix+"_compute_indices",compute_indices)
    kinds = self._table(prefix+"_kinds",kinds)
    barriers = self._table(prefix+"_register_barriers",barriers)
    dispatch_delays = self._table(prefix+"_dispatch_delays",dispatch_delays)
    pipeline_delays = self._table(prefix+"_pipeline_delays",pipeline_delays)

    @net.func(name=prefix+"_select_ready",args=("i64",),results=("i64",))
    def select_ready(unit):
      slot_places = array(slots)
      kind,begin,end = array(kinds),array(begins),array(ends)
      dep_indices = array(indices)
      selected,oldest = subi(0,1),addi(array(metadata)[0],1)
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
      guard(idx<array(metadata)[0])
      bs,ps = array(barriers),array(pipeline_delays)
      barrier = bs[idx]
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
    if activation is not None:
      interface.activation=activation
      interface.activation_producers=producer_table
      interface.activation_groups=self._table(prefix+'_act_groups',[
          (c.instruction.w[5]>>16)&255 if c.instruction.op==I.MM else 0 for c in commands])
    vector_ranks=self._table(prefix+'_vector_ranks',ranks)
    if t.vector_compatible_issue:
      from .vector_select import vector_selector_factory
      interface.vector_selector_factory=vector_selector_factory(net,slots,t.window,kinds,begins,ends,
          indices,metadata,is_done,vector_ranks)
    units = [ExecutionUnit(net,prefix+f"_unit_{u}",u,interface,parameters,memory,
                 vector_ranks if u==W.U_VPU else None,
                 command_queue=t.vector_command_queue if u==W.U_VPU else 0)
             for u in (W.U_DMA,W.U_Q,W.U_VPU,W.U_COLL,5)
             if not (u==W.U_VPU and t.vector_result_chunks)]
    vectors=[]
    if t.vector_result_chunks:
      from .vector_pool import VectorPoolEngine,result_shape
      shapes=[result_shape(c.instruction,I,cfg.LANES) if c.instruction.op==I.VOP else (0,0) for c in commands]
      if max(s[0] for s in shapes)>t.vector_result_chunks:
        raise ValueError('one vector command must fit its finite result store')
      result_chunks=self._table(prefix+'_vector_result_chunks',[s[0] for s in shapes])
      result_drains=self._table(prefix+'_vector_result_drains',[s[1] for s in shapes])
      result_port=net.place(prefix+'_vector_result_port',initial_tokens=1)
      vectors=[VectorPoolEngine(net,prefix+f'_vpu_{v}',interface,parameters,vector_ranks,
          result_chunks,result_drains,t.vector_result_chunks,result_port,W.U_VPU)
          for v in range(t.vector_engines)]
    matrices = [MatrixEngine(net,prefix+f"_mxu_{engine}",cfg,t,interface,
        parameters,memory,W.U_MXU)
        for engine in range(t.matrix_engines)]
    self.components[sid] = {"memory":memory,"units":units,"matrix_engines":matrices,"vector_engines":vectors}

    @net.transition(prefix+"_mark_done")
    def mark_done():
      guard(count(signals)>0)
      idx = peek(signals).get("id")
      bank = array(banks)[divi(idx,64)]
      guard(count(bank)>0)
      take(signals)
      token = take(bank)
      emit(bank,token.set("bits",ori(token.get("bits"),shli(1,remi(idx,64)))))
      if idx==array(metadata)[0]-1:
        emit(finished,create())

    self.slice_done.append(finished)
