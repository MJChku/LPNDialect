"""Command and row dataflow for the pinned FeSens/openTPU systolic design.

No RTL observations enter this model. Numerical execution supplies addresses
and control flow; LPN tokens supply all timing and resource synchronization.
"""

from dataclasses import dataclass
from math import ceil, gcd

from lpnlang_mlir import NetBuilder


def bank_write_cycles(width,columns,stride,lanes):
  """One write per TMEM bank per cycle, including strided column collisions."""
  banks=[0]*lanes
  for row in range(width):
    for column in range(columns):
      banks[(row+column*stride)%lanes]+=1
  return max(banks,default=0)


def matrix_batch_rows(rows,chunks,timing):
  """Group whole parallel waves; aggregation must not serialize row engines."""
  parallel=timing.matrix_row_lanes
  batch=min(max(timing.batch_rows,parallel),max(parallel,rows//64),
            max(1,timing.fifo_chunks//(2*chunks)))
  return batch//parallel*parallel if batch>=parallel else batch


def buffered_quant_cycles(rows,chunks,capacity,banks,scale_words=0):
  """Finite row/FIFO credits for the buffered quantizer's two overlapping passes.

  Each chunk takes one read and one pop. Read-to-scale-to-pop is 43 clocks;
  the output path drains in ten clocks, plus four for QST acknowledgement.
  QST sends scale_words on its separate store port before popping each row.
  This computes command metadata, without adding per-cycle simulator events.
  """
  if not rows or not chunks:
    return 3
  if capacity<chunks or banks<2:
    raise ValueError('one row must fit the quantizer FIFO and row banks')
  popped=[]
  row_ends=[]
  read=0
  for row in range(rows):
    bank_free=row_ends[row-banks]+1 if row>=banks else 1
    for chunk in range(chunks):
      index=row*chunks+chunk
      free=popped[index-capacity]+1 if index>=capacity else 1
      read=max(read+1,free,bank_free)
    first=max(read+43,(popped[-1]+2 if popped else 1))+scale_words
    popped.extend(range(first,first+chunks))
    row_ends.append(popped[-1])
  return popped[-1]+10+(4 if scale_words else 0)


@dataclass(frozen=True)
class Timing:
  dram_latency: int = 8
  dram_bytes_per_cycle: int = 128
  window: int = 32
  fifo_chunks: int = 128
  chain_length: int = 16
  epilogue_lanes: int = 2
  quant_buffer_chunks: int = 1024
  batch_rows: int = 1
  matrix_engines: int = 1
  # Hypothetical FPGA datapaths; defaults preserve the validated RTL model.
  matrix_row_lanes: int = 1
  dram_port_bytes: int = 0
  scale_lanes: int = 1
  unified_dram: bool = False
  scalar_store_bytes: int = 0
  # Opt-in hardware variant: tagged row banks and QST's prescaled row FIFO.
  # Zero retains the original validated quantizer approximation.
  quant_row_banks: int = 0
  quant_store_buffered: bool = False

  def __post_init__(self):
    if min(self.dram_latency,self.dram_bytes_per_cycle,self.window,self.fifo_chunks,
           self.chain_length,self.epilogue_lanes,self.batch_rows,self.matrix_engines)<1:
      raise ValueError("timing parameters must be positive")
    if not isinstance(self.matrix_engines,int):
      raise ValueError("matrix_engines must be an integer")
    if self.quant_buffer_chunks<0:
      raise ValueError("quant buffer size must be nonnegative")
    if (not isinstance(self.quant_row_banks,int) or self.quant_row_banks<0 or
        (self.quant_row_banks and (self.quant_row_banks<2 or self.quant_row_banks>256 or
                                  self.quant_row_banks&(self.quant_row_banks-1)))):
      raise ValueError('quant_row_banks must be zero or a power of two from 2 to 256')
    if self.quant_store_buffered and not self.quant_row_banks:
      raise ValueError('buffered QST requires explicit row banks')
    for name in ('matrix_row_lanes','scale_lanes','dram_port_bytes','scalar_store_bytes'):
      value=getattr(self,name)
      if not isinstance(value,int) or value<(0 if name in ('dram_port_bytes','scalar_store_bytes') else 1):
        raise ValueError(f'{name} must be an integer in its valid range')


class DRAM:
  def __init__(self,net,prefix,timing):
    self.spec = timing
    self.b = net.place(prefix+"_port_b",initial_tokens=1)
    self.a = net.place(prefix+"_port_a",initial_tokens=1)
    self.sw = self.b if timing.unified_dram else net.place(prefix+"_scalar_write",initial_tokens=1)


class SystolicArray:
  def __init__(self,net,prefix,timing):
    self.compute = net.place(prefix+"_compute",initial_tokens=1)
    self.commands = net.place(prefix+"_commands",initial_tokens=2)
    self.buffers = buffers = net.place(prefix+"_buffers",initial_tokens=1)
    initialize = net.place(prefix+"_initialize_buffers",initial_tokens=1)
    capacity = timing.fifo_chunks
    # Requests have different chunk counts. A single free-capacity counter
    # avoids rounding every packet up to the largest command in the program.
    @net.transition(prefix+"_initialize_buffer_capacity")
    def initialize_buffer_capacity():
      guard(count(initialize)>0)
      take(initialize)
      token = take(buffers)
      emit(buffers,token.set('free',capacity))
    self.spec = timing


VECTOR_DRAIN_RANK=100


def vector_rank(ins,I):
  """RTL n_slots ordering; reductions additionally require an empty pipeline."""
  function=(ins.w[5]>>16)&255
  if function in I.REDUCE:
    return VECTOR_DRAIN_RANK
  return {I.V_ADD:1,I.V_SUB:1,I.V_RSUB:1,I.V_MUL:1,I.V_OUTER:1,
          I.V_EXP2:9,I.V_EXP2SUB:9,I.V_RECIP:6,I.V_RSQRT:10,I.V_LOG2:10}.get(function,0)


class VectorPipeline:
  """Track live VOPs: reductions drain, shorter tap paths wait for longer ones."""
  def __init__(self,net,prefix):
    self.state=state=net.place(prefix+'_state',initial_tokens=1)
    self.completed=completed=net.place(prefix+'_completed')

    @net.transition(prefix+'_retire')
    def retire():
      guard(count(completed)>0)
      take(completed)
      token=take(state)
      emit(state,token.set('live',token.get('live')-1))


class OpenTPU:
  supports_multiple_matrix_engines = False

  def __init__(self,workload,timing=None):
    from opentpu import isa as I, waits as W
    self.workload,self.timing = workload,timing or Timing()
    if self.timing.matrix_engines!=1 and not self.supports_multiple_matrix_engines:
      raise ValueError("multiple matrix engines require CompactOpenTPU")
    self.matrix_batch_sizes = []
    cfg,t = workload.config,self.timing
    if cfg.SEQ_WC:
      raise ValueError("the timing model currently requires the range-scoreboard sequencer")
    if t.dram_bytes_per_cycle>(t.dram_port_bytes or cfg.D):
      raise ValueError("bandwidth exceeds the simulated DRAM port width")
    self.net = net = NetBuilder("fesens_opentpu")
    self.finished = net.place("program_completed",observable=True)
    measurement = net.place("measurement",initial_tokens=1)

    @net.transition
    def begin():
      take(measurement)

    self.slice_done = []
    for sid,commands in enumerate(workload.commands):
      self._slice(sid,commands,I,W)
    slice_done = self.slice_done

    @net.transition
    def complete():
      for done in slice_done:
        take(done)
      emit(self.finished,create())

  def _slice(self,sid,commands,I,W):
    cfg,t,net = self.workload.config,self.timing,self.net
    prefix = f"s{sid}"
    memory = DRAM(net,prefix+"_dram",t)
    array = SystolicArray(net,prefix+"_mxu",t)
    units = [net.place(prefix+f"_unit_{u}",initial_tokens=1) for u in range(5)]
    vector = VectorPipeline(net,prefix+'_vpu')
    window = net.place(prefix+"_window",initial_tokens=t.window)
    order = [net.place(prefix+f"_order_{i}",initial_tokens=int(i==0)) for i in range(len(commands)+1)]
    dispatched = [net.place(prefix+f"_dispatched_{i}") for i in range(len(commands))]
    done = [net.place(prefix+f"_done_{i}") for i in range(len(commands))]
    compute_order,prefetch_order,register_barrier = None,None,None
    for idx,command in enumerate(commands):
      ins,fp = command.instruction,command.footprint
      unit = W.UNIT.get(ins.op,-1)
      predecessors = [j for j,c in enumerate(commands[:idx])
          if W.conflict(fp,c.footprint) and
          (unit!=W.U_MXU or W.UNIT.get(c.instruction.op,-2)!=W.U_MXU)]
      # HALT drains all units, including independent work without memory conflicts.
      if ins.op==I.HALT:
        predecessors = list(range(idx))
      dependencies = [done[j] for j in predecessors]
      self._dispatch(prefix,idx,ins,order,dispatched,window,I,register_barrier)
      if ins.op==I.RLD:
        register_barrier = done[idx]
      if ins.op==I.MM:
        dram_deps = [done[j] for j,c in enumerate(commands[:idx])
                     if W.conflict(fp,c.footprint,dram_only=True)]
        compute_order,prefetch_order = self._matrix(prefix,idx,ins,dispatched[idx],done[idx],
            window,memory,array,dependencies,dram_deps,compute_order,prefetch_order)
      elif unit>=0:
        self._simple(prefix,idx,ins,dispatched[idx],done[idx],window,
                     units[unit],memory,dependencies,I,vector if unit==W.U_VPU else None)
      else:
        trigger,result = dispatched[idx],done[idx]

        @net.transition(prefix+f"_control_{idx}")
        def control(trigger=trigger,result=result,dependencies=dependencies):
          guard(count(trigger)>0)
          for dep in dependencies:
            guard(count(dep)>0)
          take(trigger)
          emit(result,create())
          emit(window,create())
    self.slice_done.append(done[-1])

  def _dispatch(self,prefix,idx,ins,order,dispatched,window,I,register_barrier):
    net = self.net
    previous,following,trigger = order[idx],order[idx+1],dispatched[idx]
    delay = 2 if ins.op==I.LOOP else 1
    pipeline = 5 if ins.op in I.OPNAMES and ins.op>=I.LD else 0

    @net.transition(prefix+f"_dispatch_{idx}")
    def dispatch():
      guard(count(previous)>0)
      guard(count(window)>0)
      if register_barrier is not None:
        guard(count(register_barrier)>0)
      take(previous)
      take(window)
      emit(following,create(),delay=float(delay))
      emit(trigger,create(),delay=float(pipeline))

  def _simple_timing(self,ins,memory,I):
    cfg,t = self.workload.config,self.timing
    w = ins.w
    if ins.op in (I.LD,I.ST):
      service = max(ceil(w[2]/min(cfg.LANES,cfg.D//4)),ceil(4*w[2]/t.dram_bytes_per_cycle))
      latency = service+(t.dram_latency+6 if ins.op==I.LD else 5)
      port = memory.b
    elif ins.op==I.QACT:
      rows,kb = w[1]&255,w[1]>>16
      chunks = ceil(kb*cfg.D/cfg.LANES)
      # QACT's row buffer overlaps pass 0 of the following row with pass 1.
      latency = ((rows+1)*chunks+60 if t.quant_buffer_chunks>=2*chunks else
                 rows*(2*chunks+60))
      if t.quant_row_banks and chunks<=t.quant_buffer_chunks:
        latency=buffered_quant_cycles(rows,chunks,t.quant_buffer_chunks,t.quant_row_banks)
      port,service = None,0
    elif ins.op==I.QST:
      rows,kb = w[3]&65535,w[3]>>16
      elems = kb*cfg.D
      width = 4 if w[6]==1 else 1
      store = (ceil(elems*4/t.scalar_store_bytes)
               if t.scalar_store_bytes else ceil(elems/width))
      latency = rows*(ceil(elems/cfg.LANES)+store+60)
      chunks=ceil(elems/cfg.LANES)
      half=bool(ins.flags&1 and ins.flags&2)
      if (t.quant_store_buffered and t.scalar_store_bytes==4*cfg.LANES and
          not half and chunks<=t.quant_buffer_chunks):
        scale_words=1 if ins.flags&1 else 1+ceil(kb/4)
        latency=buffered_quant_cycles(rows,chunks,t.quant_buffer_chunks,t.quant_row_banks,scale_words)
      # The shared memory link is occupied by byte-enabled stores, not by
      # scratchpad reads, max reduction or FP conversion in the quantizer.
      # Charging its complete latency here stalls unrelated matrix prefetch.
      port,service = memory.sw,(rows*store if t.unified_dram else latency)
    elif ins.op==I.VOP:
      rows,cols = w[3]&65535,w[3]>>16
      function = (w[5]>>16)&255
      composite = function in (I.V_EXP2,I.V_RECIP,I.V_RSQRT,I.V_LOG2,I.V_EXP2SUB)
      if composite and cfg.LANES==8:
        # Current RTL shares SE's three stages: 25-cycle loops on eight lanes.
        passes = {I.V_EXP2:3,I.V_EXP2SUB:3,I.V_RECIP:2,I.V_RSQRT:4,I.V_LOG2:4}[function]
        latency = rows*ceil(cols/8)*passes+25*passes+5
      else:
        lanes = max(1,cfg.LANES//4) if composite else cfg.LANES
        slots = {I.V_EXP2:9,I.V_EXP2SUB:9,I.V_RECIP:6,I.V_RSQRT:10,I.V_LOG2:10}.get(function,0)
        latency = rows*ceil(cols/lanes)+(7*slots+8 if composite else 12)
      if function==I.V_OUTER:
        latency += max(2,ceil(cols/cfg.LANES))
      if function in I.REDUCE:
        if cfg.LANES>=32:
          if function in (I.V_RSUM,I.V_RSSQ,I.V_RDOT):
            partial_chunks=64//cfg.LANES
            chunks=rows*ceil(cols/64)*partial_chunks
            # The real wide datapath preserves 64 ordered partial sums. Its
            # four-stage adder limits recurrence; the six-level tree drains
            # once, after the last issued partial (read/product/add/tree/retire).
            interval=4//partial_chunks
            latency=(chunks-1)*interval+36
            if function==I.V_RDOT and rows<=256 and w[4]&65535==1:
              latency+=ceil(rows/cfg.LANES)
          else:
            latency=rows*ceil(cols/cfg.LANES)+(cfg.LANES.bit_length()-1)+5
            if function==I.V_ARGMAX:
              latency+=4
        else:
          # The original folding tree streams rows and drains once.
          if function in (I.V_RSUM,I.V_RSSQ,I.V_RDOT):
            latency = rows*ceil(cols/64)*(64//cfg.LANES)+12
          latency += 64//cfg.LANES+12
          if function==I.V_RDOT and rows<=256:
            latency += ceil(rows/cfg.LANES)
      port,service = None,0
    elif ins.op==I.RLD:
      latency,port,service = 4,None,0
    elif ins.op in (I.BAR,I.GATHER):
      if cfg.S!=1:
        raise ValueError("multi-slice collective timing is not implemented yet")
      latency = 2 if ins.op==I.BAR else ceil((w[2]&65535)*(w[2]>>16)/cfg.LANES)+4
      port,service = None,0
    else:
      raise ValueError(f"unsupported timing opcode {ins.op:#x}")
    return latency,port,service

  def _vector_interval(self,ins,I):
    rows,cols=ins.w[3]&65535,ins.w[3]>>16
    function=(ins.w[5]>>16)&255
    lanes=self.workload.config.LANES
    if lanes==8 and vector_rank(ins,I)>1:
      # COMP8 has three shared stages and a 25-cycle recirculation loop.
      # This is its aggregate issue bandwidth; return-slot collisions are
      # approximated rather than advancing one simulator event per clock.
      passes={I.V_EXP2:3,I.V_EXP2SUB:3,I.V_RECIP:2,I.V_RSQRT:4,I.V_LOG2:4}[function]
      return rows*ceil(cols/lanes)*passes+2
    if vector_rank(ins,I)>1:
      lanes=max(1,lanes//4)
    return rows*ceil(cols/lanes)+2+(max(2,ceil(cols/self.workload.config.LANES))
        if function==I.V_OUTER else 0)

  def _simple(self,prefix,idx,ins,trigger,result,window,unit,memory,deps,I,vector=None):
    net = self.net
    latency,port,service = self._simple_timing(ins,memory,I)
    rank=vector_rank(ins,I) if vector is not None else 0
    interval=(self._vector_interval(ins,I) if vector is not None and rank<VECTOR_DRAIN_RANK else latency)

    @net.transition(prefix+f"_execute_{idx}")
    def execute():
      guard(count(trigger)>0)
      guard(count(unit)>0)
      for dep in deps:
        guard(count(dep)>0)
      if port is not None:
        guard(count(port)>0)
      if vector is not None:
        state=peek(vector.state)
        if rank==VECTOR_DRAIN_RANK:
          guard(state.get('live')==0)
        else:
          guard(select(state.get('live')==0,True,state.get('rank')<=rank,typ='i1'))
      take(trigger)
      credit = take(unit)
      emit(unit,credit,delay=float(interval))
      if vector is not None:
        state=take(vector.state)
        emit(vector.state,state.set('live',state.get('live')+1).set('rank',rank))
        emit(vector.completed,create(),delay=float(latency))
      if port is not None:
        bus = take(port)
        emit(port,bus,delay=float(service))
      emit(result,create(),delay=float(latency))
      emit(window,create(),delay=float(latency))

  def _matrix(self,prefix,idx,ins,trigger,result,window,memory,array,deps,dram_deps,previous_compute,previous_prefetch):
    from opentpu import isa as I
    cfg,t,net = self.workload.config,self.timing,self.net
    w,flags = ins.w,ins.flags
    scaled = int(not bool(flags&I.F_UNIT))
    n,kb,m = w[3]&65535,(w[3]>>16)&255,(w[5]>>16)&255
    wf,pair = (flags>>I.WF_SHIFT)&3,bool(flags&I.F_PAIR)
    chunks = kb if wf==I.W8 else ceil(kb/2)
    # Keep two packets in the FIFO so grouping does not introduce a DRAM
    # round-trip bubble between batches. Small N uses single-row events.
    batch = matrix_batch_rows(n,chunks,t)
    self.matrix_batch_sizes.append(batch)
    if chunks>t.fifo_chunks:
      raise ValueError("one matrix row must fit the prefetch FIFO")
    groups = ceil(m/cfg.MCOLS)
    steps = chunks if pair else kb
    drain_width = min(t.epilogue_lanes,cfg.MCOLS,cfg.LANES//gcd(w[5]&65535,cfg.LANES))
    drain = ceil(m/drain_width)
    # ACC's multiply/add stages increase result latency, not the row initiation
    # interval: read/modify/write is pipelined alongside subsequent rows.
    pipeline = min(cfg.D//8,t.chain_length)+cfg.MCOLS+5+9+(6 if flags&I.F_ACC else 0)
    row_service = max(steps*groups,drain)
    write_full = bank_write_cycles(batch,m,w[5]&65535,cfg.LANES)
    write_tail = bank_write_cycles(n%batch or batch,m,w[5]&65535,cfg.LANES)
    row_bytes = chunks*cfg.D+(max(4,w[6]) if scaled and t.unified_dram else 0)
    name = prefix+f"_mm_{idx}"
    buffers = array.buffers
    request,ready = net.place(name+"_request"),net.place(name+"_ready")
    compute_cursor = net.place(name+"_compute_cursor")
    prefetch_done,compute_done = net.place(name+"_prefetched"),net.place(name+"_issued")

    @net.transition(name+"_start")
    def start():
      guard(count(trigger)>0)
      guard(count(array.commands)>0)
      for dep in dram_deps:
        guard(count(dep)>0)
      if previous_prefetch is not None:
        guard(count(previous_prefetch)>0)
      take(trigger)
      take(array.commands)
      emit(request,create())
      emit(compute_cursor,create())

    @net.transition(name+"_prefetch")
    def prefetch():
      guard(count(request)>0)
      guard(count(buffers)>0)
      observed = peek(request)
      needed = muli(select(observed.get('row')+batch<n,batch,n-observed.get('row')),chunks)
      guard(peek(buffers).get('free')>=needed)
      guard(count(memory.b)>0)
      guard(count(memory.a)>=scaled)
      token = take(request)
      row = token.get("row")
      width = select(row+batch<n,batch,n-row)
      capacity = take(buffers)
      emit(buffers,capacity.set('free',capacity.get('free')-needed))
      bus = take(memory.b)
      transfer_cycles = divi(muli(width,row_bytes)+t.dram_bytes_per_cycle-1,t.dram_bytes_per_cycle)
      scale_cycles = divi(muli(muli(width,kb),scaled)+t.scale_lanes-1,t.scale_lanes)
      service = sitofp(transfer_cycles)
      emit(memory.b,bus,delay=service)
      if scaled:
        scales = take(memory.a)
        emit(memory.a,scales,delay=sitofp(scale_cycles))
      arrival = sitofp(select(transfer_cycles>scale_cycles,transfer_cycles,scale_cycles))
      emit(ready,token.set("width",width),delay=arrival+float(t.dram_latency+4))
      if row+batch<n:
        emit(request,token.set("row",row+batch),delay=service)
      else:
        emit(prefetch_done,create(),delay=arrival)

    @net.transition(name+"_compute")
    def compute():
      guard(count(ready)>0)
      guard(count(compute_cursor)>0)
      guard(count(array.compute)>0)
      guard(count(buffers)>0)
      for dep in deps:
        guard(count(dep)>0)
      if previous_compute is not None:
        guard(count(previous_compute)>0)
      token = take(ready)
      take(compute_cursor)
      port = take(array.compute)
      compute_cycles = muli(divi(token.get("width")+t.matrix_row_lanes-1,t.matrix_row_lanes),row_service)
      write_cycles = select(token.get("width")==batch,write_full,write_tail)
      service = sitofp(select(compute_cycles>write_cycles,compute_cycles,write_cycles))
      emit(array.compute,port,delay=service)
      capacity = take(buffers)
      emit(buffers,capacity.set('free',capacity.get('free')+muli(token.get('width'),chunks)))
      if token.get("row")+batch<n:
        emit(compute_cursor,create(),delay=service)
      else:
        emit(compute_done,create(),delay=service)
        # Batches finish in order and intermediate rows have no consumers:
        # publish command completion directly from the last batch. This
        # avoids per-row output events without changing resource timing.
        emit(result,create(),delay=service+float(pipeline+drain))
        emit(array.commands,create(),delay=service+float(pipeline+drain))
        emit(window,create(),delay=service+float(pipeline+drain))
    return compute_done,prefetch_done

  def build(self):
    return self.net.build()
