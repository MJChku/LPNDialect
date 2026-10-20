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


def matrix_scale_requests(ins,I,row,parallel_rows):
  """Physical 32-byte A requests for one parallel core's logical weight row.

  Parallel cores clear PAIR and walk rows p, p+parallel_rows, ... . The
  issuer requests a line per advance, even when adjacent scales share it.
  """
  if ins.flags&I.F_UNIT:return 0
  w=ins.w;kb=(w[3]>>16)&255;k0=w[3]>>24;wf=(ins.flags>>I.WF_SHIFT)&3
  if ins.flags&I.F_RSC:return kb if wf==I.W8 else ceil(kb/2)
  offset,step=(4+k0,1) if wf==I.W8 else ((16+8*k0,8) if wf==I.WNV else (8+4*k0,4))
  address=w[1]+row*w[6]
  first=(address+offset)//32
  previous=(w[1]+(row-parallel_rows)*w[6]+offset+(kb-1)*step)//32 if row>=parallel_rows else -1
  extra=int(address//32!=first and address//32!=previous)
  return kb+extra


def matrix_batch_rows(rows,chunks,timing):
  """Group whole parallel waves; aggregation must not serialize row engines."""
  parallel=timing.matrix_row_lanes
  batch=min(max(timing.batch_rows,parallel),max(parallel,rows//64),
            max(1,timing.fifo_chunks//(2*chunks)))
  return batch//parallel*parallel if batch>=parallel else batch


def quant_scale_setup(mul_latency=2,add_latency=4,scale_lookup=False):
  # Registered TMEM read, prescale, max fold, scale pipeline and reader restart.
  return 3*mul_latency+11 if scale_lookup else 9*mul_latency+4*add_latency+9


def legacy_quant_cycles(rows,reads,writes,scale_words=0,mul_latency=2,add_latency=4,scale_lookup=False):
  """Two-pass reader: scale writes and both passes repeat for every row."""
  if not rows or not reads:
    return 3
  setup=quant_scale_setup(mul_latency,add_latency,scale_lookup)
  return rows*(reads+writes+setup+scale_words-2)+3*mul_latency+(12 if scale_words else 8)


def buffered_quant_cycles(rows,chunks,capacity,banks,scale_words=0,mul_latency=2,add_latency=4,scale_lookup=False):
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
    setup=quant_scale_setup(mul_latency,add_latency,scale_lookup)
    first=max(read+setup,(popped[-1]+2 if popped else 1))+scale_words
    popped.extend(range(first,first+chunks))
    row_ends.append(popped[-1])
  return popped[-1]+mul_latency+8+(4 if scale_words else 0)


def disjoint_sum_rows(ins,I):
  """Conservative RTL address-hull check before changing a sum's row order."""
  w=ins.w
  rows,cols=w[3]&65535,w[3]>>16
  drs,ars,brs=w[4]&65535,w[4]>>16,w[5]&65535
  dst_hi=w[0]+(rows-1)*drs+1
  a_hi=w[1]+(rows-1)*ars+cols
  b_hi=w[2]+(rows-1)*brs+cols
  return (dst_hi<=w[1] or a_hi<=w[0]) and ((w[5]>>16)&255!=I.V_RDOT or
      dst_hi<=w[2] or b_hi<=w[0])


def interleaved_sum_span(rows,chunks,contexts=4,adder_latency=4):
  """Last issue clock relative to the first, with bounded row contexts."""
  cycles=0
  for base in range(0,rows,contexts):
    width=min(contexts,rows-base)
    cycles+=chunks*width+(chunks-1)*max(0,adder_latency-width)
    if base+width<rows:
      cycles+=max(0,adder_latency-width)
  return max(0,cycles-1)


def buffered_store_cycles(words,address,depth,lanes,bandwidth):
  """otpu_dma ST: TMEM gather, SRUN=32 chunk bursts, registered drain/ack.

  Gather and write runs overlap through the real chunk buffer. A run waits
  until its chunks are gathered; successive writes have one setup cycle.
  The eight-cycle tail is the request/read/data/buffer and completion pipeline.
  Addresses are bytes in DRAM; TMEM segments are aligned in DRAM (W=min(D/4,L)).
  """
  if min(words,depth,lanes,bandwidth)<1 or address<0 or address%4:
    raise ValueError('positive ST dimensions and a word-aligned DRAM address required')
  width=min(depth//4,lanes);segment=4*width
  first=address//depth;last=ceil((address+4*words)/depth);chunks=last-first
  read_base=address//segment*segment;finish=0
  for begin in range(0,chunks,32):
    end=min(chunks,begin+32)
    end_address=min(address+4*words,(first+end)*depth)
    gathered=ceil((end_address-read_base)/segment)
    write=ceil((end-begin)*depth/min(depth,bandwidth))
    finish=max(gathered,finish+(1 if begin else 0))+write
  return finish+8


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
  quant_scale_lookup: bool = False
  scalar_recip_lookup: bool = False
  reciprocal_lookup_lanes: int = 0 # proposal: independently addressed exact ROM lanes
  pinned_cache_bytes: int = 0
  cache_read_bytes_per_cycle: int = 8192
  vector_row_contexts: int = 1
  vector_command_queue: int = 0
  matrix_retirement: bool = False
  physical_scale_lines: bool = False
  scale_cache_entries: int = 0
  vector_engines: int = 1
  vector_issue_groups: int = 1 # proposal: parallel 64-lane elementwise issue groups
  vector_result_chunks: int = 0
  vector_compatible_issue: bool = False
  activation_streaming: bool = False
  fp_mul_latency: int = 2
  fp_add_latency: int = 4
  dma_store_runs: bool = False # opt-in correction: buffered 32-chunk ST write runs

  def __post_init__(self):
    if type(self.dma_store_runs) is not bool:
      raise ValueError('dma_store_runs must be a boolean')
    if type(self.vector_issue_groups) is not int or self.vector_issue_groups not in (1,2,4,8):
      raise ValueError('vector issue groups must be 1/2/4/8')
    if self.vector_issue_groups>1 and (self.vector_engines!=1 or self.vector_result_chunks or self.vector_compatible_issue):
      raise ValueError('wide elementwise issue currently requires one original VPU command path')
    if type(self.scale_cache_entries) is not int or self.scale_cache_entries not in (0,16):
      raise ValueError('scale cache entries must be zero or sixteen')
    if self.scale_cache_entries and (not self.physical_scale_lines or not self.unified_dram or
                                   self.matrix_engines!=1 or self.batch_rows!=self.matrix_row_lanes or self.dram_latency<2):
      raise ValueError('scale-cache timing requires physical unified traffic and one-wave batches on one engine')
    if type(self.vector_command_queue) is not int or self.vector_command_queue not in (0,2):
      raise ValueError('vector command queue must be 0 or the RTL two-entry FIFO')
    if self.vector_command_queue and (self.vector_result_chunks or self.vector_compatible_issue):
      raise ValueError('queued VPU model currently requires the single original issue path')
    if type(self.pinned_cache_bytes) is not int or self.pinned_cache_bytes<0 or self.pinned_cache_bytes%64:
      raise ValueError('pinned cache size must be a nonnegative multiple of 64 bytes')
    if type(self.cache_read_bytes_per_cycle) is not int or self.cache_read_bytes_per_cycle<1:
      raise ValueError('cache bandwidth must be a positive integer')
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
    if self.activation_streaming and (not self.quant_row_banks or self.batch_rows>self.matrix_row_lanes):
      raise ValueError('ACT streaming requires row banks and at most one parallel matrix wave per batch')
    if self.quant_scale_lookup and not self.quant_row_banks:
      raise ValueError('lookup quantizer scale requires explicit row banks')
    if (type(self.reciprocal_lookup_lanes) is not int or self.reciprocal_lookup_lanes not in (0,2,4,8,16)
        or (self.reciprocal_lookup_lanes and self.scalar_recip_lookup)):
      raise ValueError('choose either scalar lookup or a 2/4/8/16-lane reciprocal proposal')
    if self.quant_store_buffered and not self.quant_row_banks:
      raise ValueError('buffered QST requires explicit row banks')
    if not isinstance(self.vector_engines,int) or self.vector_engines<1:
      raise ValueError('vector_engines must be a positive integer')
    if not isinstance(self.vector_result_chunks,int) or self.vector_result_chunks<0:
      raise ValueError('vector_result_chunks must be a nonnegative integer')
    if self.vector_engines>1 and not self.vector_result_chunks:
      raise ValueError('multiple vector engines require a bounded result store')
    if self.vector_row_contexts not in (1,4):
      raise ValueError('vector_row_contexts must be 1 or 4')
    if (not isinstance(self.fp_mul_latency,int) or not 1<=self.fp_mul_latency<=2 or
        not isinstance(self.fp_add_latency,int) or not 1<=self.fp_add_latency<=4):
      raise ValueError('FP latency ranges: multiply 1..2, add 1..4')
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
    self.cache = net.place(prefix+'_cache_read',initial_tokens=1) if timing.pinned_cache_bytes else None


class SystolicArray:
  def __init__(self,net,prefix,timing):
    self.compute = net.place(prefix+"_compute",initial_tokens=1)
    self.commands = commands = net.place(prefix+"_commands",initial_tokens=0 if timing.matrix_retirement else 2)
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
      if timing.matrix_retirement:
        emit(commands,create().set('retire_slot',0))
        emit(commands,create().set('retire_slot',1))
    self.spec = timing


VECTOR_DRAIN_RANK=100


def scalar_recip_eligible(ins,I,timing):
  rows,cols=ins.w[3]&65535,ins.w[3]>>16
  slot=1+timing.fp_mul_latency+timing.fp_add_latency
  tap=1 if slot>=4 else 2
  return (timing.scalar_recip_lookup and (ins.w[5]>>16)&255==I.V_RECIP and 1<=cols<=4
          and rows*(cols-1)<=(6-tap)*slot+3-(3 if tap>1 else 0))


def reciprocal_lookup_width(ins,I,timing):
  if scalar_recip_eligible(ins,I,timing):
    return 1
  width=timing.reciprocal_lookup_lanes
  if width and (ins.w[5]>>16)&255==I.V_RECIP:
    rows,cols=ins.w[3]&65535,ins.w[3]>>16
    slot=1+timing.fp_mul_latency+timing.fp_add_latency
    tap=1 if slot>=4 else 2
    extra=rows*(ceil(cols/width)-ceil(cols/16))
    if rows>0 and cols>0 and extra<=(6-tap)*slot+3-(3 if tap>1 else 0):
      return width
  return 0


def vector_rank(ins,I,timing=None):
  """RTL n_slots ordering; reductions additionally require an empty pipeline."""
  function=(ins.w[5]>>16)&255
  if function in I.REDUCE:
    return VECTOR_DRAIN_RANK
  if timing is not None and reciprocal_lookup_width(ins,I,timing):
    return 1 if 1+timing.fp_mul_latency+timing.fp_add_latency>=4 else 2
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
    if self.timing.vector_issue_groups>1 and (not self.supports_multiple_matrix_engines or workload.config.LANES!=64):
      raise ValueError('wide elementwise proposal requires CompactOpenTPU with 64 base lanes')
    if self.timing.scale_cache_entries and not self.supports_multiple_matrix_engines:
      raise ValueError('scale-cache timing requires CompactOpenTPU')
    if self.timing.pinned_cache_bytes and not self.supports_multiple_matrix_engines:
      raise ValueError('pinned cache requires CompactOpenTPU and a validated prefetch profile')
    if self.timing.vector_command_queue and not self.supports_multiple_matrix_engines:
      raise ValueError('queued VPU requires CompactOpenTPU command acceptance')
    if self.timing.matrix_engines!=1 and not self.supports_multiple_matrix_engines:
      raise ValueError("multiple matrix engines require CompactOpenTPU")
    if self.timing.activation_streaming and not self.supports_multiple_matrix_engines:
      raise ValueError('ACT streaming requires CompactOpenTPU')
    if (self.timing.scalar_recip_lookup or self.timing.reciprocal_lookup_lanes) and not self.supports_multiple_matrix_engines:
      raise ValueError('lookup reciprocal requires CompactOpenTPU command selection')
    if self.timing.vector_compatible_issue and not self.supports_multiple_matrix_engines:
      raise ValueError('compatible vector selection requires CompactOpenTPU')
    if self.timing.vector_result_chunks and not self.supports_multiple_matrix_engines:
      raise ValueError('vector pools require CompactOpenTPU')
    self.matrix_batch_sizes = []
    cfg,t = workload.config,self.timing
    if (t.matrix_retirement or t.physical_scale_lines) and (not self.supports_multiple_matrix_engines or cfg.MCOLS not in (1,2,4,8)):
      raise ValueError('matrix details require CompactOpenTPU with 1/2/4/8-column cores')
    if (t.fp_mul_latency,t.fp_add_latency)!=(2,4) and cfg.LANES!=64:
      raise ValueError('experimental FP partition timing requires the 64-lane architecture')
    if t.activation_streaming and (cfg.MCOLS not in (1,2,4,8) or cfg.LANES!=cfg.D):
      raise ValueError('ACT streaming requires 1/2/4/8-column engines and one full block per quantizer write')
    if (t.scalar_recip_lookup or t.reciprocal_lookup_lanes) and cfg.LANES!=64:
      raise ValueError('scalar reciprocal lookup timing requires the 64-lane VPU')
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
      if ins.op==I.ST and t.dma_store_runs:
        latency=buffered_store_cycles(w[2],w[0],cfg.D,cfg.LANES,t.dram_bytes_per_cycle)
      port = memory.b
    elif ins.op==I.QACT:
      rows,kb = w[1]&255,w[1]>>16
      chunks = ceil(kb*cfg.D/cfg.LANES)
      overhead=60-12*(2-t.fp_mul_latency)-4*(4-t.fp_add_latency)
      # QACT's row buffer overlaps pass 0 of the following row with pass 1.
      latency = ((rows+1)*chunks+overhead if t.quant_buffer_chunks>=2*chunks else
                 rows*(2*chunks+overhead))
      if t.quant_row_banks and chunks<=t.quant_buffer_chunks:
        latency=buffered_quant_cycles(rows,chunks,t.quant_buffer_chunks,t.quant_row_banks,
            mul_latency=t.fp_mul_latency,add_latency=t.fp_add_latency,scale_lookup=t.quant_scale_lookup)
      elif t.quant_row_banks:
        latency=legacy_quant_cycles(rows,chunks,chunks,mul_latency=t.fp_mul_latency,
            add_latency=t.fp_add_latency,scale_lookup=t.quant_scale_lookup)
      port,service = None,0
    elif ins.op==I.QST:
      rows,kb = w[3]&65535,w[3]>>16
      elems = kb*cfg.D
      width = 4 if w[6]==1 else 1
      store = (ceil(elems*4/t.scalar_store_bytes)
               if t.scalar_store_bytes else ceil(elems/width))
      overhead=60-12*(2-t.fp_mul_latency)-4*(4-t.fp_add_latency)
      latency = rows*(ceil(elems/cfg.LANES)+store+overhead)
      chunks=ceil(elems/cfg.LANES)
      half=bool(ins.flags&1 and ins.flags&2)
      parallel_stores=t.quant_row_banks and t.scalar_store_bytes==4*cfg.LANES
      if parallel_stores:
        store=ceil((elems//2 if half else elems)/cfg.LANES)
        scale_words=1 if ins.flags&1 else 1+ceil(kb/4)
        latency=legacy_quant_cycles(rows,chunks,store,scale_words,t.fp_mul_latency,t.fp_add_latency,t.quant_scale_lookup)
      if (t.quant_store_buffered and t.scalar_store_bytes==4*cfg.LANES and
          not half and chunks<=t.quant_buffer_chunks):
        scale_words=1 if ins.flags&1 else 1+ceil(kb/4)
        latency=buffered_quant_cycles(rows,chunks,t.quant_buffer_chunks,t.quant_row_banks,scale_words,
            t.fp_mul_latency,t.fp_add_latency,t.quant_scale_lookup)
      # The shared memory link is occupied by byte-enabled stores, not by
      # scratchpad reads, max reduction or FP conversion in the quantizer.
      # Charging its complete latency here stalls unrelated matrix prefetch.
      port,service = memory.sw,(rows*store if t.unified_dram else latency)
    elif ins.op==I.VOP:
      rows,cols = w[3]&65535,w[3]>>16
      function = (w[5]>>16)&255
      # Independent elementwise lanes can issue several original 64-word
      # groups together. Ordered reductions keep the original 64-partial
      # datapath and recurrence; widening its tree would change FP results.
      issue_lanes=cfg.LANES*(t.vector_issue_groups if function not in I.REDUCE else 1)
      composite = function in (I.V_EXP2,I.V_RECIP,I.V_RSQRT,I.V_LOG2,I.V_EXP2SUB)
      if composite and cfg.LANES==8:
        # Current RTL shares SE's three stages: 25-cycle loops on eight lanes.
        passes = {I.V_EXP2:3,I.V_EXP2SUB:3,I.V_RECIP:2,I.V_RSQRT:4,I.V_LOG2:4}[function]
        latency = rows*ceil(cols/8)*passes+25*passes+5
      else:
        lanes = max(1,issue_lanes//4) if composite else issue_lanes
        slots = {I.V_EXP2:9,I.V_EXP2SUB:9,I.V_RECIP:6,I.V_RSQRT:10,I.V_LOG2:10}.get(function,0)
        slot_one=function in (I.V_ADD,I.V_SUB,I.V_RSUB,I.V_MUL,I.V_OUTER)
        ordinary=12-((2-t.fp_mul_latency)+(4-t.fp_add_latency) if slot_one else 0)
        latency = rows*ceil(cols/lanes)+((1+t.fp_mul_latency+t.fp_add_latency)*slots+8 if composite else ordinary)
      lookup_width=reciprocal_lookup_width(ins,I,t)
      if lookup_width:
        slot=vector_rank(ins,I,t)
        latency=rows*ceil(cols/lookup_width)+slot*(1+t.fp_mul_latency+t.fp_add_latency)+(3 if slot>1 else 0)+5
      if function==I.V_OUTER:
        latency += max(2,ceil(cols/issue_lanes))
      if function in I.REDUCE:
        if cfg.LANES>=32:
          if function in (I.V_RSUM,I.V_RSSQ,I.V_RDOT):
            partial_chunks=64//cfg.LANES
            chunks=rows*ceil(cols/64)*partial_chunks
            # The real wide datapath preserves 64 ordered partial sums. Its
            # four-stage adder limits recurrence; the six-level tree drains
            # once, after the last issued partial (read/product/add/tree/retire).
            interval=max(1,ceil(t.fp_add_latency/partial_chunks))
            tail=7*t.fp_add_latency+t.fp_mul_latency+6
            latency=(chunks-1)*interval+tail
            if t.vector_row_contexts==4 and cfg.LANES==64 and disjoint_sum_rows(ins,I):
              latency=interleaved_sum_span(rows,ceil(cols/64),adder_latency=t.fp_add_latency)+tail
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
    issue_lanes=lanes*(self.timing.vector_issue_groups if function not in I.REDUCE else 1)
    lookup_width=reciprocal_lookup_width(ins,I,self.timing)
    if lookup_width:
      return rows*ceil(cols/lookup_width)+2
    if lanes==8 and vector_rank(ins,I,self.timing)>1:
      # COMP8 has three shared stages and a 25-cycle recirculation loop.
      # This is its aggregate issue bandwidth; return-slot collisions are
      # approximated rather than advancing one simulator event per clock.
      passes={I.V_EXP2:3,I.V_EXP2SUB:3,I.V_RECIP:2,I.V_RSQRT:4,I.V_LOG2:4}[function]
      return rows*ceil(cols/lanes)*passes+2
    if vector_rank(ins,I,self.timing)>1:
      issue_lanes=max(1,issue_lanes//4)
    return rows*ceil(cols/issue_lanes)+2+(max(2,ceil(cols/(lanes*self.timing.vector_issue_groups)))
        if function==I.V_OUTER else 0)

  def _simple(self,prefix,idx,ins,trigger,result,window,unit,memory,deps,I,vector=None):
    net = self.net
    latency,port,service = self._simple_timing(ins,memory,I)
    rank=vector_rank(ins,I,self.timing) if vector is not None else 0
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
    pipeline = min(cfg.D//8,t.chain_length)+cfg.MCOLS+5+(5+2*t.fp_mul_latency)+(
        t.fp_mul_latency+t.fp_add_latency if flags&I.F_ACC else 0)
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
