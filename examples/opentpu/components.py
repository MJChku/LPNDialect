"""Execution components connected by command, completion and resource places.

Each matrix engine owns its request/ready queues and prefetch/compute credits.
Engines share only their command interface, dependency tracker and DRAM ports.
"""

from dataclasses import dataclass

from .model import SystolicArray, VectorPipeline, VECTOR_DRAIN_RANK


@dataclass
class CommandInterface:
  slots: list
  pending: list
  window: object
  select_ready: object
  is_done: object
  publish: object
  compute_offsets: list
  compute_indices: list
  vector_selector_factory: object = None
  activation: object = None
  activation_producers: object = None
  activation_groups: object = None


@dataclass
class CommandTiming:
  latencies: list
  services: list
  port_ids: list
  batches: list
  chunks: list
  scaled: list
  kb: list
  rows: list
  row_services: list
  tails: list
  scale_bytes: list
  write_full: list
  write_tail: list
  cache_offsets: list
  cached_prefix: list


class ExecutionUnit:
  """One scalar/control, DMA, quantization, vector or collective unit."""
  def __init__(self,net,prefix,kind,interface,parameters,memory,ranks=None):
    self.resource = resource = net.place(prefix+"_resource",initial_tokens=1)
    slots,pending,window = interface.slots,interface.pending[kind],interface.window
    select_ready,publish = interface.select_ready,interface.publish
    latencies,port_ids,services = parameters.latencies,parameters.port_ids,parameters.services
    vector=VectorPipeline(net,prefix+'_pipeline') if ranks is not None else None
    self.pipeline=vector
    if vector is not None and interface.vector_selector_factory is not None:
      select_ready=interface.vector_selector_factory(prefix,vector.state,None,None)
    intervals=services if vector is not None else latencies
    ports = [memory.b,memory.sw]
    activation=interface.activation if kind==2 else None
    start_activation=activation.start if activation is not None else None
    activation_tail=activation.tail if activation is not None else [0]

    @net.transition(prefix+"_execute")
    def execute():
      guard(count(resource)>0)
      guard(count(pending)>0)
      chosen = select_ready(const_i64(kind))
      guard(chosen>=0)
      slot_places = array(slots)
      observed = peek(slot_places[chosen])
      idx = observed.get("id")
      bus_id = array(port_ids)[idx]
      buses = array(ports)
      guard(select(bus_id>=0,count(buses[select(bus_id>=0,bus_id,0)])>0,True,typ="i1"))
      if vector is not None:
        state=peek(vector.state)
        rank=array(ranks)[idx]
        guard(select(state.get('live')==0,True,
            select(rank==VECTOR_DRAIN_RANK,False,state.get('rank')<=rank,typ='i1'),typ='i1'))
      token = take(slot_places[chosen])
      take(pending)
      emit(slot_places[chosen],create())
      latency = sitofp(array(latencies)[idx])
      credit = take(resource)
      resource_interval=array(intervals)[idx]
      if activation is not None:
        start_activation(idx)
        resource_interval=resource_interval-array(activation_tail)[idx]
      emit(resource,credit,delay=sitofp(resource_interval))
      if vector is not None:
        state=take(vector.state)
        emit(vector.state,state.set('live',state.get('live')+1).set('rank',array(ranks)[idx]))
        emit(vector.completed,create(),delay=latency)
      if bus_id>=0:
        bus = take(buses[bus_id])
        emit(buses[bus_id],bus,delay=sitofp(array(services)[idx]))
      publish(idx,latency)
      emit(window,token,delay=latency)


class MatrixEngine:
  """An independently pipelined engine accepting the existing MM commands.

  One prefetch credit admits the next command after the previous command's
  last memory transfer. The ready FIFO preserves this engine's row/command
  order; compute credit returns at the initiation interval, while command
  and window credits return after the pipeline/epilogue tail. There is no
  ordering link to another matrix engine.
  """
  def __init__(self,net,prefix,cfg,timing,interface,parameters,memory,kind):
    t = timing
    self.resources = resources = SystolicArray(net,prefix,t)
    self.prefetch_credit = prefetch_credit = net.place(prefix+"_prefetch_credit",initial_tokens=1)
    self.request = request = net.place(prefix+"_request")
    self.ready = ready = net.place(prefix+"_ready")
    slots,pending,window = interface.slots,interface.pending[kind],interface.window
    select_ready,is_done,publish = interface.select_ready,interface.is_done,interface.publish
    offsets,indices = interface.compute_offsets,interface.compute_indices
    rows,batches,chunks = parameters.rows,parameters.batches,parameters.chunks
    scaled,kbs = parameters.scaled,parameters.kb
    scale_bytes = parameters.scale_bytes
    write_full,write_tail = parameters.write_full,parameters.write_tail
    row_services,tails = parameters.row_services,parameters.tails
    cache_offsets,cached_prefix=parameters.cache_offsets,parameters.cached_prefix

    @net.transition(prefix+"_start")
    def start():
      guard(count(resources.commands)>0)
      guard(count(prefetch_credit)>0)
      guard(count(pending)>0)
      chosen = select_ready(const_i64(kind))
      guard(chosen>=0)
      place = array(slots)[chosen]
      token = take(place)
      take(pending)
      emit(place,create())
      take(resources.commands)
      take(prefetch_credit)
      emit(request,token.set("row",0))

    @net.transition(prefix+"_prefetch")
    def prefetch():
      guard(count(request)>0)
      guard(count(resources.buffers)>0)
      if not t.pinned_cache_bytes:
        guard(count(memory.b)>0)
      idx = peek(request).get("id")
      row = peek(request).get("row")
      n,batch,chunk = array(rows)[idx],array(batches)[idx],array(chunks)[idx]
      width = select(row+batch<n,batch,n-row)
      needed = muli(width,chunk)
      guard(peek(resources.buffers).get('free')>=needed)
      cached=const_i64(0)
      if t.pinned_cache_bytes:
        begin=array(cache_offsets)[idx]+row
        cached=array(cached_prefix)[begin+width]-array(cached_prefix)[begin]
        miss=muli(needed,cfg.D)-cached
        if t.unified_dram:miss=miss+muli(width,array(scale_bytes)[idx])
        guard(select(miss>0,count(memory.b)>0,True,typ='i1'))
        guard(select(cached>0,count(memory.cache)>0,True,typ='i1'))
      needs_scale = array(scaled)[idx]
      guard(count(memory.a)>=needs_scale)
      token = take(request)
      capacity = take(resources.buffers)
      emit(resources.buffers,capacity.set('free',capacity.get('free')-needed))
      bytes_per_row = muli(chunk,cfg.D)
      if t.unified_dram:
        bytes_per_row = addi(bytes_per_row,array(scale_bytes)[idx])
      transfer_cycles = divi(muli(width,bytes_per_row)+t.dram_bytes_per_cycle-1,t.dram_bytes_per_cycle)
      cache_cycles=const_i64(0)
      if t.pinned_cache_bytes:
        transfer_cycles=divi(miss+t.dram_bytes_per_cycle-1,t.dram_bytes_per_cycle)
        cache_cycles=divi(cached+t.cache_read_bytes_per_cycle-1,t.cache_read_bytes_per_cycle)
        if miss>0:
          bus=take(memory.b)
          emit(memory.b,bus,delay=sitofp(transfer_cycles))
        if cached>0:
          cache_port=take(memory.cache)
          emit(memory.cache,cache_port,delay=sitofp(cache_cycles))
      else:
        bus=take(memory.b)
        emit(memory.b,bus,delay=sitofp(transfer_cycles))
      scale_cycles = divi(muli(muli(width,array(kbs)[idx]),needs_scale)+t.scale_lanes-1,t.scale_lanes)
      service = sitofp(transfer_cycles)
      if needs_scale>0:
        scale_port = take(memory.a)
        emit(memory.a,scale_port,delay=sitofp(scale_cycles))
      stream_cycles=select(transfer_cycles>cache_cycles,transfer_cycles,cache_cycles)
      arrival = sitofp(select(stream_cycles>scale_cycles,stream_cycles,scale_cycles))
      emit(ready,token.set("width",width),delay=arrival+float(t.dram_latency+4))
      if row+batch<n:
        emit(request,token.set("row",row+batch),delay=arrival if t.pinned_cache_bytes else service)
      else:
        # A later unscaled command must not bypass this command's scale
        # decoder and reach the FIFO before its last batch. Common arrival
        # latency preserves order once both transfer paths have finished.
        emit(prefetch_credit,create(),delay=arrival)

    if interface.activation is not None:
      self._stream_compute(net,prefix,cfg,t,interface,parameters,resources,ready,publish,window,is_done)
      return

    @net.transition(prefix+"_issue_rows")
    def compute():
      guard(count(ready)>0)
      guard(count(resources.compute)>0)
      guard(count(resources.buffers)>0)
      idx = peek(ready).get("id")
      ci = array(indices)
      enabled = const_i64(0)==0
      for d in range(array(offsets)[idx],array(offsets)[idx+1]):
        enabled = select(enabled,is_done(ci[d]),False,typ="i1")
      guard(enabled)
      token = take(ready)
      port = take(resources.compute)
      compute_cycles = muli(divi(token.get("width")+t.matrix_row_lanes-1,t.matrix_row_lanes),array(row_services)[idx])
      write_cycles = select(token.get("width")==array(batches)[idx],array(write_full)[idx],array(write_tail)[idx])
      service = sitofp(select(compute_cycles>write_cycles,compute_cycles,write_cycles))
      emit(resources.compute,port,delay=service)
      capacity = take(resources.buffers)
      emit(resources.buffers,capacity.set('free',capacity.get('free')+
           muli(token.get('width'),array(chunks)[idx])))
      if token.get("row")+array(batches)[idx]>=array(rows)[idx]:
        latency = service+sitofp(array(tails)[idx])
        publish(idx,latency)
        emit(resources.commands,create(),delay=latency)
        emit(window,token,delay=latency)


  def _stream_compute(self,net,prefix,cfg,t,interface,parameters,resources,ready,publish,window,is_done):
    offsets,indices=interface.compute_offsets,interface.compute_indices
    groups,producers=interface.activation_groups,interface.activation_producers
    can_read=interface.activation.can_read
    batches,rows,chunks=parameters.batches,parameters.rows,parameters.chunks
    kbs,tails=parameters.kb,parameters.tails
    write_full,write_tail=parameters.write_full,parameters.write_tail
    self.active_rows=active=net.place(prefix+'_active_rows')
    self.returned_chunks=returned=net.place(prefix+'_returned_chunks')

    @net.transition(prefix+'_accept_rows')
    def accept_rows():
      guard(count(ready)>0)
      guard(count(resources.compute)>0)
      idx=peek(ready).get('id')
      enabled=const_i64(0)==0
      ci=array(indices)
      for d in range(array(offsets)[idx],array(offsets)[idx+1]):
        enabled=select(enabled,is_done(ci[d]),False,typ='i1')
      guard(enabled)
      token=take(ready)
      take(resources.compute)
      emit(active,token.set('group',0))

    @net.transition(prefix+'_advance_activation_row')
    def advance_activation_row():
      guard(count(active)>0)
      idx=peek(active).get('id')
      group=peek(active).get('group')
      guard(can_read(array(producers)[idx],group))
      token=take(active)
      waves=divi(token.get('width')+t.matrix_row_lanes-1,t.matrix_row_lanes)
      clocks=muli(waves,array(kbs)[idx])
      ng=array(groups)[idx]
      final=group+1>=ng
      writes=select(token.get('width')==array(batches)[idx],array(write_full)[idx],array(write_tail)[idx])
      compute=muli(clocks,ng)
      padding=select(writes>compute,writes-compute,0)
      service=sitofp(clocks+select(final,padding,0))
      if final:
        emit(resources.compute,create(),delay=service)
        emit(returned,create().set('chunks',muli(token.get('width'),array(chunks)[idx])),delay=service)
        if token.get('row')+array(batches)[idx]>=array(rows)[idx]:
          latency=service+sitofp(array(tails)[idx])
          publish(idx,latency)
          emit(resources.commands,create(),delay=latency)
          emit(window,token,delay=latency)
      else:
        emit(active,token.set('group',group+1),delay=service)

    @net.transition(prefix+'_return_chunks')
    def return_chunks():
      guard(count(returned)>0)
      token=take(returned)
      capacity=take(resources.buffers)
      emit(resources.buffers,capacity.set('free',capacity.get('free')+token.get('chunks')))
