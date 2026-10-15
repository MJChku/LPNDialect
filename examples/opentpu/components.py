"""Execution components connected by command, completion and resource places.

Each matrix engine owns its request/ready queues and prefetch/compute credits.
Engines share only their command interface, dependency tracker and DRAM ports.
"""

from dataclasses import dataclass

from .model import SystolicArray


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


class ExecutionUnit:
  """One scalar/control, DMA, quantization, vector or collective unit."""
  def __init__(self,net,prefix,kind,interface,parameters,memory):
    self.resource = resource = net.place(prefix+"_resource",initial_tokens=1)
    slots,pending,window = interface.slots,interface.pending[kind],interface.window
    select_ready,publish = interface.select_ready,interface.publish
    latencies,port_ids,services = parameters.latencies,parameters.port_ids,parameters.services
    ports = [memory.b,memory.sw]

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
      token = take(slot_places[chosen])
      take(pending)
      emit(slot_places[chosen],create())
      latency = sitofp(array(latencies)[idx])
      credit = take(resource)
      emit(resource,credit,delay=latency)
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
  def __init__(self,net,prefix,cfg,timing,interface,parameters,memory,kind,buffer_metadata=None):
    t = timing
    self.resources = resources = SystolicArray(net,prefix,t,
        max(b*c for b,c in zip(parameters.batches,parameters.chunks)),
        initial_buffers=0 if buffer_metadata is not None else None)
    self.prefetch_credit = prefetch_credit = net.place(prefix+"_prefetch_credit",initial_tokens=1)
    self.request = request = net.place(prefix+"_request")
    self.ready = ready = net.place(prefix+"_ready")
    slots,pending,window = interface.slots,interface.pending[kind],interface.window
    select_ready,is_done,publish = interface.select_ready,interface.is_done,interface.publish
    offsets,indices = interface.compute_offsets,interface.compute_indices
    rows,batches,chunks = parameters.rows,parameters.batches,parameters.chunks
    scaled,kbs = parameters.scaled,parameters.kb
    row_services,tails = parameters.row_services,parameters.tails

    if buffer_metadata is not None:
      initialize = net.place(prefix+"_initialize_buffers",initial_tokens=1)

      @net.transition(prefix+"_initialize_buffer_credits")
      def initialize_buffer_credits():
        guard(count(initialize)>0)
        take(initialize)
        for b in range(array(buffer_metadata)[1]):
          emit(resources.buffers,create())

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
      guard(count(memory.b)>0)
      idx = peek(request).get("id")
      needs_scale = array(scaled)[idx]
      guard(count(memory.a)>=needs_scale)
      token = take(request)
      row = token.get("row")
      n,batch,chunk = array(rows)[idx],array(batches)[idx],array(chunks)[idx]
      width = select(row+batch<n,batch,n-row)
      take(resources.buffers)
      bus = take(memory.b)
      service = sitofp(divi(muli(muli(width,chunk),cfg.D)+t.dram_bytes_per_cycle-1,t.dram_bytes_per_cycle))
      emit(memory.b,bus,delay=service)
      if needs_scale>0:
        scale_port = take(memory.a)
        emit(memory.a,scale_port,delay=sitofp(muli(width,array(kbs)[idx])))
      emit(ready,token.set("width",width),delay=service+float(t.dram_latency+4))
      if row+batch<n:
        emit(request,token.set("row",row+batch),delay=service)
      else:
        emit(prefetch_credit,create(),delay=service)

    @net.transition(prefix+"_issue_rows")
    def compute():
      guard(count(ready)>0)
      guard(count(resources.compute)>0)
      idx = peek(ready).get("id")
      ci = array(indices)
      enabled = const_i64(0)==0
      for d in range(array(offsets)[idx],array(offsets)[idx+1]):
        enabled = select(enabled,is_done(ci[d]),False,typ="i1")
      guard(enabled)
      token = take(ready)
      port = take(resources.compute)
      service = sitofp(muli(token.get("width"),array(row_services)[idx]))
      emit(resources.compute,port,delay=service)
      emit(resources.buffers,create())
      if token.get("row")+array(batches)[idx]>=array(rows)[idx]:
        latency = service+sitofp(array(tails)[idx])
        publish(idx,latency)
        emit(resources.commands,create(),delay=latency)
        emit(window,token,delay=latency)
