"""Proposed vector engines with private read ports and a finite result store.

There is no RTL implementation of this pool yet. Each engine reserves storage
for a complete command before issue, then shares one 64-bank result write port.
This deliberately charges an extra result drain after the existing VPU timing.
"""
from math import ceil
from .model import VectorPipeline,VECTOR_DRAIN_RANK


def result_shape(ins,I,lanes):
  rows,cols=ins.w[3]&65535,ins.w[3]>>16
  function=(ins.w[5]>>16)&255
  words=rows*(2 if function==I.V_ARGMAX else 1) if function in I.REDUCE else rows*cols
  # The result store is contiguous; the drain's address generator restores row
  # strides. Reductions drain one row per cycle, retaining scalar bank conflicts.
  clocks=rows if function in I.REDUCE else rows*ceil(cols/lanes)
  return max(1,ceil(words/lanes)),max(1,clocks)


class VectorPoolEngine:
  def __init__(self,net,prefix,interface,parameters,ranks,chunks,drains,capacity,write_port,kind,accepted=None):
    resource=net.place(prefix+'_issue',initial_tokens=1)
    buffered=net.place(prefix+'_result_capacity',initial_tokens=1)
    initialize=net.place(prefix+'_initialize',initial_tokens=1)
    output=net.place(prefix+'_results')
    returned=net.place(prefix+'_returned')
    pipeline=VectorPipeline(net,prefix+'_pipeline')
    slots,pending,window=interface.slots,interface.pending[kind],interface.window
    select_ready,publish=interface.select_ready,interface.publish
    latencies,intervals=parameters.latencies,parameters.services
    if interface.vector_selector_factory is not None:
      select_ready=interface.vector_selector_factory(prefix,pipeline.state,buffered,chunks,accepted)

    @net.transition(prefix+'_initialize_capacity')
    def initialize_capacity():
      take(initialize)
      token=take(buffered)
      emit(buffered,token.set('free',capacity))

    @net.transition(prefix+'_execute')
    def execute():
      guard(count(resource)>0)
      guard(count(pending)>0)
      chosen=select_ready(const_i64(kind))
      guard(chosen>=0)
      place=array(slots)[chosen]
      idx=peek(place).get('id')
      guard(peek(buffered).get('free')>=array(chunks)[idx])
      state=peek(pipeline.state)
      rank=array(ranks)[idx]
      guard(select(state.get('live')==0,True,
          select(rank==VECTOR_DRAIN_RANK,False,state.get('rank')<=rank,typ='i1'),typ='i1'))
      token=take(place)
      take(pending)
      emit(place,create())
      credit=take(resource)
      emit(resource,credit,delay=sitofp(array(intervals)[idx]))
      storage=take(buffered)
      emit(buffered,storage.set('free',storage.get('free')-array(chunks)[idx]))
      state=take(pipeline.state)
      emit(pipeline.state,state.set('live',state.get('live')+1).set('rank',rank))
      latency=sitofp(array(latencies)[idx])
      emit(pipeline.completed,create(),delay=latency)
      emit(output,token,delay=latency)

    @net.transition(prefix+'_drain')
    def drain():
      guard(count(output)>0)
      guard(count(write_port)>0)
      token=take(output)
      idx=token.get('id')
      credit=take(write_port)
      latency=sitofp(array(drains)[idx])
      emit(write_port,credit,delay=latency)
      publish(idx,latency)
      emit(window,token,delay=latency)
      emit(returned,create().set('chunks',array(chunks)[idx]),delay=latency)

    @net.transition(prefix+'_return_capacity')
    def return_capacity():
      guard(count(returned)>0)
      token=take(returned)
      storage=take(buffered)
      emit(buffered,storage.set('free',storage.get('free')+token.get('chunks')))
