"""Shared RTL BAR/GATHER unit with per-slice request and acknowledgement ports.

otpu_coll waits for every slice, streams all source segments through one
broadcast port and acknowledges them together. One cooldown cycle remains
after the slices observe completion (the earlier ACK cycle is already spent).
Memory-range dependencies stay in each slice's existing command interface.
"""


class CollectiveEndpoint:
  def __init__(self,net,prefix,interface,ordinals,kind):
    self.requests=requests=net.place(prefix+'_requests')
    self.replies=replies=net.place(prefix+'_replies')
    resource=net.place(prefix+'_resource',initial_tokens=1)
    cursor=net.place(prefix+'_cursor',initial_tokens=1)
    slots,pending,window=interface.slots,interface.pending[kind],interface.window
    select_ready,publish=interface.select_ready,interface.publish

    @net.transition(prefix+'_request')
    def request():
      guard(count(resource)>0)
      guard(count(pending)>0)
      chosen=select_ready(const_i64(kind))
      guard(chosen>=0)
      observed=peek(array(slots)[chosen])
      phase=peek(cursor).get('id')
      guard(array(ordinals)[observed.get('id')]==phase)
      take(resource)
      token=take(array(slots)[chosen])
      take(pending)
      emit(array(slots)[chosen],create())
      emit(requests,token)

    @net.transition(prefix+'_acknowledge')
    def acknowledge():
      guard(count(replies)>0)
      token=take(replies)
      phase=take(cursor)
      emit(cursor,phase.set('id',phase.get('id')+1))
      emit(resource,create())
      publish(token.get('id'),0.0)
      emit(window,token)


class CollectiveBus:
  def __init__(self,net,interfaces,commands,parameters,table,I,W):
    signatures=[];ordinals=[]
    for sid,stream in enumerate(commands):
      tags=[];mapping=[-1]*len(stream)
      for idx,c in enumerate(stream):
        ins,w=c.instruction,c.instruction.w
        if ins.op in (I.BAR,I.GATHER):
          mapping[idx]=len(tags)
          # RTL takes slice 0's destination, shape, destination stride and
          # segment size; source addresses/strides are independently local.
          tags.append((ins.op,*(w[j] for j in (1,2,4,5))) if ins.op==I.GATHER else (ins.op,))
      signatures.append(tags);ordinals.append(mapping)
    if any(tags!=signatures[0] for tags in signatures[1:]):
      raise ValueError('slice collective streams disagree on BAR/GATHER sequence or destination geometry')
    if not signatures[0]:
      self.endpoints=[]
      return
    ordinals=[table(f's{sid}_collective_ordinals',mapping) for sid,mapping in enumerate(ordinals)]
    self.endpoints=[CollectiveEndpoint(net,f's{sid}_collective',interface,ordinals[sid],W.U_COLL)
                    for sid,interface in enumerate(interfaces)]
    requests=[endpoint.requests for endpoint in self.endpoints]
    replies=[endpoint.replies for endpoint in self.endpoints]
    latencies=parameters[0].latencies
    resource=net.place('collective_broadcast_port',initial_tokens=1)

    @net.transition('collective_broadcast')
    def broadcast():
      guard(count(resource)>0)
      for place in requests:
        guard(count(place)>0)
      idx=peek(requests[0]).get('id')
      latency=sitofp(array(latencies)[idx])
      credit=take(resource)
      emit(resource,credit,delay=latency+1.0)
      for sid in range(len(requests)):
        token=take(array(requests)[sid])
        emit(array(replies)[sid],token,delay=latency)
