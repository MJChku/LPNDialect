"""Select the oldest dependency-ready VOP that fits the actual pipeline state."""
from .model import VECTOR_DRAIN_RANK


def vector_selector_factory(net,slots,window,kinds,begins,ends,indices,metadata,is_done,ranks):
  def build(prefix,state,capacity,chunks):
    @net.func(name=prefix+'_select_compatible',args=('i64',),results=('i64',))
    def select_compatible(unit):
      current=peek(state)
      selected,oldest=subi(0,1),addi(array(metadata)[0],1)
      for s in range(window):
        token=peek(array(slots)[s])
        idx=token.get('id')
        enabled=select(token.get('active')==1,array(kinds)[idx]==unit,False,typ='i1')
        for d in range(select(enabled,array(begins)[idx],0),select(enabled,array(ends)[idx],0)):
          enabled=select(enabled,is_done(array(indices)[d]),False,typ='i1')
        rank=array(ranks)[idx]
        compatible=select(current.get('live')==0,True,
            select(rank==VECTOR_DRAIN_RANK,False,current.get('rank')<=rank,typ='i1'),typ='i1')
        enabled=select(enabled,compatible,False,typ='i1')
        if capacity is not None:
          enabled=select(enabled,peek(capacity).get('free')>=array(chunks)[idx],False,typ='i1')
        eligible=select(enabled,idx<oldest,False,typ='i1')
        selected=select(eligible,s,selected,typ='i64')
        oldest=select(eligible,idx,oldest)
      func_return(selected)
    return select_compatible
  return build
