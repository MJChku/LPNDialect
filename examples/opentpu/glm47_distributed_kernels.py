"""Existing-ISA GLM experts owned by slices, joined through hardware GATHER.

Each whole expert belongs to one slice. Its SwiGLU is local; the shared expert
executes once. A fixed common TMEM destination keeps GATHER geometry identical
despite different local expert programs. No routing or latency is invented.
"""
from opentpu import language as ol
from opentpu.kernels.lib import rmsnorm,silu
from .glm47_kernels import expert_batches


def local_swiglu(xs,descriptor):
    """One whole expert, without the FFN-sharding gathers of swiglu_down."""
    parts=descriptor.wd.parts;chunk=descriptor.wd.pw
    def gate_up(index):
        start=index*chunk
        return (ol.dot(xs,descriptor.wg[start:start+chunk,:]),
                ol.dot(xs,descriptor.wu[start:start+chunk,:]))
    following=gate_up(0);result=None
    for index,down in enumerate(parts):
        gate,up=following
        if index+1<len(parts):following=gate_up(index+1)
        activation=silu(gate)*up
        if result is None:result=ol.dot(activation,down)
        else:ol.dot(activation,down,acc=result)
    return result


@ol.jit
def distributed_experts(m,ids):
    from opentpu import isa as I
    from opentpu.compiler import current
    sid,S=ol.program_id(),ol.num_programs();R,H=m.x.shape
    # Allocate these before the owner-dependent branch, so every program
    # presents the same broadcast destination, shape, stride and segment size.
    broadcast=ol.empty((S*H,));local=ol.empty((H,))
    owned=[(expert,batch) for expert,batch in expert_batches(ids,m.global_experts,R)
           if expert//m.experts==sid]
    if owned or sid==0:xn=rmsnorm(ol.load(m.x),ol.load(m.norm),m.eps)
    ol.store(m.partial,ol.zeros((R,H)))
    for expert,batch in owned:
        gathered=ol.empty((len(batch),H))
        for i,(token,rank) in enumerate(batch):gathered[i,:].set(xn[token,:])
        desc=m.fmt.descs(m.expert_base+(expert%m.experts)*m.fmt.nbytes)
        y=local_swiglu(ol.quantize(gathered),desc)
        for i,(token,rank) in enumerate(batch):
            weight=ol.load(m.weights[token,rank:rank+1])
            ol.store(m.partial[token,:],ol.load(m.partial[token,:])+y[i,:]*weight)
        del y,gathered,desc,weight
    if sid==0:
        desc=m.fmt.descs(m.shared_base);y=local_swiglu(ol.quantize(xn),desc)
        ol.store(m.partial,ol.load(m.partial)+y)
        del desc,y
    if owned or sid==0:del xn
    for row in range(R):
        ol.load(m.partial[row,:],out=local)
        if S>1:
            current().emit(I.gather(local.base,broadcast.base,1,H,H,S*H,H,
                                   comment='GLM: combine actual expert-owner partials'))
            current().bump_version(broadcast.buf)
            total=broadcast[:H]+broadcast[H:2*H]
            for peer in range(2,S):total=total+broadcast[peer*H:(peer+1)*H]
        else:total=local
        ol.store(m.out[row,:],total+ol.load(m.x[row,:]))
