"""GLM MLA compiler kernels emitting the existing FeSens hardware ISA.

These are functional/compiler routines, not an LPN model. They use the real
openTPU language and quantized KV format. The upstream reference's expanded
K/V formulation is preserved; no compressed/absorbed attention is substituted.
The backend uses FP32 intermediates and NVFP4-weight/int8-activation MMs, so
it is not bit-equivalent to the upstream BF16 execution.
"""
from opentpu import language as ol
from opentpu.kernels.attention import RowMasks,TileSeq,_attend_heads
from opentpu.kernels.lib import rmsnorm,rope_rows


def rotary_interleave(x,cos,sin):
    """HF GLM interleaved input pairs -> concatenated rotated halves.

    Explicit strided-column copies lower to real VOPs. Copies precede writes,
    so the source's even/odd elements cannot be overwritten prematurely.
    """
    half=x.cols//2
    even,odd=ol.empty((x.rows,half)),ol.empty((x.rows,half))
    for j in range(half):
        even[:,j:j+1].set(x[:,2*j:2*j+1])
        odd[:,j:j+1].set(x[:,2*j+1:2*j+2])
    out=ol.empty(x.shape)
    out[:,:half].set(even*cos-odd*sin)
    out[:,half:].set(odd*cos+even*sin)
    return out


def _project_cache(m,p0,interleave):
    """Low-rank Q/KV, per-rank norms, RoPE, expanded KV quantization/store."""
    s=m.spec;R=m.x.shape[0];heads=s.heads;d=s.nope+s.rope
    x=ol.load(m.x)
    xn=rmsnorm(x,ol.load(m.input_norm),s.eps)
    xs=ol.quantize(xn)
    qa=ol.dot(xs,m.qa)
    compressed=ol.dot(xs,m.kva)
    del xs,xn,x
    qn=rmsnorm(qa,ol.load(m.q_norm),1e-6)
    kn=rmsnorm(compressed[:,:s.kv_rank],ol.load(m.kv_norm),1e-6)
    qs,ks=ol.quantize(qn),ol.quantize(kn)
    cos,sin=ol.load(m.cos),ol.load(m.sin)
    kr=compressed[:,s.kv_rank:s.kv_rank+s.rope]
    kr=rotary_interleave(kr,cos,sin) if interleave else rope_rows(kr,cos,sin)
    for h in range(heads):
        q=ol.dot(qs,m.qb[h*d:(h+1)*d,:])
        qr=q[:,s.nope:d]
        qr=rotary_interleave(qr,cos,sin) if interleave else rope_rows(qr,cos,sin)
        q[:,s.nope:d].set(qr)
        expanded=ol.dot(ks,m.kvb[h*(s.nope+s.value):(h+1)*(s.nope+s.value),:])
        k=ol.empty((R,d))
        k[:,:s.nope].set(expanded[:,:s.nope]);k[:,s.nope:d].set(kr)
        ol.kv_append(m.kv,h,p0,k,expanded[:,s.nope:s.nope+s.value])
        ol.store(m.query[:,h*d:(h+1)*d],q)


def causal_sequence(p0,rows,block):
    """Compiler-side causal mask for an externally scheduled contiguous tile."""
    full=p0//block;end=p0+rows;items=[]
    for start in range(full*block,end,block):
        n=min(block,end-start)
        keep=[(r,r+1,min(n,max(0,p0+r+1-start))) for r in range(rows)]
        items.append((start,n,RowMasks(keep)))
    return TileSeq(full,items)


def _attention(m,p0,block):
    s=m.spec;d=s.nope+s.rope;rows=m.x.shape[0]
    sequence=causal_sequence(p0,rows,block)
    queries=[lambda h=h:ol.load(m.query[:,h*d:(h+1)*d]) for h in range(s.heads)]
    def store_head(h,acc,normalizer):
        ol.store(m.attention[:,h*s.value:(h+1)*s.value],acc/normalizer[:,None])
    _attend_heads(queries,m.kv,list(range(s.heads)),sequence,block,
                  scale=d**-.5,depth=3,ahead=1,emit=store_head)


def _output(m):
    x=ol.load(m.attention)
    y=ol.dot(x,m.wo)
    ol.store(m.out,y+ol.load(m.x))


@ol.jit
def mla_project_cache(m,p0,interleave=True):
    _project_cache(m,p0,interleave)


@ol.jit
def mla_attention(m,p0,block=128):
    _attention(m,p0,block)


@ol.jit
def mla_output(m):
    _output(m)


@ol.jit
def mla_layer(m,p0,block=128,interleave=True):
    """One real hardware program; external frontend owns tile/layer scheduling."""
    _project_cache(m,p0,interleave)
    _attention(m,p0,block)
    _output(m)


@ol.jit
def rotary_test(x,cos,sin,out):
    ol.store(out,rotary_interleave(ol.load(x),ol.load(cos),ol.load(sin)))


@ol.jit
def glm_router(m,topk=4,scale=1.8):
    """Actual FP32 router and GLM's single-group sigmoid+bias top-k rule.

    Router weights stay FP32. ARGMAX/RLD and indexed VOPs perform the
    data-dependent selection on hardware; expert IDs are not annotations.
    The pinned model has one group, so group selection cannot discard experts.
    """
    from opentpu import isa as I
    from opentpu.compiler import current
    from opentpu.kernels.lib import sigmoid
    b=current();R,H=m.x.shape;E=m.router.shape[0]
    xn=rmsnorm(ol.load(m.x),ol.load(m.norm),m.eps)
    # An FP32 RDOT, not an NVFP4 MM, for each router row. Store scalar
    # results at affine DRAM addresses so one hardware loop serves all E.
    for e in ol.range(E):
        weight=ol.load(m.router[e:e+1,:])
        column=ol.empty((R,1))
        for r in range(R):
            column[r,0:1].set(ol.sum(xn[r,:]*weight[0,:]))
        ol.store(m.logits[:,e:e+1],column)
    logits=ol.load(m.logits);bias=ol.load(m.bias)
    pr=ol.empty((2,));register=b.scratch()
    for r in range(R):
        scores=sigmoid(logits[r,:]);choice=scores+bias
        weights,ids=ol.empty((topk,)),ol.empty((topk,))
        for rank in range(topk):
            b.emit(I.argmax(pr.base,choice.base,1,E,comment='GLM router: best score plus correction bias'))
            b.emit(I.rld(register,pr.base+1,bound=(0,E-1),comment='GLM router: selected expert ID'))
            b.emit(I.vop(I.V_FILL,choice.base,0,0,1,1,0,0,0,I.B_SCALAR,-3e38,ra=register,
                         comment='GLM router: exclude selected expert'))
            b.emit(I.vop(I.V_COPY,weights.base+rank,scores.base,0,1,1,0,0,0,rb=register,
                         comment='GLM router: unbiased sigmoid weight'))
            ids[rank:rank+1].set(pr[1:2])
        weights.set(weights*ol.recip(ol.sum(weights)+1e-20)*scale)
        ol.store(m.ids[r,:],ids);ol.store(m.weights[r,:],weights)
    b.unscratch(register)


def expert_batches(ids,experts,max_rows):
    """Compiler/runtime dispatch from resolved router IDs, outside the LPN.

    Ascending expert order matches the pinned reference's index_add loop.
    Each selected token appears once per expert, with its original top-k rank.
    """
    groups={}
    for token,row in enumerate(ids):
        if len(set(row))!=len(row):raise ValueError('duplicate expert in one token route')
        for rank,e in enumerate(row):
            if type(e) is not int or not 0<=e<experts:raise ValueError('invalid resolved expert ID')
            groups.setdefault(e,[]).append((token,rank))
    if max_rows<1:raise ValueError('positive physical batch size required')
    return [(e,items[start:start+max_rows]) for e,items in sorted(groups.items())
            for start in range(0,len(items),max_rows)]


@ol.jit
def glm_resident_experts(m,ids):
    """Routed and unconditional shared SwiGLU, encoded as real hardware commands.

    IDs must be actual functional/router results before compilation. Dispatch
    is a compiler/runtime responsibility; the LPN sees the resulting hex only.
    This first backend holds one layer's experts resident in a 32-bit region.
    """
    from opentpu.kernels.mlp import swiglu_down
    x=ol.load(m.x);xn=rmsnorm(x,ol.load(m.norm),m.eps)
    ol.store(m.out,ol.zeros(x.shape))
    for expert,batch in expert_batches(ids,m.experts,x.rows):
        gathered=ol.empty((len(batch),x.cols))
        for i,(token,rank) in enumerate(batch):gathered[i,:].set(xn[token,:])
        descriptor=m.fmt.descs(m.expert_base+expert*m.fmt.nbytes)
        xs=ol.quantize(gathered)
        y=swiglu_down(xs,descriptor.wg,descriptor.wu,descriptor.wd,loop=True)
        for i,(token,rank) in enumerate(batch):
            weight=ol.load(m.weights[token,rank:rank+1])
            ol.store(m.out[token,:],ol.load(m.out[token,:])+y[i,:]*weight)
        del y,xs,gathered,descriptor
    shared=m.fmt.descs(m.shared_base)
    y=swiglu_down(ol.quantize(xn),shared.wg,shared.wu,shared.wd,loop=True)
    # GLM's shared expert is unconditional, without Qwen's extra sigmoid gate.
    ol.store(m.out,ol.load(m.out)+y+x)


@ol.jit
def glm_dense(m):
    """The first decoder layer's ordinary SwiGLU, without MoE routing."""
    from opentpu.kernels.mlp import swiglu_down
    x=ol.load(m.x)
    y=swiglu_down(ol.quantize(rmsnorm(x,ol.load(m.norm),m.eps)),
        m.dense.wg,m.dense.wu,m.dense.wd,loop=True)
    ol.store(m.out,x+y)


@ol.jit
def glm_embedding(m,tokens):
    """Compiler-known token IDs gathered by real MXU instructions, without host dequantization."""
    from opentpu.kernels.gather import operand,gather_row
    onehot=operand(m.onehot)
    for r,token in enumerate(tokens):
        ol.store(m.out[r,:],gather_row(onehot,m.table,token,'nvfp4'))


@ol.jit
def glm_head(m):
    """Final norm and untied vocabulary projection of the frontend's last row."""
    rows=m.x.shape[0]
    xs=ol.quantize(rmsnorm(ol.load(m.x[rows-1:rows,:]),ol.load(m.norm),m.eps))
    chunk=min(8192,ol.tmem_words()//8)
    for start in range(0,m.head.shape[0],chunk):
        end=min(start+chunk,m.head.shape[0])
        ol.store(m.out[:,start:end],ol.dot(xs,m.head[start:end,:]))
