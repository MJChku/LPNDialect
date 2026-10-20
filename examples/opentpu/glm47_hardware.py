"""Compile GLM MLA into actual openTPU programs and replay the component LPN.

This is one attention layer/tile, not complete model inference. Full-size mode
uses the pinned model dimensions and never allocates weights. Small numerical
gates use the same compiler kernels, explicit test weights and the real ISA/RTL.
The compiler frontend owns operations and masks; LPN receives only hardware hex.
"""
import argparse
from dataclasses import asdict,dataclass,replace
import json
from math import prod
from pathlib import Path
from types import SimpleNamespace
import time

from . import REVISION
from .e2e import atomic_json,digest
from .glm47 import pinned_config
from .workload import use_checkout


@dataclass(frozen=True)
class Spec:
    hidden: int
    heads: int
    q_rank: int
    kv_rank: int
    nope: int
    rope: int
    value: int
    eps: float=1e-5
    theta: float=1e6

    @classmethod
    def pinned(cls):
        c,_=pinned_config()
        return cls(c['hidden_size'],c['num_attention_heads'],c['q_lora_rank'],c['kv_lora_rank'],
            c['qk_nope_head_dim'],c['qk_rope_head_dim'],c['v_head_dim'],c['rms_norm_eps'],c['rope_theta'])


class Layout:
    """Bounded per-layer physical layout using actual packed weight/KV formats."""
    def __init__(self,spec,rows,capacity,cfg):
        from opentpu import quant as Q
        from opentpu.compiler import Affine,KVDesc,QTensor,Tensor,k_scale_bytes
        if cfg.S!=1 or spec.value!=spec.nope+spec.rope:
            raise ValueError('initial GLM MLA backend requires one slice and equal Q/K and V head widths')
        if rows<1 or rows>cfg.act_rows or capacity<rows or capacity%cfg.D:
            raise ValueError('tile must fit ACT rows and KV capacity must contain whole D blocks')
        if spec.rope%2 or any(n%cfg.D for n in (spec.hidden,spec.q_rank,spec.kv_rank,spec.nope+spec.rope,spec.heads*spec.value)):
            raise ValueError('hardware inner dimensions must be D-aligned and RoPE must be even')
        self.regions=[];self.weights={};self.next=0;self.cfg=cfg;self.spec=spec
        def alloc(name,n,kind,shape=None):
            at=(self.next+cfg.D-1)//cfg.D*cfg.D;self.next=at+n
            if self.next>cfg.DRAM_BYTES or self.next>1<<32:raise ValueError('per-layer layout exceeds hardware address/memory capacity')
            self.regions.append(dict(name=name,address=at,bytes=n,kind=kind,shape=shape));return at
        def tensor(name,shape):
            at=alloc(name,prod(shape)*4,'fp32',shape);strides=[];stride=1
            for n in reversed(shape):strides.insert(0,stride);stride*=n
            return Tensor(Affine(at),tuple(shape),tuple(strides))
        def weight(name,n,k):
            data=alloc(name+'.data',n*Q.row_bytes(k,'nvfp4',cfg.D),'nvfp4',(n,k))
            scale=alloc(name+'.scales',n*Q.scale_bytes(k,'nvfp4',cfg.D),'mxu_scale_rows')
            result=QTensor.weight(data,scale,n,k,'nvfp4',cfg.D);self.weights[name]=result;return result
        h,d=spec.hidden,spec.nope+spec.rope
        m=SimpleNamespace(spec=spec,x=tensor('input',(rows,h)),out=tensor('output',(rows,h)),
            input_norm=tensor('input_norm',(h,)),q_norm=tensor('q_norm',(spec.q_rank,)),
            kv_norm=tensor('kv_norm',(spec.kv_rank,)),cos=tensor('cos',(rows,spec.rope//2)),
            sin=tensor('sin',(rows,spec.rope//2)),query=tensor('query',(rows,spec.heads*d)),
            attention=tensor('attention',(rows,spec.heads*spec.value)))
        m.qa=weight('qa',spec.q_rank,h);m.qb=weight('qb',spec.heads*d,spec.q_rank)
        m.kva=weight('kva',spec.kv_rank+spec.rope,h)
        m.kvb=weight('kvb',spec.heads*(spec.nope+spec.value),spec.kv_rank)
        m.wo=weight('wo',h,spec.heads*spec.value)
        heads={}
        for head in range(spec.heads):
            heads[head]={key:alloc(f'kv.{head}.{key}',n,kind) for key,n,kind in [
                ('k',capacity*d,'int8'),('ks',capacity*k_scale_bytes(d,cfg.D),'mxu_scale_rows'),
                ('vt',capacity*d,'int8_transposed_tiled'),('vs',capacity*4,'fp32')]}
        m.kv=KVDesc(heads,capacity,d,cfg.D,1,0);self.m=m

    def compile(self,p0,block=128,stage='layer',interleave=True):
        from .glm47_kernels import mla_layer,mla_project_cache,mla_attention,mla_output
        from opentpu.compiler import trace_fit
        if not 0<=p0<=self.m.kv.cap-self.m.x.shape[0] or block%self.cfg.D:
            raise ValueError('invalid token position or attention block')
        kernel={'layer':mla_layer,'project':mla_project_cache,'attention':mla_attention,'output':mla_output}[stage]
        args=dict(m=self.m)
        if stage!='output':args['p0']=p0
        if stage in ('layer','attention'):args['block']=block
        if stage in ('layer','project'):args['interleave']=interleave
        program=trace_fit(kernel,self.cfg,0,lambda:args).finish()
        if len(program)*8>self.cfg.IMEM_WORDS:raise ValueError('compiled MLA program exceeds physical IMEM; split into explicit hardware launches')
        return program


def numerical_inputs(layout,p0,seed=47):
    """Small validation data only; never used to synthesize timing latencies."""
    import numpy as np
    from opentpu import quant as Q,fp32 as F
    from opentpu.runtime import quantize_rows
    from opentpu.compiler import vt_tiled
    s=layout.spec;m=layout.m;cfg=layout.cfg;rng=np.random.default_rng(seed)
    image=np.zeros(cfg.DRAM_BYTES,np.uint8);reference={}
    def put(at,data):
        data=np.ascontiguousarray(data).view(np.uint8).ravel();image[at:at+data.size]=data
    for name in ('x','input_norm','q_norm','kv_norm'):
        desc=getattr(m,name)
        data=(rng.normal(size=desc.shape).astype(np.float32) if name=='x' else
              (1+.1*rng.normal(size=desc.shape)).astype(np.float32))
        put(desc.base.static(),data);reference[name]=data
    for name,desc in layout.weights.items():
        n,k=desc.shape;w=(rng.normal(size=(n,k))/k**.5).astype(np.float32)
        q,sc=Q.quantize_mxu(w,'nvfp4',cfg.D)
        put(desc.data.static(),q);put(desc.scale.static(),sc)
        reference[name]=Q.dequantize_mxu(q,sc,'nvfp4',k,cfg.D)
    angle=np.arange(p0,p0+m.x.shape[0])[:,None]*s.theta**(-np.arange(0,s.rope,2)/s.rope)
    for name,data in [('cos',np.cos(angle)),('sin',np.sin(angle))]:
        data=data.astype(np.float32);put(getattr(m,name).base.static(),data);reference[name]=data
    old_k=rng.normal(size=(s.heads,p0,s.nope+s.rope)).astype(np.float32)
    old_v=rng.normal(size=(s.heads,p0,s.value)).astype(np.float32)
    reference['old_k']=old_k;reference['old_v']=old_v
    for h in range(s.heads):
        region=m.kv.heads[h]
        if p0:
            q,sc=quantize_rows(old_k[h],cfg.D,row=m.kv.d==cfg.D)
            put(region['k'],q);put(region['ks'],sc)
            q,scale=F.quantize(old_v[h],axis=1)
            whole=np.zeros((m.kv.cap,m.kv.d),np.int8);whole[:p0]=q
            put(region['vt'],vt_tiled(whole.T));put(region['vs'],scale)
    return image,reference


def float_reference(layout,data,interleave=True):
    """Independent NumPy expression of expanded GLM MLA using dequantized weights."""
    import numpy as np
    s=layout.spec;R=data['x'].shape[0];d=s.nope+s.rope
    def norm(x,g,eps):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)*g
    def rotate(x):
        a,b=(x[...,0::2],x[...,1::2]) if interleave else np.split(x,2,axis=-1)
        return np.concatenate((a*data['cos'][:,None,:]-b*data['sin'][:,None,:],
            b*data['cos'][:,None,:]+a*data['sin'][:,None,:]),axis=-1)
    xn=norm(data['x'],data['input_norm'],s.eps)
    qa=norm(xn@data['qa'].T,data['q_norm'],1e-6)
    q=(qa@data['qb'].T).reshape(R,s.heads,d);q[...,s.nope:]=rotate(q[...,s.nope:])
    compressed=xn@data['kva'].T
    latent=norm(compressed[:,:s.kv_rank],data['kv_norm'],1e-6)
    expanded=(latent@data['kvb'].T).reshape(R,s.heads,s.nope+s.value)
    kr=rotate(compressed[:,None,s.kv_rank:])
    k=np.concatenate((expanded[...,:s.nope],np.broadcast_to(kr,(R,s.heads,s.rope))),axis=-1)
    v=expanded[...,s.nope:]
    keys=np.concatenate((data['old_k'],k.transpose(1,0,2)),axis=1)
    values=np.concatenate((data['old_v'],v.transpose(1,0,2)),axis=1)
    output=np.empty((R,s.heads,s.value),np.float32);p0=data['old_k'].shape[1]
    for r in range(R):
        for h in range(s.heads):
            scores=keys[h,:p0+r+1]@q[r,h]*d**-.5
            p=np.exp(scores-scores.max());p/=p.sum()
            output[r,h]=p@values[h,:p0+r+1]
    return output.reshape(R,-1)@data['wo'].T+data['x']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('opentpu-root','compiler','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rows',type=int,default=4);p.add_argument('--position',type=int,default=0)
    p.add_argument('--capacity',type=int,default=1152);p.add_argument('--stage',choices=('layer','project','attention','output'),default='layer')
    p.add_argument('--small-numerical-test',action='store_true');p.add_argument('--rtl',action='store_true')
    a=p.parse_args();use_checkout(a.opentpu_root);a.output.mkdir(parents=True,exist_ok=False)
    from opentpu import Config,isa as I,rtlsim
    from .hardware_commands import load_commands,instruction_statistics
    from .native_timing import ReusableTiming
    from .model import Timing
    s=Spec(256,2,256,128,64,64,128) if a.small_numerical_test else Spec.pinned()
    cfg=Config(D=128,MCOLS=4,ACT_ROWS=16,LANES=64,DRAM_BYTES=(1<<20 if a.small_numerical_test else 64<<20),IMEM_WORDS=65536)
    before=time.perf_counter();layout=Layout(s,a.rows,a.capacity,cfg);program=layout.compile(a.position,stage=a.stage)
    metadata=dict(revision=REVISION,config=asdict(cfg),args=[],instruction_counts=[len(program)])
    atomic_json(a.output/'input.json',metadata);rtlsim._write_hex(a.output/'prog_0.hex',I.assemble(program))
    atomic_json(a.output/'layout.json',dict(regions=layout.regions,total_bytes=layout.next))
    work=load_commands(a.output)
    timing=Timing(batch_rows=2,matrix_row_lanes=2,fifo_chunks=256,scale_lanes=2,chain_length=8,
        dram_bytes_per_cycle=2816,dram_port_bytes=2816,unified_dram=True,scalar_store_bytes=256,
        quant_row_banks=8,quant_store_buffered=True,vector_row_contexts=4,vector_command_queue=2,
        matrix_retirement=True,physical_scale_lines=True)
    prepared=ReusableTiming(a.compiler,a.output/'native-cache',capacity=max(len(c) for c in work.commands)).prepare(
        replace(work,config=replace(cfg,MCOLS=2)),timing,a.output/'timing.tables')
    cycles,seconds,_,log=prepared.run();(a.output/'lpn.log').write_text(log)
    result=dict(status='complete',scope=__doc__,spec=asdict(s),rows=a.rows,position=a.position,
        capacity=a.capacity,stage=a.stage,static_instructions=len(program),
        hardware_instructions=instruction_statistics(work),lpn_cycles=cycles,lpn_seconds=seconds,
        command_sha256=digest(a.output/'prog_0.hex'),layout_bytes=layout.next,
        numerical_validated=False,rtl_validated=False,full_glm_model=False,
        source_sha256={str(f):digest(f) for f in (Path(__file__),Path(__file__).with_name('glm47_kernels.py'))})
    if a.small_numerical_test:
        if a.stage!='layer':p.error('numerical test currently requires the complete MLA layer')
        import numpy as np
        from opentpu.isasim import Machine
        image,data=numerical_inputs(layout,a.position)
        image.view('<u4').astype('>u4').tofile(a.output/'dram_0.bin')
        golden=Machine(cfg,[program],[image]).run().slices[0]
        at=layout.m.out.base.static();got=golden.dram[at:at+4*a.rows*s.hidden].view(np.float32).reshape(a.rows,s.hidden)
        ref=float_reference(layout,data);error=float(np.linalg.norm(got-ref)/np.linalg.norm(ref))
        if not np.isfinite(error) or error>.1:raise ValueError(f'MLA numerical comparison failed: {error}')
        result.update(numerical_validated=True,reference_relative_l2_error=error,
            numerical_scope='Small same-kernel test; independent FP32 expanded-MLA equations using dequantized NVFP4 weights; activation/KV quantization differences expected')
        if a.rtl:
            rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='8'
            rtlsim.VFLAGS+=['-CFLAGS','-O0','-DOTPU_FP_MUL_LAT=2','-DOTPU_FP_ADD_LAT=4']
            ua=dict(MXU_ROWS=2,MXU_COLS=2,MXU_PAIR_DOT=1,MXU_CL=8,VPU_LANES=64,QST_LANES=64)
            drams,tmems,stats=rtlsim.run(cfg,[program],[image],uarch=ua,keep=a.output/'rtl',trace=True,max_cycles=10_000_000)
            (a.output/'rtl.log').write_text(stats['trace'])
            if not np.array_equal(drams[0],golden.dram) or not np.array_equal(tmems[0],golden.tmem):
                raise ValueError('MLA complete RTL DRAM/TMEM state differs from ISA')
            error=cycles/stats['cycles']-1
            result.update(rtl_validated=True,rtl_cycles=stats['cycles'],relative_cycle_error=error,
                full_dram_and_tmem_bit_exact=True,timing_target_met=abs(error)<=.1)
    elif a.rtl:p.error('full-size RTL is not launched without numerical inputs')
    result['host_seconds']=time.perf_counter()-before;atomic_json(a.output/'report.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
