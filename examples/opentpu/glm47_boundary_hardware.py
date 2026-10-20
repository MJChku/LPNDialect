"""Compile actual GLM embedding, first dense FFN and final head hardware probes.

These are component programs, not a complete model or an inference scheduler.
Default probes use real model dimensions without weight allocation. Numerical
tests execute the same kernels with small deterministic test weights.
"""
import argparse
from dataclasses import asdict,replace
import json
from math import prod
from pathlib import Path
from types import SimpleNamespace

from . import REVISION
from .e2e import atomic_json,digest
from .workload import use_checkout


class Layout:
    def __init__(self,stage,rows,H,F,V):
        from opentpu import Config,quant as Q
        from opentpu.compiler import Affine,Tensor,QTensor
        from opentpu.kernels.gather import table,onehot
        from opentpu.llm.moe import ExpertFormat
        if stage not in ('embedding','dense','head') or not 1<=rows<=16 or H%128 or F%256 or V<1:
            raise ValueError('invalid component or physical dimensions')
        self.next=0;self.regions=[];self.stage=stage;self.fmt=None;self.onehot=None
        def alloc(name,n):
            at=(self.next+127)//128*128;self.next=at+n
            if self.next>1<<32:raise ValueError('component exceeds hardware address space')
            self.regions.append(dict(name=name,address=at,bytes=n));return at
        def tensor(name,shape):
            n=1;strides=[]
            for dim in reversed(shape):strides.insert(0,n);n*=dim
            return Tensor(Affine(alloc(name,n*4)),tuple(shape),tuple(strides))
        self.m=m=SimpleNamespace(eps=1e-5)
        if stage!='embedding':m.x=tensor('input',(rows,H));m.norm=tensor('norm',(H,))
        m.out=tensor('output',(rows,H) if stage!='head' else (1,V))
        if stage=='dense':
            self.fmt=ExpertFormat(H,F,128,'nvfp4');m.dense=self.fmt.descs(alloc('dense',self.fmt.nbytes))
        else:
            data=alloc('weight.data',V*Q.row_bytes(H,'nvfp4',128))
            scale=alloc('weight.scales',V*Q.scale_bytes(H,'nvfp4',128))
            m.head=QTensor.weight(data,scale,V,H,'nvfp4',128)
            if stage=='embedding':
                gs=alloc('embedding.gather_scales',V*Q.gather_scale_bytes(H,'nvfp4',128))
                m.table=table(data,gs,V,H,'nvfp4',128)
                self.onehot=onehot(128,4,'nvfp4');m.onehot=tensor('embedding.onehot',self.onehot.shape)
        self.cfg=Config(D=128,MCOLS=4,ACT_ROWS=16,LANES=64,IMEM_WORDS=65536,
            DRAM_BYTES=1<<max(20,(self.next-1).bit_length()))

    def compile(self,tokens=None):
        from opentpu.compiler import trace_fit
        from .glm47_kernels import glm_dense,glm_embedding,glm_head
        args=dict(m=self.m)
        if self.stage=='embedding':
            if tokens is None or len(tokens)!=self.m.out.shape[0] or any(type(t) is not int or not 0<=t<self.m.table.shape[0] for t in tokens):
                raise ValueError('actual in-range token IDs required')
            args['tokens']=tokens
        kernel=dict(dense=glm_dense,embedding=glm_embedding,head=glm_head)[self.stage]
        program=trace_fit(kernel,self.cfg,0,lambda:args).finish()
        if len(program)*8>self.cfg.IMEM_WORDS:raise ValueError('component program exceeds physical IMEM')
        return program


def numerical_inputs(layout,tokens=None):
    import numpy as np
    from opentpu import quant as Q
    m=layout.m;H=m.out.shape[1] if layout.stage=='embedding' else m.x.shape[1]
    rng=np.random.default_rng(470041);image=np.zeros(layout.cfg.DRAM_BYTES,np.uint8)
    def put(at,value):
        raw=np.ascontiguousarray(value).view(np.uint8).ravel();image[at:at+raw.size]=raw
    def dequant(q,packed=None,base=0):
        raw=image if packed is None else packed;n,k=q.shape
        return Q.dequantize_mxu(raw[q.data.static()-base:q.data.static()-base+n*q.rs].reshape(n,q.rs),
            raw[q.scale.static()-base:q.scale.static()-base+n*q.srs].reshape(n,q.srs),'nvfp4',k,128)
    if layout.stage!='embedding':
        x=rng.normal(size=m.x.shape).astype(np.float32);norm=(1+.1*rng.normal(size=H)).astype(np.float32)
        put(m.x.base.static(),x);put(m.norm.base.static(),norm)
        xn=x/np.sqrt(np.mean(x*x,axis=1,keepdims=True)+1e-5)*norm
    if layout.stage=='dense':
        fmt=layout.fmt
        wg,wu=[(rng.normal(size=(fmt.F0,H))/H**.5).astype(np.float32) for _ in range(2)]
        wd=(rng.normal(size=(H,fmt.F0))/fmt.F0**.5).astype(np.float32)
        packed=fmt.pack(wg,wu,wd);base=m.dense.wg.data.static()-fmt.wg[0];put(base,packed)
        gate,up=dequant(m.dense.wg),dequant(m.dense.wu)
        down=np.concatenate([dequant(q) for q in m.dense.wd.parts],axis=1)
        z=xn@gate.T;reference=(z/(1+np.exp(-z))*(xn@up.T))@down.T+x
    else:
        w=(rng.normal(size=m.head.shape)/H**.5).astype(np.float32)
        q,s=Q.quantize_mxu(w,'nvfp4',128);put(m.head.data.static(),q);put(m.head.scale.static(),s)
        w=dequant(m.head)
        if layout.stage=='embedding':
            put(m.table.scale.static(),Q.gather_scales(s,'nvfp4',H,128))
            put(m.onehot.base.static(),layout.onehot);reference=w[tokens]
        else:reference=xn[-1:]@w.T
    return image,reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('opentpu-root','compiler','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=('embedding','dense','head'),required=True)
    p.add_argument('--rows',type=int,default=4);p.add_argument('--small-numerical-test',action='store_true')
    p.add_argument('--rtl',action='store_true');a=p.parse_args()
    if a.rtl and not a.small_numerical_test:p.error('RTL probe requires numerical inputs')
    use_checkout(a.opentpu_root);a.output.mkdir(parents=True,exist_ok=False)
    from opentpu import isa as I,rtlsim
    from .glm47 import pinned_config
    from .glm47_target import component_timing
    from .hardware_commands import load_commands,instruction_statistics
    from .native_timing import ReusableTiming
    c,_=pinned_config();H,F,V=(128,512,256) if a.small_numerical_test else (c['hidden_size'],c['intermediate_size'],c['vocab_size'])
    layout=Layout(a.stage,a.rows,H,F,V);tokens=[(7+31*r)%V for r in range(a.rows)]
    program=layout.compile(tokens);cfg=layout.cfg
    atomic_json(a.output/'input.json',dict(revision=REVISION,config=asdict(cfg),args=[],instruction_counts=[len(program)]))
    rtlsim._write_hex(a.output/'prog_0.hex',I.assemble(program))
    work=load_commands(a.output);timing=component_timing()
    native=ReusableTiming(a.compiler,a.output/'native-cache',capacity=32768).prepare(
        replace(work,config=replace(cfg,MCOLS=2)),timing,a.output/'timing.tables')
    cycles,seconds,_,log=native.run();(a.output/'lpn.log').write_text(log)
    result=dict(status='complete',scope=__doc__,stage=a.stage,rows=a.rows,hidden=H,ffn=F,vocab=V,
        timing=asdict(timing),hardware_instructions=instruction_statistics(work),lpn_cycles=cycles,lpn_seconds=seconds,
        full_glm_model=False,numerical_validated=False,rtl_validated=False,physical_layout_bytes=layout.next,
        source_sha256={str(f):digest(f) for f in (Path(__file__),Path(__file__).with_name('glm47_kernels.py'),Path(__file__).with_name('glm47_target.py'))})
    if a.small_numerical_test:
        import numpy as np
        from .workload import load
        image,reference=numerical_inputs(layout,tokens);image.view('<u4').astype('>u4').tofile(a.output/'dram_0.bin')
        numerical=load(a.output);gold=numerical.machine.slices[0];at=layout.m.out.base.static()
        got=gold.dram[at:at+prod(layout.m.out.shape)*4].view(np.float32).reshape(layout.m.out.shape)
        error=float(np.linalg.norm(got-reference)/np.linalg.norm(reference))
        if not np.isfinite(error) or error>.1:raise ValueError(f'numerical component mismatch: {error}')
        result.update(numerical_validated=True,reference_relative_l2_error=error,
            input_sha256={n:digest(a.output/n) for n in ('input.json','prog_0.hex','dram_0.bin')})
        if a.rtl:
            rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='8'
            rtlsim.VFLAGS+=['-CFLAGS','-O0','-DOTPU_FP_MUL_LAT=2','-DOTPU_FP_ADD_LAT=4']
            ua=dict(MXU_ROWS=2,MXU_COLS=2,MXU_PAIR_DOT=1,MXU_CL=8,VPU_LANES=64,QST_LANES=64)
            drams,tmems,stats=rtlsim.run(cfg,[program],[image],uarch=ua,keep=a.output/'rtl',trace=True,max_cycles=5_000_000)
            (a.output/'rtl.log').write_text(stats['trace'])
            if not np.array_equal(drams[0],gold.dram) or not np.array_equal(tmems[0],gold.tmem):raise ValueError('complete RTL state differs')
            error=cycles/stats['cycles']-1
            result.update(rtl_validated=True,rtl_cycles=stats['cycles'],relative_cycle_error=error,
                full_dram_and_tmem_bit_exact=True,timing_target_met=abs(error)<=.1)
    atomic_json(a.output/'report.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
