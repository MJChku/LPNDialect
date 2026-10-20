"""Compile batched GLM expert dispatch from an actual executed router program.

The frontend groups real functional IDs and emits ordinary hardware programs;
the LPN receives encoded instructions only. This covers one resident MoE layer
tile. It does not solve whole-model >32-bit addressing or full GLM generation.
"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
from types import SimpleNamespace
import time

from . import REVISION
from .e2e import atomic_json,digest
from .workload import load,use_checkout


def resolved_router(directory):
    """Recover routing from fresh numerical ISA execution, binding it to hex."""
    import numpy as np
    from opentpu import Config
    from .glm47_router_hardware import compile_router
    report=json.loads((directory/'report.json').read_text())
    if report['status']!='complete' or not report['numerical_expert_selection_exact'] or report['routing_annotations_used']:
        raise ValueError('completed actual functional router result required')
    work=load(directory)
    if work.hashes!=report['input_sha256']:raise ValueError('router input files changed')
    R,H,E=report['rows'],report['hidden'],report['experts'];raw=work.images[0]
    # Reconstruct the pinned router placement and independently require exact
    # instruction-word equality before interpreting any output address.
    at=0
    def array(shape):
        nonlocal at
        at=(at+127)//128*128;n=int(np.prod(shape))
        value=raw[at:at+4*n].view(np.float32).reshape(shape).copy();at+=4*n;return value
    x=array((R,H));norm=array((H,));w=array((E,H));bias=array((E,))
    program,m,_=compile_router(work.config,x,norm,w,bias)
    if [i.encode() for i in program]!=[i.encode() for i in work.programs[0]]:
        raise ValueError('router output address reconstruction differs from canonical instructions')
    memory=work.machine.slices[0].dram
    def get(desc):
        start=desc.base.static();return memory[start:start+4*np.prod(desc.shape)].view(np.float32).reshape(desc.shape).copy()
    ids=get(m.ids).astype(int).tolist();weights=get(m.weights)
    if ids!=report['ids'] or not np.array_equal(weights,np.asarray(report['weights'],np.float32)):
        raise ValueError('fresh functional routing differs')
    return x,norm,ids,weights,E


class Layout:
    def __init__(self,H,F,E,R,k=4,base=0,tmem_words=65536,logical_columns=4):
        from opentpu import Config
        from opentpu.compiler import Affine,Tensor
        from opentpu.llm.moe import ExpertFormat
        self.fmt=fmt=ExpertFormat(H,F,128,'nvfp4');self.regions=[];self.next=base
        def alloc(name,n):
            a=(self.next+127)//128*128;self.next=a+n
            self.regions.append(dict(name=name,address=a,bytes=n));return a
        def tensor(name,shape):
            count=1;strides=[]
            for n in reversed(shape):strides.insert(0,count);count*=n
            return Tensor(Affine(alloc(name,count*4)),tuple(shape),tuple(strides))
        m=SimpleNamespace(x=tensor('input',(R,H)),norm=tensor('post_attention_norm',(H,)),
            weights=tensor('router_weights',(R,k)),out=tensor('output',(R,H)),eps=1e-5,experts=E,fmt=fmt)
        m.expert_base=alloc('resident_experts',E*fmt.nbytes);m.shared_base=alloc('shared_expert',fmt.nbytes)
        if self.next>1<<32:raise ValueError('one resident layer exceeds the current hardware address space')
        self.cfg=Config(D=128,MCOLS=logical_columns,ACT_ROWS=16,LANES=64,IMEM_WORDS=65536,TMEM_WORDS=tmem_words,
            DRAM_BYTES=1<<max(20,(self.next-1).bit_length()))
        self.m=m

    def compile(self,ids):
        from .glm47_kernels import glm_resident_experts
        program=glm_resident_experts.trace(self.cfg,0,dict(m=self.m,ids=ids)).finish()
        if len(program)*8>self.cfg.IMEM_WORDS:raise ValueError('expert program exceeds physical IMEM')
        return program


def numerical_inputs(layout,x,norm,ids,weights):
    import numpy as np
    from opentpu import quant as Q
    m=layout.m;fmt=layout.fmt;rng=np.random.default_rng(4723)
    image=np.zeros(layout.cfg.DRAM_BYTES,np.uint8);reference={}
    def put(at,data):
        raw=np.ascontiguousarray(data).view(np.uint8).ravel();image[at:at+raw.size]=raw
    put(m.x.base.static(),x);put(m.norm.base.static(),norm);put(m.weights.base.static(),weights)
    for e in range(m.experts+1):
        wg=(rng.normal(size=(fmt.F0,fmt.H))/fmt.H**.5).astype(np.float32)
        wu=(rng.normal(size=(fmt.F0,fmt.H))/fmt.H**.5).astype(np.float32)
        wd=(rng.normal(size=(fmt.H,fmt.F0))/fmt.F0**.5).astype(np.float32)
        base=m.expert_base+e*fmt.nbytes if e<m.experts else m.shared_base
        packed=fmt.pack(wg,wu,wd);put(base,packed);desc=fmt.descs(base)
        def dequant(q):
            n,k=q.shape;at=q.data.static()-base;st=q.scale.static()-base
            return Q.dequantize_mxu(packed[at:at+n*q.rs].reshape(n,q.rs),
                packed[st:st+n*q.srs].reshape(n,q.srs),'nvfp4',k,128)
        down=np.concatenate([dequant(part) for part in desc.wd.parts],axis=1)
        reference[e]=(dequant(desc.wg),dequant(desc.wu),down)
    xn=x/np.sqrt(np.mean(x*x,axis=1,keepdims=True)+1e-5)*norm
    output=np.zeros_like(x)
    # Ascending expert order is the pinned upstream experts' summation order.
    for e in range(m.experts+1):
        gate,up,down=reference[e]
        hidden=xn@gate.T;hidden=hidden/(1+np.exp(-hidden))*(xn@up.T);value=hidden@down.T
        if e==m.experts:output+=value
        else:
            for token,row in enumerate(ids):
                if e in row:output[token]+=value[token]*weights[token,row.index(e)]
    return image,output+x


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('router-case','opentpu-root','compiler','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--ffn',type=int,default=1536);p.add_argument('--numerical',action='store_true');p.add_argument('--rtl',action='store_true')
    a=p.parse_args();use_checkout(a.opentpu_root);a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from opentpu import isa as I,rtlsim
    from .hardware_commands import load_commands,instruction_statistics
    from .native_timing import ReusableTiming
    from .glm47_target import component_timing
    before=time.perf_counter();x,norm,ids,weights,E=resolved_router(a.router_case)
    R,H=x.shape;layout=Layout(H,a.ffn,E,R);cfg=layout.cfg;program=layout.compile(ids)
    rtlsim._write_hex(a.output/'prog_0.hex',I.assemble(program))
    atomic_json(a.output/'input.json',dict(revision=REVISION,config=asdict(cfg),args=[],instruction_counts=[len(program)]))
    atomic_json(a.output/'layout.json',dict(regions=layout.regions,total_bytes=layout.next,expert_slot_bytes=layout.fmt.nbytes))
    work=load_commands(a.output);stats=instruction_statistics(work)
    expected=3*R*H*layout.fmt.F*(len(ids[0])+1)
    if stats['matrix_dot_macs']!=expected:raise ValueError('encoded expert work differs from selected batched SwiGLU work')
    timing=component_timing()
    native=ReusableTiming(a.compiler,a.output/'native-cache',capacity=len(work.commands[0])).prepare(
        replace(work,config=replace(cfg,MCOLS=2)),timing,a.output/'timing.tables')
    cycles,seconds,_,log=native.run();(a.output/'lpn.log').write_text(log)
    result=dict(status='complete',scope=__doc__,timing=asdict(timing),physical_epilogue_lanes=timing.epilogue_lanes,rows=R,hidden=H,ffn=a.ffn,experts=E,
        ids=ids,router_report_sha256=digest(a.router_case/'report.json'),fresh_actual_router_reexecuted=True,
        routing_annotations_used=False,hardware_instructions=stats,lpn_cycles=cycles,lpn_seconds=seconds,
        layout_bytes=layout.next,static_instructions=len(program),numerical_validated=False,rtl_validated=False,
        full_glm_model=False,source_sha256={str(f):digest(f) for f in (Path(__file__),Path(__file__).with_name('glm47_kernels.py'))})
    if a.numerical:
        if max(H,a.ffn)>1024 or E>16:p.error('large shape mode compiles without generating weight tensors')
        from opentpu.isasim import Machine
        image,reference=numerical_inputs(layout,x,norm,ids,weights)
        image.view('<u4').astype('>u4').tofile(a.output/'dram_0.bin')
        gold=Machine(cfg,[program],[image]).run().slices[0]
        at=layout.m.out.base.static();got=gold.dram[at:at+4*R*H].view(np.float32).reshape(R,H)
        error=float(np.linalg.norm(got-reference)/np.linalg.norm(reference))
        if not np.isfinite(error) or error>.15:raise ValueError(f'expert FP32-reference error: {error}')
        result.update(numerical_validated=True,reference_relative_l2_error=error)
        if a.rtl:
            rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='8';rtlsim.VFLAGS+=['-CFLAGS','-O0','-DOTPU_FP_MUL_LAT=2','-DOTPU_FP_ADD_LAT=4']
            ua=dict(MXU_ROWS=2,MXU_COLS=2,MXU_PAIR_DOT=1,MXU_CL=8,VPU_LANES=64,QST_LANES=64)
            drams,tmems,observed=rtlsim.run(cfg,[program],[image],uarch=ua,keep=a.output/'rtl',trace=True,max_cycles=2_000_000)
            (a.output/'rtl.log').write_text(observed['trace'])
            if not np.array_equal(drams[0],gold.dram) or not np.array_equal(tmems[0],gold.tmem):raise ValueError('expert full RTL state differs from ISA')
            result.update(rtl_validated=True,rtl_cycles=observed['cycles'],full_dram_and_tmem_bit_exact=True,
                relative_cycle_error=cycles/observed['cycles']-1,timing_target_met=abs(cycles/observed['cycles']-1)<=.1)
    elif a.rtl:p.error('RTL test requires numerical inputs')
    result['host_seconds']=time.perf_counter()-before;atomic_json(a.output/'report.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
