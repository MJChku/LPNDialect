"""Compile and validate GLM's real FP32 router hardware instructions.

Expert IDs are computed by ARGMAX/RLD from numerical input, never supplied as
annotations. This tests the router component, not full GLM text generation.
"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
from types import SimpleNamespace
import time

from .e2e import atomic_json,digest
from .workload import load,use_checkout


def compile_router(cfg,x,norm,weight,bias,topk=4):
    from opentpu.runtime import Input,Output,place
    from .glm47_kernels import glm_router
    R,H=x.shape;E=weight.shape[0]
    if weight.shape[1]!=H or not 1<=topk<=E:raise ValueError('invalid router dimensions')
    memory,layout,bound=place(cfg,dict(x=Input(x),norm=Input(norm),router=Input(weight),bias=Input(bias),
        logits=Output((R,E)),ids=Output((R,topk)),weights=Output((R,topk)),eps=1e-5))
    m=SimpleNamespace(**bound[0])
    # Device workspace is written through an affine loop address. It is not
    # a host-collected static store record; callers explicitly read its DRAM.
    m.logits=replace(m.logits,host=None)
    builder=glm_router.trace(cfg,0,dict(m=m,topk=topk,scale=1.8))
    program=builder.finish()
    if len(program)*8>cfg.IMEM_WORDS:raise ValueError('router program exceeds physical IMEM')
    return program,m,memory.images[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('opentpu-root','compiler','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rows',type=int,default=4);p.add_argument('--hidden',type=int,default=2048)
    p.add_argument('--experts',type=int,default=64);p.add_argument('--rtl',action='store_true')
    a=p.parse_args();use_checkout(a.opentpu_root);a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from opentpu import Config,isa as I,rtlsim
    from .benchmark import write_inputs
    from .hardware_commands import instruction_statistics
    from .native_timing import ReusableTiming
    from .glm47_target import component_timing
    if min(a.rows,a.hidden)<1 or a.experts<4:p.error('positive shapes and at least four experts required')
    rng=np.random.default_rng(1447);R,H,E=a.rows,a.hidden,a.experts;topk=4
    x=rng.normal(size=(R,H)).astype(np.float32);norm=(1+.1*rng.normal(size=H)).astype(np.float32)
    weight=(rng.normal(size=(E,H))/H**.5).astype(np.float32)
    bias=(rng.normal(size=E)*.3).astype(np.float32)
    # Deliberately force an otherwise weak expert into the selection; its
    # unbiased sigmoid score must still supply its mixture weight.
    bias[-1]=2
    cfg=Config(D=128,MCOLS=4,ACT_ROWS=16,LANES=64,IMEM_WORDS=65536)
    before=time.perf_counter();program,m,image=compile_router(cfg,x,norm,weight,bias,topk)
    write_inputs(a.output,cfg,[program],[image]);work=load(a.output);golden=work.machine.slices[0]
    def get(desc):
        start=desc.base.static();return golden.dram[start:start+4*np.prod(desc.shape)].view(np.float32).reshape(desc.shape)
    ids=get(m.ids).astype(int);weights=get(m.weights);logits=get(m.logits)
    xn=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5)*norm
    ref_logits=xn@weight.T;scores=1/(1+np.exp(-ref_logits));choice=scores+bias
    expected=np.argsort(-choice,axis=-1,kind='stable')[:,:topk]
    selected=np.take_along_axis(scores,expected,axis=1)
    reference_weights=selected/(selected.sum(axis=1,keepdims=True)+1e-20)*1.8
    if not np.array_equal(ids,expected):raise ValueError('actual hardware router selected different experts')
    error=float(np.max(np.abs(weights-reference_weights)))
    if error>2e-5:raise ValueError(f'FP32 router weights differ: {error}')
    if not np.all(ids[:,0]==E-1):raise ValueError('correction-bias test did not select forced expert')
    arithmetic=instruction_statistics(work)
    if arithmetic['matrix_commands'] or arithmetic['opcode_counts'].get('RLD',0)!=R*topk:
        raise ValueError('router did not use FP32 vector math and real data-dependent selection')
    timing=component_timing()
    prepared=ReusableTiming(a.compiler,a.output/'native-cache',capacity=len(work.commands[0])).prepare(
        replace(work,config=replace(cfg,MCOLS=2)),timing,a.output/'timing.tables')
    cycles,seconds,_,log=prepared.run();(a.output/'lpn.log').write_text(log)
    result=dict(status='complete',scope=__doc__,timing=asdict(timing),physical_epilogue_lanes=timing.epilogue_lanes,rows=R,hidden=H,experts=E,
        hardware_instructions=arithmetic,ids=ids.tolist(),weights=weights.tolist(),
        logits_max_absolute_error=float(np.max(np.abs(logits-ref_logits))),weights_max_absolute_error=error,
        numerical_expert_selection_exact=True,routing_annotations_used=False,
        lpn_cycles=cycles,lpn_seconds=seconds,rtl_validated=False,full_glm_model=False,
        input_sha256=work.hashes,
        source_sha256={str(f):digest(f) for f in (Path(__file__),Path(__file__).with_name('glm47_kernels.py'))})
    if a.rtl:
        rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='8'
        rtlsim.VFLAGS+=['-CFLAGS','-O0','-DOTPU_FP_MUL_LAT=2','-DOTPU_FP_ADD_LAT=4']
        ua=dict(MXU_ROWS=2,MXU_COLS=2,MXU_PAIR_DOT=1,MXU_CL=8,VPU_LANES=64,QST_LANES=64)
        drams,tmems,stats=rtlsim.run(cfg,[program],[image],uarch=ua,keep=a.output/'rtl',trace=True,max_cycles=10_000_000)
        (a.output/'rtl.log').write_text(stats['trace'])
        if not np.array_equal(drams[0],golden.dram) or not np.array_equal(tmems[0],golden.tmem):
            raise ValueError('router full RTL DRAM/TMEM differs from independent ISA')
        result.update(rtl_validated=True,rtl_cycles=stats['cycles'],relative_cycle_error=cycles/stats['cycles']-1,
            full_dram_and_tmem_bit_exact=True,timing_target_met=abs(cycles/stats['cycles']-1)<=.1)
    result['host_seconds']=time.perf_counter()-before;atomic_json(a.output/'report.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
