"""Run one integrated GLM decoder layer through actual hardware programs.

MLA, FP32 routing and grouped routed/shared experts share physical DRAM.
The functional ISA resolves router IDs before the expert program is compiled.
This is a test-weight layer gate, not trained checkpoint/full-model inference.
Host program-loading, route-readback and compilation overhead are not device
kernel cycles and are not folded into a claimed model-inference latency.
"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
from types import SimpleNamespace
import time

from .e2e import atomic_json,digest
from .workload import execute,use_checkout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('opentpu-root','compiler','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--small-test',action='store_true');p.add_argument('--rtl',action='store_true')
    p.add_argument('--position',type=int,default=0)
    a=p.parse_args();use_checkout(a.opentpu_root);a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from opentpu import Config,isa as I,rtlsim,waits as W
    from opentpu.compiler import Affine,Tensor
    from .glm47_hardware import Layout as MLA,Spec,numerical_inputs as mla_inputs,float_reference
    from .glm47_expert_hardware import Layout as Experts
    from .glm47_kernels import glm_router
    from .benchmark import write_inputs
    from .model import Timing
    from .native_timing import ReusableTiming
    from .hardware_commands import instruction_statistics
    started=time.perf_counter();R=2 if a.small_test else 4
    spec=Spec(128,2,256,128,64,64,128) if a.small_test else Spec.pinned()
    H=spec.hidden;E=4 if a.small_test else 64;F=512 if a.small_test else 1536
    capacity=max(256 if a.small_test else 1152,(a.position+R+127)//128*128)
    provisional=Config(D=128,MCOLS=4,ACT_ROWS=16,LANES=64,DRAM_BYTES=64<<20,IMEM_WORDS=65536)
    mla=MLA(spec,R,capacity,provisional);cursor=mla.next
    def tensor(shape):
        nonlocal cursor
        base=(cursor+127)//128*128;n=1;strides=[]
        for dim in reversed(shape):strides.insert(0,n);n*=dim
        cursor=base+4*n;return Tensor(Affine(base),tuple(shape),tuple(strides))
    router=SimpleNamespace(x=mla.m.out,norm=tensor((H,)),router=tensor((E,H)),bias=tensor((E,)),
        logits=tensor((R,E)),ids=tensor((R,4)),weights=tensor((R,4)),eps=1e-5)
    experts=Experts(H,F,E,R,base=cursor)
    experts.m.x=mla.m.out;experts.m.norm=router.norm;experts.m.weights=router.weights
    cfg=experts.cfg;mla.cfg=cfg
    program_mla=mla.compile(a.position)
    program_router=glm_router.trace(cfg,0,dict(m=router,topk=4,scale=1.8)).finish()
    image,data=mla_inputs(mla,a.position,seed=477)
    rng=np.random.default_rng(4747)
    gamma=(1+.1*rng.normal(size=H)).astype(np.float32)
    weight=(rng.normal(size=(E,H))/H**.5).astype(np.float32);bias=(rng.normal(size=E)*.1).astype(np.float32)
    def put(memory,at,value):
        raw=np.ascontiguousarray(value).view(np.uint8).ravel();memory[at:at+raw.size]=raw
    put(image,router.norm.base.static(),gamma);put(image,router.router.base.static(),weight)
    put(image,router.bias.base.static(),bias)
    first=execute(cfg,[program_mla],[image]);second=execute(cfg,[program_router],[first.machine.slices[0].dram])
    # The later expert weights cannot affect either prefix stage. Validate
    # this from actual resolved hardware footprints before lazy placement.
    pool=experts.m.expert_base
    for work in (first,second):
        for c in work.commands[0]:
            if any(space==W.SP_DRAM and hi>pool for space,lo,hi in c.footprint.rd+c.footprint.wr):
                raise ValueError('prefix accesses expert region before its data-independent lazy placement')
    def get(memory,desc):
        start=desc.base.static();return memory[start:start+4*np.prod(desc.shape)].view(np.float32).reshape(desc.shape).copy()
    mid=get(first.machine.slices[0].dram,mla.m.out)
    ids=get(second.machine.slices[0].dram,router.ids).astype(int).tolist()
    weights=get(second.machine.slices[0].dram,router.weights)
    xn=mid/np.sqrt(np.mean(mid*mid,axis=1,keepdims=True)+1e-5)*gamma
    scores=1/(1+np.exp(-(xn@weight.T)));expected=np.argsort(-(scores+bias),axis=1,kind='stable')[:,:4]
    if not np.array_equal(np.asarray(ids),expected):raise ValueError('integrated numerical router IDs differ')
    ref_weights=np.take_along_axis(scores,expected,axis=1);ref_weights=ref_weights/ref_weights.sum(axis=1,keepdims=True)*1.8
    if np.max(np.abs(weights-ref_weights))>2e-5:raise ValueError('integrated router weights differ')
    # Materialize only experts actually selected by this real functional
    # prefix, plus the shared expert. All E physical slots remain allocated.
    reference=np.zeros_like(mid);fmt=experts.fmt
    for e in sorted({e for row in ids for e in row})+[E]:
        erng=np.random.default_rng(47700+e)
        wg=(erng.normal(size=(fmt.F0,H))/H**.5).astype(np.float32)
        wu=(erng.normal(size=(fmt.F0,H))/H**.5).astype(np.float32)
        wd=(erng.normal(size=(H,fmt.F0))/fmt.F0**.5).astype(np.float32)
        packed=fmt.pack(wg,wu,wd);base=pool+e*fmt.nbytes if e<E else experts.m.shared_base
        # Prefixes only read lower regions, so adding the same immutable
        # weights to their initial/final states preserves exact continuity.
        for memory in (image,first.machine.slices[0].dram,second.machine.slices[0].dram):put(memory,base,packed)
        from opentpu import quant as Q
        desc=fmt.descs(base)
        def dequant(q):
            n,k=q.shape;at=q.data.static()-base;st=q.scale.static()-base
            return Q.dequantize_mxu(packed[at:at+n*q.rs].reshape(n,q.rs),
                packed[st:st+n*q.srs].reshape(n,q.srs),'nvfp4',k,128)
        gate,up=dequant(desc.wg),dequant(desc.wu)
        down=np.concatenate([dequant(q) for q in desc.wd.parts],axis=1)
        z=xn@gate.T;value=(z/(1+np.exp(-z))*(xn@up.T))@down.T
        if e==E:reference+=value
        else:
            for token,row in enumerate(ids):
                if e in row:reference[token]+=value[token]*weights[token,row.index(e)]
    program_experts=experts.compile(ids)
    third=execute(cfg,[program_experts],[second.machine.slices[0].dram])
    got=get(third.machine.slices[0].dram,experts.m.out)
    mla_error=float(np.linalg.norm(mid-float_reference(mla,data))/np.linalg.norm(mid))
    expert_error=float(np.linalg.norm(got-(reference+mid))/np.linalg.norm(got))
    if not np.isfinite(mla_error+expert_error) or max(mla_error,expert_error)>.15:
        raise ValueError(f'integrated stage math differs: {mla_error}, {expert_error}')
    timing=Timing(batch_rows=2,matrix_row_lanes=2,fifo_chunks=256,scale_lanes=2,chain_length=8,
        dram_bytes_per_cycle=2816,dram_port_bytes=2816,unified_dram=True,scalar_store_bytes=256,
        quant_row_banks=8,quant_store_buffered=True,vector_row_contexts=4,vector_command_queue=2,
        matrix_retirement=True,physical_scale_lines=True)
    runner=ReusableTiming(a.compiler,a.output/'native-cache',capacity=32768);records=[]
    items=[('mla',first,image),('router',second,first.machine.slices[0].dram),('experts',third,second.machine.slices[0].dram)]
    previous_rtl=None
    for name,work,initial in items:
        directory=a.output/name;directory.mkdir()
        write_inputs(directory,cfg,work.programs,[initial])
        native=runner.prepare(replace(work,config=replace(cfg,MCOLS=2)),timing,directory/'timing.tables')
        cycles,seconds,_,log=native.run();(directory/'lpn.log').write_text(log)
        record=dict(name=name,static_instructions=len(work.programs[0]),hardware_instructions=instruction_statistics(work),
            lpn_cycles=cycles,lpn_seconds=seconds,input_sha256={n:digest(directory/n) for n in ('input.json','prog_0.hex','dram_0.bin')})
        if a.rtl:
            if previous_rtl is not None and not np.array_equal(previous_rtl,initial):raise ValueError('actual RTL stage continuity failed')
            rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='8'
            flags=['-CFLAGS','-O0','-DOTPU_FP_MUL_LAT=2','-DOTPU_FP_ADD_LAT=4']
            if '-DOTPU_FP_ADD_LAT=4' not in rtlsim.VFLAGS:rtlsim.VFLAGS+=flags
            ua=dict(MXU_ROWS=2,MXU_COLS=2,MXU_PAIR_DOT=1,MXU_CL=8,VPU_LANES=64,QST_LANES=64)
            drams,tmems,observed=rtlsim.run(cfg,work.programs,[initial],uarch=ua,keep=directory/'rtl',trace=True,max_cycles=5_000_000)
            (directory/'rtl.log').write_text(observed['trace']);gold=work.machine.slices[0]
            if not np.array_equal(drams[0],gold.dram) or not np.array_equal(tmems[0],gold.tmem):raise ValueError('integrated full RTL state differs')
            previous_rtl=drams[0]
            record.update(rtl_cycles=observed['cycles'],full_dram_and_tmem_bit_exact=True,
                relative_cycle_error=cycles/observed['cycles']-1)
        records.append(record)
        atomic_json(a.output/'progress.json',dict(status='running',records=records))
    result=dict(status='complete',scope=__doc__,spec=asdict(spec),rows=R,experts=E,ffn=F,position=a.position,
        physical_layer_bytes=experts.next,config=asdict(cfg),records=records,ids=ids,
        mla_reference_relative_error=mla_error,expert_reference_relative_error=expert_error,
        all_stages_numerically_executed=True,routing_annotations_used=False,shared_physical_dram=True,
        compiler_expert_dispatch_from_actual_router_outputs=True,full_glm_model=False,weights='Deterministic numerical test weights',
        rtl_validated=a.rtl,host_seconds=time.perf_counter()-started,
        lpn_cycles_sum=sum(r['lpn_cycles'] for r in records),lpn_seconds_sum=sum(r['lpn_seconds'] for r in records),
        source_sha256={str(f):digest(f) for f in [Path(__file__),*[Path(__file__).with_name(n) for n in
            ('glm47_kernels.py','glm47_hardware.py','glm47_expert_hardware.py')]]})
    if a.rtl:
        result.update(rtl_cycles_sum=sum(r['rtl_cycles'] for r in records),
            all_full_dram_and_tmem_bit_exact=True,worst_relative_cycle_error=max(abs(r['relative_cycle_error']) for r in records))
    atomic_json(a.output/'report.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
