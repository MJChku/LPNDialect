"""Compile real FP32 routing and slice-owned GLM experts with hardware GATHER.

This is a numerical MoE component gate. Each slice has its own existing
32-bit DRAM address space; experts are partitioned and the shared expert runs
once. LPN receives canonical hardware instructions for all slices.
"""
import argparse
from dataclasses import asdict,replace
import json
from math import prod
from pathlib import Path
from types import SimpleNamespace
import time

from .e2e import atomic_json,digest
from .workload import use_checkout,execute


class Layout:
    def __init__(self,H,F,E,R,S):
        from opentpu.compiler import Affine,Tensor
        from .glm47_expert_hardware import Layout as ExpertLayout
        if S<1 or E%S or E<4 or S>64 or not 1<=R<=16:
            raise ValueError('experts must divide the slices and tile must fit ACT rows')
        local=ExpertLayout(H,F,E//S,R,tmem_words=262144 if R>4 else 65536,
            logical_columns=1<<max(2,(R-1).bit_length()))
        self.next=local.next;self.regions=list(local.regions);self.fmt=local.fmt;self.m=m=local.m
        def tensor(name,shape):
            base=(self.next+127)//128*128;n=prod(shape);strides=[];stride=1
            for dim in reversed(shape):strides.insert(0,stride);stride*=dim
            self.next=base+4*n;self.regions.append(dict(name=name,address=base,bytes=4*n))
            return Tensor(Affine(base),tuple(shape),tuple(strides))
        m.global_experts=E;m.router=tensor('router',(E,H));m.bias=tensor('correction_bias',(E,))
        m.logits=tensor('router_logits',(R,E));m.ids=tensor('router_ids',(R,4));m.partial=tensor('local_partial',(R,H))
        self.cfg=replace(local.cfg,S=S,DRAM_BYTES=1<<max(20,(self.next-1).bit_length()))
        if self.next>1<<32:raise ValueError('one slice exceeds its hardware address space')

    def router_programs(self):
        from .glm47_kernels import glm_router
        return [glm_router.trace(self.cfg,sid,dict(m=self.m,topk=4,scale=1.8)).finish()
                for sid in range(self.cfg.S)]

    def expert_programs(self,ids):
        from .glm47_distributed_kernels import distributed_experts
        programs=[distributed_experts.trace(self.cfg,sid,dict(m=self.m,ids=ids)).finish()
                  for sid in range(self.cfg.S)]
        if any(len(p)*8>self.cfg.IMEM_WORDS for p in programs):raise ValueError('expert program exceeds physical IMEM')
        return programs


def numerical_case(layout,correction_bias=None):
    import numpy as np
    from opentpu import quant as Q,waits as W
    m=layout.m;R,H=m.x.shape;E=m.global_experts;S=layout.cfg.S;rng=np.random.default_rng(47008)
    images=[np.zeros(layout.cfg.DRAM_BYTES,np.uint8) for _ in range(S)]
    def put(raw,at,value):
        data=np.ascontiguousarray(value).view(np.uint8).ravel();raw[at:at+data.size]=data
    x=rng.normal(size=(R,H)).astype(np.float32);norm=(1+.1*rng.normal(size=H)).astype(np.float32)
    router=(rng.normal(size=(E,H))/H**.5).astype(np.float32);bias=(rng.normal(size=E)*.1).astype(np.float32)
    if correction_bias is not None:
        bias=np.asarray(correction_bias,np.float32)
        if bias.shape!=(E,):raise ValueError('correction bias shape differs from expert count')
    for image in images:
        for tensor,value in ((m.x,x),(m.norm,norm),(m.router,router),(m.bias,bias)):put(image,tensor.base.static(),value)
    first=execute(layout.cfg,layout.router_programs(),images)
    def get(raw,desc):
        at=desc.base.static();return raw[at:at+prod(desc.shape)*4].view(np.float32).reshape(desc.shape).copy()
    ids=get(first.machine.slices[0].dram,m.ids).astype(int).tolist();weights=get(first.machine.slices[0].dram,m.weights)
    for machine in first.machine.slices:
        if get(machine.dram,m.ids).astype(int).tolist()!=ids or not np.array_equal(get(machine.dram,m.weights),weights):
            raise ValueError('replicated actual hardware routers disagree')
    # Lazy packed weights are immutable and invisible to the routing prefix.
    pool_end=m.expert_base+m.experts*m.fmt.nbytes
    shared_end=m.shared_base+m.fmt.nbytes
    for stream in first.commands:
        for command in stream:
            for space,lo,hi in command.footprint.rd+command.footprint.wr:
                if space==W.SP_DRAM and (max(lo,m.expert_base)<min(hi,pool_end) or max(lo,m.shared_base)<min(hi,shared_end)):
                    raise ValueError('router prefix touches lazily materialized expert weights')
    xn=x/np.sqrt(np.mean(x*x,axis=1,keepdims=True)+1e-5)*norm;reference=np.zeros_like(x)
    for expert in sorted({e for row in ids for e in row})+[E]:
        owner=expert//m.experts if expert<E else 0
        base=m.expert_base+(expert%m.experts)*m.fmt.nbytes if expert<E else m.shared_base
        erng=np.random.default_rng(4700800+expert);fmt=m.fmt
        wg,wu=[(erng.normal(size=(fmt.F0,H))/H**.5).astype(np.float32) for _ in range(2)]
        wd=(erng.normal(size=(H,fmt.F0))/fmt.F0**.5).astype(np.float32)
        packed=fmt.pack(wg,wu,wd)
        put(images[owner],base,packed);put(first.machine.slices[owner].dram,base,packed)
        descriptors=fmt.descs(base)
        def dequant(q):
            n,k=q.shape;at=q.data.static()-base;st=q.scale.static()-base
            return Q.dequantize_mxu(packed[at:at+n*q.rs].reshape(n,q.rs),
                packed[st:st+n*q.srs].reshape(n,q.srs),'nvfp4',k,128)
        gate,up=dequant(descriptors.wg),dequant(descriptors.wu)
        down=np.concatenate([dequant(p) for p in descriptors.wd.parts],axis=1)
        z=xn@gate.T;value=(z/(1+np.exp(-z))*(xn@up.T))@down.T
        if expert==E:reference+=value
        else:
            for token,row in enumerate(ids):
                if expert in row:reference[token]+=value[token]*weights[token,row.index(expert)]
    second=execute(layout.cfg,layout.expert_programs(ids),[s.dram for s in first.machine.slices])
    output=get(second.machine.slices[0].dram,m.out)
    for machine in second.machine.slices:
        if not np.array_equal(output,get(machine.dram,m.out)):raise ValueError('expert-owner GATHER outputs disagree')
    error=float(np.linalg.norm(output-reference-x)/np.linalg.norm(output))
    if not np.isfinite(error) or error>.1:raise ValueError(f'distributed expert numerical mismatch: {error}')
    return first,second,images,ids,error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('opentpu-root','compiler','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--small-test',action='store_true');p.add_argument('--rtl',action='store_true')
    p.add_argument('--slices',type=int);p.add_argument('--rows',type=int,default=4);a=p.parse_args()
    root=use_checkout(a.opentpu_root);a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    import numpy as np
    from opentpu import rtlsim
    from .benchmark import write_inputs
    from .glm47_target import component_timing
    from .native_timing import ReusableTiming
    from .hardware_commands import instruction_statistics
    H,F,E=(128,512,8) if a.small_test else (2048,1536,64)
    S=a.slices or (2 if a.small_test else 8);layout=Layout(H,F,E,a.rows,S)
    first,second,images,ids,error=numerical_case(layout);timing=replace(component_timing(),dma_store_runs=True)
    capacity=1<<max(len(c) for w in (first,second) for c in w.commands).bit_length()
    runner=ReusableTiming(a.compiler,a.output/'native-cache',capacity=capacity);records=[];previous_rtl=None
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='8'
    rtlsim.VFLAGS+=['-CFLAGS','-O0','-DOTPU_FP_MUL_LAT=2','-DOTPU_FP_ADD_LAT=4']
    ua=dict(MXU_ROWS=2,MXU_COLS=2,MXU_PAIR_DOT=1,MXU_CL=8,VPU_LANES=64,QST_LANES=64)
    for name,work,initial in [('router',first,images),('experts',second,[s.dram for s in first.machine.slices])]:
        directory=a.output/name;directory.mkdir();write_inputs(directory,layout.cfg,work.programs,initial)
        native=runner.prepare(replace(work,config=replace(layout.cfg,MCOLS=2)),timing,directory/'timing.tables')
        cycles,seconds,_,text=native.run();(directory/'lpn.log').write_text(text)
        record=dict(name=name,lpn_cycles=cycles,lpn_seconds=seconds,hardware_instructions=instruction_statistics(work),
            input_sha256={n:digest(directory/n) for n in ['input.json',*[f'prog_{s}.hex' for s in range(S)],*[f'dram_{s}.bin' for s in range(S)]]})
        if a.rtl:
            if previous_rtl is not None and any(not np.array_equal(a,b) for a,b in zip(previous_rtl,initial)):
                raise ValueError('actual cross-stage RTL DRAM continuity failed')
            drams,tmems,stats=rtlsim.run(layout.cfg,work.programs,initial,uarch=ua,keep=directory/'rtl',trace=True,max_cycles=10000000)
            (directory/'rtl.log').write_text(stats['trace'])
            for sid,gold in enumerate(work.machine.slices):
                if not np.array_equal(drams[sid],gold.dram) or not np.array_equal(tmems[sid],gold.tmem):
                    raise ValueError('complete distributed RTL state differs from ISA')
            previous_rtl=drams;relative=cycles/stats['cycles']-1
            record.update(rtl_cycles=stats['cycles'],relative_cycle_error=relative,
                all_full_memories_bit_exact=True,timing_target_met=abs(relative)<=.1)
        records.append(record);atomic_json(a.output/'progress.json',dict(status='running',records=records))
    files=[Path(__file__),*[Path(__file__).with_name(n) for n in ('glm47_distributed_kernels.py','glm47_kernels.py',
        'glm47_expert_hardware.py','collective.py','compact.py','model.py','components.py','glm47_target.py')],
        *[rtlsim.RTL/n for n in rtlsim.RTL_SOURCES],root/'opentpu/rtlsim.py']
    result=dict(status='complete',scope=__doc__,slices=S,rows=a.rows,hidden=H,experts=E,ffn=F,
        config=asdict(layout.cfg),per_slice_layout_bytes=layout.next,ids=ids,records=records,timing=asdict(timing),
        reference_relative_l2_error=error,routing_annotations_used=False,shared_expert_executions=1,
        expert_partition='contiguous whole experts, owned by floor(e/experts_per_slice)',
        actual_hardware_gather=S>1,rtl_validated=a.rtl,whole_glm_inference=False,
        all_timing_targets_met=all(r['timing_target_met'] for r in records) if a.rtl else None,
        source_sha256={str(f):digest(f) for f in files},uarch=ua,rtl_compile_flags=rtlsim.VFLAGS,
        host_seconds=time.perf_counter()-started,lpn_cycles_sum=sum(r['lpn_cycles'] for r in records))
    atomic_json(a.output/'report.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
