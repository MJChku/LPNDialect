"""Validate actual multi-slice BAR/GATHER inputs against ISA, Python/native LPN and RTL."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path

from .e2e import atomic_json,digest
from .workload import execute,use_checkout


def collective_case(slices=2,rows=3,cols=129,stagger=True):
    import numpy as np
    from opentpu import Config,isa as I
    cfg=Config(S=slices,D=128,MCOLS=4,LANES=64,ACT_ROWS=16,DRAM_BYTES=1<<20)
    programs=[];images=[];expected=[];drs=slices*cols+19
    for sid in range(slices):
        srs=cols+7+sid;src=512*sid
        x=(np.arange(rows*srs,dtype=np.float32)/128+sid).reshape(rows,srs)
        memory=np.zeros(cfg.DRAM_BYTES,np.uint8);memory[:x.nbytes]=x.view(np.uint8).ravel()
        program=[I.ld(0,src,rows*srs)]
        delta=5 if stagger and sid==slices-1 else 0
        for _ in range(delta):
            program.append(I.vop(I.V_ADD,src,src,0,rows,cols,srs,srs,0,I.B_SCALAR,1.0))
        expected.append(x[:,:cols]+delta)
        for stage in range(2):
            target=8192+stage*8192
            source=src if stage==0 else 8192+sid*cols
            stride=srs if stage==0 else drs
            program.append(I.gather(source,target,rows,cols,stride,drs,cols))
            for row in range(rows):program.append(I.st(16384+stage*32768+4*row*slices*cols,
                target+row*drs,slices*cols))
            program.append(I.bar())
        program.append(I.halt());programs.append(program);images.append(memory)
    return cfg,programs,images,np.concatenate(expected,axis=1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('opentpu-root','compiler','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rtl',action='store_true');a=p.parse_args()
    root=use_checkout(a.opentpu_root);a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from opentpu import rtlsim
    from .benchmark import write_inputs
    from .compact import CompactOpenTPU
    from .glm47_target import component_timing
    from .native_timing import ReusableTiming
    timing=replace(component_timing(),dma_store_runs=True)
    runner=ReusableTiming(a.compiler,a.output/'native-cache',capacity=64)
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='8'
    rtlsim.VFLAGS+=['-CFLAGS','-O0','-DOTPU_FP_MUL_LAT=2','-DOTPU_FP_ADD_LAT=4']
    uarch=dict(MXU_ROWS=2,MXU_COLS=2,MXU_PAIR_DOT=1,MXU_CL=8,VPU_LANES=64,QST_LANES=64)
    records=[]
    for S,R,C,stagger in ((2,3,129,True),(4,3,129,True),(2,1,1024,False)):
        name=f's{S}-r{R}-c{C}-stagger{int(stagger)}';directory=a.output/name;directory.mkdir()
        cfg,programs,images,expected=collective_case(S,R,C,stagger)
        write_inputs(directory,cfg,programs,images);work=execute(cfg,programs,images)
        for machine in work.machine.slices:
            for stage in range(2):
                at=16384+stage*32768
                actual=machine.dram[at:at+expected.nbytes].view(np.float32).reshape(expected.shape)
                np.testing.assert_array_equal(actual,expected)
        physical=replace(work,config=replace(cfg,MCOLS=2))
        sim=CompactOpenTPU(physical,timing).net.python_simulator();ticks=sim.run(max_time=1000000)
        if sim.place_dicts('program_completed')!=[{}]:raise ValueError('Python collective LPN deadlocked')
        native=runner.prepare(physical,timing,directory/'timing.tables')
        cycles,seconds,_,text=native.run();(directory/'lpn.log').write_text(text)
        if cycles!=ticks:raise ValueError('Python/native collective timing differs')
        record=dict(name=name,slices=S,config=asdict(cfg),lpn_cycles=cycles,lpn_seconds=seconds,
            python_native_equal=True,independent_gather_values_bit_exact=True,rtl_validated=False,
            input_sha256={n:digest(directory/n) for n in ['input.json',*[f'prog_{s}.hex' for s in range(S)],*[f'dram_{s}.bin' for s in range(S)]]})
        if a.rtl:
            drams,tmems,stats=rtlsim.run(cfg,programs,images,uarch=uarch,keep=directory/'rtl',trace=True,max_cycles=1000000)
            (directory/'rtl.log').write_text(stats['trace'])
            for sid,gold in enumerate(work.machine.slices):
                if not np.array_equal(drams[sid],gold.dram) or not np.array_equal(tmems[sid],gold.tmem):
                    raise ValueError('multi-slice complete RTL memory differs from ISA')
            error=cycles/stats['cycles']-1
            record.update(rtl_validated=True,rtl_cycles=stats['cycles'],relative_cycle_error=error,
                all_full_dram_and_tmem_bit_exact=True,timing_target_met=abs(error)<=.1)
        records.append(record);atomic_json(a.output/'progress.json',dict(status='running',records=records))
    project=Path(__file__).parent
    files=[Path(__file__),*[project/n for n in ('collective.py','compact.py','components.py','model.py','workload.py','shape_workload.py','native_timing.py')],
        root/'opentpu/isasim.py',root/'opentpu/rtlsim.py',root/'rtl/top/otpu_coll.sv']
    atomic_json(a.output/'report.json',dict(status='complete',scope=__doc__,records=records,
        timing=asdict(timing),uarch=uarch,rtl_compile_flags=rtlsim.VFLAGS,source_sha256={str(f):digest(f) for f in files},
        whole_glm_inference=False,scheduler_changed=False,
        all_timing_targets_met=all(r.get('timing_target_met',False) for r in records) if a.rtl else None))
    print(json.dumps(records,indent=2),flush=True)


if __name__=='__main__':main()
