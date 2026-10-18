"""Check two-command prefetch, uneven engine groups, wraparound and RMAX/WAW ordering."""
import argparse
from pathlib import Path
import sys,json,time,re


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--opentpu-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rows',type=int,default=32)
    ap.add_argument('--columns',type=int,choices=(1,2,4,8),default=1)
    a=ap.parse_args()
    sys.path.insert(0,str(a.opentpu_root.resolve()))
    import numpy as np
    from opentpu import Config,isa as I,rtlsim
    from opentpu.isasim import Machine
    # Numerical/cycle gates need no optimizing host compiler; long benchmarks
    # separately use -O3. Verilator's SV optimization and all assertions stay on.
    from examples.opentpu.rtl_evolution.check_matrix import main as primitive_gates
    sys.argv=[sys.argv[0],'--opentpu-root',str(a.opentpu_root),'--output',str(a.output),
              '--rows',str(a.rows),'--columns',str(a.columns)]
    primitive_gates()
    report=json.loads((a.output/'report.json').read_text())
    report['status']='running'
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    cfg=Config(D=64,MCOLS=4,ACT_ROWS=16,LANES=64)
    ua=dict(MXU_ROWS=a.rows,VPU_LANES=64,QST_LANES=64)
    if a.columns!=1:ua['MXU_COLS']=a.columns
    from examples.opentpu.e2e import digest
    report['pipeline_script_sha256']=digest(Path(__file__))
    for overlap in (False,True):
        rng=np.random.default_rng(713+overlap)
        kb=3;k=kb*cfg.D;maxn=67;m=8
        image=np.zeros(cfg.DRAM_BYTES,np.uint8)
        x=rng.normal(size=(m,k)).astype(np.float32)
        image[:x.nbytes]=x.view(np.uint8).ravel()
        data=rng.integers(-127,128,(maxn,k),dtype=np.int8).view(np.uint8)
        image[65536:65536+data.size]=data.ravel()
        alpha=rng.uniform(.8,1.2,m).astype(np.float32)
        image[0x48000:0x48000+alpha.nbytes]=alpha.view(np.uint8)
        old=rng.normal(size=8192).astype(np.float32)
        image[0x50000:0x50000+old.nbytes]=old.view(np.uint8)
        program=[I.ld(0,0,x.size),I.qact(0,m,0,kb,k),
                 I.ld(0x48000,8192,m),I.ld(0x50000,10000,8192)]
        shapes=list(zip((5,67,1,35,64,3),(3,8,1,2,4,3)))
        for j,(n,rows) in enumerate(shapes):
            out=10000 if overlap else 10000+1024*j
            program.append(I.mm(65536,0,out,n,kb,k,n+3,rows,0,4,
                unit=True,acc=True,ascale=8192 if rows<=cfg.MCOLS else None,rmax=j%2==0))
        program.append(I.halt())
        golden=Machine(cfg,[program],[image.copy()]).run().slices[0]
        name='pipeline_waw' if overlap else 'pipeline_independent'
        before=time.perf_counter()
        drams,tmems,stats=rtlsim.run(cfg,[program],[image],uarch=ua,
            keep=a.output/name,max_cycles=10000,trace=True)
        (a.output/name/'rtl.log').write_text(stats['trace'])
        starts=[int(x) for x in re.findall(r'T0 S c=(\d+) s=\d+ u=1 ',stats['trace'])]
        ends=[int(x) for x in re.findall(r'T0 U c=(\d+) u=1 ',stats['trace'])]
        overlaps=sum(starts[i+1]<ends[i] for i in range(min(len(starts)-1,len(ends))))
        if len(starts)!=len(shapes) or len(ends)!=len(shapes) or overlaps==0:
            raise RuntimeError(f'{name}: two-command admission did not overlap: {starts}, {ends}')
        if not np.array_equal(drams[0],golden.dram) or not np.array_equal(tmems[0],golden.tmem):
            bad=np.flatnonzero(tmems[0]!=golden.tmem)
            raise RuntimeError(f'{name}: numerical mismatch at {bad[:16].tolist()}')
        result=dict(case=dict(name=name,commands=len(shapes),shapes=shapes),
                    rtl_cycles=stats['cycles'],seconds=time.perf_counter()-before,
                    all_dram_and_tmem_bit_exact=True,overlapping_command_admissions=overlaps,
                    matrix_start_cycles=starts,matrix_end_cycles=ends)
        report['records'].append(result)
        (a.output/'report.json').write_text(json.dumps(dict(report,status='running'),indent=2)+'\n')
        print(json.dumps(result),flush=True)
    if report['pipeline_script_sha256']!=digest(Path(__file__)):raise RuntimeError('pipeline gate changed')
    report['status']='complete'
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    try:main()
    except Exception as error:
        output=Path(sys.argv[sys.argv.index('--output')+1])
        path=output/'report.json'
        if path.exists():
            report=json.loads(path.read_text())
            report.update(status='failed',error_type=type(error).__name__,error=str(error))
            path.write_text(json.dumps(report,indent=2)+'\n')
        raise
