"""Check ACT streaming against independent ISA and an unchanged-scoreboard control.

The program and input image are identical for each control/candidate pair. This
is a numerical/dependency component gate, not an end-to-end inference benchmark.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opentpu-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.opentpu_root.resolve()))
    import numpy as np
    from opentpu import Config, isa as I, rtlsim
    from opentpu.isasim import Machine
    a.output.mkdir(parents=True, exist_ok=False)
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1] = '12'
    rtlsim.VFLAGS += ['-CFLAGS', '-O0']
    cfg = Config(D=64, MCOLS=4, ACT_ROWS=16, LANES=64)
    sources = [rtlsim.RTL / s for s in rtlsim.RTL_SOURCES]
    sources += [rtlsim.ROOT / 'sim/verilator/tb_top.sv']
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    records = []
    specs = [(f'row{m}_kb{kb}',m,kb,False) for m in (1,8) for kb in (1,9,33)]
    specs += [(name,8,9,False) for name in (
        'output_war','younger_overwrite','partial_blocks','producer_rewrite','delayed_acceptance')]
    specs += [('duplicated_rows',2,9,True),('unbuffered_fallback',8,9,False)]
    for seed,(name,m,kb,dup) in enumerate(specs):
        rng = np.random.default_rng(1740+seed)
        k=kb*64; n=9
        image=np.zeros(cfg.DRAM_BYTES,np.uint8)
        inputk=(kb+3)*64 if name=='partial_blocks' else k
        x=rng.normal(size=(m,inputk)).astype(np.float32)
        if name in ('partial_blocks','producer_rewrite'): x.fill(1)
        image[:x.nbytes]=x.view(np.uint8).ravel()
        # A second producer has different input values, not just a repeated write.
        x2=rng.normal(size=(m,inputk)).astype(np.float32)
        if name in ('partial_blocks','producer_rewrite'): x2.fill(-1)
        image[32768:32768+x2.nbytes]=x2.view(np.uint8).ravel()
        ab=3 if name=='partial_blocks' else 0
        mkb=kb+ab if name=='partial_blocks' else kb
        weights=rng.integers(-127,128,(n,mkb*64),dtype=np.int8).view(np.uint8)
        image[131072:131072+weights.size]=weights.ravel()
        out=0 if name=='output_war' else 20000
        mmrows=2*m if dup else m
        prog=[I.ld(0,0,x.size),I.ld(32768,8192,x2.size)]
        if name=='partial_blocks': prog += [I.qact(0,m,0,mkb,inputk)]
        prog += [I.qact(8192 if name=='partial_blocks' else 0,m,ab,kb,inputk,dup=dup)]
        if name=='producer_rewrite': prog += [I.qact(8192,m,ab+2,kb-2,k)]
        prog += [I.mm(131072,0,out,n,mkb,mkb*64,n+3,mmrows,0,4,unit=True)]
        if name=='younger_overwrite':
            prog += [I.qact(8192,m,0,kb,k),I.mm(131072,0,20000,n,kb,k,n+3,m,0,4,unit=True)]
        prog += [I.halt()]
        golden=Machine(cfg,[prog],[image.copy()]).run().slices[0]
        results=[]
        for enabled in (False,True):
            ua=dict(MXU_ROWS=4,VPU_LANES=64,QST_LANES=64,ACT_STREAMING=int(enabled))
            if name=='unbuffered_fallback': ua['QUANT_XD']=4
            extra=['+tm_sdly=30'] if name=='delayed_acceptance' else []
            before=time.perf_counter()
            drams,tmems,stats=rtlsim.run(cfg,[prog],[image.copy()],uarch=ua,
                keep=a.output/name/('stream' if enabled else 'control'),trace=True,
                max_cycles=100000,plusargs=extra)
            if not np.array_equal(drams[0],golden.dram) or not np.array_equal(tmems[0],golden.tmem):
                bad=np.flatnonzero(tmems[0]!=golden.tmem)
                raise RuntimeError(f'{name} streaming={enabled}: mismatch at {bad[:16].tolist()}')
            trace=stats['trace']
            (a.output/name/('stream' if enabled else 'control')/'rtl.log').write_text(trace)
            # For the alias case the MM must retain its whole-command WAR edge.
            qdone=[int(c) for c in re.findall(r'T0 U c=(\d+) u=2 ',trace)]
            release=[int(c) for c in re.findall(r'T0 G c=(\d+) ',trace)]
            if not qdone or not release:
                raise RuntimeError('missing quantizer completion or matrix release trace')
            if name=='output_war' and release[0]<qdone[0]:
                raise RuntimeError('MM released before its overlapping QACT reader completed')
            result=dict(streaming=enabled,cycles=stats['cycles'],seconds=time.perf_counter()-before,
                        quant_done=qdone,matrix_release=release,all_state_bit_exact=True)
            if name!='delayed_acceptance':
                from examples.opentpu.model import Timing
                from examples.opentpu.compact import CompactOpenTPU
                from examples.opentpu.shape_workload import resolve_static
                timing=Timing(batch_rows=4,matrix_row_lanes=4,fifo_chunks=4*128,
                    dram_bytes_per_cycle=2816,dram_port_bytes=2816,unified_dram=True,
                    scale_lanes=4,scalar_store_bytes=256,epilogue_lanes=1,
                    quant_row_banks=8,quant_store_buffered=True,vector_row_contexts=4,
                    quant_buffer_chunks=ua.get('QUANT_XD',1024),activation_streaming=enabled)
                workload=resolve_static(replace(cfg,MCOLS=1),[prog])
                model=CompactOpenTPU(workload,timing)
                sim=model.net.python_simulator()
                prediction=sim.run(max_time=100000)
                if sim.place_dicts('program_completed')!=[{}]:raise RuntimeError('LPN deadlocked')
                result.update(lpn_cycles=prediction,lpn_error_percent=100*(prediction/stats['cycles']-1))
            results.append(result)
        records.append(dict(name=name,rows=mmrows,kb=mkb,results=results,
                            cycle_speedup=results[0]['cycles']/results[1]['cycles']))
        report=dict(status='running',scope=__doc__,config=cfg.__dict__,source_sha256=hashes,records=records)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(records[-1]),flush=True)
    if hashes!={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}:
        raise RuntimeError('RTL sources changed during validation')
    report['status']='complete'
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
