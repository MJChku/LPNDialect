"""Check scale-cache coherence through real DMA and quantizer writes."""
import argparse
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('opentpu-root','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    from examples.opentpu.workload import use_checkout
    from examples.opentpu.e2e import atomic_json,digest
    use_checkout(a.opentpu_root.resolve())
    import numpy as np
    from opentpu import Config,isa as I,rtlsim
    from opentpu.isasim import Machine
    cfg=Config(D=64,MCOLS=4,ACT_ROWS=16,LANES=64)
    ua=dict(MXU_ROWS=8,VPU_LANES=64,QST_LANES=64,SCALE_CACHE_ENTRIES=16)
    rtlsim.VFLAGS+=['-CFLAGS','-O0','-DOTPU_FP_MUL_LAT=2','-DOTPU_FP_ADD_LAT=4']
    files=[rtlsim.RTL/n for n in rtlsim.RTL_SOURCES]+[rtlsim.TB/'tb_top.sv',rtlsim.TB/'lpn_wall.cpp',Path(__file__)]
    hashes={str(f):digest(f) for f in files};records=[]
    for name in ('dma_scale_overwrite','quantizer_partial_scale_overwrite','disjoint_write_with_reads'):
        rng=np.random.default_rng(458);n,m,kb=19,3,5;k=kb*cfg.D;srs=16;sbase=1<<18;wbase=1<<16
        image=np.zeros(cfg.DRAM_BYTES,np.uint8)
        x=rng.normal(size=(m,k)).astype(np.float32);image[:x.nbytes]=x.view(np.uint8).ravel()
        weights=rng.integers(-127,128,(n,k),dtype=np.int8).view(np.uint8)
        image[wbase:wbase+weights.size]=weights.ravel()
        scales=np.zeros((n,srs),np.uint8);scales[:,:4]=np.array([.01],dtype=np.float32).view(np.uint8)
        image[sbase:sbase+scales.size]=scales.ravel()
        changed=scales.copy();changed[:,:4]=np.array([.02],dtype=np.float32).view(np.uint8)
        image[32768:32768+changed.size]=changed.ravel()
        program=[I.ld(0,0,x.size),I.qact(0,m,0,kb,k),I.ld(32768,4096,changed.size//4)]
        matrix=lambda out:I.mm(wbase,sbase,out,n,kb,k,n+3,m,0,srs)
        program += [matrix(16384),matrix(17408)]
        if name=='dma_scale_overwrite':program += [I.st(sbase,4096,changed.size//4)]
        elif name=='quantizer_partial_scale_overwrite':
            program += [I.qst(8000,sbase+1,3<<18,1,1,cfg.D,cfg.D,1)]
        else:program += [I.st(700000,8000,128)]
        program += [matrix(18432),I.halt()]
        golden=Machine(cfg,[program],[image.copy()]).run().slices[0]
        drams,tmems,stats=rtlsim.run(cfg,[program],[image],uarch=ua,max_cycles=50000,
            trace=True,keep=a.output/name)
        if not np.array_equal(drams[0],golden.dram) or not np.array_equal(tmems[0],golden.tmem):
            raise RuntimeError(f'{name}: cached RTL differs from full independent ISA state')
        different=not np.array_equal(golden.tmem[16384:16384+n],golden.tmem[18432:18432+n])
        if different!=(name!='disjoint_write_with_reads'):raise RuntimeError('write gate did not affect expected outputs')
        log=stats['trace'];(a.output/name/'rtl.log').write_text(log)
        counters=re.findall(r'SCACHE s=0 p=(\d+) req=(\d+) miss=(\d+)',log)
        if len(counters)!=8 or sum(int(q)-int(z) for _,q,z in counters)<=0:raise RuntimeError('cache hits were not exercised')
        records.append(dict(case=name,rtl_cycles=stats['cycles'],all_dram_and_tmem_bit_exact=True,
            overwrite_changes_output=different,cache_requests=sum(int(q) for _,q,_ in counters),
            cache_misses=sum(int(z) for _,_,z in counters)))
        atomic_json(a.output/'report.json',dict(status='running',records=records,source_sha256=hashes))
    if hashes!={str(f):digest(f) for f in files}:raise RuntimeError('gate sources changed')
    exe=rtlsim.build_top(cfg,uarch=ua,dram_lat=8)
    report=dict(status='complete',scope=__doc__,config=cfg.__dict__,uarch=ua,records=records,
        source_sha256=hashes,simulator=str(exe),simulator_sha256=digest(exe))
    atomic_json(a.output/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
