"""Compare packed/unpacked QST under shared-link pressure with identical data.

This is a synthetic hardware gate, not an inference workload result.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('opentpu-root','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from examples.opentpu.workload import use_checkout
    from examples.opentpu.e2e import atomic_json,digest
    use_checkout(a.opentpu_root.resolve())
    import numpy as np
    from opentpu import Config,isa as I,rtlsim
    from opentpu.isasim import Machine
    cfg=Config(D=64,MCOLS=4,ACT_ROWS=16,LANES=64,DRAM_BYTES=4<<20)
    rng=np.random.default_rng(501);n,kb=2049,9;k=kb*cfg.D
    image=np.zeros(cfg.DRAM_BYTES,np.uint8);x=rng.normal(size=k).astype(np.float32)
    image[:x.nbytes]=x.view(np.uint8);weights=rng.integers(-127,128,(n,k),dtype=np.int8).view(np.uint8)
    image[1<<20:(1<<20)+weights.size]=weights.ravel()
    scales=np.zeros((n,16),np.uint8);scales[:,:4]=np.array([.01],np.float32).view(np.uint8)
    image[3<<20:(3<<20)+scales.size]=scales.ravel()
    program=[I.ld(0,0,k),I.qact(0,1,0,kb,k),I.mm(1<<20,3<<20,16384,n,kb,k,n,1,0,16),
             I.qst(0,7<<19,15<<18,128,kb,0,k,1),I.halt()]
    golden=Machine(cfg,[program],[image.copy()]).run().slices[0]
    rtlsim.VFLAGS+=['-CFLAGS','-O0','-DOTPU_FP_MUL_LAT=2','-DOTPU_FP_ADD_LAT=4']
    files=[rtlsim.RTL/n for n in rtlsim.RTL_SOURCES]+[rtlsim.TB/'tb_top.sv',rtlsim.TB/'lpn_wall.cpp',Path(__file__)]
    hashes={str(f):digest(f) for f in files};records=[]
    for packed in (0,1):
        ua=dict(MXU_ROWS=8,VPU_LANES=64,QST_LANES=64,MEM_BYTES=256,QST_PACKED_WORDS=packed)
        directory=a.output/f'packed{packed}'
        drams,tmems,stats=rtlsim.run(cfg,[program],[image],uarch=ua,max_cycles=100000,keep=directory,trace=True)
        if not np.array_equal(drams[0],golden.dram) or not np.array_equal(tmems[0],golden.tmem):
            raise RuntimeError('QST pressure gate differs from full independent ISA state')
        (directory/'rtl.log').write_text(stats['trace'])
        exe=rtlsim.build_top(cfg,uarch=ua,dram_lat=8)
        records.append(dict(packed=bool(packed),rtl_cycles=stats['cycles'],uarch=ua,
            input_sha256={n:digest(directory/n) for n in ['dram_0.bin','prog_0.hex']},
            all_full_dram_and_tmem_bit_exact=True,simulator=str(exe),simulator_sha256=digest(exe)))
        atomic_json(a.output/'report.json',dict(status='running',records=records,source_sha256=hashes))
    if records[0]['input_sha256']!=records[1]['input_sha256']:raise RuntimeError('pressure input differs')
    if hashes!={str(f):digest(f) for f in files}:raise RuntimeError('gate sources changed')
    result=dict(status='complete',scope=__doc__,records=records,source_sha256=hashes,
        cycle_speedup=records[0]['rtl_cycles']/records[1]['rtl_cycles'],identical_input_files=True)
    atomic_json(a.output/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
