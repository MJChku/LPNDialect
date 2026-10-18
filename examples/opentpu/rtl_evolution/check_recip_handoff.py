"""Component gate: LD -> QACT -> MM -> reciprocal -> QACT -> MM -> ST.

Identical instructions/images are run in a reciprocal feature-off control and
feature-on RTL, against independent ISA state. This is not inference timing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('opentpu-root','table','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.opentpu_root.resolve()))
    import numpy as np
    from opentpu import Config,isa as I,rtlsim
    from opentpu.isasim import Machine
    from .rom_inputs import reciprocal_inputs
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    _,tables,extra=reciprocal_inputs(a.table)
    cfg=Config(D=64,MCOLS=4,ACT_ROWS=16,LANES=64)
    rng=np.random.default_rng(4389);m,k,n=8,192,11
    image=np.zeros(cfg.DRAM_BYTES,np.uint8)
    x=rng.uniform(.25,2,(m,k)).astype(np.float32)
    image[:x.nbytes]=x.view(np.uint8).ravel()
    weights=rng.integers(1,64,(n,k),dtype=np.int8)
    image[65536:65536+weights.size]=weights.view(np.uint8).ravel()
    program=[I.ld(0,0,x.size),I.qact(0,m,0,3,k),
        I.mm(65536,0,16384,n,3,k,n+3,m,0,4,unit=True),
        I.vop(I.V_RECIP,28000,16384,0,1,3,3,3,1),
        I.vop(I.V_MUL,28128,28000,28000,1,3,3,3,3),
        I.vop(I.V_RECIP,30000,16384,0,1,192,192,192,1),
        I.qact(28128,1,8,1,64),I.mm(65536,0,32000,3,1,k,4,1,8,4,unit=True),
        I.st(524288,32000,3),I.halt()]
    expected=Machine(cfg,[program],[image.copy()]).run().slices[0]
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='12'
    rtlsim.VFLAGS+=['-CFLAGS','-O0','-DOTPU_FP_MUL_LAT=2','-DOTPU_FP_ADD_LAT=4']
    sources=[rtlsim.RTL/s for s in rtlsim.RTL_SOURCES]+[rtlsim.TB/'tb_top.sv']
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    records=[]
    for enabled in (False,True):
        ua=dict(MXU_ROWS=4,VPU_LANES=64,QST_LANES=64,ACT_STREAMING=1,QUANT_LOOKUP=1,
                VPU_SCALAR_RECIP=int(enabled))
        before=time.perf_counter()
        dram,tmem,stats=rtlsim.run(cfg,[program],[image.copy()],uarch=ua,
            keep=a.output/str(int(enabled)),max_cycles=100000,trace=True,plusargs=extra)
        if not np.array_equal(dram[0],expected.dram) or not np.array_equal(tmem[0],expected.tmem):
            raise RuntimeError('component chain differs from independent ISA')
        (a.output/str(int(enabled))/'rtl.log').write_text(stats['trace'])
        records.append(dict(reciprocal_lookup=enabled,rtl_cycles=stats['cycles'],
                            all_dram_and_tmem_bit_exact=True,seconds=time.perf_counter()-before))
    if hashes!={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}:
        raise RuntimeError('RTL sources changed during handoff gate')
    # Canonical instruction and DRAM files must also be identical between runs.
    for name in ('prog_0.hex','dram_0.bin'):
        if (a.output/'0'/name).read_bytes()!=(a.output/'1'/name).read_bytes():raise RuntimeError('input files differ')
    result=dict(status='complete',scope=__doc__,records=records,source_sha256=hashes,
                lookup_table_sha256=tables,identical_input_files=True,
                cycle_speedup=records[0]['rtl_cycles']/records[1]['rtl_cycles'])
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
