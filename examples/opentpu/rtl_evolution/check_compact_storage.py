"""Check the smaller physical ACT/FIFO allocation at its highest used addresses."""
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
    # The logical ISA configuration and inputs retain the original capacities.
    # Only actual RTL storage is reduced by top-level hardware parameters.
    cfg=Config(D=64,MCOLS=4,ACT_ROWS=16,ACT_BLOCKS=64,LANES=64,DRAM_BYTES=4<<20)
    ua=dict(MXU_ROWS=8,VPU_LANES=64,QST_LANES=64,ACT_ROWS=8,ACT_BLOCKS=64,FIFO_DEPTH=64)
    rtlsim.VFLAGS+=['-CFLAGS','-O0','-DOTPU_FP_MUL_LAT=2','-DOTPU_FP_ADD_LAT=2']
    files=[rtlsim.RTL/n for n in rtlsim.RTL_SOURCES]+[rtlsim.TB/'tb_top.sv',rtlsim.TB/'lpn_wall.cpp',Path(__file__)]
    hashes={str(f):digest(f) for f in files};records=[]
    for name in ('whole_activation','partial_high_blocks'):
        rng=np.random.default_rng(610);n,m,kb,ab=19,8,9,55
        x=rng.normal(size=(m,64*cfg.D)).astype(np.float32)
        image=np.zeros(cfg.DRAM_BYTES,np.uint8);image[:x.nbytes]=x.view(np.uint8).ravel()
        weights=rng.integers(-127,128,(n,kb*cfg.D),dtype=np.int8).view(np.uint8)
        image[1<<20:(1<<20)+weights.size]=weights.ravel()
        scales=np.zeros((n,16),np.uint8);scales[:,:4]=np.array([.01],np.float32).view(np.uint8)
        image[2<<20:(2<<20)+scales.size]=scales.ravel()
        load=I.qact(0,m,0,64,64*cfg.D) if name=='whole_activation' else I.qact(ab*cfg.D,m,ab,kb,64*cfg.D)
        program=[I.ld(0,0,x.size),load,
                 I.mm(1<<20,2<<20,40000,n,kb,kb*cfg.D,n,m,ab,16),
                 I.mm(1<<20,2<<20,41000,n,kb,kb*cfg.D,n,m,ab,16),I.halt()]
        golden=Machine(cfg,[program],[image.copy()]).run().slices[0]
        directory=a.output/name
        drams,tmems,stats=rtlsim.run(cfg,[program],[image],uarch=ua,max_cycles=100000,
                                  keep=directory,trace=True)
        if not np.array_equal(drams[0],golden.dram) or not np.array_equal(tmems[0],golden.tmem):
            raise RuntimeError('compact physical storage differs from full ISA state')
        (directory/'rtl.log').write_text(stats['trace'])
        records.append(dict(case=name,rtl_cycles=stats['cycles'],physical_last_activation_row=7,
            physical_last_activation_block=63,all_full_dram_and_tmem_bit_exact=True))
        atomic_json(a.output/'report.json',dict(status='running',records=records,source_sha256=hashes))
    if hashes!={str(f):digest(f) for f in files}:raise RuntimeError('gate sources changed')
    exe=rtlsim.build_top(cfg,uarch=ua,dram_lat=8)
    result=dict(status='complete',scope=__doc__,config=cfg.__dict__,uarch=ua,records=records,
        simulator=str(exe),simulator_sha256=digest(exe),source_sha256=hashes,
        limitations=['Boundary numerical gates, not a complete inference replay or FPGA synthesis'])
    atomic_json(a.output/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
