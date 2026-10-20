"""Import NVFP4 E2M1/E4M3 tensors and verify actual packed MXU full numerical state."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('opentpu-root','compiler','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    from examples.opentpu.workload import use_checkout
    from examples.opentpu.e2e import atomic_json,digest
    from examples.opentpu.shape_workload import resolve_static
    from examples.opentpu.model import Timing
    from examples.opentpu.native_timing import ReusableTiming
    from .parallel_inference import simulator_cache_key
    use_checkout(a.opentpu_root)
    import numpy as np
    from opentpu import Config,isa as I,quant as Q,rtlsim
    from opentpu.isasim import Machine
    cfg=Config(D=128,MCOLS=4,ACT_ROWS=16,LANES=64)
    ua=dict(MXU_ROWS=2,MXU_COLS=2,MXU_PAIR_DOT=1,VPU_LANES=64,QST_LANES=64)
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='8'
    rtlsim.VFLAGS+=['-CFLAGS','-O0','-DOTPU_FP_MUL_LAT=2','-DOTPU_FP_ADD_LAT=4']
    files=[rtlsim.RTL/name for name in rtlsim.RTL_SOURCES]+[rtlsim.TB/'tb_top.sv',rtlsim.TB/'lpn_wall.cpp',Path(__file__)]
    hashes={str(f):digest(f) for f in files};records=[];start=time.perf_counter()
    def save(status):
        atomic_json(a.output/'report.json',dict(status=status,scope=__doc__,config=vars(cfg),uarch=ua,
            source_sha256=hashes,rtl_compile_flags=rtlsim.VFLAGS,records=records,
            seconds=time.perf_counter()-start,weight_format='NVFP4 E2M1, E4M3 per16 and FP32 tensor scale',
            full_glm_inference=False,dsp58_or_fpga_fit_proven=False))
    save('running')
    for index,(name,n,m,kb,pair,acc) in enumerate([
        ('nvfp4_scaled',5,3,3,False,False),('nvfp4_replay_acc',9,8,5,False,True),
        ('nvfp4_pair',7,2,3,True,False)]):
        rng=np.random.default_rng(1447+index);k=kb*cfg.D
        codes=rng.integers(0,16,(n,k),dtype=np.uint8)
        codes[0,:16]=np.arange(16,dtype=np.uint8)
        scales=rng.integers(32,104,(n,k//16),dtype=np.uint8)
        scales[0,0]=0
        rows,srows=Q.import_nvfp4(codes,scales,.125,D=128)
        x=rng.normal(size=(m,k)).astype(np.float32)
        image=np.zeros(cfg.DRAM_BYTES,np.uint8);image[:x.nbytes]=x.view(np.uint8).ravel()
        weight_base,scale_base=65536,262144;pitch=rows.shape[1]
        image[weight_base:weight_base+rows.nbytes]=rows.ravel()
        image[scale_base:scale_base+srows.nbytes]=srows.ravel()
        out,ors=16384,n+3
        program=[I.ld(0,0,x.size),I.qact(0,m,0,kb,k,dup=pair)]
        if acc:
            old=rng.normal(size=(m,ors)).astype(np.float32)
            image[32768:32768+old.nbytes]=old.view(np.uint8).ravel()
            program.append(I.ld(32768,out,old.size))
        program+=[I.mm(weight_base,scale_base,out,n,kb,pitch,ors,m,0,srows.shape[1],wf=I.WNV,pair=pair,acc=acc),I.halt()]
        expected=Machine(cfg,[program],[image.copy()]).run().slices[0]
        directory=a.output/name
        dram,tmem,stats=rtlsim.run(cfg,[program],[image],keep=directory,uarch=ua,max_cycles=500000,trace=True)
        (directory/'rtl.log').write_text(stats['trace'])
        if not np.array_equal(dram[0],expected.dram) or not np.array_equal(tmem[0],expected.tmem):
            raise ValueError('NVFP4 full numerical state differs: '+name)
        raw=re.search(r'RESULT cycles=(\d+) halted=1 error=0',stats['trace'])
        if not raw or int(raw[1])!=stats['cycles']:raise ValueError('raw completion differs')
        work=resolve_static(cfg,[program]);work=replace(work,config=replace(cfg,MCOLS=2))
        timing=Timing(batch_rows=2,matrix_row_lanes=2,fifo_chunks=256,scale_lanes=2,
            dram_bytes_per_cycle=2816,dram_port_bytes=2816,unified_dram=True,scalar_store_bytes=256,
            quant_row_banks=8,quant_store_buffered=True,vector_row_contexts=4,
            vector_command_queue=2,matrix_retirement=True,physical_scale_lines=True)
        native=ReusableTiming(a.compiler,a.output/'native-cache',capacity=128).prepare(work,timing,directory/'timing.tables')
        cycles,_,_,log=native.run();(directory/'lpn.log').write_text(log)
        exe=rtlsim.build_top(cfg,uarch=ua,dram_lat=8)
        if exe.parent.name!=simulator_cache_key(rtlsim,cfg,ua):raise ValueError('simulator identity differs')
        records.append(dict(name=name,n=n,m=m,kb=kb,pair=pair,acc=acc,
            rtl_cycles=stats['cycles'],lpn_cycles=cycles,relative_cycle_error=cycles/stats['cycles']-1,
            all_full_dram_and_tmem_bit_exact=True,zero_rollbacks=True,
            simulator=str(exe),simulator_sha256=digest(exe),
            input_sha256={f:digest(directory/f) for f in ('prog_0.hex','dram_0.bin')},
            output_sha256={f:digest(directory/f) for f in ('dram_out_0.bin','tmem_0.hex')},
            raw_log_sha256=digest(directory/'rtl.log')))
        save('running')
    if hashes!={str(f):digest(f) for f in files}:raise ValueError('gate sources changed')
    save('complete')


if __name__=='__main__':main()
