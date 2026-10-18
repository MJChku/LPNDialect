"""Check scalar reciprocal fast taps, bulk fallback and mixed VPU retirement."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('opentpu-root','table','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--lookup',type=int,choices=(0,1),default=1)
    p.add_argument('--pipeline',action='store_true')
    p.add_argument('--stall',type=int,choices=(0,1,2),default=0)
    p.add_argument('--write-buffer',type=int,choices=(0,1),default=0)
    p.add_argument('--fp-mul-latency',type=int,choices=(1,2),default=2)
    p.add_argument('--fp-add-latency',type=int,choices=(1,2,3,4),default=4)
    a=p.parse_args();sys.path.insert(0,str(a.opentpu_root.resolve()))
    import numpy as np
    from opentpu import Config,isa as I,rtlsim
    from opentpu.isasim import Machine
    from .rom_inputs import reciprocal_inputs
    table_files,table_hashes,table_args=reciprocal_inputs(a.table)
    cfg=Config(D=64,LANES=64,TMEM_WORDS=32768,DRAM_BYTES=65536)
    rng=np.random.default_rng(41092)
    x=np.ldexp(rng.normal(size=cfg.TMEM_WORDS).astype(np.float32),rng.integers(-100,101,cfg.TMEM_WORDS)).view(np.uint32)
    # All exponent/sign pairs, plus adjacent mantissas and exceptional payloads.
    for j in range(4096):
        frac=(0,1,0x7fffff,0x400000,0x400001,0x123456,0x3fffff,0x400002)[j//512]
        x[j]=((j>>8)&1)<<31|(j&255)<<23|frac
    specs=[]
    for rows in (1,2,8,17,63,128):
        for stride in (1,5):
            specs.append((f'recip_{rows}x1_stride{stride}',I.vop(I.V_RECIP,16384,0,8192,rows,1,3,stride,1)))
    slot=1+a.fp_mul_latency+a.fp_add_latency
    tap=1 if slot>=4 else 2
    saved=(6-tap)*slot+3-(3 if tap>1 else 0)
    shapes=[(1,2),(1,3),(1,4),(1,5),(8,2),(8,3),(8,4),(128,2),
            (8,63),(8,64),(8,65),(1,193),(8,193)]
    for cols in (2,3,4):
        boundary=saved//(cols-1)
        shapes.extend(((boundary,cols),(boundary+1,cols)))
    for rows,cols in dict.fromkeys(shapes):
        specs.append((f'recip_{rows}x{cols}',I.vop(I.V_RECIP,16384,0,8192,rows,cols,cols+3,cols+7,cols+9)))
    specs.append(('recip_in_place',I.vop(I.V_RECIP,0,0,8192,128,1,1,1,1)))
    for fun in (I.V_COPY,I.V_ADD,I.V_MUL,I.V_EXP2,I.V_RSQRT,I.V_RSSQ):
        specs.append((I.VFUNCS[fun]+'_after_recip',I.vop(fun,22000,8192,12288,8,1,3,7,9)))
    if a.pipeline:
        # Sources remain disjoint from all outputs. Repeated destinations test
        # ordered writeback of mixed fast/slow commands, without a RAW pretext.
        specs=[]
        for j in range(48):
            f,cols=((I.V_RECIP,3),(I.V_ADD,1),(I.V_RECIP,193),(I.V_COPY,1),
                    (I.V_RECIP,1),(I.V_EXP2,1),(I.V_RECIP,4),(I.V_MUL,1))[j%8]
            specs.append((f'mixed_{j}',I.vop(f,16384+(j%8)*2048,0,8192,1 if j%3==0 else 8,cols,cols+3,cols+7,cols+9)))
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'tmem.hex').write_text(''.join(f'{int(v):08x}\n' for v in x))
    words=[]
    for _,ins in specs:
        packed=(ins.op<<232)|(ins.flags<<224)
        for k,w in enumerate(ins.w):packed|=int(w)<<(32*(6-k))
        words.append(f'{packed:060x}')
    (a.output/'commands.hex').write_text('\n'.join(words)+'\n')
    sources=[rtlsim.RTL/s for s in ('vpu/otpu_fp.sv','vpu/otpu_fpipe.sv','top/otpu_pkg.sv',
        'vpu/otpu_vtree.sv','vpu/otpu_se_comp.sv','vpu/otpu_se_tail.sv','vpu/otpu_recip_rom.sv','vpu/otpu_vpu.sv')]
    sources.append(Path(__file__).with_name('tb_scalar_recip.sv'))
    files=sources+table_files;hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='12'
    rtlsim.VFLAGS+=['-CFLAGS','-O0',f'-DOTPU_FP_MUL_LAT={a.fp_mul_latency}',f'-DOTPU_FP_ADD_LAT={a.fp_add_latency}']
    exe=rtlsim.build('tb_scalar_recip',sources,dict(LANES=64,LOOKUP=a.lookup,BUFFERED=a.write_buffer))
    run=subprocess.run([str(exe),f'+dir={a.output}',f'+ncmd={len(specs)}',f'+stall={a.stall}',
                        f'+pipeline={int(a.pipeline)}']+table_args,capture_output=True,text=True,timeout=180)
    (a.output/'rtl.log').write_text(run.stdout+run.stderr);run.check_returncode()
    counts=re.search(r'SCALAR_RECIP_GATE cycles=(\d+) completed=(\d+)',run.stdout)
    if not counts or int(counts[2])!=len(specs):raise RuntimeError('missing VPU completions')
    cycles=[int(c) for _,c in re.findall(r'VCMD index=(\d+) cycles=(\d+)',run.stdout)]
    if not a.pipeline and len(cycles)!=len(specs):raise RuntimeError('missing command timing')
    records=[]
    prefixes=[len(specs)-1] if a.pipeline else range(len(specs))
    for index in prefixes:
        machine=Machine(cfg,[[ins for _,ins in specs[:index+1]]+[I.halt()]],[np.zeros(cfg.DRAM_BYTES,np.uint8)])
        machine.slices[0].tmem[:]=x
        expected=machine.run().slices[0].tmem
        actual=np.fromfile(a.output/('tmem_out.bin' if a.pipeline else f'tmem_{index}.bin'),dtype='<u4')
        bad=np.flatnonzero(actual!=expected)
        if len(bad):raise RuntimeError(f'{specs[index][0]}: mismatch at {bad[:16].tolist()}')
        records.append(dict(name='mixed_pipeline' if a.pipeline else specs[index][0],all_tmem_bit_exact=True,
                            rtl_cycles=int(counts[1]) if a.pipeline else cycles[index]))
    if hashes!={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}:raise RuntimeError('sources changed during gate')
    report=dict(status='complete',scope=__doc__,lookup=bool(a.lookup),pipeline=a.pipeline,stall=a.stall,
        write_buffer=a.write_buffer,fp_mul_latency=a.fp_mul_latency,fp_add_latency=a.fp_add_latency,
        accepted_and_completed_commands=len(specs),total_clocks=int(counts[1]),records=records,
        source_and_table_sha256=hashes,simulator=str(exe),simulator_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
        lookup_table_sha256=table_hashes)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
