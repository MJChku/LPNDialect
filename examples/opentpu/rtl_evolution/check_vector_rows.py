"""Bit-exact gates for bounded interleaved-row reductions in the actual VPU."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opentpu-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--row-contexts',type=int,choices=(1,4),default=4)
    p.add_argument('--stall',type=int,choices=(0,1),default=0)
    p.add_argument('--write-buffer',type=int,choices=(0,1),default=0)
    p.add_argument('--fp-mul-latency',type=int,choices=(1,2),default=2)
    p.add_argument('--fp-add-latency',type=int,choices=(1,2,3,4),default=4)
    a=p.parse_args();sys.path.insert(0,str(a.opentpu_root.resolve()))
    import numpy as np
    from opentpu import Config,isa as I,rtlsim
    from opentpu.isasim import Machine
    cfg=Config(D=64,LANES=64,TMEM_WORDS=32768,DRAM_BYTES=65536)
    rng=np.random.default_rng(4721)
    x=np.ldexp(rng.normal(size=32768).astype(np.float32),rng.integers(-18,19,32768))
    # Cancellation-sensitive values, denormals, signed zeros and nonfinite inputs.
    x[13:20]=np.array([0x80000000,1,0x007fffff,0x7f800000,0xff800000,0x7fc00001,0],np.uint32).view(np.float32)
    specs=[]
    for rows,cols in ((1,193),(2,193),(3,193),(4,193),(5,193),(8,193),(9,193),
                      (8,1),(8,63),(8,64),(8,65),(8,576),(9,641)):
        for func in (I.V_RSUM,I.V_RSSQ,I.V_RDOT):
            specs.append((f'{I.VFUNCS[func]}_{rows}x{cols}',
                          I.vop(func,16384,0,8192,rows,cols,3,cols+7,cols+9)))
    for rows in (3,4,5,8):
        for mode in (I.B_ROW,I.B_COL,I.B_SCALAR):
            specs.append((f'rdot_mode{mode}_r{rows}',
                          I.vop(I.V_RDOT,16384,0,8192,rows,193,1,200,202,bmode=mode,imm=1.0)))
    # Aliased outputs keep original row-major issue; each output aliases only
    # its own row's first input, so the ISA and original RTL agree as well.
    specs.append(('alias_a',I.vop(I.V_RSSQ,0,0,8192,8,193,200,200,202)))
    specs.append(('alias_b',I.vop(I.V_RDOT,8192,0,8192,8,193,202,200,202)))
    # Ordinary VOPs after a sum prove retirement and reused datapath state.
    specs.append(('mul_after_sums',I.vop(I.V_MUL,20000,0,8192,8,193,200,200,202)))
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'tmem.hex').write_text('\n'.join(f'{int(v):08x}' for v in x.view(np.uint32))+'\n')
    words=[]
    for _,ins in specs:
        packed=(ins.op<<232)|(ins.flags<<224)
        for k,w in enumerate(ins.w):packed|=int(w)<<(32*(6-k))
        words.append(f'{packed:060x}')
    (a.output/'commands.hex').write_text('\n'.join(words)+'\n')
    sources=[rtlsim.RTL/s for s in ('vpu/otpu_fp.sv','vpu/otpu_fpipe.sv','top/otpu_pkg.sv',
                                  'vpu/otpu_vtree.sv','vpu/otpu_se_comp.sv','vpu/otpu_se_tail.sv','vpu/otpu_vpu.sv')]
    sources.append(Path(__file__).with_name('tb_vpu_rows.sv'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='12'
    rtlsim.VFLAGS += [f'-DOTPU_FP_MUL_LAT={a.fp_mul_latency}',f'-DOTPU_FP_ADD_LAT={a.fp_add_latency}']
    exe=rtlsim.build('tb_vpu_rows',sources,dict(LANES=64,ROWS=a.row_contexts,BUFFERED=a.write_buffer))
    run=subprocess.run([str(exe),f'+dir={a.output}',f'+ncmd={len(specs)}',f'+stall={a.stall}'],
                       capture_output=True,text=True,timeout=120)
    (a.output/'rtl.log').write_text(run.stdout+run.stderr);run.check_returncode()
    cycles=[int(c) for _,c in re.findall(r'VCMD index=(\d+) cycles=(\d+)',run.stdout)]
    if len(cycles)!=len(specs):raise RuntimeError('incomplete RTL command measurements')
    records=[]
    for index,(name,_) in enumerate(specs):
        m=Machine(cfg,[[ins for _,ins in specs[:index+1]]+[I.halt()]],[np.zeros(cfg.DRAM_BYTES,np.uint8)])
        m.slices[0].tmem[:]=x.view(np.uint32);golden=m.run().slices[0].tmem
        got=np.fromfile(a.output/f'tmem_{index}.bin',dtype='<u4')
        bad=np.flatnonzero(got!=golden)
        if len(bad):raise RuntimeError(f'{name}: TMEM differs at {bad[:8].tolist()}, actual {got[bad[:8]]}, expected {golden[bad[:8]]}')
        records.append(dict(name=name,rtl_cycles=cycles[index],all_tmem_bit_exact=True))
    if hashes!={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
        raise RuntimeError('RTL sources changed during the gate')
    report=dict(status='complete',records=records,row_contexts=a.row_contexts,stall=a.stall,write_buffer=a.write_buffer,
                fp_mul_latency=a.fp_mul_latency,fp_add_latency=a.fp_add_latency,
                simulator=str(exe),source_sha256=hashes)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
