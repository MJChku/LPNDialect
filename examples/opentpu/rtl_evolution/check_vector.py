"""Bit-exact wide-vector and ordered-reduction RTL gates."""
import argparse
import json
from pathlib import Path
import subprocess
import re
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opentpu-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fp-mul-latency',type=int,choices=(1,2),default=2)
    p.add_argument('--fp-add-latency',type=int,choices=(1,2,3,4),default=4)
    a=p.parse_args();sys.path.insert(0,str(a.opentpu_root.resolve()))
    import numpy as np
    from opentpu import Config,isa as I,rtlsim
    from opentpu.isasim import Machine
    a.output.mkdir(parents=True,exist_ok=True)
    cfg=Config(D=64,LANES=64,TMEM_WORDS=8192,DRAM_BYTES=1<<16)
    rng=np.random.default_rng(319)
    memory=rng.normal(0,.7,8192).astype(np.float32).view(np.uint32)
    # Positive RSQRT inputs, and cancellation-sensitive sum terms of differing magnitude.
    memory[1024:2048]=np.abs(memory[1024:2048].view(np.float32)).view(np.uint32)
    memory[:512]=np.ldexp(rng.normal(size=512).astype(np.float32),rng.integers(-10,11,512)).view(np.uint32)
    memory[:8]=[0,0x80000000,1,0x00800000,0x7f7fffff,0x7f800000,0x7fc00001,0xbf800000]
    memory[1024:1032]=[0,0x80000000,1,0x00800000,0x7f7fffff,0x7f800000,0x7fc00001,0x3f800000]
    programs=[]
    for i,func in enumerate([I.V_ADD,I.V_SUB,I.V_MUL,I.V_EXP2,I.V_EXP2SUB,I.V_RECIP,I.V_RSQRT,I.V_LOG2,
                            I.V_RSUM,I.V_RSSQ,I.V_RDOT,I.V_RMAX,I.V_ARGMAX]):
        src=1024 if func==I.V_RSQRT else 0
        # More than 64 terms forces feedback, and the final chunk is partial.
        programs.append(I.vop(func,4096+256*i,src,2048,1,193,256,256,256))
    golden=Machine(cfg,[programs+[I.halt()]],[np.zeros(cfg.DRAM_BYTES,np.uint8)])
    golden.slices[0].tmem[:]=memory
    golden.run()
    (a.output/'tmem.hex').write_text('\n'.join(f'{int(v):08x}' for v in memory)+'\n')
    cmdwords=[]
    for ins in programs:
        packed=(ins.op<<232)|(ins.flags<<224)
        for k,w in enumerate(ins.w):packed|=int(w)<<(32*(6-k))
        cmdwords.append(f'{packed:060x}')
    (a.output/'commands.hex').write_text('\n'.join(cmdwords)+'\n')
    tb=Path(__file__).with_name('tb_vpu_wide.sv')
    sources=['vpu/otpu_fp.sv','vpu/otpu_fpipe.sv','top/otpu_pkg.sv',
             'vpu/otpu_vtree.sv','vpu/otpu_se_comp.sv','vpu/otpu_se_tail.sv','vpu/otpu_vpu.sv']
    rtlsim.VFLAGS += [f'-DOTPU_FP_MUL_LAT={a.fp_mul_latency}',f'-DOTPU_FP_ADD_LAT={a.fp_add_latency}']
    exe=rtlsim.build('tb_vpu_wide',[rtlsim.RTL/s for s in sources]+[tb],dict(LANES=64))
    run=subprocess.run([str(exe),f'+dir={a.output}',f'+ncmd={len(programs)}'],capture_output=True,text=True)
    (a.output/'rtl.log').write_text(run.stdout+run.stderr)
    run.check_returncode()
    got=np.fromfile(a.output/'tmem_out.bin',dtype='<u4')
    bad=np.flatnonzero(got!=golden.slices[0].tmem)
    if len(bad):
        raise RuntimeError(f'wide VPU differs at {bad[:24].tolist()}: actual {got[bad[:8]].tolist()}, expected {golden.slices[0].tmem[bad[:8]].tolist()}')
    report=dict(status='complete',lanes=64,all_tmem_bit_exact=True,
                fp_mul_latency=a.fp_mul_latency,fp_add_latency=a.fp_add_latency,
                functions=[I.VFUNCS[ins.w[5]>>16&255] for ins in programs],elements_per_row=193,
                command_cycles=[int(v) for _,v in re.findall(r'VCMD index=(\d+) cycles=(\d+)',run.stdout)])
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
