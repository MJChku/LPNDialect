"""Check selectable arithmetic stage boundaries against independent FP32 values."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opentpu-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.opentpu_root.resolve()))
    import numpy as np
    from opentpu import fp32 as F,rtlsim
    a.output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(567)
    words=rng.integers(0,1<<32,(8192,4),dtype=np.uint32)
    corners=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,
                      0x00800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,
                      0x7fc00001,0xffc00001,0x7f800001,0x3f800000,0xbf800000],np.uint32)
    aa,bb=np.meshgrid(corners,corners,indexing='ij');n=aa.size
    words[:n,0]=aa.ravel();words[:n,1]=bb.ravel()
    words[:n,2]=np.roll(aa.ravel(),17);words[:n,3]=np.roll(bb.ravel(),113)
    x,y,c,d=[words[:,k].copy().view(np.float32) for k in range(4)]
    expected=np.column_stack([F.bits(F.add(x,y)),F.bits(F.mul(x,y)),
        F.bits(F.add(F.mul(x,y),c)),F.bits(F.add(F.mul(x,y),F.mul(c,d)))])
    for name,table in (('inputs',words),('expected',expected)):
        (a.output/(name+'.hex')).write_text('\n'.join(''.join(f'{int(w):08x}' for w in row) for row in table)+'\n')
    sources=[rtlsim.RTL/'vpu/otpu_fp.sv',rtlsim.RTL/'vpu/otpu_fpipe.sv',Path(__file__).with_name('tb_fp_partition.sv')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    exe=rtlsim.build('tb_fp_partition',sources)
    run=subprocess.run([str(exe),f'+dir={a.output}','+n=8192'],capture_output=True,text=True,timeout=120)
    (a.output/'rtl.log').write_text(run.stdout+run.stderr);run.check_returncode()
    if 'FP_PARTITION vectors=8192 ' not in run.stdout:raise RuntimeError('incomplete arithmetic gate')
    if hashes!={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
        raise RuntimeError('sources changed')
    result=dict(status='complete',vectors=8192,corner_pairs=256,
        add_latencies=[1,2,3,4,5,6],multiply_latencies=[1,2,3],
        madd_latencies=[2,6,4],mma_latency=2,enable_stalls_verified=True,
        all_outputs_bit_exact_to_numpy=True,source_sha256=hashes,simulator=str(exe),
        physical_clock_or_fpga_fit_validated=False)
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
