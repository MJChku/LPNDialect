"""Compare original and lookup QACT scale units under identical enabled clocks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('opentpu-root','table','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--fp-mul-latency',type=int,choices=(1,2),default=2)
    p.add_argument('--fp-add-latency',type=int,choices=(1,2,3,4),default=4)
    a=p.parse_args();a.table=a.table.resolve();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    calibration=json.loads((a.table/'report.json').read_text())
    sys.path.insert(0,str(a.opentpu_root.resolve()))
    from opentpu import rtlsim
    root=Path(__file__).parent
    sources=[rtlsim.RTL/s for s in ('vpu/otpu_fp.sv','vpu/otpu_fpipe.sv','top/otpu_pkg.sv','vpu/otpu_quant.sv')]
    sources += [root/s for s in ('otpu_recip_rom.sv','otpu_qscale_lookup.sv','tb_qscale_lookup.sv')]
    files=sources+[a.table/'coeff.hex',a.table/'correction.hex']
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    for name in ('coeff.hex','correction.hex'):
        if hashes[str(a.table/name)]!=calibration['files'][name]['sha256']:raise RuntimeError('calibration table changed')
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='12'
    rtlsim.VFLAGS+=['-CFLAGS','-O3',f'-DOTPU_FP_MUL_LAT={a.fp_mul_latency}',f'-DOTPU_FP_ADD_LAT={a.fp_add_latency}']
    before=time.perf_counter();exe=rtlsim.build('tb_qscale_lookup',sources,{})
    built=time.perf_counter()-before;before=time.perf_counter()
    run=subprocess.run([str(exe),f'+recip_coeff={a.table/"coeff.hex"}',f'+recip_correction={a.table/"correction.hex"}'],
        capture_output=True,text=True,timeout=600)
    (a.output/'rtl.log').write_text(run.stdout+run.stderr);run.check_returncode()
    observed=re.search(r'QSCALE_LOOKUP_GATE checked=(\d+) old_latency=(\d+) new_latency=(\d+) held=(\d+)',run.stdout)
    if not observed or int(observed[1])!=131072:raise RuntimeError('incomplete scale-unit comparison')
    if hashes!={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}:raise RuntimeError('sources changed during comparison')
    report=dict(status='complete',scope=__doc__,checked=int(observed[1]),old_enabled_clocks=int(observed[2]),
        new_enabled_clocks=int(observed[3]),enable_holds=int(observed[4]),all_scale_and_inverse_bits_identical=True,
        fp_mul_latency=a.fp_mul_latency,fp_add_latency=a.fp_add_latency,
        simulator=str(exe),simulator_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),source_and_table_sha256=hashes,
        build_seconds=built,run_seconds=time.perf_counter()-before,
        limitations=['Scale-unit component comparison','Not yet integrated with quantizer, VPU or inference','FPGA synthesis/clock not verified'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
