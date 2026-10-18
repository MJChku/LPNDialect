"""Check the normalized reciprocal RTL against its original SV arithmetic function."""
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
    p.add_argument('--opentpu-root',type=Path,required=True)
    p.add_argument('--table',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=1<<23)
    p.add_argument('--full-reciprocal',action='store_true')
    a=p.parse_args()
    if not 1<=a.limit<=1<<23:p.error('limit must be 1..2^23')
    a.table=a.table.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    calibration=json.loads((a.table/'report.json').read_text())
    if calibration['status']!='complete' or calibration['correction_bits']!=2:
        raise RuntimeError('requires the complete signed-two-bit calibration table')
    sys.path.insert(0,str(a.opentpu_root.resolve()))
    from opentpu import rtlsim
    root=Path(__file__).parent
    top='tb_recip_full' if a.full_reciprocal else 'tb_recip_rom'
    sources=[rtlsim.RTL/'vpu/otpu_fp.sv',root/'otpu_recip_rom.sv',root/(top+'.sv')]
    files=sources+[a.table/'coeff.hex',a.table/'correction.hex']
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    for name in ('coeff.hex','correction.hex'):
        if hashes[str(a.table/name)]!=calibration['files'][name]['sha256']:
            raise RuntimeError('calibration table hash differs')
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='12'
    rtlsim.VFLAGS+=['-CFLAGS','-O3']
    before=time.perf_counter();exe=rtlsim.build(top,sources,dict(PORTS=2))
    built=time.perf_counter()-before
    before=time.perf_counter()
    run=subprocess.run([str(exe),f'+recip_coeff={a.table/"coeff.hex"}',
        f'+recip_correction={a.table/"correction.hex"}',f'+limit={a.limit}'],capture_output=True,text=True,timeout=1800)
    (a.output/'rtl.log').write_text(run.stdout+run.stderr);run.check_returncode()
    observed=re.search((r'RECIP_FULL_GATE checked=' if a.full_reciprocal else r'RECIP_ROM_GATE checked=')+r'(\d+) clocks=(\d+) held=(\d+) ports=(\d+)',run.stdout)
    if not observed or int(observed[1])!=a.limit:raise RuntimeError('missing complete primitive result')
    if hashes!={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}:
        raise RuntimeError('source/table changed during verification')
    report=dict(status='complete',scope=__doc__,tested_inputs=a.limit,full_reciprocal=a.full_reciprocal,exhaustive_normalized_inputs=not a.full_reciprocal and a.limit==1<<23,
        original_sv_function_bit_exact=True,enable_holds=int(observed[3]),ports=int(observed[4]),latency_enabled_clocks=4,
        build_seconds=built,run_seconds=time.perf_counter()-before,simulator=str(exe),
        simulator_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),source_and_table_sha256=hashes,
        calibration_report_sha256=hashlib.sha256((a.table/'report.json').read_bytes()).hexdigest(),
        limitations=['Full reciprocal primitive; sign/exponent edge grid and shuffled inputs' if a.full_reciprocal else 'Normalized primitive only','Not integrated with QACT or VPU','FPGA mapping and clock unverified'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
