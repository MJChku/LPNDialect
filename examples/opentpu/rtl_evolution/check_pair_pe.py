"""Compare one-DSP paired int8 arithmetic against two original scalar chains."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

UNISIM_REVISION='1c8e05fd1e9a79ceb8b996a0996674122eed086f'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('opentpu-root','unisim-root','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--exhaustive',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from examples.opentpu.workload import use_checkout
    from examples.opentpu.e2e import atomic_json,digest
    use_checkout(a.opentpu_root.resolve())
    from opentpu import rtlsim
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.unisim_root,text=True).strip()
    if revision!=UNISIM_REVISION:raise ValueError('unexpected vendor simulation-library revision')
    root=Path(__file__).parent
    vendor=a.unisim_root/'verilog/src/unisims/DSP48E2.v'
    # Every primitive pin is explicitly driven with binary values. Verilator
    # cannot elaborate the vendor's floating-input default tests on tied pins.
    # Specialize those tests only; retain the complete arithmetic/register model.
    binary,count=re.subn(r"\(([A-Za-z0-9_]+(?:\[\d+\])?)\s+(===|!==)\s+1'bz\)",
        lambda m:"1'b0" if m[2]=='===' else "1'b1",vendor.read_text())
    if count!=168 or "1'bz" in binary:raise ValueError('unexpected vendor floating-input test structure')
    generated=rtlsim.BUILD/'lpn_vendor_binary'/('DSP48E2_'+hashlib.sha256(binary.encode()).hexdigest()+'.v')
    generated.parent.mkdir(parents=True,exist_ok=True)
    if generated.exists() and generated.read_text()!=binary:raise RuntimeError('generated vendor model differs')
    if not generated.exists():generated.write_text(binary)
    sources=[a.unisim_root/'verilog/src/glbl.v',generated,
             rtlsim.RTL/'tech/generic/otpu_tech_dsp.sv',root/'otpu_pair_pe.sv',root/'tb_pair_pe.sv']
    files=sources+[vendor,Path(__file__)];hashes={str(f):digest(f) for f in files}
    rtlsim.VFLAGS+=['-CFLAGS','-O3']
    before=time.perf_counter();exe=rtlsim.build('tb_pair_pe',sources,{})
    built=time.perf_counter()-before;command=[str(exe)]+(['+exhaustive'] if a.exhaustive else [])
    atomic_json(a.output/'launch.json',dict(command=command,source_sha256=hashes,simulator_sha256=digest(exe)))
    before=time.perf_counter();run=subprocess.run(command,capture_output=True,text=True,timeout=1800)
    (a.output/'rtl.log').write_text(run.stdout+run.stderr);run.check_returncode()
    m=re.search(r'PAIR_PE single=(\d+) chain=(\d+) stalls=(\d+) overflow_corner=(\d+)',run.stdout)
    expected=65536+(16777216 if a.exhaustive else 0)
    if not m or int(m[1])!=expected or min(map(int,m.groups()[1:]))<=0:raise RuntimeError('incomplete paired arithmetic gate')
    if hashes!={str(f):digest(f) for f in files}:raise RuntimeError('gate sources changed')
    report=dict(status='complete',scope=__doc__,vendor_source='https://github.com/Xilinx/XilinxUnisimLibrary',
        vendor_revision=revision,exhaustive_signed_int8_triplets=a.exhaustive,
        checked_single_clocks=int(m[1]),checked_chain_clocks=int(m[2]),chain_positions=8,
        enable_stalls=int(m[3]),max_positive_low_sum_cases=int(m[4]),
        both_behavioral_and_vendor_primitive_match_two_scalar_PEs=True,
        vendor_model_specialization=dict(binary_driven_pins_only=True,floating_input_tests_replaced=count,
            arithmetic_and_register_logic_unchanged=True),
        primitives_per_pair=1,source_sha256=hashes,simulator=str(exe),simulator_sha256=digest(exe),
        build_seconds=built,run_seconds=time.perf_counter()-before,
        limitations=['Component arithmetic gate, not integrated MXU or model inference',
                     'Shared operand required; independent asynchronous engines cannot simply be packed',
                     'DSP48E2 tested; DSP58 mapping, FPGA placement and Fmax not established',
                     'Column-end scale multipliers and FP epilogues are not packed by this component'])
    atomic_json(a.output/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
