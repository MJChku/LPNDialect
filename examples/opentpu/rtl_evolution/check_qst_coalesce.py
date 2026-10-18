"""Check coalesced QST write data, byte masks, grant bubbles and scatter fallback."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('opentpu-root','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from examples.opentpu.workload import use_checkout
    from examples.opentpu.e2e import digest,atomic_json
    use_checkout(a.opentpu_root.resolve())
    from opentpu import rtlsim
    root=Path(__file__).parent;sources=[rtlsim.RTL/'mem/otpu_qst_coalesce.sv',root/'tb_qst_coalesce.sv']
    if digest(sources[0])!=digest(root/'otpu_qst_coalesce.sv'):raise ValueError('packaged coalescer differs from RTL')
    hashes={str(f):digest(f) for f in sources+[Path(__file__)]}
    exe=rtlsim.build('tb_qst_coalesce',sources,{})
    run=subprocess.run([str(exe)],capture_output=True,text=True,timeout=60)
    (a.output/'rtl.log').write_text(run.stdout+run.stderr);run.check_returncode()
    m=re.search(r'QST_COALESCE checked=(\d+) raw_words=(\d+) merged_words=(\d+)',run.stdout)
    if not m or int(m[1])!=720:raise RuntimeError('incomplete coalescer gate')
    if hashes!={str(f):digest(f) for f in sources+[Path(__file__)]}:raise RuntimeError('gate sources changed')
    report=dict(status='complete',scope=__doc__,checked=int(m[1]),original_word_transactions=int(m[2]),
        coalesced_word_transactions=int(m[3]),all_data_and_byte_masks_equal=True,
        source_sha256=hashes,simulator=str(exe),simulator_sha256=digest(exe),
        limitations=['Component gate; full accelerator and workload gates are separate','No FPGA mapping or clock measurement'])
    atomic_json(a.output/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
