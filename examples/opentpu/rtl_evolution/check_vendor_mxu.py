"""Compare the integrated vendor-DSP MXU with behavioral RTL and independent ISA."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('opentpu-root', 'unisim-root', 'compiler', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--rows', type=int, default=2)
    p.add_argument('--columns', type=int, choices=(2,4,8), default=2)
    a = p.parse_args()
    for name in ('opentpu_root','unisim_root','compiler','output'):
        setattr(a,name,getattr(a,name).resolve())
    from examples.opentpu.e2e import atomic_json, digest
    if a.rows < 2: p.error('parallel MXU requires at least two rows')
    a.output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), *[Path(__file__).with_name(n) for n in
        ('vendor_model.py','check_matrix.py','check_matrix_pipeline.py','audit_packed_mxu.py')]]
    hashes = {str(f): digest(f) for f in files}
    atomic_json(a.output/'launch.json', dict(status='running', source_sha256=hashes,
        rows=a.rows, columns=a.columns))
    audited = []
    for name, tech in [('behavioral',0), ('vendor',1)]:
        output = a.output/name
        command = [sys.executable, '-m', 'examples.opentpu.rtl_evolution.check_matrix_pipeline',
            '--opentpu-root', str(a.opentpu_root), '--output', str(output),
            '--rows', str(a.rows), '--columns', str(a.columns), '--paired-dsp']
        if tech: command += ['--pair-tech','1','--unisim-root',str(a.unisim_root)]
        with (a.output/(name+'.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        audit_output = a.output/(name+'-audit')
        code = ('from pathlib import Path; from examples.opentpu.rtl_evolution.audit_packed_mxu import audit; '
                'import sys; audit(*map(Path,sys.argv[1:]))')
        subprocess.run([sys.executable,'-c',code,str(output),str(a.opentpu_root),
                        str(a.compiler),str(audit_output)], check=True)
        audited.append(json.loads((audit_output/'report.json').read_text()))
    behavior, vendor = audited
    if not vendor['vendor_model'] or vendor['uarch'] != behavior['uarch'] | {'MXU_PAIR_TECH':1}:
        raise ValueError('vendor implementation was not selected exclusively')
    if vendor['config'] != behavior['config']: raise ValueError('configuration differs')
    records = []
    for original, changed in zip(behavior['records'],vendor['records']):
        for field in ('case','input_sha256','output_sha256','rtl_cycles','lpn_cycles'):
            if original[field] != changed[field]: raise ValueError('vendor/behavior mismatch: '+field)
        records.append(changed)
    if len(records) != 11: raise ValueError('missing matrix gate')
    if hashes != {str(f): digest(f) for f in files}: raise ValueError('gate sources changed')
    result = dict(status='complete',scope=__doc__,source_sha256=hashes,
        audit_sha256={n:digest(a.output/n/'report.json') for n in ('behavioral-audit','vendor-audit')},
        config=vendor['config'],uarch=vendor['uarch'],vendor_model=vendor['vendor_model'],
        records=records,all_inputs_full_outputs_cycles_and_lpn_predictions_identical=True,
        worst_relative_cycle_error=max(abs(r['relative_cycle_error']) for r in records),
        limitations=['Binary-pin specialized vendor DSP48E2 simulation, not four-state floating-pin verification',
                    'Small integrated matrix gates; no DSP58, full vendor-primitive inference or FPGA fit/Fmax proof'])
    atomic_json(a.output/'report.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as error:
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1])
            if output.is_dir():
                from examples.opentpu.e2e import atomic_json
                atomic_json(output/'failure.json',dict(status='failed',
                    error_type=type(error).__name__,error=str(error)))
        raise
