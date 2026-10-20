"""Bind actual-component LPN architecture results to retained RTL programs."""
import argparse
import json
from pathlib import Path
import re

from .e2e import atomic_json,digest
from .hardware_commands import load_commands,instruction_statistics
from .retained_inputs import verified_program
from .workload import use_checkout


def audit(case,reference,control):
    d=json.loads((case/'report.json').read_text());original=json.loads((reference/'report.json').read_text())
    prior=json.loads(control.read_text())
    if d['status']!='complete' or d['failures'] or d['indices']!=list(range(48)):
        raise ValueError('complete successful full-inference DSE required')
    if digest(reference/'report.json')!=d['reference_report_sha256']:raise ValueError('reference changed')
    if not original['all_dram_and_tmem_bit_exact'] or not original['all_inputs_identical_to_baseline']:
        raise ValueError('numerical RTL/input evidence missing')
    if prior['status']!='complete' or not prior['all_full_dram_and_tmem_bit_exact']:
        raise ValueError('fresh cached-RTL audit missing')
    for path,h in d['source_sha256'].items():
        if digest(Path(path))!=h:raise ValueError('timing source changed')
    expected={};canonical={}
    for record in original['kernels']:
        index=record['step'];source=reference/'kernels'/f'{index:04d}'
        work=load_commands(source)
        if [len(c) for c in work.commands]!=record['instructions']:raise ValueError('actual RTL instruction count differs')
        expected[index]=instruction_statistics(work)
        encoded={}
        for sid in range(work.config.S):
            _,identity=verified_program(source,sid);encoded[f'program_{sid}.json']=identity
            if work.hashes[f'prog_{sid}.hex']!=record['input_sha256'][f'prog_{sid}.hex']:
                raise ValueError('canonical hardware hex changed')
        canonical[index]=encoded
    results=[];control_checked=False
    for point in d['points']:
        name=point['architecture']['name'];records=point['kernels']
        if len(records)!=48 or [r['kernel'] for r in records]!=list(range(48)):
            raise ValueError('missing or reordered kernels')
        cycles=0;operations=0
        for record in records:
            index=record['kernel'];root=case/name/f'{index:04d}'
            if record['statistics']['hardware_instructions']!=expected[index]:raise ValueError('instruction arithmetic differs')
            if record['program_json_sha256']!=canonical[index]:raise ValueError('decoded hardware words differ')
            log=(root/'lpn.log').read_text()
            observed=float(re.search(r'Simulation finished at t=([0-9.]+)',log)[1])
            if observed!=record['cycles'] or 'rollbacks: 0' not in log or 'place program_completed: 1 token(s)' not in log:
                raise ValueError('native replay did not finish correctly')
            wall=float(re.search(r'Wall window elapsed: ([0-9.]+)',log)[1])
            if wall!=record['native_seconds']:raise ValueError('native time differs')
            if record['statistics']['transitions']<20:raise ValueError('actual component net expected')
            cycles+=observed;operations+=expected[index]['matrix_dot_operations']
        if cycles!=point['full_inference_cycles']:raise ValueError('aggregate cycles differ')
        if operations!=point['hardware_instruction_arithmetic']['matrix_dot_operations']:
            raise ValueError('aggregate instruction arithmetic differs')
        if name=='physical_scale16_128x1_add2':
            if cycles!=prior['lpn_cycles']:raise ValueError('validated control prediction changed')
            control_checked=True
        results.append(dict(architecture=name,full_inference_cycles=cycles,
            matrix_dot_operations=operations,native_simulation_seconds_sum=sum(r['native_seconds'] for r in records),
            all_hardware_hex_and_decoded_words_verified=True,all_native_logs_verified=True,
            all_actual_rtl_dynamic_instruction_counts_verified=True,
            new_architecture_rtl_validation=False if name.startswith('proposal_') else None))
    if not control_checked:raise ValueError('validated control missing')
    return dict(status='complete',scope=__doc__,dse_report_sha256=digest(case/'report.json'),
        original_reference_sha256=digest(reference/'report.json'),prior_cached_rtl_audit_sha256=digest(control),
        audit_source_sha256=digest(Path(__file__)),records=results,
        unchanged_control_cycles=prior['lpn_cycles'],control_actual_rtl_cycles=prior['full_rtl_cycles'],
        control_worst_rtl_kernel_error=prior['worst_kernel_error'],
        tensor_dag_used=False,glm_hardware_execution=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('case','reference','control-audit','opentpu-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();use_checkout(a.opentpu_root)
    if a.output.exists():p.error('refusing to replace evidence')
    atomic_json(a.output,audit(a.case,a.reference,a.control_audit));print('actual hardware-command DSE audit passed',flush=True)


if __name__=='__main__':main()
