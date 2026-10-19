"""Freshly audit retained-input hardware replay results and numerical dumps."""
import argparse
import json
from pathlib import Path
import re
import struct


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selected-kernels',action='store_true',help='Audit an explicitly labeled subset without claiming complete inference')
    a=p.parse_args()
    from examples.opentpu.e2e import atomic_json,digest
    from .parallel_inference import files_equal,validate_variant
    from examples.opentpu.retained_inputs import verified_program
    report=json.loads((a.case/'report.json').read_text())
    validate_variant(report['timing'],report['uarch'])
    if report.get('timing',{}).get('pinned_cache_bytes',0):
        raise ValueError('pinned cache proposal has no matching RTL implementation')
    if report.get('timing',{}).get('reciprocal_lookup_lanes',0):
        raise ValueError('wide reciprocal proposal has no matching RTL implementation')
    full=report['indices']==list(range(48))
    if report['status']!='complete' or report['failures'] or (not full and not a.selected_kernels):
        raise ValueError('requires all 48 completed kernels, or --selected-kernels for a subset')
    if not report['indices'] or sorted(set(report['indices']))!=report['indices'] or not set(report['indices'])<=set(range(48)):
        raise ValueError('invalid kernel indices')
    if [r['step'] for r in report['records']]!=report['indices']:
        raise ValueError('records do not match the requested kernel indices')
    reference=Path(report['case'])
    if digest(reference/'report.json')!=report['reference_report_sha256']:
        raise RuntimeError('reference report changed')
    original=json.loads((reference/'report.json').read_text())
    if not original['all_dram_and_tmem_bit_exact'] or not original['all_inputs_identical_to_baseline']:
        raise RuntimeError('reference was not numerically verified')
    for name,expected in report['rtl_and_model_source_sha256'].items():
        if digest(Path(name))!=expected:raise RuntimeError(f'source changed: {name}')
    records=[];binary_hashes={}
    for record in report['records']:
        index=record['step']
        source=original['kernels'][index]
        if not record['all_dram_and_tmem_bit_exact'] or not record['identical_input_files']:
            raise RuntimeError('invalid kernel evidence')
        if record['timing']!=report['timing']:
            raise RuntimeError('worker timing parameters differ from the requested architecture')
        if 'rtl_top_parameters' in record and bool(record['rtl_top_parameters'].get('ACT_STREAMING',0))!=bool(record['timing'].get('activation_streaming',False)):
            raise RuntimeError('RTL streaming parameter differs from model timing')
        if 'rtl_top_parameters' in record and bool(record['rtl_top_parameters'].get('QUANT_LOOKUP',0))!=bool(record['timing'].get('quant_scale_lookup',False)):
            raise RuntimeError('RTL lookup scale parameter differs from model timing')
        if 'rtl_top_parameters' in record and bool(record['rtl_top_parameters'].get('VPU_SCALAR_RECIP',0))!=bool(record['timing'].get('scalar_recip_lookup',False)):
            raise RuntimeError('RTL reciprocal lookup parameter differs from model timing')
        if record['timing'].get('quant_scale_lookup',False) or record['timing'].get('scalar_recip_lookup',False):
            expected_tables={Path(n).name:h for n,h in report['rtl_and_model_source_sha256'].items()
                             if Path(n).name in ('coeff.hex','correction.hex')}
            if len(expected_tables)!=2 or record.get('lookup_table_sha256')!=expected_tables:
                raise RuntimeError('worker lookup table data differs from global source evidence')
        if 'rtl_compile_flags' in record:
            flags=record['rtl_compile_flags']
            for field,define,default in (('fp_mul_latency','OTPU_FP_MUL_LAT',2),
                                          ('fp_add_latency','OTPU_FP_ADD_LAT',4)):
                selected=[f for f in flags if f.startswith('-D'+define+'=')]
                actual=int(selected[0].split('=')[1]) if selected else default
                if len(selected)>1 or actual!=record['timing'].get(field,default):
                    raise RuntimeError('RTL arithmetic defines differ from model timing')
        directory=a.case/'kernels'/f'{index:04d}'
        if record['input_sha256']!=source['input_sha256']:
            raise RuntimeError('input hashes differ from reference')
        for name,expected in record['input_sha256'].items():
            if digest(directory/name)!=expected:raise RuntimeError(f'input changed: {index}/{name}')
        exe=record['simulator']
        if exe not in binary_hashes:binary_hashes[exe]=digest(Path(exe))
        if binary_hashes[exe]!=record['baseline_binary_sha256']:
            raise RuntimeError('simulator binary changed')
        log=(directory/'rtl.log').read_text()
        result=re.search(r'RESULT cycles=(\d+) halted=(\d+) error=(\d+)',log)
        if not result or (result[2],result[3])!=('1','0') or int(result[1])!=record['rtl_cycles']:
            raise RuntimeError('raw RTL log disagrees')
        instructions=[int(n) for n in re.findall(r'SLICE \d+ icount=(\d+)',log)]
        if instructions!=record['instructions'] or instructions!=source['instructions']:
            raise RuntimeError('RTL instruction count differs from upstream trace')
        if report['timing'].get('scale_cache_entries',0):
            observed=re.findall(r'SCACHE s=(\d+) p=(\d+) req=(\d+) miss=(\d+)',log)
            expected={(s,p) for s in range(original['config']['S']) for p in range(report['uarch']['MXU_ROWS'])}
            if len(observed)!=len(expected) or {(int(s),int(p)) for s,p,_,_ in observed}!=expected:
                raise RuntimeError('scale cache counters missing or duplicated')
            cache=record['scale_cache_counts']
            if (cache['requests']!=sum(int(n) for _,_,n,_ in observed) or
                cache['requests']!=cache['predicted_requests'] or
                cache['misses']!=sum(int(n) for _,_,_,n in observed)):
                raise RuntimeError('scale cache raw counts differ')
        native_log=(directory/'lpn.log').read_text()
        native_time=re.search(r'Simulation finished at t=([0-9.]+)',native_log)
        if ('rollbacks: 0' not in native_log or 'place program_completed: 1 token(s)' not in native_log
            or not native_time or float(native_time[1])!=record['lpn_cycles']):
            raise RuntimeError('native completion log disagrees with recorded timing')
        if abs((record['lpn_cycles']/record['rtl_cycles']-1)-record['evolved_lpn_relative_cycle_error'])>1e-12:
            raise RuntimeError('recorded model error disagrees with cycle counts')
        for sid in range(original['config']['S']):
            _,identity=verified_program(directory,sid)
            if 'program_json_sha256' in record and record['program_json_sha256'][f'program_{sid}.json']!=identity:
                raise RuntimeError('decoded program hash changed')
            if not files_equal(directory/f'dram_out_{sid}.bin',reference/'kernels'/f'{index:04d}'/f'dram_out_{sid}.bin'):
                raise RuntimeError('full DRAM differs from previously validated independent ISA state')
            words=[int(w,16) for w in (directory/f'tmem_{sid}.hex').read_text().split()]
            if len(words)!=original['config']['TMEM_WORDS'] or struct.pack('<'+'I'*len(words),*words)!=(directory/f'golden_tmem_{sid}.bin').read_bytes():
                raise RuntimeError('full scratchpad differs from independent ISA state')
        records.append(dict(step=index,phase=source['phase'],rtl_cycles=record['rtl_cycles'],
            original_rtl_cycles=record['baseline_rtl_cycles'],lpn_cycles=record['lpn_cycles'],
            relative_cycle_error=record['evolved_lpn_relative_cycle_error']))
    rtl=sum(r['rtl_cycles'] for r in records);native=sum(r['lpn_cycles'] for r in records)
    before=sum(r['original_rtl_cycles'] for r in records)
    expected_rtl=report['full_inference_rtl_cycles'] if full else report['selected_rtl_cycles']
    expected_native=report['full_inference_lpn_cycles'] if full else report['selected_lpn_cycles']
    expected_before=sum(original['kernels'][i]['baseline_rtl_cycles'] for i in report['indices']) if not full else original['baseline_rtl_cycles']
    if rtl!=expected_rtl or native!=expected_native or before!=expected_before:
        raise RuntimeError('aggregate cycles disagree')
    result=dict(status='complete',report_sha256=digest(a.case/'report.json'),script_sha256=digest(Path(__file__)),
        full_inference=full,kernel_indices=report['indices'],selected_rtl_cycles=rtl,original_rtl_cycles=before,lpn_cycles=native,cycle_speedup=before/rtl,
        relative_cycle_error=native/rtl-1,worst_kernel_error=max(abs(r['relative_cycle_error']) for r in records),
        all_inputs_and_simulator_hashes_verified=True,all_program_json_matches_canonical_hex=True,all_full_dram_and_tmem_bit_exact=True,
        all_raw_rtl_logs_and_instruction_counts_verified=True,all_model_and_rtl_source_hashes_verified=True,
        schedule='Retained unmodified upstream Engine.generate, 128 input and 32 output tokens',
        generated_ids_preserved_by_full_output_equivalence=full,simulator_sha256=binary_hashes,
        phases={phase:dict(rtl_cycles=sum(r['rtl_cycles'] for r in records if r['phase']==phase),
                          original_rtl_cycles=sum(r['original_rtl_cycles'] for r in records if r['phase']==phase))
                for phase in ('prefill','decode')},records=records)
    if full:result['full_rtl_cycles']=rtl
    atomic_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
