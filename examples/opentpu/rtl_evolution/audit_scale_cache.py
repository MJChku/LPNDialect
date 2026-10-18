"""Freshly audit a retained-input scale-cache RTL workload measurement."""
import argparse
import json
from pathlib import Path
import re
import struct


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('case','rtl-dir','uncached-replay','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    from examples.opentpu.e2e import atomic_json,digest
    from examples.opentpu.retained_inputs import verified_program
    from .parallel_inference import files_equal
    path=a.rtl_dir/'report.json';report=json.loads(path.read_text())
    original=json.loads((a.case/'report.json').read_text())
    control=json.loads(a.uncached_replay.read_text())
    if report['status']!='complete' or original['status']!='complete' or control['status']!='complete':
        raise ValueError('completed workload and control required')
    if not report['all_full_dram_and_tmem_bit_exact'] or not report['identical_inputs'] or report['lpn_scale_cache_modeled']:
        raise ValueError('invalid hardware-only numerical result')
    if (not original['all_dram_and_tmem_bit_exact'] or not original['all_inputs_identical_to_baseline'] or
        original['entrypoint']!='unmodified upstream Engine.generate' or
        original['prompt_tokens']!=128 or original['output_tokens']!=32):
        raise ValueError('requires verified upstream normal 128/32 inference')
    if (report['source_sha256'].get(str(a.case/'report.json'))!=digest(a.case/'report.json') or
        control['reference_report_sha256']!=digest(a.case/'report.json')):
        raise ValueError('workload reference differs')
    for name,expected in report['source_sha256'].items():
        if digest(Path(name))!=expected:raise ValueError('cached hardware source or table changed')
    for name,expected in control['rtl_and_model_source_sha256'].items():
        if digest(Path(name))!=expected:raise ValueError('uncached control source or table changed')
    index=report['kernel'];cfg=original['config'];record=original['kernels'][index]
    previous=next(r for r in control['records'] if r['step']==index)
    if report['config']!=cfg or report['args']!=record['args'] or report['input_sha256']!=previous['input_sha256'] or report['input_sha256']!=record['input_sha256']:
        raise ValueError('canonical inputs/configuration/arguments differ')
    ua=dict(report['uarch']);entries=ua.pop('SCALE_CACHE_ENTRIES')
    if entries!=16 or ua!=control['uarch']:raise ValueError('hardware parameter comparison differs beyond cache')
    for field,define in [('mul','OTPU_FP_MUL_LAT'),('add','OTPU_FP_ADD_LAT')]:
        def selected(flags):
            values=[f.split('=')[1] for f in flags if f.startswith('-D'+define+'=')]
            if len(values)!=1:raise ValueError('missing or duplicate FP latency define')
            return values[0]
        if selected(report['rtl_compile_flags'])!=selected(previous['rtl_compile_flags']):
            raise ValueError('FP latency differs from control')
    if digest(Path(report['simulator']))!=report['simulator_sha256'] or digest(Path(previous['simulator']))!=previous['baseline_binary_sha256']:
        raise ValueError('cached or control simulator changed')
    log=(a.rtl_dir/'rtl.log').read_text()
    if digest(a.rtl_dir/'rtl.log')!=report['raw_log_sha256']:raise ValueError('raw cached log changed')
    result=re.search(r'RESULT cycles=(\d+) halted=(\d+) error=(\d+)',log)
    timer=re.search(r'LPN_WALL seconds=([0-9.]+) cycles=(\d+)',log)
    if not result or (result[2],result[3])!=('1','0') or int(result[1])!=report['rtl_cycles'] or not timer or int(timer[2])!=int(result[1]):
        raise ValueError('raw RTL completion or cycle timer differs')
    if [int(n) for n in re.findall(r'SLICE \d+ icount=(\d+)',log)]!=record['instructions']:
        raise ValueError('instruction count differs')
    counts=re.findall(r'SCACHE s=(\d+) p=(\d+) req=(\d+) miss=(\d+)',log)
    keys={(int(s),int(p)) for s,p,_,_ in counts}
    if len(counts)!=cfg['S']*ua['MXU_ROWS'] or keys!={(s,p) for s in range(cfg['S']) for p in range(ua['MXU_ROWS'])}:
        raise ValueError('cache counters do not cover all physical engines exactly once')
    requests=sum(int(n) for _,_,n,_ in counts);misses=sum(int(n) for _,_,_,n in counts)
    if (requests,misses)!=(report['cache_requests'],report['cache_misses']):raise ValueError('cache counter totals differ')
    source=a.case/'kernels'/f'{index:04d}'
    for sid in range(cfg['S']):
        for name in (f'prog_{sid}.hex',f'dram_{sid}.bin'):
            if digest(a.rtl_dir/name)!=record['input_sha256'][name]:raise ValueError('canonical input changed')
        _,identity=verified_program(a.rtl_dir,sid)
        if identity!=report['program_json_sha256'][f'program_{sid}.json']:raise ValueError('decoded program changed')
        if not files_equal(a.rtl_dir/f'dram_out_{sid}.bin',source/f'golden_dram_{sid}.bin'):
            raise ValueError('full DRAM differs from independent ISA')
        words=[int(w,16) for w in (a.rtl_dir/f'tmem_{sid}.hex').read_text().split()]
        if len(words)!=cfg['TMEM_WORDS'] or struct.pack('<'+'I'*len(words),*words)!=(source/f'golden_tmem_{sid}.bin').read_bytes():
            raise ValueError('full scratchpad differs from independent ISA')
    result=dict(status='complete',scope=__doc__,kernel=index,uncached_rtl_cycles=previous['rtl_cycles'],
        cached_rtl_cycles=report['rtl_cycles'],cycle_speedup=previous['rtl_cycles']/report['rtl_cycles'],
        fraction_fewer_cycles=1-report['rtl_cycles']/previous['rtl_cycles'],
        original_rtl_cycles=record['baseline_rtl_cycles'],cache_requests=requests,cache_misses=misses,
        backing_scale_bytes=misses*32,uncached_scale_bytes=requests*32,cache_hit_fraction=1-misses/requests,
        all_sources_tables_inputs_binaries_and_raw_counts_verified=True,all_full_dram_and_tmem_bit_exact=True,
        report_sha256=digest(path),control_report_sha256=digest(a.uncached_replay),script_sha256=digest(Path(__file__)),
        numerical_arithmetic_repeated=False,lpn_cache_timing_validated=False,
        limitations=['One retained '+record['phase']+' kernel, not complete inference','No FPGA mapping or clock proof'])
    atomic_json(a.output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
