"""Freshly audit cached RTL memories/counts, then predict timing from instructions."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('case','opentpu-root','compiler','uncached-replay','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gates',type=Path,nargs='+',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from examples.opentpu.e2e import digest,atomic_json
    from examples.opentpu.workload import use_checkout
    from examples.opentpu.retained_inputs import verified_program
    from examples.opentpu.shape_workload import resolve_static
    from examples.opentpu.model import Timing
    from examples.opentpu.scale_cache_timing import cache_metadata
    from examples.opentpu.native_timing import ReusableTiming
    use_checkout(a.opentpu_root)
    from opentpu import Config,isa as I,waits as W
    reference=json.loads((a.case/'report.json').read_text())
    control=json.loads(a.uncached_replay.read_text());records=[];gates={}
    root=Path(__file__).resolve().parents[1]
    files=[root/n for n in ['model.py','compact.py','components.py','scale_cache_timing.py',
        'shape_workload.py','retained_inputs.py','native_timing.py','rtl_evolution/audit_scale_cache.py']]+[Path(__file__)]
    hashes={str(f):digest(f) for f in files}
    for gate in a.gates:
        actual=json.loads((gate/'report.json').read_text());index=actual['kernel']
        # Numerical data and counters are evidence only. They never enter the
        # metadata/prediction interface, which takes the original commands.
        subprocess.run([sys.executable,'-m','examples.opentpu.rtl_evolution.audit_scale_cache',
            '--case',str(a.case),'--rtl-dir',str(gate),'--uncached-replay',str(a.uncached_replay),
            '--output',str(a.output/f'audit-{index}.json')],check=True,capture_output=True,text=True)
        cfg=Config(**reference['config']);source=a.case/'kernels'/f'{index:04d}'
        encoded,identity=verified_program(source,0)
        work=resolve_static(cfg,[[I.Instr.decode(w) for w in encoded]],reference['kernels'][index]['args'])
        ua=actual['uarch'];work=replace(work,config=replace(cfg,MCOLS=ua.get('MXU_COLS',1),LANES=64))
        timing=replace(Timing(**control['timing']),scale_cache_entries=ua['SCALE_CACHE_ENTRIES'])
        if timing.matrix_row_lanes!=ua['MXU_ROWS']:raise ValueError('physical core count differs')
        metadata=cache_metadata(work.commands[0],I,W,timing.matrix_row_lanes,timing.scale_cache_entries,timing.dram_latency)
        native=ReusableTiming(a.compiler,a.output/'native-cache',capacity=65536).prepare(work,timing,a.output/f'{index}.tables')
        cycles,_,_,log=native.run();(a.output/f'{index}.log').write_text(log)
        if metadata['requests']!=actual['cache_requests']:raise ValueError('physical request count differs')
        result=dict(kernel=index,phase=reference['kernels'][index]['phase'],program_json_sha256=identity,
            rtl_cycles=actual['rtl_cycles'],lpn_cycles=cycles,relative_error=cycles/actual['rtl_cycles']-1,
            requests=metadata['requests'],actual_requests=actual['cache_requests'],
            predicted_misses=metadata['backing_reads'],actual_misses=actual['cache_misses'],
            miss_count_relative_error=metadata['backing_reads']/actual['cache_misses']-1,
            timing=asdict(timing),fresh_rtl_audit_sha256=digest(a.output/f'audit-{index}.json'))
        records.append(result);gates[str(gate/'report.json')]=digest(gate/'report.json');print(json.dumps(result),flush=True)
    if hashes!={str(f):digest(f) for f in files}:raise ValueError('timing sources changed')
    result=dict(status='complete',scope=__doc__,records=records,source_sha256=hashes,gate_report_sha256=gates,
        reference_report_sha256=digest(a.case/'report.json'),uncached_report_sha256=digest(a.uncached_replay),
        timing_target_met_all_cases=all(abs(r['relative_error'])<=.1 for r in records),
        worst_case_relative_error=max(abs(r['relative_error']) for r in records),
        numerical_prepass_repeated=False,
        limitations=['Four retained kernels, not full inference',
            'Private direct-mapped state is bounded; coherence follows canonical command order',
            'Asynchronous invalidations and their exact miss counts remain approximate',
            'No FPGA placement or clock proof'])
    atomic_json(a.output/'report.json',result)
    if not result['timing_target_met_all_cases']:raise ValueError('cache timing exceeds ten percent error')


if __name__=='__main__':main()
