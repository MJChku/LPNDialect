"""Verify retained graph work, native logs, binaries and v5e-screen accounting."""
import argparse
import json
from pathlib import Path
import re

from .e2e import atomic_json,digest
from .glm47_trace_audit import audit as audit_trace


def audit(case,traces,compiler):
    summary=json.loads((case/'report.json').read_text())
    if summary['status']!='complete' or len(summary['records'])!=10:raise ValueError('complete ten-point screen required')
    for path,expected in summary['source_sha256'].items():
        if digest(Path(path))!=expected:raise ValueError('architecture source changed')
    trace_audits={s:audit_trace(traces/('trace-'+s)) for s in ('balanced','hotspot')}
    records=[]
    for r in summary['records']:
        path=case/(r['scenario']+'-'+r['name']);retained=json.loads((path/'report.json').read_text())
        if digest(path/'report.json')!=r['report_sha256']:raise ValueError('point report changed')
        if any(retained[k]!=v for k,v in r.items() if k in retained):raise ValueError('summary/point disagreement')
        for file,key in [('timing.native','binary_sha256'),('timing.tables','input_tables_sha256')]:
            if digest(path/file)!=r[key]:raise ValueError('native artifact changed')
        source=Path(next(p for p in summary['source_sha256'] if p.endswith('/tensor_dag.py')))
        if digest(source)!=r['source_sha256']:raise ValueError('point source changed')
        trace=traces/('trace-'+r['scenario']);graph=json.loads((trace/'commands.json').read_text())
        if digest(trace/'commands.json')!=r['commands_sha256']:raise ValueError('point used different commands')
        if digest(trace/'report.json')!=r['upstream_trace_sha256']:raise ValueError('upstream report changed')
        log=(path/'lpn.log').read_text();cycles=float(re.search(r'Simulation finished at t=([0-9.]+)',log)[1])
        if cycles!=r['cycles'] or 'rollbacks: 0' not in log or 'place program_completed: 1 token(s)' not in log:
            raise ValueError('native run did not finish correctly')
        if float(re.search(r'Wall window elapsed: ([0-9.]+)',log)[1])!=r['native_seconds_samples'][0]:
            raise ValueError('native wall time differs')
        actual={};by_call={};traffic=0
        for c in graph['commands']:
            traffic+=c['weight_bytes']+c['activation_read_bytes']+c['activation_write_bytes']
            if c['unit']!='matrix':continue
            shape=c['matrix_shape'];macs=shape['batch']*shape['m']*shape['n']*shape['k']
            if macs!=c['macs']:raise ValueError('shape arithmetic differs')
            p=c['matrix_precision'];actual[p]=actual.get(p,0)+2*macs
            key=str(c['call']);by_call[key]=by_call.get(key,0)+2*macs
        a=r['arithmetic'];seconds=cycles/(r['hardware']['clock_mhz']*1e6)
        if a['matrix_flops']!=sum(actual.values()) or a['matrix_flops_by_call']!=by_call:
            raise ValueError('useful FLOP accounting differs')
        if actual!={p:v['matrix_flops'] for p,v in a['by_precision'].items()}:
            raise ValueError('precision accounting differs')
        if abs(a['achieved_matrix_tflops']-sum(actual.values())/seconds/1e12)>1e-10:
            raise ValueError('achieved rate differs')
        if traffic!=r['statistics']['hbm_transfer_bytes']:raise ValueError('traffic differs')
        floor=traffic/(r['hardware']['hbm_gbs']*1e9)
        if seconds+1e-9<floor:raise ValueError('shared HBM capacity violated')
        if not r['shared_matrix_unit']:raise ValueError('matrix cluster not shared')
        records.append(dict(scenario=r['scenario'],name=r['name'],cycles=cycles,
            matrix_flops=sum(actual.values()),matrix_flops_by_precision=actual,
            assumed_hardware_seconds=seconds,hbm_service_lower_bound_seconds=floor,
            useful_matrix_tflops=a['achieved_matrix_tflops'],all_retained_artifacts_verified=True))
    return dict(status='complete',scope=__doc__,report_sha256=digest(case/'report.json'),
        audit_source_sha256=digest(Path(__file__)),compiler_sha256=digest(compiler),
        upstream_shape_audits=trace_audits,records=records,
        rtl_or_numerical_validation=False,hardware_design_assumptions_unchanged=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--traces',type=Path,required=True)
    p.add_argument('--compiler',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('refusing to replace evidence')
    atomic_json(a.output,audit(a.case,a.traces,a.compiler));print('v5e screen accounting audit passed')


if __name__=='__main__':main()
