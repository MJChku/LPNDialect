"""Refresh an archived DSE resource bill without rerunning or altering timing evidence."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('report','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    from .tpu_dse import Architecture,resources
    from .e2e import atomic_json,digest
    report=json.loads(a.report.read_text());reference=json.loads(a.reference.read_text())
    if report['status']!='complete' or reference['status']!='complete':
        raise ValueError('completed evidence required')
    if digest(a.reference)!=report['reference_report_sha256']:
        raise ValueError('reference differs from original timing study')
    if a.output.exists():raise ValueError('refusing to replace existing resource evidence')
    points=[]
    for point in report['points']:
        arch=Architecture(**point['architecture'])
        corrected=resources(arch,reference['config'],report['clock_mhz'])
        points.append(dict(architecture=asdict(arch),resources=corrected,
            previously_reported_onchip_bytes=point['resources']['onchip_bytes_lower_bound'],
            added_onchip_bytes=corrected['onchip_bytes_lower_bound']-point['resources']['onchip_bytes_lower_bound']))
    files=[Path(__file__),Path(__file__).with_name('tpu_dse.py')]
    atomic_json(a.output,dict(status='complete',scope=__doc__,config=reference['config'],
        original_timing_report_sha256=digest(a.report),reference_report_sha256=digest(a.reference),
        source_sha256={str(f):digest(f) for f in files},timing_evidence_modified=False,
        clock_mhz=report['clock_mhz'],points=points))


if __name__=='__main__':main()
