"""Run upstream GLM shape traces and explicit, uncalibrated LPN sensitivity points."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--prompt-tokens',type=int,default=1024);p.add_argument('--output-tokens',type=int,default=8)
    a=p.parse_args();a.output=a.output.resolve();a.compiler=a.compiler.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    from .e2e import atomic_json,digest
    root=Path(__file__).resolve().parents[1]
    files=[Path(__file__),Path(__file__).with_name('glm47_trace.py'),Path(__file__).with_name('glm47.py'),root/'tensor_dag.py']
    sources={str(f):digest(f) for f in files};started=time.perf_counter();records=[];failures=[];traces={}
    def save(status):
        atomic_json(a.output/'report.json',dict(status=status,scope=__doc__,source_sha256=sources,
            prompt_tokens=a.prompt_tokens,output_tokens=a.output_tokens,
            clock_mhz=1500,hbm_gbs=858.9934592,records=records,failures=failures,
            seconds=time.perf_counter()-started,routing_is_sensitivity_assumption=True,
            existing_opentpu_rtl_executed=False,rtl_timing_calibrated=False,generated_text_predicted=False,
            implementation='Proposed tensor-command accelerator; not the existing FeSens ISA',
            activation_policy='Every intermediate access spills to HBM; no persistent weight cache',
            purpose='Determine whether more arithmetic helps this explicit reference dataflow before implementing a larger accelerator'))
    def run(command,log):
        with log.open('w') as out:subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,check=True)
    save('tracing')
    for scenario in ('balanced','hotspot'):
        destination=a.output/('trace-'+scenario)
        run([sys.executable,'-m','examples.opentpu.glm47_trace','--output',str(destination),
            '--prompt-tokens',str(a.prompt_tokens),'--output-tokens',str(a.output_tokens),
            '--routing-scenario',scenario],a.output/('trace-'+scenario+'.log'))
        traces[scenario]=destination
    save('timing')
    points=[('base',12288,1024),('weight4x',49152,1024),('balanced4x',49152,4096),
            ('balanced16x',196608,16384)]
    def timing(scenario,name,weight,bf16):
        dest=a.output/f'{scenario}-{name}'
        run([sys.executable,'-m','examples.tensor_dag','--commands',str(traces[scenario]/'commands.json'),
            '--output',str(dest),'--compiler',str(a.compiler),'--clock-mhz','1500',
            '--hbm-gbs','858.9934592','--weight-macs-per-cycle',str(weight),
            '--bf16-macs-per-cycle',str(bf16)],a.output/f'{scenario}-{name}.log')
        d=json.loads((dest/'report.json').read_text())
        return dict(scenario=scenario,name=name,**d,report_sha256=digest(dest/'report.json'),
            upstream_trace_sha256=digest(traces[scenario]/'report.json'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(timing,s,*point):(s,point[0]) for s in traces for point in points}
        for future in as_completed(pending):
            try:records.append(future.result())
            except Exception as error:failures.append(dict(point=pending[future],error=repr(error)))
            save('timing')
    if sources!={str(f):digest(f) for f in files}:raise ValueError('sources changed during run')
    records.sort(key=lambda r:(r['scenario'],r['name']));save('failed' if failures else 'complete')
    if failures:raise RuntimeError('some timing screens failed')


if __name__=='__main__':main()
