"""Check scale-cache data, fixed latency, pending hits, conflicts and invalidation."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('opentpu-root','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    from examples.opentpu.workload import use_checkout
    from examples.opentpu.e2e import digest,atomic_json
    use_checkout(a.opentpu_root.resolve())
    from opentpu import rtlsim
    root=Path(__file__).parent
    sources=[rtlsim.RTL/'mem/otpu_scale_cache.sv',root/'tb_scale_cache.sv']
    if digest(sources[0])!=digest(root/'otpu_scale_cache.sv'):raise ValueError('RTL cache differs from packaged source')
    hashes={str(f):digest(f) for f in sources+[Path(__file__)]};records=[]
    for latency,entries in ((2,4),(8,16),(16,16)):
        before=time.perf_counter();exe=rtlsim.build('tb_scale_cache',sources,dict(LAT=latency,ENTRIES=entries))
        built=time.perf_counter()-before
        run=subprocess.run([str(exe)],capture_output=True,text=True,timeout=60)
        (a.output/f'lat{latency}.log').write_text(run.stdout+run.stderr);run.check_returncode()
        result=re.search(r'SCALE_CACHE requests=(\d+) responses=(\d+) misses=(\d+) hits=(\d+) blocked=(\d+) writes=(\d+) cancelled=(\d+)',run.stdout)
        if not result:raise RuntimeError('cache gate did not complete')
        req,res,miss,hit,blocked,writes,cancelled=map(int,result.groups())
        if req!=res+cancelled or req!=miss+hit:raise RuntimeError('cache gate counts disagree')
        records.append(dict(latency=latency,entries=entries,requests=req,responses=res,misses=miss,hits=hit,
            conflict_stalls=blocked,writes=writes,cancelled_by_reset=cancelled,
            all_response_bits_and_latencies_match=True,simulator=str(exe),simulator_sha256=digest(exe),build_seconds=built))
        atomic_json(a.output/'report.json',dict(status='running',scope=__doc__,records=records,source_sha256=hashes))
    if hashes!={str(f):digest(f) for f in sources+[Path(__file__)]}:raise RuntimeError('gate sources changed')
    report=dict(status='complete',scope=__doc__,records=records,source_sha256=hashes,
                limitations=['Component gate; full-accelerator numerical validation is separate','FPGA mapping and clock not measured'])
    atomic_json(a.output/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
