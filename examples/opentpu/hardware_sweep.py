"""Replay unchanged canonical hardware programs across explicit physical profiles.

The optional functional frontend resolves actual data-dependent instructions
once per input. It supplies concrete control/memory footprints, never timing.
Every candidate then runs the existing native component LPN on those commands.
"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import time

from .e2e import atomic_json,digest
from .hardware_commands import load_commands,instruction_statistics
from .model import Timing
from .native_timing import ReusableTiming
from .workload import load,use_checkout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('opentpu-root','compiler','profiles','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--functional',action='store_true',help='Numerically resolve actual controls; requires retained DRAM images')
    p.add_argument('--clock-mhz',type=float,required=True,help='Common assumed clock, not achieved Fmax')
    a=p.parse_args()
    if not 0<a.clock_mhz<1e6:p.error('positive finite assumed clock required')
    use_checkout(a.opentpu_root);a.output.mkdir(parents=True,exist_ok=False)
    profiles=json.loads(a.profiles.read_text())
    if not profiles or len({r['name'] for r in profiles})!=len(profiles):raise ValueError('unique physical profiles required')
    for profile in profiles:
        name=profile['name'];columns=profile['physical_matrix_columns']
        if not name or Path(name).name!=name or name in ('.','..'):raise ValueError('profile name must be a directory name')
        if columns not in (1,2,4,8):raise ValueError('unsupported physical column count')
        Timing(**profile['timing'])
    files=[Path(__file__),*[Path(__file__).with_name(n) for n in ('model.py','compact.py','components.py',
        'workload.py','hardware_commands.py','shape_workload.py','native_timing.py','benchmark.py',
        'activation_stream.py','scale_cache_timing.py','vector_pool.py','vector_select.py')]]
    sources={str(f):digest(f) for f in files};started=time.perf_counter();records=[];functional_seconds=0
    runner=ReusableTiming(a.compiler,a.output/'native-cache',capacity=32768)
    def report(status):
        return dict(status=status,scope=__doc__,profiles=profiles,records=records,source_sha256=sources,
            profile_sha256=digest(a.profiles),assumed_clock_mhz=a.clock_mhz,
            numerical_control_resolved=a.functional,functional_frontend_seconds=functional_seconds,
            tensor_dag_used=False,architecture_rtl_validated_by_this_run=False,
            full_model_inference_established_by_this_run=False,host_seconds=time.perf_counter()-started)
    for index,source in enumerate(a.inputs):
        before=time.perf_counter();work=(load if a.functional else load_commands)(source)
        functional_seconds+=time.perf_counter()-before;stats=instruction_statistics(work)
        identity={n:digest(source/n) for n in ('input.json',*[f'prog_{sid}.hex' for sid in range(work.config.S)],
            *([f'dram_{sid}.bin' for sid in range(work.config.S)] if a.functional else []))}
        for profile in profiles:
            name=profile['name'];timing=Timing(**profile['timing']);directory=a.output/name/f'{index:04d}'
            directory.mkdir(parents=True)
            physical=replace(work,config=replace(work.config,MCOLS=profile['physical_matrix_columns']))
            native=runner.prepare(physical,timing,directory/'timing.tables')
            cycles,seconds,_,log=native.run();(directory/'lpn.log').write_text(log)
            if native.statistics['hardware_instructions']!=stats:raise ValueError('profile changed instruction work')
            record=dict(profile=name,input=str(source.resolve()),input_index=index,input_sha256=identity,
                original_config=asdict(work.config),physical_config=asdict(physical.config),
                timing=asdict(timing),hardware_instructions=stats,cycles=cycles,native_seconds=seconds,
                assumed_hardware_seconds=cycles/(a.clock_mhz*1e6),
                statistics=native.statistics,native_binary_sha256=digest(native.binary),
                input_tables_sha256=digest(native.inputs))
            records.append(record);atomic_json(directory/'report.json',record)
            atomic_json(a.output/'report.json',report('running'))
        if identity!={n:digest(source/n) for n in identity}:raise ValueError('canonical input changed during sweep')
    if sources!={str(f):digest(f) for f in files}:raise ValueError('sweep source changed')
    result=report('complete');result['totals']=[dict(profile=p['name'],
        cycles=sum(r['cycles'] for r in records if r['profile']==p['name']),
        native_seconds=sum(r['native_seconds'] for r in records if r['profile']==p['name'])) for p in profiles]
    atomic_json(a.output/'report.json',result);print(json.dumps(result['totals'],indent=2),flush=True)


if __name__=='__main__':main()
