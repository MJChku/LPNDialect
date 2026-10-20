"""Run the actual openTPU component LPN directly from canonical RTL commands.

No tensor-DAG input or numerical matrix execution. The ISA and timing components
are those used by the existing openTPU validation path. New timing parameters
remain proposals until validated against corresponding RTL.
"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import time

from .e2e import atomic_json,digest
from .hardware_commands import load_commands
from .model import Timing
from .native_timing import ReusableTiming
from .workload import use_checkout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','opentpu-root','compiler','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--timing',type=Path,help='Explicit Timing fields; defaults use original hardware timing')
    p.add_argument('--physical-matrix-columns',type=int,choices=(1,2,4,8),help='Physical MXU_COLS override, as in the RTL microarchitecture')
    p.add_argument('--physical-vector-lanes',type=int,choices=(8,16,32,64),help='Physical TMEM/VPU lane count override')
    p.add_argument('--clock-mhz',type=float,required=True,help='Assumed hardware clock, not host speed or achieved Fmax')
    p.add_argument('--runs',type=int,default=3);a=p.parse_args()
    if a.runs<1 or not 0<a.clock_mhz<1e6:p.error('positive repeat count and finite clock required')
    use_checkout(a.opentpu_root)
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();work=load_commands(a.input)
    input_config=asdict(work.config)
    overrides={k:v for k,v in [('MCOLS',a.physical_matrix_columns),('LANES',a.physical_vector_lanes)] if v is not None}
    if any(v<1 for v in overrides.values()):p.error('positive physical dimensions required')
    if overrides:work=replace(work,config=replace(work.config,**overrides))
    if work.config.LANES not in (8,16,32,64):
        raise ValueError('the existing physical VPU/TMEM model supports 8/16/32/64 lanes; wider elementwise proposals retain 64 reduction lanes')
    timing=Timing(**json.loads(a.timing.read_text())) if a.timing else Timing(dram_bytes_per_cycle=min(work.config.D,128))
    files=[Path(__file__),Path(__file__).with_name('hardware_commands.py'),
        *[Path(__file__).with_name(n) for n in ('model.py','compact.py','components.py','shape_workload.py',
            'native_timing.py','workload.py','benchmark.py','activation_stream.py','scale_cache_timing.py','vector_pool.py','vector_select.py')]]
    sources={str(f):digest(f) for f in files}
    runner=ReusableTiming(a.compiler,a.output/'native-cache',capacity=max(len(c) for c in work.commands))
    prepare=runner.prepare(
        work,timing,a.output/'timing.tables')
    samples=[]
    for index in range(a.runs):
        cycles,core,process,log=prepare.run();(a.output/f'native-{index}.log').write_text(log)
        samples.append(dict(cycles=cycles,simulation_seconds=core,process_seconds=process))
    if len({s['cycles'] for s in samples})!=1:raise ValueError('nonrepeatable hardware-command replay')
    cycles=samples[0]['cycles'];seconds=cycles/(a.clock_mhz*1e6)
    arithmetic=prepare.statistics['hardware_instructions']
    if sources!={str(f):digest(f) for f in files}:raise ValueError('timing sources changed during replay')
    atomic_json(a.output/'report.json',dict(status='complete',scope=__doc__,
        input_directory=str(a.input.resolve()),input_sha256=work.hashes,input_config=input_config,
        hardware_config=asdict(work.config),explicit_physical_overrides=overrides,
        timing=asdict(timing),assumed_clock_mhz=a.clock_mhz,cycles=cycles,
        assumed_hardware_seconds=seconds,statistics=prepare.statistics,
        achieved_matrix_dot_tops=arithmetic['matrix_dot_operations']/seconds/1e12,
        achieved_logical_fp32_vector_gflops=arithmetic['logical_fp32_vector_flops']/seconds/1e9,
        samples=samples,preparation_seconds=prepare.preparation_seconds,compile_seconds=prepare.compile_seconds,
        host_total_seconds=time.perf_counter()-started,source_sha256=sources,
        binary=str(prepare.binary),binary_sha256=digest(prepare.binary),
        compiler_runtime_identity=runner.compiler_identity,
        tables_sha256=digest(prepare.inputs),tensor_dag_used=False,weights_loaded=False,
        numerical_arithmetic_executed=False,zero_rollbacks=True,
        architecture_rtl_validated_by_this_run=False,
        dependency_source='Actual resolved hardware instruction memory footprints; LPN resources supply component timing'))
    print(json.dumps(dict(cycles=cycles,simulation_seconds=[s['simulation_seconds'] for s in samples],
        matrix_dot_operations=arithmetic['matrix_dot_operations'],statistics=prepare.statistics)),flush=True)


if __name__=='__main__':main()
