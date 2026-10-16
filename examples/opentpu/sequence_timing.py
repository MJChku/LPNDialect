"""Resumable, weight-free timing for every position of a dense Qwen sequence.

Each upstream kernel drains at HALT. Independent prescribed-position simulations
can run in separate host processes; hardware sequence cycles are their sum.
The LPN scheduler itself remains sequential. No numerical inference is performed.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from .fpga_dse import architecture, measure, resources, V80_SOURCE
from .model import Timing
from .native_timing import ReusableTiming
from .shape_workload import compile_shape, program_hash, qwen17_spec
from .workload import use_checkout


def digest(path):
  with Path(path).open('rb') as stream:
    return hashlib.file_digest(stream,'sha256').hexdigest()


def atomic_json(path,value):
  path=Path(path)
  path.parent.mkdir(parents=True,exist_ok=True)
  with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as stream:
    temporary=Path(stream.name)
    try:
      json.dump(value,stream,indent=2)
      stream.write('\n')
      stream.flush()
      os.fsync(stream.fileno())
    except BaseException:
      temporary.unlink(missing_ok=True)
      raise
  temporary.replace(path)


def initialize(identity,output):
  global IDENTITY,ROOT,RUNNER,SPEC,IDENTITY_HASH
  IDENTITY,ROOT=identity,Path(output)
  use_checkout(identity['opentpu_root'])
  SPEC=qwen17_spec()
  RUNNER=ReusableTiming(identity['compiler'],ROOT/'cache',capacity=32768)
  IDENTITY_HASH=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def simulate(position):
  start=time.perf_counter()
  workload,image=compile_shape(SPEC,position,IDENTITY['cap'])
  shape_seconds=time.perf_counter()-start
  scratch=ROOT/'workers'/str(os.getpid())
  results={}
  for name,rows in [('original',1),('rows32',32),('rows64',64)]:
    if name=='original':
      cfg,timing=workload.config,Timing(batch_rows=8)
    else:
      cfg=replace(workload.config,MCOLS=1,LANES=64)
      timing=architecture((name,rows,1,64,2816),IDENTITY['memory_latency'])
    results[name]=measure(RUNNER,replace(workload,config=cfg),timing,scratch/name)
  result=dict(status='complete',identity_sha256=IDENTITY_HASH,position=position,
      phase='prefill' if position<IDENTITY['prompt_tokens'] else 'decode',
      shape_seconds=shape_seconds,host_step_seconds=time.perf_counter()-start,
      program_sha256=program_hash(workload.programs[0]),
      image_bytes=image.nbytes,instructions=len(workload.commands[0]),architectures=results)
  atomic_json(ROOT/'steps'/f'{position:05d}.json',result)
  return result


def aggregate(records,identity):
  architectures={}
  for name in ('original','rows32','rows64'):
    phases={phase:sum(r['architectures'][name]['cycles'] for r in records if r['phase']==phase)
            for phase in ('prefill','decode')}
    cycles=sum(phases.values())
    architectures[name]=dict(cycles=cycles,phase_cycles=phases,
        hardware_seconds_at_assumed_clock=cycles/(identity['clock_mhz']*1e6),
        native_seconds=sum(r['architectures'][name]['native_seconds'] for r in records),
        summed_host_model_seconds=sum(r['architectures'][name]['host_total_seconds'] for r in records),
        rollbacks=sum(r['architectures'][name]['rollbacks'] for r in records))
  base=architectures['original']
  for name,value in architectures.items():
    value['cycle_speedup']=base['cycles']/value['cycles'] if value['cycles'] else None
    value['phase_cycle_speedup']={phase:base['phase_cycles'][phase]/cycles if cycles else None
                                for phase,cycles in value['phase_cycles'].items()}
  return architectures


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--opentpu-root',type=Path,required=True)
  parser.add_argument('--compiler',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--prompt-tokens',type=int,default=1024)
  parser.add_argument('--output-tokens',type=int,default=8)
  parser.add_argument('--workers',type=int,default=8)
  parser.add_argument('--clock-mhz',type=float,default=200)
  parser.add_argument('--memory-latency',type=int,default=64)
  args=parser.parse_args()
  if min(args.prompt_tokens,args.output_tokens,args.workers,args.clock_mhz,args.memory_latency)<1:
    parser.error('lengths, workers, clock and latency must be positive')
  args.opentpu_root=args.opentpu_root.resolve()
  args.compiler=args.compiler.resolve()
  args.output=args.output.resolve()
  use_checkout(args.opentpu_root)
  nsteps=args.prompt_tokens+args.output_tokens
  cap=((nsteps+127)//128)*128
  source=Path(__file__).parent
  identity=dict(model=asdict(qwen17_spec()),prompt_tokens=args.prompt_tokens,
      output_tokens=args.output_tokens,cap=cap,clock_mhz=args.clock_mhz,
      memory_latency=args.memory_latency,opentpu_root=str(args.opentpu_root),
      compiler=str(args.compiler),compiler_sha256=digest(args.compiler),
      runtime_sha256=digest(args.compiler.parent/'libLPNNativeRuntime.a'),
      upstream_python_sha256={str(p.relative_to(args.opentpu_root)):digest(p)
        for p in sorted((args.opentpu_root/'opentpu').rglob('*.py'))},
      source_sha256={name:digest(source/name) for name in
        ['sequence_timing.py','fpga_dse.py','shape_workload.py','model.py','compact.py',
         'components.py','native_timing.py','workload.py','benchmark.py']})
  workload,image=compile_shape(qwen17_spec(),0,cap)
  budget=SimpleNamespace(clock_mhz=args.clock_mhz,macs_per_dsp=1,
      dsp_fraction=.8,sram_fraction=.5,hbm_efficiency=.7)
  identity['resource_assumptions']=dict(board_reference=V80_SOURCE,
      peak_hbm_gbs=810,hbm_efficiency=.7,dsp_slices=10848,dsp_fraction=.8,
      embedded_memory_bits=673_000_000,sram_fraction=.5,macs_per_dsp=1)
  identity['architectures']={}
  for name,rows in [('original',1),('rows32',32),('rows64',64)]:
    columns,lanes,bw=(4,8,128) if name=='original' else (1,64,2816)
    point=('validated_architecture' if name=='original' else name,rows,1,lanes,bw)
    profile=resources(replace(workload.config,MCOLS=columns),image,point,budget)
    if not profile['passes_resource_screen']:
      parser.error(f'{name} exceeds resource assumptions: {profile["rejection_reasons"]}')
    identity['architectures'][name]=dict(matrix_columns=columns,parallel_output_rows=rows,
        vector_lanes=lanes,timing=asdict(architecture(point,args.memory_latency)),resources=profile)
  # Normalize tuples and JSON types once so a saved manifest compares on resume.
  identity=json.loads(json.dumps(identity))
  identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
  args.output.mkdir(parents=True,exist_ok=True)
  with (args.output/'run.lock').open('w') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:parser.error('another sequence run owns this output directory')
    manifest=args.output/'manifest.json'
    if manifest.exists():
      if json.loads(manifest.read_text())!=identity:
        parser.error('resume identity changed; choose a new output directory')
    else:atomic_json(manifest,identity)
    records={}
    for path in sorted((args.output/'steps').glob('*.json')):
      result=json.loads(path.read_text())
      pos=result['position']
      if (result['status']!='complete' or result['identity_sha256']!=identity_hash
          or not 0<=pos<nsteps or pos in records
          or path.stem!=f'{pos:05d}'
          or result['phase']!=('prefill' if pos<args.prompt_tokens else 'decode')
          or set(result['architectures'])!={'original','rows32','rows64'}
          or any(r['rollbacks']!=0 or r['cycles']<=0 for r in result['architectures'].values())):
        parser.error(f'invalid checkpoint: {path}')
      records[pos]=result
    started=time.perf_counter()
    progress=dict(status='running',pid=os.getpid(),workers=args.workers,
        expected_steps=nsteps,completed_steps=len(records),identity_sha256=identity_hash)
    atomic_json(args.output/'progress.json',progress)
    pending=[p for p in range(nsteps) if p not in records]
    # Spawn avoids inheriting imported simulator state or native runtime bindings.
    with ProcessPoolExecutor(max_workers=args.workers,
        mp_context=multiprocessing.get_context('spawn'),initializer=initialize,
        initargs=(identity,str(args.output))) as pool:
      futures={pool.submit(simulate,p):p for p in pending}
      try:
        for future in as_completed(futures):
          result=future.result()
          records[result['position']]=result
          progress.update(completed_steps=len(records),elapsed_this_session_seconds=time.perf_counter()-started)
          atomic_json(args.output/'progress.json',progress)
          print(json.dumps(dict(completed_steps=len(records),expected_steps=nsteps,
            position=result['position'],elapsed_seconds=progress['elapsed_this_session_seconds'])),flush=True)
      except BaseException as error:
        for future in futures:future.cancel()
        progress.update(status='failed',error=str(error))
        atomic_json(args.output/'progress.json',progress)
        raise
    if sorted(records)!=list(range(nsteps)):raise RuntimeError('sequence positions are missing')
    ordered=[records[p] for p in range(nsteps)]
    # Verify original-model anchors where the saved capacity/layout is identical.
    baseline=json.loads((source/'validation/qwen17/completed_fibonacci.json').read_text())['baseline']
    reference={r['position']:r for r in baseline['records']}
    checked=[]
    if cap==128:
      for result in ordered:
        if result['position'] in reference:
          r=reference[result['position']]
          if result['program_sha256']!=r['input_sha256']['prog_0.hex'] or result['architectures']['original']['cycles']!=r['lpn_cycles']:
            raise RuntimeError('original shape timing differs from validated baseline')
          checked.append(result['position'])
    report=dict(status='complete',identity=identity,identity_sha256=identity_hash,
        scope='Every position of a prescribed-length, weight-free Qwen3-1.7B sequence. Prefill is token-by-token. Hardware kernels drain at HALT, so sequence cycles sum per-position runs. No weights, tensor arithmetic, generated text or EOS. Modified FPGA architectures and larger KV capacities are not RTL-validated.',
        token_steps=nsteps,prompt_tokens=args.prompt_tokens,output_tokens=args.output_tokens,
        workers=args.workers,positions_verified_contiguous=True,
        recorded_program_and_cycle_positions_checked=checked,
        host_session_seconds=time.perf_counter()-started,
        host_session_scope='This invocation, including worker startup and parallel execution; resumed checkpoints retain prior stage timings. Summed worker times are not wall makespan.',
        summed_shape_seconds=sum(r['shape_seconds'] for r in ordered),
        architectures=aggregate(ordered,identity),records=ordered)
    atomic_json(args.output/'report.json',report)
    progress.update(status='complete',elapsed_this_session_seconds=time.perf_counter()-started)
    atomic_json(args.output/'progress.json',progress)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','identity')},indent=2),flush=True)


if __name__=='__main__':main()
