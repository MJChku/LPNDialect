"""Measure LPN-only generation from an RTL-validated report and verify every step.

Full pipeline timers and complete process wall time are reported separately.
No RTL timestamps enter the timing model; they are used only for comparison.
"""
def main():
  import argparse
  from pathlib import Path
  from dataclasses import asdict
  import json,hashlib,re,time,subprocess
  from types import SimpleNamespace

  p=argparse.ArgumentParser()
  p.add_argument('--baseline',type=Path,required=True)
  p.add_argument('--compiler',type=Path,required=True)
  p.add_argument('--mode',default='heartbeat')
  p.add_argument('--opentpu-root',type=Path,required=True)
  p.add_argument('--model-dir',type=Path)
  p.add_argument('--allow-partial',action='store_true',help='Explicitly replay a snapshot of an unfinished generation case')
  p.add_argument('--output',type=Path,required=True)
  a=p.parse_args()
  session_start=time.perf_counter()
  baseline_path=a.baseline
  baseline=json.loads(baseline_path.read_text())
  expected=baseline['records']
  if not expected:p.error('baseline has no observed steps')
  if baseline.get('status') not in ('complete','eos') and not a.allow_partial:
   p.error('baseline generation is unfinished; use --allow-partial to replay only its observed snapshot')
  if any(not r.get('all_dram_and_tmem_bit_exact') or not r.get('identical_input_files') for r in expected):
   p.error('baseline must have independent bit-exact ISA/RTL validation on identical inputs for every step')
  from examples.opentpu.workload import use_checkout,execute
  from examples.opentpu.e2e import CheckedBackend,atomic_json,digest
  from examples.opentpu.benchmark import write_inputs
  from examples.opentpu.native_timing import ReusableTiming
  from examples.opentpu.model import Timing
  use_checkout(a.opentpu_root)
  import numpy as np
  import torch
  from opentpu.llm import load_spec
  from opentpu.llm.qwen3 import Engine,load_weights,device_config
  torch.set_num_threads(1)
  model_dir=a.model_dir or Path(baseline['identity']['model_directory'])
  for name,value in baseline['identity']['checkpoint_sha256'].items():
   assert digest(model_dir/name)==value,('checkpoint changed',name)
  spec=load_spec(model_dir)
  cfg=device_config(spec,baseline['identity']['cap'],**{
   key:baseline['identity']['config'][key] for key in
   ['S','D','MCOLS','ACT_ROWS','ACT_BLOCKS','LANES','IMEM_WORDS','TMEM_WORDS']})
  assert asdict(cfg)==baseline['identity']['config']
  a.output.mkdir(parents=True,exist_ok=True)
  options=SimpleNamespace(output=a.output,lpn_compiler=a.compiler.resolve())

  class ReplayBackend(CheckedBackend):
   def __init__(self,cfg,images,options):
    super().__init__(cfg,images,options)
    self.timing_runner=ReusableTiming(options.lpn_compiler,options.output/'timing-cache',capacity=baseline['identity'].get('command_capacity',8192))

   def run(self,programs,args=None):
    r=expected[len(self.records)]
    current=a.output/'current'
    before=time.perf_counter()
    write_inputs(current,self.cfg,programs,self.drams,args)
    serialization=time.perf_counter()-before
    before=time.perf_counter()
    hashes={name:digest(current/name) for s in range(self.cfg.S)
            for name in (f'prog_{s}.hex',f'dram_{s}.bin')}
    hash_seconds=time.perf_counter()-before
    assert hashes==r['input_sha256'],('input differs',r['position'],hashes,r['input_sha256'])
    before=time.perf_counter()
    workload=execute(self.cfg,programs,self.drams,args,hashes)
    frontend=time.perf_counter()-before
    assert [len(c) for c in workload.commands]==r['instructions']
    timing=Timing(**r['timing'])
    prepared=self.timing_runner.prepare(workload,timing,current/'timing.tables')
    cycles,core,process,log=prepared.run()
    (current/'lpn.log').write_text(log)
    assert cycles==r['lpn_cycles'],('predicted cycles changed',r['position'],cycles,r['lpn_cycles'])
    counters=re.search(r'Guard checks: (\d+); body executions: (\d+); rollbacks: (\d+)',log)
    assert counters and int(counters[3])==0
    hit=re.search(r'Guard cache hits: (\d+); dependency reads: (\d+)',log)
    record=dict(position=r['position'],phase=r['phase'],input_token=r['input_token'],
      instructions=r['instructions'],input_sha256=hashes,identical_input_files=True,
      lpn_cycles=cycles,baseline_lpn_cycles=r['lpn_cycles'],rtl_cycles=r['rtl_cycles'],
      relative_cycle_error=cycles/r['rtl_cycles']-1,
      functional_frontend_seconds=frontend,model_build_seconds=prepared.preparation_seconds,
      native_compile_seconds=prepared.compile_seconds,lpn_core_seconds=core,
      lpn_process_seconds=process,single_use_lpn_seconds=frontend+prepared.preparation_seconds+prepared.compile_seconds+process,
      input_serialization_seconds=serialization,input_hash_seconds=hash_seconds,
      timing_cache_hit=prepared.cache_hit,timing_net=prepared.statistics,
      guard_evaluations=int(counters[1]),body_executions=int(counters[2]),rollbacks=int(counters[3]),
      guard_cache_hits=int(hit[1]) if hit else None)
    self.records.append(record)
    # The original RTL validation established bit-exact ISA/RTL output memory.
    # Feeding the independent ISA result forward regenerates each next input;
    # hashes above verify this against the original RTL path on every step.
    self.drams=[s.dram[:len(im)].copy() for s,im in zip(workload.machine.slices,self.drams)]
    return record

  before=time.perf_counter()
  engine=Engine(spec,load_weights(model_dir),baseline['identity']['cap'],cfg,
   backend=lambda c,im:ReplayBackend(c,im,options),pipeline=False,resident=False)
  engine_init=time.perf_counter()-before
  print(json.dumps({'mode':a.mode,'stage':'engine_ready','engine_init_seconds':engine_init}),flush=True)
  workload_start=time.perf_counter()
  for reference in expected:
   assert engine.pos==reference['position']
   if reference['phase']=='prefill':
    assert reference['input_token']==baseline['identity']['prompt_ids'][engine.pos]
   else:
    assert reference['input_token']==int(np.argmax(logits)),('generated token changed',engine.pos)
   before=time.perf_counter()
   logits=engine.step(reference['input_token'])
   step_seconds=time.perf_counter()-before
   logit_hash=hashlib.sha256(logits.tobytes()).hexdigest()
   assert logit_hash==reference['logits_sha256'],('logits differ',reference['position'])
   assert int(np.argmax(logits))==reference['next_token']
   record=engine.backend.records[-1]
   record.update(total_step_seconds=step_seconds,logits_sha256=logit_hash,
     next_token=int(np.argmax(logits)),logits_match_baseline=True)
   atomic_json(a.output/'progress.json',{'mode':a.mode,'completed':engine.pos,'records':engine.backend.records})
   print(json.dumps({'mode':a.mode,'completed':engine.pos,'full_lpn_seconds':record['single_use_lpn_seconds'],
     'total_step_seconds':step_seconds,'core_seconds':record['lpn_core_seconds'],
     'cycles':record['lpn_cycles'],'cache_hit':record['timing_cache_hit']}),flush=True)
  workload_elapsed=time.perf_counter()-workload_start
  records=engine.backend.records
  report={'mode':a.mode,'status':'complete','steps':len(records),'workload':f'Exactly {len(expected)} recorded {model_dir.name} steps; persistent KV state, all transformer layers and vocabulary logits',
   'baseline_generation_status':baseline.get('status'),
   'complete_generation_replayed':baseline.get('status') in ('complete','eos'),
   'generated_tokens':[r['input_token'] for r in records if r['phase']=='decode'],
   'replay_source_sha256':digest(Path(__file__)),
   'comparison':'Original recorded RTL timings; RTL was not rerun. Inputs, logits and LPN predictions checked against every saved step.',
   'baseline_report_sha256':digest(baseline_path),'baseline_identity':baseline['identity'],
   'compiler_sha256':digest(a.compiler),'runtime_sha256':digest(a.compiler.parent/'libLPNNativeRuntime.a'),
   'engine_init_seconds':engine_init,'workload_elapsed_seconds':workload_elapsed,
   'session_seconds':time.perf_counter()-session_start,
   'records':records}
  for key in ['functional_frontend_seconds','model_build_seconds','native_compile_seconds','lpn_core_seconds',
   'lpn_process_seconds','single_use_lpn_seconds','total_step_seconds','input_serialization_seconds','input_hash_seconds',
   'guard_evaluations','body_executions','rollbacks','lpn_cycles','rtl_cycles']:
   report[key]=sum(r[key] for r in records)
  report.update(recorded_rtl_core_seconds=baseline['rtl_core_seconds'],
   recorded_rtl_process_seconds=sum(r['rtl_process_seconds'] for r in expected),
   recorded_lpn_accounted_seconds=baseline['single_use_lpn_seconds'],
   recorded_lpn_core_seconds=baseline['lpn_core_seconds'],
   all_inputs_identical=True,all_logits_identical=True,all_predicted_cycles_identical=True,
   all_steps_within_5_percent=all(abs(r['relative_cycle_error'])<=.05 for r in records),
   compilation_misses=sum(not r['timing_cache_hit'] for r in records))
  report['full_lpn_speedup_vs_recorded_lpn']=report['recorded_lpn_accounted_seconds']/report['single_use_lpn_seconds']
  report['full_lpn_speedup_vs_recorded_rtl_core']=report['recorded_rtl_core_seconds']/report['single_use_lpn_seconds']
  report['core_speedup_vs_recorded_rtl_core']=report['recorded_rtl_core_seconds']/report['lpn_core_seconds']
  assert report['generated_tokens']==baseline['generated_tokens'],'generated sequence differs'
  report['complete_replay_speedup_vs_recorded_rtl_core']=report['recorded_rtl_core_seconds']/report['session_seconds']
  atomic_json(a.output/'report.json',report)
  print(json.dumps({k:v for k,v in report.items() if k not in ['records','baseline_identity']},indent=2),flush=True)

if __name__ == '__main__':
  main()
