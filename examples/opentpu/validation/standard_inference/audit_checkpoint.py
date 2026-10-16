"""Audit a completed standard-inference checkpoint run and replay its final kernel.

Earlier numerical comparisons are the runner's in-run checks. This audit
independently checks their metadata/program identities and replays the last
kernel's numerical execution and native timing from its retained input bundle.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
  with Path(path).open('rb') as f:
    return hashlib.file_digest(f,'sha256').hexdigest()


def require(condition,message):
  if not condition:
    raise RuntimeError(message)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--case',type=Path,required=True)
  parser.add_argument('--project-root',type=Path,required=True)
  parser.add_argument('--opentpu-root',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  project=args.project_root.resolve()
  sys.path.insert(0,str(project))
  from examples.opentpu.workload import use_checkout,load
  from examples.opentpu.model import Timing
  from examples.opentpu.native_timing import ReusableTiming
  from examples.opentpu.e2e import atomic_json
  use_checkout(args.opentpu_root)
  import numpy as np
  from opentpu import rtlsim
  from tokenizers import Tokenizer

  job=json.loads((args.case/'job.json').read_text())
  require(job.get('status')=='complete' and job.get('exit_code')==0,'job is not complete')
  root=args.case/'result'
  report=json.loads((root/'report.json').read_text())
  require(report['status']=='complete' and report['mode']=='validate-rtl','not numerical RTL validation')
  require(report['entrypoint']=='opentpu.llm.qwen3.Engine.generate','unexpected workload entrypoint')
  require(job['source_sha256']==report['project_python_sha256'],'sources changed during run')
  for name,expected in report['project_python_sha256'].items():
    require(sha(project/'examples/opentpu'/name)==expected,'source differs: '+name)
  for name,expected in report['upstream_python_sha256'].items():
    require(sha(args.opentpu_root/name)==expected,'upstream Python differs: '+name)
  for name,expected in report['upstream_rtl_sha256'].items():
    require(sha(args.opentpu_root/name)==expected,'upstream RTL differs: '+name)
  model=Path(report['model_directory'])
  command=job['command']
  def option(name):return command[command.index(name)+1]
  tokenizer=Tokenizer.from_file(str(model/'tokenizer.json'))
  require(tokenizer.encode(option('--prompt')).ids==report['prompt_ids'],'prompt differs')
  require(len(report['prompt_ids'])==report['prompt_tokens'],'prompt length differs')
  require(int(option('--output-tokens'))==report['requested_output_tokens'],'output request differs')
  require(int(option('--depth'))==report['config']['D'],'hardware mapping differs')
  require(tokenizer.decode(report['generated_ids'])==report['generated_text'],'decoded output differs')
  for name,expected in job['checkpoint_sha256'].items():
    require(sha(model/name)==expected,'checkpoint differs: '+name)
  cfg=json.loads((model/'config.json').read_text())
  for field,key in [('hidden','hidden_size'),('layers','num_hidden_layers'),
                    ('ffn','intermediate_size'),('vocab','vocab_size')]:
    require(report['model'][field]==cfg[key],'model shape differs: '+field)
  compiler=Path(job['command'][job['command'].index('--compiler')+1])
  require(sha(compiler)==report['compiler_sha256'],'compiler differs')
  require(sha(compiler.parent/'libLPNNativeRuntime.a')==report['runtime_sha256'],'runtime differs')

  records=report['kernels']
  position=0
  for index,record in enumerate(records):
    require(record['kernel_index']==index and record['step']==index,'kernel numbering differs')
    require(record['position_before']==position,'sequence position differs')
    rows=record['input_rows']
    if record['phase']=='prefill':
      require(position<report['prompt_tokens'],'late prefill')
    else:
      require(record['phase']=='decode' and position>=report['prompt_tokens'] and rows==1,'invalid decode')
    position+=rows
    error=record['lpn_cycles']/record['rtl_cycles']-1
    require(abs(error-record['relative_cycle_error'])<1e-12 and abs(error)<=.05,'cycle error fails')
    for field in ('accuracy_within_5_percent','all_dram_and_tmem_bit_exact','identical_input_files','zero_rollbacks'):
      require(record[field] is True,'failed check: '+field)
    for sid,expected in enumerate(record['program_sha256']):
      encoded=json.loads((root/'kernels'/f'{index:04d}'/f'program_{sid}.json').read_text())
      words=[word for instruction in encoded for word in instruction]
      end=next((i+1 for i in range(len(words)-1,-1,-1) if words[i]),1)
      raw=('\n'.join(f'{word:08x}' for word in words[:end])+'\n').encode()
      observed=hashlib.sha256(raw).hexdigest()
      require(observed==expected==record['input_sha256'][f'prog_{sid}.hex'],'program encoding differs')
  pre=[r for r in records if r['phase']=='prefill']
  require(sum(r['input_rows'] for r in pre)==report['prompt_tokens'],'prompt coverage differs')
  require(any(r['input_rows']>1 for r in pre),'no batched prefill observed')
  require(position==report['final_position'],'final position differs')
  require(len(report['generated_ids'])==report['returned_output_tokens'],'output count differs')
  require(0<report['returned_output_tokens']<=report['requested_output_tokens'],'invalid output count')
  for name in ('rtl_cycles','lpn_cycles','rtl_core_seconds','lpn_core_seconds'):
    require(abs(sum(r[name] for r in records)-report['timing_totals'][name])<1e-8,'total differs: '+name)

  last=records[-1]
  for name,expected in last['input_sha256'].items():
    require(sha(root/'current'/name)==expected,'retained input differs: '+name)
  workload=load(root/'current')
  for sid,state in enumerate(workload.machine.slices):
    dram=np.fromfile(root/'current'/f'dram_out_{sid}.bin',dtype=np.uint8)
    tmem=rtlsim._read_hex(root/'current'/f'tmem_{sid}.hex',workload.config.TMEM_WORDS)
    require(np.array_equal(dram,state.dram) and np.array_equal(tmem,state.tmem),'final numerical replay differs')
  runner=ReusableTiming(compiler,root/'timing-cache',capacity=65536)
  prepared=runner.prepare(workload,Timing(**last['timing']),root/'audit-timing.tables')
  cycles,_,_,_=prepared.run()
  require(cycles==last['lpn_cycles'],'final native replay differs')
  result=dict(status='complete',scope=__doc__,model_directory=str(model),kernels=len(records),
      prompt_tokens=report['prompt_tokens'],output_tokens=report['returned_output_tokens'],
      prefill_rows=[r['input_rows'] for r in pre],
      source_checkpoint_and_program_hashes_match=True,reported_checks_pass=True,
      final_dram_and_tmem_independently_replayed_bit_exact=True,
      final_native_cycles_replayed=cycles,report_sha256=sha(root/'report.json'),
      audit_source_sha256=sha(__file__),timing_totals=report['timing_totals'])
  result['matched_source_revision']=subprocess.check_output(
      ['git','rev-parse','HEAD'],cwd=project,text=True).strip()
  atomic_json(args.output,result)
  print(json.dumps(result,indent=2))


if __name__=='__main__':main()
