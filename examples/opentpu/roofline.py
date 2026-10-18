"""Necessary dense-MAC and uncached-weight bandwidth budgets, not timing predictions.

Weight bytes exclude scale metadata, DMA, KV writes, burst rounding and latency,
so this is optimistic for a design that fetches each MM's weights from HBM.
Caching or lossless compression can change that traffic assumption.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from math import ceil
from pathlib import Path


def kernel_demands(task):
  from .workload import use_checkout
  from .shape_workload import resolve_static
  from .retained_inputs import verified_program
  from .e2e import digest
  use_checkout(Path(task['opentpu_root']))
  from opentpu import Config,isa as I
  cfg=Config(**task['config']);source=Path(task['source']);programs=[];identities={}
  for sid in range(cfg.S):
    if digest(source/f'prog_{sid}.hex')!=task['record']['input_sha256'][f'prog_{sid}.hex']:
      raise ValueError('canonical program changed')
    encoded,identity=verified_program(source,sid)
    programs.append([I.Instr.decode(words) for words in encoded]);identities[str(sid)]=identity
  work=resolve_static(cfg,programs,task['record']['args'])
  macs=weight_bytes=commands=0;formats={}
  for items in work.commands:
    for c in items:
      ins=c.instruction
      if ins.op!=I.MM:continue
      w=ins.w;n,kb,m=w[3]&65535,(w[3]>>16)&255,(w[5]>>16)&255
      wf=(ins.flags>>I.WF_SHIFT)&3
      size=n*kb*(cfg.D if wf==I.W8 else cfg.D//2)
      weight_bytes+=size;macs+=n*m*kb*cfg.D;commands+=1
      formats[str(wf)]=formats.get(str(wf),0)+size
  return dict(kernel=task['record']['step'],phase=task['record']['phase'],dense_macs=macs,
      weight_only_uncached_bytes=weight_bytes,matrix_commands=commands,
      weight_bytes_by_format=formats,program_json_sha256=identities)


def main():
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('case','opentpu-root','output'):p.add_argument('--'+name,type=Path,required=True)
  p.add_argument('--bytes-per-cycle',type=int,default=2816)
  p.add_argument('--mac-lanes',type=int,default=8678)
  p.add_argument('--clock-mhz',type=float,default=200)
  p.add_argument('--workers',type=int,default=8)
  a=p.parse_args()
  if min(a.bytes_per_cycle,a.mac_lanes,a.clock_mhz,a.workers)<=0:p.error('positive budgets required')
  from .e2e import atomic_json,digest
  reference=json.loads((a.case/'report.json').read_text())
  if (reference['status']!='complete' or reference['prompt_tokens']!=128 or reference['output_tokens']!=32
      or reference['entrypoint']!='unmodified upstream Engine.generate'
      or not reference['all_dram_and_tmem_bit_exact'] or not reference['all_inputs_identical_to_baseline']):
    raise ValueError('completed verified normal inference required')
  files=[Path(__file__).with_name(n) for n in ('roofline.py','shape_workload.py','retained_inputs.py','workload.py')]
  sources={str(f):digest(f) for f in files}
  tasks=[dict(record=record,config=reference['config'],source=str(a.case/'kernels'/f'{record["step"]:04d}'),
              opentpu_root=str(a.opentpu_root.resolve())) for record in reference['kernels']]
  with ProcessPoolExecutor(max_workers=a.workers) as pool:records=list(pool.map(kernel_demands,tasks))
  if len(records)!=48 or [r['kernel'] for r in records]!=list(range(48)):raise ValueError('full inference required')
  macs=sum(r['dense_macs'] for r in records);traffic=sum(r['weight_only_uncached_bytes'] for r in records)
  original=reference['baseline_rtl_cycles'];target=original/100
  if sources!={str(f):digest(f) for f in files}:raise RuntimeError('analysis sources changed')
  result=dict(status='complete',scope=__doc__,records=records,reference_report_sha256=digest(a.case/'report.json'),
      source_sha256=sources,original_rtl_cycles=original,target_cycles=target,dense_macs=macs,
      weight_only_uncached_bytes=traffic,mac_lanes=a.mac_lanes,bytes_per_cycle=a.bytes_per_cycle,
      assumed_clock_mhz=a.clock_mhz,dense_compute_floor_cycles=ceil(macs/a.mac_lanes),
      uncached_weight_only_floor_cycles=ceil(traffic/a.bytes_per_cycle),
      dense_compute_speedup_ceiling=original/(macs/a.mac_lanes),
      uncached_weight_only_speedup_ceiling=original/(traffic/a.bytes_per_cycle),
      minimum_mac_lanes_for_100x=ceil(macs/target),
      minimum_uncached_weight_bytes_per_cycle_for_100x=traffic/target,
      minimum_uncached_weight_gbs_for_100x=traffic/target*a.clock_mhz/1000,
      numerical_arithmetic_executed=False,hardware_implemented=False,
      limitations=['Necessary budgets only; costs overlap and are not added',
                   'No proof of realizable Fmax, DSP packing, utilization or memory efficiency',
                   'Uncached traffic bound changes if a coherent cache or lossless memory compression is implemented'])
  atomic_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
