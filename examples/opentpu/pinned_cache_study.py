"""Bounded persistent weight-region cache traffic on unchanged inference inputs.

This is a traffic study, not LPN timing or implemented RTL. The region starts
cold. Every changed host input line is invalidated, as is every line potentially
written by the current kernel. Cold lines become valid only for the next kernel,
so no intra-kernel fill timing or ordering is assumed.
"""
import argparse
import json
from pathlib import Path
import time


def study_kernel(cfg,commands,image,previous,states,base,I,W):
  import numpy as np
  maximum=max(len(valid) for valid in states.values())
  current=image[base:base+64*maximum].reshape(-1,64)
  changed=np.ones(maximum,dtype=bool) if previous is None else np.any(current!=previous,axis=1)
  written=np.zeros(maximum+1,dtype=np.int64)
  reads=np.zeros(maximum+1,dtype=np.int64)
  requests=[];total=0
  for c in commands:
    for space,lo,hi in c.footprint.wr:
      if space==W.SP_DRAM:
        begin=max(0,min(maximum,(lo-base)//64));end=max(0,min(maximum,(hi-base+63)//64))
        if begin<end:written[begin]+=1;written[end]-=1
    ins=c.instruction
    if ins.op!=I.MM:continue
    if ((ins.flags>>I.WF_SHIFT)&3)!=I.W8:raise ValueError('current study requires int8 MM weights')
    w=ins.w;n,kb=w[3]&65535,(w[3]>>16)&255;size=kb*cfg.D
    if w[0]%64 or w[4]%64 or size%64:raise ValueError('aligned MM weight blocks required')
    addresses=w[0]+np.arange(n,dtype=np.int64)*w[4]
    begin=np.clip((addresses-base)//64,0,maximum);end=np.clip((addresses+size-base)//64,0,maximum)
    requests.append((begin,end));total+=n*size
    np.add.at(reads,begin,1);np.add.at(reads,end,-1)
  forbidden=np.cumsum(written[:-1])>0;fetched=np.cumsum(reads[:-1])>0
  result={}
  for size,valid in states.items():
    n=len(valid);invalidated=int(np.count_nonzero(valid & (changed[:n]|forbidden[:n])))
    valid &= ~(changed[:n]|forbidden[:n])
    prefix=np.empty(n+1,dtype=np.int64);prefix[0]=0;np.cumsum(valid,out=prefix[1:])
    hits=sum(int((prefix[np.minimum(end,n)]-prefix[np.minimum(begin,n)]).sum())*64 for begin,end in requests)
    result[str(size)]=dict(requested_weight_bytes=total,cache_hit_bytes=hits,
        hbm_weight_bytes=total-hits,invalidated_valid_lines=invalidated,
        valid_lines_at_start=int(valid.sum()),cold=True if previous is None else False)
    valid |= fetched[:n]&~forbidden[:n]
  return result,current.copy()


def main():
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('case','opentpu-root','output'):p.add_argument('--'+name,type=Path,required=True)
  p.add_argument('--base',type=lambda x:int(x,0),default=2<<20)
  p.add_argument('--mib',default='8,16,24,32,44,48,64')
  a=p.parse_args()
  import numpy as np
  from .e2e import atomic_json,digest
  from .workload import use_checkout
  from .shape_workload import resolve_static
  from .retained_inputs import verified_program
  from .weight_storage import read_image
  use_checkout(a.opentpu_root.resolve())
  from opentpu import Config,isa as I,waits as W
  reference=json.loads((a.case/'report.json').read_text())
  if (reference['status']!='complete' or reference['prompt_tokens']!=128 or reference['output_tokens']!=32
      or reference['entrypoint']!='unmodified upstream Engine.generate'
      or not reference['all_dram_and_tmem_bit_exact'] or not reference['all_inputs_identical_to_baseline']):
    raise ValueError('completed verified normal inference required')
  cfg=Config(**reference['config']);sizes=sorted(set(int(x)*(1<<20) for x in a.mib.split(',')))
  if cfg.S!=1 or a.base<0 or a.base%64 or not sizes or sizes[0]<=0 or a.base+sizes[-1]>cfg.DRAM_BYTES:
    p.error('one slice, aligned valid region and positive bounded sizes required')
  files=[Path(__file__).with_name(n) for n in ('pinned_cache_study.py','weight_storage.py','shape_workload.py','retained_inputs.py')]
  hashes={str(f):digest(f) for f in files};states={size:np.zeros(size//64,dtype=bool) for size in sizes}
  previous=None;records=[];started=time.perf_counter()
  for record in reference['kernels']:
    index=record['step'];source=a.case/'kernels'/f'{index:04d}'
    for name in ('prog_0.hex','dram_0.bin'):
      if digest(source/name)!=record['input_sha256'][name]:raise ValueError('retained input changed')
    encoded,identity=verified_program(source,0)
    work=resolve_static(cfg,[[I.Instr.decode(words) for words in encoded]],record['args'])
    if [len(c) for c in work.commands]!=record['instructions']:raise ValueError('instruction count differs from verified RTL')
    observations,previous=study_kernel(cfg,work.commands[0],read_image(source/'dram_0.bin',cfg.DRAM_BYTES),previous,states,a.base,I,W)
    records.append(dict(kernel=index,phase=record['phase'],program_json_sha256=identity,
                        input_sha256=record['input_sha256'],caches=observations))
    print(json.dumps(dict(kernel=index,caches=observations)),flush=True)
  if len(records)!=48 or [r['kernel'] for r in records]!=list(range(48)):raise ValueError('full inference required')
  if hashes!={str(f):digest(f) for f in files}:raise RuntimeError('study sources changed')
  original=reference['baseline_rtl_cycles'];points=[]
  for size in sizes:
    hits=sum(r['caches'][str(size)]['cache_hit_bytes'] for r in records)
    misses=sum(r['caches'][str(size)]['hbm_weight_bytes'] for r in records)
    points.append(dict(cache_bytes=size,valid_bitmap_bytes=size//64//8,cache_hit_bytes=hits,
        hbm_weight_bytes=misses,weight_hit_fraction=hits/(hits+misses),
        optimistic_hbm_weight_floor_cycles=misses/2816,
        optimistic_hbm_weight_speedup_ceiling=original/(misses/2816),
        optimistic_parallel_cache_read_floor_cycles=hits/8192,
        assumed_hbm_bytes_per_cycle=2816,assumed_cache_read_bytes_per_cycle=8192))
  atomic_json(a.output,dict(status='complete',scope=__doc__,base=a.base,records=records,points=points,
      source_sha256=hashes,reference_report_sha256=digest(a.case/'report.json'),
      seconds=time.perf_counter()-started,numerical_arithmetic_executed=False,hardware_implemented=False,
      limitations=['Weight traffic only, not a cycle prediction or hardware speedup',
          'Cold lines are unavailable for the rest of that kernel, conservatively',
          'Every changed input line and potentially written line invalidates prior cache state',
          'Data/valid storage is counted; bank conflicts, control, mapping and Fmax remain unproven',
          'Independent SRAM and HBM read ports are assumed by separate floor calculations']))


if __name__=='__main__':main()
