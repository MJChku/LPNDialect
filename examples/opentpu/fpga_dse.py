"""Weight-free Qwen3 architecture exploration with the event-driven LPN.

Cycle reductions describe hypothetical datapaths, not simulator wall speedup
or synthesis-proven FPGA performance. No checkpoint, tokenizer or GEMM runs.
"""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import time

from .workload import use_checkout
from .shape_workload import compile_shape, program_hash, qwen17_spec
from .native_timing import ReusableTiming
from .model import Timing


V80_SOURCE='https://www.amd.com/content/dam/amd/en/documents/products/accelerators/alveo/v80/alveo-v80-product-brief.pdf'


def candidates():
  # Mechanism ablations and balanced points, rather than an unbounded grid.
  points=[('validated_architecture',1,1,8,128)]
  for rows in (1,2,4,8,16,32):
    points.append((f'balanced_{rows}',rows,1,min(64,8*rows),128*rows))
  for rows in (4,8,16):
    points.append((f'compute_only_{rows}',rows,1,8,128))
  for bw in (512,2048):
    points.append((f'memory_only_{bw}',1,1,8,bw))
  for engines,rows in ((2,1),(4,1),(2,4),(2,8),(4,4)):
    points.append((f'engines_{engines}_rows_{rows}',rows,engines,64,2048))
  for lanes in (8,16,32):
    points.append((f'vector_{lanes}',8,1,lanes,1024))
  for bw in (256,512,1024):
    points.append((f'bandwidth_{bw}',16,1,64,bw))
  return points


def architecture(point,latency):
  label,rows,engines,lanes,bw=point
  if label=='validated_architecture':
    return Timing(batch_rows=8)
  return Timing(batch_rows=max(8,rows),matrix_engines=engines,
      matrix_row_lanes=rows,dram_bytes_per_cycle=bw,dram_port_bytes=bw,
      dram_latency=latency,fifo_chunks=128*rows,scale_lanes=rows,
      unified_dram=True,scalar_store_bytes=4*lanes)


def resources(cfg,image,point,args):
  label,rows,engines,lanes,bw=point
  macs=cfg.D*cfg.MCOLS*rows*engines
  # Explicit screening assumptions. These are not a LUT/DSP implementation model.
  dsp_equivalent=(macs+args.macs_per_dsp-1)//args.macs_per_dsp
  activation=cfg.act_rows*cfg.ACT_BLOCKS*(cfg.D+8)*rows*engines
  fifo=128*rows*cfg.D*engines
  accumulation=rows*engines*cfg.MCOLS*(cfg.D//8+cfg.MCOLS+32)*4
  onchip=4*cfg.TMEM_WORDS+activation+fifo+accumulation
  reasons=[]
  if dsp_equivalent>int(10848*args.dsp_fraction):reasons.append('MXU DSP screening budget')
  if onchip>int(673_000_000/8*args.sram_fraction):reasons.append('on-chip byte screening budget')
  if image.nbytes>32_000_000_000:reasons.append('HBM capacity')
  if bw*args.clock_mhz/1000>810*args.hbm_efficiency:reasons.append('sustained HBM bandwidth budget')
  return dict(matrix_mac_lanes=macs,mxu_dsp_equivalent=dsp_equivalent,
      onchip_bytes_lower_bound=onchip,image_bytes=image.nbytes,
      requested_memory_gbs=bw*args.clock_mhz/1000,
      passes_resource_screen=not reasons,rejection_reasons=reasons,
      scope='Capacity/throughput screening only. DSP packing is assumed; non-MXU DSPs, LUTs, shell, routing, bank implementation and achievable Fmax require synthesis. Reserves are configurable.')


def stream_statistics(workload):
  from opentpu import isa as I
  counts=Counter()
  macs=memory_bytes=0
  for c in workload.commands[0]:
    ins,w=c.instruction,c.instruction.w
    counts[I.OPNAMES[ins.op]]+=1
    if ins.op==I.MM:
      n,kb,m=w[3]&65535,(w[3]>>16)&255,(w[5]>>16)&255
      chunks=kb if (ins.flags>>I.WF_SHIFT)&3==I.W8 else (kb+1)//2
      macs+=n*kb*workload.config.D*m
      memory_bytes+=n*(chunks*workload.config.D+(max(4,w[6]) if not ins.flags&I.F_UNIT else 0))
    elif ins.op in (I.LD,I.ST):memory_bytes+=4*w[2]
    elif ins.op==I.QST:
      memory_bytes+=(w[3]&65535)*(w[3]>>16)*workload.config.D*(4 if w[6]==1 else 1)
  return dict(instruction_counts=dict(counts),matrix_macs=macs,
      modeled_stream_bytes=memory_bytes,byte_scope='Matrix data/scales, LD/ST and QST data; excludes QST scale writes, controller overhead and padding. No persistent weight cache assumed.')


def measure(runner,workload,timing,dest):
  start=time.perf_counter()
  prepared=runner.prepare(workload,timing,dest/'timing.tables')
  cycles,core,process,log=prepared.run()
  counters=re.search(r'Guard checks: (\d+); body executions: (\d+); rollbacks: (\d+)',log)
  if not counters or int(counters[3]):raise RuntimeError('missing runtime counters or rollback')
  (dest/'lpn.log').write_text(log)
  return dict(cycles=cycles,host_total_seconds=time.perf_counter()-start,
      native_seconds=core,preparation_seconds=prepared.preparation_seconds,
      compile_seconds=prepared.compile_seconds,process_seconds=process,
      cache_hit=prepared.cache_hit,rollbacks=0,net=prepared.statistics)


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--opentpu-root',type=Path,required=True)
  p.add_argument('--compiler',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  p.add_argument('--position',type=int,default=15,help='Decode position used for architecture sweep')
  p.add_argument('--prompt-tokens',type=int,default=16)
  p.add_argument('--output-tokens',type=int,default=8)
  p.add_argument('--clock-mhz',type=float,default=200,help='Assumed common hardware clock, not a measured Fmax')
  p.add_argument('--memory-latency',type=int,default=64,help='Assumed HBM round-trip latency in hardware cycles')
  p.add_argument('--hbm-efficiency',type=float,default=.7)
  p.add_argument('--dsp-fraction',type=float,default=.8)
  p.add_argument('--sram-fraction',type=float,default=.5)
  p.add_argument('--macs-per-dsp',type=int,default=1,help='Explicit MXU resource accounting assumption')
  p.add_argument('--baseline',type=Path,default=Path(__file__).parent/'validation/qwen17/completed_fibonacci.json')
  p.add_argument('--skip-sequence',action='store_true',help='Only run the representative-position sweep')
  args=p.parse_args()
  if min(args.clock_mhz,args.memory_latency,args.macs_per_dsp,args.prompt_tokens,args.output_tokens)<1 or args.position<0:
    p.error('sizes, latency and clock must be positive')
  if any(not 0<x<=1 for x in (args.hbm_efficiency,args.dsp_fraction,args.sram_fraction)):
    p.error('resource fractions must lie in (0,1]')
  use_checkout(args.opentpu_root)
  args.output.mkdir(parents=True,exist_ok=True)
  baseline=json.loads(args.baseline.read_text())
  baseline=baseline.get('baseline',baseline)
  reference={r['position']:r for r in baseline['records']}
  spec=qwen17_spec()
  nsteps=args.prompt_tokens+args.output_tokens
  cap=((max(args.position+1,nsteps)+127)//128)*128
  before=time.perf_counter()
  workload,image=compile_shape(spec,args.position,cap)
  shape_seconds=time.perf_counter()-before
  runner=ReusableTiming(args.compiler,args.output/'cache',capacity=32768)
  points=candidates()
  report=dict(scope='Weight-free fixed-length dense Qwen3-1.7B architecture predictions. No numerical inference, generated text, EOS or evolved-RTL validation. Hardware cycles, simulator seconds and assumed-clock latency are reported separately.',
      model=asdict(spec),position=args.position,cap=cap,
      assumptions=dict(clock_mhz=args.clock_mhz,memory_latency_cycles=args.memory_latency,
        peak_hbm_gbs=810,hbm_efficiency=args.hbm_efficiency,hbm_capacity_bytes=32_000_000_000,
        dsp_slices=10848,dsp_fraction=args.dsp_fraction,macs_per_dsp=args.macs_per_dsp,
        embedded_memory_bits=673_000_000,sram_fraction=args.sram_fraction,
        board_reference=V80_SOURCE,bank_conflicts='TMEM strided output ports modeled; HBM bank conflicts and detailed controller queues not modeled.'),
      compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
      source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in
        ['fpga_dse.py','shape_workload.py','model.py','compact.py','components.py','native_timing.py']},
      shape_compilation_seconds=shape_seconds,workload=stream_statistics(workload),points=[])
  base_cycles=None
  for point in points:
    label,rows,engines,lanes,bw=point
    timing=architecture(point,args.memory_latency)
    item=dict(label=label,timing=asdict(timing),vector_lanes=lanes,
      resources=resources(workload.config,image,point,args))
    if item['resources']['passes_resource_screen']:
      cfg=replace(workload.config,LANES=lanes)
      measured=measure(runner,replace(workload,config=cfg),timing,args.output/label)
      item.update(measured)
      if base_cycles is None:base_cycles=measured['cycles']
      if label=='validated_architecture' and cap==128 and args.position in reference:
        r=reference[args.position]
        if measured['cycles']!=r['lpn_cycles'] or program_hash(workload.programs[0])!=r['input_sha256']['prog_0.hex']:
          raise RuntimeError('shape-only baseline does not reproduce recorded program/cycles')
        item['matches_recorded_program_and_lpn_cycles']=True
        item['recorded_rtl_cycles']=r['rtl_cycles']
        item['baseline_cycle_error']=measured['cycles']/r['rtl_cycles']-1
      item.update(cycle_speedup=base_cycles/measured['cycles'],
        hardware_seconds_at_assumed_clock=measured['cycles']/(args.clock_mhz*1e6),
        matrix_utilization=report['workload']['matrix_macs']/(measured['cycles']*item['resources']['matrix_mac_lanes']))
    report['points'].append(item)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in item.items() if k in ('label','cycles','cycle_speedup','resources')}),flush=True)
  feasible=[x for x in report['points'] if 'cycles' in x]
  best=min(feasible,key=lambda x:x['cycles'])
  report['best_point']=best['label']
  if not args.skip_sequence:
    best_point=next(x for x in points if x[0]==best['label'])
    sequence=[]
    for pos in range(nsteps):
      before=time.perf_counter()
      w,img=compile_shape(spec,pos,cap)
      shape=time.perf_counter()-before
      base=measure(runner,w,Timing(batch_rows=8),args.output/'sequence'/str(pos)/'base')
      if cap==128 and pos in reference:
        if program_hash(w.programs[0])!=reference[pos]['input_sha256']['prog_0.hex'] or base['cycles']!=reference[pos]['lpn_cycles']:
          raise RuntimeError(f'static shape differs from validated baseline at position {pos}')
      evolved=measure(runner,replace(w,config=replace(w.config,LANES=best_point[3])),
        architecture(best_point,args.memory_latency),args.output/'sequence'/str(pos)/'evolved')
      sequence.append(dict(position=pos,phase='prefill' if pos<args.prompt_tokens else 'decode',
        shape_seconds=shape,baseline=base,evolved=evolved,
        baseline_program_and_cycles_checked=cap==128 and pos in reference))
      print(json.dumps(dict(sequence_completed=pos+1,baseline_cycles=base['cycles'],evolved_cycles=evolved['cycles'])),flush=True)
    report['sequence']=dict(prompt_tokens=args.prompt_tokens,output_tokens=args.output_tokens,
      token_steps=nsteps,records=sequence,baseline_cycles=sum(x['baseline']['cycles'] for x in sequence),
      evolved_cycles=sum(x['evolved']['cycles'] for x in sequence),
      shape_compilation_seconds=sum(x['shape_seconds'] for x in sequence),
      simulator_seconds=sum(x[k]['host_total_seconds'] for x in sequence for k in ('baseline','evolved')))
    report['sequence']['cycle_speedup']=report['sequence']['baseline_cycles']/report['sequence']['evolved_cycles']
  report['status']='complete'
  (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(dict(status='complete',best_point=report['best_point'],best_cycle_speedup=best['cycle_speedup'])),flush=True)


if __name__=='__main__':main()
