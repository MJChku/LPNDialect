"""Compare a numerically checked RTL trace with an observed LPN command timeline.

Acceptance and execution are distinct stages. Start-to-end differences alone
cannot identify unit latency when a command was accepted into a FIFO earlier.
"""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import re
import statistics
import struct


def main():
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('case','rtl-dir','lpn-dir','opentpu-root','output'):p.add_argument('--'+name,type=Path,required=True)
  p.add_argument('--kernel',type=int,required=True);a=p.parse_args()
  from .e2e import atomic_json,digest
  from .retained_inputs import verified_program
  from .shape_workload import resolve_static
  from .workload import use_checkout
  from .rtl_evolution.parallel_inference import files_equal
  use_checkout(a.opentpu_root.resolve())
  from opentpu import Config,isa as I,waits as W,profile
  reference=json.loads((a.case/'report.json').read_text());record=reference['kernels'][a.kernel]
  if reference['status']!='complete' or not reference['all_dram_and_tmem_bit_exact']:raise ValueError('verified reference required')
  source=a.case/'kernels'/f'{a.kernel:04d}'
  for name in ('prog_0.hex','dram_0.bin'):
    if digest(a.rtl_dir/name)!=record['input_sha256'][name]:raise ValueError('RTL trace inputs differ from canonical input')
  encoded,identity=verified_program(a.rtl_dir,0);program=[I.Instr.decode(w) for w in encoded]
  if not files_equal(a.rtl_dir/'dram_out_0.bin',source/'golden_dram_0.bin'):raise ValueError('traced DRAM differs from ISA')
  words=[int(w,16) for w in (a.rtl_dir/'tmem_0.hex').read_text().split()]
  if len(words)!=reference['config']['TMEM_WORDS'] or struct.pack('<'+'I'*len(words),*words)!=(source/'golden_tmem_0.bin').read_bytes():
    raise ValueError('traced scratchpad differs from ISA')
  launch=json.loads((a.rtl_dir/'launch.json').read_text())
  if digest(Path(launch['command'][0]))!=launch['simulator_sha256']:raise ValueError('traced executable changed')
  trace=(a.rtl_dir/'rtl.log').read_text();result=re.search(r'RESULT cycles=(\d+) halted=1 error=0',trace)
  if not result or int(result[1])!=launch['expected_cycles']:raise ValueError('trace did not complete at the expected cycle')
  if [int(n) for n in re.findall(r'SLICE \d+ icount=(\d+)',trace)]!=record['instructions']:
    raise ValueError('trace instruction count differs from RTL reference')
  cfg=Config(**reference['config']);work=resolve_static(cfg,[program],record['args'])
  parsed=profile.parse(trace,replace(cfg,LANES=64,MCOLS=1),[program])
  observed=json.loads((a.lpn_dir/'report.json').read_text())
  if (not observed['native_python_cycles_agree'] or observed['kernel']!=a.kernel
      or observed['program_json_sha256']['0']!=identity):raise ValueError('LPN trace identity differs')
  for name,expected in observed['source_sha256'].items():
    if digest(Path(name))!=expected:raise ValueError('observed LPN sources changed')
  if digest(a.lpn_dir/'commands.json')!=observed['command_trace_sha256']:raise ValueError('LPN command trace changed')
  times={r['index']:r for r in json.loads((a.lpn_dir/'commands.json').read_text())}
  units=[(i,c) for i,c in enumerate(work.commands[0]) if W.UNIT.get(c.instruction.op,5)!=5]
  if len(parsed.recs)!=len(units):raise ValueError('dispatch coverage differs')
  gaps=defaultdict(list);vectors={};matrices=[]
  for rtl,(idx,c) in zip(parsed.recs,units):
    if rtl.pc!=c.pc or rtl.op!=c.instruction.op or not 0<=rtl.start<=rtl.end:raise ValueError('dynamic command alignment differs')
    item=times[idx];t=item['times'];gaps[item['operation']].append(rtl.end-t['completed'])
    if rtl.unit==3:vectors[idx]=rtl.start
    if rtl.unit==1:
      matrices.append(dict(index=idx,pc=c.pc,rtl_accept=rtl.start,rtl_release=rtl.release,
          rtl_completed=rtl.end,lpn_admit=t['started'],lpn_compute=t['compute_started'],lpn_completed=t['completed']))
  actual=sorted(vectors,key=vectors.get)
  predicted=sorted(vectors,key=lambda i:times[i]['times'].get('accepted',times[i]['times']['started']))
  first=next((dict(position=k,rtl_index=x,lpn_index=y) for k,(x,y) in enumerate(zip(actual,predicted)) if x!=y),None)
  summary=dict(status='complete',scope=__doc__,kernel=a.kernel,rtl_cycles=int(result[1]),lpn_cycles=observed['cycles'],
      relative_cycle_error=observed['cycles']/int(result[1])-1,timing_target_met=abs(observed['cycles']/int(result[1])-1)<=.10,
      all_full_dram_and_tmem_bit_exact=True,all_dynamic_commands_aligned=True,dispatches=len(units),
      first_vector_acceptance_order_difference=first,first_matrix_commands=matrices[:20],
      completion_gaps_by_operation={name:dict(commands=len(v),minimum=min(v),maximum=max(v),mean=statistics.mean(v)) for name,v in gaps.items()},
      simulator_sha256=launch['simulator_sha256'],program_json_sha256=identity,
      rtl_trace_sha256=digest(a.rtl_dir/'rtl.log'),lpn_profile_sha256=digest(a.lpn_dir/'report.json'),
      script_sha256=digest(Path(__file__)),numerical_arithmetic_executed=False,
      limitations=['Completion gaps accumulate upstream delays; they are not isolated unit latencies',
                   'One kernel, not a full-inference timing validation'])
  atomic_json(a.output,summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
