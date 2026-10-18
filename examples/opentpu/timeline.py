"""Observe existing Python LPN transactions without changing model or scheduler.

Profiling is deliberately separate from simulation speed measurements. Command
waits overlap across units and must not be added into an end-to-end runtime.
"""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import re
import time

from lpnlang_mlir import PythonSimulator


class TimelineSimulator(PythonSimulator):
  def reset(self):
    super().reset()
    self.command_times=defaultdict(dict)
    self.busy=defaultdict(list)
    self.held={}
    self.definition=''

  def _invoke(self,definition,*args,**kwargs):
    before=self.definition;self.definition=definition.name
    try:return super()._invoke(definition,*args,**kwargs)
    finally:self.definition=before

  @staticmethod
  def resource(name):
    return bool(re.search(r'_(resource|compute|port_[ab]|scalar_write|issue|vector_result_port)$',name))

  def _commit_transaction(self):
    ctx=self._require_transaction();now=self.current_time
    for place,token in ctx.taken:
      if re.fullmatch(r's\d+_pipeline',place.name):
        sid=int(place.name.split('_')[0][1:])
        self.command_times[sid,token.get('id')]['arrived']=now
      if re.fullmatch(r's\d+_slot_\d+',place.name) and self.definition.endswith(('_execute','_start')):
        sid=int(place.name.split('_')[0][1:])
        self.command_times[sid,token.get('id')]['started']=now
      if re.fullmatch(r's\d+_slot_\d+',place.name) and self.definition.endswith('_accept'):
        sid=int(place.name.split('_')[0][1:])
        self.command_times[sid,token.get('id')]['accepted']=now
      if re.fullmatch(r's\d+_unit_\d+_commands',place.name) and self.definition.endswith('_execute'):
        sid=int(place.name.split('_')[0][1:])
        self.command_times[sid,token.get('id')]['started']=now
      if re.fullmatch(r's\d+_mxu_\d+_ready',place.name) and self.definition.endswith(('_issue_rows','_accept_rows')):
        sid=int(place.name.split('_')[0][1:])
        self.command_times[sid,token.get('id')].setdefault('compute_started',now)
      if self.resource(place.name):
        if place.name in self.held:raise RuntimeError('profile found a multiply-held resource')
        self.held[place.name]=now
    for place,token,ready in ctx.emits:
      if re.fullmatch(r's\d+_pipeline',place.name):
        sid=int(place.name.split('_')[0][1:])
        self.command_times[sid,token.get('id')]['dispatched']=now
      if re.fullmatch(r's\d+_signals',place.name):
        sid=int(place.name.split('_')[0][1:]);entry=self.command_times[sid,token.get('id')]
        if 'completed' in entry:raise RuntimeError('duplicate command completion')
        entry['completed']=ready
      if self.resource(place.name) and place.name in self.held:
        start=self.held.pop(place.name)
        if ready>start:self.busy[place.name].append((start,ready))
    super()._commit_transaction()


def main():
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('case','prediction','opentpu-root','output'):p.add_argument('--'+name,type=Path,required=True)
  p.add_argument('--architecture',required=True);p.add_argument('--kernel',type=int,required=True)
  a=p.parse_args()
  from .e2e import atomic_json,digest
  from .retained_inputs import verified_program
  from .workload import use_checkout
  from .shape_workload import resolve_static
  from .compact import CompactOpenTPU,dependencies
  from .model import Timing
  use_checkout(a.opentpu_root.resolve())
  from opentpu import Config,isa as I,waits as W
  reference=json.loads((a.case/'report.json').read_text())
  prediction=json.loads(a.prediction.read_text())
  if prediction['status']!='complete' or prediction['reference_report_sha256']!=digest(a.case/'report.json'):
    raise ValueError('completed prediction on the same verified reference is required')
  if reference['status']!='complete' or not reference['all_dram_and_tmem_bit_exact'] or not reference['all_inputs_identical_to_baseline']:
    raise ValueError('verified original inference required')
  point=next(x for x in prediction['points'] if x['architecture']['name']==a.architecture)
  expected=next(k for k in point['kernels'] if k['kernel']==a.kernel)
  record=reference['kernels'][a.kernel];source=a.case/'kernels'/f'{a.kernel:04d}'
  cfg=Config(**reference['config']);programs=[];identities={}
  for sid in range(cfg.S):
    if digest(source/f'prog_{sid}.hex')!=record['input_sha256'][f'prog_{sid}.hex']:
      raise ValueError('canonical program changed')
    encoded,identity=verified_program(source,sid)
    programs.append([I.Instr.decode(w) for w in encoded]);identities[str(sid)]=identity
  work=replace(resolve_static(cfg,programs,record['args']),config=replace(cfg,LANES=64,MCOLS=point['architecture']['columns']))
  timing=Timing(**expected['timing']);model=CompactOpenTPU(work,timing)
  files=[Path(__file__).with_name(n) for n in ('timeline.py','model.py','compact.py','components.py','activation_stream.py','shape_workload.py','retained_inputs.py')]
  hashes={str(f):digest(f) for f in files}
  a.output.mkdir(parents=True,exist_ok=False)
  started=time.perf_counter();sim=TimelineSimulator(model.net)
  ticks=sim.run(max_time=expected['cycles']+1)
  if ticks!=expected['cycles'] or sim.place_dicts('program_completed')!=[{}]:
    raise RuntimeError('observer simulation differs from the native prediction')
  if sim.held:raise RuntimeError('unfinished observed resource hold')
  commands=[];aggregates=defaultdict(lambda:defaultdict(float))
  for sid,items in enumerate(work.commands):
    deps,_=dependencies(items,I,W,timing.matrix_engines)
    for index,c in enumerate(items):
      times=sim.command_times[sid,index]
      if 'completed' not in times:raise RuntimeError('unobserved command completion')
      ins=c.instruction
      label=I.VFUNCS[(ins.w[5]>>16)&255] if ins.op==I.VOP else I.OPNAMES[ins.op]
      # MM admission waits for DRAM dependencies while compute has additional
      # activation constraints. Report its stages without attributing the wait.
      if ins.op!=I.MM and 'started' in times and 'arrived' in times:
        inputs=max([times['arrived']]+[sim.command_times[sid,d]['completed'] for d in deps[index]])
        wait=times['started']-inputs
        if wait<0:raise RuntimeError('command started before its input dependencies')
        times['resource_or_selection_wait']=wait
        aggregates[label]['resource_or_selection_wait']+=wait
      aggregates[label]['commands']+=1
      if 'started' in times:aggregates[label]['start_to_completion']+=times['completed']-times['started']
      commands.append(dict(slice=sid,index=index,pc=c.pc,operation=label,times=times))
  resources={name:dict(busy_cycles=sum(end-start for start,end in intervals),
                       utilization=sum(end-start for start,end in intervals)/ticks,
                       holds=len(intervals)) for name,intervals in sim.busy.items()}
  if hashes!={str(f):digest(f) for f in files}:raise RuntimeError('profile sources changed')
  atomic_json(a.output/'commands.json',commands)
  atomic_json(a.output/'resources.json',dict(sim.busy))
  result=dict(status='complete',scope=__doc__,kernel=a.kernel,architecture=a.architecture,
      cycles=ticks,native_cycles=expected['cycles'],native_python_cycles_agree=True,
      observer_seconds=time.perf_counter()-started,program_json_sha256=identities,
      source_sha256=hashes,reference_report_sha256=digest(a.case/'report.json'),
      prediction_report_sha256=digest(a.prediction),resources=resources,operations=dict(aggregates),
      command_trace_sha256=digest(a.output/'commands.json'),
      resource_trace_sha256=digest(a.output/'resources.json'),
      limitations=['Wait totals overlap across commands; do not sum as elapsed time',
                   'Matrix streaming dependency waits are not attributed',
                   'Profiling wall time is not a simulator performance measurement'])
  atomic_json(a.output/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
