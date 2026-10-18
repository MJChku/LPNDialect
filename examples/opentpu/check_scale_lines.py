"""Check physical matrix scale-line requests against a complete RTL trace."""
import argparse
import json
from pathlib import Path
import re
import struct


def main():
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('case','replay','rtl-dir','opentpu-root','output'):p.add_argument('--'+name,type=Path,required=True)
  p.add_argument('--kernel',type=int,required=True);a=p.parse_args()
  from .e2e import atomic_json,digest
  from .retained_inputs import verified_program
  from .shape_workload import resolve_static
  from .workload import use_checkout
  from .model import matrix_scale_requests
  from .rtl_evolution.parallel_inference import files_equal
  use_checkout(a.opentpu_root.resolve())
  from opentpu import Config,isa as I
  reference=json.loads((a.case/'report.json').read_text());replay=json.loads(a.replay.read_text())
  record=reference['kernels'][a.kernel];observed=next(v for v in replay['records'] if v['step']==a.kernel)
  if replay['status']!='complete' or not observed['all_dram_and_tmem_bit_exact']:raise ValueError('completed numerical replay required')
  if replay['reference_report_sha256']!=digest(a.case/'report.json'):raise ValueError('reference differs')
  for name in ('prog_0.hex','dram_0.bin'):
    if digest(a.rtl_dir/name)!=record['input_sha256'][name]:raise ValueError('trace input changed')
  source=a.case/'kernels'/f'{a.kernel:04d}'
  if not files_equal(a.rtl_dir/'dram_out_0.bin',source/'golden_dram_0.bin'):raise ValueError('traced DRAM differs')
  words=[int(x,16) for x in (a.rtl_dir/'tmem_0.hex').read_text().split()]
  if len(words)!=reference['config']['TMEM_WORDS'] or struct.pack('<'+'I'*len(words),*words)!=(source/'golden_tmem_0.bin').read_bytes():
    raise ValueError('traced scratchpad differs')
  if digest(Path(observed['simulator']))!=observed['baseline_binary_sha256']:raise ValueError('simulator changed')
  trace=(a.rtl_dir/'rtl.log').read_text()
  if f'RESULT cycles={observed["rtl_cycles"]} halted=1 error=0' not in trace:raise ValueError('trace cycles differ')
  if [int(n) for n in re.findall(r'SLICE \d+ icount=(\d+)',trace)]!=record['instructions']:raise ValueError('instruction count differs')
  encoded,identity=verified_program(a.rtl_dir,0);cfg=Config(**reference['config'])
  work=resolve_static(cfg,[[I.Instr.decode(w) for w in encoded]],record['args'])
  parallel=observed['rtl_top_parameters']['MXU_ROWS'];predicted=storage_proxy=matrices=0
  for c in work.commands[0]:
    ins=c.instruction
    if ins.op!=I.MM:continue
    matrices+=1;n=ins.w[3]&65535
    predicted+=sum(matrix_scale_requests(ins,I,row,parallel) for row in range(n))
    if not ins.flags&I.F_UNIT:storage_proxy+=n*max(4,ins.w[6])
  counts=re.findall(r'MXS0 a=(\d+) sw=(\d+)',trace)
  actual=sum(int(n) for n,_ in counts)
  if not counts or actual!=predicted:raise ValueError(f'physical request count differs: {predicted} vs {actual}')
  result=dict(status='complete',scope=__doc__,kernel=a.kernel,parallel_rows=parallel,matrix_commands=matrices,
      predicted_A_requests=predicted,rtl_A_requests=actual,rtl_sw_only_requests=sum(int(n) for _,n in counts),
      physical_scale_line_bytes=actual*32,old_compact_storage_byte_proxy=storage_proxy,
      all_requests_match=True,all_full_dram_and_tmem_bit_exact=True,numerical_arithmetic_executed=False,
      program_json_sha256=identity,rtl_trace_sha256=digest(a.rtl_dir/'rtl.log'),
      simulator_sha256=observed['baseline_binary_sha256'],replay_report_sha256=digest(a.replay),
      source_sha256={str(f):digest(f) for f in (Path(__file__),Path(__file__).with_name('model.py'))})
  atomic_json(a.output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
