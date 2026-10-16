"""Observe upstream Engine.generate without choosing its batching or decode order.

The backend consumes upstream programs. Planning mode replaces numerical storage
and results with placeholders, but uses the same unmodified Engine scheduling.
It predicts no text or EOS. Validation mode computes real values and compares
the same program/DRAM files in the ISA model, LPN and Verilator.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from .workload import use_checkout


def run_engine(spec,cfg,cap,prompt,max_new,backend_factory,weights=None):
  """Use the upstream constructor and generate entrypoint in both modes."""
  from opentpu.llm.qwen3 import Engine
  if weights is None:
    import numpy as np
    from unittest.mock import patch
    from opentpu import imgcache
    # No weight tensor is materialized. This broadcast view only supplies the
    # host input interface. The planning backend never consumes numerical data.
    placeholder={'model.embed_tokens.weight':np.broadcast_to(
      np.zeros(spec.hidden,np.float32),(spec.vocab,spec.hidden))}
    # Only device image materialization is replaced. Engine.generate/prefill,
    # fit_chunk, compilation, default row count and decode all remain upstream.
    with patch.object(imgcache,'build',return_value=[]):
      engine=Engine(spec,placeholder,cap,cfg,backend=backend_factory,
                    pipeline=False,resident=False)
  else:
    engine=Engine(spec,weights,cap,cfg,backend=backend_factory,
                  pipeline=False,resident=False)
  generated=engine.generate(prompt,max_new=max_new)
  save=getattr(engine.backend,'save_previous',None)
  if save is not None:
    save(engine.pos)
  return engine,generated


def make_backend(output,prompt_length,compiler=None,options=None,capacity=65536):
  import numpy as np
  from .shape_workload import resolve_static,program_hash
  from .e2e import CheckedBackend
  from .e2e import atomic_json
  from .fpga_dse import measure
  from .native_timing import ReusableTiming
  from .model import Timing
  output=Path(output)
  class Backend(CheckedBackend):
    def __init__(self,cfg,images):
      if options is not None:
        super().__init__(cfg,images,options)
      else:
        self.cfg=cfg
        self.records=[]
      self.engine=None
      self.runner=(ReusableTiming(compiler,output/'timing-cache',capacity=capacity)
                   if compiler is not None and options is None else None)

    def attach(self,engine):
      self.engine=engine

    def write(self,s,addr,data):
      if options is not None:
        super().write(s,addr,data)

    def save_previous(self,position_after):
      if self.records:
        record=self.records[-1]
        rows=record.get('rows',position_after-record['position_before'])
        if rows>0:
          record['input_rows']=rows
        atomic_json(output/'kernels'/f"{record['kernel_index']:04d}"/'record.json',record)

    def read(self,s,addr,nbytes):
      if self.records:
        logits=self.engine.image.io['logits']
        if logits<=addr<logits+4*self.engine.spec.vocab*self.engine.rows:
          self.records[-1]['logit_rows_read']+=nbytes//(4*self.engine.image.v_loc)
          self.save_previous(self.engine.pos)
      if options is not None:
        return super().read(s,addr,nbytes)
      # Fixed scheduling placeholders: argmax 0, not a predicted model output.
      # Qwen's EOS ids differ from 0, so fixed-length planning does not stop early.
      return np.zeros(nbytes,np.uint8)

    def run(self,programs,args=None):
      index=len(self.records)
      before=self.engine.pos
      self.save_previous(before)
      directory=output/'kernels'/f'{index:04d}'
      directory.mkdir(parents=True,exist_ok=True)
      for sid,program in enumerate(programs):
        (directory/f'program_{sid}.json').write_text(json.dumps(
          [ins.encode() for ins in program])+'\n')
      observed=dict(kernel_index=index,position_before=before,
        phase='prefill' if before<prompt_length else 'decode',
        # Device-resident inputs need no host embedding write. Observe rows
        # from upstream stats/position advancement after the run, not writes.
        input_rows=None if before<prompt_length else 1,logit_rows_read=0,
        program_sha256=[program_hash(p) for p in programs],args=list(args or []))
      atomic_json(output/'progress.json',dict(status='running',pid=os.getpid(),
          completed_kernels=index,active_kernel=observed))
      if options is not None:
        try:
          result=super().run(programs,args)
        except Exception as error:
          atomic_json(directory/'failure.json',dict(observed,
              error_type=type(error).__name__,error=str(error)))
          atomic_json(output/'progress.json',dict(status='failed',pid=os.getpid(),
              completed_kernels=index,active_kernel=observed,error=str(error)))
          raise
      else:
        workload=resolve_static(self.cfg,programs,args)
        result=dict(instructions=[len(c) for c in workload.commands])
        if self.runner is not None:
          result['lpn']=measure(self.runner,workload,
              Timing(batch_rows=8,dram_bytes_per_cycle=min(128,self.cfg.D)),directory)
        self.records.append(result)
      result.update(observed)
      atomic_json(directory/'record.json',result)
      atomic_json(output/'progress.json',dict(status='running',pid=os.getpid(),
          completed_kernels=index+1,last_kernel=result))
      return result
  return Backend


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--opentpu-root',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--mode',choices=['plan','validate-rtl'],default='plan')
  parser.add_argument('--compiler',type=Path)
  parser.add_argument('--model-dir',type=Path)
  parser.add_argument('--prompt',default='Once upon a time in a quiet village')
  parser.add_argument('--prompt-tokens',type=int,default=16,
      help='Prescribed placeholder token count in weight-free planning only')
  parser.add_argument('--output-tokens',type=int,default=8)
  parser.add_argument('--cap',type=int,default=128)
  parser.add_argument('--depth',type=int,choices=(64,128),default=128,
      help='Hardware D; SmolLM2-135M requires 64 for its 64-element attention heads')
  parser.add_argument('--rtl-timeout',type=float,default=28800)
  args=parser.parse_args()
  if args.mode=='validate-rtl' and (args.model_dir is None or args.compiler is None):
    parser.error('RTL validation requires --model-dir and --compiler')
  if min(args.prompt_tokens,args.output_tokens,args.cap)<1:
    parser.error('lengths must be positive')
  args.opentpu_root=use_checkout(args.opentpu_root)
  from opentpu.llm.qwen3 import load_weights,device_config
  from .shape_workload import qwen17_spec
  from .e2e import atomic_json,digest
  args.output=args.output.resolve()
  args.output.mkdir(parents=True,exist_ok=False)
  if args.mode=='plan':
    spec=qwen17_spec()
    prompt=[0]*args.prompt_tokens
    weights=None
  else:
    from tokenizers import Tokenizer
    from opentpu.llm import load_spec
    from opentpu import rtlsim
    import torch
    torch.set_num_threads(1)
    spec=load_spec(args.model_dir)
    prompt=Tokenizer.from_file(str(args.model_dir/'tokenizer.json')).encode(args.prompt).ids
    weights=load_weights(args.model_dir)
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='4'
    rtlsim.VFLAGS+=['-CFLAGS','-O3']
  if not prompt or len(prompt)+args.output_tokens>args.cap:
    parser.error('prompt plus output must fit capacity')
  cfg=device_config(spec,args.cap,S=1,D=args.depth,MCOLS=4,ACT_ROWS=16,
      ACT_BLOCKS=64,LANES=8,IMEM_WORDS=32768,TMEM_WORDS=65536)
  from types import SimpleNamespace
  options=(SimpleNamespace(output=args.output,lpn_compiler=args.compiler,
      batch_rows=8,reuse_timing=True,command_capacity=65536,
      rtl_timeout=args.rtl_timeout) if args.mode=='validate-rtl' else None)
  start=time.perf_counter()
  engine,generated=run_engine(spec,cfg,args.cap,prompt,args.output_tokens,
      make_backend(args.output,len(prompt),args.compiler,options),weights)
  records=engine.backend.records
  # Backend records are the dictionaries returned to Engine; upstream _run_rows
  # itself adds 'rows'. Preserve that observed choice instead of choosing it.
  for r in records:r['input_rows']=r.get('rows',r['input_rows'])
  for index,r in enumerate(records):
    atomic_json(args.output/'kernels'/f'{index:04d}'/'record.json',r)
  report=dict(status='complete',mode=args.mode,entrypoint='opentpu.llm.qwen3.Engine.generate',
      scheduling='Unmodified upstream prefill_chunks/fit_chunk and autoregressive decode; default PREFILL_ROWS. Host compile pipeline disabled; resident device decode disabled.',
      scope=('Weight-free fixed-token placeholders; no text, logits, EOS or RTL accuracy claim.'
             if args.mode=='plan' else 'Real checkpoint numerical execution; identical ISA/LPN/RTL inputs and bit-exact device memory checks.'),
      source_sha256=digest(Path(__file__)),model=asdict(spec),config=asdict(cfg),
      model_directory=str(args.model_dir.resolve()) if args.model_dir else None,
      prompt_ids=prompt,
      prompt_tokens=len(prompt),requested_output_tokens=args.output_tokens,
      returned_output_tokens=len(generated),final_position=engine.pos,
      generated_ids=generated,
      generated_text=(Tokenizer.from_file(str(args.model_dir/'tokenizer.json')).decode(generated)
                      if args.mode=='validate-rtl' else None),
      engine_default_rows=engine.rows,host_seconds=time.perf_counter()-start,
      kernels=records,project_python_sha256={p.name:digest(p)
        for p in sorted(Path(__file__).parent.glob('*.py'))},
      compiler_sha256=digest(args.compiler) if args.compiler else None,
      runtime_sha256=digest(args.compiler.resolve().parent/'libLPNNativeRuntime.a') if args.compiler else None,
      upstream_python_sha256={str(p.relative_to(args.opentpu_root)):digest(p)
        for p in sorted((args.opentpu_root/'opentpu').rglob('*.py'))})
  if args.mode=='validate-rtl':
    rtl_cycles=sum(r['rtl_cycles'] for r in records)
    lpn_cycles=sum(r['lpn_cycles'] for r in records)
    rtl_core=sum(r['rtl_core_seconds'] for r in records)
    lpn_core=sum(r['lpn_core_seconds'] for r in records)
    report['timing_totals']=dict(rtl_cycles=rtl_cycles,lpn_cycles=lpn_cycles,
        relative_cycle_error=lpn_cycles/rtl_cycles-1,
        worst_kernel_cycle_error=max(abs(r['relative_cycle_error']) for r in records),
        rtl_core_seconds=rtl_core,lpn_core_seconds=lpn_core,
        core_speedup=rtl_core/lpn_core,
        lpn_pipeline_seconds_excluding_input_serialization=sum(r['single_use_lpn_seconds'] for r in records),
        rtl_process_seconds=sum(r['rtl_process_seconds'] for r in records),
        all_device_memories_bit_exact=all(r['all_dram_and_tmem_bit_exact'] for r in records),
        all_input_files_identical=all(r['identical_input_files'] for r in records),
        zero_rollbacks=all(r['zero_rollbacks'] for r in records))
    report['upstream_rtl_sha256']={str(p.relative_to(args.opentpu_root)):digest(p)
      for folder in ('rtl','sim/verilator')
      for p in sorted((args.opentpu_root/folder).rglob('*'))
      if p.is_file() and p.suffix in ('.sv','.svh','.cpp','.h')}
  atomic_json(args.output/'report.json',report)
  atomic_json(args.output/'progress.json',dict(status='complete',pid=os.getpid(),
      completed_kernels=len(records),report='report.json'))
  print(json.dumps({k:v for k,v in report.items() if not k.endswith('_sha256') and k!='kernels'},indent=2),flush=True)
  print(json.dumps(dict(kernel_schedule=[{k:r[k] for k in
      ['position_before','phase','input_rows','logit_rows_read','instructions']} for r in records])),flush=True)


if __name__=='__main__':main()
