"""Observe upstream Engine.generate without choosing its batching or decode order.

The backend consumes upstream programs. Planning mode replaces numerical storage
and results with placeholders, but uses the same unmodified Engine scheduling.
It predicts no text or EOS. Validation mode computes real values and compares
the same program/DRAM files in the ISA model, LPN and Verilator.
"""

import argparse
from dataclasses import asdict
import json
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
  return engine,generated


def make_backend(output,prompt_length,compiler=None,options=None,capacity=65536):
  import numpy as np
  from .shape_workload import resolve_static,program_hash
  from .e2e import CheckedBackend
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
      self.input_rows=1
      self.runner=(ReusableTiming(compiler,output/'timing-cache',capacity=capacity)
                   if compiler is not None and options is None else None)

    def attach(self,engine):
      self.engine=engine

    def write(self,s,addr,data):
      if self.engine is not None and addr==self.engine.image.io['x']:
        self.input_rows=max(1,np.asarray(data).nbytes//(4*self.engine.spec.hidden))
      if options is not None:
        super().write(s,addr,data)

    def read(self,s,addr,nbytes):
      if self.records:
        logits=self.engine.image.io['logits']
        if logits<=addr<logits+4*self.engine.spec.vocab*self.engine.rows:
          self.records[-1]['logit_rows_read']+=nbytes//(4*self.engine.image.v_loc)
      if options is not None:
        return super().read(s,addr,nbytes)
      # Fixed scheduling placeholders: argmax 0, not a predicted model output.
      # Qwen's EOS ids differ from 0, so fixed-length planning does not stop early.
      return np.zeros(nbytes,np.uint8)

    def run(self,programs,args=None):
      index=len(self.records)
      before=self.engine.pos
      directory=output/'kernels'/f'{index:04d}'
      directory.mkdir(parents=True,exist_ok=True)
      for sid,program in enumerate(programs):
        (directory/f'program_{sid}.json').write_text(json.dumps(
          [ins.encode() for ins in program])+'\n')
      observed=dict(kernel_index=index,position_before=before,
        phase='prefill' if before<prompt_length else 'decode',
        input_rows=self.input_rows,logit_rows_read=0,
        program_sha256=[program_hash(p) for p in programs],args=list(args or []))
      if options is not None:
        result=super().run(programs,args)
      else:
        workload=resolve_static(self.cfg,programs,args)
        result=dict(instructions=[len(c) for c in workload.commands])
        if self.runner is not None:
          result['lpn']=measure(self.runner,workload,Timing(batch_rows=8),directory)
        self.records.append(result)
      result.update(observed)
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
    spec=load_spec(args.model_dir)
    prompt=Tokenizer.from_file(str(args.model_dir/'tokenizer.json')).encode(args.prompt).ids
    weights=load_weights(args.model_dir)
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='4'
    rtlsim.VFLAGS+=['-CFLAGS','-O3']
  if not prompt or len(prompt)+args.output_tokens>args.cap:
    parser.error('prompt plus output must fit capacity')
  cfg=device_config(spec,args.cap,S=1,D=128,MCOLS=4,ACT_ROWS=16,
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
  report=dict(status='complete',mode=args.mode,entrypoint='opentpu.llm.qwen3.Engine.generate',
      scheduling='Unmodified upstream prefill_chunks/fit_chunk and autoregressive decode; default PREFILL_ROWS. Host compile pipeline disabled; resident device decode disabled.',
      scope=('Weight-free fixed-token placeholders; no text, logits, EOS or RTL accuracy claim.'
             if args.mode=='plan' else 'Real checkpoint numerical execution; identical ISA/LPN/RTL inputs and bit-exact device memory checks.'),
      source_sha256=digest(Path(__file__)),model=asdict(spec),config=asdict(cfg),
      model_directory=str(args.model_dir.resolve()) if args.model_dir else None,
      prompt_ids=prompt,
      prompt_tokens=len(prompt),requested_output_tokens=args.output_tokens,
      returned_output_tokens=len(generated),final_position=engine.pos,
      engine_default_rows=engine.rows,host_seconds=time.perf_counter()-start,
      kernels=records,project_python_sha256={p.name:digest(p)
        for p in sorted(Path(__file__).parent.glob('*.py'))},
      compiler_sha256=digest(args.compiler) if args.compiler else None,
      runtime_sha256=digest(args.compiler.resolve().parent/'libLPNNativeRuntime.a') if args.compiler else None,
      upstream_python_sha256={str(p.relative_to(args.opentpu_root)):digest(p)
        for p in sorted((args.opentpu_root/'opentpu').rglob('*.py'))})
  if args.mode=='validate-rtl':
    report['upstream_rtl_sha256']={str(p.relative_to(args.opentpu_root)):digest(p)
      for folder in ('rtl','sim/verilator')
      for p in sorted((args.opentpu_root/folder).rglob('*'))
      if p.is_file() and p.suffix in ('.sv','.svh','.cpp','.h')}
  atomic_json(args.output/'report.json',report)
  print(json.dumps({k:v for k,v in report.items() if not k.endswith('_sha256') and k!='kernels'},indent=2),flush=True)
  print(json.dumps(dict(kernel_schedule=[{k:r[k] for k in
      ['position_before','phase','input_rows','logit_rows_read','instructions']} for r in records])),flush=True)


if __name__=='__main__':main()
