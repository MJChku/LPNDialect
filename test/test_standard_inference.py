"""Upstream inference scheduling and shared-backend integration checks."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from examples.opentpu.workload import use_checkout


class StandardInferenceTest(unittest.TestCase):
  compiler=None
  rtl=False
  artifacts=None

  def test_upstream_plans_multirow_prefill_and_last_prompt_logits(self):
    from examples.opentpu.shape_workload import qwen17_spec
    from examples.opentpu.standard_inference import make_backend,run_engine
    from opentpu.llm.qwen3 import device_config
    spec=qwen17_spec()
    cfg=device_config(spec,128,S=1,D=128,MCOLS=4,ACT_ROWS=16,
      ACT_BLOCKS=64,LANES=8,IMEM_WORDS=32768,TMEM_WORDS=65536)
    with tempfile.TemporaryDirectory() as directory:
      engine,generated=run_engine(spec,cfg,128,[0]*16,8,make_backend(directory,16))
      pre=[x for x in engine.backend.records if x['phase']=='prefill']
      decode=[x for x in engine.backend.records if x['phase']=='decode']
      self.assertEqual(sum(x['rows'] for x in pre),16)
      self.assertTrue(any(x['rows']>1 for x in pre))
      self.assertEqual([x['logit_rows_read'] for x in pre],[0]*(len(pre)-1)+[1])
      self.assertEqual(len(decode),8)
      self.assertEqual(engine.pos,24)
      self.assertEqual(len(generated),8)
      self.assertFalse(list(Path(directory).rglob('dram_*.bin')))

  def test_shared_upstream_generate_matches_real_isa_and_rtl(self):
    if not self.rtl:
      self.skipTest('pass --rtl and --compiler to compare actual Verilator')
    import numpy as np
    from opentpu import rtlsim
    from opentpu.llm.qwen3 import Spec,Engine,device_config
    from examples.opentpu.standard_inference import make_backend,run_engine
    spec=Spec(hidden=128,layers=1,n_q=1,n_kv=1,head_dim=128,ffn=128,vocab=128)
    rng=np.random.default_rng(41)
    weights={'model.embed_tokens.weight':rng.normal(0,.05,(128,128)).astype(np.float32),
             'model.norm.weight':np.ones(128,np.float32)}
    prefix='model.layers.0.'
    for name in ['input_layernorm.weight','post_attention_layernorm.weight',
                 'self_attn.q_norm.weight','self_attn.k_norm.weight']:
      weights[prefix+name]=np.ones(128,np.float32)
    for name in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
                 'mlp.gate_proj','mlp.up_proj','mlp.down_proj']:
      weights[prefix+name+'.weight']=rng.normal(0,.02,(128,128)).astype(np.float32)
    cfg=device_config(spec,128,S=1,D=128,MCOLS=4,ACT_ROWS=16,
      ACT_BLOCKS=64,LANES=8,IMEM_WORDS=32768,TMEM_WORDS=65536)
    prompt=list(range(16))
    reference=Engine(spec,weights,128,cfg,backend='isa',pipeline=False,resident=False)
    expected=reference.generate(prompt,max_new=2)
    with (nullcontext(str(self.artifacts)) if self.artifacts else
          tempfile.TemporaryDirectory()) as directory:
      Path(directory).mkdir(parents=True,exist_ok=True)
      output=Path(directory)
      options=SimpleNamespace(output=output,lpn_compiler=self.compiler,
        batch_rows=8,reuse_timing=True,command_capacity=65536,rtl_timeout=600)
      rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='4'
      rtlsim.VFLAGS+=['-CFLAGS','-O3']
      engine,actual=run_engine(spec,cfg,128,prompt,2,
        make_backend(output,len(prompt),self.compiler,options),weights)
      self.assertEqual(actual,expected)
      pre=[x for x in engine.backend.records if x['phase']=='prefill']
      self.assertEqual(sum(x['rows'] for x in pre),len(prompt))
      self.assertGreater(len(pre),1)
      self.assertTrue(any(x['rows']>1 for x in pre))
      self.assertEqual([x['logit_rows_read'] for x in pre],[0]*(len(pre)-1)+[1])
      for record in engine.backend.records:
        self.assertTrue(record['all_dram_and_tmem_bit_exact'])
        self.assertTrue(record['identical_input_files'])
        self.assertTrue(record['zero_rollbacks'])
        self.assertLessEqual(abs(record['relative_cycle_error']),.05)
        record['numerical_validation']='Independent ISA versus RTL; synthetic quantized weights fixture'
      for sid,state in enumerate(reference.backend.machine.slices):
        self.assertTrue(np.array_equal(engine.backend.drams[sid],state.dram[:len(engine.backend.drams[sid])]))
      report={
        'scope':'Small synthetic numerical fixture through unmodified upstream Engine.generate; not full Qwen checkpoint validation',
        'prompt_rows':[x['rows'] for x in pre],'generated_ids':actual,
        'kernels':len(engine.backend.records),
        'rtl_cycles':sum(x['rtl_cycles'] for x in engine.backend.records),
        'lpn_cycles':sum(x['lpn_cycles'] for x in engine.backend.records),
        'max_cycle_error':max(abs(x['relative_cycle_error']) for x in engine.backend.records)}
      from examples.opentpu.e2e import atomic_json,digest
      from examples.opentpu import REVISION
      from dataclasses import asdict
      atomic_json(output/'fixture-report.json',dict(report,model=asdict(spec),config=asdict(cfg),
        seed=41,prompt=prompt,records=engine.backend.records,
        upstream_revision=REVISION,compiler_sha256=digest(self.compiler),
        runtime_sha256=digest(self.compiler.resolve().parent/'libLPNNativeRuntime.a'),
        source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):digest(p)
          for p in [Path(__file__).resolve(),*sorted((Path(__file__).resolve().parents[1]/'examples/opentpu').glob('*.py'))]}))
      print('shared_inference_validation',report,flush=True)


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--opentpu-root',type=Path,required=True)
  p.add_argument('--compiler',type=Path)
  p.add_argument('--rtl',action='store_true')
  p.add_argument('--artifacts',type=Path,help='Retain numerical fixture inputs, traces and reports')
  args=p.parse_args()
  if args.rtl and args.compiler is None:p.error('--rtl requires --compiler')
  use_checkout(args.opentpu_root)
  StandardInferenceTest.compiler=args.compiler
  StandardInferenceTest.rtl=args.rtl
  StandardInferenceTest.artifacts=args.artifacts
  unittest.main(argv=[__file__],verbosity=2)
