"""Lookup quantizer timing against measured RTL and native dataflow execution."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

from examples.opentpu.model import OpenTPU,Timing
from examples.opentpu.compact import CompactOpenTPU
from examples.opentpu.shape_workload import resolve_static
from examples.opentpu.workload import use_checkout


class LookupScaleTest(unittest.TestCase):
  tool=None
  report=None

  def test_timing_matches_every_production_quantizer_case(self):
    if not self.report:self.skipTest('requires measured lookup quantizer report')
    from opentpu import Config,isa as I
    evidence=json.loads(Path(self.report).read_text())
    self.assertEqual(evidence['status'],'complete');self.assertTrue(evidence['lookup_scale'])
    self.assertEqual(len(evidence['records']),28)
    model=object.__new__(OpenTPU)
    model.workload=SimpleNamespace(config=Config(D=64,MCOLS=1,LANES=64))
    model.timing=Timing(quant_row_banks=8,quant_store_buffered=True,quant_scale_lookup=True,
                        scalar_store_bytes=256,unified_dram=True)
    memory=SimpleNamespace(b=object(),sw=object())
    for record in evidence['records']:
      name=record['name']
      if match:=re.fullmatch(r'qst_r(\d+)_kb(\d+)_row(\d)_half(\d)_es(\d+)',name):
        rows,kb,row,half,es=map(int,match.groups())
        ins=I.qst(0,65537,700000,rows,kb,kb*64,kb*64*es+3,es,row=bool(row),half=bool(half))
      elif match:=re.fullmatch(r'qact_r(\d+)_kb(\d+)',name):
        rows,kb=map(int,match.groups());ins=I.qact(0,rows,2,kb,kb*64)
      elif name=='qact_prescaled':ins=I.qact(0,8,2,9,576,cscale=31000,rscale=31900)
      elif name=='qact_dup':ins=I.qact(0,8,2,3,192,dup=True)
      elif name=='qst_ties':ins=I.qst(22000,65537,700000,8,3,192,193,1,row=True)
      else:self.fail('unknown measured gate case')
      with self.subTest(name=name):
        self.assertTrue(record['all_state_bit_exact'])
        self.assertEqual(model._simple_timing(ins,memory,I)[0],record['rtl_cycles'])

  def test_native_matches_python_and_retains_activation_readiness(self):
    from opentpu import Config,isa as I
    from examples.opentpu.native_timing import ReusableTiming
    cfg=Config(D=64,MCOLS=1,LANES=64,ACT_ROWS=16,DRAM_BYTES=1<<20)
    program=[I.qact(0,8,0,9,576),I.mm(65536,0,16384,9,9,576,12,8,0,4,unit=True),
             I.qst(16384,0x80000,0x90000,8,1,12,64,1),I.halt()]
    work=resolve_static(cfg,[program])
    fast=Timing(quant_row_banks=8,quant_store_buffered=True,quant_scale_lookup=True,
        activation_streaming=True,scalar_store_bytes=256,unified_dram=True,
        matrix_row_lanes=4,batch_rows=4,fifo_chunks=512,dram_bytes_per_cycle=64)
    times=[]
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);runner=ReusableTiming(self.tool,root/'cache',capacity=128) if self.tool else None
      for timing in (replace(fast,quant_scale_lookup=False),fast):
        model=CompactOpenTPU(work,timing);sim=model.net.python_simulator()
        ticks=sim.run(max_time=100000);times.append(ticks)
        self.assertEqual(sim.place_dicts('program_completed'),[{}])
        if runner:
          prepared=runner.prepare(work,timing,root/f'{timing.quant_scale_lookup}.tables')
          self.assertEqual(prepared.run()[0],ticks)
      self.assertLess(times[1],times[0])
    with self.assertRaises(ValueError):Timing(quant_scale_lookup=True)


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True)
  p.add_argument('--tool');p.add_argument('--gate-report',type=Path)
  a=p.parse_args();use_checkout(a.opentpu_root);LookupScaleTest.tool=a.tool;LookupScaleTest.report=a.gate_report
  unittest.main(argv=[__file__],verbosity=2)
