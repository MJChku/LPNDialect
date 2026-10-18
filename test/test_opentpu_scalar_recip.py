"""Scalar reciprocal tap selection, bulk fallback and native event execution."""
import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from examples.opentpu.model import OpenTPU,Timing,vector_rank
from examples.opentpu.compact import CompactOpenTPU
from examples.opentpu.shape_workload import resolve_static
from examples.opentpu.workload import use_checkout


class ScalarReciprocalTest(unittest.TestCase):
  tool=None

  def test_rank_tracks_real_tap_and_bulk_fallback(self):
    from opentpu import isa as I
    scalar=I.vop(I.V_RECIP,4096,0,8192,8,1,3,5,1)
    bulk=I.vop(I.V_RECIP,4096,0,8192,128,2,3,5,1)
    fast=Timing(scalar_recip_lookup=True)
    self.assertEqual(vector_rank(scalar,I),6)
    self.assertEqual(vector_rank(scalar,I,fast),1)
    self.assertEqual(vector_rank(bulk,I,fast),6)
    small=I.vop(I.V_RECIP,4096,0,8192,1,3,3,5,1)
    self.assertEqual(vector_rank(small,I,fast),1)
    self.assertEqual(vector_rank(scalar,I,replace(fast,fp_mul_latency=1,fp_add_latency=1)),2)

  def test_native_matches_compact_timing_for_mixed_functions(self):
    from opentpu import Config,isa as I
    from examples.opentpu.native_timing import ReusableTiming
    cfg=Config(D=64,MCOLS=1,LANES=64,ACT_ROWS=16,DRAM_BYTES=1<<20)
    program=[I.vop(I.V_RECIP,4096,0,8192,8,1,3,5,1),
             I.vop(I.V_ADD,12288,4096,8192,8,1,3,3,1),
             I.vop(I.V_RECIP,16384,0,8192,8,193,200,200,200),
             I.vop(I.V_RECIP,20000,0,8192,8,1,3,5,1),
             I.vop(I.V_EXP2,21000,0,8192,8,1,3,5,1),I.halt()]
    work=resolve_static(cfg,[program])
    common=Timing(dram_bytes_per_cycle=64,scalar_recip_lookup=True)
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);runner=ReusableTiming(self.tool,root/'cache',capacity=128) if self.tool else None
      for timing in (common,replace(common,fp_mul_latency=1,fp_add_latency=1)):
        model=CompactOpenTPU(work,timing);sim=model.net.python_simulator()
        ticks=sim.run(max_time=100000)
        self.assertEqual(sim.place_dicts('program_completed'),[{}])
        if runner:
          prepared=runner.prepare(work,timing,root/f'{timing.fp_add_latency}.tables')
          self.assertEqual(prepared.run()[0],ticks)
      with self.assertRaisesRegex(ValueError,'CompactOpenTPU'):
        OpenTPU(work,common)
      chain=resolve_static(cfg,[[I.vop(I.V_RECIP,4096,0,8192,8,3,3,5,1),
          I.vop(I.V_ADD,8192,4096,12288,8,3,3,3,3),
          I.vop(I.V_RECIP,12288,8192,0,8,3,3,3,3),I.halt()]])
      chain_fast=CompactOpenTPU(chain,common).net.python_simulator().run(max_time=100000)
      chain_slow=CompactOpenTPU(chain,replace(common,scalar_recip_lookup=False)).net.python_simulator().run(max_time=100000)
      self.assertLess(chain_fast,chain_slow)


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--tool')
  a=p.parse_args();use_checkout(a.opentpu_root);ScalarReciprocalTest.tool=a.tool
  unittest.main(argv=[__file__],verbosity=2)
