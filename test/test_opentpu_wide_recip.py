"""Costed reciprocal proposals: finite ports, fallback and native timing."""
import argparse
from dataclasses import replace,asdict
from pathlib import Path
import tempfile
import unittest
from examples.opentpu.model import Timing,reciprocal_lookup_width
from examples.opentpu.compact import CompactOpenTPU
from examples.opentpu.shape_workload import resolve_static
from examples.opentpu.workload import use_checkout
from examples.opentpu.tpu_dse import Architecture,resources
from examples.opentpu.rtl_evolution.parallel_inference import validate_variant


class WideReciprocalTest(unittest.TestCase):
  tool=None

  def test_geometry_falls_back_and_rtl_replay_rejects_proposal(self):
    from opentpu import isa as I
    bulk=I.vop(I.V_RECIP,16384,0,8192,8,192,192,192,192)
    short=I.vop(I.V_RECIP,16384,0,8192,1,3,3,3,3)
    self.assertEqual(reciprocal_lookup_width(bulk,I,Timing(reciprocal_lookup_lanes=4)),0)
    self.assertEqual(reciprocal_lookup_width(bulk,I,Timing(reciprocal_lookup_lanes=16)),16)
    self.assertEqual(reciprocal_lookup_width(short,I,Timing(reciprocal_lookup_lanes=2)),2)
    for lanes in (1,3,32,True):
      with self.assertRaises(ValueError):Timing(reciprocal_lookup_lanes=lanes)
    with self.assertRaises(ValueError):Timing(reciprocal_lookup_lanes=2,scalar_recip_lookup=True)
    with self.assertRaisesRegex(ValueError,'no matching RTL'):
      validate_variant(asdict(Timing(reciprocal_lookup_lanes=16)),{})

  def test_native_matches_python_and_wide_dependent_chain_improves(self):
    from opentpu import Config,isa as I
    from examples.opentpu.native_timing import ReusableTiming
    cfg=Config(D=64,MCOLS=1,LANES=64,ACT_ROWS=16,DRAM_BYTES=1<<20)
    program=[I.vop(I.V_RECIP,16384,0,8192,8,192,192,192,192),
             I.vop(I.V_ADD,24576,16384,8192,8,192,192,192,192),
             I.vop(I.V_RECIP,32768,24576,8192,1,3,3,3,3),I.halt()]
    work=resolve_static(cfg,[program]);cycles=[]
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);runner=ReusableTiming(self.tool,root/'cache',capacity=128) if self.tool else None
      for lanes in (0,4,16):
        timing=Timing(dram_bytes_per_cycle=64,reciprocal_lookup_lanes=lanes)
        sim=CompactOpenTPU(work,timing).net.python_simulator();ticks=sim.run(max_time=100000)
        self.assertEqual(sim.place_dicts('program_completed'),[{}]);cycles.append(ticks)
        if runner:
          prepared=runner.prepare(work,timing,root/f'{lanes}.tables')
          self.assertEqual(prepared.run()[0],ticks)
      self.assertLess(cycles[2],cycles[1]);self.assertLess(cycles[1],cycles[0])

  def test_table_cost_counts_dual_port_replicas(self):
    from opentpu import Config
    cfg=asdict(Config(D=64,MCOLS=4,LANES=64))
    base=Architecture('control',128,1,2816)
    original=resources(base,cfg,200)['onchip_bytes_lower_bound']
    for lanes in (2,4,8,16):
      cost=resources(replace(base,reciprocal_lookup_lanes=lanes),cfg,200)
      self.assertEqual(cost['onchip_bytes_lower_bound']-original,(lanes//2)*2123264)
      self.assertEqual(cost['reciprocal_predictor_integer_multipliers'],lanes)
      self.assertFalse(cost['wide_reciprocal_rtl_implemented'])


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--tool')
  a=p.parse_args();use_checkout(a.opentpu_root);WideReciprocalTest.tool=a.tool
  unittest.main(argv=[__file__],verbosity=2)
