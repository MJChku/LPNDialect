import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from examples.opentpu.compact import CompactOpenTPU
from examples.opentpu.model import Timing,OpenTPU
from examples.opentpu.shape_workload import resolve_static
from examples.opentpu.workload import use_checkout
from examples.opentpu.rtl_evolution.parallel_inference import validate_variant


class PinnedTimingTest(unittest.TestCase):
  tool=None

  def fixture(self):
    from opentpu import Config,isa as I
    cfg=Config(D=64,MCOLS=1,LANES=64,ACT_ROWS=16,DRAM_BYTES=1<<20)
    program=[I.mm(0,0,8192,32,8,512,64,1,0,4,unit=True),
             I.mm(32768,0,16384,32,8,512,64,1,0,4,unit=True),I.halt()]
    work=resolve_static(cfg,[program])
    timing=Timing(batch_rows=32,matrix_row_lanes=32,matrix_engines=2,fifo_chunks=4096,
        scale_lanes=32,dram_bytes_per_cycle=256,dram_port_bytes=256,pinned_cache_bytes=65536)
    profile=dict(version=1,base=0,cache_bytes=65536,hit_bytes_by_command_row=[[512]*32,[512]*32,[]])
    return work,timing,profile

  def observe(self,work,timing,profile,root):
    sim=CompactOpenTPU(work,timing,prefetch_profile=profile).net.python_simulator()
    ticks=sim.run(max_time=100000)
    self.assertEqual(sim.place_dicts('program_completed'),[{}])
    if self.tool:
      from examples.opentpu.native_timing import ReusableTiming
      prepared=ReusableTiming(self.tool,root/'cache',capacity=128).prepare(work,timing,root/'inputs',prefetch_profile=profile)
      self.assertEqual(prepared.run()[0],ticks)
    return ticks

  def test_shared_cache_bandwidth_is_charged_and_native_agrees(self):
    work,timing,profile=self.fixture()
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp)
      slow=self.observe(work,replace(timing,cache_read_bytes_per_cycle=64),profile,root/'slow')
      fast=self.observe(work,replace(timing,cache_read_bytes_per_cycle=4096),profile,root/'fast')
      self.assertGreaterEqual(slow,32768/64)
      self.assertLess(fast,slow)
      miss=dict(profile,hit_bytes_by_command_row=[[0]*32,[0]*32,[]])
      cold=self.observe(work,timing,miss,root/'cold')
      original=self.observe(work,replace(timing,pinned_cache_bytes=0),None,root/'original')
      self.assertEqual(cold,original)

  def test_missing_or_invalid_metadata_and_unimplemented_rtl_are_rejected(self):
    work,timing,profile=self.fixture()
    with self.assertRaises(ValueError):CompactOpenTPU(work,timing)
    with self.assertRaises(ValueError):CompactOpenTPU(work,timing,prefetch_profile=dict(profile,cache_bytes=128))
    with self.assertRaises(ValueError):CompactOpenTPU(work,timing,prefetch_profile=dict(profile,base=65536))
    bad=dict(profile,hit_bytes_by_command_row=[[513]*32,[512]*32,[]])
    with self.assertRaises(ValueError):CompactOpenTPU(work,timing,prefetch_profile=bad)
    with self.assertRaisesRegex(ValueError,'pinned cache'):
      OpenTPU(work,replace(timing,matrix_engines=1))
    with self.assertRaisesRegex(ValueError,'no matching RTL'):
      validate_variant(dict(pinned_cache_bytes=65536),{})


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--tool')
  a=p.parse_args();use_checkout(a.opentpu_root);PinnedTimingTest.tool=a.tool
  unittest.main(argv=[__file__],verbosity=2)
