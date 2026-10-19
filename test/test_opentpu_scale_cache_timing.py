"""Cache hits preserve decoder work; bounded replacement pacing leaves HBM free."""
import argparse
from pathlib import Path
import tempfile
import unittest

from examples.opentpu.workload import use_checkout
from examples.opentpu.scale_cache_timing import scale_addresses,cache_metadata
from examples.opentpu.shape_workload import resolve_static
from examples.opentpu.model import Timing,matrix_scale_requests


class ScaleTimingTest(unittest.TestCase):
  tool=None

  def test_queries_match_uncached_protocol_and_pending_hits_coalesce(self):
    from opentpu import Config,isa as I,waits as W
    cfg=Config(D=64,LANES=64,DRAM_BYTES=1<<20)
    for k0 in (0,24,28,63):
      ins=I.mm(0,65536,8192,4,3,192,4,1,0,256,k0=k0)
      for row in range(4):
        self.assertEqual(len(scale_addresses(ins,I,row,2)),matrix_scale_requests(ins,I,row,2))
    ins=I.mm(0,65536,8192,4,3,192,4,1,0,256,rsc=True)
    work=resolve_static(cfg,[[ins,I.halt()]])
    meta=cache_metadata(work.commands[0],I,W,2,16,8)
    self.assertEqual(meta['requests'],12);self.assertEqual(meta['backing_reads'],4)
    self.assertEqual(meta['pacing'][0],[10,0,3,0])
    self.assertEqual(meta['data_bytes'],1024)

  def test_disjoint_write_invalidates_and_bad_geometry_is_rejected(self):
    from opentpu import Config,isa as I,waits as W
    cfg=Config(D=64,LANES=64,DRAM_BYTES=1<<20)
    ins=I.mm(0,65536,8192,2,3,192,4,1,0,4,rsc=True)
    def misses(write):
      work=resolve_static(cfg,[[ins]+([I.st(262144,0,1)] if write else [])+[ins,I.halt()]])
      return cache_metadata(work.commands[0],I,W,2,16,8)['backing_reads']
    self.assertEqual(misses(False),2);self.assertEqual(misses(True),4)
    with self.assertRaisesRegex(ValueError,'int8'):
      scale_addresses(I.mm(0,65536,8192,2,3,192,4,1,0,4,wf=I.W4F),I,0,2)
    with self.assertRaises(ValueError):Timing(scale_cache_entries=16)

  def test_native_matches_and_replacement_does_not_hold_hbm(self):
    from opentpu import Config,isa as I
    from examples.opentpu.compact import CompactOpenTPU
    from examples.opentpu.timeline import TimelineSimulator
    from examples.opentpu.native_timing import ReusableTiming
    cfg=Config(D=64,MCOLS=1,LANES=64,DRAM_BYTES=1<<20)
    ins=I.mm(0,65536,8192,4,3,192,4,1,0,256,rsc=True)
    work=resolve_static(cfg,[[ins,I.halt()]])
    timing=Timing(batch_rows=2,matrix_row_lanes=2,fifo_chunks=128,scale_lanes=2,
        dram_bytes_per_cycle=1<<20,dram_port_bytes=1<<20,unified_dram=True,
        matrix_retirement=True,physical_scale_lines=True,scale_cache_entries=16)
    sim=TimelineSimulator(CompactOpenTPU(work,timing).net);ticks=sim.run(max_time=10000)
    self.assertEqual(sim.place_dicts('program_completed'),[{}]);self.assertFalse(sim.held)
    def busy(name):return sum(b-a for a,b in sim.busy[name])
    self.assertEqual(busy('s0_dram_port_b'),2)
    self.assertEqual(busy('s0_dram_port_a'),6)
    self.assertEqual(busy('s0_mxu_0_scale_cache_pace'),13)
    if self.tool:
      with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);p=ReusableTiming(self.tool,root/'cache',capacity=128).prepare(work,timing,root/'tables')
        self.assertEqual(p.run()[0],ticks)

  def test_weight_and_scale_hits_do_not_take_hbm_credit(self):
    from opentpu import Config,isa as I
    from examples.opentpu.compact import CompactOpenTPU
    from examples.opentpu.timeline import TimelineSimulator
    from examples.opentpu.native_timing import ReusableTiming
    cfg=Config(D=64,MCOLS=1,LANES=64,DRAM_BYTES=1<<20)
    ins=I.mm(0,65536,8192,2,3,192,4,1,0,4,rsc=True)
    work=resolve_static(cfg,[[ins,ins,I.halt()]])
    timing=Timing(batch_rows=2,matrix_row_lanes=2,fifo_chunks=128,scale_lanes=2,
        dram_bytes_per_cycle=1<<20,dram_port_bytes=1<<20,unified_dram=True,
        matrix_retirement=True,physical_scale_lines=True,scale_cache_entries=16,pinned_cache_bytes=65536)
    profile=dict(version=1,base=0,cache_bytes=65536,hit_bytes_by_command_row=[[192,192],[192,192],[]])
    model=CompactOpenTPU(work,timing,prefetch_profile=profile)
    sim=TimelineSimulator(model.net);ticks=sim.run(max_time=10000)
    self.assertEqual(sim.place_dicts('program_completed'),[{}]);self.assertFalse(sim.held)
    self.assertEqual(sum(b-a for a,b in sim.busy['s0_dram_port_b']),1)
    self.assertEqual(sum(b-a for a,b in sim.busy['s0_dram_port_a']),6)
    self.assertEqual(model.statistics()['scale_cache'][0]['requests'],12)
    self.assertEqual(model.statistics()['scale_cache'][0]['backing_reads'],2)
    if self.tool:
      with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);p=ReusableTiming(self.tool,root/'cache',capacity=128).prepare(work,timing,root/'tables',prefetch_profile=profile)
        self.assertEqual(p.run()[0],ticks)


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--tool',type=Path)
  a=p.parse_args();use_checkout(a.opentpu_root);ScaleTimingTest.tool=a.tool
  unittest.main(argv=[__file__],verbosity=2)
