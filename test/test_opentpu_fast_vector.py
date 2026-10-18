"""Proposal routing, real hazards, bounded results and shared write bandwidth."""
import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from examples.opentpu.model import Timing
from examples.opentpu.compact import CompactOpenTPU
from examples.opentpu.timeline import TimelineSimulator
from examples.opentpu.shape_workload import resolve_static
from examples.opentpu.workload import use_checkout
from examples.opentpu.rtl_evolution.parallel_inference import validate_variant


class FastVectorTest(unittest.TestCase):
  tool=None

  def observe(self,program,timing):
    from opentpu import Config,isa as I
    cfg=Config(D=64,MCOLS=1,LANES=64,DRAM_BYTES=1<<20)
    work=resolve_static(cfg,[program+[I.halt()]])
    sim=TimelineSimulator(CompactOpenTPU(work,timing).net)
    ticks=sim.run(max_time=100000)
    self.assertEqual(sim.place_dicts('program_completed'),[{}]);self.assertFalse(sim.held)
    if self.tool:
      from examples.opentpu.native_timing import ReusableTiming
      with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        prepared=ReusableTiming(self.tool,root/'cache',capacity=128).prepare(work,timing,root/'tables')
        self.assertEqual(prepared.run()[0],ticks)
    return ticks,sim

  def common(self):
    return Timing(dram_bytes_per_cycle=64,vector_result_chunks=128,vector_compatible_issue=True)

  def test_fill_copy_bypass_long_fp_command_without_duplicating_fp_engine(self):
    from opentpu import isa as I
    program=[I.vop(I.V_RSQRT,8192,0,4096,1,64,64,64,64),
             I.vop(I.V_FILL,16384,0,0,1,64,64,0,0),
             I.vop(I.V_COPY,24576,4096,0,1,64,64,64,0)]
    serial,_=self.observe(program,self.common())
    parallel,sim=self.observe(program,replace(self.common(),vector_fast_engine=True))
    self.assertLess(parallel,serial)
    self.assertLess(sim.command_times[0,1]['completed'],sim.command_times[0,0]['completed'])
    self.assertTrue(sim.busy['s0_vpu_fast_issue']);self.assertTrue(sim.busy['s0_vpu_0_issue'])

  def test_true_read_write_dependencies_are_preserved(self):
    from opentpu import isa as I
    program=[I.vop(I.V_RSQRT,8192,0,4096,1,64,64,64,64),
             I.vop(I.V_FILL,0,0,0,1,64,64,0,0),
             I.vop(I.V_COPY,16384,0,0,1,64,64,64,0)]
    _,sim=self.observe(program,replace(self.common(),vector_fast_engine=True))
    for previous,current in [(0,1),(1,2)]:
      self.assertGreaterEqual(sim.command_times[0,current]['started'],sim.command_times[0,previous]['completed'])

  def test_private_capacity_and_shared_result_port_are_charged(self):
    from opentpu import isa as I
    program=[I.vop(I.V_FILL if i%2 else I.V_ADD,8192+i*4096,0,4096,1,4096,4096,4096,4096)
             for i in range(8)]
    timing=replace(self.common(),vector_fast_engine=True)
    ticks,_=self.observe(program,timing)
    self.assertGreaterEqual(ticks,8*64)
    with self.assertRaisesRegex(ValueError,'finite result store'):
      self.observe(program,replace(timing,vector_result_chunks=32))
    with self.assertRaisesRegex(ValueError,'fast vector engine'):
      Timing(vector_fast_engine=True)
    with self.assertRaisesRegex(ValueError,'no matching RTL'):
      validate_variant(dict(vector_fast_engine=True),{})


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--tool')
  a=p.parse_args();use_checkout(a.opentpu_root);FastVectorTest.tool=a.tool
  unittest.main(argv=[__file__],verbosity=2)
