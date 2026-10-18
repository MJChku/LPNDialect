"""Bounded vector results, shared drain bandwidth and native scheduling."""
import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from examples.opentpu.model import Timing
from examples.opentpu.compact import CompactOpenTPU
from examples.opentpu.shape_workload import resolve_static
from examples.opentpu.workload import use_checkout


class VectorPoolTest(unittest.TestCase):
  tool=None

  def work(self,functions,cols=64):
    from opentpu import Config,isa as I
    cfg=Config(D=64,MCOLS=1,LANES=64,ACT_ROWS=16,DRAM_BYTES=1<<20)
    program=[I.vop(f,8192+i*4096,0,4096,1,cols,cols,cols,cols) for i,f in enumerate(functions)]
    return resolve_static(cfg,[program+[I.halt()]])

  def run_model(self,work,timing):
    model=CompactOpenTPU(work,timing)
    sim=model.net.python_simulator();ticks=sim.run(max_time=100000)
    self.assertEqual(sim.place_dicts('program_completed'),[{}])
    if self.tool:
      from examples.opentpu.native_timing import ReusableTiming
      with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        prepared=ReusableTiming(self.tool,root/'cache',capacity=128).prepare(work,timing,root/'inputs')
        self.assertEqual(prepared.run()[0],ticks)
    return ticks

  def test_parallel_long_and_short_functions(self):
    from opentpu import isa as I
    work=self.work([I.V_RSQRT,I.V_COPY,I.V_RECIP,I.V_ADD,I.V_EXP2,I.V_MUL])
    common=Timing(dram_bytes_per_cycle=64,vector_result_chunks=64)
    serial=self.run_model(work,common)
    parallel=self.run_model(work,replace(common,vector_engines=4))
    self.assertLess(parallel,serial)

  def test_result_store_backpressure_and_capacity_check(self):
    from opentpu import isa as I
    work=self.work([I.V_ADD]*8,cols=512)
    common=Timing(dram_bytes_per_cycle=64,vector_result_chunks=8)
    tiny=self.run_model(work,common)
    bigger=self.run_model(work,replace(common,vector_result_chunks=128))
    self.assertGreater(tiny,bigger)
    with self.assertRaisesRegex(ValueError,'finite result store'):
      CompactOpenTPU(work,replace(common,vector_result_chunks=4))
    with self.assertRaises(ValueError):Timing(vector_engines=2)

  def test_compatible_selection_can_bypass_an_independent_blocked_rank(self):
    from opentpu import isa as I
    work=self.work([I.V_RSQRT,I.V_COPY]*4)
    normal=self.run_model(work,Timing(dram_bytes_per_cycle=64))
    compatible=self.run_model(work,Timing(dram_bytes_per_cycle=64,vector_compatible_issue=True))
    self.assertLess(compatible,normal)

  def test_shared_output_port_bounds_aggregate_throughput(self):
    from opentpu import isa as I
    work=self.work([I.V_COPY]*8,cols=4096)
    ticks=self.run_model(work,Timing(dram_bytes_per_cycle=64,vector_result_chunks=128,vector_engines=4))
    self.assertGreaterEqual(ticks,8*64)


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--tool')
  a=p.parse_args();use_checkout(a.opentpu_root);VectorPoolTest.tool=a.tool
  unittest.main(argv=[__file__],verbosity=2)
