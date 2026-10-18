"""ACT row readiness, retained hazards and native event-driven execution."""
import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from examples.opentpu.model import Timing
from examples.opentpu.compact import CompactOpenTPU,dependencies
from examples.opentpu.activation_stream import quant_row_first_pops,stream_producers
from examples.opentpu.shape_workload import resolve_static
from examples.opentpu.workload import use_checkout


class ActivationTest(unittest.TestCase):
  tool=None

  def workload(self,m=8,kb=9,war=False,partial=False):
    from opentpu import Config,isa as I
    cfg=Config(D=64,MCOLS=1,LANES=64,ACT_ROWS=16,DRAM_BYTES=1<<20)
    out=0 if war else 20000
    prog=[I.qact(0,m,0,kb,kb*64)]
    if partial:prog+=[I.qact(8192,m,2,kb-2,kb*64)]
    prog+=[I.mm(65536,0,out,9,kb,kb*64,12,m,0,4,unit=True),I.halt()]
    return resolve_static(cfg,[prog])

  def timing(self):
    return Timing(activation_streaming=True,quant_row_banks=8,matrix_row_lanes=4,
                  batch_rows=4,fifo_chunks=256,dram_bytes_per_cycle=64)

  def test_row_schedule_finite_banks_and_fifo(self):
    self.assertEqual(quant_row_first_pops(1,1,1024,8),[44])
    self.assertEqual(quant_row_first_pops(8,9,1024,8),list(range(52,123,10)))
    large=quant_row_first_pops(9,3,1024,8)
    tiny=quant_row_first_pops(9,3,4,8)
    self.assertGreater(tiny[-1],large[-1])
    with self.assertRaises(ValueError):quant_row_first_pops(1,9,4,8)
    with self.assertRaises(ValueError):Timing(activation_streaming=True)

  def test_only_fully_covered_act_raw_becomes_partial_dependency(self):
    from opentpu import isa as I,waits as W
    t=self.timing()
    for war,partial,expected in ((False,False,0),(True,False,-1),(False,True,-1)):
      work=self.workload(war=war,partial=partial);commands=work.commands[0]
      producers=stream_producers(commands,I,W,t);i=len(commands)-2
      self.assertEqual(producers[i],expected)
      deps,_=dependencies(commands,I,W,streaming_producers=producers)
      if expected>=0:self.assertNotIn(expected,deps[i])
      else:self.assertTrue(deps[i])
      self.assertTrue(deps[-1]) # HALT still drains every incomplete producer

  def test_python_and_native_progress_and_fallback(self):
    from examples.opentpu.native_timing import ReusableTiming
    times=[]
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp)
      runner=ReusableTiming(self.tool,root/'cache',capacity=128) if self.tool else None
      for war,partial in ((False,False),(True,False),(False,True)):
        work=self.workload(war=war,partial=partial)
        t=self.timing()
        model=CompactOpenTPU(work,t);sim=model.net.python_simulator()
        ticks=sim.run(max_time=100000)
        self.assertEqual(sim.place_dicts('program_completed'),[{}])
        if runner:
          prepared=runner.prepare(work,t,root/f'{war}-{partial}.tables')
          self.assertEqual(prepared.run()[0],ticks)
        if not war and not partial:
          normal=CompactOpenTPU(work,replace(t,activation_streaming=False)).net.python_simulator().run(max_time=100000)
          self.assertLess(ticks,normal)
        times.append(ticks)
      if runner:
        work=self.workload(m=1,kb=3)
        prepared=runner.prepare(work,self.timing(),root/'different-shape.tables')
        self.assertTrue(prepared.cache_hit)
        sim=CompactOpenTPU(work,self.timing()).net.python_simulator()
        self.assertEqual(prepared.run()[0],sim.run(max_time=100000))


if __name__=='__main__':
  p=argparse.ArgumentParser()
  p.add_argument('--opentpu-root',type=Path,required=True)
  p.add_argument('--tool')
  a=p.parse_args()
  use_checkout(a.opentpu_root)
  ActivationTest.tool=a.tool
  unittest.main(argv=[__file__],verbosity=2)
