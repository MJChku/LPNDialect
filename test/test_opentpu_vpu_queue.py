import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from examples.opentpu.model import Timing
from examples.opentpu.compact import CompactOpenTPU
from examples.opentpu.timeline import TimelineSimulator
from examples.opentpu.workload import use_checkout
from examples.opentpu.shape_workload import resolve_static


class VPUQueueTest(unittest.TestCase):
  tool=None

  def test_accepted_fifo_does_not_reselect_a_later_ready_older_command(self):
    from opentpu import Config,isa as I
    cfg=Config(D=64,MCOLS=1,LANES=64,DRAM_BYTES=1<<20)
    program=[I.vop(I.V_RSQRT,8192,0,0,1,64,64,64,64),
             I.vop(I.V_ADD,12288,8192,0,1,64,64,64,64),
             I.vop(I.V_EXP2,16384,0,0,1,64,64,64,64),
             I.vop(I.V_COPY,20480,0,0,1,64,64,64,64),I.halt()]
    work=resolve_static(cfg,[program]);orders=[]
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp)
      for depth in (0,2):
        timing=Timing(dram_bytes_per_cycle=64,vector_command_queue=depth)
        model=CompactOpenTPU(work,timing);sim=TimelineSimulator(model.net)
        ticks=sim.run(max_time=100000);self.assertEqual(sim.place_dicts('program_completed'),[{}])
        orders.append(sorted(range(4),key=lambda i:sim.command_times[0,i]['started']))
        if depth:
          for i in range(4):self.assertGreater(sim.command_times[0,i]['started'],sim.command_times[0,i]['accepted'])
        if self.tool:
          from examples.opentpu.native_timing import ReusableTiming
          prepared=ReusableTiming(self.tool,root/'cache',capacity=128).prepare(work,timing,root/f'{depth}.tables')
          self.assertEqual(prepared.run()[0],ticks)
    self.assertLess(orders[0].index(1),orders[0].index(3))
    self.assertEqual(orders[1],[0,2,3,1])


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--tool')
  a=p.parse_args();use_checkout(a.opentpu_root);VPUQueueTest.tool=a.tool
  unittest.main(argv=[__file__],verbosity=2)
