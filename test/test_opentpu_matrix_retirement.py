import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from examples.opentpu.compact import CompactOpenTPU
from examples.opentpu.model import Timing
from examples.opentpu.shape_workload import resolve_static
from examples.opentpu.timeline import TimelineSimulator
from examples.opentpu.workload import use_checkout


class MatrixRetirementTest(unittest.TestCase):
  tool=None

  def test_overlapped_integer_work_has_ordered_serial_epilogues(self):
    from opentpu import Config,isa as I
    from examples.opentpu.native_timing import ReusableTiming
    cfg=Config(D=64,MCOLS=1,LANES=64,ACT_ROWS=16,DRAM_BYTES=1<<20)
    program=[I.mm(0,0,8192,9,1,64,9,4,0,4,unit=True),
             I.mm(65536,0,16384,9,1,64,9,4,0,4,unit=True),I.halt()]
    work=resolve_static(cfg,[program])
    common=Timing(batch_rows=128,matrix_row_lanes=128,fifo_chunks=16384,scale_lanes=128,
        dram_bytes_per_cycle=2816,dram_port_bytes=2816,epilogue_lanes=1)
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);times=[]
      for flag in (False,True):
        timing=replace(common,matrix_retirement=flag)
        model=CompactOpenTPU(work,timing);sim=TimelineSimulator(model.net)
        ticks=sim.run(max_time=100000)
        self.assertEqual(sim.place_dicts('program_completed'),[{}]);self.assertFalse(sim.held)
        times.append([sim.command_times[0,i]['completed'] for i in (0,1)])
        if self.tool:
          prepared=ReusableTiming(self.tool,root/'cache',capacity=128).prepare(work,timing,root/f'{flag}.tables')
          self.assertEqual(prepared.run()[0],ticks)
      self.assertEqual(times[1][1]-times[1][0],15)
      self.assertLess(times[0][1]-times[0][0],15)

  def test_streaming_multiple_batches_and_accumulation_keep_slots_bounded(self):
    from opentpu import Config,isa as I
    from examples.opentpu.native_timing import ReusableTiming
    cfg=Config(D=64,MCOLS=1,LANES=64,ACT_ROWS=16,DRAM_BYTES=1<<20)
    program=[I.qact(0,4,0,3,192),
             I.mm(0,0,8192,129,3,192,129,4,0,4,unit=True),
             I.mm(65536,0,8192,129,3,192,129,4,0,4,unit=True,acc=True),I.halt()]
    work=resolve_static(cfg,[program]);timing=Timing(batch_rows=32,matrix_row_lanes=32,
        fifo_chunks=4096,scale_lanes=32,dram_bytes_per_cycle=2816,dram_port_bytes=2816,
        quant_row_banks=8,quant_store_buffered=True,activation_streaming=True,matrix_retirement=True)
    sim=TimelineSimulator(CompactOpenTPU(work,timing).net);ticks=sim.run(max_time=100000)
    self.assertEqual(sim.place_dicts('program_completed'),[{}]);self.assertFalse(sim.held)
    self.assertLess(sim.command_times[0,1]['completed'],sim.command_times[0,2]['completed'])
    if self.tool:
      with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);prepared=ReusableTiming(self.tool,root/'cache',capacity=128).prepare(work,timing,root/'tables')
        self.assertEqual(prepared.run()[0],ticks)


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--tool')
  a=p.parse_args();use_checkout(a.opentpu_root);MatrixRetirementTest.tool=a.tool
  unittest.main(argv=[__file__],verbosity=2)
