import argparse
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from examples.opentpu.model import Timing,matrix_scale_requests
from examples.opentpu.compact import CompactOpenTPU
from examples.opentpu.timeline import TimelineSimulator
from examples.opentpu.shape_workload import resolve_static
from examples.opentpu.workload import use_checkout


class ScaleLineTest(unittest.TestCase):
  tool=None

  def test_rsc_repeated_lines_and_sw_only_requests(self):
    from opentpu import isa as I
    rsc=I.mm(0,0,8192,2,9,576,2,1,0,4,rsc=True)
    self.assertEqual(matrix_scale_requests(rsc,I,0,128),9)
    unit=I.mm(0,0,8192,2,9,576,2,1,0,4,unit=True)
    self.assertEqual(matrix_scale_requests(unit,I,0,128),0)
    # Sw at 28, block 0 at 32: its first request needs Sw separately.
    split=I.mm(0,28,8192,3,2,128,3,1,0,4)
    self.assertEqual(matrix_scale_requests(split,I,0,1),3)
    # The previous row's last line already carried the next row's Sw.
    self.assertEqual(matrix_scale_requests(split,I,1,1),2)
    self.assertEqual(matrix_scale_requests(split,I,1,128),2)
    self.assertEqual(matrix_scale_requests(split,I,8,1),2)
    self.assertEqual(matrix_scale_requests(split,I,8,128),3)
    packed=I.mm(0,0,8192,1,9,320,1,1,0,8,wf=I.W4I,rsc=True)
    self.assertEqual(matrix_scale_requests(packed,I,0,128),5)

  def test_physical_reads_consume_shared_link_and_native_agrees(self):
    from opentpu import Config,isa as I
    from examples.opentpu.native_timing import ReusableTiming
    cfg=Config(D=64,MCOLS=1,LANES=64,DRAM_BYTES=1<<20)
    program=[I.mm(0,65536,8192,192,9,576,192,1,0,16),I.halt()]
    work=resolve_static(cfg,[program]);common=Timing(batch_rows=128,matrix_row_lanes=128,
        fifo_chunks=16384,scale_lanes=128,dram_bytes_per_cycle=2816,dram_port_bytes=2816,unified_dram=True)
    cycles=[]
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp)
      for flag in (False,True):
        timing=replace(common,physical_scale_lines=flag)
        model=CompactOpenTPU(work,timing);sim=TimelineSimulator(model.net);ticks=sim.run(max_time=100000)
        self.assertEqual(sim.place_dicts('program_completed'),[{}]);cycles.append(ticks)
        if flag:self.assertEqual(sim.busy['s0_dram_port_b'][0][1]-sim.busy['s0_dram_port_b'][0][0],40)
        if self.tool:
          prepared=ReusableTiming(self.tool,root/'cache',capacity=128).prepare(work,timing,root/f'{flag}.tables')
          self.assertEqual(prepared.run()[0],ticks)
    self.assertGreater(cycles[1],cycles[0])


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--tool')
  a=p.parse_args();use_checkout(a.opentpu_root);ScaleLineTest.tool=a.tool
  unittest.main(argv=[__file__],verbosity=2)
