"""Independent MXUs must share the finite scratchpad output bandwidth."""
import argparse
from pathlib import Path
import tempfile
import unittest

from examples.opentpu.model import Timing
from examples.opentpu.compact import CompactOpenTPU
from examples.opentpu.timeline import TimelineSimulator
from examples.opentpu.shape_workload import resolve_static
from examples.opentpu.workload import use_checkout


class MatrixClustersTest(unittest.TestCase):
  tool=None

  def test_shared_output_bandwidth_streaming_and_nonstreaming(self):
    from opentpu import Config,isa as I
    from examples.opentpu.native_timing import ReusableTiming
    cfg=Config(D=64,MCOLS=8,ACT_ROWS=8,LANES=64,DRAM_BYTES=1<<20)
    for streaming in (False,True):
      for conflicting in (False,True):
        # Independent weight ranges and ACT reads, 512 output words each.
        program=[I.mm(i*4096,0,0 if conflicting else i*512,64,1,64,8,8,0,8,unit=True)
                 for i in range(16)]+[I.halt()]
        work=resolve_static(cfg,[program])
        timing=Timing(window=32,batch_rows=64,matrix_row_lanes=64,matrix_engines=16,
            fifo_chunks=4096,scale_lanes=1024,dram_bytes_per_cycle=1<<20,
            dram_port_bytes=1<<20,unified_dram=True,epilogue_lanes=4,
            activation_streaming=streaming,matrix_retirement=True,quant_row_banks=8)
        sim=TimelineSimulator(CompactOpenTPU(work,timing).net)
        ticks=sim.run(max_time=100000)
        self.assertEqual(sim.place_dicts('program_completed'),[{}]);self.assertFalse(sim.held)
        intervals=sim.busy['s0_matrix_result_port']
        self.assertEqual(len(intervals),16)
        self.assertEqual(sum(b-a for a,b in intervals),128)
        ordered=sorted(intervals)
        self.assertTrue(all(b<=c for (a,b),(c,d) in zip(ordered,ordered[1:])))
        self.assertGreaterEqual(ticks,128)
        if conflicting:
          # WAW commands cannot begin computation before predecessor completion.
          for idx in range(1,16):
            self.assertGreaterEqual(sim.command_times[0,idx]['compute_started'],
                                    sim.command_times[0,idx-1]['completed'])
        if self.tool:
          with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p=ReusableTiming(self.tool,root/'cache',capacity=128).prepare(work,timing,root/'tables')
            self.assertEqual(p.run()[0],ticks)

  def test_unimplemented_cluster_cannot_be_reported_as_rtl(self):
    from examples.opentpu.rtl_evolution.parallel_inference import validate_variant
    with self.assertRaisesRegex(ValueError,'multiple matrix engines'):
      validate_variant({'matrix_engines':2},{})


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--tool')
  a=p.parse_args();use_checkout(a.opentpu_root);MatrixClustersTest.tool=a.tool
  unittest.main(argv=[__file__],verbosity=2)
