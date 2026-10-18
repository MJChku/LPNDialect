"""An optional observer preserves command timing and tracks held resources."""
import argparse
from pathlib import Path
import tempfile
import unittest
from examples.opentpu.timeline import TimelineSimulator
from examples.opentpu.compact import CompactOpenTPU
from examples.opentpu.model import Timing
from examples.opentpu.shape_workload import resolve_static
from examples.opentpu.workload import use_checkout


class TimelineTest(unittest.TestCase):
  tool=None

  def test_streaming_and_vector_observation_preserves_simulation(self):
    from opentpu import Config,isa as I
    cfg=Config(D=64,MCOLS=1,LANES=64,ACT_ROWS=16,DRAM_BYTES=1<<20)
    program=[I.qact(0,8,0,3,192),I.mm(65536,0,8192,9,3,192,12,8,0,4,unit=True),
             I.vop(I.V_RECIP,16384,8192,0,1,3,3,3,1),I.halt()]
    work=resolve_static(cfg,[program]);timing=Timing(dram_bytes_per_cycle=64,batch_rows=4,matrix_row_lanes=4,
        fifo_chunks=512,scale_lanes=4,quant_row_banks=8,quant_store_buffered=True,
        activation_streaming=True,scalar_recip_lookup=True,unified_dram=True)
    model=CompactOpenTPU(work,timing)
    plain=model.net.python_simulator();expected=plain.run(max_time=100000)
    observed=TimelineSimulator(model.net)
    self.assertEqual(observed.run(max_time=100000),expected)
    self.assertEqual(observed.place_dicts('program_completed'),[{}]);self.assertFalse(observed.held)
    self.assertEqual(len(observed.command_times),len(program))
    for index in range(len(program)):
      self.assertIn('completed',observed.command_times[0,index])
    self.assertIn('compute_started',observed.command_times[0,1])
    for intervals in observed.busy.values():
      self.assertTrue(all(0<=start<end<=expected for start,end in intervals))
      self.assertLessEqual(sum(end-start for start,end in intervals),expected)
    # Resetting and observing again must not retain any prior trace.
    before={key:dict(value) for key,value in observed.command_times.items()}
    self.assertEqual(observed.run(max_time=100000),expected)
    self.assertEqual(dict(observed.command_times),before)
    if self.tool:
      from examples.opentpu.native_timing import ReusableTiming
      with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);prepared=ReusableTiming(self.tool,root/'cache',capacity=128).prepare(work,timing,root/'inputs')
        self.assertEqual(prepared.run()[0],expected)


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--tool')
  a=p.parse_args();use_checkout(a.opentpu_root);TimelineTest.tool=a.tool
  unittest.main(argv=[__file__],verbosity=2)
