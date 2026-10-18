"""Multi-column waves wait for every useful ACT row, including partial groups."""
import argparse
from pathlib import Path
import tempfile
import unittest

from examples.opentpu.model import Timing
from examples.opentpu.compact import CompactOpenTPU
from examples.opentpu.timeline import TimelineSimulator
from examples.opentpu.shape_workload import resolve_static
from examples.opentpu.workload import use_checkout


class RowTrace(TimelineSimulator):
  def reset(self):
    super().reset();self.rows=[]

  def _commit_transaction(self):
    ctx=self._require_transaction()
    if self.definition.endswith('_advance_activation_row'):
      for place,token in ctx.taken:
        if place.name.endswith('_active_rows'):
          self.rows.append((token.get('id'),token.get('group'),self.current_time))
    super()._commit_transaction()


class MultiColumnTest(unittest.TestCase):
  tool=None

  def test_group_waits_for_last_useful_row_and_native_agrees(self):
    from opentpu import Config,isa as I
    from examples.opentpu.activation_stream import quant_row_first_pops
    from examples.opentpu.native_timing import ReusableTiming
    for cols in (1,2,4,8):
      cfg=Config(D=64,MCOLS=cols,ACT_ROWS=8,LANES=64,DRAM_BYTES=1<<20)
      program=[I.qact(0,5,0,3,192),I.mm(65536,262144,8192,8,3,192,11,5,0,8),I.halt()]
      work=resolve_static(cfg,[program])
      timing=Timing(batch_rows=8,matrix_row_lanes=8,fifo_chunks=512,scale_lanes=8,
          dram_bytes_per_cycle=2816,dram_port_bytes=2816,unified_dram=True,
          epilogue_lanes=max(1,cols//2),quant_row_banks=8,quant_store_buffered=True,
          activation_streaming=True,matrix_retirement=True,physical_scale_lines=True)
      sim=RowTrace(CompactOpenTPU(work,timing).net);ticks=sim.run(max_time=100000)
      self.assertEqual(sim.place_dicts('program_completed'),[{}]);self.assertFalse(sim.held)
      firsts=quant_row_first_pops(5,3,1024,8)
      start=sim.command_times[0,0]['started']
      self.assertEqual(len(sim.rows),(5+cols-1)//cols)
      for idx,group,clock in sim.rows:
        self.assertEqual(idx,1);last=min(5,(group+1)*cols)-1
        self.assertGreaterEqual(clock,start+firsts[last]+timing.fp_mul_latency+10)
      if self.tool:
        with tempfile.TemporaryDirectory() as tmp:
          root=Path(tmp);p=ReusableTiming(self.tool,root/'cache',capacity=128).prepare(work,timing,root/'tables')
          self.assertEqual(p.run()[0],ticks)


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--tool')
  a=p.parse_args();use_checkout(a.opentpu_root);MultiColumnTest.tool=a.tool
  unittest.main(argv=[__file__],verbosity=2)
