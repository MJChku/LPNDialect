"""Actual numerical routing, whole-expert ownership and hardware result combination."""
import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from examples.opentpu.workload import use_checkout


class DistributedTest(unittest.TestCase):
    tool=None

    def check(self,slices,bias=None):
        import numpy as np
        from opentpu import isa as I
        from examples.opentpu.glm47_distributed_hardware import Layout,numerical_case
        from examples.opentpu.hardware_commands import instruction_statistics
        from examples.opentpu.glm47_target import component_timing
        from examples.opentpu.compact import CompactOpenTPU
        from examples.opentpu.native_timing import ReusableTiming
        layout=Layout(128,512,8,4,slices)
        first,work,images,ids,error=numerical_case(layout,bias)
        self.assertLess(error,.02)
        # One shared expert, not one per slice, and every actual selected
        # token visits exactly four routed experts.
        self.assertEqual(instruction_statistics(work)['matrix_dot_macs'],3*4*128*512*5)
        for sid,program in enumerate(work.programs):
            gathered=[p for p in program if p.op==I.GATHER]
            self.assertEqual(len(gathered),4 if slices>1 else 0)
            if slices>1:self.assertTrue(all(p.w[1]==0 for p in gathered))
            if sid and not any(e//layout.m.experts==sid for row in ids for e in row):
                self.assertFalse(any(p.op==I.MM for p in program))
        physical=replace(work,config=replace(work.config,MCOLS=2))
        timing=replace(component_timing(),dma_store_runs=True)
        sim=CompactOpenTPU(physical,timing).net.python_simulator();ticks=sim.run(max_time=1000000)
        self.assertEqual(sim.place_dicts('program_completed'),[{}])
        if self.tool:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);cap=1<<max(len(c) for c in work.commands).bit_length()
                native=ReusableTiming(self.tool,root/'cache',capacity=cap).prepare(physical,timing,root/'inputs')
                self.assertEqual(native.run()[0],ticks)
        return ids

    def test_two_owner_slices_use_real_routes_and_one_shared_expert(self):self.check(2)

    def test_empty_owners_still_participate_in_every_collective(self):
        import numpy as np
        ids=self.check(4,np.asarray([5]*4+[-5]*4,np.float32))
        self.assertTrue(all(set(row)==set(range(4)) for row in ids))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--tool',type=Path)
    a=p.parse_args();use_checkout(a.opentpu_root);DistributedTest.tool=a.tool
    unittest.main(argv=[__file__],verbosity=2)
