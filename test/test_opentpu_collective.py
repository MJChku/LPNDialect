"""Cross-slice BAR/GATHER synchronization on actual encoded hardware programs."""
import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from examples.opentpu.workload import use_checkout,execute


from examples.opentpu.collective_hardware import collective_case


class CollectiveTest(unittest.TestCase):
    tool=None

    def test_native_and_python_wait_for_all_producers_and_repeated_collectives(self):
        import numpy as np
        from examples.opentpu.compact import CompactOpenTPU
        from examples.opentpu.glm47_target import component_timing
        from examples.opentpu.native_timing import ReusableTiming
        for slices in (2,4):
            with self.subTest(slices=slices):
                cfg,programs,images,expected=collective_case(slices)
                work=execute(cfg,programs,images)
                for machine in work.machine.slices:
                    for stage in range(2):
                        at=16384+stage*32768
                        got=machine.dram[at:at+expected.nbytes].view(np.float32).reshape(expected.shape)
                        np.testing.assert_array_equal(got,expected)
                timing=component_timing();physical=replace(work,config=replace(cfg,MCOLS=2))
                model=CompactOpenTPU(physical,timing);sim=model.net.python_simulator()
                ticks=sim.run(max_time=100000)
                self.assertEqual(sim.place_dicts('program_completed'),[{}])
                self.assertEqual(len(model.collective.endpoints),slices)
                if self.tool:
                    with tempfile.TemporaryDirectory() as temp:
                        root=Path(temp)
                        native=ReusableTiming(self.tool,root/'cache',capacity=64).prepare(physical,timing,root/'inputs')
                        self.assertEqual(native.run()[0],ticks)

    def test_mismatched_peer_geometry_is_rejected(self):
        from opentpu import isa as I
        from examples.opentpu.shape_workload import resolve_static
        from examples.opentpu.compact import CompactOpenTPU
        from examples.opentpu.glm47_target import component_timing
        cfg,programs,images,_=collective_case(2,stagger=False)
        programs[1][1]=I.gather(512,8193,3,129,137,277,129)
        with self.assertRaisesRegex(ValueError,'collective streams disagree'):
            CompactOpenTPU(resolve_static(cfg,programs),component_timing())

    def test_multiple_slices_without_collectives_remain_independent(self):
        from opentpu import isa as I
        from examples.opentpu.shape_workload import resolve_static
        from examples.opentpu.compact import CompactOpenTPU
        from examples.opentpu.glm47_target import component_timing
        cfg,_,_,_=collective_case(2)
        model=CompactOpenTPU(resolve_static(cfg,[[I.halt()],[I.halt()]]),component_timing())
        sim=model.net.python_simulator();sim.run(max_time=10000)
        self.assertEqual(sim.place_dicts('program_completed'),[{}])
        self.assertEqual(model.collective.endpoints,[])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--tool',type=Path)
    a=p.parse_args();use_checkout(a.opentpu_root);CollectiveTest.tool=a.tool
    unittest.main(argv=[__file__],verbosity=2)
