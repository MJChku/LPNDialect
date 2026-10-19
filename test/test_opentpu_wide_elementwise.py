"""Wide elementwise timing proposal preserves ordered reductions and hazards."""
import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from examples.opentpu.compact import CompactOpenTPU
from examples.opentpu.model import Timing
from examples.opentpu.shape_workload import resolve_static
from examples.opentpu.workload import use_checkout


class WideElementwiseTest(unittest.TestCase):
    tool = None

    def timing(self, groups):
        return Timing(vector_issue_groups=groups, vector_command_queue=2,
            vector_row_contexts=4, fp_add_latency=2, dram_bytes_per_cycle=64)

    def test_only_elementwise_issue_is_widened(self):
        from opentpu import Config, isa as I
        cfg=Config(D=64,LANES=64)
        work=resolve_static(cfg,[[I.halt()]])
        memory=SimpleNamespace()
        for function in [I.V_COPY,I.V_MUL,I.V_EXP2,I.V_OUTER,*I.REDUCE]:
            ins=I.vop(function,8192,0,4096,3,1024,2048,2048,2048,I.B_FULL)
            if function==I.V_OUTER:ins.w[6]=6000
            latencies=[];intervals=[]
            for groups in (1,2,4,8):
                model=CompactOpenTPU(work,self.timing(groups))
                latencies.append(model._simple_timing(ins,memory,I)[0])
                intervals.append(model._vector_interval(ins,I))
            if function in I.REDUCE:
                self.assertEqual(len(set(latencies)),1)
                self.assertEqual(len(set(intervals)),1)
            else:
                self.assertTrue(all(b<a for a,b in zip(latencies,latencies[1:])))
                self.assertTrue(all(b<a for a,b in zip(intervals,intervals[1:])))
        # Sub-group tails remain whole issue clocks, never fractional cycles.
        for count in (1,63,64,65,255,256,257):
            ins=I.vop(I.V_COPY,8192,0,0,1,count,count,count,0,I.B_SCALAR)
            model=CompactOpenTPU(work,self.timing(4))
            self.assertEqual(model._vector_interval(ins,I),(count+255)//256+2)

    def test_native_chain_keeps_dependencies_and_reductions(self):
        from opentpu import Config,isa as I
        from examples.opentpu.native_timing import ReusableTiming
        cfg=Config(D=64,LANES=64)
        program=[I.vop(I.V_COPY,8192,0,0,3,2048,2048,2048,0,I.B_SCALAR),
            I.vop(I.V_MUL,16384,8192,0,3,2048,2048,2048,0,I.B_SCALAR,imm=2),
            I.vop(I.V_RSUM,24576,16384,0,3,2048,1,2048,0,I.B_SCALAR),I.halt()]
        work=resolve_static(cfg,[program]);ticks=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            runner=ReusableTiming(self.tool,root/'cache',capacity=128) if self.tool else None
            for groups in (1,2,4):
                timing=self.timing(groups)
                sim=CompactOpenTPU(work,timing).net.python_simulator()
                cycles=sim.run(max_time=100000)
                self.assertEqual(sim.place_dicts('program_completed'),[{}])
                if runner:
                    prepared=runner.prepare(work,timing,root/f'{groups}.tables')
                    self.assertEqual(prepared.run()[0],cycles)
                ticks.append(cycles)
        self.assertTrue(ticks[2]<ticks[1]<ticks[0])

    def test_rtl_rejected_and_port_costs_explicit(self):
        from opentpu import Config
        from examples.opentpu.tpu_dse import Architecture,resources
        from examples.opentpu.rtl_evolution.parallel_inference import validate_variant
        with self.assertRaises(ValueError):validate_variant({'vector_issue_groups':4},{})
        with self.assertRaises(ValueError):Timing(vector_issue_groups=3)
        with self.assertRaises(ValueError):Timing(vector_issue_groups=4,vector_engines=2,vector_result_chunks=1024)
        report=resources(Architecture('wide',96,2,2816,vector_issue_groups=4),vars(Config()),200)
        self.assertEqual(report['vector_elementwise_issue_lanes'],256)
        self.assertEqual(report['vector_ordered_reduction_lanes'],64)
        self.assertEqual(report['vector_tmem_read_words_per_cycle'],512)
        self.assertEqual(report['vector_tmem_write_words_per_cycle'],256)
        self.assertEqual(report['vector_fp32_multiply_sites'],1088)
        self.assertFalse(report['wide_elementwise_rtl_implemented'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True)
    p.add_argument('--tool',type=Path);a=p.parse_args();use_checkout(a.opentpu_root)
    WideElementwiseTest.tool=a.tool;unittest.main(argv=[__file__],verbosity=2)
