"""Storage-version dependency screening; no finite FPGA implementation claimed."""
import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from examples.opentpu.act_versions import renamed_commands
from examples.opentpu.compact import CompactOpenTPU, dependencies
from examples.opentpu.model import OpenTPU, Timing
from examples.opentpu.shape_workload import resolve_static
from examples.opentpu.workload import use_checkout


class ActVersionsTest(unittest.TestCase):
    tool = None

    def test_partial_writes_keep_each_preceding_page_version(self):
        from opentpu import Config, isa as I, waits as W
        cfg = Config(D=64, MCOLS=8, ACT_ROWS=8)
        work = resolve_static(cfg, [[I.qact(0, 2, 0, 2, 128),
            I.qact(1024, 1, 1, 1, 64),
            I.mm(65536, 0, 2048, 4, 2, 128, 4, 2, 0, 4, unit=True), I.halt()]])
        old = work.commands[0]
        before = [c.footprint for c in old]
        new = renamed_commands(old, cfg, I, W)
        pages = cfg.act_rows * cfg.ACT_BLOCKS
        actual = {a for sp, lo, hi in new[2].footprint.rd if sp == W.SP_ACT for a in range(lo, hi)}
        self.assertEqual(actual, {pages, 2*pages+1, pages+cfg.ACT_BLOCKS, pages+cfg.ACT_BLOCKS+1})
        deps, _ = dependencies(new, I, W)
        self.assertEqual(deps[2], [0, 1])
        self.assertIs(new[2].instruction, old[2].instruction)
        self.assertEqual([c.footprint for c in old], before)

    def test_barrier_and_unknown_accessor(self):
        from opentpu import Config, isa as I, waits as W
        cfg = Config(D=64, MCOLS=8, ACT_ROWS=8)
        work = resolve_static(cfg, [[I.qact(0, 1, 0, 1, 64), I.bar(),
            I.qact(1024, 1, 0, 1, 64), I.halt()]])
        new = renamed_commands(work.commands[0], cfg, I, W)
        deps, _ = dependencies(new, I, W)
        self.assertIn(0, deps[1])
        self.assertIn(1, deps[2])
        command = work.commands[0][0]
        unknown = replace(command, instruction=I.Instr(I.NOP))
        with self.assertRaises(ValueError): renamed_commands([unknown], cfg, I, W)

    def test_dup_pair_and_initial_pages(self):
        from opentpu import Config, isa as I, waits as W
        cfg = Config(D=64, MCOLS=8, ACT_ROWS=8)
        work = resolve_static(cfg, [[I.qact(0, 2, 1, 1, 64, dup=True),
            I.mm(65536, 0, 2048, 4, 2, 128, 4, 2, 0, 4, unit=True, pair=True, wf=I.W4I), I.halt()]])
        new = renamed_commands(work.commands[0], cfg, I, W)
        pages = cfg.act_rows * cfg.ACT_BLOCKS
        actual = {a for sp, lo, hi in new[1].footprint.rd if sp == W.SP_ACT for a in range(lo, hi)}
        self.assertEqual(actual, {r*cfg.ACT_BLOCKS for r in range(4)} |
            {pages+r*cfg.ACT_BLOCKS+1 for r in range(4)})

    def workload(self, alias=False):
        from opentpu import Config, isa as I
        cfg = Config(D=64, MCOLS=1, LANES=64, ACT_ROWS=8, DRAM_BYTES=1<<20)
        return resolve_static(cfg, [[I.qact(0, 4, 0, 8, 512),
            I.mm(65536, 0, 20000, 128, 8, 512, 128, 4, 0, 4, unit=True),
            I.qact(20000 if alias else 4096, 4, 0, 8, 512),
            I.mm(131072, 0, 21000, 128, 8, 512, 128, 4, 0, 4, unit=True), I.halt()]])

    def test_only_act_overwrite_hazards_are_removed(self):
        from opentpu import isa as I, waits as W
        for alias in (False, True):
            work = self.workload(alias)
            old = work.commands[0]
            new = renamed_commands(old, work.config, I, W)
            deps, _ = dependencies(new, I, W)
            self.assertIn(0, deps[1]) # old activation producer still required
            self.assertEqual(1 in deps[2], alias) # TMEM RAW is never removed
            self.assertIn(2, deps[3]) # new activation producer still required
            for a, b in zip(old, new):
                for field in ('rd', 'wr'):
                    self.assertEqual(tuple(r for r in getattr(a.footprint, field) if r[0] != W.SP_ACT),
                        tuple(r for r in getattr(b.footprint, field) if r[0] != W.SP_ACT))

    def test_screen_cannot_claim_rtl_or_fpga_capacity(self):
        from examples.opentpu.rtl_evolution.parallel_inference import validate_variant
        from examples.opentpu.tpu_dse import Architecture, resources
        work = self.workload()
        with self.assertRaises(ValueError):
            OpenTPU(work, Timing(act_versions_unbounded=True))
        with self.assertRaises(ValueError):
            validate_variant({'act_versions_unbounded': True}, {})
        point = Architecture('screen', 4, 1, 64, ceiling_only=True, act_versions_unbounded=True)
        report = resources(point, vars(work.config), 200)
        self.assertFalse(report['act_version_storage_bounded'])
        for name, value in report.items():
            if name.startswith('passes_'): self.assertFalse(value, name)
        with self.assertRaises(ValueError):
            resources(replace(point, ceiling_only=False), vars(work.config), 200)

    def test_overlap_python_native_and_default(self):
        from examples.opentpu.native_timing import ReusableTiming
        timing = Timing(quant_row_banks=8, activation_streaming=True,
            matrix_row_lanes=4, batch_rows=4, fifo_chunks=256, dram_bytes_per_cycle=64)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = ReusableTiming(self.tool, root/'cache', capacity=128) if self.tool else None
            for alias in (False, True):
                work = self.workload(alias)
                ticks = []
                for enabled in (False, True):
                    t = replace(timing, act_versions_unbounded=enabled)
                    sim = CompactOpenTPU(work, t).net.python_simulator()
                    cycles = sim.run(max_time=100000)
                    self.assertEqual(sim.place_dicts('program_completed'), [{}])
                    ticks.append(cycles)
                    if runner:
                        prepared = runner.prepare(work, t, root/f'{alias}-{enabled}.tables')
                        self.assertEqual(prepared.run()[0], cycles)
                if alias: self.assertEqual(*ticks)
                else: self.assertLess(ticks[1], ticks[0])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--opentpu-root', type=Path, required=True)
    p.add_argument('--tool', type=Path)
    args = p.parse_args()
    use_checkout(args.opentpu_root)
    ActVersionsTest.tool = args.tool
    unittest.main(argv=[__file__], verbosity=2)
