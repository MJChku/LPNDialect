"""Optional FeSens integration tests; requires its pinned checkout and numpy."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from examples.opentpu.benchmark import gemm, mlp, native_observation, write_inputs
from examples.opentpu.model import OpenTPU, Timing
from examples.opentpu.workload import load, use_checkout


class OpenTPUTest(unittest.TestCase):
  tool = None

  def test_real_program_native_matches_python_dataflow(self):
    with tempfile.TemporaryDirectory() as directory:
      out = Path(directory)
      gemm(out,n=32,kb=8,jobs=2)
      w = load(out)
      self.assertEqual(len(w.commands[0]),7)
      self.assertEqual([c.pc for c in w.commands[0]],[0,1,2,3,3,4,5])
      model = OpenTPU(w)
      sim = model.net.python_simulator()
      python_cycles = sim.run(max_time=100000)
      self.assertEqual(len(sim.place_dicts("program_completed")),1)
      if self.tool:
        ir,binary = out/"model.mlir",out/"model.native"
        ir.write_text(model.build())
        subprocess.run([self.tool,str(ir),"-o",str(binary)],check=True,capture_output=True,text=True)
        cycles,_,_,_ = native_observation(binary)
        self.assertEqual(cycles,python_cycles)

  def test_batched_rows_preserve_long_workload_timing(self):
    with tempfile.TemporaryDirectory() as directory:
      out = Path(directory)
      gemm(out,n=128,kb=8,jobs=2)
      w = load(out)
      fine,batched = OpenTPU(w),OpenTPU(w,Timing(batch_rows=8))
      cycles = []
      for model in (fine,batched):
        sim = model.net.python_simulator()
        cycles.append(sim.run(max_time=100000))
        self.assertEqual(len(sim.place_dicts("program_completed")),1)
      self.assertLess(abs(cycles[1]/cycles[0]-1),0.02)
      self.assertEqual(batched.matrix_batch_sizes,[2,2])

  def test_register_relative_loop_uses_actual_inputs(self):
    from opentpu import Config,isa as I
    import numpy as np
    with tempfile.TemporaryDirectory() as directory:
      out = Path(directory)
      cfg = Config(D=128,MCOLS=4,DRAM_BYTES=1024)
      program = [I.li(1,2),I.loop(2,0,rcount=1),I.addi(2,2,4),I.nop(),I.halt()]
      write_inputs(out,cfg,[program],[np.zeros(1024,np.uint8)])
      w = load(out)
      self.assertEqual([c.pc for c in w.commands[0]],[0,1,2,3,2,3,4])
      self.assertEqual(w.machine.slices[0].R[2],8)

  def test_accumulator_pipeline_does_not_serialize_rows(self):
    from opentpu import isa as I
    with tempfile.TemporaryDirectory() as directory:
      out = Path(directory)
      gemm(out,n=128,kb=2,jobs=3)
      baseline = load(out)
      cycles = []
      for accumulate in (False,True):
        program = baseline.programs[0]
        program[3].flags = I.F_UNIT|(I.F_ACC if accumulate else 0)
        write_inputs(out,baseline.config,[program],baseline.images)
        workload = load(out)
        model = OpenTPU(workload)
        sim = model.net.python_simulator()
        cycles.append(sim.run(max_time=100000))
        self.assertEqual(len(sim.place_dicts("program_completed")),1)
      self.assertLess(abs(cycles[1]-cycles[0]),30)

  def test_unsupported_host_synchronization_is_rejected(self):
    from opentpu import Config,isa as I
    import numpy as np
    with tempfile.TemporaryDirectory() as directory:
      cfg = Config(D=128,MCOLS=4,DRAM_BYTES=1024)
      write_inputs(directory,cfg,[[I.waitw(0,0,1,I.C_EQ),I.halt()]],
                   [np.zeros(1024,np.uint8)])
      with self.assertRaisesRegex(ValueError,"WAITW"):
        load(directory)

  def test_truncated_input_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      out = Path(directory)
      gemm(out,n=16,kb=2,jobs=1)
      (out/"prog_0.hex").write_text("00000000\n")
      with self.assertRaisesRegex(ValueError,"length"):
        load(out)

  def test_recorded_rtl_accuracy(self):
    if not self.tool:
      self.skipTest("native compiler not supplied")
    fixtures = Path(__file__).resolve().parents[1]/"examples/opentpu/validation"
    for case in ("gemm_100k","mlp","replay_m8"):
      with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        report = json.loads((fixtures/case/"comparison.json").read_text())
        if case=="mlp":
          mlp(out)
        else:
          gemm(out,n=report["n"],kb=report["k"]//128,jobs=report["jobs"],m=report["m"])
        workload = load(out)
        self.assertEqual(workload.hashes,report["input_sha256"])
        model = OpenTPU(workload,Timing(**report["timing"]))
        ir,binary = out/"model.mlir",out/"model.native"
        ir.write_text(model.build())
        subprocess.run([self.tool,str(ir),"-o",str(binary)],check=True,capture_output=True,text=True)
        cycles,_,_,_ = native_observation(binary)
        self.assertLess(abs(cycles/report["rtl_cycles"]-1),0.05)


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--opentpu-root",type=Path,required=True)
  parser.add_argument("--tool")
  args = parser.parse_args()
  use_checkout(args.opentpu_root)
  OpenTPUTest.tool = args.tool
  unittest.main(argv=[__file__],verbosity=2)
