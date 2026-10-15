"""Optional FeSens integration tests; requires its pinned checkout and numpy."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from examples.opentpu.benchmark import gemm, mlp, native_observation, write_inputs
from examples.opentpu.model import OpenTPU, Timing
from examples.opentpu.compact import CompactOpenTPU,dependencies
from examples.opentpu.workload import load, use_checkout


class OpenTPUTest(unittest.TestCase):
  tool = None

  def test_reusable_timing_accepts_changed_command_count_without_recompilation(self):
    if not self.tool:
      self.skipTest("native compiler not supplied")
    from examples.opentpu.native_timing import ReusableTiming
    with tempfile.TemporaryDirectory() as directory:
      out = Path(directory)
      runner = ReusableTiming(self.tool,out/"cache",capacity=128)
      binary = None
      for jobs in (2,70):
        gemm(out/"input",n=8,kb=1,jobs=jobs)
        workload = load(out/"input")
        expected = CompactOpenTPU(workload).net.python_simulator().run(max_time=100000)
        prepared = runner.prepare(workload,Timing(),out/"input.tables")
        self.assertEqual(prepared.run()[0],expected)
        self.assertEqual(prepared.cache_hit,binary is not None)
        if binary is not None:
          self.assertEqual(prepared.binary,binary)
        binary = prepared.binary
      with self.assertRaisesRegex(ValueError,"input_capacity"):
        CompactOpenTPU(workload,input_capacity=4)

  def test_multiple_engines_overlap_independent_commands_and_keep_write_hazards(self):
    import copy
    from examples.opentpu.workload import execute
    from opentpu import isa as I
    with tempfile.TemporaryDirectory() as directory:
      out = Path(directory)
      gemm(out,n=128,kb=8,jobs=1,m=16)
      baseline = load(out)
      first = baseline.programs[0][3]
      independent_times = []
      for conflicting in (False,True):
        second = copy.deepcopy(first)
        if not conflicting:
          second.w[2] += 16*128
        program = baseline.programs[0][:2]+[first,second,I.halt()]
        workload = execute(baseline.config,[program],baseline.images)
        predictions = []
        for engines in (1,2):
          model = CompactOpenTPU(workload,Timing(matrix_engines=engines,batch_rows=8))
          sim = model.net.python_simulator()
          cycles = sim.run(max_time=100000)
          self.assertEqual(sim.place_dicts("program_completed"),[{}])
          self.assertEqual(model.statistics()["transitions"],14+3*(engines-1))
          self.assertEqual(len(model.components[0]["matrix_engines"]),engines)
          if self.tool:
            ir,binary = out/"multi.mlir",out/"multi.native"
            ir.write_text(model.build())
            subprocess.run([self.tool,str(ir),"-o",str(binary)],check=True,capture_output=True,text=True)
            self.assertEqual(native_observation(binary)[0],cycles)
          predictions.append(cycles)
        if conflicting:
          self.assertGreaterEqual(predictions[1],predictions[0])
          self.assertGreater(predictions[1],independent_times[1])
        else:
          self.assertLess(predictions[1],0.75*predictions[0])
          independent_times = predictions

  def test_parallel_matrix_dependencies_preserve_raw_war_waw_and_barriers(self):
    from examples.opentpu.workload import Command
    from opentpu import isa as I,waits as W
    matrix = I.mm(0,0,0,1,1,128,1,1,0,4,unit=True)
    for space in (W.SP_DRAM,W.SP_TMEM,W.SP_ACT):
      for hazard in ("RAW","WAR","WAW"):
        rd = ((space,0,8),)
        a = W.Fp(rd=rd) if hazard=="WAR" else W.Fp(wr=rd)
        b = W.Fp(rd=rd) if hazard=="RAW" else W.Fp(wr=rd)
        commands = [Command(0,matrix,a),Command(1,matrix,b),Command(2,I.halt(),W.Fp())]
        compute,dram = dependencies(commands,I,W,matrix_engines=2)
        self.assertEqual(compute[1],[0],(space,hazard))
        self.assertEqual(dram[1],[0] if space==W.SP_DRAM else [])
        self.assertEqual(compute[2],[1])
    commands = [Command(0,matrix,W.Fp()),Command(1,I.bar(),W.Fp(all=True)),
                Command(2,matrix,W.Fp()),Command(3,I.halt(),W.Fp())]
    compute,dram = dependencies(commands,I,W,matrix_engines=2)
    self.assertEqual(compute[2],[1])
    self.assertEqual(dram[2],[1])

  def test_matrix_admission_cannot_overtake_an_older_blocked_command(self):
    from examples.opentpu.workload import Command,Workload
    from opentpu import Config,isa as I,waits as W
    cfg = Config(D=128,MCOLS=4)
    program = [I.st(0,0,16),I.mm(0,0,128,8,1,128,8,1,0,4,unit=True),
        I.mm(1<<20,0,128,8,1,128,8,1,0,4,unit=True),I.halt()]
    commands = [Command(i,ins,W.footprint(ins,[(0,0)]*16,cfg.D,cfg.S))
                for i,ins in enumerate(program)]
    workload = Workload(cfg,[program],[],[commands],None,{},[])
    for engines in (1,2):
      model = CompactOpenTPU(workload,Timing(matrix_engines=engines))
      sim = model.net.python_simulator()
      original = sim._invoke
      admitted = []
      def observe(definition,*args,**kwargs):
        fired = original(definition,*args,**kwargs)
        if fired and definition.name.endswith("_start"):
          prefix = definition.name[:-len("_start")]
          admitted.append(sim.place_dicts(prefix+"_request")[-1]["id"])
        return fired
      sim._invoke = observe
      sim.run(max_time=10000)
      self.assertEqual(admitted,[1,2])
      self.assertEqual(sim.place_dicts("program_completed"),[{}])

  def test_multi_engine_configuration_requires_compact_model(self):
    with self.assertRaises(ValueError):
      Timing(matrix_engines=0)
    with self.assertRaises(ValueError):
      Timing(matrix_engines=1.5)
    with tempfile.TemporaryDirectory() as directory:
      gemm(directory,n=8,kb=1,jobs=1)
      with self.assertRaisesRegex(ValueError,"CompactOpenTPU"):
        OpenTPU(load(directory),Timing(matrix_engines=2))

  def test_compact_completion_banks_match_expanded_dataflow(self):
    for jobs in (2,70):
      with self.subTest(jobs=jobs),tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        gemm(out,n=8,kb=1,jobs=jobs)
        workload = load(out)
        expanded = OpenTPU(workload).net.python_simulator()
        expected = expanded.run(max_time=100000)
        model = CompactOpenTPU(workload)
        sim = model.net.python_simulator()
        self.assertEqual(sim.run(max_time=100000),expected)
        self.assertEqual(sim.place_dicts("program_completed"),[{}])
        self.assertLess(len(model.net._transitions),20)
        if self.tool:
          ir,binary = out/"compact.mlir",out/"compact.native"
          ir.write_text(model.build())
          subprocess.run([self.tool,str(ir),"-o",str(binary)],check=True,capture_output=True,text=True)
          self.assertEqual(native_observation(binary)[0],expected)

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
        predictions = []
        for cls in (OpenTPU,CompactOpenTPU):
          model = cls(workload,Timing(**report["timing"]))
          ir,binary = out/"model.mlir",out/"model.native"
          ir.write_text(model.build())
          subprocess.run([self.tool,str(ir),"-o",str(binary)],check=True,capture_output=True,text=True)
          cycles,_,_,_ = native_observation(binary)
          predictions.append(cycles)
          self.assertLess(abs(cycles/report["rtl_cycles"]-1),0.05)
        self.assertEqual(predictions[0],predictions[1])


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--opentpu-root",type=Path,required=True)
  parser.add_argument("--tool")
  args = parser.parse_args()
  use_checkout(args.opentpu_root)
  OpenTPUTest.tool = args.tool
  unittest.main(argv=[__file__],verbosity=2)
