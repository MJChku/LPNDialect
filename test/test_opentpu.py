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

  def test_shape_trace_matches_numerical_control_and_footprints(self):
    from examples.opentpu.shape_workload import resolve_static
    from opentpu import Config,isa as I
    import numpy as np
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as directory:
      cfg=Config(D=128,MCOLS=4,DRAM_BYTES=1024)
      program=[I.li(1,3),I.loop(3,0,rcount=1),I.addi(2,2,4),
               I.ld(0,0,1,ra=2),I.nop(),I.halt()]
      write_inputs(directory,cfg,[program],[np.zeros(1024,np.uint8)])
      numerical=load(directory)
      with patch('opentpu.isasim.Machine',side_effect=AssertionError('numerical execution forbidden')):
        static=resolve_static(cfg,[program])
      self.assertIsNone(static.machine)
      self.assertEqual(static.images,[])
      self.assertEqual([(c.pc,c.instruction.encode(),c.footprint) for c in static.commands[0]],
                       [(c.pc,c.instruction.encode(),c.footprint) for c in numerical.commands[0]])

  def test_shape_trace_rejects_data_dependent_control(self):
    from examples.opentpu.shape_workload import resolve_static
    from opentpu import Config,isa as I
    with self.assertRaisesRegex(ValueError,'RLD'):
      resolve_static(Config(),[[I.Instr(I.RLD),I.halt()]])
    with self.assertRaisesRegex(ValueError,'limit'):
      resolve_static(Config(),[[I.nop(),I.halt()]],max_steps=1)

  def test_fpga_output_bank_service_counts_actual_collisions(self):
    from examples.opentpu.model import bank_write_cycles
    self.assertEqual(bank_write_cycles(3,8,2,8),4)
    self.assertEqual(bank_write_cycles(8,4,64,8),4)
    self.assertEqual(bank_write_cycles(8,4,64,1),32)

  def test_shared_dram_load_can_finish_while_quantizer_computes(self):
    from examples.opentpu.shape_workload import resolve_static
    from opentpu import Config,isa as I
    cfg=Config(D=64,MCOLS=1,LANES=64,DRAM_BYTES=1<<20)
    program=[I.qst(0,65536,131072,8,1,64,64,1),I.ld(0,1024,16),I.halt()]
    workload=resolve_static(cfg,[program])
    timing=Timing(unified_dram=True,scalar_store_bytes=256,
        dram_bytes_per_cycle=256,dram_port_bytes=256)
    ends=[]
    for cls in (OpenTPU,CompactOpenTPU):
      sim=cls(workload,timing).net.python_simulator()
      sim.run(max_time=100)
      if cls is OpenTPU:
        self.assertEqual(len(sim.place_dicts('s0_done_1')),1)
        self.assertEqual(len(sim.place_dicts('s0_done_0')),0)
      else:
        bits=sim.place_dicts('s0_flags_0')[0]['bits']
        self.assertTrue(bits&2)
        self.assertFalse(bits&1)
      ends.append(sim.run(max_time=10000,reset=False))
    self.assertEqual(*ends)

  def test_small_matrix_prefetch_uses_actual_fifo_capacity(self):
    import re
    from examples.opentpu.shape_workload import resolve_static
    from opentpu import Config,isa as I
    cfg=Config(D=128,MCOLS=4,ACT_BLOCKS=64)
    # The first matrix fits all 64 one-chunk rows in the 128-chunk FIFO.
    # A later, barrier-separated 64-chunk command must not shrink that FIFO
    # into two slots for the first command and insert repeated DRAM delays.
    program=[I.mm(0,0,0,64,1,128,64,1,0,4,unit=True),I.bar(),
             I.mm(65536,0,4096,8,64,8192,8,4,0,4,unit=True),I.halt()]
    workload=resolve_static(cfg,[program])
    timing=Timing(batch_rows=8,dram_latency=64)
    with tempfile.TemporaryDirectory() as directory:
      for cls in (OpenTPU,CompactOpenTPU):
        with self.subTest(model=cls.__name__):
          model=cls(workload,timing)
          sim=model.net.python_simulator()
          sim.run(max_time=250)
          if cls is CompactOpenTPU:
            self.assertTrue(sim.place_dicts('s0_flags_0')[0]['bits']&1)
          else:
            self.assertEqual(sim.place_dicts('s0_done_0'),[{}])
          if self.tool:
            out=Path(directory)
            ir,binary=out/'fifo.mlir',out/'fifo.native'
            ir.write_text(model.build())
            subprocess.run([self.tool,str(ir),'-o',str(binary)],check=True,capture_output=True,text=True)
            native=subprocess.run([str(binary),'--max-time=250','--tokens','--stats'],
                check=True,capture_output=True,text=True).stdout
            self.assertIn('rollbacks: 0',native)
            if cls is CompactOpenTPU:
              flags=re.search(r"place s0_flags_0: 1 token\(s\)\s+token \{[^}]*'bits': (\d+)",native)
              self.assertIsNotNone(flags)
              self.assertTrue(int(flags[1])&1)
            else:
              self.assertIn('place s0_done_0: 1 token(s)',native)

  def test_parallel_rows_do_not_depend_on_aggregation_hint(self):
    from dataclasses import replace
    from examples.opentpu.shape_workload import resolve_static
    from opentpu import Config,isa as I
    cfg=Config(D=128,MCOLS=4,ACT_ROWS=8,ACT_BLOCKS=64)
    program=[I.mm(0,0,0,64,16,2048,64,8,0,4,unit=True),I.halt()]
    baseline=resolve_static(cfg,[program])
    serial=CompactOpenTPU(baseline,Timing()).net.python_simulator().run(max_time=100000)
    parallel=replace(baseline,config=replace(cfg,MCOLS=1,LANES=64))
    results=[]
    with tempfile.TemporaryDirectory() as directory:
      for batch in (1,32,64):
        timing=Timing(batch_rows=batch,matrix_row_lanes=32,fifo_chunks=4096,
            dram_port_bytes=2048,dram_bytes_per_cycle=2048)
        for cls in (OpenTPU,CompactOpenTPU):
          model=cls(parallel,timing)
          sim=model.net.python_simulator()
          cycles=sim.run(max_time=100000)
          self.assertEqual(sim.place_dicts('program_completed'),[{}])
          results.append(cycles)
        if self.tool:
          from examples.opentpu.native_timing import ReusableTiming
          out=Path(directory)
          prepared=ReusableTiming(self.tool,out/'cache',capacity=128).prepare(
              parallel,timing,out/'timing.tables')
          self.assertEqual(prepared.run()[0],cycles)
    self.assertEqual(len(set(results)),1)
    self.assertLess(results[0],serial/4)

  def test_wide_vector_timing_matches_rtl_latency_fixture(self):
    from examples.opentpu.shape_workload import resolve_static
    from opentpu import Config,isa as I
    fixture=Path(__file__).resolve().parents[1]/'examples/opentpu/rtl_evolution/validation/vector64-latency.json'
    report=json.loads(fixture.read_text())
    functions={name:code for code,name in I.VFUNCS.items()}
    program=[I.vop(functions[name],4096+256*i,1024 if name=='rsqrt' else 0,
                2048,1,193,256,256,256) for i,name in enumerate(report['functions'])]
    cfg=Config(D=64,LANES=64,TMEM_WORDS=8192,DRAM_BYTES=65536)
    workload=resolve_static(cfg,[program+[I.halt()]])
    model=CompactOpenTPU(workload,Timing(dram_bytes_per_cycle=64))
    for ins,actual in zip(program,report['command_cycles']):
      with self.subTest(function=I.VFUNCS[ins.w[5]>>16&255]):
        prediction,_,_=model._simple_timing(ins,model.components[0]['memory'],I)
        self.assertLessEqual(abs(prediction-actual),1)

  def test_fpga_unscaled_prefetch_does_not_overtake_scale_fetch(self):
    if not self.tool:
      self.skipTest('native compiler not supplied')
    from dataclasses import replace
    from examples.opentpu.shape_workload import compile_shape,qwen17_spec
    from examples.opentpu.native_timing import ReusableTiming
    workload,_=compile_shape(qwen17_spec(),1023,1024)
    workload=replace(workload,config=replace(workload.config,MCOLS=1,LANES=64))
    timing=Timing(batch_rows=8,matrix_engines=4,matrix_row_lanes=4,
        dram_port_bytes=2048,dram_bytes_per_cycle=2048,dram_latency=64,
        scale_lanes=4,fifo_chunks=512,unified_dram=True,scalar_store_bytes=256)
    with tempfile.TemporaryDirectory() as directory:
      out=Path(directory)
      prepared=ReusableTiming(self.tool,out/'cache',capacity=32768).prepare(workload,timing,out/'timing.tables')
      cycles,_,_,log=prepared.run()
      self.assertGreater(cycles,1_000_000)
      self.assertIn('rollbacks: 0',log)

  def test_fpga_datapath_matches_expanded_and_native_models(self):
    from examples.opentpu.shape_workload import resolve_static
    from dataclasses import replace
    with tempfile.TemporaryDirectory() as directory:
      out=Path(directory)
      gemm(out/'input',n=32,kb=2,jobs=2,m=4)
      numerical=load(out/'input')
      workload=resolve_static(numerical.config,numerical.programs)
      # Fast weight bus but slow scale decoder; readiness must wait for both.
      timing=Timing(batch_rows=8,matrix_row_lanes=4,dram_port_bytes=1024,
          dram_bytes_per_cycle=1024,scale_lanes=1,unified_dram=True,
          scalar_store_bytes=32)
      predictions=[]
      for cls in (OpenTPU,CompactOpenTPU):
        model=cls(workload,timing)
        sim=model.net.python_simulator()
        predictions.append(sim.run(max_time=1_000_000))
        self.assertEqual(len(sim.place_dicts('program_completed')),1)
      self.assertEqual(*predictions)
      if self.tool:
        from examples.opentpu.native_timing import ReusableTiming
        prepared=ReusableTiming(self.tool,out/'cache',capacity=128).prepare(workload,timing,out/'timing.tables')
        cycles,_,_,log=prepared.run()
        self.assertEqual(cycles,predictions[0])
        self.assertIn('rollbacks: 0',log)

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
          self.assertEqual(model.statistics()["transitions"],16+4*(engines-1))
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

  def test_ready_scalar_units_keep_hazards_without_false_fifo_edges(self):
    from examples.opentpu.workload import Command
    from opentpu import isa as I,waits as W
    region=((W.SP_TMEM,0,8),)
    for ins in (I.ld(0,0,8),I.qact(0,1,0,1,128)):
      for hazard in ('independent','RAW','WAR','WAW'):
        first=W.Fp(rd=region) if hazard=='WAR' else W.Fp(wr=region)
        second=(W.Fp() if hazard=='independent' else
                W.Fp(rd=region) if hazard=='RAW' else W.Fp(wr=region))
        compute,_=dependencies([Command(0,ins,first),Command(1,ins,second)],I,W)
        self.assertEqual(compute[1],[] if hazard=='independent' else [0],hazard)

  def test_vector_overlap_drains_for_reductions_and_shorter_paths(self):
    from examples.opentpu.workload import Workload,Command
    from opentpu import Config,isa as I,waits as W
    cfg=Config(D=128,MCOLS=4,LANES=8)
    def cycles(second,dependent=False):
      first=I.vop(I.V_EXP2,0,128,0,1,8,8,8,0)
      tail=I.vop(second,8,0 if dependent else 256,0,1,8,8,8,0)
      program=[first,tail,I.halt()]
      commands=[Command(i,ins,W.footprint(ins,[(0,0)]*16,cfg.D,cfg.S))
                for i,ins in enumerate(program)]
      workload=Workload(cfg,[program],[],[commands],None,{},[])
      predictions=[]
      for cls in (OpenTPU,CompactOpenTPU):
        sim=cls(workload).net.python_simulator()
        predictions.append(sim.run(max_time=10000))
        self.assertEqual(sim.place_dicts('program_completed'),[{}])
      self.assertEqual(*predictions)
      return predictions[0]
    parallel=cycles(I.V_EXP2)
    self.assertLess(parallel,cycles(I.V_EXP2,dependent=True))
    self.assertLess(parallel,cycles(I.V_RSUM))
    self.assertGreater(cycles(I.V_COPY),parallel)

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
