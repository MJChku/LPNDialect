"""Verify actual ELF inputs, compact scoreboards, and immutable literal arrays."""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from examples.coralnpu.benchmark_workload import build_elf, gemm_source
from examples.coralnpu.elf import execute_elf
from examples.coralnpu.stream import InstructionStream
from lpnlang_mlir import NetBuilder
from test_native_compiler import native_result


class ELFWorkloadTest(unittest.TestCase):
  tool = None
  clang = "clang-20"

  def test_same_elf_frontend_executes_real_operands_and_loop_counts(self):
    with tempfile.TemporaryDirectory() as directory:
      source,elf = Path(directory)/"gemm.S",Path(directory)/"gemm.elf"
      asm,expected = gemm_source(k=8,jobs=2)
      source.write_text(asm)
      build_elf(source,elf,self.clang)
      workload = execute_elf(elf)
      self.assertEqual(workload.scalar_outputs[-256:],expected)
      self.assertEqual(workload.matrices,8)
      self.assertEqual(workload.macs,2*16*16*8)
      self.assertEqual(workload.instructions[workload.roi[0]].pc,
                       workload.image.symbols["roi_start"])
      # The end of the executable includes a real scalar readback loop.
      self.assertGreater(len(workload.instructions),workload.roi[1]+1000)
      self.assertEqual(len(workload.instructions),len(workload.addresses))
      if not self.tool:
        return
      # Check a complete GEMM execution region against the Python LPN, while
      # avoiding the slow reference's replay of the unrelated output oracle.
      stop = workload.roi[1]
      workload = replace(workload,instructions=workload.instructions[:stop],
                         addresses=workload.addresses[:stop])
      model = InstructionStream(workload)
      sim = model.net.python_simulator()
      log = io.StringIO()
      with redirect_stdout(log):
        sim.run(max_time=100000,debug=True)
      self.assertEqual(len(sim.place_dicts(model.finished)),1)
      ir,binary = Path(directory)/"model.mlir",Path(directory)/"model"
      ir.write_text(model.build())
      subprocess.run([self.tool,str(ir),"-o",str(binary)],check=True,capture_output=True,text=True)
      result = subprocess.run([str(binary),"--trace","--stats",
          "--wall-start=observe_roi_start","--wall-end=observe_roi_end"],
          check=True,capture_output=True,text=True)
      pattern = r"\[t=([0-9.]+)\] fired observe_roi_(?:start|end)"
      self.assertEqual(re.findall(pattern,result.stdout),re.findall(pattern,log.getvalue()))
      self.assertIn("rollbacks: 0",result.stdout)
      self.assertIn("Wall window elapsed:",result.stdout)
      self.assertIn("place program_completed: 1 token(s)",result.stdout)

  def test_literal_array_updates_do_not_change_cached_input(self):
    if not self.tool:
      self.skipTest("native compiler not provided")
    net = NetBuilder("immutable_literal")
    counter = net.place("counter",initial_tokens=1)
    out = net.place("out")

    @net.transition
    def step():
      token = take(counter)
      idx = token.get("idx")
      values = array([10,20])
      changed = values.set(0,idx)
      emit(out,create().set("original",values[0]).set("changed",changed[0]))
      if idx<2:
        emit(counter,token.set("idx",idx+1),delay=1.0)

    _,places = native_result(net,self.tool,100)
    self.assertEqual(places["out"],[{"original":10,"changed":i} for i in range(3)])

  def test_unsupported_elf_instruction_fails(self):
    with tempfile.TemporaryDirectory() as directory:
      source,elf = Path(directory)/"bad.S",Path(directory)/"bad.elf"
      asm,_ = gemm_source(k=2,jobs=1)
      for replacement in (".word 0x00000000", "vle16.v v8, (x11), v0.t",
                          "csrw vstart, x0", ".word 0x04001013"):
        with self.subTest(instruction=replacement):
          source.write_text(asm.replace("vle16.v v8, (x11)",replacement))
          build_elf(source,elf,self.clang)
          with self.assertRaises(ValueError):
            execute_elf(elf)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--tool")
  parser.add_argument("--clang",default="clang-20")
  args = parser.parse_args()
  ELFWorkloadTest.tool,ELFWorkloadTest.clang = args.tool,args.clang
  unittest.main(argv=[__file__],verbosity=2)
