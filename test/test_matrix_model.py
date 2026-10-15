import json
from pathlib import Path
import unittest

from examples.coralnpu.matrix import GemmModel, MatrixArray, MatrixCommand, MatrixSpec
from examples.coralnpu.model import CoralNPU, Instruction, MemorySpec
from examples.coralnpu.trace import load_dispatch_trace
from examples.coralnpu.validate_matrix import cases, compare_array, load_events


class MatrixModelTest(unittest.TestCase):
  def test_staggered_blocks_and_tail_masks(self):
    spec = MatrixSpec()
    self.assertEqual(spec.latency(MatrixCommand()), 12)
    self.assertEqual(spec.latency(MatrixCommand(m=3, n=5, k=3)), 8)
    self.assertEqual(spec.latency(MatrixCommand(m=9, n=11, k=2)), 11)

  def test_accumulator_pipeline_reuse(self):
    commands = [MatrixCommand() for _ in range(4)]
    self.assertEqual(MatrixArray(commands).simulate()["cycles"], 24)
    tail = [MatrixCommand(m=3, n=5, k=3) for _ in range(4)]
    self.assertEqual(MatrixArray(tail).simulate()["cycles"], 20)

  def test_core_includes_matrix_dependencies(self):
    commands = [MatrixCommand() for _ in range(4)]
    instructions = [Instruction("matrix", ("mt0", "v8", "v16"), ("mt0",), matrix=c)
                    for c in commands]
    self.assertEqual(CoralNPU(instructions).simulate()["cycles"], 27)

  def test_invalid_matrix_shapes_fail(self):
    for options in ({"k": 5}, {"m": 17}, {"tile": 2}, {"dtype": "fp32", "k": 2}):
      with self.assertRaises(ValueError):
        MatrixCommand(**options)

  def test_operand_buffering_allows_overlap(self):
    one = GemmModel(16, 16, 8, buffers=1).simulate()
    two = GemmModel(16, 16, 8, buffers=2).simulate()
    self.assertLess(two["cycles"], one["cycles"])
    self.assertEqual(two["macs"], 2048)

  def test_memory_bandwidth_affects_gemm(self):
    fast = GemmModel(16, 16, 4, memory=MemorySpec(3, 16)).simulate()
    slow = GemmModel(16, 16, 4, memory=MemorySpec(3, 4)).simulate()
    self.assertGreater(slow["cycles"], fast["cycles"])

  def test_recorded_full_core_matrix_rtl(self):
    root = Path(__file__).resolve().parents[1] / "examples/coralnpu/validation/matrix"
    report = json.loads((root / "comparison.json").read_text())
    for record in report["kernels"]:
      name = record["kernel"]
      with self.subTest(kernel=name):
        comparison = compare_array(cases()[name], load_events(root / f"{name}.matrix.csv"))
        self.assertEqual(comparison["array_lpn_cycles"], record["array_rtl_cycles"])
        instructions, _ = load_dispatch_trace(root / f"{name}.dispatch.csv")
        self.assertEqual([instruction.matrix for instruction in instructions if instruction.matrix], cases()[name])


if __name__ == "__main__":
  unittest.main()
