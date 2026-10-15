import unittest
from pathlib import Path
import json

from examples.coralnpu.model import CoralNPU, CoreConfig, Instruction, MemorySpec
from examples.coralnpu.trace import load_dispatch_trace


class CoralNPUTest(unittest.TestCase):
  def test_dependency_chain(self):
    model = CoralNPU([Instruction("alu", ("x1",), ("x1",)) for _ in range(8)])
    self.assertEqual(model.simulate()["cycles"], 8)

  def test_four_way_independent_dispatch(self):
    model = CoralNPU([Instruction("alu") for _ in range(8)])
    self.assertEqual(model.simulate()["cycles"], 2)

  def test_multiplier_is_pipelined(self):
    model = CoralNPU([Instruction("mul") for _ in range(4)])
    self.assertEqual(model.simulate()["cycles"], 5)

  def test_external_memory_latency_and_bandwidth(self):
    cfg = CoreConfig(external=MemorySpec(latency=10, bytes_per_cycle=4))
    model = CoralNPU([Instruction("lsu", bytes=16, external=True)], cfg)
    self.assertEqual(model.simulate()["cycles"], 14)

  def test_decoupled_vector_dependency_chain(self):
    model = CoralNPU([Instruction("valu", ("v1", "v2"), ("v1",)) for _ in range(4)])
    self.assertEqual(model.simulate()["cycles"], 19)

  def test_unknown_unit_fails(self):
    with self.assertRaisesRegex(ValueError, "unsupported CoralNPU unit"):
      CoralNPU([Instruction("unmodeled")])

  def test_recorded_verilator_microkernels(self):
    root = Path(__file__).resolve().parents[1] / "examples/coralnpu/validation"
    for repetitions in (8, 16, 32):
      directory = root / f"rep{repetitions}"
      report = json.loads((directory / "comparison.json").read_text())
      for kernel in report["kernels"]:
        with self.subTest(repetitions=repetitions, kernel=kernel["kernel"]):
          instructions, baseline = load_dispatch_trace(
              directory / f"{kernel['kernel']}.csv")
          self.assertEqual(CoralNPU(instructions).simulate()["cycles"],
                           baseline["dispatch_span"])


if __name__ == "__main__":
  unittest.main()
