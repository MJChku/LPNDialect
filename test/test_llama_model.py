"""Check Llama's dataflow graph and native/reference execution on a small shape."""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path
import unittest

from examples.coralnpu.llama import LlamaConfig, LlamaWorkload
from examples.coralnpu.model import MemorySpec


class LlamaModelTest(unittest.TestCase):
  tool = None

  def small(self, **kwargs):
    return LlamaWorkload(4, 2, config=LlamaConfig(hidden=32,intermediate=64,
        layers=1,heads=4,kv_heads=2,vocabulary=64), **kwargs)

  def test_architecture_and_tied_parameter_count(self):
    cfg = LlamaConfig()
    self.assertEqual((cfg.layers,cfg.head_dim,cfg.kv_dim), (28,128,1024))
    self.assertEqual(cfg.parameters,3212749824)

  def test_grouped_queries_and_separate_key_value_dependencies(self):
    model = self.small()
    ops = {op.name: (idx,op) for idx,op in enumerate(model.operators)}
    self.assertEqual(ops["p0.l0.q"][1].macs,2*ops["p0.l0.k"][1].macs)
    qk = ops["p0.l0.qk"][1]
    self.assertEqual(qk.dependencies,(ops["p0.l0.q_rope"][0],ops["p0.l0.key_append"][0]))
    self.assertIn(ops["p0.l0.value_append"][0],ops["p0.l0.attention_value"][1].dependencies)

  def test_prefill_produces_first_token_and_decode_extends_cache(self):
    report = self.small().simulate()
    self.assertEqual(len(report["decode_cycles"]),1)
    self.assertEqual(report["kv_cache_bytes"],320)
    self.assertGreater(report["total_cycles"],report["time_to_first_token_cycles"])
    self.assertFalse(report["accuracy"]["passed"])
    self.assertFalse(report["numerical_inference"])

  def test_memory_bandwidth_changes_completion(self):
    fast = self.small(memory=MemorySpec(64,16)).simulate()
    slow = self.small(memory=MemorySpec(64,4)).simulate()
    self.assertGreater(slow["total_cycles"],fast["total_cycles"])

  def test_feedback_layers_match_expanded_dataflow(self):
    cfg = LlamaConfig(hidden=32,intermediate=64,layers=3,heads=4,kv_heads=2,vocabulary=64)
    compact = LlamaWorkload(4,2,config=cfg)
    expanded = LlamaWorkload(4,2,config=cfg,fold_layers=False)
    a,b = compact.simulate(),expanded.simulate()
    self.assertEqual(a["phase_completion_cycles"],b["phase_completion_cycles"])
    self.assertEqual((a["macs"],a["dram_bytes"],a["operators"]),
                     (b["macs"],b["dram_bytes"],b["operators"]))
    self.assertLess(a["operator_templates"],b["operator_templates"])

  def test_invalid_shapes_fail(self):
    with self.assertRaises(ValueError):
      LlamaConfig(hidden=31)
    with self.assertRaises(ValueError):
      LlamaWorkload(131072,2)

  def test_native_dataflow_matches_reference_without_rollback(self):
    if not self.tool:
      self.skipTest("native compiler not provided")
    model = self.small()
    reference = model.simulate()
    with tempfile.TemporaryDirectory() as directory:
      source,binary = Path(directory)/"llama.mlir",Path(directory)/"llama"
      source.write_text(model.build())
      subprocess.run([self.tool,str(source),"-o",str(binary)],
                     capture_output=True,text=True,check=True)
      result = subprocess.run([str(binary),"--stats","--trace"],
                              capture_output=True,text=True,check=True)
    self.assertEqual(model.phase_times(result.stdout),reference["phase_completion_cycles"])
    match = re.search(r"body executions: (\d+); rollbacks: (\d+)",result.stdout)
    self.assertEqual(int(match.group(2)),0)
    self.assertEqual(int(match.group(1)),3*len(model.operators)+model.generate_tokens)
    self.assertIn("place llama_completed: 1 token(s)",result.stdout)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--tool")
  args = parser.parse_args()
  LlamaModelTest.tool = args.tool
  unittest.main(argv=[__file__],verbosity=2)
