"""Guard synthesis must reject token flows its metadata cannot represent."""

import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path

from examples.network_net import create_network_net
from lpnlang_mlir import NetBuilder
from test_native_compiler import (build_conditional_take_net, build_priority_net,
                                 build_blocked_guard_net)
from test_native_compiler import build_read_only_helper_guard_net
from test_native_compiler import build_bitmap_net


class GuardRejectionTest(unittest.TestCase):
  tool = ""

  def test_read_only_helper_predicate_is_preserved(self):
    result = self.synthesize(build_read_only_helper_guard_net(1))
    self.assertEqual(result.returncode,0,result.stderr)
    self.assertIn('lpn.guard_mode = "manual"',result.stdout)

  def test_cpp_interpreter_bitmap_high_bit(self):
    result = self.synthesize(build_bitmap_net(),("--lpn-simulate",))
    self.assertEqual(result.returncode,0,result.stderr)
    self.assertIn("place sink: 1 token(s)",result.stdout)

  def test_effectful_helper_before_manual_guard_is_rejected(self):
    net = NetBuilder("effectful_predicate")
    source = net.place("source",initial_tokens=1)

    @net.func(args=("Place",),results=("i64",))
    def consume(place):
      func_return(take(place).get("ready"))

    @net.transition
    def invalid():
      ready = builder.consume(source)
      guard(ready>0)

    result = self.synthesize(net)
    self.assertNotEqual(result.returncode,0)
    self.assertIn("precede",result.stderr)

  def synthesize(self, net, extra_args=()):
    with tempfile.TemporaryDirectory(prefix="lpn-guards-test-") as directory:
      source = Path(directory) / "model.mlir"
      source.write_text(net.build(), encoding="utf-8")
      return subprocess.run([self.tool, "--lpn-synthesize-guards", *extra_args, str(source)],
                            capture_output=True, text=True)

  def test_repeated_takes_and_emits_are_rejected(self):
    result = self.synthesize(create_network_net(3, 2))
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("repeating or dynamic scf.for loop", result.stderr)

  def test_token_flow_in_helper_is_rejected(self):
    net = NetBuilder("helper_guard")
    source = net.place("source", initial_tokens=1)
    sink = net.place("sink")

    @net.func(args=("Place",), results=())
    def consume_token(place):
      take(place)
      func_return()

    @net.transition
    def consume():
      builder.consume_token(source)
      emit(sink, create())

    result = self.synthesize(net)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("token flow inside a helper function", result.stderr)

  def test_data_dependent_nested_take_is_rejected(self):
    result = self.synthesize(build_conditional_take_net(1, 1))
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("data-dependent scf.if", result.stderr)

  def test_explicit_priority_predicate_is_preserved(self):
    result = self.synthesize(build_priority_net())
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn('lpn.guard_mode = "manual"', result.stdout)
    self.assertIn("lpn.guard", result.stdout)
    self.assertIn("lpn.peek", result.stdout)

  def test_cpp_interpreter_executes_manual_priority(self):
    result = self.synthesize(build_priority_net(), ("--lpn-simulate",))
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("place low: 1 token(s)", result.stdout)
    self.assertIn("place high: 0 token(s)", result.stdout)
    self.assertIn("place out: 1 token(s)", result.stdout)

  def test_manual_guards_cannot_be_silently_reduced(self):
    result = self.synthesize(build_priority_net(), ("--lpn-retain-hypergraph",))
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("does not support manual guard predicates", result.stderr)

  def test_cpp_interpreter_false_manual_guard(self):
    result = self.synthesize(build_blocked_guard_net(), ("--lpn-simulate",))
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("place source: 1 token(s)", result.stdout)
    self.assertIn("place sink: 0 token(s)", result.stdout)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--tool", required=True)
  args = parser.parse_args()
  GuardRejectionTest.tool = args.tool
  unittest.main(argv=[__file__], verbosity=2)
