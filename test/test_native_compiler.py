"""Compare native executables with the Python simulator on representative nets."""

import argparse
import ast
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from examples.network_net import create_network_net
from examples.simple_array_net import build_demo_net
from lpnlang_mlir import NetBuilder


def build_conditional_take_net(route_value: int, branch_tokens: int):
  net = NetBuilder("conditional_take")
  source = net.place("source", initial_tokens=1)
  branch = net.place("branch", initial_tokens=branch_tokens)
  sink = net.place("sink", observable=True)

  @net.transition
  def route():
    token = take(source).set("route", route_value)
    if token.get("route") == 1:
      take(branch)
    emit(sink, token)

  return net


def build_priority_net(high_priority=10, low_priority=5):
  net = NetBuilder("priority")
  seed = net.place("seed", initial_tokens=1)
  latch = net.place("latch", initial_tokens=1)
  high = net.place("high", observable=True)
  low = net.place("low", observable=True)
  out = net.place("out", observable=True)

  @net.transition
  def bootstrap():
    take(seed)
    emit(high, create().set("priority", high_priority))
    emit(low, create().set("priority", low_priority))

  @net.transition
  def arbitrate():
    guard(count(latch) > 0)
    guard(count(high) > 0)
    guard(count(low) > 0)
    first = peek(high)
    second = peek(low)
    take(latch)
    if first.get("priority") >= second.get("priority"):
      emit(out, take(high))
    else:
      emit(out, take(low))

  return net


def build_blocked_guard_net():
  net = NetBuilder("guard_blocked")
  source = net.place("source", initial_tokens=1)
  sink = net.place("sink", observable=True)

  @net.transition
  def blocked():
    guard(False)
    emit(sink, take(source))

  return net


def native_result(net, tool: str, max_time: float):
  with tempfile.TemporaryDirectory(prefix="lpn-native-test-") as directory:
    source = Path(directory) / "model.mlir"
    binary = Path(directory) / "model"
    source.write_text(net.build(), encoding="utf-8")
    subprocess.run([tool, str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    output = subprocess.run(
        [str(binary), f"--max-time={max_time}", "--tokens"],
        check=True, capture_output=True, text=True, timeout=30).stdout

  match = re.search(r"Simulation finished at t=([0-9.]+)", output)
  if not match:
    raise AssertionError(f"native executable did not report final time:\n{output}")
  places = {}
  current_place = None
  for line in output.splitlines():
    place = re.match(r"  place (.+): ([0-9]+) token\(s\)", line)
    if place:
      current_place = place.group(1)
      places[current_place] = []
      continue
    if line.startswith("    token "):
      places[current_place].append(ast.literal_eval(line[len("    token "):]))
  return float(match.group(1)), places


class NativeCompilerTest(unittest.TestCase):
  tool = ""

  def check_net(self, net, max_time):
    native_time, native_places = native_result(net, self.tool, max_time)
    simulator = net.python_simulator()
    python_time = simulator.run(max_time=max_time)
    self.assertAlmostEqual(native_time, python_time, places=3)
    for handle, *_ in net._places:
      self.assertEqual(native_places[handle.name],
                       simulator.place_dicts(handle), handle.name)
    return native_places

  def test_all_to_all_network(self):
    self.check_net(create_network_net(3, 2), 1_000_000)

  def test_array_helper_and_loop_values(self):
    self.check_net(build_demo_net(3), 1_000)

  def test_take_inside_token_dependent_if(self):
    for route_value, branch_tokens in ((0, 0), (1, 0), (1, 1)):
      with self.subTest(route_value=route_value, branch_tokens=branch_tokens):
        places = self.check_net(
            build_conditional_take_net(route_value, branch_tokens), 1_000)
        should_fire = route_value == 0 or branch_tokens > 0
        self.assertEqual(places["sink"],
                         [{"route": route_value}] if should_fire else [])
        self.assertEqual(len(places["source"]), 0 if should_fire else 1)

  def test_manual_priority_guard_and_nonconsuming_peek(self):
    for high, low in ((10, 5), (3, 9)):
      with self.subTest(high=high, low=low):
        places = self.check_net(build_priority_net(high, low), 1_000)
        self.assertEqual(places["out"], [{"priority": max(high, low)}])
        loser = "low" if high >= low else "high"
        self.assertEqual(places[loser], [{"priority": min(high, low)}])

  def test_false_manual_guard_leaves_marking_unchanged(self):
    places = self.check_net(build_blocked_guard_net(), 1_000)
    self.assertEqual(places["source"], [{}])
    self.assertEqual(places["sink"], [])

  def test_coralnpu_resource_model(self):
    from examples.coralnpu.model import CoralNPU, Instruction
    model = CoralNPU([Instruction("mul", ("x1",), ("x1",)) for _ in range(8)])
    self.check_net(model.net, 1_000)
    vector_model = CoralNPU([
        Instruction("valu", ("v1", "v2"), ("v1",)) for _ in range(4)])
    self.check_net(vector_model.net, 1_000)

  def test_matrix_pipeline_and_gemm_memory(self):
    from examples.coralnpu.matrix import MatrixArray, MatrixCommand, GemmModel
    model = MatrixArray([MatrixCommand() for _ in range(4)])
    places = self.check_net(model.net, 1_000)
    self.assertEqual(len(places["matrix_completed"]), 4)
    gemm = GemmModel(16, 16, 8)
    places = self.check_net(gemm.net, 1_000)
    self.assertEqual(len(places["gemm_completed"]), 1)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--tool", required=True)
  args = parser.parse_args()
  NativeCompilerTest.tool = args.tool
  unittest.main(argv=[__file__], verbosity=2)
