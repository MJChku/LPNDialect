"""Guard-cache behavior across dynamic dependencies and delayed markings."""

import argparse
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from lpnlang_mlir import NetBuilder
from test_native_compiler import native_result


def dynamic_route_net():
  net = NetBuilder("dynamic_guard_dependencies")
  route = net.place("route", initial_tokens=1)
  a, b = net.place("a"), net.place("b", initial_tokens=1)
  seed, change = net.place("seed", initial_tokens=1), net.place("change")
  out = net.place("out")

  @net.func(args=("Place",), results=("i1",))
  def available(place):
    func_return(peek(place).get("value") == 0)

  @net.transition
  def consume():
    guard(count(route) > 0)
    chosen = array(a, b)[peek(route).get("destination")]
    guard(available(chosen))
    take(route)
    emit(out, take(chosen))

  @net.transition
  def schedule_change():
    emit(change, take(seed), delay=7.0)

  @net.transition
  def reroute():
    take(change)
    token = take(route)
    emit(route, token.set("destination", 1))
  return net


def expanding_dependencies_net():
  net = NetBuilder("expanding_guard_dependencies")
  seed = net.place("seed", initial_tokens=1)
  source, branch, out = net.place("source"), net.place("branch"), net.place("out")

  @net.transition
  def consume():
    guard(count(source) > 0)
    guard(count(branch) > 0)
    take(source)
    emit(out, take(branch))

  @net.transition
  def schedule():
    take(seed)
    emit(source, create(), delay=3.0)
    emit(branch, create().set("value", 99), delay=8.0)
  return net


def helper_peek_net():
  net = NetBuilder("empty_helper_peek")
  seed = net.place("seed", initial_tokens=1)
  source, out = net.place("source"), net.place("out")

  @net.func(args=("Place",), results=("i1",))
  def available(place):
    func_return(peek(place).get("value") == 9)

  @net.transition
  def consume():
    guard(available(array(source)[0]))
    emit(out, take(source))

  @net.transition
  def schedule():
    emit(source, take(seed).set("value", 9), delay=9.0)
  return net


def unchanged_guard_net():
  net = NetBuilder("unchanged_guard")
  cursor = net.place("cursor", initial_tokens=1)
  latch = net.place("latch", initial_tokens=1)
  out = net.place("out")

  @net.func(args=("Place",), results=("i1",))
  def available(place):
    token = peek(place).set("probe", 1)
    func_return(token.get("enabled") == 1)

  @net.transition
  def blocked():
    guard(available(array(latch)[0]))
    emit(out, create())

  @net.transition
  def step():
    guard(count(cursor) > 0)
    guard(peek(cursor).get("id") < 20)
    token = take(cursor)
    emit(cursor, token.set("id", token.get("id") + 1), delay=1.0)
  return net


def incomplete_annotation_net():
  net = NetBuilder("cached_true_atomic_fallback")
  latch, seed = net.place("latch", initial_tokens=1), net.place("seed", initial_tokens=1)
  source, out = net.place("source"), net.place("out")

  @net.transition
  def consume():
    # Intentionally incomplete: the atomic body must still handle an empty
    # source when the independently cached predicate remains true.
    guard(count(latch) > 0)
    emit(out, take(source))

  @net.transition
  def schedule():
    emit(source, take(seed).set("value", 13), delay=5.0)
  return net


class GuardHeartbeatTest(unittest.TestCase):
  tool = ""

  def compare(self, net, expected_time, expected_out):
    with tempfile.TemporaryDirectory(prefix="lpn-heartbeat-") as d:
      ir, binary = Path(d)/"model.mlir", Path(d)/"model"
      ir.write_text(net.build())
      subprocess.run([self.tool, str(ir), "-o", str(binary)], check=True, capture_output=True, text=True)
      results = []
      for options in ([], ["--no-guard-cache"]):
        run = subprocess.run([str(binary), "--stats", "--tokens", "--trace"] + options,
                             check=True, capture_output=True, text=True, timeout=30)
        results.append(run.stdout)
      # Preserve the complete firing order, timestamps, FIFO contents and token
      # fields, including fallback rollbacks. Only performance counters differ.
      def semantic(output):
        return [s for s in output.splitlines() if not s.startswith(("Guard", "Token payload"))]
      self.assertEqual(semantic(results[0]), semantic(results[1]))
      cached = int(re.search(r"Guard checks: (\d+)", results[0])[1])
      uncached = int(re.search(r"Guard checks: (\d+)", results[1])[1])
      self.assertLess(cached, uncached)
      self.assertGreater(int(re.search(r"Guard cache hits: (\d+)", results[0])[1]), 0)
      self.assertIn("Guard cache hits: 0", results[1])
    time, places = native_result(net, self.tool, 1000)
    self.assertEqual(time, expected_time)
    self.assertEqual(places["out"], expected_out)
    reference = net.python_simulator()
    self.assertEqual(reference.run(), expected_time)
    for handle, *_ in net._places:
      self.assertEqual(places[handle.name], reference.place_dicts(handle))
    return results

  def test_dynamic_selector_and_same_count_token_replacement(self):
    self.compare(dynamic_route_net(), 7, [{}])

  def test_short_circuit_watch_set_expands_after_first_input_arrives(self):
    self.compare(expanding_dependencies_net(), 8, [{"value": 99}])

  def test_empty_peek_in_helper_wakes_on_delayed_arrival(self):
    self.compare(helper_peek_net(), 9, [{"value": 9}])

  def test_unrelated_events_do_not_recheck_a_false_guard(self):
    self.compare(unchanged_guard_net(), 20, [])

  def test_cached_true_keeps_atomic_fallback_for_incomplete_annotations(self):
    results = self.compare(incomplete_annotation_net(), 5, [{"value": 13}])
    for output in results:
      self.assertGreater(int(re.search(r"rollbacks: (\d+)", output)[1]), 0)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--tool", required=True)
  args = parser.parse_args()
  GuardHeartbeatTest.tool = args.tool
  unittest.main(argv=[__file__], verbosity=2)
