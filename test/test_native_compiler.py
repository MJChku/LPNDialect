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


def build_conditional_take_net(route_value: int, branch_tokens: int, manual_guard=False):
  net = NetBuilder("conditional_take")
  source = net.place("source", initial_tokens=1)
  branch = net.place("branch", initial_tokens=branch_tokens)
  sink = net.place("sink", observable=True)

  if manual_guard:
    required = int(route_value == 1)

    @net.transition
    def route():
      guard(count(source) > 0)
      guard(count(branch) >= required)
      token = take(source).set("route", route_value)
      if token.get("route") == 1:
        take(branch)
      emit(sink, token)
  else:
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


def build_blocked_inferred_net():
  net = NetBuilder("inferred_blocked")
  source = net.place("source", initial_tokens=1)
  sink = net.place("sink")

  @net.transition
  def needs_two():
    for _ in range(2):
      emit(sink, take(source))

  return net


def build_peek_after_take_net(tokens):
  net = NetBuilder("peek_after_take")
  source = net.place("source", initial_tokens=tokens)
  sink = net.place("sink")

  @net.transition
  def read_next():
    first = take(source)
    second = peek(source)
    emit(sink, first.set("next", second.get("value")))

  return net


def build_read_only_helper_guard_net(tokens,use_peek=False):
  net = NetBuilder("helper_predicate")
  source = net.place("source",initial_tokens=tokens)
  sink = net.place("sink")

  @net.func(args=("Place",),results=("i1",))
  def available(place):
    if use_peek:
      func_return(peek(place).get("value")==0)
    else:
      func_return(count(place)>0)

  @net.transition
  def consume():
    ready = available(array(source)[0])
    guard(ready)
    emit(sink,take(source))
  return net


def build_peek_snapshot_net():
  net = NetBuilder("peek_snapshot")
  seed = net.place("seed",initial_tokens=1)
  source = net.place("source")
  sink = net.place("sink")

  @net.transition
  def initialize():
    take(seed)
    emit(source,create().set("value",73))

  @net.transition
  def consume():
    guard(count(source)>0)
    saved = peek(source)
    take(source)
    emit(sink,saved)
  return net


def build_empty_predicate_peek_net():
  net = NetBuilder("empty_predicate_peek")
  source = net.place("source")
  sink = net.place("sink")

  @net.transition
  def consume():
    guard(peek(source).get("value")>0)
    emit(sink,take(source))
  return net


def build_literal_place_cache_net():
  net = NetBuilder("literal_place_cache")
  source = net.place("source",initial_tokens=2)
  sink = net.place("sink")

  @net.transition
  def transfer():
    places = array(source,sink)
    indices = array(0,1)
    guard(count(places[indices[0]])>0)
    emit(places[indices[1]],take(places[indices[0]]))
  return net


def build_mixed_token_net(rollback):
  net = NetBuilder("mixed_tokens")
  seed = net.place("seed",initial_tokens=1)
  source = net.place("source")
  sink = net.place("sink")
  required = 6 if rollback else 5

  @net.transition
  def initialize():
    take(seed)
    emit(source,create())
    emit(source,create().set("value",7),delay=1.0)
    emit(source,create(),delay=1.0)
    emit(source,create(),delay=2.0)
    emit(source,create().set("value",9),delay=3.0)

  @net.transition
  def consume():
    guard(count(source)>=5)
    for _ in range(required):
      emit(sink,take(source))
  return net


def build_token_handle_net():
  net = NetBuilder("token_handles")
  seed = net.place("seed",initial_tokens=1)
  hops = [net.place(f"hop_{i}") for i in range(17)]
  left,right = net.place("left"),net.place("right")

  @net.transition
  def initialize():
    take(seed)
    emit(hops[0],create().set("value",99))

  for previous,following in zip(hops,hops[1:]):
    @net.transition(previous.name+"_move")
    def move(previous=previous,following=following):
      emit(following,take(previous).clone())

  @net.transition
  def fanout():
    token = take(hops[-1])
    emit(left,token)
    emit(right,token.set("value",100))
  return net


def build_billion_credit_net():
  net = NetBuilder("billion_credits")
  seed = net.place("seed",initial_tokens=1)
  source = net.place("source",initial_tokens=1_000_000_000)
  sink = net.place("sink")

  @net.transition
  def move_credits():
    take(seed)
    for _ in range(32):
      emit(sink,take(source))
  return net


def build_bitmap_net():
  net = NetBuilder("bitmap")
  seed = net.place("seed",initial_tokens=1)
  source,sink = net.place("source"),net.place("sink")

  @net.transition
  def initialize():
    take(seed)
    bits = const_i64(0)
    for bit in range(64):
      bits = ori(bits,shli(1,bit))
    emit(source,create().set("mask",bits))

  @net.transition
  def consume():
    guard(count(source)>0)
    observed = peek(source)
    guard(andi(observed.get("mask"),shli(1,63))!=0)
    emit(sink,take(source).set("low",andi(observed.get("mask"),63)))
  return net


def build_guard_only_helper_net():
  net = NetBuilder("guard_only_helper")
  source = net.place("source",initial_tokens=1)
  sink = net.place("sink")

  @net.func(args=("Place",),results=("i1",))
  def enabled(place):
    temporary = peek(place).set("probe",1)
    func_return(temporary.get("probe")==1)

  @net.transition
  def consume():
    guard(enabled(array(source)[0]))
    emit(sink,take(source))
  return net


def native_output(net, tool: str, max_time: float,print_tokens=True):
  with tempfile.TemporaryDirectory(prefix="lpn-native-test-") as directory:
    source = Path(directory) / "model.mlir"
    binary = Path(directory) / "model"
    source.write_text(net.build(), encoding="utf-8")
    subprocess.run([tool, str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return subprocess.run(
        [str(binary), f"--max-time={max_time}", "--stats"]+(["--tokens"] if print_tokens else []),
        check=True, capture_output=True, text=True, timeout=30).stdout


def native_result(net, tool: str, max_time: float):
  output = native_output(net,tool,max_time)
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

  def test_read_only_boolean_helper_in_guard(self):
    for tokens in (0,2):
      for use_peek in (False,True):
        self.check_net(build_read_only_helper_guard_net(tokens,use_peek),1000)

  def test_body_peek_remains_valid_after_take(self):
    result = self.check_net(build_peek_snapshot_net(),1000)
    self.assertEqual(result["sink"],[{"value":73}])

  def test_empty_peek_disables_predicate_safely(self):
    self.check_net(build_empty_predicate_peek_net(),1000)

  def test_literal_and_place_array_caches_are_distinct(self):
    result = self.check_net(build_literal_place_cache_net(),1000)
    self.assertEqual(result["sink"],[{},{}])

  def test_mixed_empty_and_valued_fifo_and_rollback(self):
    for rollback in (False,True):
      with self.subTest(rollback=rollback):
        result = self.check_net(build_mixed_token_net(rollback),1000)
        expected = [{},{"value":7},{},{},{"value":9}]
        self.assertEqual(result["source" if rollback else "sink"],expected)
        self.assertEqual(result["sink" if rollback else "source"],[])

  def test_moving_cloning_and_fanout_share_immutable_payloads(self):
    net = build_token_handle_net()
    result = self.check_net(net,1000)
    self.assertEqual(result["left"],[{"value":99}])
    self.assertEqual(result["right"],[{"value":100}])
    output = native_output(net,self.tool,1000)
    self.assertIn("Token payload allocations: 2; field copies on set: 1",output)

  def test_empty_tokens_use_counters_for_billion_credits(self):
    output = native_output(build_billion_credit_net(),self.tool,1000,print_tokens=False)
    self.assertIn("place source: 999999968 token(s)",output)
    self.assertIn("place sink: 32 token(s)",output)
    self.assertIn("Token payload allocations: 0; field copies on set: 0",output)

  def test_bitmap_includes_signed_high_bit(self):
    result = self.check_net(build_bitmap_net(),1000)
    self.assertEqual(result["sink"],[{"mask":-1,"low":63}])

  def test_guard_only_helper_is_not_reexecuted_in_body(self):
    net = build_guard_only_helper_net()
    self.check_net(net,1000)
    output = native_output(net,self.tool,1000)
    self.assertIn("Token payload allocations: 1;",output)

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
            build_conditional_take_net(route_value, branch_tokens, manual_guard=True), 1_000)
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

  def test_disabled_guards_skip_transition_bodies(self):
    for net in (build_blocked_guard_net(), build_blocked_inferred_net(),
                build_peek_after_take_net(1),
                build_conditional_take_net(1, 0, manual_guard=True)):
      with self.subTest(net=net.name), tempfile.TemporaryDirectory() as directory:
        source, binary = Path(directory) / "model.mlir", Path(directory) / "model"
        source.write_text(net.build())
        subprocess.run([self.tool, str(source), "-o", str(binary)],
                       check=True, capture_output=True, text=True)
        result = subprocess.run([str(binary), "--stats"], check=True,
                                capture_output=True, text=True)
        self.assertIn("body executions: 0; rollbacks: 0", result.stdout)

  def test_inferred_peek_requires_remaining_fifo_token(self):
    places = self.check_net(build_peek_after_take_net(2), 1_000)
    self.assertEqual(len(places["source"]), 1)
    self.assertEqual(places["sink"], [{"next": 0}])

  def test_uninferable_guard_requires_manual_annotation(self):
    with tempfile.TemporaryDirectory() as directory:
      source, binary = Path(directory) / "model.mlir", Path(directory) / "model"
      source.write_text(build_conditional_take_net(1, 1).build())
      result = subprocess.run([self.tool, str(source), "-o", str(binary)],
                              capture_output=True, text=True)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("add an explicit guard(...)", result.stderr)

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
