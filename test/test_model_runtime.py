"""Regression checks for the modeling path and its Python execution."""

import unittest

from examples.network_net import build_network_example, create_network_net
from examples.protoacc_components import DEFAULT_WORKLOAD_PATH
from examples.protoacc_helpers import load_workload
from lpnlang_mlir import NetBuilder, PythonRuntimeAPI, TransitionBuilder
from helper_component import connect_helper


class ModelRuntimeTest(unittest.TestCase):
  def test_component_can_call_helper_from_another_module(self):
    import helper_component as component
    net = NetBuilder("cross_module_helper")
    source = net.place("source",initial_tokens=1)
    destination = net.place("destination")

    @net.func(args=("i64",),results=("i64",))
    def next_value(value):
      func_return(addi(value,1))

    connect_helper(net,source,destination,next_value)
    sim = net.python_simulator()
    sim.run()
    self.assertEqual(sim.place_dicts("destination"),[{"value":1}])
    self.assertNotIn("take",component.__dict__)
    self.assertNotIn("func_return",globals())

  def test_all_to_all_delivers_every_flow(self):
    net = create_network_net(3, 2)
    sim = net.python_simulator()
    sim.run(max_time=1_000_000)
    for dst in range(3):
      tokens = sim.place_dicts(f"d{dst}_in")
      self.assertEqual(len(tokens), 4)
      self.assertEqual(
          {(t["src"], t["dst"], t["id"], t["hops"]) for t in tokens},
          {(src, dst, dst * 2 + seq, 1)
           for src in range(3) if src != dst for seq in range(2)})

  def test_all_to_all_uses_one_injector_per_source(self):
    ir = build_network_example(3, 2)
    self.assertEqual(ir.count("lpn.transition @d0_inject_all"), 1)
    self.assertEqual(ir.count("lpn.transition @d1_inject_all"), 1)
    self.assertEqual(ir.count("lpn.transition @d2_inject_all"), 1)
    self.assertIn("scf.for", ir)

  def test_signed_division_and_remainder_match_mlir(self):
    api = PythonRuntimeAPI(None)
    self.assertEqual(api.divi(-7, 3), -2)
    self.assertEqual(api.remi(-7, 3), -1)
    self.assertEqual(api.divi(7, -3), -2)
    self.assertEqual(api.remi(7, -3), 1)

  def test_small_f64_delay_is_preserved(self):
    builder = TransitionBuilder("small_delay")
    builder.const_f64(1e-9)
    self.assertIn("1e-09", "\n".join(builder.render()))

  def test_invalid_place_and_network_parameters(self):
    net = NetBuilder("validation")
    net.place("buffer")
    with self.assertRaises(ValueError):
      net.place("buffer")
    with self.assertRaises(ValueError):
      net.place("negative", initial_tokens=-1)
    with self.assertRaises(ValueError):
      create_network_net(1)
    with self.assertRaises(ValueError):
      create_network_net(2, 0)

  def test_bundled_protoacc_workload_is_parseable(self):
    workload = load_workload(DEFAULT_WORKLOAD_PATH)
    self.assertTrue(workload.control_tokens)
    self.assertTrue(workload.field_meta_tokens)
    self.assertTrue(workload.unit_tokens)
    self.assertTrue(workload.writes)


if __name__ == "__main__":
  unittest.main()
