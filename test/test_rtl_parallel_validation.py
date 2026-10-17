import unittest
from types import SimpleNamespace

from examples.opentpu.rtl_evolution.parallel_inference import configure_rtlsim


class WorkerConfigurationTest(unittest.TestCase):
    def test_reused_worker_keeps_the_same_simulator_cache_key(self):
        simulation = SimpleNamespace(VFLAGS=['--binary', '-j', '4', '-O3'])
        configure_rtlsim(simulation)
        first = repr(simulation.VFLAGS)
        for _ in range(4):
            configure_rtlsim(simulation)
            self.assertEqual(repr(simulation.VFLAGS), first)
        self.assertEqual(simulation.VFLAGS[simulation.VFLAGS.index('-j') + 1], '12')


if __name__ == '__main__':
    unittest.main()
