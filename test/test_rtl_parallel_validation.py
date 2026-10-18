import unittest
from types import SimpleNamespace

from examples.opentpu.rtl_evolution.parallel_inference import configure_rtlsim


class WorkerConfigurationTest(unittest.TestCase):
    def test_reused_worker_selects_and_resets_arithmetic_architecture(self):
        simulation=SimpleNamespace(VFLAGS=['--binary','-j','4','-O3'])
        configure_rtlsim(simulation,1,2)
        first=list(simulation.VFLAGS)
        configure_rtlsim(simulation,1,2)
        self.assertEqual(simulation.VFLAGS,first)
        self.assertIn('-DOTPU_FP_MUL_LAT=1',first)
        self.assertIn('-DOTPU_FP_ADD_LAT=2',first)
        configure_rtlsim(simulation,2,4)
        self.assertFalse(any(f.startswith('-DOTPU_FP_') for f in simulation.VFLAGS))
        original=SimpleNamespace(VFLAGS=['--binary','-j','4','-O3'])
        configure_rtlsim(original)
        self.assertEqual(simulation.VFLAGS,original.VFLAGS)

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
