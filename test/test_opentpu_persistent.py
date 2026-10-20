"""Resident functional state and real controls across hardware program launches."""
import argparse
from pathlib import Path
import unittest

from examples.opentpu.workload import use_checkout,execute_on_machine


class PersistentTest(unittest.TestCase):
    def machine(self):
        import numpy as np
        from opentpu import Config,isa as I
        from opentpu.isasim import Machine
        cfg=Config(D=128,MCOLS=4,LANES=64,ACT_ROWS=16,DRAM_BYTES=1<<20)
        image=np.zeros(cfg.DRAM_BYTES,np.uint8);image[:64]=np.arange(16,dtype=np.float32).view(np.uint8)
        return Machine(cfg,[[I.halt()]],[image])

    def test_resident_values_and_tmem_control_feed_the_next_program_without_copy(self):
        import numpy as np
        from opentpu import isa as I
        machine=self.machine();address=id(machine.slices[0].dram)
        first=[I.ld(0,0,16),I.vop(I.V_ADD,0,0,0,1,16,16,16,0,I.B_SCALAR,2.0),I.st(64,0,16),I.halt()]
        before=execute_on_machine(machine,[first])
        second=[I.rld(1,0,mul=4),I.ld(0,32,1,ra=1),I.st(128,32,1),I.halt()]
        after=execute_on_machine(machine,[second])
        self.assertEqual(id(machine.slices[0].dram),address)
        self.assertEqual(after.images,[])
        self.assertEqual(after.commands[0][1].instruction.w[0],8)
        np.testing.assert_array_equal(machine.slices[0].dram[64:128].view(np.float32),np.arange(16,dtype=np.float32)+2)
        self.assertEqual(machine.slices[0].dram[128:132].view(np.float32)[0],2)
        self.assertEqual(len(before.commands[0]),4)
        self.assertIs(before.programs[0],first)
        self.assertNotIn('step',machine.slices[0].__dict__)

    def test_failed_recording_restores_the_numerical_machine(self):
        from opentpu import isa as I
        machine=self.machine()
        with self.assertRaisesRegex(ValueError,'not supported'):
            execute_on_machine(machine,[[I.Instr(I.WAITW),I.halt()]])
        self.assertNotIn('step',machine.slices[0].__dict__)
        work=execute_on_machine(machine,[[I.ld(0,0,1),I.halt()]])
        self.assertEqual(len(work.commands[0]),2)

    def test_native_nvfp4_factory_matches_independent_element_values(self):
        import numpy as np
        from opentpu import quant as Q
        from examples.opentpu.glm47_test_weights import native_matrix,rng_for
        n,k=3,256;rng=rng_for('unit')
        codes=rng.integers(0,16,(n,k),dtype=np.uint8)
        scales=rng.integers(48,65,(n,k//16),dtype=np.uint8)
        mag=np.asarray([0,.5,1,1.5,2,3,4,6],np.float32)[codes&7]
        sign=np.where(codes&8,-1.,1.).astype(np.float32)
        exponent=(scales>>3).astype(np.int32);mantissa=(scales&7).astype(np.float32)
        scale=np.ldexp(1+mantissa/8,exponent-7)
        expected=mag*sign*np.repeat(scale,16,axis=1)*np.float32(1/(3.5*k**.5))
        q,s=native_matrix('unit',n,k);actual=Q.dequantize_mxu(q,s,'nvfp4',k,128)
        np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-6)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);a=p.parse_args()
    use_checkout(a.opentpu_root);unittest.main(argv=[__file__],verbosity=2)
