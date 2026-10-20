"""Hardware-input and arithmetic checks against the independent ISA execution."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from examples.opentpu import REVISION
from examples.opentpu.hardware_commands import load_commands,instruction_statistics
from examples.opentpu.workload import execute,use_checkout


class HardwareCommandsTest(unittest.TestCase):
    tool=None

    def write(self,root,cfg,program,args=()):
        from opentpu import isa as I,rtlsim
        (root/'input.json').write_text(json.dumps(dict(revision=REVISION,config=vars(cfg),
            args=list(args),instruction_counts=[len(program)])))
        rtlsim._write_hex(root/'prog_0.hex',I.assemble(program))

    def test_same_hex_and_control_as_numerical_isa(self):
        import numpy as np
        from opentpu import Config,isa as I
        from examples.opentpu.model import Timing
        from examples.opentpu.native_timing import ReusableTiming
        cfg=Config(D=128,MCOLS=4,ACT_ROWS=16,LANES=64)
        # Two iterations of 3x5x384 and one PAIR 2x5x384: PAIR changes
        # issue service, not mathematical work. No weight values are needed
        # for static control; the independent ISA still computes the zeros.
        program=[I.li(1,2),I.loop(1,0,rcount=1),
            I.mm(0,0,0,5,3,256,8,3,0,8,unit=True,wf=I.WNV),
            I.mm(4096,0,128,5,3,256,8,2,0,8,unit=True,wf=I.WNV,pair=True),
            I.vop(I.V_MUL,256,0,128,2,5,8,8,8),I.halt()]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.write(root,cfg,program)
            static=load_commands(root)
            numerical=execute(cfg,[program],[np.zeros(cfg.DRAM_BYTES,np.uint8)])
            self.assertEqual([(c.pc,c.instruction.encode()) for c in static.commands[0]],
                             [(c.pc,c.instruction.encode()) for c in numerical.commands[0]])
            a=instruction_statistics(static)
            self.assertEqual(a,instruction_statistics(numerical))
            self.assertEqual(a['matrix_commands'],3)
            self.assertEqual(a['matrix_dot_macs'],15360)
            self.assertEqual(a['matrix_dot_operations'],30720)
            self.assertEqual(a['logical_fp32_vector_flops'],10)
            self.assertEqual(a['matrix_by_weight_format']['nvfp4']['commands'],3)
            self.assertEqual(a['dynamic_instructions'],7)
            if self.tool:
                t=Timing(dram_bytes_per_cycle=128)
                p=ReusableTiming(self.tool,root/'cache',capacity=32)
                x=p.prepare(static,t,root/'static.tables');y=p.prepare(numerical,t,root/'numeric.tables')
                self.assertEqual(x.run()[0],y.run()[0]);self.assertEqual(x.binary,y.binary)

    def test_tensor_graphs_wide_addresses_and_dynamic_control_fail(self):
        from opentpu import Config,isa as I
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'commands.json').write_text(json.dumps(dict(version=1,commands=[dict(unit='matrix',macs=100)])))
            with self.assertRaisesRegex(ValueError,'tensor graphs'):load_commands(root)
            self.write(root,Config(DRAM_BYTES=32_000_000_000),[I.halt()])
            with self.assertRaisesRegex(ValueError,'32-bit'):load_commands(root)
            self.write(root,Config(),[I.rld(1,0),I.halt()])
            with self.assertRaisesRegex(ValueError,'numerical/control'):load_commands(root)

    def test_word_tampering_and_manifest_mismatch_rejected(self):
        from opentpu import Config,isa as I
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg=Config();self.write(root,cfg,[I.halt()])
            (root/'prog_0.hex').write_text('100000000\n')
            with self.assertRaisesRegex(ValueError,'32 bits'):load_commands(root)
            self.write(root,cfg,[I.halt()])
            m=json.loads((root/'input.json').read_text());m['instruction_counts']=[2]
            (root/'input.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'length'):load_commands(root)
            self.write(root,replace(cfg,IMEM_WORDS=4),[I.halt()])
            with self.assertRaisesRegex(ValueError,'IMEM'):load_commands(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True)
    p.add_argument('--tool',type=Path);a=p.parse_args();use_checkout(a.opentpu_root)
    HardwareCommandsTest.tool=a.tool;unittest.main(argv=[__file__],verbosity=2)
