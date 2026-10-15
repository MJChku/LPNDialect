"""Bit-exact scalar ISA, control flow and timing metadata versus upstream."""

import argparse
import tempfile
import unittest

from examples.opentpu.native_scalar import execute_scalar
from examples.opentpu.workload import execute,use_checkout


class NativeScalarTest(unittest.TestCase):
  def compare(self,program,words=(),args=()):
    import numpy as np
    from opentpu import Config
    cfg = Config(D=128,MCOLS=4,DRAM_BYTES=4096,IMEM_WORDS=4096)
    image = np.zeros(cfg.DRAM_BYTES,np.uint8)
    image.view(np.uint32)[:len(words)] = words
    golden = execute(cfg,[program],[image],list(args))
    native = execute_scalar(cfg,[program],[image],list(args),max_steps=1000)
    a,b = golden.machine.slices[0],native.machine.slices[0]
    np.testing.assert_array_equal(a.dram,b.dram)
    np.testing.assert_array_equal(a.tmem,b.tmem)
    self.assertEqual(a.R,b.R)
    self.assertEqual(a.icount,b.icount)
    self.assertEqual([c.pc for c in golden.commands[0]],[c.pc for c in native.commands[0]])
    for expected,got in zip(golden.commands[0],native.commands[0]):
      self.assertEqual(expected.instruction.encode(),got.instruction.encode())
      self.assertEqual(expected.footprint,got.footprint)
    return native

  def test_nested_loops_data_dependent_count_and_wraparound(self):
    from opentpu import isa as I
    p = [I.li(0,999),I.addi(5,8,1),I.ld(0,0,1),I.rld(1,0,raw=True),I.li(2,0xfffffffc),
         I.loop(5,0,rcount=1),I.addi(2,2,4),I.loop(1,2),I.addi(3,3,7),
         I.st(1024,0,1,ra=2),I.nop(),I.halt()]
    n = self.compare(p,[2],[0xffffffff])
    self.assertEqual(n.machine.slices[0].R[:6],[0,2,4,28,0,0])

  def test_zero_count_and_single_slice_barrier(self):
    from opentpu import isa as I
    self.compare([I.loop(2,0),I.li(1,99),I.nop(),I.bar(),I.halt()])

  def test_float_to_integer_rld_special_values_and_multiplier(self):
    from opentpu import isa as I
    words = [0,0x80000000,1,0x3f000000,0x3fc00000,0xbfc00000,0x4f000000,
             0x7f800000,0x7fc00000,0x4effffff,0xceffffff]
    p = [I.ld(0,0,len(words))]+[I.rld(j+1,j) for j in range(len(words))]+[
        I.rld(12,4,mul=3,bound=(3,3)),I.halt()]
    self.compare(p,words)

  def test_register_relative_memory_and_bound(self):
    from opentpu import isa as I
    p = [I.li(1,4),I.li(2,7),I.ld(0,0,2,ra=1,rb=2),I.rld(3,0,ra=2,raw=True,bound=(17,17)),
         I.st(64,0,2,rb=2),I.halt()]
    self.compare(p,[0,17,29])

  def test_unsupported_numeric_kernels_are_explicit(self):
    import numpy as np
    from opentpu import Config,isa as I
    cfg = Config(D=128,MCOLS=4,DRAM_BYTES=1024)
    for ins in (I.qact(0,1,0,1,128),I.mm(0,0,0,1,1,128,1,1,0,4,unit=True)):
      with self.subTest(op=ins.op),self.assertRaisesRegex(ValueError,"not ported"):
        execute_scalar(cfg,[[ins,I.halt()]],[np.zeros(1024,np.uint8)])

  def test_invalid_memory_and_run_limits(self):
    import numpy as np
    from opentpu import Config,isa as I
    cfg = Config(D=128,MCOLS=4,DRAM_BYTES=1024)
    for p,pattern,limit in (([I.ld(1,0,1),I.halt()],"memory",10),
                            ([I.loop(1,100),I.nop(),I.halt()],"step limit",10),
                            ([I.loop(0,1),I.halt()],"empty loop",10)):
      with self.subTest(pattern=pattern),self.assertRaisesRegex(ValueError,pattern):
        execute_scalar(cfg,[p],[np.zeros(1024,np.uint8)],max_steps=limit)


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--opentpu-root",required=True)
  args = parser.parse_args()
  use_checkout(args.opentpu_root)
  unittest.main(argv=[__file__],verbosity=2)
