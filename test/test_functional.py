"""Native functional-model authoring, mutable state, and failure semantics."""

import argparse
from pathlib import Path
import shutil
import tempfile
import unittest

from lpnfunc import functional,Buffer,u32,i32,i64,f32,boolean,bitcast


@functional
def route(priority:Buffer[u32],result:Buffer[u32],n:i64)->i64:
  position:i64 = 0
  total:u32 = 0
  while position<n:
    if priority[position]>3:
      total = u32(total+priority[position])
      result[position] = total
    else:
      result[position] = 0
    position += 1
  return position


@functional
def copy_words(source:Buffer[u32],destination:Buffer[u32],n:i64)->i64:
  for j in range(n):
    destination[j] = source[j]
  return n


@functional
def affine(source:Buffer[f32],out:Buffer[f32],n:i64)->i64:
  for j in range(n):
    out[j] = source[j]*2.0+1.0
  return n


@functional
def divide(a:i64,b:i64)->i64:
  return a//b


@functional
def shift(a:u32,n:u32)->u32:
  return u32(a<<n)


@functional
def float_bits(a:f32)->u32:
  return bitcast(a,u32)


@functional
def is_nonzero(a:i64)->boolean:
  return boolean(a)


@functional
def short_circuit(a:Buffer[u32],n:i64)->boolean:
  return n<len(a) and a[n]>0


@functional
def first_error(a:Buffer[u32],b:i64)->i64:
  x:i64 = 1//b
  a[0] = u32(x)
  return x


@functional
def conditional_local(flag:boolean)->i64:
  value:i64
  if flag:
    value = 7
  else:
    value = 11
  return value


@functional
def uninitialized(flag:boolean)->i64:
  if flag:
    value:i64 = 7
  return value


@functional
def unsupported(a:i64)->i64:
  return abs(a)


@functional
def range_target(lo:i64,hi:i64,step:i64)->i64:
  j:i64 = 42
  total:i64 = 0
  for j in range(lo,hi,step):
    total += j
    j = 9
  return total+j


@functional
def value_and(a:i64,b:i64)->i64:
  return a and b


@functional
def bool_sum(a:boolean,b:boolean)->i64:
  return a+b


class FunctionalTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not all(shutil.which(n) for n in ("mlir-opt-20","mlir-translate-20","clang++-20")):
      raise unittest.SkipTest("MLIR/LLVM 20 functional compiler tools not installed")
    try:
      import numpy as np
    except ImportError:
      raise unittest.SkipTest("NumPy not installed")
    cls.np = np
    cls.temporary = tempfile.TemporaryDirectory(prefix="lpn-functional-tests-")
    cls.cache = Path(cls.temporary.name)

  @classmethod
  def tearDownClass(cls):
    if hasattr(cls,"temporary"):
      cls.temporary.cleanup()

  def test_data_dependent_control_and_mutable_buffers(self):
    np = self.np
    a = np.array([1,0xffffffff,7,3,9,4],np.uint32)
    golden,got = np.zeros_like(a),np.zeros_like(a)
    with np.errstate(over="ignore"):
      expected = route(a,golden,len(a))
    self.assertEqual(route.compile(self.cache)(a,got,len(a)),expected)
    np.testing.assert_array_equal(got,golden)

  def test_aliasing_retains_sequential_buffer_semantics(self):
    np = self.np
    golden,got = np.arange(8,dtype=np.uint32),np.arange(8,dtype=np.uint32)
    copy_words(golden[:-1],golden[1:],7)
    copy_words.compile(self.cache)(got[:-1],got[1:],7)
    np.testing.assert_array_equal(got,golden)

  def test_f32_and_bitcast(self):
    np = self.np
    a = np.array([-0.0,-1.5,0.25,1024.0],np.float32)
    golden,got = np.zeros_like(a),np.zeros_like(a)
    affine(a,golden,len(a))
    affine.compile(self.cache)(a,got,len(a))
    np.testing.assert_array_equal(got.view(np.uint32),golden.view(np.uint32))
    native = float_bits.compile(self.cache)
    for x in a:
      self.assertEqual(native(x),float_bits(x))

  def test_signed_floor_division(self):
    native = divide.compile(self.cache)
    for a,b in ((-7,3),(7,-3),(-7,-3),(7,3)):
      self.assertEqual(native(a,b),divide(a,b))
    for a,b in ((1,0),(-(1<<63),-1)):
      with self.assertRaises(ValueError):
        native(a,b)

  def test_unsigned_shift_and_boolean_cast(self):
    native = shift.compile(self.cache)
    self.assertEqual(native(1,31),0x80000000)
    self.assertEqual(native(0x80000000,1),0)
    with self.assertRaises(ValueError):
      native(1,32)
    check = is_nonzero.compile(self.cache)
    for x in (0,1,2,-2):
      self.assertEqual(check(x),bool(x))

  def test_bounds_errors_prevent_later_stores(self):
    np = self.np
    native = copy_words.compile(self.cache)
    source,destination = np.arange(2,dtype=np.uint32),np.full(4,99,np.uint32)
    with self.assertRaises(IndexError):
      native(source,destination,4)
    np.testing.assert_array_equal(destination,[0,1,99,99])
    with self.assertRaises(IndexError):
      native(np.zeros(0,np.uint32),destination,1)

  def test_first_error_is_preserved(self):
    with self.assertRaises(ValueError):
      first_error.compile(self.cache)(self.np.zeros(0,self.np.uint32),0)

  def test_short_circuit_does_not_access_missing_element(self):
    native = short_circuit.compile(self.cache)
    a = self.np.array([7],self.np.uint32)
    self.assertFalse(native(a,1))
    self.assertTrue(native(a,0))

  def test_branch_results_and_uninitialized_local_rejection(self):
    native = conditional_local.compile(self.cache)
    self.assertEqual(native(True),7)
    self.assertEqual(native(False),11)
    with self.assertRaisesRegex(TypeError,"before assignment"):
      uninitialized.emit()

  def test_no_silent_python_fallback(self):
    with self.assertRaisesRegex(TypeError,"unsupported expression"):
      unsupported.compile(self.cache)
    with self.assertRaisesRegex(TypeError,"boolean operands"):
      value_and.emit()
    with self.assertRaisesRegex(TypeError,"explicit integer cast"):
      bool_sum.emit()

  def test_range_target_and_overflow(self):
    native = range_target.compile(self.cache)
    for lo,hi,step in ((0,4,1),(5,5,1),(0,10,3),((1<<63)-2,(1<<63)-1,2)):
      self.assertEqual(native(lo,hi,step),range_target(lo,hi,step))

  def test_buffer_contract_and_cache(self):
    np = self.np
    native = copy_words.compile(self.cache)
    self.assertTrue(copy_words.compile(self.cache).cache_hit)
    with self.assertRaises(TypeError):
      native(np.zeros(2,np.int64),np.zeros(2,np.uint32),2)
    with self.assertRaises(TypeError):
      native(np.zeros(4,np.uint32)[::2],np.zeros(2,np.uint32),2)
    readonly = np.zeros(2,np.uint32)
    readonly.flags.writeable = False
    with self.assertRaises(TypeError):
      native(readonly,np.zeros(2,np.uint32),2)


if __name__=="__main__":
  unittest.main(verbosity=2)
