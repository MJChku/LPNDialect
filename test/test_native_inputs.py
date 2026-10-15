"""Native producer/timing ABI, table validation, and model reuse across streams."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from lpnlang_mlir import NetBuilder,InputArray
from lpnlang_mlir.native_inputs import write_input_tables


def duration_net(values):
  net = NetBuilder("runtime_durations")
  durations = InputArray("durations",values)
  cursor = net.place("cursor",initial_tokens=1)
  done = net.place("done",observable=True)

  @net.func(args=(),results=("i64",))
  def command_count():
    func_return(index_cast(array_len(array(durations)),src_type="index",dst_type="i64"))

  @net.transition
  def step():
    guard(count(cursor)>0)
    idx = peek(cursor).get("id")
    guard(idx<command_count())
    token = take(cursor)
    emit(cursor,token.set("id",idx+1),delay=sitofp(array(durations)[idx]))

  @net.transition
  def complete():
    guard(count(cursor)>0)
    guard(peek(cursor).get("id")==command_count())
    take(cursor)
    emit(done,create())

  return net


def input_copy_net():
  net = NetBuilder("immutable_input")
  values = InputArray("values",[7,11])
  seed = net.place("seed",initial_tokens=1)
  out = net.place("out",observable=True)

  @net.transition
  def copy():
    take(seed)
    changed = array_set(array(values),0,999)
    emit(out,create().set("copied",changed[0]).set("original",array(values)[0]))

  return net


class NativeInputsTest(unittest.TestCase):
  tool = None

  def compile(self,out,shared=False):
    ir,binary = out/"model.mlir",out/("model.so" if shared else "model.native")
    ir.write_text(duration_net([2,3]).build())
    subprocess.run([self.tool,str(ir),"-o",str(binary)]+(["--shared"] if shared else []),
        check=True,capture_output=True,text=True)
    return binary

  def test_one_binary_accepts_different_stream_lengths_and_values(self):
    with tempfile.TemporaryDirectory() as d:
      out = Path(d);binary = self.compile(out)
      for values in ([2,3],[100,1,7],[]):
        tables = out/"input.tables"
        write_input_tables(tables,{"durations":values})
        r = subprocess.run([str(binary),"--inputs="+str(tables),"--stats"],
            check=True,capture_output=True,text=True)
        cycles = float(re.search(r"finished at t=([0-9.]+)",r.stdout)[1])
        self.assertEqual(cycles,sum(values))
        self.assertIn("place done: 1 token(s)",r.stdout)
        self.assertIn("rollbacks: 0",r.stdout)
        self.assertEqual(duration_net(values).python_simulator().run(),sum(values))

  def test_missing_and_malformed_tables_fail(self):
    with tempfile.TemporaryDirectory() as d:
      out = Path(d);binary = self.compile(out)
      self.assertNotEqual(subprocess.run([str(binary)],capture_output=True).returncode,0)
      for data in (b"bad",b"LPNTBL1\0"+(1).to_bytes(8,"little"),
                   b"LPNTBL1\0"+bytes(8)+b"extra"):
        tables = out/"bad.tables";tables.write_bytes(data)
        r = subprocess.run([str(binary),"--inputs="+str(tables)],capture_output=True)
        self.assertNotEqual(r.returncode,0)

  def test_updating_a_bound_array_copies_without_changing_input(self):
    with tempfile.TemporaryDirectory() as d:
      out = Path(d);ir,binary,tables = out/"copy.mlir",out/"copy.native",out/"copy.tables"
      ir.write_text(input_copy_net().build())
      subprocess.run([self.tool,str(ir),"-o",str(binary)],check=True,capture_output=True,text=True)
      write_input_tables(tables,{"values":[7,11]})
      r = subprocess.run([str(binary),"--inputs="+str(tables),"--tokens"],
          check=True,capture_output=True,text=True)
      self.assertIn("'copied': 999",r.stdout)
      self.assertIn("'original': 7",r.stdout)
      sim = input_copy_net().python_simulator();sim.run()
      self.assertEqual(sim.place_dicts("out"),[{"copied":999,"original":7}])
      with self.assertRaises(TypeError):
        write_input_tables(tables,{"bad":[1.5]})

  def test_cpp_producer_binds_borrowed_arrays_and_reuses_shared_model(self):
    with tempfile.TemporaryDirectory() as d:
      out = Path(d);library = self.compile(out,shared=True)
      source = out/"driver.cpp"
      source.write_text('''#include "LPN/Runtime/NativeInterface.h"
#include <cstdint>
int main() {
  uint64_t first[] = {2,3}, second[] = {7,11,13};
  char name[] = "native-producer", stats[] = "--stats";
  char *args[] = {name,stats};
  if (lpn_bind_input_array("durations",-1,first)!=2) return 10;
  if (lpn_bind_input_array("durations",2,first)) return 11;
  if (lpn_simulate(2,args)) return 12;
  if (lpn_simulated_time()!=5) return 15;
  if (lpn_bind_input_array("durations",3,second)) return 13;
  if (lpn_simulate(2,args)) return 14;
  if (lpn_simulated_time()!=31) return 16;
  return lpn_clear_input_arrays();
}
''')
      binary = out/"driver"
      include = Path(__file__).resolve().parents[1]/"include"
      subprocess.run([shutil.which("clang++-20"),"-O3","-I",str(include),str(source),
          str(library),"-Wl,-rpath,"+str(out),"-o",str(binary)],check=True,capture_output=True,text=True)
      r = subprocess.run([str(binary)],check=True,capture_output=True,text=True)
      self.assertEqual([float(v) for v in re.findall(r"finished at t=([0-9.]+)",r.stdout)],[5,31])
      self.assertEqual(r.stdout.count("rollbacks: 0"),2)


if __name__=="__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--tool",required=True)
  a = p.parse_args()
  NativeInputsTest.tool = a.tool
  unittest.main(argv=[__file__],verbosity=2)
