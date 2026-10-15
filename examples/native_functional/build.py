"""Build a Python-authored functional producer and LPN into one C++ program."""

import argparse
from pathlib import Path
import shutil
import subprocess

from .model import resolve_transfers,transfer_timing


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--lpn-compiler",type=Path,required=True)
  p.add_argument("--output",type=Path,required=True)
  a = p.parse_args()
  out = a.output.resolve()
  out.mkdir(parents=True,exist_ok=True)
  functional = resolve_transfers.compile(out/"functional-cache")
  ir,library = out/"timing.mlir",out/"timing.so"
  ir.write_text(transfer_timing().build())
  subprocess.run([str(a.lpn_compiler.resolve()),str(ir),"-o",str(library),"--shared"],check=True)
  root = Path(__file__).resolve().parents[2]
  command = [shutil.which("clang++-20"),"-O3","-I",str(root/"include"),
      str(Path(__file__).with_name("driver.cpp")),str(functional.path),str(library),
      "-Wl,-rpath,"+str(functional.path.parent),"-Wl,-rpath,"+str(out),
      "-o",str(out/"native-functional-timing")]
  subprocess.run(command,check=True)
  print(out/"native-functional-timing")


if __name__=="__main__":
  main()
