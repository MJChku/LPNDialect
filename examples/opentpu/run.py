"""Compile a shared openTPU input bundle into an event-driven LPN executable."""

import argparse
from pathlib import Path
import subprocess
import time

from .model import OpenTPU, Timing
from .workload import load, use_checkout


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--opentpu-root",type=Path,required=True)
  p.add_argument("--input",type=Path,required=True)
  p.add_argument("--emit",type=Path,required=True)
  p.add_argument("--lpn-compiler",type=Path)
  p.add_argument("--output",type=Path)
  p.add_argument("--batch-rows",type=int,default=1)
  p.add_argument("--compact",action="store_true")
  p.add_argument("--matrix-engines",type=int,default=1,
      help="independent matrix engines sharing memory; requires --compact")
  a = p.parse_args()
  if bool(a.lpn_compiler)!=bool(a.output):
    p.error("--lpn-compiler and --output must be supplied together")
  if a.matrix_engines<1 or a.matrix_engines!=1 and not a.compact:
    p.error("positive --matrix-engines requires --compact when greater than one")
  use_checkout(a.opentpu_root)
  before = time.perf_counter()
  workload = load(a.input)
  from .compact import CompactOpenTPU
  cls = CompactOpenTPU if a.compact else OpenTPU
  model = cls(workload,Timing(batch_rows=a.batch_rows,dram_bytes_per_cycle=min(workload.config.D,128),
      matrix_engines=a.matrix_engines))
  a.emit.write_text(model.build())
  print(f"Loaded and emitted in {time.perf_counter()-before:.3f}s; "
        f"instructions per slice: {[len(c) for c in workload.commands]}")
  if a.compact:
    print(f"Net: {model.statistics()}")
  if a.lpn_compiler:
    subprocess.run([str(a.lpn_compiler.resolve()),str(a.emit),"-o",str(a.output)],check=True)


if __name__=="__main__":
  main()
