"""Measure native functional execution and its remaining Python trace adapter.

This scalar/control microbenchmark is not a neural-network end-to-end claim.
It runs exactly the same FeSens instructions and image in both ISA frontends.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

from . import REVISION
from .native_scalar import execute_scalar,scalar_isa
from .workload import execute,use_checkout


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--opentpu-root",type=Path,required=True)
  p.add_argument("--output",type=Path,required=True)
  p.add_argument("--cache",type=Path)
  p.add_argument("--jobs",type=int,default=8000)
  p.add_argument("--runs",type=int,default=5)
  a = p.parse_args()
  if not 1<=a.jobs<=60000 or a.runs<3:
    p.error("jobs must be 1..60000 and runs must be at least three")
  use_checkout(a.opentpu_root)
  import numpy as np
  from opentpu import Config,isa as I
  cfg = Config(S=1,D=128,MCOLS=4,ACT_ROWS=16,ACT_BLOCKS=64,LANES=8,
               DRAM_BYTES=4<<20,IMEM_WORDS=4096)
  image = np.zeros(cfg.DRAM_BYTES,np.uint8)
  image.view(np.uint32)[0] = a.jobs
  image.view(np.uint32)[16:32] = np.arange(16,dtype=np.uint32)*1234567
  program = [I.ld(0,0,1),I.rld(1,0,raw=True),I.li(2,0x10000),I.loop(4,0,rcount=1),
      I.ld(64,32,16),I.st(0,32,16,ra=2),I.addi(2,2,64),I.nop(),I.halt()]
  programs,images = [program],[image]
  max_steps = 4*a.jobs+6
  native = scalar_isa.compile(a.cache)
  golden = execute(cfg,programs,images)
  got = execute_scalar(cfg,programs,images,cache_dir=a.cache,max_steps=max_steps)
  expected,actual = golden.machine.slices[0],got.machine.slices[0]
  np.testing.assert_array_equal(actual.dram,expected.dram)
  np.testing.assert_array_equal(actual.tmem,expected.tmem)
  assert actual.R==expected.R and actual.icount==expected.icount
  assert len(golden.commands[0])==len(got.commands[0])
  for left,right in zip(golden.commands[0],got.commands[0]):
    assert left.pc==right.pc and left.footprint==right.footprint
    assert left.instruction.encode()==right.instruction.encode()
  del golden,got
  encoded = I.assemble(program)
  dram = image.copy().view(np.uint32)
  tmem,regs,stack = np.zeros(cfg.TMEM_WORDS,np.uint32),np.zeros(16,np.uint32),np.zeros(12,np.int64)
  trace = np.empty(max_steps*9,np.uint32)
  def reset():
    dram[:] = image.view(np.uint32)
    tmem.fill(0)
    regs.fill(0)
    stack.fill(0)
  def raw():
    return native(encoded,dram,tmem,regs,stack,trace,max_steps)
  reset()
  assert raw()==expected.icount
  np.testing.assert_array_equal(dram.view(np.uint8),expected.dram)
  np.testing.assert_array_equal(tmem,expected.tmem)
  assert regs.tolist()==expected.R
  functions = {"python_prepass":lambda:execute(cfg,programs,images),
      "native_prepass_with_python_adapter":lambda:execute_scalar(cfg,programs,images,
          cache_dir=a.cache,max_steps=max_steps),"native_kernel_only":raw}
  samples = {name:[] for name in functions}
  for repeat in range(a.runs):
    names = list(functions)
    names = names[repeat%3:]+names[:repeat%3]
    for name in names:
      gc.collect()
      if name=="native_kernel_only":
        reset()  # Deliberately outside the raw-kernel timed region.
      before = time.perf_counter()
      value = functions[name]()
      samples[name].append(time.perf_counter()-before)
      del value
  medians = {name:statistics.median(values) for name,values in samples.items()}
  report = {"workload":"data-dependent hardware loop, register-relative LD/ST, 16 words/transfer",
      "scope":"scalar/control prepass only; no matrix, quantizer, VPU, LPN, or RTL timing",
      "opentpu_revision":REVISION,"jobs":a.jobs,"dynamic_instructions":expected.icount,
      "numerical_memory_register_and_metadata_match":True,
      "program_sha256":hashlib.sha256(encoded.tobytes()).hexdigest(),
      "image_sha256":hashlib.sha256(image.tobytes()).hexdigest(),
      "host":platform.platform(),"python":platform.python_version(),"numpy":np.__version__,
      "compile_or_cache_load_seconds":native.compile_seconds,"compile_cache_hit":native.cache_hit,
      "seconds_samples":samples,"seconds_median":medians,
      "prepass_with_adapter_speedup":medians["python_prepass"]/medians["native_prepass_with_python_adapter"],
      "kernel_only_vs_python_prepass_ratio":medians["python_prepass"]/medians["native_kernel_only"],
      "raw_kernel_excludes":"buffer allocation/reset, instruction assembly, trace adaptation and compilation"}
  a.output.parent.mkdir(parents=True,exist_ok=True)
  a.output.write_text(json.dumps(report,indent=2)+"\n")
  print(json.dumps(report,indent=2))


if __name__=="__main__":
  main()
