"""Compare complete old and reusable LPN paths on one frozen real input bundle.

The reference checkout is read only; both workers execute identical instructions
and data. Worker timers exclude initial input-file loading and package imports,
matching the trained runner's resident-input timing scope.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def worker(a):
  from examples.opentpu.workload import load,execute,use_checkout
  from examples.opentpu.model import Timing
  from examples.opentpu.compact import CompactOpenTPU
  from examples.opentpu.benchmark import native_observation
  use_checkout(a.opentpu_root)
  original = load(a.input)
  before = time.perf_counter()
  workload = execute(original.config,original.programs,original.images,original.args,original.hashes)
  front = time.perf_counter()-before
  timing = Timing(batch_rows=8,dram_bytes_per_cycle=min(128,workload.config.D))
  a.output.mkdir(parents=True,exist_ok=True)
  if a.mode=="original":
    before = time.perf_counter()
    model = CompactOpenTPU(workload,timing)
    source = model.build()
    ir,binary = a.output/"model.mlir",a.output/"model.native"
    ir.write_text(source)
    preparation = time.perf_counter()-before
    before = time.perf_counter()
    subprocess.run([str(a.compiler),str(ir),"-o",str(binary)],check=True,capture_output=True,text=True)
    compile_seconds = time.perf_counter()-before
    cycle,wall,process,log = native_observation(binary)
    hit = False
    objects = {"places":len(model.net._places),"transitions":len(model.net._transitions)}
  else:
    from examples.opentpu.native_timing import ReusableTiming
    runner = ReusableTiming(a.compiler,a.output/"cache",capacity=8192)
    prepared = runner.prepare(workload,timing,a.output/"input.tables")
    preparation,compile_seconds = prepared.preparation_seconds,prepared.compile_seconds
    cycle,wall,process,log = prepared.run()
    hit,objects = prepared.cache_hit,prepared.statistics
    source = (prepared.binary.parent/"timing.mlir").read_text()
  memory = [{"dram":hashlib.sha256(s.dram).hexdigest(),"tmem":hashlib.sha256(s.tmem).hexdigest()}
            for s in workload.machine.slices]
  result = {"mode":a.mode,"cycles":cycle,"functional_seconds":front,
      "preparation_seconds":preparation,"compile_seconds":compile_seconds,
      "native_core_seconds":wall,"native_process_seconds":process,
      "full_lpn_seconds":front+preparation+compile_seconds+process,
      "cache_hit":hit,"ir_bytes":len(source.encode()),"net":objects,
      "input_sha256":workload.hashes,"numerical_memory_sha256":memory}
  result["compiler_sha256"] = hashlib.sha256(a.compiler.read_bytes()).hexdigest()
  result["binary_sha256"] = hashlib.sha256((binary if a.mode=="original" else prepared.binary).read_bytes()).hexdigest()
  result["timing_ir_sha256"] = hashlib.sha256(source.encode()).hexdigest()
  print(json.dumps(result))


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--opentpu-root",type=Path,required=True)
  p.add_argument("--input",type=Path,required=True)
  p.add_argument("--output",type=Path,required=True)
  p.add_argument("--reference-root",type=Path)
  p.add_argument("--compiler",type=Path,required=True)
  p.add_argument("--reference-compiler",type=Path)
  p.add_argument("--runs",type=int,default=3)
  p.add_argument("--mode",choices=("original","reusable"),help=argparse.SUPPRESS)
  a = p.parse_args()
  if a.mode:
    worker(a)
    return
  if not a.reference_root or not a.reference_compiler or a.runs<3:
    p.error("reference-root, reference-compiler and at least three runs are required")
  a.output.mkdir(parents=True,exist_ok=True)
  project = Path(__file__).resolve().parents[2]
  records = {"original":[],"reusable":[]}
  warmups = {}
  for repeat in range(a.runs+1):
    names = ["original","reusable"] if repeat%2==0 else ["reusable","original"]
    for name in names:
      root = a.reference_root.resolve() if name=="original" else project
      compiler = a.reference_compiler if name=="original" else a.compiler
      command = [sys.executable,str(Path(__file__).resolve()),"--mode",name,
          "--opentpu-root",str(a.opentpu_root.resolve()),"--input",str(a.input.resolve()),
          "--output",str(a.output.resolve()/name),"--compiler",str(compiler.resolve())]
      env = dict(os.environ,PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE="1")
      result = subprocess.run(command,cwd=root,env=env,check=True,capture_output=True,text=True)
      record = json.loads(result.stdout)
      if repeat:
        records[name].append(record)
      else:
        warmups[name] = record
  all_records = [r for rows in records.values() for r in rows]
  expected = all_records[0]
  for r in all_records:
    assert r["cycles"]==expected["cycles"],"timing prediction changed"
    assert r["input_sha256"]==expected["input_sha256"],"input differs"
    assert r["numerical_memory_sha256"]==expected["numerical_memory_sha256"],"numerical result differs"
  medians = {name:{key:statistics.median(r[key] for r in rows) for key in (
      "functional_seconds","preparation_seconds","compile_seconds","native_core_seconds",
      "native_process_seconds","full_lpn_seconds")} for name,rows in records.items()}
  report = {"scope":"same real input, functional execution + metadata/model preparation + compilation/cache + timing process",
      "warmups":warmups,"samples":records,"seconds_median":medians,
      "full_lpn_path_speedup":medians["original"]["full_lpn_seconds"]/medians["reusable"]["full_lpn_seconds"],
      "cycles_and_all_numerical_memory_identical":True,
      "remaining":"Python/NumPy functional execution and Python metadata construction",
      "note":"three or more alternating samples after warmup; initial input loading and package imports excluded; concurrent RTL validation"}
  (a.output/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
  print(json.dumps({"seconds_median":medians,"full_lpn_path_speedup":report["full_lpn_path_speedup"]},indent=2))


if __name__=="__main__":
  main()
