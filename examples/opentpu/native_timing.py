"""Compile a capacity-bounded timing architecture once; bind new command tables."""

import fcntl
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile
import time

from lpnlang_mlir.native_inputs import write_input_tables
from .compact import CompactOpenTPU
from .benchmark import native_observation


class ReusableTiming:
  def __init__(self,compiler,cache,capacity=8192):
    self.compiler = Path(compiler).resolve()
    self.cache = Path(cache).resolve()
    self.capacity = capacity
    with self.compiler.open("rb") as f:
      identity = hashlib.file_digest(f,"sha256").hexdigest()
    runtime = self.compiler.parent/"libLPNNativeRuntime.a"
    with runtime.open("rb") as f:
      identity += hashlib.file_digest(f,"sha256").hexdigest()
    self.compiler_identity = identity

  def prepare(self,workload,timing,inputs,prefetch_profile=None):
    before = time.perf_counter()
    model = CompactOpenTPU(workload,timing,input_capacity=self.capacity,prefetch_profile=prefetch_profile)
    # Builder IDs are per-process counters. They distinguish SSA scopes but
    # carry no model semantics; normalize them for a stable architecture key.
    source = re.sub(r"%t[0-9]+_","%v_",model.build())
    key = hashlib.sha256((self.compiler_identity+source).encode()).hexdigest()
    root = self.cache/key
    root.mkdir(parents=True,exist_ok=True)
    inputs = Path(inputs).resolve()
    inputs.parent.mkdir(parents=True,exist_ok=True)
    write_input_tables(inputs,model.input_tables)
    preparation = time.perf_counter()-before
    binary,ir = root/"timing.native",root/"timing.mlir"
    before = time.perf_counter()
    with (root/"compile.lock").open("a") as lock:
      fcntl.flock(lock,fcntl.LOCK_EX)
      hit = binary.exists()
      if not hit:
        with tempfile.TemporaryDirectory(dir=root) as tmp:
          temporary = Path(tmp)/"timing.native"
          ir.write_text(source)
          result = subprocess.run([str(self.compiler),str(ir),"-o",str(temporary)],
              capture_output=True,text=True)
          (root/"compile.log").write_text(result.stdout+result.stderr)
          result.check_returncode()
          temporary.replace(binary)
    return PreparedTiming(binary,inputs,preparation,time.perf_counter()-before,hit,
        model.statistics(),len(source.encode()))


class PreparedTiming:
  def __init__(self,binary,inputs,preparation_seconds,compile_seconds,cache_hit,statistics,ir_bytes):
    self.binary,self.inputs = binary,inputs
    self.preparation_seconds,self.compile_seconds = preparation_seconds,compile_seconds
    self.cache_hit,self.statistics,self.ir_bytes = cache_hit,statistics,ir_bytes

  def run(self,trace=False):
    return native_observation(self.binary,trace=trace,inputs=self.inputs)
