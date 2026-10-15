"""Run identical executable programs and operands in FeSens RTL and LPN."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

from . import REVISION
from .model import OpenTPU, Timing
from .workload import load, use_checkout


def gemm(out,n=512,kb=32,jobs=7,m=4):
  import numpy as np
  from opentpu import Config, isa as I
  if not (1<=m<=16 and 1<=kb<=64 and 1<=n<=2048 and jobs>=1):
    raise ValueError("GEMM dimensions exceed the benchmark configuration")
  k = 128*kb
  if m*k+m*n>65536:
    raise ValueError("GEMM does not fit the benchmark memories")
  weight_base = max(0x10000,((4*m*k+127)//128)*128)
  output_base = max(0x300000,((weight_base+n*k+127)//128)*128)
  dram_bytes = 1<<max(22,(output_base+4*m*n-1).bit_length())
  cfg = Config(S=1,D=128,MCOLS=4,ACT_ROWS=16,ACT_BLOCKS=64,LANES=8,
               DRAM_BYTES=dram_bytes,IMEM_WORDS=4096)
  rng = np.random.default_rng(47)
  x = rng.standard_normal((m,k)).astype(np.float32)
  weights = rng.integers(-127,128,(n,k),dtype=np.int8)
  image = np.zeros(cfg.DRAM_BYTES,np.uint8)
  image[:x.nbytes] = x.view(np.uint8).reshape(-1)
  image[weight_base:weight_base+weights.nbytes] = weights.view(np.uint8).reshape(-1)
  output = m*k
  program = [I.ld(0,0,x.size),I.qact(0,m,0,kb,k),I.loop(1,jobs),
      I.mm(weight_base,0,output,n,kb,k,n,m,0,4,unit=True),
      I.st(output_base,output,m*n),I.halt()]
  write_inputs(out,cfg,[program],[image])
  np.save(Path(out)/"float_reference.npy",x@weights.astype(np.float32).T)
  return {"name":"repeated quantized GEMM","m":m,"n":n,"k":k,"jobs":jobs,
          "macs":m*n*k*jobs,"checked_output_elements":m*n,
          "seed":47,"output_tmem_word":output,"output_dram_byte":output_base}


def write_inputs(out,cfg,programs,images,args=None):
  import numpy as np
  from opentpu import isa as I
  out = Path(out)
  out.mkdir(parents=True,exist_ok=True)
  (out/"input.json").write_text(json.dumps({"revision":REVISION,"config":asdict(cfg),
      "args":list(args or []),"instruction_counts":[len(p) for p in programs]},indent=2)+"\n")
  for s in range(cfg.S):
    words = I.assemble(programs[s])
    # Exact same hex formatting and truncation as rtlsim._write_hex.
    nonzero = np.nonzero(words)[0]
    end = int(nonzero[-1])+1 if len(nonzero) else 1
    # HALT's trailing zero words are omitted by the RTL writer too.
    (out/f"prog_{s}.hex").write_text("\n".join(f"{int(w):08x}" for w in words[:end])+"\n")
    images[s].view("<u4").astype(">u4").tofile(out/f"dram_{s}.bin")


def mlp(out,m=4,h=1024,f=1024,tmem_words=65536):
  import numpy as np
  from opentpu import Config, reference, quant as Q
  from opentpu.kernels import mlp as kernel
  from opentpu.runtime import Input,Output,Weight,compile_kernel,collect
  from opentpu.isasim import Machine
  if min(m,h,f,tmem_words)<1 or h%128 or f%128:
    raise ValueError("MLP dimensions must be positive and hidden widths multiples of 128")
  allocations = [4*m*h,4*h]+2*[f*Q.row_bytes(h,"int8"),f*Q.scale_bytes(h,"int8")]+[
      h*Q.row_bytes(f,"int8"),h*Q.scale_bytes(f,"int8"),4*m*h]
  needed = sum(((n+127)//128)*128 for n in allocations)
  cfg = Config(S=1,D=128,MCOLS=4,ACT_ROWS=16,ACT_BLOCKS=64,LANES=8,
               DRAM_BYTES=1<<max(22,(needed-1).bit_length()),IMEM_WORDS=4096,
               TMEM_WORDS=tmem_words)
  rng = np.random.default_rng(47)
  x = rng.standard_normal((m,h)).astype(np.float32)
  gamma = (1+0.1*rng.standard_normal(h)).astype(np.float32)
  gate = (rng.standard_normal((f,h))/np.sqrt(h)).astype(np.float32)
  up = (rng.standard_normal((f,h))/np.sqrt(h)).astype(np.float32)
  down = (rng.standard_normal((h,f))/np.sqrt(f)).astype(np.float32)
  compiled,images = compile_kernel(kernel,cfg,h=Input(x),gamma=Input(gamma),
      w_gate=Weight(gate,0),w_up=Weight(up,0),w_down=Weight(down,0),
      out=Output((m,h)),eps=1e-6)
  write_inputs(out,cfg,compiled.programs,images)
  golden = Machine(cfg,compiled.programs,images).run()
  got = collect(compiled,[s.dram for s in golden.slices],[s.tmem for s in golden.slices],{}).outputs["out"]
  expected = reference.mlp(x,gamma,gate,up,down,1e-6)
  relative = float(np.linalg.norm(got.astype(np.float64)-expected)/np.linalg.norm(expected))
  if relative>0.02:
    raise RuntimeError("MLP numerical error exceeds the quantization tolerance")
  return {"name":"upstream-compiled SwiGLU MLP with RMSNorm and residual", "m":m,
      "hidden":h,"intermediate":f,"seed":47,"checked_output_elements":m*h,
      "float_reference_relative_error":relative,"macs":3*m*h*f}


def native_observation(binary,trace=False):
  args = [str(binary),"--stats","--wall-start=begin","--wall-end=complete","--max-time=1000000000000"]
  if trace:
    args.append("--trace")
  before = time.perf_counter()
  r = subprocess.run(args,check=True,capture_output=True,text=True)
  process = time.perf_counter()-before
  wall = re.search(r"Wall window elapsed: ([0-9.]+) seconds",r.stdout)
  cycles = re.search(r"Simulation finished at t=([0-9.]+)",r.stdout)
  if not wall or not cycles or "rollbacks: 0" not in r.stdout or "place program_completed: 1 token(s)" not in r.stdout:
    raise RuntimeError("native simulation did not complete with zero rollbacks")
  return float(cycles[1]),float(wall[1]),process,r.stdout


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--opentpu-root",type=Path,required=True)
  p.add_argument("--output",type=Path,required=True)
  p.add_argument("--lpn-compiler",type=Path,required=True)
  p.add_argument("--n",type=int,default=512)
  p.add_argument("--kb",type=int,default=32)
  p.add_argument("--jobs",type=int,default=7)
  p.add_argument("--m",type=int,default=4)
  p.add_argument("--hidden",type=int,default=1024)
  p.add_argument("--intermediate",type=int,default=1024)
  p.add_argument("--tmem-words",type=int,default=65536)
  p.add_argument("--rtl-timeout",type=float,default=3600)
  p.add_argument("--batch-rows",type=int,default=1)
  p.add_argument("--compact",action="store_true",help="reuse a fixed set of transitions for long traces")
  p.add_argument("--workload",choices=("gemm","mlp"),default="gemm")
  p.add_argument("--runs",type=int,default=5)
  p.add_argument("--build-jobs",type=int,default=16)
  p.add_argument("--input",type=Path,help="existing input bundle instead of generated GEMM")
  a = p.parse_args()
  if a.runs<3:
    p.error("use at least three measured samples")
  root = use_checkout(a.opentpu_root)
  revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
  if revision!=REVISION:
    raise ValueError(f"check out pinned FeSens revision {REVISION}")
  import numpy as np
  from opentpu import rtlsim
  # Both generated RTL and its host runtime use optimized C++, with bounded build parallelism.
  rtlsim.VFLAGS[rtlsim.VFLAGS.index("-j")+1] = str(a.build_jobs)
  rtlsim.VFLAGS += ["-CFLAGS","-O3"]
  out = a.output.resolve()
  out.mkdir(parents=True,exist_ok=True)
  inputs = a.input.resolve() if a.input else out/"input"
  report = {} if a.input else (mlp(inputs,a.m,a.hidden,a.intermediate,a.tmem_words)
      if a.workload=="mlp" else gemm(inputs,a.n,a.kb,a.jobs,a.m))
  before = time.perf_counter()
  workload = load(inputs)
  report["functional_frontend_seconds"] = time.perf_counter()-before
  if (inputs/"float_reference.npy").exists() and "output_dram_byte" in report:
    golden = workload.machine.slices[0].dram[report["output_dram_byte"]:][
        :4*report["checked_output_elements"]].view(np.float32)
    expected = np.load(inputs/"float_reference.npy").reshape(-1)
    relative = float(np.linalg.norm(golden.astype(np.float64)-expected)/np.linalg.norm(expected))
    report["float_reference_relative_error"] = relative
    if relative>0.02:
      raise RuntimeError("GEMM numerical error exceeds the quantization tolerance")
  timing = Timing(batch_rows=a.batch_rows,dram_bytes_per_cycle=min(128,workload.config.D))
  before = time.perf_counter()
  if a.compact:
    from .compact import CompactOpenTPU
    model = CompactOpenTPU(workload,timing)
  else:
    model = OpenTPU(workload,timing)
  report["matrix_batch_rows"] = model.matrix_batch_sizes
  ir,binary = out/"model.mlir",out/"model.native"
  ir.write_text(model.build())
  report["model_build_seconds"] = time.perf_counter()-before
  before = time.perf_counter()
  compiled = subprocess.run([str(a.lpn_compiler.resolve()),str(ir),"-o",str(binary)],
                            capture_output=True,text=True)
  (out/"compile.log").write_text(compiled.stdout+compiled.stderr)
  if compiled.returncode:
    raise RuntimeError(f"native compilation failed: {compiled.stderr}")
  report["native_compile_seconds"] = time.perf_counter()-before
  cycles,_,_,log = native_observation(binary,trace=True)
  (out/"lpn-observed.log").write_text(log)
  report["lpn_cycles"] = cycles
  print(f"LPN predicted {cycles} cycles; building/running RTL",flush=True)
  before = time.perf_counter()
  exe = rtlsim.build_top(workload.config,dram_lat=timing.dram_latency)
  report["rtl_build_seconds"] = time.perf_counter()-before
  drams,tmems,stats = rtlsim.run(workload.config,workload.programs,workload.images,
      dram_lat=timing.dram_latency,trace=True,keep=out/"rtl-observed",args=workload.args)
  (out/"rtl-observed.log").write_text(stats["trace"])
  if stats["instructions"]!=[len(c) for c in workload.commands]:
    raise RuntimeError("RTL and functional frontend instruction counts differ")
  for s,golden in enumerate(workload.machine.slices):
    if not np.array_equal(drams[s],golden.dram) or not np.array_equal(tmems[s],golden.tmem):
      raise RuntimeError(f"RTL and functional frontend differ in slice {s}")
    for name in (f"prog_{s}.hex",f"dram_{s}.bin"):
      if hashlib.sha256((out/"rtl-observed"/name).read_bytes()).hexdigest()!=workload.hashes[name]:
        raise RuntimeError(f"RTL did not consume identical {name}")
    from opentpu.waits import UNIT
    actual = [int(pc) for pc in re.findall(rf"T{s} D c=\d+ s=\d+ pc=(\d+)",stats["trace"])]
    expected = [c.pc for c in workload.commands[s] if c.instruction.op in UNIT]
    if actual!=expected:
      raise RuntimeError(f"RTL and frontend command paths differ in slice {s}")
  report.update(rtl_cycles=stats["cycles"],all_dram_and_tmem_bit_exact=True,
      identical_input_files=True,identical_command_paths=True,lpn_zero_rollbacks=True,
      dynamic_instructions=[len(c) for c in workload.commands],input_sha256=workload.hashes)
  report["relative_cycle_error"] = cycles/stats["cycles"]-1
  report["accuracy_within_5_percent"] = abs(report["relative_cycle_error"])<=0.05
  # The prepared RTL run directory contains the identical inputs and can be reused directly.
  roi,process = {"verilator":[],"lpn":[]},{"verilator":[],"lpn":[]}
  for run in range(a.runs+1):
    order = ("verilator","lpn") if run%2==0 else ("lpn","verilator")
    for engine in order:
      if engine=="lpn":
        checked,seconds,total,_ = native_observation(binary)
        if checked!=cycles:
          raise RuntimeError("native model time changed between runs")
      else:
        before = time.perf_counter()
        r = subprocess.run([str(exe),f"+dir={out/'rtl-observed'}","+lpn_wall"]+
            [f"+arg{k}={int(v)&0xffffffff}" for k,v in enumerate(workload.args)],
            check=True,capture_output=True,text=True,timeout=a.rtl_timeout)
        total = time.perf_counter()-before
        wall = re.search(r"LPN_WALL seconds=([0-9.]+) cycles=(\d+)",r.stdout)
        if not wall or int(wall[2])!=stats["cycles"] or "halted=1 error=0" not in r.stdout:
          raise RuntimeError("RTL timer or cycle count differs")
        seconds = float(wall[1])
      if run:
        roi[engine].append(seconds)
        process[engine].append(total)
    print(json.dumps({"run":run,"wall_samples":{k:v[-1:] for k,v in roi.items()}}),flush=True)
  report["wall_core_samples_seconds"],report["wall_process_samples_seconds"] = roi,process
  report["wall_core_median_seconds"] = {k:statistics.median(v) for k,v in roi.items()}
  report["wall_process_median_seconds"] = {k:statistics.median(v) for k,v in process.items()}
  core,whole = report["wall_core_median_seconds"],report["wall_process_median_seconds"]
  report["core_speedup"] = core["verilator"]/core["lpn"]
  report["cycles_per_second"] = {"verilator":stats["cycles"]/core["verilator"],"lpn":cycles/core["lpn"]}
  report["single_use_lpn_seconds"] = sum(report[k] for k in (
      "functional_frontend_seconds","model_build_seconds","native_compile_seconds"))+whole["lpn"]
  report["single_use_speedup"] = whole["verilator"]/report["single_use_lpn_seconds"]
  report.update(upstream_revision=revision,config=asdict(workload.config),timing=asdict(timing),
      verilator=rtlsim.verilator_version(),verilator_flags=rtlsim.VFLAGS,host=platform.node(),
      numpy_version=np.__version__,python_version=platform.python_version(),
      measurement_date_utc=time.strftime("%Y-%m-%d",time.gmtime()),
      baseline_binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
      method="identical hex instructions and big-endian DRAM files; independent ISA numerical frontend; LPN row/resource dataflow; no observed timestamps in model; untraced core wall timers; one warmup then alternating engine order; preparation reported separately")
  (out/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
  print(json.dumps(report,indent=2),flush=True)
  if not report["accuracy_within_5_percent"]:
    raise RuntimeError("LPN cycle error exceeds 5%; see comparison.json")


if __name__=="__main__":
  main()
