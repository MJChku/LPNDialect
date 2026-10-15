"""Run the same executable BF16 GEMM workload in VME RTL and native LPN."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import re
import statistics
from pathlib import Path
import subprocess
import time

from .elf import execute_elf
from .stream import InstructionStream
from .validate_matrix import bits32, configure, word


def gemm_source(k=64, jobs=64):
  if k < 2 or k%2 or jobs < 1:
    raise ValueError("K must be positive/even and jobs positive")
  if k*64+1024 > 32768:
    raise ValueError("the benchmark operands and output must fit in 32 KB DTCM")
  code = [".option norvc", ".section .text", ".globl _start", "_start:",
          "li x5, 0x60006600", "csrs mstatus, x5",f"li x20, {jobs}",
          ".globl roi_start", "roi_start:", "gemm_job:"]
  code += configure(16,16,1,"fp32")
  code += [f".word 0x{word(0x10,30,0,6,0):08x}"]
  code += configure(16,16,2,"bf16")
  code += ["la x11, operand_a", "la x12, operand_b",f"li x6, {k//2}","gemm_inner:"]
  for reg,address in ((8,11),(12,11),(16,12),(20,12)):
    code += [f"vle16.v v{reg}, (x{address})",f"addi x{address}, x{address}, 32"]
  code += [f".word 0x{word(0x3c,8,16,1,1,0x77):08x}",
           "addi x6, x6, -1", "bnez x6, gemm_inner"]
  code += configure(16,16,1,"fp32")
  code += ["la x11, output", "li a0, 0", "li x6, 16", "store_row:",
           f".word 0x{word(0x10,31,10,6,4):08x}", "vse32.v v4, (x11)",
           "addi x11, x11, 64", "addi a0, a0, 1", "addi x6, x6, -1",
           "bnez x6, store_row", "addi x20, x20, -1", "bnez x20, gemm_job",
           "la x11, output", "lw x5, 1020(x11)", ".globl roi_end", "roi_end:",
           ".word 0x01228f93", "li x6, 256", "check_output:", "lw x5, 0(x11)",
           "addi x31, x5, 0", "addi x11, x11, 4", "addi x6, x6, -1",
           "bnez x6, check_output", ".word 0x08000073", "1: j 1b",
           ".section .data", ".balign 16"]
  a = [[(inner+row)%7-3 for row in range(16)] for inner in range(k)]
  b = [[(inner*3+col)%5-2 for col in range(16)] for inner in range(k)]
  for name,values in (("operand_a",a),("operand_b",b)):
    code += [name+":", ".short "+",".join(str(bits32(x)>>16) for row in values for x in row)]
  code += [".balign 16","output:",".zero 1024"]
  expected = [bits32(sum(a[inner][row]*b[inner][col] for inner in range(k)))
              for row in range(16) for col in range(16)]
  return "\n".join(code)+"\n",expected


def build_elf(source, elf, clang):
  subprocess.run([clang,"--target=riscv32-unknown-elf",
      "-march=rv32imaf_zve32x_zicsr_zifencei","-mabi=ilp32","-nostdlib",
      "-fuse-ld=lld","-Wl,-Ttext=0,-Tdata=0x10000,-e,_start",str(source),"-o",str(elf)],
      check=True,capture_output=True,text=True)


def read_csv(path):
  with Path(path).open() as source:
    return list(csv.DictReader(source))


def observe(root, elf, out, expected, workload, max_cycles=1000000):
  trace,writes = out/"dispatch.csv",out/"writes.csv"
  simulator = root/"bazel-bin/tests/verilator_sim/vme_core_mini_axi_sim"
  before = time.perf_counter()
  result = subprocess.run([str(simulator),f"--binary={elf}",f"--cycles={max_cycles}","--backdoor_load"],
      env={**os.environ,"LPN_CORAL_TRACE":str(trace),"LPN_CORAL_WRITEBACK":str(writes),
           "LPN_CORAL_WALL_FILE":str(out/"rtl-observed-wall.csv"),
           "LPN_CORAL_WALL_START_PC":hex(workload.image.symbols["roi_start"]),
           "LPN_CORAL_WALL_END_PC":hex(workload.image.symbols["roi_end"])},
      capture_output=True,text=True,timeout=300)
  seconds = time.perf_counter()-before
  (out/"rtl.log").write_text(result.stdout+result.stderr)
  if result.returncode:
    raise RuntimeError("RTL workload failed; see rtl.log")
  observed = [int(row["value"],0) for row in read_csv(writes) if int(row["reg"])==31]
  if observed[-256:] != expected:
    raise RuntimeError("RTL GEMM output differs from independent reference")
  rows = read_csv(trace)
  actual = [(int(row["pc"],0),int(row["inst"],0)) for row in rows]
  decoded = [(instruction.pc,instruction.bits) for instruction in workload.instructions]
  if actual != decoded:
    mismatch = next((i for i,(a,b) in enumerate(zip(actual,decoded)) if a!=b),None)
    raise RuntimeError(f"ELF frontend and RTL instruction paths differ: {len(actual)}/{len(decoded)}, first={mismatch}")
  first,last = workload.roi[0],workload.roi[1]-1
  cycles = int(rows[last]["cycle"])-int(rows[first]["cycle"])+1
  return {"roi_rtl_cycles":cycles,"rtl_observed_wall_seconds":seconds,
          "elf_sha256":workload.image.sha256,"dynamic_instructions":len(decoded),
          "roi_instructions":last-first+1,"matrix_commands":workload.matrices,
          "macs":workload.macs,"checked_output_elements":256,
          "frontend_and_rtl_instruction_paths_identical":True}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--coral-root",type=Path,required=True)
  parser.add_argument("--output",type=Path,required=True)
  parser.add_argument("--clang",default="clang-20")
  parser.add_argument("--k",type=int,default=64)
  parser.add_argument("--jobs",type=int,default=160)
  parser.add_argument("--runs",type=int,default=5)
  parser.add_argument("--lpn-compiler",type=Path,required=True)
  parser.add_argument("--baseline-build",default="Bazel -c opt --copt=-O3")
  args = parser.parse_args()
  if args.runs<3:
    parser.error("use at least three measured runs")
  out = args.output.resolve()
  out.mkdir(parents=True,exist_ok=True)
  source,elf = out/"gemm.S",out/"gemm.elf"
  asm,expected = gemm_source(args.k,args.jobs)
  source.write_text(asm)
  build_elf(source,elf,args.clang)
  before = time.perf_counter()
  workload = execute_elf(elf)
  frontend_seconds = time.perf_counter()-before
  if workload.scalar_outputs[-256:] != expected:
    raise RuntimeError("ELF frontend GEMM output differs from independent reference")
  report = observe(args.coral_root.resolve(),elf,out,expected,workload)
  report.update(k=args.k,jobs=args.jobs,frontend_seconds=frontend_seconds)
  before = time.perf_counter()
  model = InstructionStream(workload)
  ir,binary = out/"model.mlir",out/"model.native"
  ir.write_text(model.build())
  report["model_build_seconds"] = time.perf_counter()-before
  before = time.perf_counter()
  subprocess.run([str(args.lpn_compiler.resolve()),str(ir),"-o",str(binary)],
                 check=True,capture_output=True,text=True)
  report["native_compile_seconds"] = time.perf_counter()-before
  checked = subprocess.run([str(binary),"--stats","--trace"],
                           check=True,capture_output=True,text=True)
  (out/"lpn-observed.log").write_text(checked.stdout)
  start = re.search(r"\[t=([0-9.]+)\] fired observe_roi_start",checked.stdout)
  end = re.search(r"\[t=([0-9.]+)\] fired observe_roi_end",checked.stdout)
  if not start or not end or "place program_completed: 1 token(s)" not in checked.stdout:
    raise RuntimeError("native LPN workload did not complete")
  if "rollbacks: 0" not in checked.stdout:
    raise RuntimeError("native LPN workload unexpectedly rolled back")
  report["roi_lpn_cycles"] = float(end.group(1))-float(start.group(1))
  report["relative_cycle_error"] = report["roi_lpn_cycles"]/report["roi_rtl_cycles"]-1
  report["accuracy_within_5_percent"] = abs(report["relative_cycle_error"])<=0.05
  report["benchmark_complete"] = False
  (out/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
  samples = {"verilator":[],"lpn":[]}
  process = {"verilator":[],"lpn":[]}
  simulator = args.coral_root.resolve()/"bazel-bin/tests/verilator_sim/vme_core_mini_axi_sim"
  for run in range(args.runs+1):
    order = ("verilator","lpn") if run%2==0 else ("lpn","verilator")
    for engine in order:
      before = time.perf_counter()
      if engine=="verilator":
        timing = out/f"rtl-wall-{run}.csv"
        env = {name:value for name,value in os.environ.items() if not name.startswith("LPN_CORAL_")}
        env.update(LPN_CORAL_WALL_FILE=str(timing),
            LPN_CORAL_WALL_START_PC=hex(workload.image.symbols["roi_start"]),
            LPN_CORAL_WALL_END_PC=hex(workload.image.symbols["roi_end"]))
        result = subprocess.run([str(simulator),f"--binary={elf}","--cycles=1000000","--backdoor_load"],
                               env=env,capture_output=True,text=True,check=True,timeout=300)
        elapsed = time.perf_counter()-before
        row = read_csv(timing)
        if len(row)!=1 or int(row[0]["cycles"])!=report["roi_rtl_cycles"]:
          raise RuntimeError("Verilator wall timer did not cover the same cycle window")
        seconds = float(row[0]["wall_seconds"])
      else:
        result = subprocess.run([str(binary),"--stats","--wall-start=observe_roi_start",
            "--wall-end=observe_roi_end"],capture_output=True,text=True,check=True,timeout=300)
        elapsed = time.perf_counter()-before
        wall = re.search(r"Wall window elapsed: ([0-9.]+) seconds",result.stdout)
        if not wall or "rollbacks: 0" not in result.stdout:
          raise RuntimeError("LPN wall timer or guard verification failed")
        seconds = float(wall.group(1))
      if run>0:
        samples[engine].append(seconds)
        process[engine].append(elapsed)
    print(json.dumps({"run":run,"roi_cycles":report["roi_rtl_cycles"],
                      "wall_samples":{k:v[-1:] for k,v in samples.items()}}),flush=True)
    (out/"progress.json").write_text(json.dumps({"run":run,"wall_roi_samples_seconds":samples,
        "wall_process_samples_seconds":process},indent=2)+"\n")
  report["wall_roi_samples_seconds"] = samples
  report["wall_process_samples_seconds"] = process
  report["wall_roi_median_seconds"] = {k:statistics.median(v) for k,v in samples.items()}
  report["wall_process_median_seconds"] = {k:statistics.median(v) for k,v in process.items()}
  report["roi_speedup"] = report["wall_roi_median_seconds"]["verilator"]/report["wall_roi_median_seconds"]["lpn"]
  report["single_use_lpn_seconds"] = (report["frontend_seconds"]+report["model_build_seconds"]+
      report["native_compile_seconds"]+report["wall_process_median_seconds"]["lpn"])
  report["single_use_speedup"] = report["wall_process_median_seconds"]["verilator"]/report["single_use_lpn_seconds"]
  report["method"] = "same ELF and complete dynamic instruction path; timestamp-free LPN input; real BF16 operands; ROI wall timers; traces disabled in timed runs; one warmup; alternating engine order; compilation and functional frontend excluded from runtime"
  report["coral_revision"] = subprocess.check_output(["git","rev-parse","HEAD"],cwd=args.coral_root,text=True).strip()
  report["lpn_zero_rollbacks"] = True
  report["benchmark_complete"] = True
  report["host"] = platform.node()
  report["baseline_build"] = args.baseline_build
  report["baseline_binary_sha256"] = hashlib.sha256(simulator.read_bytes()).hexdigest()
  report["lpn_native_binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
  report["measurement_date_utc"] = datetime.now(timezone.utc).date().isoformat()
  report["clang_version"] = subprocess.check_output([args.clang,"--version"],text=True).splitlines()[0]
  report["host_kernel"] = platform.platform()
  report["model_input"] = "the identical RV32/VME ELF binary used by Verilator"
  report["dispatch_sha256"] = hashlib.sha256((out/"dispatch.csv").read_bytes()).hexdigest()
  report["output_reference_sha256"] = hashlib.sha256(
      b"".join(value.to_bytes(4,"little") for value in expected)).hexdigest()
  (out/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
  print(json.dumps(report,indent=2),flush=True)
  if not report["accuracy_within_5_percent"]:
    raise RuntimeError("GEMM timing error exceeds the 5% target; see comparison.json")


if __name__ == "__main__":
  main()
