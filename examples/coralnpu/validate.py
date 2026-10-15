"""Build small RV32/RVV kernels and compare LPN timing with real Verilator RTL.

Run on a host with the CoralNPU simulator and clang/lld:
python -m examples.coralnpu.validate --coral-root ../coralnpu --output results
"""

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

from .model import CoralNPU
from .trace import load_dispatch_trace


START = 0x01100013  # addi x0,x0,17
END = 0x01228f93    # addi x31,x5,18: also waits for the final result


def kernels(repeats=32):
  return {
      "alu_chain": (["li x5, 1"], ["addi x5, x5, 1"] * repeats, []),
      "alu_independent": ([f"li x{reg}, 1" for reg in (5, 6, 7, 8)],
          [f"addi x{5 + idx % 4}, x{5 + idx % 4}, 1" for idx in range(repeats)],
          ["add x5, x5, x6", "add x5, x5, x7", "add x5, x5, x8"]),
      "mul_chain": (["li x5, 1", "li x6, 3"],
                    ["mul x5, x5, x6"] * repeats, []),
      "dtcm_loads": (["lui x10, 0x10", "li x5, 7", "sw x5, 0(x10)",
                      "lw x5, 0(x10)", "addi x0, x5, 0"],
                     ["lw x5, 0(x10)"] * repeats, []),
      "vector_add": (["li x5, 0x6600", "csrs mstatus, x5", "li x10, 4",
                      "vsetvli x10, x10, e32, m1, ta, ma", "vmv.v.i v1, 1",
                      "vmv.v.i v2, 2", "lui x11, 0x10", "vse32.v v1, (x11)",
                      "lw x5, 0(x11)", "addi x0, x5, 0"],
                     ["vadd.vv v1, v1, v2"] * repeats,
                     ["vse32.v v1, (x11)", "lw x5, 0(x11)"]),
  }


def assembly(init, body, tail):
  lines = [".option norvc", ".section .text", ".globl _start", "_start:"]
  lines += init + [".word 0x01100013"] + body + tail
  lines += [".word 0x01228f93", ".word 0x08000073", "1: j 1b"]
  return "\n".join(lines) + "\n"


def roi_trace(full, destination):
  with full.open() as source:
    rows = list(csv.DictReader(source))
  starts = [i for i, row in enumerate(rows) if int(row["inst"], 0) == START]
  ends = [i for i, row in enumerate(rows) if int(row["inst"], 0) == END]
  if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
    raise RuntimeError("kernel did not dispatch both ROI markers exactly once")
  roi = rows[starts[0] + 1:ends[0] + 1]
  with destination.open("w") as target:
    writer = csv.DictWriter(target, fieldnames=("cycle", "lane", "pc", "inst"),
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(roi)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--coral-root", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--clang", default="clang-18")
  parser.add_argument("--lpn-compiler", type=Path)
  parser.add_argument("--repeats", type=int, default=32)
  args = parser.parse_args()
  root, out = args.coral_root.resolve(), args.output.resolve()
  out.mkdir(parents=True, exist_ok=True)
  sim = root / "bazel-bin/tests/verilator_sim/rvv_core_mini_axi_sim"
  revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                     text=True).strip()
  reports = []
  for name, (init, body, tail) in kernels(args.repeats).items():
    src, elf = out / f"{name}.S", out / f"{name}.elf"
    full, trace = out / f"{name}.full.csv", out / f"{name}.csv"
    writes = out / f"{name}.writes.csv"
    src.write_text(assembly(init, body, tail))
    subprocess.run([args.clang, "--target=riscv32-unknown-elf", "-march=rv32imaf_zve32x_zicsr_zifencei",
                    "-mabi=ilp32", "-nostdlib", "-fuse-ld=lld", "-Wl,-Ttext=0,-e,_start",
                    str(src), "-o", str(elf)], check=True, capture_output=True)
    before = time.perf_counter()
    result = subprocess.run([str(sim), f"--binary={elf}", "--cycles=100000"],
                            env={**os.environ, "LPN_CORAL_TRACE": str(full),
                                 "LPN_CORAL_WRITEBACK": str(writes)},
                            text=True, capture_output=True, timeout=120)
    rtl_seconds = time.perf_counter() - before
    (out / f"{name}.rtl.log").write_text(result.stdout + result.stderr)
    if result.returncode:
      raise RuntimeError(f"Verilator failed for {name}; see {name}.rtl.log")
    expected = {"alu_chain": 1 + args.repeats,
                "alu_independent": 4 + args.repeats,
                "mul_chain": pow(3, args.repeats, 1 << 32),
                "dtcm_loads": 7, "vector_add": 1 + 2 * args.repeats}[name]
    with writes.open() as source:
      values = [int(row["value"], 0) for row in csv.DictReader(source)
                if int(row["reg"]) == 31]
    if not values or values[-1] != ((expected + 18) & 0xffffffff):
      raise RuntimeError(f"incorrect RTL kernel result for {name}: {values}")
    roi_trace(full, trace)
    instructions, baseline = load_dispatch_trace(trace)
    model = CoralNPU(instructions)
    ir = out / f"{name}.mlir"
    ir.write_text(model.build())
    before = time.perf_counter()
    prediction = model.simulate()
    python_seconds = time.perf_counter() - before
    report = {"kernel": name, "instructions": len(instructions),
              "verilator_dispatch_cycles": baseline["dispatch_span"],
              "lpn_completion_cycles": prediction["cycles"],
              "rtl_wall_seconds": rtl_seconds, "python_wall_seconds": python_seconds}
    report["rtl_result_checked"] = True
    # The final ROI instruction consumes the previous result. Its dispatch
    # span includes its single execution cycle. This provides a consistent
    # comparison window without including program loading or reset cycles.
    report["relative_cycle_error"] = (prediction["cycles"] / baseline["dispatch_span"] - 1)
    if args.lpn_compiler:
      binary = out / f"{name}.native"
      subprocess.run([str(args.lpn_compiler.resolve()), str(ir), "-o", str(binary)],
                     check=True, capture_output=True)
      before = time.perf_counter()
      native = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
      report["native_wall_seconds"] = time.perf_counter() - before
      match = re.search(r"Simulation finished at t=([0-9.]+)", native.stdout)
      if not match or float(match.group(1)) != prediction["cycles"]:
        raise RuntimeError("native and Python timing disagree")
    reports.append(report)
    print(json.dumps(report), flush=True)
  (out / "comparison.json").write_text(json.dumps(
      {"coral_revision": revision, "kernels": reports}, indent=2) + "\n")


if __name__ == "__main__":
  main()
