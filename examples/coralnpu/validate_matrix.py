"""Run full-core VME RTL kernels, check every accumulator value and PE timing."""

import argparse
import csv
import json
import os
import re
from pathlib import Path
import struct
import subprocess
import time

from .matrix import MatrixArray, MatrixCommand, MatrixSpec


def word(f6, vs2, vs1, f3, rd, opcode=0x57):
  return (f6 << 26) | (1 << 25) | (vs2 << 20) | (vs1 << 15) | (f3 << 12) | (rd << 7) | opcode


def configure(m, n, k, dtype):
  sew, widen, vtype = {"int8": (8, 3, 0xc0), "uint8": (8, 3, 0xc0),
                       "bf16": (16, 2, 0x1c9), "fp32": (32, 1, 0xd2)}[dtype]
  return [f"li a0, {(m << 10) | (k << 5) | widen}", f"li a1, {vtype}",
          ".insn r 0x57, 7, 0x41, x0, a0, a1", f"li a0, {n}",
          ".insn r 0x57, 7, 0x42, x0, a0, x0"]


def bits32(value):
  return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def operands(dtype):
  slots = {"int8": 4, "uint8": 4, "bf16": 2, "fp32": 1}[dtype]
  a = [[(i + k) % 7 - (3 if dtype == "int8" else 0) for i in range(16)] for k in range(slots)]
  b = [[1 + (j + 2 * k) % 5 for j in range(16)] for k in range(slots)]
  return a, b


def make_kernel(commands):
  dtype = commands[0].dtype
  if any(command.dtype != dtype for command in commands):
    raise ValueError("a validation kernel uses one operand format")
  groups = sorted({command.tile for command in commands})
  a, b = operands(dtype)
  code = [".option norvc", ".section .text", ".globl _start", "_start:",
          "li x5, 0x60006600", "csrs mstatus, x5"]
  code += configure(16, 16, 1, "fp32")
  for tile in groups:
    code += [f".word 0x{word(0x10, 30, 0, 6, tile << 1):08x}"]
  code += configure(16, 16, commands[0].k, dtype)
  sew = {"int8": 8, "uint8": 8, "bf16": 16, "fp32": 32}[dtype]
  stride = {8: 2, 16: 4, 32: 4}[sew]
  for name, start in (("operand_a", 8), ("operand_b", 16)):
    code += [f"la x11, {name}"]
    for row in range(len(a)):
      code += [f"vle{sew}.v v{start + row * stride}, (x11)"]
      if row + 1 < len(a):
        code += [f"addi x11, x11, {16 * sew // 8}"]
  # The PC list explicitly identifies matrix commands, separate from setup.
  previous_shape = (16, 16, commands[0].k)
  for idx, command in enumerate(commands):
    shape = (command.m, command.n, command.k)
    if shape != previous_shape:
      code += configure(command.m, command.n, command.k, dtype)
    previous_shape = shape
    code += [f"matrix_command_{idx}:"]
    f3 = 0 if dtype in ("int8", "uint8") else 1
    flag = int(dtype in ("int8", "bf16"))
    code += [f".word 0x{word(0x3c, 8, 16, f3, (command.tile << 1) | flag, 0x77):08x}"]
  # Read every row back through the architectural vector and memory paths.
  code += configure(16, 16, 1, "fp32")
  for group_idx, tile in enumerate(groups):
    code += [f"la x11, output_{tile}"]
    for row in range(16):
      code += [f"li a0, {(tile << 27) | row}",
               f".word 0x{word(0x10, 31, 10, 6, 4):08x}",
               "vse32.v v4, (x11)", "addi x11, x11, 64"]
  # Scalar x31 writes make every architectural result observable to the host.
  for tile in groups:
    code += [f"la x11, output_{tile}", "li x6, 256", f"check_tile_{tile}:",
             "lw x5, 0(x11)", "addi x31, x5, 0", "addi x11, x11, 4",
             "addi x6, x6, -1", f"bnez x6, check_tile_{tile}"]
  code += [".word 0x08000073", "1: j 1b", ".section .data", ".balign 16"]
  for name, values in (("operand_a", a), ("operand_b", b)):
    code += [f"{name}:"]
    if sew == 8:
      code += [".byte " + ",".join(str(item & 255) for row in values for item in row)]
    elif sew == 16:
      code += [".short " + ",".join(str(bits32(item) >> 16) for row in values for item in row)]
    else:
      code += [".word " + ",".join(str(bits32(item)) for row in values for item in row)]
  for tile in groups:
    code += [".balign 16", f"output_{tile}:", ".zero 1024"]
  expected = []
  for tile in groups:
    for i in range(16):
      for j in range(16):
        value = sum(sum(a[k][i] * b[k][j] for k in range(command.k))
                    for command in commands if command.tile == tile and i < command.m and j < command.n)
        expected.append(bits32(value) if dtype in ("bf16", "fp32") else value & 0xffffffff)
  return "\n".join(code) + "\n", expected


def cases():
  return {
      "uint8_full": [MatrixCommand(dtype="uint8")],
      "int8_full": [MatrixCommand(dtype="int8")],
      "bf16_full": [MatrixCommand(dtype="bf16", k=2)],
      "fp32_full": [MatrixCommand(dtype="fp32", k=1)],
      "int8_tail": [MatrixCommand(m=3, n=5, k=3)],
      "int8_quadrants": [MatrixCommand(m=9, n=11, k=2)],
      "int8_same_accumulator": [MatrixCommand() for _ in range(4)],
      "int8_independent_accumulators": [MatrixCommand(tile=tile) for tile in (0, 4, 8, 12)],
      "int8_tail_same_accumulator": [MatrixCommand(m=3, n=5, k=3) for _ in range(4)],
      "bf16_same_accumulator": [MatrixCommand(dtype="bf16", k=2) for _ in range(4)],
      "fp32_same_accumulator": [MatrixCommand(dtype="fp32", k=1) for _ in range(4)],
      "bf16_decode_long": [MatrixCommand(m=1,n=16,k=2,dtype="bf16") for _ in range(64)],
      "bf16_prefill_long": [MatrixCommand(dtype="bf16",k=2) for _ in range(64)],
  }


def load_events(path):
  with Path(path).open() as source:
    return list(csv.DictReader(source))


def compare_array(commands, events, simulate=True):
  launches = [event for event in events if event["event"] == "launch"]
  if len(launches) != len(commands):
    raise RuntimeError(f"RTL launched {len(launches)} of {len(commands)} commands")
  latencies = []
  spec = MatrixSpec()
  for command, launch in zip(commands, launches):
    done = [event for event in events if event["event"] == "write" and event["pc"] == launch["pc"]]
    active = {block for block, *_ in spec.blocks(command)}
    if {int(event["block"]) for event in done} != active:
      raise RuntimeError("RTL did not complete every active PE block")
    measured = max(int(event["cycle"]) for event in done) - int(launch["cycle"])
    if measured != spec.latency(command):
      raise RuntimeError(f"PE latency mismatch: RTL {measured}, LPN {spec.latency(command)}")
    latencies.append({"rtl_cycles": measured, "lpn_cycles": spec.latency(command)})
  if simulate:
    result = MatrixArray(commands).simulate()
  else:
    # Recorded long, identical streams were also run through the native LPN
    # at capture time. Check their max-plus recurrence without replaying the
    # much slower Python interpreter in every unit-test run.
    if not commands or any(command != commands[0] for command in commands):
      raise ValueError("the compact sequence check requires identical commands")
    command = commands[0]
    result = {"cycles": (len(commands)-1)*max(spec.issue_cycles(command),
               spec.reuse_cycles(command))+spec.latency(command),
              "macs": sum(c.macs for c in commands)}
  origin = int(launches[0]["cycle"])
  rtl_cycles = max(int(event["cycle"]) for event in events if event["event"] == "write") - origin
  if rtl_cycles != result["cycles"]:
    raise RuntimeError(f"array completion mismatch: RTL {rtl_cycles}, LPN {result['cycles']}")
  return {"commands": len(commands), "macs": result["macs"],
          "array_lpn_cycles": result["cycles"], "array_rtl_cycles": rtl_cycles,
          "command_latencies": latencies,
          "rtl_launch_intervals": [int(b["cycle"]) - int(a["cycle"])
                                   for a, b in zip(launches, launches[1:])]}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--coral-root", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--clang", default="clang-18")
  parser.add_argument("--case", choices=list(cases()))
  parser.add_argument("--lpn-compiler", type=Path)
  args = parser.parse_args()
  root, out = args.coral_root.resolve(), args.output.resolve()
  out.mkdir(parents=True, exist_ok=True)
  simulator = root / "bazel-bin/tests/verilator_sim/vme_core_mini_axi_sim"
  reports = []
  for name, commands in cases().items():
    if args.case and args.case != name:
      continue
    asm, expected = make_kernel(commands)
    source, elf = out / f"{name}.S", out / f"{name}.elf"
    source.write_text(asm)
    subprocess.run([args.clang, "--target=riscv32-unknown-elf", "-march=rv32imaf_zve32x_zicsr_zifencei",
                    "-mabi=ilp32", "-nostdlib", "-fuse-ld=lld", "-Wl,-Ttext=0,-Tdata=0x10000,-e,_start",
                    str(source), "-o", str(elf)], check=True, capture_output=True)
    writes, events = out / f"{name}.writes.csv", out / f"{name}.matrix.csv"
    before = time.perf_counter()
    rtl = subprocess.run([str(simulator), f"--binary={elf}", "--cycles=100000",
                          f"+lpn_matrix_trace={events}"],
                         env={**os.environ, "LPN_CORAL_WRITEBACK": str(writes),
                              "LPN_CORAL_TRACE": str(out / f"{name}.dispatch.csv")},
                         capture_output=True, text=True, timeout=120)
    seconds = time.perf_counter() - before
    (out / f"{name}.rtl.log").write_text(rtl.stdout + rtl.stderr)
    if rtl.returncode:
      raise RuntimeError(f"RTL kernel {name} failed; see {name}.rtl.log")
    observed = [int(row["value"], 0) for row in load_events(writes) if int(row["reg"]) == 31]
    if observed != expected:
      mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(observed, expected)) if a != b]
      raise RuntimeError(f"{name}: result count {len(observed)}/{len(expected)}, mismatches {mismatches[:8]}")
    report = compare_array(commands, load_events(events))
    report.update(kernel=name, checked_elements=len(expected), rtl_wall_seconds=seconds)
    model = MatrixArray(commands)
    ir = out / f"{name}.mlir"
    ir.write_text(model.build())
    if args.lpn_compiler:
      binary = out / f"{name}.native"
      subprocess.run([str(args.lpn_compiler.resolve()), str(ir), "-o", str(binary)], check=True)
      before = time.perf_counter()
      native = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
      report["native_wall_seconds"] = time.perf_counter() - before
      final = re.search(r"Simulation finished at t=([0-9.]+)", native.stdout)
      if not final or float(final.group(1)) != report["array_lpn_cycles"]:
        raise RuntimeError("native matrix timing disagrees with the Python model")
      (out / f"{name}.native.log").write_text(native.stdout)
    reports.append(report)
    print(json.dumps(report), flush=True)
  revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
  (out / "comparison.json").write_text(json.dumps(
      {"coral_revision": revision, "kernels": reports}, indent=2) + "\n")


if __name__ == "__main__":
  main()
