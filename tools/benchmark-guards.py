#!/usr/bin/env python3
"""Compare native guard-first and transactional execution with identical LLVM -O3.

Builds a temporary comparison compiler/runtime; never changes the built tools.
Run from the repository root after building: PYTHONPATH=. python3
tools/benchmark-guards.py --output /tmp/guard-benchmark.json
"""

import argparse
import json
import platform
from pathlib import Path
import shlex
import statistics
import subprocess
import tempfile
import time

from examples.coralnpu.matrix import GemmModel
from examples.network_net import create_network_net
from lpnlang_mlir import NetBuilder


def blocked_workload(manual=False):
  net = NetBuilder("many_disabled")
  available = net.place("available", initial_tokens=1)
  empty = net.place("empty")
  clock = net.place("clock", initial_tokens=1)
  done = net.place("done")
  for idx in range(256):
    if manual:
      @net.transition(f"blocked_{idx}")
      def blocked():
        guard(count(available) > 0)
        guard(count(empty) > 0)
        token = take(available)
        take(empty)
        emit(available, token)
    else:
      @net.transition(f"blocked_{idx}")
      def blocked():
        token = take(available)
        take(empty)
        emit(available, token)

  @net.transition
  def tick():
    token = take(clock)
    steps = token.get("steps") + 1
    token = token.set("steps", steps)
    if steps < 256:
      emit(clock, token, delay=1.0)
    else:
      emit(done, token)

  return net


def flags(path):
  result = []
  for line in path.read_text().splitlines():
    if line.startswith(("CXX_DEFINES =", "CXX_INCLUDES =", "CXX_FLAGS =")):
      result += shlex.split(line.split("=", 1)[1])
  return result


def comparison_compiler(root, temp):
  build = root / "build/tools/lpn-compile"
  source = (root / "tools/lpn-compile/lpn-compile.cpp").read_text()
  assert source.count("skipManualGuardChecks = true;") == 1
  source = source.replace("skipManualGuardChecks = true;", "skipManualGuardChecks = false;")
  (temp / "lpn-compile.cpp").write_text(source)
  runtime = (root / "tools/lpn-compile/NativeRuntime.cpp").read_text()
  start = runtime.index("  ++guardChecks;", runtime.index("bool tryFire("))
  end = runtime.index("  ++bodyExecutions;", start)
  runtime = runtime[:start] + runtime[end:]
  (temp / "NativeRuntime.cpp").write_text(runtime)
  obj, archive = temp / "runtime.o", temp / "runtime.a"
  subprocess.run(["c++", *flags(build / "CMakeFiles/LPNNativeRuntime.dir/flags.make"),
                  "-c", str(temp / "NativeRuntime.cpp"), "-o", str(obj)], check=True)
  subprocess.run(["ar", "rcs", str(archive), str(obj)], check=True)
  options = [arg for arg in flags(build / "CMakeFiles/lpn-compile.dir/flags.make")
             if not arg.startswith("-DLPN_NATIVE_RUNTIME_PATH=")]
  options += [f'-DLPN_NATIVE_RUNTIME_PATH="{archive}"',
              f'-I{root / "tools/lpn-compile"}']
  compiler_obj = temp / "compiler.o"
  subprocess.run(["c++", *options, "-c", str(temp / "lpn-compile.cpp"),
                  "-o", str(compiler_obj)], check=True)
  link = shlex.split((build / "CMakeFiles/lpn-compile.dir/link.txt").read_text())
  binary = temp / "transactional-compiler"
  link[link.index("-o") + 1] = str(binary)
  link = [str(compiler_obj) if arg.endswith("lpn-compile.cpp.o") else
          "-Wl,-rpath,/usr/lib/llvm-20/lib" if arg.startswith("-Wl,-rpath") else arg
          for arg in link]
  subprocess.run(link, cwd=build, check=True)
  return binary


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--runs", type=int, default=9)
  args = parser.parse_args()
  if args.runs < 3:
    parser.error("use at least three runs")
  root = Path(__file__).resolve().parents[1]
  reports = []
  with tempfile.TemporaryDirectory(prefix="lpn-guard-benchmark-") as directory:
    temp = Path(directory)
    old = comparison_compiler(root, temp)
    new = root / "build/tools/lpn-compile/lpn-compile"
    cases = {"inferred_many_disabled": blocked_workload(),
             "manual_many_disabled": blocked_workload(manual=True),
             "all_to_all_32x4": create_network_net(32, 4),
             "gemm_16x16x256": GemmModel(16, 16, 256).net}
    for name, net in cases.items():
      source = temp / f"{name}.mlir"
      source.write_text(net.build())
      binaries = {}
      for label, compiler in (("transactional", old), ("guard_first", new)):
        binary = temp / f"{name}-{label}"
        subprocess.run([str(compiler), str(source), "-o", str(binary)],
                       check=True, capture_output=True, text=True)
        binaries[label] = binary
      samples = {label: [] for label in binaries}
      outputs, stats = {}, {}
      for iteration in range(args.runs + 2):
        # Alternate execution order to reduce temperature/load order bias.
        order = list(binaries) if iteration % 2 == 0 else list(reversed(binaries))
        for label in order:
          before = time.perf_counter()
          result = subprocess.run([str(binaries[label]), "--max-time=1000000000", "--stats"],
                                  check=True, capture_output=True, text=True, timeout=60)
          seconds = time.perf_counter() - before
          lines = result.stdout.splitlines()
          outputs[label] = [line for line in lines if not line.startswith("Guard checks:")]
          stats[label] = next(line for line in lines if line.startswith("Guard checks:"))
          if iteration >= 2:
            samples[label].append(seconds)
        assert outputs["transactional"] == outputs["guard_first"], name
      medians = {label: statistics.median(times) for label, times in samples.items()}
      report = {"workload": name, "median_seconds": medians, "samples_seconds": samples,
                "speedup": medians["transactional"] / medians["guard_first"],
                "stats": stats, "final_marking_and_time_equal": True}
      reports.append(report)
      print(json.dumps(report), flush=True)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps({"host": platform.node(), "runs": args.runs,
      "method": "same LLVM -O3; transactional variant skips enabling functions and retains body guards; process startup and output included; two warmups",
      "workloads": reports}, indent=2) + "\n")


if __name__ == "__main__":
  main()
