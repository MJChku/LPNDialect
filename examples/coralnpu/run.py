"""python -m examples.coralnpu.run --trace dispatch.csv [--emit model.mlir]"""

import argparse
import json
from pathlib import Path

from .model import CoralNPU, MemorySpec
from .matrix import GemmModel
from .trace import load_dispatch_trace


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  workload = parser.add_mutually_exclusive_group(required=True)
  workload.add_argument("--trace", type=Path)
  workload.add_argument("--gemm", nargs=3, type=int, metavar=("M", "N", "K"))
  parser.add_argument("--dtype", choices=("int8", "uint8", "bf16", "fp32"), default="int8")
  parser.add_argument("--memory-latency", type=int, default=3)
  parser.add_argument("--memory-bandwidth", type=float, default=16,
                      help="Bytes per cycle for the GEMM memory endpoint")
  parser.add_argument("--emit", type=Path)
  args = parser.parse_args()
  baseline = None
  if args.gemm:
    model = GemmModel(*args.gemm, dtype=args.dtype,
                      memory=MemorySpec(args.memory_latency, args.memory_bandwidth))
  else:
    instructions, baseline = load_dispatch_trace(args.trace)
    model = CoralNPU(instructions)
  if args.emit:
    args.emit.write_text(model.build())
  else:
    result = model.simulate()
    if baseline is not None:
      result["verilator"] = baseline
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
