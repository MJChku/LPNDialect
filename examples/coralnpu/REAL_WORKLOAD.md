# Executable CoralNPU GEMM benchmark

This benchmark executes a real RISC-V/VME program in Verilator and feeds the
identical ELF to the LPN frontend. It computes **160 GEMMs of A[16,64] × B[64,16]**,
with BF16 operands and FP32 accumulators. Each GEMM zeros the accumulator,
loads its operands, issues 32 packed matrix operations, and stores all 16
output rows. Scalar branches control the inner and outer loops. The operands
vary across rows, columns, and K; repetitions reuse the same A and B in local
memory. This is a compute kernel benchmark, not full Llama inference.

The program issues 5,120 matrix commands and performs 2,621,440 MACs. A scalar
readback loop verifies the final 256 outputs after the timed region. Small
signed integer values encoded as BF16 keep the reference result exact; these
checks do not establish arbitrary IEEE floating-point equivalence.

## Run

Build LPN as described in the root README. In a fresh CoralNPU checkout at
`517f4c09bd5aabb4e8a16436d8853c912a24790b`, apply the observer patch once and
build an optimized baseline:

```sh
cd ../coralnpu
git apply ../LPNDialect/examples/coralnpu/verilator-observer.patch
bazel build -c opt --copt=-O3 \
  //tests/verilator_sim:vme_core_mini_axi_sim --jobs=12 --linkopt=-latomic
cd ../LPNDialect
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m examples.coralnpu.benchmark_workload \
  --coral-root ../coralnpu --clang clang-20 \
  --output /tmp/coral-real-100k --jobs 160 --runs 5 \
  --lpn-compiler build/tools/lpn-compile/lpn-compile
```

The output directory contains `gemm.S`, `gemm.elf`, observed RTL dispatch and
scalar writeback CSVs, generated MLIR, the native executable, wall samples,
and `comparison.json`. The command fails for a numerical/path mismatch,
incomplete simulation, unexpected rollback, or timing error exceeding 5%.
Use at least three measured samples; the default is five after one warmup.

## What is compared

`elf.py` loads the same ELF as Verilator and functionally executes its supported
RV32/VME instructions. It determines branches from real register values,
executes the matrix arithmetic and memory operations, and records effective
addresses. Unsupported encodings fail explicitly. The functional frontend
does not consume any RTL trace or measured timing.

`stream.py` compiles this instruction path into an LPN with reusable issue,
vector execution, retirement, and ROI observer transitions. Tokens represent
register readiness, memory aliases, execution units, accumulator reuse, ROB
credits, and queue capacity. Matrix latencies derive from the PE pipeline and
command geometry. The existing event-driven scheduler and guard-before-body
execution are retained. Immutable literal tables are cached in the native
runtime instead of copied on every firing.

An observation run compares every dynamic PC/instruction word between the
independent frontend and RTL, and checks all 256 final elements against an
independent dot-product reference. Subsequent speed runs disable dispatch,
writeback, and transition traces. Passive timers measure the same region:
first GEMM setup instruction dispatch through one cycle after a scalar end
marker that depends on the final output load. ELF loading, reset, and the
readback oracle are outside that region. Engine order alternates per run.

LPN's native timing run does not repeat numerical arithmetic: the functional
frontend already executed it to determine the path. Reports therefore expose
frontend time, graph construction, and native compilation separately, plus
their sum with the median complete native process time as `single_use_lpn_seconds`.
`single_use_speedup` compares that sum with the median complete RTL process
time. Common assembly-to-ELF compilation and building either simulator are
excluded from both sides.

## Recorded result

The [recorded report](validation/real_100k/comparison.json) and
[assembly input](validation/real_100k/gemm.S) were captured on `icdslab2` on
2026-10-15, using an AMD EPYC 7702P, Clang/LLVM 20, and the optimized Verilator
baseline above. Five measured samples follow one warmup:

| Metric | Verilator | Native LPN |
| --- | ---: | ---: |
| Timed region, modeled cycles | 103,372 | 100,158 |
| Median region wall time | 21.009 s | 1.005 s |
| Median complete process wall time | 21.386 s | 1.020 s |

The cycle prediction is 3.11% below RTL; the timing run is 20.90× faster.
LPN functional preprocessing takes 2.075 s, graph construction 1.451 s, and
native compilation 1.165 s. Adding these to its complete process time gives
5.711 s for a single use, or 3.74× faster than the complete RTL process.
All 77,770 dynamic instructions match the RTL path, all 256 checked outputs
match the reference, and the native run records zero rollbacks. The timed
region contains 76,484 instructions. Binary and observation hashes and the
individual samples are retained in the report.

## Limits

The shared ELF is an actual executable workload; the LPN model still predicts
timing for its precomputed path. It does not react to new runtime inputs after
compilation. The compact scoreboard currently requires local-memory accesses;
external DRAM traffic is a future extension. It abstracts instruction fetch,
vector scheduling/renaming, and other frontend details. Matching this kernel
within 5% establishes a result for this workload, not a whole-core or
whole-Llama accuracy guarantee.
