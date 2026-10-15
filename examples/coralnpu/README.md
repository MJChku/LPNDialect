# CoralNPU LPN timing model

This model targets the RV32 scalar/RVV `RvvCoreMiniAxi` and matrix-enabled
`VmeCoreMiniAxi` configurations of
Google's [CoralNPU RTL](https://github.com/google-coral/coralnpu). The baseline
revision is `517f4c09bd5aabb4e8a16436d8853c912a24790b`. Units are classes composed
inside `CoralNPU`, using the existing LPN scheduling and native compiler.

`model.py` models four-wide in-order dispatch, register RAW/WAW dependencies,
shared execution ports with separate latency and initiation interval,
finite outstanding instructions, in-order retirement, branch redirect delay,
and local/external memory ports. Time is expressed in cycles, although the
generic native executable's output label remains `ns`. No cycle-by-cycle loop
is added to the LPN simulator.

The input is a dynamic instruction stream. `trace.py` decodes that stream from
Verilator dispatch observations and discards measured timestamps when building
the model. Consequently this predicts timing for a fixed execution path. It
does not execute ISA arithmetic or independently predict control-flow changes.

## Run the RTL baseline

Apply `verilator-observer.patch` to the pinned CoralNPU checkout. This exposes
existing debug ports without enabling the full verification ROB, and samples
dispatch and scalar writeback. It does not change execution-unit behavior.

```sh
cd ../coralnpu
git apply ../LPNDialect/examples/coralnpu/verilator-observer.patch
bazel build //tests/verilator_sim:rvv_core_mini_axi_sim --jobs=12 --linkopt=-latomic
cd ../LPNDialect
PYTHONPATH=. python3 -m examples.coralnpu.validate \
  --coral-root ../coralnpu --clang clang-20 \
  --output /tmp/coralnpu-validation \
  --lpn-compiler build/tools/lpn-compile/lpn-compile
```

The host needs CoralNPU's Bazel version, Clang/LLD, and `liblz4-dev`. The
validation script builds assembly kernels with no compressed instructions,
runs the actual Verilator core, checks a final scalar result (including the
vector-add result stored to memory), and records timing comparisons. The ROI
ends with a scalar instruction dependent on the final result; reset, ELF
loading, and initialization are excluded. Reports distinguish measured
dispatch span from predicted completion time. Errors are reported rather than
hidden by replaying baseline delays. Wall times include process startup and
trace output; they are not steady-state simulator-throughput measurements.

Recorded observations are in `validation/rep8` and `validation/rep32`, captured
on 2026-10-15. The final 32-repeat run on `icdslab2` produced:

| Kernel | Verilator window (cycles) | LPN (cycles) |
| --- | ---: | ---: |
| Dependent scalar adds | 33 | 33 |
| Independent scalar adds | 12 | 12 |
| Dependent multiplies | 65 | 65 |
| DTCM loads | 129 | 129 |
| RVV adds plus store/readback | 140 | 140 |

The separate 8-repeat runs also match for all five kernels. The native LPN
executables ran in approximately 5–7 ms, compared with 110–203 ms for this
Verilator build including initialization and observation output. The baseline
used Bazel's default fastbuild configuration and debug ports. These are
microkernel checks, not an accuracy or speed guarantee for arbitrary programs.
Model latency parameters were refined against the 32-repeat kernels; the
8-repeat cases check those parameters at another workload size.

## Compile a recorded workload

```sh
PYTHONPATH=. python3 -m examples.coralnpu.run \
  --trace /tmp/coralnpu-validation/alu_chain.csv --emit /tmp/coral.mlir
build/tools/lpn-compile/lpn-compile /tmp/coral.mlir -o /tmp/coral
/tmp/coral
```

## Accuracy and extension limits

This is an initial performance abstraction, not a complete cycle-exact RTL
replacement. Fetch buffering, lane pairing restrictions, operand-dependent
division, vector reservation stations/renaming, vector length and LMUL
expansion, and precise trap behavior are not fully modeled. The RVV trace
adapter tracks vector/matrix configuration when its operands can be recovered
from scalar constants. The vector baseline kernel uses LMUL=1. Matrix arithmetic,
configuration, tile moves, zeroing, and aligned tile memory transfers are now
decoded; unknown runtime configuration and unsupported vector memory modes
raise errors rather than choosing a default matrix shape.
Vector arithmetic latency parameters require comparison on additional kernels.
Command handoff and decoding contribute three cycles before vector execution;
the command/uop queues are represented by a combined credit pool. Per-cycle
dispatch and retirement patterns can differ even when total kernel time agrees.

Memory requests currently use conservative program ordering because the
dispatch trace contains no effective memory addresses. `MemoryPort` provides
an independent bandwidth token and outstanding-request credits. `MemorySpec`
computes service time as `ceil(bytes / bytes_per_cycle)` and completion delay
as `latency + service_time`. Requests marked `Instruction(external=True)` use
the external endpoint. This is the extension point for the requested simple
DRAM latency/bandwidth model. Real AXI address mapping and overlapping
independent requests still need address-bearing workload records.

Each issue transition has explicit resource guards. Dependency tokens represent
ready register results, while execution-port tokens are released at the
initiation interval. Retire transitions release ROB credits in program order.
The instruction-stream model has two transitions per scalar instruction and
three per vector instruction; large traces
will increase compile time and transition scanning cost. Compile time is
separate from execution time, and the scheduler algorithm is unchanged.

## Matrix PE array and GEMM

`matrix.py` adds an explicit `MatrixArray` and a composable `GemmModel` around it.
The pinned RTL has four staggered PE blocks. Each block processes two rows by
eight columns per cycle; packed integer products reduce four INT8 multiplies
into each accumulator, BF16 reduces two, and FP32 uses one. The logical
16x16 32-bit accumulator spans four 256-byte tile registers. Groups begin at
mt0, mt4, mt8, or mt12. The model therefore uses those aligned groups.

```mermaid
flowchart LR
    M[Memory port] --> B[Operand buffer credits]
    B --> P[Four PE blocks<br/>stagger: 0 / 1 / 1 / 2 cycles]
    P --> X[Multiplier pipeline: 3 cycles]
    X --> F[Product register: 1 cycle]
    F --> A[Accumulator pipeline: 3 cycles]
    A --> T[Tile storage]
    T --> M
```

The LPN uses one token per row group instead of per MAC. Multiply and add
ports have one-cycle initiation intervals. An accumulator can accept another
full-tile operation after four cycles, while results from the previous
operation continue through the pipeline. Shorter M tails still respect the
four-cycle multiplier-tag hazard. A full command takes 12 cycles; partial
3x5 and 9x11 commands take 8 and 11 cycles. These timings come from the RTL
pipeline and active-block geometry, not a fixed GEMM delay.

`GemmModel` decomposes M/N/K into matrix commands, prefetches operands through
a finite number of buffers, models accumulator ownership and zeroing, and
stores each output tile before reusing its physical accumulator. Its memory
endpoint uses the same latency/bandwidth abstraction as the existing core
model. For example:

```sh
PYTHONPATH=. python3 -m examples.coralnpu.run \
  --gemm 16 16 64 --dtype int8 --emit /tmp/gemm.mlir
build/tools/lpn-compile/lpn-compile /tmp/gemm.mlir -o /tmp/gemm
/tmp/gemm
```

`--memory-latency` and `--memory-bandwidth` set cycles and bytes/cycle for the
GEMM memory endpoint. Defaults represent the local-memory abstraction. The
GEMM model predicts performance; numerical ISA/FP execution is delegated to
the RTL validation baseline. DMA/AXI protocol details are not fully modeled.

### Full-core matrix validation

The observer patch now includes a VME simulator target, bindings for its extra
debug ports, and passive PE ingress/write observations. It also pads mini-ROB
instruction constants before mset bit extraction, fixing an upstream VME
debug elaboration error. Apply the patch once to a fresh pinned checkout:

```sh
cd ../coralnpu
git apply ../LPNDialect/examples/coralnpu/verilator-observer.patch
bazel build //tests/verilator_sim:vme_core_mini_axi_sim --jobs=12 --linkopt=-latomic
cd ../LPNDialect
PYTHONPATH=. python3 -m examples.coralnpu.validate_matrix \
  --coral-root ../coralnpu --clang clang-20 \
  --output /tmp/coralnpu-matrix-validation \
  --lpn-compiler build/tools/lpn-compile/lpn-compile
```

The validation program runs actual matrix instructions on the full VME RTL.
It initializes vector operands, zeros accumulators, performs matrix operations,
reads every tile row back through vector stores, and checks every scalar
readback against a separate reference calculation. PE observations compare
array launch-to-write timing, excluding setup/readback overhead.

Recorded observations in `validation/matrix` cover 11 kernels and 3,584
checked elements: UINT8, signed INT8, BF16, FP32, masked tails, multiple active
quadrants, repeated accumulator use, and independent accumulators. Single
full-tile operations take 12 cycles; four full-tile accumulations take 24;
four 3x5 tail accumulations take 20. All measured array completion times match
the LPN. Partial-tile validation uses actual write enables; the RTL's per-block
last-uop retire pulses do not identify every partial block's final write.

This validates the PE-array abstraction and architectural matrix results for
these kernels. It does not establish cycle equivalence of the entire frontend,
tile controller, exceptions, arbitrary FP values, or memory backpressure.
The complete instruction-stream model can decode these workloads, while the
array and GEMM APIs avoid replaying scalar setup and verification instructions.
