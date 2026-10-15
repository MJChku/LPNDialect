# FeSens/openTPU LPN model

This is the active accelerator modeling target. The upstream is
[FeSens/openTPU](https://github.com/FeSens/openTPU), pinned at
`f237dab2f196855a59a083b1454786df8f01291f`. It is distinct from the older
UCSBarchlab/OpenTPU project. On `icdslab2`, the checkouts are `~/FeSens-openTPU`
and `~/LPNDialect`.

The model reads the **same `prog_<slice>.hex` instructions and `dram_<slice>.bin`
operand files as the RTL testbench**. A JSON manifest adds configuration,
instruction counts, and initial argument registers. DRAM files contain the
big-endian words expected by Verilog `$fread`; the memory they initialize is
little-endian. The independent upstream ISA simulator executes these inputs
to determine control flow and effective addresses and to check numerical
results. No observed RTL timestamps enter the LPN model.

## Setup and run

Build LPN following the root README. Install a recent Verilator; the recorded
runs use **5.052**. Ubuntu's Verilator 5.020 fails to elaborate this pinned
upstream revision. Use a Python environment with NumPy:

```sh
git clone https://github.com/FeSens/openTPU.git ../FeSens-openTPU
git -C ../FeSens-openTPU checkout f237dab2f196855a59a083b1454786df8f01291f
git -C ../FeSens-openTPU apply ../LPNDialect/examples/opentpu/verilator-observer.patch
python3 -m venv ../FeSens-openTPU/.venv
../FeSens-openTPU/.venv/bin/pip install numpy

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ../FeSens-openTPU/.venv/bin/python \
  -m examples.opentpu.benchmark --opentpu-root ../FeSens-openTPU \
  --output /tmp/opentpu-gemm --batch-rows 8 --runs 5 \
  --lpn-compiler build/tools/lpn-compile/lpn-compile

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ../FeSens-openTPU/.venv/bin/python \
  -m examples.opentpu.benchmark --opentpu-root ../FeSens-openTPU \
  --workload mlp --output /tmp/opentpu-mlp --batch-rows 8 --runs 5 \
  --lpn-compiler build/tools/lpn-compile/lpn-compile
```

The observer patch adds two optional host clock readings around the RTL core
run (`+lpn_wall`) and links their DPI helper. It changes no DUT logic. The
benchmark explicitly requests host C++ `-O3`, bounds build parallelism, and
records the complete Verilator flags and executable hash. It checks every
DRAM byte and TMEM word against the ISA simulator, checks dynamic command PCs,
and verifies the actual RTL input-file hashes match LPN's. A timing error over
5%, differing numerical results/inputs/paths, or incomplete simulation fails
the command. Native enabling functions execute before bodies; successful
benchmark runs require zero rollbacks.

Replay a generated or externally prepared input bundle directly:

```sh
PYTHONPATH=. ../FeSens-openTPU/.venv/bin/python -m examples.opentpu.run \
  --opentpu-root ../FeSens-openTPU --input /tmp/opentpu-gemm/input \
  --emit /tmp/opentpu.mlir --batch-rows 8 \
  --lpn-compiler build/tools/lpn-compile/lpn-compile --output /tmp/opentpu
/tmp/opentpu --stats --wall-start=begin --wall-end=complete
```

Use `benchmark --input DIRECTORY` to compare an existing input bundle against
RTL. `write_inputs()` in `benchmark.py` exports programs and memory images
from upstream `compile_kernel()`; the MLP workload demonstrates this route.

## Workloads and measurements

The default GEMM program performs seven **A[4,4096] × W[512,4096]^T** matrix
products, with INT8 weights, quantized activations, and FP32 results. It loads
actual FP32 inputs, runs QACT, uses a hardware loop around the matrix command,
and stores 2,048 output values. The workload performs 58,720,256 MACs. Input
and weight values vary across every dimension; iterations reuse the same
operands. It is a real executed kernel, not a full language model.

The MLP workload uses the **upstream compiler and SwiGLU MLP kernel**: four
rows, hidden and intermediate widths 1,024, pre-RMSNorm, gate/up projections,
SiLU, a down projection, and a residual addition. Its 50 dynamic instructions
include matrix operations, vector arithmetic/composites/reductions,
quantization, loads, and stores. All 4,096 outputs are checked; the INT8 result
also meets a 2% relative error tolerance against independent floating-point
math. That numerical tolerance is separate from the 5% timing target.

Recorded results on `icdslab2`, 2026-10-15, with LLVM/MLIR 20 and an optimized
Verilator 5.052 baseline (`--batch-rows 8`, bounded as described below):

| Workload | RTL cycles | LPN cycles | Timing error | RTL core wall | LPN core wall | LPN cycles/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Seven 4×4096×512 GEMMs](validation/gemm_100k/comparison.json) | 119,723 | 119,674 | −0.041% | 5.458 s | 21.44 ms | 5.58 M |
| [Compiled SwiGLU MLP](validation/mlp/comparison.json) | 27,673 | 27,629 | −0.159% | 1.319 s | 24.71 ms | 1.12 M |
| [Held-out 8×2048×128 GEMMs, three repeats](validation/replay_m8/comparison.json) | 16,939 | 16,894 | −0.266% | 0.781 s | 2.04 ms | 8.29 M |

Core speedups are **254.5×** for the large GEMM and **53.4×** for the MLP.
Including numerical preprocessing, graph construction, compilation, and the
complete native process, single-use LPN costs are **0.404 s** and **0.609 s**,
versus complete RTL process medians of **5.665 s** and **1.529 s**: **14.0×**
and **2.51×** respectively. The large GEMM uses two-row batches; the MLP uses
four or eight. Its predictions match the one-row model for these two cases.
The third case exercises ACT-row replay with eight rows on four columns and
was checked after the initial model parameters were fixed.

Every DRAM byte and TMEM word matches bit for bit in all three cases; this
includes 7,168 checked output elements in total. Independent floating-point
relative errors are 0.795%, 1.131%, and 0.774%, respectively, reflecting INT8
quantization. Each report contains all wall samples, configuration, software
versions, input hashes, and baseline executable hash. The encoded programs
and input manifests are saved beside it; operand bytes are reproducible from
the generator and seed (47), using the recorded NumPy version (2.5.4).

Timers measure core execution from reset release through halt. They exclude
RTL image initialization and memory dumps; the complete process times are
also recorded. Native timing excludes its prior numerical execution and
compilation, which are reported separately and included in
`single_use_lpn_seconds`. One warmup precedes alternating engine order and
five measured runs. Input generation, upstream kernel compilation, reference
calculations, and building the RTL simulator are setup costs shared by the
comparison, outside these runtime totals.

## Dataflow and current scope

`model.py` composes DRAM ports and a systolic array inside per-slice control.
The upstream footprint rules supply RAW/WAR/WAW dependencies over TMEM, ACT
RAM, and DRAM. Tokens provide synchronization, dispatch-window credits,
in-order DMA/quantizer execution, and the two-entry matrix command queue.
Matrix prefetch may start before ACT operands become ready. A bounded FIFO
feeds row groups to the array, whose initiation interval is separate from its
chain/skew, epilogue, and accumulation latency. The final row group publishes
command completion; unused intermediate output events are omitted.

`--batch-rows 1` keeps one event per streamed row. Higher values group rows,
bounded by the shape and FIFO space for two packets. Small commands stay at
one row. This reduces LPN events without adding a clock-by-clock loop or
changing the existing scheduler. The benchmark records each command's actual
batch size. Grouping is an abstraction to validate on additional workloads,
not a blanket guarantee of identical intermediate timing.

Validated configuration: one slice, depth 128, four matrix columns, two
epilogue lanes, eight TMEM/VPU lanes, replicated TMEM, a 32-entry range-based
sequencer window, 128-chunk prefetch FIFO, and behavioral DRAM delivering up
to 128 bytes/cycle with eight-cycle read latency. This is an upstream RTL test
configuration, not a claim of equivalence to the production FPGA's memory
controller, TMEM implementation, or WAITC sequencer. `Timing` exposes latency
and bandwidth for the simple DRAM abstraction; alternate values require
validation against a matching baseline.

The frontend handles arithmetic, hardware loops, register-relative addresses,
and data-derived RLD values through the numerical ISA execution. The timing
backend covers LD/ST, MM, QACT/QST, VOP, scalar control, RLD, and single-slice
BAR/GATHER. It rejects WAITW host interactions, HALT CHAIN, DSTEP/STREAM,
WAITC, and multi-slice collective timing. Instruction fetch, detailed TMEM
arbitration, VPU instruction overlap, and all protocol/controller details are
approximations. Whole Llama accuracy and real Llama 3.2 3B inference have not
been established.

## Integration checks

Run the optional tests with NumPy and the pinned upstream checkout:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ../FeSens-openTPU/.venv/bin/python \
  test/test_opentpu.py --opentpu-root ../FeSens-openTPU \
  --tool build/tools/lpn-compile/lpn-compile
```

To include them in CTest, configure with
`-DLPN_OPENTPU_ROOT=/absolute/path/to/FeSens-openTPU` and
`-DPython3_EXECUTABLE=/absolute/path/to/FeSens-openTPU/.venv/bin/python`.
The tests compare Python/native model completion, exercise hardware loops and
register values, check row grouping and pipelined accumulator reuse, and
reject unsupported synchronization and truncated inputs.
