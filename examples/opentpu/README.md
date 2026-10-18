# FeSens/openTPU LPN model

The implemented parallel RTL completes SmolLM2-135M with 128 prompt tokens and
32 generated tokens in 15,435,996 cycles, versus 132,828,263 for original RTL
(**8.605x at a common clock**), with identical inputs and bit-exact full device
memory. Corrected LPN total error is +0.622% / +1.793% for original/evolved RTL.
See [RTL implementation, reproduction and audited evidence](rtl_evolution/README.md).
FPGA resource fit, Fmax and a vendor HBM controller are not established.
The buffered-quantizer follow-up completes the same full inference in
15,335,479 RTL cycles (**8.662x**), with bit-exact memory and LPN total error
-0.649% (worst kernel 2.958%). See its
[report and fresh audit](TPU_EXPLORATION.md#implemented-quantizer-change).
The current [FPGA-first architecture exploration](TPU_EXPLORATION.md) retains
this verified baseline and reports new predictions separately from fresh RTL
validation. Its 100x cycle-reduction target has not yet been achieved.
The next [shared-bank and interleaved-reduction candidate](FPGA_REDUCTIONS.md)
predicts 13.067x and has passed numerical component gates; full inference RTL
validation is still required.
The separate [arithmetic partition experiment](LOWLAT_FP.md) checks shorter
FP32 pipelines bit-exactly and predicts 15.108x for a two-clock-adder candidate.
Its operating clock and full-inference RTL timing remain unvalidated.

**Inference workload correction:** earlier `e2e.py`, `replay_e2e.py` and
weight-free FPGA sequence results used a custom token-by-token prompt loop.
Their checked numerical/cycle comparisons remain measurements of those streams,
but their speedups do not represent openTPU's normal batched-prefill inference
path. Inference scheduling belongs to the upstream runner, not the LPN model.
Use `standard_inference.py` below to observe `Engine.generate()` and feed its
emitted commands to the backends.

See [ARCHITECTURE.md](ARCHITECTURE.md) for component boundaries, explicit
dependencies, matrix-engine experiments and net-size/CPU-parallelism details.
See [NATIVE_INTERFACE.md](NATIVE_INTERFACE.md) for runtime input tables,
compile-once timing, and the C++ functional/timing ABI.
See [FPGA_DSE.md](FPGA_DSE.md) for weight-free Qwen3 architecture sweeps that
skip numerical computation and report hardware-cycle reductions, memory
bandwidth and resource assumptions.

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

### Normal upstream inference entrypoint

`standard_inference.py` calls upstream `Engine.generate()` without replacing
its prefill, fit/chunk selection or decode methods. The backend receives the
resulting programs. `plan` mode replaces weight/image materialization and
numerical readbacks with explicit placeholders; it does not predict text or EOS.
The upstream scheduler and compiler remain unchanged. The existing configuration
has eight-row input capacity, and the upstream fit logic may choose fewer rows
per kernel to fit its instruction and tensor memories.

```sh
PYTHONPATH=. ../FeSens-openTPU/.venv/bin/python -m examples.opentpu.standard_inference \
  --opentpu-root ../FeSens-openTPU --mode plan \
  --prompt-tokens 16 --output-tokens 8 --cap 128 --output /tmp/qwen-standard-plan
```

To check actual numerical inference on identical ISA/LPN/Verilator inputs:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  ../FeSens-openTPU/.venv/bin/python -m examples.opentpu.standard_inference \
  --opentpu-root ../FeSens-openTPU-qwen --mode validate-rtl \
  --model-dir ../FeSens-openTPU/models/Qwen3-1.7B \
  --compiler build/tools/lpn-compile/lpn-compile \
  --prompt 'Once upon a time' --output-tokens 8 --cap 128 \
  --output /tmp/qwen-standard-validation
```

The numerical path uses real checkpoint values, the same upstream
`Engine.generate()` and shared `CheckedBackend` for ISA/LPN/RTL comparisons.
It retains the real KV state across kernels and verifies complete device
memories. It runs Verilator and needs the pinned checkout, model dependencies
and RTL patches documented below. Each completed kernel is saved under
`kernels/<index>/record.json`; `progress.json` identifies the active process,
completed-kernel count and failures. This adapter does not resume a partially
completed generation. Planning can optionally
run LPN on each upstream-emitted kernel with `--compiler`. Kernel program words,
hashes, selected row counts and logit-readback counts are recorded separately
from hardware timing.

For the available SmolLM2-135M checkpoint, use the same command with
`--model-dir ../FeSens-openTPU/models/SmolLM2-135M --depth 64`. Its 64-element
attention heads require the 64-wide hardware mapping. Qwen3-1.7B uses the
default `--depth 128`. Both use upstream batching and preserve real KV state.
Completed reports include generated IDs/text, total and per-kernel cycle errors,
native/RTL core time and measured LPN pipeline stages. Pipeline stage totals
exclude input serialization; the joint validation wall time includes both engines.

At this pin, upstream `generate()` also calls `step()` after selecting the last
requested output, unless EOS/capacity stops it. The adapter preserves that final
decode kernel in both backends rather than silently changing the runner.

The [numerical fixture](validation/standard_inference/fixture16.json) checks a
small synthetic transformer using two eight-token prefill chunks and two decode
steps. ISA and RTL device memories are bit-exact on every kernel; RTL totals
29,595 cycles and LPN 29,218 cycles, with a worst per-kernel error of 2.64%.
This establishes that small shared path, not full Qwen3-1.7B RTL accuracy.
The [regression log](validation/standard_inference/regression.txt) records the
20 accelerator checks, including recorded RTL fixtures. Run the numerical check
with the Verilator environment described below:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  ../FeSens-openTPU/.venv/bin/python test/test_standard_inference.py \
  --opentpu-root ../FeSens-openTPU-qwen \
  --compiler build/tools/lpn-compile/lpn-compile --rtl \
  --artifacts /tmp/opentpu-standard-fixture
```

A real [SmolLM2-135M checkpoint run](validation/standard_inference/smollm2_16in8out/report.json)
now checks all 30 layers with **16 prompt tokens and eight generated tokens**.
Unmodified upstream inference selected two eight-token prefill kernels, reading
prompt logits only after the second, followed by eight decode kernels (including
the upstream final step described above). Every DRAM byte and TMEM word matched
the independent ISA execution on every kernel; both engines used identical
programs and numerical inputs, with zero LPN rollbacks.

The real [Qwen3-1.7B checkpoint run](validation/standard_inference/qwen17_8in2out/report.json)
also completed: all 28 layers, **eight prompt tokens and two generated tokens**,
with two four-token prefill kernels and two upstream decode kernels. It passed
the same complete device-memory, input-identity and zero-rollback checks.

| Case | RTL cycles | LPN cycles | Total cycle error | Worst kernel error | RTL core time | Native LPN core time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-135M, 16 input / 8 output | 25,455,239 | 25,896,361 | +1.73% | 2.79% | 984.25 s | 12.30 s |
| Qwen3-1.7B, 8 input / 2 output | 53,251,060 | 53,651,773 | +0.75% | 1.08% | 3,013.09 s | 13.24 s |

SmolLM2 achieved **80.02x native timing-core speedup** in this single run. The measured
functional/preparation/compilation/native-process stages total **54.36 s**;
that total excludes input serialization and is not an entire-process wall
measurement. Joint validation wall time, including RTL and its build, was
1,245.99 s. The generated continuation was ` wise old man named Mr. Oak.`

Qwen achieved **227.50x native timing-core speedup**. Its measured LPN pipeline
stages total **94.88 s**, excluding input serialization; joint validation wall
time was 3,614.66 s. Its continuation was `, there`. Its raw output, terminal
job manifest and [independent archive audit](validation/standard_inference/qwen17_8in2out/audit.json)
are saved beside its report. These are single-run observations of the stated
batched checkpoint cases, not accuracy claims for arbitrary context lengths.

SmolLM2's raw output, terminal job manifest and an
[independent archive audit](validation/standard_inference/smollm2_16in8out/audit.json)
are retained beside the report.

The audit verifies checkpoint/source/program hashes and all reported checks,
then independently replays the retained final kernel's ISA memory results and
native LPN timing. Earlier kernels' numerical comparisons remain the runner's
in-run checks. Its matching source revision is `c82de17`; later harness changes
fix progress row logging and narrow planning bandwidth without changing the
hardware timing model. To audit the original server artifacts, use a frozen
worktree at that revision:

```sh
git worktree add --detach /tmp/lpn-checkpoint-source c82de17
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  ../FeSens-openTPU/.venv/bin/python \
  examples/opentpu/validation/standard_inference/audit_checkpoint.py \
  --case "$HOME/lpn-runs/standard-checkpoint-validation/smollm2_16in8out" \
  --project-root /tmp/lpn-checkpoint-source --opentpu-root ../FeSens-openTPU-qwen \
  --output /tmp/smollm2-standard-audit.json
```

The [1,024-input/8-output Qwen3-1.7B planning report](validation/standard_inference/qwen17_plan1024.json)
records upstream selecting 256 four-token prefill kernels, one prompt-logit
readback, and eight decode kernels. Its 207.23 seconds are host planning and
compilation time, with no numerical tensor execution, LPN timing simulation or
Verilator run. The logical memories, instruction fit and chunk size are choices
of the configured upstream runner/compiler. The report includes source hashes
and every emitted program hash; it establishes the schedule rather than an
architecture speedup.

The [16-input/8-output weight-free LPN run](validation/standard_inference/qwen17_plan16_lpn.json)
executes the full Qwen3-1.7B shapes on the original architecture: four four-token
prefill kernels and eight upstream decode kernels. It predicts 159,799,171
cycles with zero rollbacks. Host time was 117.33 seconds, including 38.31 seconds
of native timing and 74.42 seconds of model/table preparation. This is a
single-run observation; there is no paired Verilator measurement for this
full-size batched stream yet. Add `--compiler` to the planning command above
to reproduce this mode.

The legacy token-by-token artifacts below are retained for reproducibility.

### Compiler and RTL setup

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
| [Eight 8×4096×2048 GEMMs](validation/gemm_million/comparison.json) | 1,059,504 | 1,059,454 | −0.00472% | 49.974 s | 71.66 ms | 14.78 M |
| [MLP, H=3072 F=8192 M=4](validation/mlp_3072_8192/comparison.json) | 610,507 | 598,253 | −2.007% | 30.053 s | 374.26 ms | 1.60 M |

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
five measured runs (three for the million-cycle GEMM and larger MLP). Input generation, upstream kernel compilation, reference
calculations, and building the RTL simulator are setup costs shared by the
comparison, outside these runtime totals.

The million-cycle GEMM performs 536,870,912 MACs and uses row replay on the
four-column array. Its core speedup is **697.4×**, with **2.770 s** for numerical
preprocessing, model construction, compilation and one native process (**18.3×**
versus the complete RTL process). The larger MLP uses the hidden/intermediate
dimensions of Llama 3.2 3B, with generated operands: it is a kernel test, not
Llama inference. It performs 301,989,888 MACs, uses 128 MiB DRAM and 131,072 TMEM
words, and gives **80.3×** core / **15.6×** single-use speedup. Both new cases
check every DRAM byte and TMEM word. Reproduce them with:

```sh
python -m examples.opentpu.benchmark --opentpu-root ../FeSens-openTPU \
  --lpn-compiler build/tools/lpn-compile/lpn-compile --output /tmp/opentpu-million \
  --n 2048 --kb 32 --jobs 8 --m 8 --batch-rows 8 --runs 3
python -m examples.opentpu.benchmark --opentpu-root ../FeSens-openTPU \
  --lpn-compiler build/tools/lpn-compile/lpn-compile --output /tmp/opentpu-large-mlp \
  --workload mlp --hidden 3072 --intermediate 8192 --tmem-words 131072 \
  --batch-rows 8 --runs 3
```

## Long trained-model runs

The completed SmolLM2 story validation contains 72 prompt tokens and 64
generated tokens (136 separate RTL invocations with persistent KV state):
298,017,840 RTL cycles and 11,179.11 s of RTL core time. Every DRAM/TMEM result
matched independently executed ISA results; maximum per-token timing error was
3.28%. The [complete heartbeat replay](validation/guard_heartbeats/smollm2_136.json)
checks all original input hashes, logits and LPN predictions. Its native timing
was 55.11 s; functional/preparation/compilation/native process time was 354.72 s,
and complete replay wall time was 405.34 s. The RTL observations were retained,
not rerun for the heartbeat replay.

Qwen3-1.7B bring-up uses the official `Qwen/Qwen3-1.7B` checkpoint at
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, compiled to the upstream INT8
execution path. The [first-token reports](validation/qwen17/first_tokens.json)
cover four concurrent cases, each executing all 28 layers and the vocabulary
head: 54,499,684 aggregate RTL cycles, 0.53% timing error, and bit-exact full
DRAM/TMEM results. Measured aggregate RTL core time was 2954.02 s, native LPN
timing 7.31 s (403.86x), and full LPN pipeline 38.65 s (76.43x versus RTL core).
These historical first-token observations use RTL command-path tracing. The
completed generation campaign below uses D=128, a 2 GiB memory, the same
behavioral memory latency/ports as the earlier runs, and a larger command-table
capacity of 32768. Real Llama 3.2 remains unvalidated.

Ten legacy token-by-token Qwen generation cases completed: eight generated eight tokens each, and
two generated twelve each. Each case executes every transformer layer and the
vocabulary head with persistent KV state. Prompt tokens and generated tokens
each require one full forward step; the RTL runner launches a separate
Verilator process per step. All baseline steps passed bit-exact full DRAM/TMEM
validation. Every complete LPN-only regeneration matched all program/input
hashes, logits, generated tokens and predicted cycle counts.

The table compares summed whole Verilator process times with whole LPN replay
process times. LPN time includes functional computation, preparation, compilation,
serialization, hashing, initialization and process cleanup. Baselines ran with
up to six concurrent workers; LPN replays ran one at a time. Earlier replays
overlapped baselines, and the last replay ran after all baselines finished.

| Completed case | Input / output tokens | Whole RTL processes | Native LPN timing | Whole LPN process | Whole-process speedup | Worst cycle error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Once](validation/qwen17/completed_once.json) | 1 / 8 | 2.06 h | 16.67 s | 140.74 s | 52.66x | 0.541% |
| [France](validation/qwen17/completed_france.json) | 5 / 8 | 2.96 h | 23.97 s | 183.97 s | 57.99x | 0.557% |
| [Sky](validation/qwen17/completed_sky.json) | 5 / 8 | 2.96 h | 23.97 s | 183.95 s | 57.96x | 0.557% |
| [Robot story](validation/qwen17/completed_robot.json) | 7 / 8 | 3.44 h | 27.74 s | 211.38 s | 58.57x | 0.565% |
| [Mountain village](validation/qwen17/completed_mountain.json) | 5 / 8 | 2.96 h | 24.08 s | 182.42 s | 58.36x | 0.557% |
| [Translation prompt](validation/qwen17/completed_translation.json) | 7 / 8 | 3.46 h | 27.68 s | 213.33 s | 58.35x | 0.565% |
| [Arithmetic prompt](validation/qwen17/completed_arithmetic.json) | 5 / 8 | 2.99 h | 24.01 s | 182.87 s | 58.82x | 0.557% |
| [Why](validation/qwen17/completed_why12.json) | 1 / 12 | 2.98 h | 23.87 s | 184.21 s | 58.28x | 0.557% |
| [Hello](validation/qwen17/completed_hello12.json) | 1 / 12 | 3.01 h | 23.94 s | 181.84 s | 59.56x | 0.557% |
| [Fibonacci](validation/qwen17/completed_fibonacci.json) | 18 / 8 | 5.94 h | 48.11 s | 372.41 s | 57.46x | 0.610% |

The [completed batch observations](validation/qwen17/overnight_progress.json)
contain 143 passing token comparisons and 88 generated tokens. The
[full replay summary](validation/qwen17/completed_replays_summary.json)
covers 1,948,459,995 RTL cycles. Aggregate speedup is
**57.89x** comparing whole processes, or **51.81x** comparing whole
LPN processes with RTL core time alone. Native timing alone is **399.67x** faster.
Worst per-step cycle error is **0.610%**. Native timing throughput is
7.38 million RTL-equivalent cycles/s; whole LPN process throughput is
0.956 million/s. These rates use summed per-case processing times, not
concurrent batch wall time. Accuracy applies to this fixed behavioral-memory
RTL configuration; it does not establish physical DDR-controller accuracy.

The [raw observations](validation/qwen17/overnight_records.csv) preserve every
paired measurement. The [completion audit](validation/qwen17/completion_audit.json)
verifies all ten durable snapshot hashes and final logits, complete generation
replays, raw CSV records, and checkpoint/model/compiler/runtime/RTL fingerprints.
The [audit program](validation/qwen17/audit_completed.py) rejects unfinished
campaigns and can recheck the archived run on icdslab2. It checks source hashes
strictly; use the frozen validation revision, before later replay changes:

```sh
git worktree add --detach /tmp/lpn-qwen-audit cea0c77
~/FeSens-openTPU/.venv/bin/python examples/opentpu/validation/qwen17/audit_completed.py \
  --root ~/lpn-runs/qwen3-1.7b-parallel --project-root /tmp/lpn-qwen-audit \
  --opentpu-root ~/FeSens-openTPU-qwen --expected-cases 10 \
  --output /tmp/qwen-completion-audit.json
```

The [measured overhead breakdown](validation/qwen17/functional_overhead.json)
uses the 18-input/8-output Fibonacci case: 372.41 s end to end, with 183.48 s
in the functional frontend, 89.74 s serializing/hashing inputs, 36.65 s preparing
the timing model, and 48.11 s in native LPN timing. The frontend timer includes
ISA execution and timing-metadata extraction; it does not isolate pure GEMM.
Eliminating that entire stage with other costs unchanged would leave 188.92 s,
a 1.97x end-to-end upper bound for that change. This is an arithmetic bound,
not a measured optimization; numerical results remain necessary for generation.
At similar context lengths, 16 input plus 8 output tokens should take roughly
5m45s with this configuration; that is an estimate, not an additional RTL case.

The [Qwen guard-cache comparison](validation/qwen17/guard_cache_ablation.json)
uses one copied timing input from an independently RTL-validated decode step.
Three alternating untraced runs per mode measured median native time of
5.388 s with `--no-guard-cache` and 1.880 s with heartbeat caching (2.87x).
Guard evaluations dropped from 16,737,495 to 1,352,959. Separate traced runs
matched all 526,080 firings and the final token marking exactly. This isolates
native timing; it is not an end-to-end speedup. The accompanying
[measurement script](validation/qwen17/guard_cache_ablation.py) accepts an
immutable native binary and its input tables. `replay_e2e.py --mode` only labels
a replay report; it does not select guard-cache behavior.

For Qwen's memory size, apply [verilator-large-memory.patch](verilator-large-memory.patch)
in an isolated pinned FeSens worktree, after the observer patch. It splits the
simulation memory's storage array into rows to avoid Verilator's range limit;
it adds no architectural bank conflicts or DRAM timing. The
[storage equivalence check](validation/qwen17/memory_equivalence.json) exercises
all ports, byte writes, boundary crossings, partial input loading and read
response cycles against the original memory and an independent byte result.
Reproduce it with:

```sh
PYTHONPATH=. ../FeSens-openTPU/.venv/bin/python -m examples.opentpu.check_memory_storage \
  --reference-root ../FeSens-openTPU --candidate-root ../FeSens-openTPU-qwen \
  --verilator "$VERILATOR_ROOT/bin/verilator" --verilator-root "$VERILATOR_ROOT" \
  --output /tmp/lpn-memory-storage-check
```

On icdslab2, set `VERILATOR_ROOT=$HOME/.local/opt/lpn-verilator-5.052` and put
its `bin` directory first on `PATH`. The system Verilator 5.020 is insufficient.
The completed campaign is under `~/lpn-runs/qwen3-1.7b-parallel`: ten independent
generation cases, with a limit of six concurrent processes. The original eight
request up to eight new tokens each; two added cases request twelve each.
`batch.json` records PIDs, commands, progress and failures. Large active
files/snapshots live under `/dev/shm/lpn-qwen3-17-validation`; reports and logs
are mirrored to disk. The separate `overnight_monitor.py` process copies and
hash-verifies each completed token's latest snapshot into `recovery/<case>`
using content-addressed files and an atomically selected manifest. It publishes
`overnight-summary.json` and `overnight-records.csv` with paired RTL/LPN data.
The supervisor archives a final snapshot when each case ends; the monitor then
removes its duplicate backup. RAM snapshots do not survive a reboot, so only
verified disk recovery snapshots are durable. Each case freezes RTL/observer
source hashes as well as the upstream revision in its resume identity.

After twenty passing token comparisons, `supervisor_v2.py` adopted the four
live workers and admitted two queued cases under measured memory headroom.
The [handoff audit](validation/qwen17/control_handoff.json) records unchanged
worker identities and unchanged RTL PIDs. No model/RTL workers were restarted.
The new scheduler preserves PID start times to distinguish exited processes
from reused PIDs, checks complete numerical/timing reports before accepting an
adopted worker's result, and hard-links an existing verified recovery snapshot
into the final archive when available. `batch.json` records the current
parallelism and its change history. The monitor and replay driver follow that
manifest; do not assume the original supervisor PID is still current.

After four completed generations freed two slots, `supervisor_v3.py` adopted
the remaining four workers and added the `Hello` and `Why` cases, each with a
one-token prompt and twelve requested output tokens. The
[queue extension audit](validation/qwen17/queue_extension.json) verifies that
all existing worker identities and running RTL PIDs were preserved. The
parallel limit remains six; the collector and replay driver follow the ten-case
manifest. Model, compiler, runtime and RTL sources remain frozen.

`replay_e2e.py` measures an LPN-only regeneration of a recorded RTL-validated
case, preserving real numerical execution and greedy decoding. It checks
checkpoint fingerprints, every input image/program hash, logits, instruction
counts and predicted cycles. Timing-table capacity comes from the recorded
case, so larger Qwen streams do not inherit SmolLM2's smaller limit. Completed
generation reports are required by default; `--allow-partial` explicitly opts
into a frozen unfinished snapshot and marks that scope in the result.

```sh
PYTHONPATH=. ../FeSens-openTPU/.venv/bin/python -m examples.opentpu.replay_e2e \
  --baseline /path/to/completed/report.json --opentpu-root ../FeSens-openTPU-qwen \
  --compiler build/tools/lpn-compile/lpn-compile --output /tmp/qwen-lpn-replay
```

For timing experiments, add **`--timing-only`**. This skips writing the full RTL
DRAM images/program files and skips their per-step SHA-256 checks. Functional
ISA execution, persistent KV state, greedy generation, instruction counts,
logit comparisons and cycle comparisons remain enabled. The report records
`input_file_validation: "skipped"` and `all_inputs_identical: null`; skipped
checks are not reported as passes. Small timing input tables are still written;
checkpoint verification and compiler/cache fingerprinting remain enabled.
Omit the option to retain full input-file validation.

To use the checked-in Qwen example, extract its baseline and provide your local
copy of the official checkpoint at the pinned revision above. Install the
Torch/safetensors/tokenizer dependencies listed below and build LPN first:

```sh
python3 - <<'PY'
import json
from pathlib import Path
case = json.loads(Path('examples/opentpu/validation/qwen17/completed_fibonacci.json').read_text())
Path('/tmp/qwen-baseline.json').write_text(json.dumps(case['baseline']))
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  ../FeSens-openTPU/.venv/bin/python -m examples.opentpu.replay_e2e \
  --baseline /tmp/qwen-baseline.json --opentpu-root ../FeSens-openTPU \
  --model-dir ../FeSens-openTPU/models/Qwen3-1.7B \
  --compiler build/tools/lpn-compile/lpn-compile \
  --output /tmp/qwen-timing-only --timing-only
```

This LPN-only command does not launch Verilator. Model weights and generated
binaries remain external; the model, runtime/compiler sources, RTL patches,
baseline reports and replay example are included in this repository.

The [full timing-only check](validation/qwen17/timing_only_fibonacci.json)
replayed the 18-input/8-output Qwen case in **292.31 s (4m52s)**, versus
372.41 s for the earlier validated replay: **1.27x** faster end to end and
**73.20x** faster than the recorded whole RTL processes. All 26 steps matched
the saved logits, generated tokens and predicted cycles, with zero rollbacks.
No RTL DRAM/program files were created; their serialization/hash timers are zero.
This is one measured run, not a controlled alternating benchmark. Initialization
took 23.63 s versus 0.65 s in the earlier replay, which used a RAM image cache.
The original RTL was not rerun; the existing per-step cycle errors are unchanged.

The [three-step replay pilot](validation/qwen17/replay_pilot.json) covers one
prefill and two generated-token decodes. All hashes/logits/cycles match; its
complete replay session took 45.63 s versus 2235.28 s of recorded RTL core time.
This is a partial-case check, not a completed eight-token generation result.
The `replay_completed.py` driver waited for each full batch case to end,
then ran one LPN-only replay at a time, recording complete external process
wall time in `~/lpn-runs/qwen3-1.7b-parallel/lpn-replays.json`. It removes only
the replay's regenerable large input dump after verification; reports and
validation checkpoints remain. All ten baselines and replays are complete;
the supervisor, collector and replay driver have exited normally. Missing
historical controller PIDs are expected after their terminal manifests are written.

`e2e.py` runs the upstream model compiler and Engine with real checkpoint
weights and tokenizer. Every prompt/decode token executes all layers,
attention, SwiGLU, KV-cache writes/reads and the vocabulary head. Host greedy
selection feeds the predicted token into the next execution. Prefill is
token-by-token; this does not yet benchmark tiled multi-token prefill.

The current bring-up checkpoint is
[HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, a trained Llama-family model.
Download its safetensors, configuration and tokenizer files at that revision
into `../FeSens-openTPU/models/SmolLM2-135M`. This is an additional public
validation fixture. Real Llama 3.2 3B execution is still unvalidated: access to
the official checkpoint returned HTTP 401, and upstream also currently rejects
its `llama3` RoPE scaling. Do not describe shape-only MLP tests or SmolLM2 as
successful Llama 3.2 inference.

```sh
../FeSens-openTPU/.venv/bin/pip install safetensors tokenizers
../FeSens-openTPU/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
../FeSens-openTPU/.venv/bin/python -m examples.opentpu.e2e \
  --opentpu-root ../FeSens-openTPU --model-dir ../FeSens-openTPU/models/SmolLM2-135M \
  --lpn-compiler build/tools/lpn-compile/lpn-compile \
  --output "$HOME/lpn-runs/smollm2-135m-long" \
  --depth 64 --cap 512 --prompt Once --tokens 64
```

Depth 64 matches SmolLM2's 64-dimensional heads; LPN and RTL use exactly the
same array/configuration. One first-position pilot executed 5,811 instructions
and roughly 2.18 million RTL cycles, took about 83–85 seconds of RTL core time,
and passed full memory checks with approximately **1.25%** timing error.
At that measured rate, 64 decoded tokens plus the one-token prompt is expected
to take **about 1.5 hours** of RTL core time. This is an estimate, not a completed
hour-scale accuracy result. Generation stops at EOS; choose a larger
`--tokens`/`--cap` for longer runs, within the checkpoint's supported context.
For example, `--tokens 256 --cap 512` targets roughly six hours at the pilot's
rate, subject to EOS and changing attention cost as context grows.

The [four-step pilot](validation/smollm2_pilot/report.json) includes one prompt
token and three actual decode steps, the fourth executed after exiting and
resuming from the snapshot. It generates `Once you have a`, checks **8,725,620
RTL cycles** in total, and stays within **1.251%** on every step. RTL core time
totals **327.18 s**, native LPN core time **10.90 s** (about **30×**, **0.81 M
LPN cycles/s**), and numerical execution plus model construction/compilation
and native processes **28.68 s**. These full-model numbers differ substantially
from GEMM throughput. The three untraced decode steps give about 30× core
speedup too. The pilot is a validation sample, not a completed long generation.
Its compiler/runtime hashes are retained in the report; a later cache-isolation
fix distinguishes identical integer and place-index arrays in other nets.

An additional [whole-sequence FP32 check](validation/smollm2_pilot/float-reference.json)
at position three obtains the same next-token argmax (253), cosine similarity
0.99992 and 2.54% relative logit error against the quantized device result.
This uses upstream's independent NumPy transformer reference and the trained
weights. It is one numerical quality sample, not a quality guarantee over the
checkpoint or the long generation.

The longer workload uses [this 72-token story prompt](prompts/story.txt) and
up to 64 decode tokens. Use the same command with
`--prompt "$(cat examples/opentpu/prompts/story.txt)"` and a new output path
such as `$HOME/lpn-runs/smollm2-135m-story`. At the pilot rate, prefill alone
takes about 1.6 hours of RTL core execution and a complete 64-token continuation
about three hours; an early EOS reduces only the decode portion.
The [initial long-run checkpoint](validation/smollm2_story_initial/report.json)
records progress, not a completed generation. Its first step, with the optimized
token runtime and packed completion flags, passes all memory checks at **1.245%
timing error**: RTL core **79.40 s**, LPN core **2.265 s** (**35.0×**, **0.975 M
modeled cycles/s**); functional/native preparation plus execution **6.644 s**.
The persistent run lives at `~/lpn-runs/smollm2-135m-story` on `icdslab2`.

`report.json` is updated after each checked token. `state.json` selects one of
two alternating snapshots containing DRAM (including KV cache) and logits.
The input/program/native executable for the current token is retained in
`current/`; memory dumps and snapshots are replaced rather than accumulated.
Restart with the same arguments plus `--resume`. `--max-steps 3` deliberately
checkpoints after three input tokens for a short bring-up/resume check; omit it
for the full generation. A process lock prevents concurrent writers. Resume
checks checkpoint, tokenizer, prompt, model/compiler/runtime source hashes,
configuration and snapshot hash, so changes cannot silently mix measurements.

Every token checks all DRAM/TMEM, instruction counts, identical input file
hashes and zero native rollbacks; the first token also checks the RTL command
PC path. Timing error over 5% stops the run and writes `failed-step.json`.
The first token is traced for validation; later tokens are untraced. Token
records identify tracing and separately report RTL build/process/core time,
functional execution, LPN graph construction, compilation and core time.
These are individual real generation steps, not repeated timing samples of
one fixed input. The numerical check establishes ISA/RTL equivalence for the
quantized execution, not independent FP32 Hugging Face quality equivalence.

## Dataflow and current scope

`model.py` composes DRAM ports and a systolic array inside per-slice control.
The upstream footprint rules supply RAW/WAR/WAW dependencies over TMEM, ACT
RAM, and DRAM. Tokens provide synchronization, dispatch-window credits,
in-order DMA/quantizer execution, and the two-entry matrix command queue.
Matrix prefetch may start before ACT operands become ready. A bounded FIFO
feeds row groups to the array, whose initiation interval is separate from its
chain/skew, epilogue, and accumulation latency. The final row group publishes
command completion; unused intermediate output events are omitted.

For long traces, `--compact` in `run`/`benchmark` selects `compact.py`, which
`e2e` uses by default. It indexes footprint ranges and removes transitive
completion dependencies, reuses a fixed set of transitions and 32 command
slots, and stores out-of-order completion flags in immutable 64-bit token
banks. Delayed messages update those banks atomically. Guard checks run first;
slot/flag latches remain present while read-only predicates inspect them.
Immutable place-handle arrays are resolved once by the native compiler/runtime.
This changes the net representation and avoids guard token copies; the
event-driven scheduler and its event ordering implementation are unchanged.
The expanded model remains a reference, with matching predictions checked on
the three original RTL fixtures and across completion-bank boundaries.

The native runtime represents empty-only ready queues with counters and a
shared empty-token sentinel. Queues containing data preserve FIFO with runs
of identical immutable handles. Take/emit/clone/peek retain or transfer payload
pointers instead of copying field maps; `.set()` creates a new payload. A
peek remains valid after its original token is taken. Failed firings restore
the same handles and queue order. `--stats` exposes payload allocations,
field copies on `.set()` and empty-token takes. Tests cover a billion empty
credits with zero payload allocations, mixed empty/data queues, delayed ties,
rollback, fan-out and signed bit 63. AddressSanitizer/UndefinedBehaviorSanitizer
also checked the ownership paths. The frontend/native compiler/C++ interpreter
support `andi`, `ori`, and `shli` for packed state (shift counts must fit the
integer width).

Guard-only helper calculations are removed from native bodies after the
separate enabling predicate has checked them. Prefix values still used by
the body are retained. The [same-MLIR native experiment](validation/gemm_million/native-optimization.json)
records the saved original executable versus the updated compiler/runtime,
including repeated samples and both binary hashes. The allocation changes
give a modest gain on this already grouped GEMM; their larger benefit is
avoiding token-object growth and copies in more heavily populated nets.

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
