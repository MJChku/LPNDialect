# RTL architecture validation

The implemented RTL runs the complete **SmolLM2-135M, 128-input/32-output**
workload in **15,435,996 cycles**, versus **132,828,263 cycles** for the original
RTL: **8.6051x fewer hardware cycles at a common clock**. Both execute the same
compiled programs and numerical DRAM inputs. Every kernel's complete DRAM and
scratchpad matches the independent ISA model, and all 32 generated token IDs
match. Upstream `Engine.generate()` owns batching, tiling, decode and EOS.

| Phase | Original RTL cycles | Evolved RTL cycles | Cycle speedup |
| --- | ---: | ---: | ---: |
| Prefill: 128 tokens, 16 upstream batches | 62,013,323 | 10,862,492 | 5.709x |
| Decode: 32 tokens | 70,814,940 | 4,573,504 | 15.484x |
| Complete inference: 48 kernels | 132,828,263 | 15,435,996 | **8.605x** |

The earlier 8.23x exploration used historical token-at-a-time prompt processing;
it was not a normal-inference result. The table above is a new measurement of
actual RTL using standard upstream batched prefill and autoregressive decode.

Corrected native LPN predictions were replayed from all 48 identical programs,
without rerunning tensor arithmetic or deriving timing from an RTL oracle:

| LPN target | Predicted cycles | Total cycle error | Worst kernel error |
| --- | ---: | ---: | ---: |
| Original RTL | 133,654,573 | +0.622% | 2.013% |
| Evolved RTL | 15,712,791 | +1.793% | 3.462% |

All kernels completed with zero rollbacks. The event-driven scheduler, guard
synthesis and heartbeat cache are unchanged. The 24 integration tests passed,
including Python/native/expanded-model agreement, dependency hazards, recorded
RTL fixtures, actual FIFO capacity and independent DRAM work during quantization.

These are hardware cycle reductions. They do not establish FPGA Fmax or resource
fit. Memory remains a numerical behavioral model with eight-cycle read latency
and shared byte-budget arbitration; the evolved limit is 2,816 link bytes per
cycle. A vendor HBM controller, channel/bank timing and board integration are
not implemented. Host workers validate separate HALT-drained kernels in parallel;
the modeled accelerator latency is the sum of all 48 kernels.

## Implemented hardware

`parallel-rtl.patch` applies to FeSens revision
`f237dab2f196855a59a083b1454786df8f01291f`. Apply it in a clean separate worktree;
preserve the original baseline checkout. A clean application was checked against
all 25 simulated source files, with exact SHA-256 matches. The patch includes the
passive wall timer and large behavioral-memory representation used for validation.

* 32 independent single-column systolic engines, with original integer accumulation
  and FP epilogues. Hardware stripes each decoded MM across weight rows; software
  and compiled instruction streams are unchanged. Each core has weight/scale
  FIFOs and an activation RAM copy.
* Two-command matrix prefetch. Per-core completion epochs and a logical head
  barrier preserve output ordering and the final reduction of matrix maxima.
* Shared scratchpad-bank arbitration and shared memory byte-budget arbitration
  for weights, scales, DMA and quantizer stores.
* A 64-lane VPU and reduction tree. Ordered sums preserve the ISA's 64 partials
  and the real four-stage adder feedback limit.
* 64 parallel QST byte converters and byte-enabled word links for contiguous
  and strided stores. Each active lane charges four link bytes. This is a
  logical memory interface, not an HBM burst/coalescing controller.

Parameters are `MXU_ROWS=32`, `LANES=64`, `VPU_LANES=64`, `QST_LANES=64`,
`MEM_BYTES=2816`. The measured model uses hardware D=64 for its 64-element
attention heads. This result does not validate a different model or FPGA mapping.

The eleven matrix/store numerical gates cover partial engine groups, activation
replay, nonzero ACC/ASCALE inputs, RMAX, FP4 PAIR, QST strides 1/4/17 and queued
sequences with overlapping outputs and epoch wraparound. Traces show younger
command admission before older completion. The independent VPU gate checks ten
operators, including reductions with 193-element rows, against bit-exact full
scratchpad state. Gate builds use unoptimized host C++; checkpoint simulation
uses `-O3`. Neither flag changes hardware cycle counts.

## Reproduce

Use Verilator 5.052 and a Python environment with NumPy, Torch, tokenizers and
safetensors. Set `VERILATOR_ROOT` and `PATH` consistently when using a private
Verilator installation. Run from the LPN project root with `PYTHONPATH=.` and
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Compile LPN as documented in the
project README. Keep both project and RTL sources frozen during measurements.

Prepare two clean checkouts at the pinned revision. The original uses the
observer and large-memory patches; the evolved patch already includes both:

```sh
git -C ../FeSens-openTPU-original apply "$PWD/examples/opentpu/verilator-observer.patch"
git -C ../FeSens-openTPU-original apply "$PWD/examples/opentpu/verilator-large-memory.patch"
git -C ../FeSens-openTPU-arch-pipeline apply "$PWD/examples/opentpu/rtl_evolution/parallel-rtl.patch"
```

First run the original RTL. An empty/nonexistent `--reuse` directory means no
observations are reused; existing observations require identical inputs and
simulator hashes and retain their provenance.

```sh
python -m examples.opentpu.rtl_evolution.baseline_inference \
  --opentpu-root ../FeSens-openTPU-original \
  --model-dir /path/to/SmolLM2-135M \
  --prompt-json examples/opentpu/rtl_evolution/validation/prompt128.json \
  --reuse /tmp/no-retained-observations --output /tmp/original128x32 \
  --compiler build/tools/lpn-compile/lpn-compile

python -m examples.opentpu.rtl_evolution.parallel_inference \
  --opentpu-root ../FeSens-openTPU-arch-pipeline \
  --baseline /tmp/original128x32/report.json \
  --output /tmp/evolved128x32 --workers 16 \
  --compiler build/tools/lpn-compile/lpn-compile
```

`parallel_inference` prepares the full workload in upstream inference order using
independent ISA execution. Workers compare every canonical program/DRAM input hash
with original RTL, execute the simulator, compare every output byte and scratchpad
word, and run native LPN. Its simulator configuration is idempotent across reused
workers, so an equivalent flag list does not trigger another simulator build.

Optional `--reuse-rtl PREVIOUS_CASE` reuses only completed identical-source RTL
observations after matching configuration, input and binary hashes and comparing
retained full outputs against fresh ISA execution. It records the source of each
reused observation. Output directories must be new.

Replay corrected timing and audit the full experiment:

```sh
python -m examples.opentpu.rtl_evolution.replay_baseline_timing \
  --opentpu-root ../FeSens-openTPU-original --case /tmp/original128x32 \
  --output /tmp/original-timing --workers 4 \
  --compiler build/tools/lpn-compile/lpn-compile

python -m examples.opentpu.rtl_evolution.replay_baseline_timing \
  --opentpu-root ../FeSens-openTPU-arch-pipeline --case /tmp/evolved128x32 \
  --parallel-rtl --output /tmp/evolved-timing --workers 4 \
  --compiler build/tools/lpn-compile/lpn-compile

python -m examples.opentpu.rtl_evolution.audit_completed \
  --baseline /tmp/original128x32 --evolved /tmp/evolved128x32 \
  --replay /tmp/original-timing --evolved-replay /tmp/evolved-timing \
  --baseline-root ../FeSens-openTPU-original \
  --evolved-root ../FeSens-openTPU-arch-pipeline \
  --output /tmp/rtl-architecture-audit.json
```

The audit requires all 48 verified kernels and the actual 128/32 token IDs and
schedule. It freshly compares all evolved raw DRAM/TMEM dumps, rehashes sources,
simulators and inputs, checks raw RTL/native completion logs, and recomputes
phase and total cycle counts. Corrected predictions remain separate from the
original recorded predictions; errors are not silently overwritten.

## Retained evidence

[The complete audit](validation/normal128x32-audit.json),
[original RTL report](validation/normal128x32-baseline.json),
[evolved RTL report](validation/normal128x32-evolved.json),
[corrected original timing](validation/normal128x32-original-timing.json),
[corrected evolved timing](validation/normal128x32-evolved-timing.json),
[clean patch proof](validation/normal128x32-patch-proof.json),
[original patch proof](validation/normal128x32-original-patch-proof.json),
[upstream/checkpoint proof](validation/normal128x32-upstream-proof.json),
[final retained-worker check](validation/normal128x32-reuse-proof.json), and
[integration log](validation/normal128x32-tests.log) retain the measurements and
source/input/binary hashes. Numerical weights and large memory dumps stay on
the server rather than in Git.

Raw data on `icdslab2.epfl.ch` is under `~/lpn-runs/rtl-architecture-128x32`:
`baseline-smol128x32-continued`, `evolved-pipeline-final`,
`corrected-original-final`, `corrected-evolved-full`, and
`completed-comparison-audit.json`. Both complete inference jobs and the final
audit are terminal. The original baseline explicitly reused five previously
verified observations; the evolved successor reused sixteen completed prefill
observations after fresh input and full numerical rechecks. Superseded jobs and
all completed evidence are preserved.
Both RTL checkouts retain all 80 upstream functional/compiler Python files
unchanged; their hashes and checkpoint file hashes are archived separately.

The long run exposed two model issues. FIFO requests now reserve their actual
chunk counts rather than a worst-case packet slot. Quantization reserves shared
DRAM only for stores, allowing unrelated matrix prefetch during its scratchpad
reads and conversion. Both fixes model hardware resources and preserve the
original default timing behavior. Early predictor misses remain in the raw
reports; the two corrected full replays establish the accuracy in the table.
