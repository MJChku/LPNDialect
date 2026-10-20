# GLM through actual openTPU instructions

The compiler emits canonical `input.json`, `prog_0.hex` and, for numerical
gates, `dram_0.bin`. Numerical ISA execution resolves data-dependent control
before the existing `CompactOpenTPU` sequencer/DMA/quantizer/MXU/VPU model
receives hardware commands. RTL consumes the same bytes. No tensor-DAG model,
annotated router IDs, timing formulas derived from model shapes, scheduler
changes or guard/heartbeat changes are involved.

## Completed validation

The joined layer performs MLA, FP32 routing, then grouped routed/shared
experts in one physical DRAM image. Each RTL output is exactly the next
stage's input. Prefix memory footprints prove the lazily initialized expert
weights cannot affect attention or routing. Expert selection is computed by
real ARGMAX/RLD instructions; the compiler groups the actual IDs by expert.
Shared experts are unconditional. Rank norms use epsilon 1e-6; outer norms
use 1e-5; interleaved RoPE and causal masks have independent numerical gates.

The [fresh profile audit](profile-audit.json) reproduces every old prediction,
then corrects a physical target error: the two-column RTL defaults to **one**
epilogue lane (`otpu_mxu.sv`, parameter `E=0` means max(1,MCOLS/2)). Earlier
GLM runners inherited the generic two-lane `Timing` default. All original
reports remain immutable; the correction changes configuration only. It
removes the four-row attention timing failure without fitting a multiplier.

| Actual hardware gate | RTL cycles | Correct physical LPN | Error |
| --- | ---: | ---: | ---: |
| MLA, two rows, cache boundary | 3,858 | 4,109 | +6.51% |
| MLA, four rows, cache boundary | 6,257 | 6,325 | +1.09% |
| FP32 router, H2048 / 64 experts / four rows | 48,898 | 49,964 | +2.18% |
| Resident experts, small numerical gate | 7,161 | 6,775 | -5.39% |
| Joined small layer, attention | 3,679 | 3,938 | +7.04% |
| Joined small layer, router | 1,575 | 1,516 | -3.75% |
| Joined small layer, experts | 7,161 | 6,775 | -5.39% |
| First dense FFN, four rows | 2,987 | 2,893 | -3.15% |
| NVFP4 embedding gathers, four rows | 22,765 | 22,731 | -0.15% |
| Untied head and final norm, last row | 428 | 396 | -7.48% |

Every listed RTL gate matches complete DRAM and TMEM against independently
executed hardware instructions. This is hardware timing accuracy and backend
numerical validation with deterministic test weights, not trained-model text
accuracy. The backend uses NVFP4 weights, int8 activation dots/expanded KV and
FP32 intermediates; it is not bit-equivalent to upstream BF16 execution.
Numerical errors against independent equations using dequantized weights are
0.85% for the dense FFN, 0.68% for the head, and below 1e-6 for embedding gathers.

The reference target has two matrix row engines, two physical columns,
128-deep dots, eight-term chains, paired arithmetic and a 64-lane VPU/quantizer.
This is a compiler correctness target, not a TPU-v5e-sized design or proof of
FPGA placement. RTL C++ gates use `-O0`; their wall times must not be used as
a performance comparison against optimized native LPN.

## Real model dimensions

The [full-width layer](full-width-layer-original.json) uses H2048, 20 heads,
query rank 768, KV rank 512, 192+64 key dimensions, value dimension 256,
64 routed experts, top four and one shared expert of width 1536. Four rows
are processed together. Its 386,464,384-byte physical layout fits a 512-MiB
device image. Actual numerical MLA output feeds the actual router; its
resolved IDs determine the emitted expert instructions. No routing annotation
is used. The independent MLA and expert comparisons have relative errors
6.43% and 0.87%, including activation/KV quantization.

Corrected native predictions are 193,035 attention cycles, 49,964 router
cycles and 572,568 expert cycles, totaling **815,567 kernel cycles**. Native
timing takes about **1.1 seconds for this one four-row layer**; creating test
weights, numerical functional execution and input serialization took about
53 seconds in the retained run. Host route-readback, program loading and
compilation are excluded from kernel-cycle sums. These are not full GLM
latency or runtime numbers. Full-width RTL validation is running separately
under `~/lpn-runs/glm47-flash/layer-retained-full-rtl` on icdslab2; a running
case is not a passed timing gate.

The first dense FFN (width 10240), untied 154880-row vocabulary head and
embedding table also compile at full dimensions without weight allocation.
Their predictions are probes; only the small numerical versions above have
completed RTL checks. Eight compiler tests pass, including exact projection
MAC accounting, future-token independence, actual router decisions/unbiased
weights, expert batching and independent boundary equations.

## Reproduce

Use the pinned FeSens revision and packed-MXU RTL patches documented in
[the RTL evolution workflow](../../../rtl_evolution/README.md). Set paths
to that checkout and the LPN compiler built with input-array support:

```sh
export OTPU_ROOT=/path/to/FeSens-openTPU-packed-mxu
export LPN_COMPILER=/path/to/build/tools/lpn-compile/lpn-compile
export PYTHONPATH=.
python -m examples.opentpu.glm47_layer_hardware \
  --opentpu-root "$OTPU_ROOT" --compiler "$LPN_COMPILER" \
  --small-test --rtl --position 127 --output /tmp/glm-layer-gate
python -m examples.opentpu.glm47_boundary_hardware \
  --opentpu-root "$OTPU_ROOT" --compiler "$LPN_COMPILER" \
  --stage embedding --rows 4 --small-numerical-test --rtl \
  --output /tmp/glm-embedding-gate
python test/test_opentpu_glm47_compiler.py \
  --opentpu-root "$OTPU_ROOT" --tool "$LPN_COMPILER"
```

The other boundary stages are `dense` and `head`. Omit `--small-test` for
the numerical full-width joined layer (about 1.5 GiB of retained input
snapshots). Omit `--small-numerical-test --rtl` for weight-free full-size
boundary probes. All output directories must be new. On icdslab2 fresh
Verilator builds require both `PATH=$HOME/.local/opt/lpn-verilator-5.052/bin:$PATH`
and `VERILATOR_ROOT=$HOME/.local/opt/lpn-verilator-5.052`.

`validate_glm47_hardware` independently executes a retained layer's exact
inputs and runs its three RTL snapshots with `--workers 3`. Those are
parallel validation jobs; each LPN remains single threaded and device stage
ordering is checked using full memory hashes. Never restart a live long run
because a tool observation times out. Check its owner PID and `worker.json`.

## Remaining work

The complete 47-layer inference frontend must use standard upstream
generation scheduling and preserve layer/cache dependencies while emitting
hardware launches. LPN must continue to accept hardware commands only.
Whole-model weights exceed the existing 32-bit whole-image interface; an
explicit wider/banked hardware mapping or accounted streaming implementation
is still required. Per-layer launches do not justify silently treating host
weight replacement as free. Compressed/absorbed MLA and a larger, calibrated
matrix architecture remain separate hardware work. No full-model GLM speedup,
text quality, 100x gain or FPGA fit is claimed by these component gates.
