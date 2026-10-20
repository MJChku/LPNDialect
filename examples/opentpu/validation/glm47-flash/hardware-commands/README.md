# GLM through actual openTPU instructions

The compiler emits canonical `input.json`, `prog_0.hex` and, for numerical
gates, `dram_0.bin`. Numerical ISA execution resolves data-dependent control
before the existing `CompactOpenTPU` sequencer/DMA/quantizer/MXU/VPU model
receives hardware commands. RTL consumes the same bytes. No tensor-DAG model,
annotated router IDs, timing formulas derived from model shapes, scheduler
changes or guard/heartbeat changes are involved.

The current compiler batches FP32 router reductions into multi-row RDOT
instructions. The [full-width router RTL gate](batched-router-full-rtl.json)
decreases from 48,898 to **20,885 cycles (2.34x)**, preserving numerical
results; LPN predicts 19,544 cycles, -6.42%. The [joined 16-row small gate](
batched-small-sixteenrows-rtl.json) matches complete memories in all stages
with worst timing error **5.32%**, confirmed by a [fresh audit](
batched-sixteen-row-audit.json). [Four- and sixteen-row continuity checks](
batched-lowering-continuity.json) verify unchanged DRAM inputs, exact router
IDs, and identical attention/expert programs across the compiler optimization.
The baseline evidence below remains preserved.

The [optimized full-width joined-layer RTL gate](batched-full-o3-rtl.json)
and its [fresh audit](batched-full-o3-audit.json) also pass. With batched
routing, RTL totals **783,848 cycles**, LPN **785,147 cycles**: **+0.17%
total error, 6.42% worst stage**. Every complete DRAM/TMEM state and
inter-stage DRAM handoff matches. The same actual inputs run on both paths.
This is one four-row MoE layer with deterministic numerical test weights.

Verilator is compiled with C++ `-O3` for this comparison. Its three simulation
clock windows sum to **266.784 seconds**, versus **1.091 seconds** for native
LPN windows, approximately **245x faster timing replay**. Native child-process
times sum to 1.129 seconds. These measurements exclude source compilation,
functional preparation, weight packing, image serialization and memory dumps;
they do not establish an end-to-end or full-GLM speedup. Three host workers
accelerate independent RTL validation snapshots; kernel times are summed in
the real device's stage order and each LPN runtime remains single threaded.

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
| Scalar-reduction router, H2048 / 64 experts / four rows | 48,898 | 49,964 | +2.18% |
| Batched-reduction router, same dimensions | 20,885 | 19,544 | -6.42% |
| Full-width MLA, H2048 / 20 heads / four rows | 190,958 | 193,035 | +1.09% |
| Full-width resident experts, same joined layer | 572,005 | 572,568 | +0.10% |
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
latency or runtime numbers. The [full-width RTL validation](
full-width-layer-rtl.json) is complete: all three stages match complete
memories and their continuity checks. RTL totals **811,861 cycles**, versus
815,567 predicted: **+0.46% total error, 2.18% worst stage**. The independent
[fresh ISA/native/full-memory audit](full-width-layer-audit.json) also passes.
This baseline retains scalar router reductions. Optimized-router programs
are tracked separately and preserve their attention/expert hardware inputs.

The unchanged four-row programs also pass through a [physical-profile sweep](
four-row-sweep.json), using the [explicit parameters](profiles.json).
At 572 DRAM bytes/cycle and 120-cycle latency, modeled layer cycles are:

| Hardware proposal | Kernel cycles |
| --- | ---: |
| Two matrix row engines, two columns | 828,217 |
| 96 matrix row engines, two columns | 282,015 |
| 128 matrix row engines, eight columns | 280,025 |
| Same matrix plus 256 elementwise lanes, reductions retained at 64 | 279,491 |

This is a **2.96x** predicted improvement at equal memory parameters, for
one four-row decoder layer. Routing alone remains about 57,000 cycles.
The original fast-memory control reproduces 815,567 cycles exactly. No
proposal in this table has new RTL timing, resource-fit or Fmax validation.
All profiles replay identical hardware work and actual resolved routes;
this is not a full-model speedup or a comparison to a commercial TPU.
The 1.5-GHz clock field is an explicit common assumption only.

With the optimized router and the same physical-profile list, unchanged
numerically resolved hardware commands give:

| Tile size | Two-row/two-column HBM proposal | 96-row/two-column proposal | 128-row/eight-column, 256-elementwise proposal |
| --- | ---: | ---: | ---: |
| Four rows | 797,797 | 251,595 | 249,071 |
| Sixteen rows | 2,584,781 | 502,647 | 496,910 |

These [four-row](batched-four-row-sweep.json) and [sixteen-row](
batched-sixteen-row-sweep.json) results are actual instruction-driven LPN
predictions, with equal memory parameters within each row. The latter is a
**5.20x** predicted cycle reduction for one 16-row layer, not full inference.
Its native simulation windows sum to about 0.23 seconds at the larger point.
The [numerical 16-row layer](batched-full-sixteenrows.json) uses explicit
1-MiB SRAM and real, computed routes; it is not a complete GLM workload.

Larger compiler tiles expose physical limits: 16 rows exhaust the original
65,536-word scratchpad, and attention row-max handling requires a logical
`MCOLS` capacity of at least the tile size. The joined-layer runner accepts
`--rows 16 --tmem-words 262144` to request a 1-MiB scratchpad and explicitly
sizes the row-max area. Physical dot columns remain a separate RTL/LPN
parameter. This is a larger hardware configuration, not free storage or
additional compute assumed for the old target. A regression checks both
original compilation failures and all new scratchpad footprints.

The first dense FFN (width 10240), untied 154880-row vocabulary head and
embedding table also compile at full dimensions without weight allocation.
Their predictions are probes; only the small numerical versions above have
completed RTL checks. Ten compiler tests pass, including exact projection
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
python -m examples.opentpu.hardware_sweep \
  --opentpu-root "$OTPU_ROOT" --compiler "$LPN_COMPILER" \
  --profiles examples/opentpu/validation/glm47-flash/hardware-commands/profiles.json \
  --inputs /path/to/retained-layer/mla /path/to/retained-layer/router \
    /path/to/retained-layer/experts \
  --functional --clock-mhz 1500 --output /tmp/glm-hardware-sweep
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
