# Shared-bank matrix engines and interleaved reductions

The 128-engine implementation completes all 48 unchanged SmolLM2-135M
128-input / 32-output kernels in **10,400,210 RTL cycles**, versus 132,828,263
for original RTL: **12.7717x fewer cycles** at an assumed common clock. Every
full DRAM and scratchpad state is bit-exact against independent ISA execution.
Generated IDs are preserved by full output equivalence. The 100x objective is
still unfulfilled; FPGA resource fit, clock and vendor HBM remain unverified.

LPN predicts 10,165,357 cycles: **−2.258% aggregate error**, with **6.243% worst
kernel error**. All kernels meet the accepted 10% timing tolerance. The earlier
13.0668x prediction is retained as a prediction, not substituted for the actual
12.7717x result. The 32-engine quantizer predecessor takes 15,335,479 cycles
([8.6615x](TPU_EXPLORATION.md)).

A fresh audit rereads all canonical program/DRAM inputs, decoded instruction
JSON, simulator/source hashes, cycle/instruction logs and complete memory
outputs. It also checks raw native timing/completion logs. The original
upstream `Engine.generate()` schedule and compiled inputs are unchanged.

| Phase | Original RTL | New RTL |
| --- | ---: | ---: |
| Prefill | 62,013,323 | 6,284,346 |
| Decode | 70,814,940 | 4,115,864 |
| Total | 132,828,263 | 10,400,210 |

Reports are `validation/fpga-reductions/full-inference-{rtl,audit}.json`.

## Bounded row contexts

The 64-lane VPU now has four independent sets of the ISA's 64 partial sums.
It interleaves chunks from up to four rows through the same multipliers and
four-stage adders. Every row retains the original accumulation and folding
order, including floating-point rounding. Partial groups of one, two or three
rows insert the required feedback bubbles. The contexts occupy 1 KiB of raw
state; this is not a complete mapped FPGA resource report.

A conservative address-hull test keeps the original row-major schedule when
reduction outputs overlap either input. It therefore does not silently change
the behavior of in-place commands. RSUM, RSSQ and RDOT share the new path;
other vector operations retain their existing datapaths and scheduling.

The LPN opts in with `Timing(vector_row_contexts=4)` and physical `LANES=64`.
The default remains one context. The same address-hull check selects the
interleaved issue span, including finite context capacity and partial-group
feedback gaps. The scheduler, dependency guards and heartbeat cache are unchanged.

Actual VPU RTL, with identical input data and complete scratchpad comparison
after every command:

| RSSQ shape | One context | Four contexts |
| --- | ---: | ---: |
| 1 x 193 | 49 cycles | 49 cycles |
| 8 x 193 | 161 cycles | 68 cycles |
| 8 x 576 | 321 cycles | 108 cycles |
| Aliased 8 x 193 | 161 cycles | 161 cycles |

Each of the original-schedule control, interleaved path, interleaved path under
stalls, and buffered-write path under stalls passes 54 numerical cases. Cases
cover partial context groups, tail columns, broadcast operands, cancellation,
denormals, NaNs/infinities, alias fallback and a following ordinary VOP. All 28
LPN integration tests pass, including native/expanded agreement.

## Share scratchpad ports across engines

The first 128-engine RTL gate exposed the previous wrapper's fixed assignment
of two scratchpad lanes per engine. It correctly rejected 128 engines on a
64-lane scratchpad. The larger geometry predictions did not establish that an
implementation existed; the failed gate is retained in the evidence.

The wrapper now routes accepted requests by physical bank. Engines contend for
the same 64 banks; a fair rotating priority chooses at most one read and one
write per bank. It retains every requesting engine's synchronous read response
while other engines reuse the bank, so a stalled engine cannot consume another
engine's value. No extra scratchpad ports are assumed.

Nine actual RTL gates pass with 128 engines / 64 banks, including 259 output
rows, scaled weights, ACC, ASCALE, RMAX, packed FP4 and strided QST. A second
nine-case gate passes with four engines / four banks, deliberately oversubscribing
the original two-private-lanes requirement. Both compare all DRAM and scratchpad
against independent ISA execution.

The narrow-bank gate also exposed generate-local constant functions in the
folding tree that current Verilator rejects. `portable-tree.patch` moves those
functions and their required constants to module scope. It changes no arithmetic
or pipeline. The narrow-bank gate uses that portable tree; the first wide-bank
gate used the equivalent wide branch before the scope-only move.

## Reproduce

Apply these patches in order to a **separate** FeSens checkout at
`f237dab2f196855a59a083b1454786df8f01291f`:

1. `rtl_evolution/parallel-rtl.patch`
2. `rtl_evolution/buffered-quant.patch`
3. `rtl_evolution/interleaved-reductions.patch`
4. `rtl_evolution/shared-matrix-banks.patch`
5. `rtl_evolution/portable-tree.patch`

The clean-patch proof matches all 25 simulation source files. Preserve the
original, 8.605x and 8.6615x source/binary/evidence checkouts.

From the LPN root, with the Python environment, native compiler and Verilator
setup in [README.md](README.md):

```sh
python -m examples.opentpu.rtl_evolution.check_vector_rows \
  --opentpu-root /path/to/FeSens-openTPU-fpga-reductions \
  --stall 1 --write-buffer 1 --output /tmp/vector-row-gates

python -m examples.opentpu.rtl_evolution.check_matrix \
  --opentpu-root /path/to/FeSens-openTPU-fpga-reductions \
  --rows 128 --output /tmp/matrix-128-gates

python -m examples.opentpu.rtl_evolution.replay_architecture \
  --case /path/to/evolved-pipeline-final \
  --baseline /path/to/baseline-smol128x32-continued/report.json \
  --opentpu-root /path/to/FeSens-openTPU-fpga-reductions \
  --compiler build/tools/lpn-compile/lpn-compile \
  --matrix-rows 128 --indices all --workers 12 --output /tmp/fpga-128-inference
```

The replay compares new RTL against every retained original program/input and
complete independent ISA output. Kernel cycles sum; validation workers do not
add hardware concurrency. `--matrix-rows 32` gives a smaller control with the
same quantizer, row contexts and bank router. Do not edit frozen sources while
a replay is running.

Reports are in [validation/fpga-reductions](rtl_evolution/validation/fpga-reductions).
Raw data is under `~/lpn-runs/tpu-100x` on `icdslab2.epfl.ch`.

## Resource implications for the 100x target

The retained dense command stream requests 18,468,182,016 matrix MACs. A 100x
result must fit within 1,328,282.63 cycles, requiring at least **13,904 useful
MACs/cycle** before other costs. The conservative V80 screen reserves 20% of
10,848 DSPs and assumes one MAC per DSP: 8,678 MACs, which gives only a 62.4x
matrix-compute ceiling under those assumptions even at perfect utilization.

Therefore a 100x FPGA design needs more arithmetic per DSP, fewer executed
dense operations, or a different resource envelope, as well as better memory
reuse and shorter serial dependencies. Passing the current capacity screen
does not prove placement, routing, HBM integration or clock frequency. Larger
ASIC-scale points remain separate experiments; the FPGA design remains the
priority.
