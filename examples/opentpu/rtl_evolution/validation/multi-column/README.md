# Multi-column parallel row engines

`multi-column.patch` applies after `packed-qst.patch` and its preceding patches.
A clean application matches all 29 simulated source/harness files; see
`patch-proof.json`. The original baseline and earlier measured architectures
remain in separate frozen checkouts.

`MXU_COLS=1/2/4/8` selects the number of activation columns per parallel weight
row engine (`MXU_ROWS>1`). Each engine reuses its weight stream across those
columns, retaining the original systolic integer datapath and FP epilogue.
ACT rows are separately banked, so more columns read more row banks rather
than requiring additional complete activation copies. The epilogue has
`max(1, MXU_COLS/2)` lanes and twice that many core scratchpad request slots.
The shared scratchpad arbiter still enforces the original global bank limits.

The streaming readiness mask includes only useful columns of a partial final
group. It is independent of grant/readiness, avoiding a combinational loop.
The ACT group index is multiplied by the physical column count when selecting
row banks. LPN similarly waits for the last useful ACT row in each group and
uses the corresponding number of parallel groups. These are hardware/model
changes; upstream programs and functional inference are unchanged.

All **36** primitive accelerator cases pass full independent-ISA DRAM and
scratchpad comparisons, across all four column configurations. Fresh timing
comparison rechecks canonical inputs, raw logs, instruction counts, hardware
source/parameter/flag cache keys, simulator hashes and full numerical outputs.
All 36 native LPN predictions are within **7.4074%** of RTL, meeting the 10%
target for this synthetic scope. The largest error is the short 5x3 case at
eight columns; this is not complete inference validation.

| Case, eight row engines | C=1 RTL | C=2 RTL | C=4 RTL | C=8 RTL |
| --- | ---: | ---: | ---: | ---: |
| Unit-scaled 19x8 | 286 | 264 | 254 | 254 |
| Scaled 17x8 | 400 | 362 | 346 | 340 |
| ACC/ASCALE/RMAX | 189 | 184 | 180 | 183 |
| Small 5x3 | 126 | 127 | 131 | 135 |

Queued six-command gates additionally pass for C=2 and C=8, including changing
shapes, independent and overlapping destinations, ACC, ASCALE, RMAX and epoch
wraparound. Every such gate observes five overlapping command admissions.
The four-column-count native/Python readiness test and all 30 previous general
integration tests pass. Native input-table registration now rejects duplicate
names; this caught a naming collision while developing the extra row metadata.

The DSP cells are still the original **unpacked** cells. The independently
verified [paired-DSP component](../paired-dsp/README.md) is not integrated.
The `proposal_pair_*` DSE points are paired-column array shapes; their resource
estimates do not assume two MACs per DSP. Higher MAC counts, ASIC-scale points,
window sizes and multiple FP engines remain proposals requiring implementation
and full-workload validation. FPGA mapping, clock and the 100x goal are unproven.

```sh
python -m examples.opentpu.rtl_evolution.check_matrix \
  --opentpu-root ../FeSens-openTPU-multi-column --rows 8 --columns 2 \
  --stream-act --output /tmp/multi-c2
python -m examples.opentpu.rtl_evolution.check_matrix_pipeline \
  --opentpu-root ../FeSens-openTPU-multi-column --rows 8 --columns 2 \
  --output /tmp/multi-pipeline
```

Full unchanged 48-kernel LPN exploration has completed (see
`fpga-full-predictions.json`). At the same 8192 unpacked MAC lanes, 64 rows x
2 columns predicts 8,546,863 cycles versus 8,477,788 for compact 128x1: it is
slightly slower. Higher-resource 128x2 predicts 8,287,843 cycles (16.0269x
versus original RTL), only 2.24% fewer cycles than the compact control and
outside the conservative unpacked DSP screen. These results do not justify
promoting a wider array to full RTL inference validation yet.

The separate ASIC pilot, even with 1024 rows, eight columns, sixteen FP
engines and much higher bandwidth, extrapolates only 24.4x. It covers four
kernels, not all 48, and has no matching RTL implementation. Increasing one
array alone does not approach 100x on this instruction stream.
