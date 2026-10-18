# FPGA experiment: single-lane reciprocal for small rows

The default-off `VPU_SCALAR_RECIP` variant adds one exact reciprocal lookup
lane to the existing VPU. It uses the table validated in [RECIP_ROM.md](RECIP_ROM.md)
and leaves wide reciprocal commands on the original arithmetic pipeline.
The same compiled instructions select the hardware path; upstream inference
and the event-driven LPN runtime, guards and heartbeat caching are unchanged.

## Geometry and hardware

The real 128-input / 32-output SmolLM2-135M workload contains 14,400 reciprocal
commands that qualify for this path. Its important small shape is three columns,
not one. A one-column-only prototype did not improve complete-workload timing.

The implemented lane serializes one to four columns when the extra issue clocks
are no greater than the pipeline clocks saved. For rows `r`, columns `c`,
`s = 1 + multiply_latency + add_latency`, and tap `t = 1` if `s >= 4`, otherwise
`t = 2`, eligibility requires:

```
1 <= c <= 4
r * (c - 1) <= (6 - t) * s + 3 - (3 if t > 1 else 0)
```

The four-clock lookup is delayed to an existing retirement tap. Metadata carries
the selected class, lane mask and destination through stalls and ordered
writeback. Wider or unprofitable commands retain the original path. This avoids
an isolated-command latency regression; it does not guarantee a whole-program
improvement because issue and retirement can interact with other commands.

The VPU needs its own 2,123,264-byte table copy and a small integer predictor
multiplier. The quantizer has a separate copy because its enable can differ.
Together the tables occupy about 4.05 MiB before RAM width rounding. There is
one VPU lookup read port; no 16- or 64-port lookup memory is assumed. Resource
reports include these bytes, but DSP mapping, banking, routing and Fmax require
FPGA synthesis. The current implementation is restricted to the 64-lane VPU.
The timing variant requires `CompactOpenTPU` command selection.

## Measured numerical gates and timing forecasts

All full scratchpad comparisons pass against the independent ISA for 38 command
prefixes, including sign/exponent extremes, exceptional floats, strides,
in-place operations, cutoff boundaries and fallback widths. Another 48 mixed
commands pass with bursty stalls, both with and without buffered writeback.
The shorter 1/1 FP-stage configuration passes the same buffered checks.

With original 2/4 FP stages, isolated reciprocal latency changes as follows:

| Shape | Original clocks | Lookup clocks |
| --- | ---: | ---: |
| 1 × 3 | 52 | 16 |
| 8 × 3 | 59 | 37 |
| 13 × 4, fallback | 64 | 64 |

Nine matrix/quantizer integration cases also pass. The paired full-top
LD/QACT/MM/reciprocal/MUL/QACT/MM/ST gate has identical inputs and bit-exact full
DRAM/TMEM states in both configurations. It takes **411 clocks without this
feature and 413 with it**; this numerical integration gate is not a speedup.
Native and Python LPN execution agree, and the existing 30 integration tests pass.

All 48 retained kernels were timed in LPN with ACT streaming and quantizer lookup:

| Candidate | Predicted full cycles | Predicted reduction vs original RTL |
| --- | ---: | ---: |
| 32 engines, original FP stages | 14,035,542 | 9.464× |
| 128 engines, original FP stages | 9,348,242 | 14.209× |
| 128 engines, two-stage adder | 8,156,090 | 16.286× |

These are forecasts, pending real-inference RTL replay. An oversized ASIC
diagnostic reaches only 25.245× and is not an FPGA candidate. The 100× target
remains open. The latest audited full RTL result is the earlier
[128-engine architecture at 12.7717×](FPGA_REDUCTIONS.md).

## Reproduce

Apply `small-recip.patch` after all eight patches in [LOOKUP_SCALE.md](LOOKUP_SCALE.md)
on pinned FeSens revision `f237dab2f196855a59a083b1454786df8f01291f`.
The nine-patch proof reproduces all 54 candidate RTL/harness files.
Generate the verified reciprocal tables using [RECIP_ROM.md](RECIP_ROM.md), then:

```sh
PYTHONPATH=. python -m examples.opentpu.rtl_evolution.check_scalar_recip \
  --opentpu-root /path/to/FeSens-openTPU-small-recip \
  --table /tmp/recip-tables --output /tmp/small-recip-unit

PYTHONPATH=. python -m examples.opentpu.rtl_evolution.check_scalar_recip \
  --opentpu-root /path/to/FeSens-openTPU-small-recip \
  --table /tmp/recip-tables --pipeline --stall 2 --write-buffer 1 \
  --output /tmp/small-recip-pipeline

PYTHONPATH=. python test/test_opentpu_scalar_recip.py \
  --opentpu-root /path/to/FeSens-openTPU-small-recip \
  --tool build/tools/lpn-compile/lpn-compile
```

For retained real-inference replay, use the command in [LOWLAT_FP.md](LOWLAT_FP.md)
with `--activation-streaming --lookup-scale --scalar-recip-lookup
--recip-table /tmp/recip-tables`. The ROM initialization parameters are
`VPU_RECIP_COEFF_FILE` and `VPU_RECIP_CORRECTION_FILE`; they default to the
corresponding quantizer table paths. Table load time is excluded from inference
cycle comparisons, consistently with the existing memory-preload convention.

Reports, source/table hashes and patch proof are in
[validation/small-recip](rtl_evolution/validation/small-recip).
