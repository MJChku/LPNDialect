# FPGA experiment: exact lookup scale unit

The reciprocal ROM is now connected to QACT/QST in a default-off RTL variant.
The table replaces the normalized Newton sequence without changing quantized
bytes or scale records. The VPU still uses its original arithmetic. Full
inference RTL validation and FPGA synthesis/frequency remain pending.

## Measured components and complete-workload forecasts

The scale-unit comparison checks 131,072 inputs, including every exponent and
sign across selected mantissas, zeros, denormals, infinities, NaNs and shuffled
words, under held clock enables. Both scale and inverse outputs are bit-exact:

| Multiply/add stages | Original scale latency | Lookup scale latency |
| --- | ---: | ---: |
| 2/4 | 32 | 6 |
| 2/2 | 24 | 6 |
| 1/1 | 13 | 5 |

Latencies count enabled clocks. With the original FP stages, the complete
quantizer takes 28 rather than 54 clocks for QACT of one 64-element row, and
114 rather than 140 for eight 576-element rows. A HALF QST of eight 64-element
rows takes 162 rather than 370 clocks. These are component measurements.

The final numerical gates pass 28 commands each with the production FIFO,
bursty TMEM grants and a tiny FIFO, plus a feature-off control. They compare
all ACT bytes, scale records and DRAM at every command boundary against the
independent ISA. Nine matrix integration gates pass full DRAM/TMEM comparisons,
including scaling, accumulation, RMAX, packed weights, strides and replay.
The LPN quantizer equations match every measured production case exactly;
Python/native event execution also agrees.

All 48 retained normal-inference kernels were evaluated in LPN:

| Candidate | LPN cycles | Predicted reduction vs original RTL |
| --- | ---: | ---: |
| 32 engines, original FP stages | 14,440,002 | 9.199x |
| 128 engines, original FP stages | 9,754,592 | 13.617x |
| 128 engines, two-stage adder | 8,449,670 | 15.720x |

These are forecasts, not measured full inference results. The 100x target is
still open. The original audited 8.6615x result remains the verified full-run
baseline. Timing consumes the original compiled instruction words; neither
this model nor the ROM generator chooses an inference schedule.

## Storage, initialization and input identity

The correction array uses 2 MiB, plus 25.5 KiB of coefficient bits before RAM
width rounding. These arithmetic constants are independent of model weights.
The quantizer uses one ROM read port and one small signed predictor multiplier,
plus the original two final FP multiplies. Physical DSP/RAM mapping, routing
and clock still require synthesis; the resource screen includes the table bytes
as a lower bound.

For simulation, pass `+recip_coeff=...` and `+recip_correction=...`. For synthesis,
set top parameters `QUANT_COEFF_FILE` and `QUANT_CORRECTION_FILE` to the generated
hex paths and enable `QUANT_LOOKUP`. They propagate to constant `$readmemh`
arguments. An exhaustive Verilator gate with `SYNTHESIS` defined checks this
initialization branch; it is not a Vivado synthesis or fit result. Add both new
SV modules to the FPGA synthesis fileset when adapting a board build.

Inference cycle comparisons exclude configuration/table initialization, as they
exclude the baseline harness's initial memory preload. A cold table transfer
would be an additional setup cost. The tables are generated once and re-used.
Workers validate them against the exact hashes checked by the exhaustive SV
gate, save those hashes with each result, and reject mismatched retained data.

The decoded JSON used by the timing prepass is now checked word for word
against canonical RTL hex, with the same trailing-zero serialization. All 48
retained kernels pass. New replays record JSON hashes as well as the original
canonical instruction/DRAM hashes. The auditor repeats this identity check.
No scheduler, guard synthesis, event handling or heartbeat implementation changes.

## Reproduce

Apply `lookup-scale.patch` after the seven patches listed in
[ACT_STREAMING.md](ACT_STREAMING.md), on pinned FeSens revision
`f237dab2f196855a59a083b1454786df8f01291f`. The eight-patch proof reproduces all
54 RTL/harness files in the candidate checkout. Generate tables with
[RECIP_ROM.md](RECIP_ROM.md), then from the LPN root:

```sh
PYTHONPATH=. python -m examples.opentpu.rtl_evolution.check_quant_buffered \
  --opentpu-root /path/to/FeSens-openTPU-fast-scale --output /tmp/lookup-quant \
  --chunks 1024 --stream-act --lookup-scale --recip-table /tmp/recip-tables

PYTHONPATH=. python -m examples.opentpu.rtl_evolution.check_matrix \
  --opentpu-root /path/to/FeSens-openTPU-fast-scale --output /tmp/lookup-matrix \
  --rows 4 --stream-act --lookup-scale --recip-table /tmp/recip-tables

PYTHONPATH=. python test/test_opentpu_lookup_scale.py \
  --opentpu-root /path/to/FeSens-openTPU-fast-scale \
  --tool build/tools/lpn-compile/lpn-compile \
  --gate-report /tmp/lookup-quant/report.json
```

For retained-input RTL replay, use the command in [LOWLAT_FP.md](LOWLAT_FP.md)
with `--activation-streaming --lookup-scale --recip-table /tmp/recip-tables`.
The default FP stages remain 2/4. Predictions use architectures
`experimental_lookup_32x1`, `experimental_lookup_128x1` and
`experimental_lookup_128x1_add2` in `tpu_dse`. Its optional `--cache` reuses the
compiled timing binary across studies while binding each workload's own tables.

Reports and source/table fingerprints are in
[validation/lookup-scale](rtl_evolution/validation/lookup-scale).
