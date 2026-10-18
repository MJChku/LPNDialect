# Configurable arithmetic pipeline experiment

The FPGA-first 128-engine full RTL run remains frozen separately at
`~/lpn-runs/tpu-100x/fpga-128-normal128x32`. This experiment changes arithmetic
stage boundaries in another checkout. **The 100x objective is not achieved.**

The complete 48-kernel LPN predictions use the same retained upstream
SmolLM2-135M 128-input/32-output instruction streams:

| Architecture assumption | Add / multiply latency | Predicted cycles | Reduction vs original RTL |
| --- | --- | ---: | ---: |
| 128 row engines, 2,816 memory bytes/cycle | 2 / 2 clocks | 8,791,967 | 15.1079x |
| Oversized diagnostic: 1,024 engines, 65,536 bytes/cycle | 1 / 1 clocks | 6,726,981 | 19.7456x |

These are predictions, not completed inference RTL validations. The second
point is an intentionally oversized throughput experiment, not a specified
HBM implementation. Reducing arithmetic latency alone still leaves substantial
quantizer, vector and command dependency costs.

No clock frequency or FPGA fit is established for the shorter partitions.
An addition in one clock has more combinational work between registers than
the original four-clock implementation. The reported 200 MHz used for byte/s
conversion is an assumption; do not convert these cycle reductions into
measured hardware throughput or claim an FPGA can run them at that clock.

## Implementation and checks

Apply [lowlat-fp.patch](rtl_evolution/lowlat-fp.patch) after the five patches in
[FPGA_REDUCTIONS.md](FPGA_REDUCTIONS.md), in a separate pinned FeSens checkout.
The clean application reproduces all 25 simulated source files.

The defaults remain `OTPU_FP_MUL_LAT=2` and `OTPU_FP_ADD_LAT=4`. Verilator
defines select shorter partitions, for example:

```text
-DOTPU_FP_MUL_LAT=1 -DOTPU_FP_ADD_LAT=1
```

The FP32 functions, flush-to-zero behavior, canonical arithmetic NaNs and
round-to-nearest-even operations are unchanged. Adds may place the four existing
arithmetic substages behind one, two, three or four register boundaries;
additional clocks pad the result. Multiplies support one or two arithmetic
stages and optional padding. Multiply-add retains separate product/add rounding.
The quantizer, VPU, matrix epilogue and dependent metadata pipelines use the
selected latencies consistently. Timing support for non-default partitions is
currently restricted to the 64-lane architecture.

The independent NumPy arithmetic gate checks 8,192 vectors, including all
256 pairs of its special-value set, at add latencies 1..6, multiply latencies
1..3, multiply-add latencies 2/6/4 and double-product-add latency 2. It checks
both output values and holding state when the enable is low.

With one-clock add/multiply stages, additional actual RTL gates pass:

- 13 VPU functions, including all composite functions, reductions, NaNs,
  denormals, signed zeros, infinities and partial rows;
- 54 interleaved-row cases under stalled buffered writes;
- 28 quantizer cases under stalls and 28 without stalls, plus a separate
  28-case small-FIFO run;
- nine matrix/store numerical gates including ACC, ASCALE, RMAX and FP4.

These compare complete architectural output states with independent ISA
execution. They validate component arithmetic and control, not a completed new
full-inference timing target. All 30 LPN integration tests also pass, including
native/expanded agreement, the new partitions and the earlier default fixtures.

## Correct two-pass quantizer timing

The explicit row-bank model now distinguishes the per-row two-pass work from
the command's final pipeline drain. QST HALF reads the full row to find its
scale, then writes only half the elements. The previous approximation charged
the full row's stores and repeated a pipeline tail for every row.

The corrected two-pass formulas derive the scale/restart interval from the
selected multiply/add stages. They match all measured HALF cases at the
original and shorter partitions: for example eight 64-element rows take
370 clocks at 2/4 stages and 199 clocks at 1/1 stages. A nine-row, 33-block
unbuffered QACT takes 977 and 785 clocks respectively. Buffered paths retain
finite FIFO and row-bank credits. The original model defaults preserve their
previous approximation; explicit row-bank variants use the corrected formula.
The correction does not change the measured full inference cycle counts.

## Reproduce

Use the pinned Python/Verilator setup in [README.md](README.md), from the LPN
root with `PYTHONPATH=.` and single-threaded BLAS:

```sh
python -m examples.opentpu.rtl_evolution.check_fp_partition \
  --opentpu-root /path/to/FeSens-openTPU-lowlat-fp --output /tmp/fp-partition

python -m examples.opentpu.rtl_evolution.check_vector \
  --opentpu-root /path/to/FeSens-openTPU-lowlat-fp \
  --fp-mul-latency 1 --fp-add-latency 1 --output /tmp/fast-composites

python -m examples.opentpu.rtl_evolution.check_quant_buffered \
  --opentpu-root /path/to/FeSens-openTPU-lowlat-fp --chunks 32 \
  --fp-mul-latency 1 --fp-add-latency 1 --output /tmp/fast-quantizer

python -m examples.opentpu.tpu_dse \
  --case /path/to/evolved-pipeline-final \
  --opentpu-root /path/to/FeSens-openTPU-lowlat-fp \
  --compiler build/tools/lpn-compile/lpn-compile \
  --architectures experimental_fpga_128x1_add2,asic_1024x1_add1_mul1_memory65536 \
  --indices all --workers 8 --output /tmp/arithmetic-partition-predictions
```

Select corresponding LPN fields `Timing(fp_mul_latency=1,fp_add_latency=1)`
explicitly. The retained-input replay launcher passes these settings to each
RTL worker and records its exact compile flags and top parameters. Reused
observations must also match the current simulator source/parameter/flag cache
key; changing FP stages cannot silently reuse another architecture's binary.
The two-cycle-adder real-workload pilot is:

```sh
python -m examples.opentpu.rtl_evolution.replay_architecture \
  --case /path/to/evolved-pipeline-final \
  --baseline /path/to/baseline-smol128x32-continued/report.json \
  --opentpu-root /path/to/FeSens-openTPU-lowlat-fp \
  --compiler build/tools/lpn-compile/lpn-compile \
  --matrix-rows 32 --fp-mul-latency 2 --fp-add-latency 2 \
  --indices 0,15,16,47 --workers 4 --output /tmp/fp-add2-inference-pilot
```

This smaller matrix configuration isolates the arithmetic change. A four-kernel
pilot cannot establish a complete inference cycle reduction.
The event-driven runtime, guards and heartbeat caching are unchanged.

Reports and source fingerprints are in
[validation/lowlat-fp](rtl_evolution/validation/lowlat-fp). `initial-pilot.json`
and `initial-tests.log` preserve the stage before the two-pass quantizer
correction, at Git revision `9cd6b68`; `pilot.json`, `full-predictions.json`
and `tests.log` use the corrected model. The prescribed length and actual
upstream schedule are unchanged; no numerical tensor work enters predictions.
