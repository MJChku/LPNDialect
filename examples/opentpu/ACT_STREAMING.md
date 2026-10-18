# FPGA experiment: stream quantized activation rows

The last audited full inference result remains **8.6615x fewer RTL cycles**
for the buffered-quantizer 32-engine implementation. ACT streaming is a new,
default-off hardware and LPN experiment. It has not established a new full
inference RTL result, FPGA fit, or operating clock. The 100x target is open.

## Behavior and dependencies

The original instruction stream and upstream `Engine.generate()` scheduling
are unchanged. A buffered QACT computes its row scale from the complete row,
then emits the quantized blocks consecutively. Its writer now runs on an
independent enable, so a denied TMEM read cannot interrupt that row's ACT
writes. QST and unbuffered QACT retain their original grant behavior.

An MM can release before a conflicting QACT completes only when:

- The QACT row fits the existing bounded FIFO and covers every MM activation
  row and block, including DUP coverage.
- The conflict is solely an ACT read-after-write dependency. In particular,
  overlapping MM output and QACT TMEM input keep their full WAR dependency.
- The QACT has actually started. Dispatch alone cannot invalidate ACT: a younger
  overwrite may still be waiting for an older matrix reader.

A central row/block validity map clears the producer's target on acceptance
and publishes each write only when it lands in ACT RAM. A matrix core waits
at the first block of each activation row. Once it starts, both producer and
consumer advance one block per clock, preserving the systolic core's no-gap
contract. Partial replacements retain the whole-command completion barrier;
an untouched first block must not enable reads of later replaced blocks.

Only parallel MXU configurations with one block per quantizer write
(`MXU_ROWS > 1`, `ULANES == D`) and the footprint scoreboard are supported.
Defaults preserve the previous hardware behavior. The finite FIFO and row
banks remain in use. For 16 ACT rows and 64 blocks, the central ready map is
128 bytes; a 32-slot ACT-start dependency matrix is another 128 bytes before
metadata and pipeline tags. This reuses the existing row-data buffer and adds
no multiplier lanes. Routing and physical cost still require synthesis.

The LPN adds row-first availability events, a current-producer latch and
activation-row compute steps. Completed producers reuse the existing completion
flags. FIFO capacity returns after the final replayed activation row. Partial
coverage uses the same conservative completion barrier as RTL. The runtime's
event-driven scheduler, guard synthesis and place heartbeats are unchanged.

## Predictions on the complete retained inference

All 48 original kernels are consumed, without tensor arithmetic or a new
workload schedule. These are predictions at an assumed common clock:

| Candidate | LPN cycles | Predicted reduction vs original RTL |
| --- | ---: | ---: |
| 32 engines, original FP stages | 14,735,898 | 9.014x |
| 128 engines, original FP stages | 10,010,812 | 13.268x |
| 128 engines, two-stage FP adder | 8,654,012 | 15.349x |

The 128-engine baseline without streaming predicts 10,165,357 cycles. The gain
is modest; this does not solve the remaining compute and vector bottlenecks.
The capacity reference remains provisional V80/HBM assumptions documented in
[FPGA_REDUCTIONS.md](FPGA_REDUCTIONS.md), not an implemented board memory system.
The shorter adder also needs a frequency comparison after FPGA synthesis.

## Validation and reproduction

The quantizer numerical suite compares full ACT data, scale records and DRAM
at every command boundary against the independent ISA model: 28 cases each
for the production FIFO, tiny FIFO, bursty TMEM stalls, and feature-off control.
The streamed path additionally asserts that no row has a gap after its first
ACT write. Matrix component gates cover numerical scaling, accumulation,
RMAX, packed weights, strides and activation replay.

The paired ACT dependency suite runs identical programs and images through
feature-off and feature-on RTL. It covers a late quantizer acceptance, younger
ACT overwrites, partial replacements with different signs but identical row
scales, duplicated rows and unbuffered fallback. Partial replacements and TMEM
WAR keep their completion barriers. All states must match the ISA model.

The 30 existing integration tests pass, plus three new tests of dependency
classification, finite row schedules, Python/native timing agreement and
reuse of one compiled timing binary across different program shapes.
The 13 paired dependency cases pass bit for bit, with up to **1.1545x**
component cycle reduction. The streamed LPN's worst error is **8.621%** on the
12 cases without artificial handshake delay; that delay is tested for hardware
correctness but is outside the timing model. The feature-off one-block toy case
has −11.865% LPN error, exposing the existing fixed startup approximation.
Partial replacement and WAR cases retain their original cycles and stay within
1% model error. Component validation does not establish full inference accuracy.

Apply these patches in order to FeSens revision
`f237dab2f196855a59a083b1454786df8f01291f`:

```sh
git apply /path/to/LPNDialect/examples/opentpu/rtl_evolution/parallel-rtl.patch
git apply /path/to/LPNDialect/examples/opentpu/rtl_evolution/buffered-quant.patch
git apply /path/to/LPNDialect/examples/opentpu/rtl_evolution/interleaved-reductions.patch
git apply /path/to/LPNDialect/examples/opentpu/rtl_evolution/shared-matrix-banks.patch
git apply /path/to/LPNDialect/examples/opentpu/rtl_evolution/portable-tree.patch
git apply /path/to/LPNDialect/examples/opentpu/rtl_evolution/lowlat-fp.patch
git apply /path/to/LPNDialect/examples/opentpu/rtl_evolution/act-streaming.patch
```

From the LPN root, with the pinned Python/Verilator environment in
[README.md](README.md):

```sh
PYTHONPATH=. python -m examples.opentpu.rtl_evolution.check_act_streaming \
  --opentpu-root /path/to/FeSens-openTPU-act-streaming --output /tmp/act-gate

PYTHONPATH=. python test/test_opentpu_activation_stream.py \
  --opentpu-root /path/to/FeSens-openTPU-act-streaming \
  --tool build/tools/lpn-compile/lpn-compile

PYTHONPATH=. python -m examples.opentpu.tpu_dse \
  --case /path/to/evolved-pipeline-final \
  --opentpu-root /path/to/FeSens-openTPU-act-streaming \
  --compiler build/tools/lpn-compile/lpn-compile --indices all --workers 8 \
  --architectures experimental_fpga_stream_32x1,experimental_fpga_stream_128x1,experimental_fpga_stream_128x1_add2 \
  --output /tmp/act-predictions
```

Real-workload RTL replay uses the same `replay_architecture` command as
[LOWLAT_FP.md](LOWLAT_FP.md), adding `--activation-streaming`. Workers reject a
streaming mismatch between the hardware parameters and LPN timing. Source and
binary hashes distinguish these simulators from previous architectures.
Reports and the clean seven-patch reproduction proof are in
[validation/act-streaming](rtl_evolution/validation/act-streaming).


## Completed real-workload streaming pilot

The 32-engine, original-FP-stage replay of original kernels 0, 15, 16 and 47
completed at 1,583,564 RTL cycles, against 1,600,365 for its matching control
(1.01061x fewer cycles). Every full DRAM/TMEM state is bit-exact. LPN predicts
1,582,986 cycles: −0.0365% total error and 2.308% worst kernel error. A fresh
selected-kernel audit verified inputs, binaries, sources, logs and memories.
This is four selected kernels, not a complete inference timing result.
See `validation/act-streaming/inference-pilot{,-audit}.json`.
