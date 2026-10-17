# RTL architecture validation (in progress)

The prior 8.23× LPN result used historical token-at-a-time prefill. It is not
an established normal-inference speedup. This experiment implements hardware
and measures original/evolved RTL on the unmodified upstream inference path.

`parallel-rtl.patch` applies to FeSens revision
`f237dab2f196855a59a083b1454786df8f01291f`. It includes the passive wall timer
and large behavioral-memory representation used by the original validation.
Apply it in a separate clean worktree; preserve the original baseline checkout.

The implementation adds the following hardware (primitive numerical validation passed; checkpoint validation is in progress):

* 32 independent single-column systolic engines, with integer accumulation and
  the original FP epilogue. A decoded MM is striped across weight rows inside
  hardware. Each engine has its own weight/scale FIFO and activation RAM copy.
* Shared scratchpad-bank arbitration and shared memory byte-budget arbitration,
  including weight data, scale lines, DMA and scalar quantizer writes.
* A completion barrier and reduction of per-engine matrix maxima. One logical
  instruction completes only when every engine and final maximum write finishes.
* Narrow-column and replay-row indexing in the fused epilogue.
* A 64-lane VPU and reduction tree. Ordered sums retain the ISA's 64 partials;
  their issue rate respects the existing four-stage FP adder's feedback latency.
* 64 parallel QST byte converters and byte-enabled word links, supporting
  contiguous and strided stores. The shared memory budget charges four link
  bytes per active store lane (256 bytes for 64 lanes). This is an abstract
  logical memory interface, not an implemented HBM burst/coalescing controller.

The current configuration uses `MXU_ROWS=32`, `LANES=64`, `VPU_LANES=64`,
`QST_LANES=64`, `MEM_BYTES=2816`. Two-command parallel-MXU prefetch is still
missing. This is not yet an equivalent implementation of all earlier LPN
assumptions; in particular, wide ordered sums need feedback stalls. The multi-link behavioral DRAM retains actual data and fixed
read latency. It has no HBM channel/bank controller or board integration.
The independent 64-lane VPU gate has passed bit-exact full scratchpad comparison
for ADD, MUL, EXP2, RECIP, RSQRT, RSUM, RSSQ, RDOT, RMAX and ARGMAX with
193-element rows (including feedback and a partial final chunk). Full matrix,
checkpoint gates are still required. All nine full matrix/store gates passed after
correcting bank arbitration, including ACC/ASCALE/RMAX, FP4 PAIR, partial engine
groups, activation replay and QST strides 1, 4 and 17. The fused-epilogue gate found
that duplicate ASCALE reads overflowed the slice's per-bank lane counters even
when they named the same word. The wrapper now admits at most one read per
bank, and full gates have been restarted with this correction.
Cycle speedup at a common clock does not prove FPGA resource fit or Fmax.

Run numerical gates with Verilator 5.052 and the NumPy environment:

```sh
python examples/opentpu/rtl_evolution/check_matrix.py \
  --opentpu-root ../FeSens-openTPU-arch-rtl --output /tmp/parallel-matrix
```

Every gate compares the entire DRAM and TMEM against the independent upstream
ISA model. Gates include incomplete engine groups, replayed activation rows,
strided outputs, scales, ACC/ASCALE/RMAX, and packed weights. Gate completion
and the requested 128-prompt/32-output real-checkpoint experiment are still
required before reporting a verified architecture speedup.

The original-RTL long run is under
`~/lpn-runs/rtl-architecture-128x32` on `icdslab2.epfl.ch`. Its
`baseline-job.json` identifies the controller and worker; inspect their live
processes and `baseline-smol128x32/progress.json` before starting another job.
It uses SmolLM2-135M real weights, an exactly 128-token story prompt, capacity
256, 32 requested output tokens, and a frozen LPN project at `9672222`.
Upstream `Engine.generate()` owns batching, autoregressive decode and EOS.
A shorter EOS-terminated run must not be labeled a completed 32-output case.

The active corrected implementation on the server is `~/FeSens-openTPU-arch-wide`;
`matrix-wide-fixed.log` and `matrix-wide-fixed/source-identity.json` identify its
current full gates. Verify the owning PID in that identity and its descendants.
The older `matrix-gates` and `matrix-wide-gates` jobs were deliberately stopped
when the ASCALE grant deadlock was diagnosed; their stopped manifests explain
why. Their directories are diagnostic evidence, not completed validation.

Once the baseline and full gates pass, compare the complete checkpoint run:

```sh
PYTHONPATH=. python -m examples.opentpu.rtl_evolution.compare_inference \
  --opentpu-root ../FeSens-openTPU-arch-wide \
  --baseline "$HOME/lpn-runs/rtl-architecture-128x32/baseline-smol128x32-continued/report.json" \
  --output "$HOME/lpn-runs/rtl-architecture-128x32/evolved-smol128x32" \
  --compiler ../LPNDialect-heartbeat/build/tools/lpn-compile/lpn-compile
```

This checks every program/DRAM input hash against the original run, every final
kernel memory against independent ISA execution, and generated IDs and kernel
count against the original report. It also records evolved LPN predictions and
their errors rather than treating the old architecture forecast as validated.
Freeze both project and RTL sources for the long comparison. The final report
must show all 128 prompt tokens and all 32 actual output tokens before this
experiment can establish the requested end-to-end architecture speedup.

The first strict original-RTL run stopped at prompt position 32: numerical
memory checks passed, but the LPN prediction was 4,007,659 cycles versus
3,798,036 measured RTL cycles (+5.52%). This misses the 5% timing target.
`baseline-continued-job.json` owns the successor, using frozen project
`95fd1e6` in `~/LPNDialect-rtl-arch-validation`. Its progress is under
`baseline-smol128x32-continued`. The successor reconstructs upstream state by
ISA replay, rechecks the program/DRAM hashes and simulator binary before
reusing each of the first five verified RTL observations, and freshly compares
the last retained complete RTL memory dump. Later kernels execute RTL normally.
It records predictor misses explicitly while collecting actual hardware-cycle
measurements; this policy does not establish the 95% timing-accuracy target.
The normal `standard_inference` validation still fails on errors over 5%.
The predictor regression remains work to investigate and correct.

`compare_inference --baseline-job BASELINE_JOB.json` can consume each completed
original-RTL observation as the original job progresses. It verifies the live
worker's PID and command while waiting, then checks the same input hashes and
full numerical memories for the evolved RTL. It still requires the final
completed original 128/32 report and matching generated IDs before success.
This overlaps the two simulations without changing inference scheduling.
