# Weight-cache study with corrected memory traffic

This repeats the bounded persistent-cache timing study with explicit physical
scale-line traffic, ordered matrix retirement and the two-entry VPU FIFO.
All 48 original upstream 128-input/32-output programs and their arguments
are unchanged. The validated cold/warm/invalidation profiles are reused with
their reference, canonical program and manifest hashes checked per kernel.
No tensor arithmetic is repeated.

| Weight cache | Full LPN cycles | Fewer cycles vs uncached | Predicted speedup vs original RTL |
| --- | ---: | ---: | ---: |
| None | 8,476,740 | — | 15.6697x |
| 24 MiB | 8,433,764 | 0.5070% | 15.7496x |
| 48 MiB | 8,391,204 | 1.0091% | 15.8295x |

`full-predictions.json` contains all 96 cache predictions. The uncached control
is retained under `../matrix-retirement/physical-full-prediction.json`.
The larger traffic savings do not translate into comparable cycle savings:
the separate cache path retains read latency, scale traffic remains uncached,
and compute/dependencies can dominate. The caches have no matching RTL and
are not selected for implementation given this small benefit.

The corrected decode profile in `decode-profile.json` agrees exactly with
native LPN at 108,415 cycles. Its shared memory link is occupied about 74.9%
of elapsed cycles, matrix compute about 49.2%, matrix retirement about 14.5%,
VPU about 27.7%, and quantization about 22.3%. These overlapping occupancies
must not be summed as elapsed time. Profiling Python execution time is not
a simulator speed benchmark.

At the provisional 200 MHz V80 capacity assumptions, 24 MiB plus the counted
base architecture fits the 50% SRAM capacity screen. The 48 MiB proposal
requires the less conservative 80% SRAM screen. These screens are not FPGA
mapping or Fmax evidence. Neither cache result is a measured hardware speedup.

Reproduce from a frozen project checkout:

```sh
python -m examples.opentpu.tpu_dse --case /path/to/normal128x32 \
  --opentpu-root ../FeSens-openTPU-scalar-recip --compiler /path/to/lpn-compile \
  --architectures physical_pinned24m_128x1_add2,physical_pinned48m_128x1_add2 \
  --memory-profile-root /path/to/pinned-cache-profiles \
  --indices all --workers 8 --output /tmp/physical-cache-study
```
