# FPGA resource occupancy and necessary budgets

This diagnostic observes existing Python simulator transactions without changing
the event scheduler, model equations, guards or heartbeat implementation. It
records command dispatch, arrival, start, matrix compute admission, completion,
and intervals when a resource credit is held. Profiling wall time is excluded
from simulator performance comparisons. The observer must reproduce the native
prediction exactly before its report is accepted.

## Decode observation

Original real-inference kernel 16, using the 128-engine small-reciprocal proposal
with ACT streaming, quantizer lookup and the two-stage adder, completes in
**94,640 LPN cycles** in both native and observed Python execution. The upstream
instructions and canonical hex are unchanged. These are model observations;
this candidate's real-inference RTL pilot is separate and still pending.

| Resource | Occupied cycles | Fraction of kernel cycles |
| --- | ---: | ---: |
| Shared memory port B | 58,092 | 61.38% |
| Scale decoder port A | 17,062 | 18.03% |
| Matrix compute credit | 58,204 | 61.50% |
| Vector issue credit | 30,041 | 31.74% |
| Quantizer issue credit | 24,213 | 25.58% |
| DMA issue credit | 8,524 | 9.01% |

Intervals overlap and must not be added. Occupancy means a credit is held,
including waits for streaming inputs; it is not arithmetic/DSP utilization.
Per-command waits also overlap, so their sums are diagnostics rather than
contributions to elapsed time. Matrix dependency waits are not attributed.
The raw command/resource timelines are retained on `icdslab2` under
`~/lpn-runs/tpu-100x/timeline-small128-add2-16`.

The observer test checks streaming matrix and vector commands, exact agreement
with ordinary Python and native execution, reset behavior, completion coverage
and bounded resource occupancy.

## Full-workload necessary budgets

An independent instruction-only pass counts all 48 original kernels:

- **18,468,182,016 dense integer MACs**.
- **6,212,192,256 requested weight bytes**, before scale metadata, DMA, KV writes,
  burst padding and latency, assuming each MM fetches its weights from HBM.
- The 100× target is **1,328,282.63 cycles**, relative to original 132,828,263.

At the provisional FPGA budget of 8,678 matrix MAC lanes and 2,816 bytes/clock:

| Necessary resource bound | Optimistic floor cycles | Speedup ceiling |
| --- | ---: | ---: |
| Dense arithmetic, full utilization | 2,128,162 | 62.415× |
| Uncached weight bytes alone | 2,206,035 | 60.211× |

These are separate bounds and are not added. They assume neither a cache nor
lossless memory compression, and the arithmetic bound assumes the dense MAC
algorithm. They are not universal FPGA limits or LPN cycle predictions.

For 100× under those assumptions, even with perfect utilization, the design
would need at least **13,904 concurrent dense MAC lanes** and **4,676.86 weight
bytes/clock**—**935.37 GB/s at an assumed 200 MHz**, before other traffic. Thus
more execution units alone cannot achieve 100× within the current capacity and
bandwidth screen. DSP packing and fewer physical weight transfers are necessary
directions to investigate; neither is implemented by this diagnostic.

## Reproduce

```sh
PYTHONPATH=. python -m examples.opentpu.timeline \
  --case /path/to/evolved-pipeline-final \
  --prediction /path/to/small-recip-full-predictions/report.json \
  --architecture experimental_small_recip_128x1_add2 --kernel 16 \
  --opentpu-root /path/to/FeSens-openTPU-small-recip --output /tmp/profile-decode

PYTHONPATH=. python -m examples.opentpu.roofline \
  --case /path/to/evolved-pipeline-final \
  --opentpu-root /path/to/FeSens-openTPU-small-recip --output /tmp/roofline.json
```

Reports and source/input fingerprints are in
[validation/fpga-profile](rtl_evolution/validation/fpga-profile). These diagnostics
preserve the original workload and do not establish a new RTL speedup or FPGA fit.
