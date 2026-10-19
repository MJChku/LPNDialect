# Fixed-budget paired-array shape pilot

The native LPN pilot uses original real kernels 0/15/16/47 from upstream
SmolLM2-135M 128/32 inference, with a private sixteen-entry scale cache.
Every candidate has at most 8528 source-level integer DSP cells, including
column ends. These are timing predictions, not new RTL or FPGA-fit results.

| Array | Dot plus column-end integer DSP cells | Extrapolated cycle speedup |
| --- | ---: | ---: |
| 96x2 control | 7872 | 16.9376x |
| 104x2 | 8528 | 16.9298x |
| 48x4 | 7872 | 16.5147x |
| 52x4 | 8528 | 16.1039x |
| 24x8 | 7872 | 14.4334x |
| 26x8 | 8528 | 14.2312x |

The 96x2 control remains preferable. More columns lose utilization on many
single-column decode operations; more rows do not necessarily improve wave
occupancy. No RTL implementation of these extra shapes is pursued from this
evidence. Packing only halves integer dot cells, and DSP58 mapping, FP and
other units, placement and clock still need proof. Intermediate attention
positions are extrapolated; none of these values is a full inference result.
