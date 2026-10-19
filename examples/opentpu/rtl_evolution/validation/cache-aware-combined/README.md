# Combined cache and paired-array pilot

`pilot.json` contains native timing on unchanged retained real kernels
0/15/16/47 of upstream SmolLM2-135M 128-input/32-output inference. It executes
no numerical tensor arithmetic and is candidate selection only. Intermediate
attention positions are extrapolated; this is neither a full inference result
nor RTL validation of these combined variants.

| Proposed configuration | Extrapolated cycle speedup | Counted SRAM lower bound |
| --- | ---: | ---: |
| 128x1, sixteen-entry scale cache | 16.5802x | 16,777,472 B |
| Compact ACT/FIFO 128x1, scale cache | 16.5717x | 11,534,592 B |
| Paired 96x2, scale cache | 16.9376x | 10,204,096 B |
| Paired 96x2, scale cache and 28 MiB weight cache | 17.1165x | 39,621,568 B |
| Compact 128x1 control, no cache | 15.9150x | 11,455,488 B |

Storage uses [the corrected resource bill](corrected-resources.json), which
adds five previously omitted TMEM read copies, instruction memory and the
quantizer FIFO: 1,703,936 bytes per point. The original timing report remains
unchanged. Bank-granularity waste, multiwrite implementation and pipeline/
control storage still need mapping; this is a capacity lower bound.

The weight-cache proposal consumes verified coherent prepass profiles; scale
misses are reconstructed from original instructions, with the documented
canonical-order coherence approximation. No RTL cycles enter the predictor.
DSP packing only halves dot cells. The paired array counts 6144 dot cells,
1728 column-end integer DSP cells and 192 matrix FP multiply sites; mapping
to DSP58, remaining units, memory banking, routing and Fmax need synthesis.
The 128x1 point exceeds the provisional 80% integer-DSP reserve screen.

The separate **full uncached paired 96x2 RTL** result is **15.4406x**, with
all 48 memory states exact and worst LPN error 5.3560%. These pilot estimates
do not supersede that measured result or establish the 100x target.
