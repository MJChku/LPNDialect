# Actual hardware-command LPN replay

This is the existing `CompactOpenTPU` component model, consuming the actual
FeSens eight-word instruction stream. **No tensor-DAG model is used.** GLM
has no complete hardware program yet; the full-model workload here is the
retained, numerically validated upstream SmolLM2-135M inference: 128 input
tokens, 32 output tokens and 48 kernels, with its original schedule intact.

The input entry point is documented in [hardware command replay](../../../HARDWARE_COMMANDS.md).
The three new tests verify exact agreement of hardware PCs, encoded resolved
instructions, arithmetic counts and native LPN cycles with numerical ISA
execution, including scalar loops, NVFP4 and PAIR. They reject tensor records,
32-bit-address overflow, malformed words, manifest/IMEM mismatches and unresolved
data-dependent control. All three pass (0.431 s). The 30 existing Python/native
integration tests also pass (113.740 s, `general-tests.log`).

## Direct canonical-hex replay

`control-replay.json` reads the retained kernel 0 `input.json` and `prog_0.hex`
without loading weights or tensor values. With the known cached hardware
parameters it predicts **302,421 cycles**, identical to the prior LPN result,
versus **314,371 RTL cycles**. Three native samples are
0.5462 / 0.5471 / 0.5465 seconds; metadata/model preparation takes 8.194 s and
compilation 0.318 s. Those separate costs are preserved explicitly.

`nvfp4-replay.json` similarly loads the exact canonical hex files from the
already validated NVFP4 RTL gate, with manifests specifying the original
configuration. It reproduces all three prior predictions exactly:

| Actual NVFP4 hardware program | RTL cycles | Direct hex-input LPN cycles |
| --- | ---: | ---: |
| Scaled | 190 | 179 |
| Activation replay and ACC | 477 | 467 |
| PAIR | 166 | 153 |

The inputs' program hashes equal the retained RTL hashes. Numerical output
equivalence is established by the original component gate, not recomputed or
claimed as a new full GLM run here.

## Full command-driven architecture study

`full-dse.json` contains four completed 48-kernel runs. All count exactly
**36,936,364,032 matrix dot operations** and **70,419,009 logical VPU FP32 FLOPs**
from resolved hardware instructions. The MXU arithmetic is quantized integer
dot products with FP32 epilogues; its dot TOPS is not native BF16 TFLOP/s.
Encoded K padding and loop iterations are included. Quantizer arithmetic,
matrix epilogues and special-function equivalents are excluded from the FP32
FLOP subtotal.

The known cached 128x1 control gives **8,142,924 cycles**, exactly its previous
prediction, versus **8,410,568 cycles in fully validated cached RTL**. The fresh
retained full-RTL audit reports −3.1822% total error and 4.7960% worst kernel.
This control has its original 2816-byte/cycle, 8-cycle behavioral memory
parameters; it is a regression reference, not an equal-memory comparison with
the new HBM proposals.

All three proposed organizations have 131,072 dot MAC sites, or nominal
393.216 integer dot TOPS at the **assumed** 1.5 GHz. The actual command encoding
uses D=64. They propose four bounded VPU result engines, a 128-command window,
572 bytes/cycle shared HBM (858 decimal GB/s), 120-cycle assumed response
latency, and existing quantizer/activation-streaming/physical-scale protocols.
They have no scale-cache timing extension for multiple matrix engines and
no wider shared scratchpad output port. No FPGA fit or 1.5-GHz Fmax is proven.

| Proposed engines x parallel output rows x columns | Predicted full cycles | Assumed hardware ms | Achieved dot TOPS | Native window seconds, sum over 48 kernels |
| --- | ---: | ---: | ---: | ---: |
| 4 x 128 x 4 | 18,162,384 | 12.108 | 3.051 | 247.52 |
| 1 x 512 x 4 | 19,181,161 | 12.787 | 2.888 | 256.78 |
| 4 x 256 x 2 | 18,128,591 | 12.086 | 3.056 | 245.60 |

These proposals demonstrate low utilization of added arithmetic on the actual
compiled stream. They are **not completed faster RTL designs** and should not
be promoted based on nominal compute capacity. Memory, quantizer/VPU service,
shared output throughput and the compiled dependency/issue pattern remain
architectural constraints. No result is extrapolated to GLM.

The cached control's summed native window time is **23.35 seconds**. The whole
four-point/192-kernel study took **380.07 seconds wall time**, including metadata
preparation and compilation, with four independent host workers. Every LPN
simulation is single threaded. Native-window sums were measured while those
workers ran, and are not a separately measured sequential end-to-end wall time.
These measurements do not support the historical 1–2-second GLM claim.

`full-dse-audit.json` independently reloads every canonical hardware program,
checks it against retained decoded words and actual RTL instruction counts,
verifies the source and native logs, checks identical operation counts across
architectures, and verifies the unchanged cached-control prediction. It does
not supply measured RTL cycles to the predictor or claim RTL validation for
the proposals.

Run source is frozen at `d2b89dc` in
`~/LPNDialect-hardware-command-replay` on icdslab2; audit source is `7234bab`
in `~/LPNDialect-hardware-command-audit`. Native logs, tables and binaries stay
under `~/lpn-runs/tpu-100x/hardware-command-scale-full`, and direct-input
controls under `~/lpn-runs/tpu-100x/hardware-command-control` and
`~/lpn-runs/glm47-flash/nvfp4-hardware-input`. Existing scheduler, guard synthesis
and heartbeat caching remain unchanged.
