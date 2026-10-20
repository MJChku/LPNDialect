# GLM-4.7-Flash: NVFP4 capacity and first MLA/MoE timing screen

**Scope correction:** the user requires hardware commands as the LPN input.
The historical tensor-DAG studies below do not satisfy that requirement and
are retained as experimental evidence only. Their 1–2-second simulation time
does not establish GLM execution on the actual openTPU LPN. No complete GLM
hardware instruction stream exists yet. Use the actual
[hardware command path](../../HARDWARE_COMMANDS.md) for accelerator modeling.

The subsequent [v5e-scale LPN study](v5e-scale/README.md) adds shared matrix
resources, padded systolic-array service and useful FLOP accounting. Its ten
points are complete and audited; the timings remain architecture estimates.

This target is the real **zai-org/GLM-4.7-Flash** configuration, pinned at
`7dd20894a642a0aa287e9827cb1a1f7f91386b67`. It has 47 main-model layers,
2048 hidden dimensions, 20 attention heads, query rank 768, KV rank 512,
192 non-rotary plus 64 rotary query/key dimensions and 256 value dimensions.
The first layer has a dense MLP; the other 46 have 64 routed experts,
four selected experts per token and one shared expert. The optional MTP
layer in the checkpoint is excluded explicitly.

This is **not a completed GLM implementation on existing openTPU RTL**.
The first frontend and tensor-command architecture screen are executable;
the FeSens ISA adapter, compressed MLA cache, routing implementation,
physical memory mapping and full numerical/timing validation remain open.

## Capacity and arithmetic

The independent inventory matches every main-model parameter shape in a
meta instance of upstream Transformers 5.12.0: **29,943,390,976 parameters**.
Proposed storage quantizes matrix weights, embeddings and the LM head to
E2M1, with one E4M3 scale per 16 elements and one FP32 global tensor scale.
Router parameters and norms remain FP32. Including 64-byte alignment, the
weight layout is **16,865,359,940 bytes (15.707 GiB)**. Actual FeSens scale-row
repacking has separate row headers and is not included in this proposed layout.
The checkpoint index's `total_size` is not used to infer parameter or byte counts.

At BF16 cache precision, compressed MLA needs 54,144 bytes per cached token
per sequence over all layers. The ordinary expanded K/V reference needs
962,560 bytes. At 1,032 tokens that is 55,876,608 versus 993,361,920 bytes;
at 128K it is 7,096,762,368 versus 126,164,664,320 bytes. These exclude
temporary workspaces. **The traced reference uses expanded K/V**, matching
the pinned upstream code; compressed capacity is an architectural option,
not an unimplemented optimization silently applied to timing.

Weights alone require at least **34 address bits**, exceeding the existing
32-bit FeSens whole-image interface. We do not wrap addresses. A wider HBM
interface or explicit bank/window translation is required for the real device.
`capacity.json` contains the complete immutable layout and necessary service
bounds, not latency predictions. Packed weights do not automatically double
MAC throughput or establish numerical quantization accuracy.

## Standard upstream inference and explicit routing

The unmodified upstream `model.generate` owns **1024-input / 8-output**
scheduling: one 1024-row prefill call followed by seven one-row decode calls.
Every call visits all 47 layers. Torch executes the real model implementation
on meta tensors, without allocating trained weights or computing tensor values.
Placeholder logits enable a fixed-length planning run; token IDs are not
model predictions. No custom token-by-token prefill loop is introduced.

Expert routing needs numerical values. This screen therefore uses two
explicit **sensitivity annotations**, not fabricated real routing:
balanced expert usage and a concentrated four-expert hotspot. External
per-call/per-layer IDs can be supplied through `--routing-json`. The tensor
frontend records actual expert batch shapes for those IDs. Router-readiness
and previous-logit completion dependencies remain in the graph.
The omitted sigmoid/group-selection/top-k arithmetic has an explicit,
uncalibrated routing-cost annotation.

Fresh audits independently verify all linear and attention MACs against
configuration formulas, all DAG edges, every expert's router dependency,
every decode call's prior-logit dependency, and source/artifact hashes.
Both scenarios have **3,873,905,156,096 matrix MACs**. They differ in expert
batching and weight-transfer reuse. These are real model shapes and upstream
control, with annotated routing; they are **not numerical/text validation**.
The trace/frontend/timing source is frozen at **194cf8b** on the remote run.
Negative audit tests also reject missing router readiness, missing previous
logits and a forged matrix MAC count (`negative-audit-tests.json`).

## First LPN screen

`examples/tensor_dag.py` consumes the external graph and contains no GLM or
inference scheduling. Its seven transitions model dispatch, three execution
paths, shared HBM and completion. The usual event scheduler, guard synthesis
and heartbeat cache are unchanged. The native runtime completes each case
with zero rollbacks in about 1.2–2.0 seconds, after trace preparation/compilation.

All points use **1.5 GHz and 858.9934592 GB/s HBM**, or about 573 bytes/cycle.
The base proposes 12,288 NVFP4-weight MACs/cycle, 1024 BF16 attention MACs/cycle,
256 FP32 MACs/cycle and 256 vector elements/cycle. The NVFP4 and BF16 paths
are independent hypothetical datapaths, not existing RTL capability. The
fourfold and sixteenfold balanced points increase both matrix paths; vector
and FP32 budgets stay fixed. No point establishes FPGA resource fit.

| Routing sensitivity | Base | Weight matrix 4x | Both matrix paths 4x | Both matrix paths 16x |
| --- | ---: | ---: | ---: | ---: |
| Balanced | 914,523,811 cycles | 768,900,528 | 399,277,488 | 290,902,613 |
| Four-expert hotspot | 899,075,677 cycles | 698,327,926 | 328,704,886 | 204,375,785 |

For balanced routing, these are **1.19x / 2.29x / 3.14x** relative improvements
on the same graph. They are not comparisons against original openTPU or a
commercial TPU/GPU, and are not validated latency predictions.

Every intermediate read/write spills to HBM in this initial reference:
140,526,164,348 bytes in the balanced case. No persistent weight cache or
SRAM reuse is invented. Aggregate compute/transfer service overlaps
optimistically per operation; detailed tiling, scale decode, FP recurrence,
bank conflicts, refills and out-of-order activation-allocation capacity are
not calibrated. Each execution path owns one 64-KiB tile-buffer credit,
but detailed tile service still requires a real implementation. **Do not
claim the usual 10% RTL accuracy target for this new screen.**

Five target tests pass, including complete upstream meta parameter-shape
comparison and independent NVFP4 import/dequantization equality. Four DAG
tests pass in Python and compiled native execution, covering dependencies,
shared HBM contention, finite command windows and completion-flag wrap.

## Reproduce

Use a separate environment with NumPy, Torch and `transformers==5.12.0`.
The recorded run uses Torch 2.14.1+cpu. Existing accelerator environments need
no package updates. No checkpoint weights or tokenizer download is needed.

```sh
PYTHONPATH=. python -m examples.opentpu.glm47 --output /tmp/glm-capacity.json
PYTHONPATH=. python test/test_opentpu_glm47.py --opentpu-root ../FeSens-openTPU
PYTHONPATH=. python test/test_tensor_dag.py --tool build/tools/lpn-compile/lpn-compile
PYTHONPATH=. python -m examples.opentpu.glm47_screen \
  --compiler build/tools/lpn-compile/lpn-compile \
  --prompt-tokens 1024 --output-tokens 8 --output /tmp/glm-screen
```

The separate NVFP4 matrix gate imports actual E2M1/E4M3 test tensors into
the packed 128-deep RTL, with eight-term DSP chains. Its result is tracked
separately; a component gate must not be described as full GLM execution.

That component gate and a fresh audit are now **complete**. All full DRAM
and TMEM outputs match independently rerun ISA states. The actual instructions
select WNV, and new native LPN predictions match the retained predictions:

| NVFP4 component case | RTL cycles | Native LPN | Timing error |
| --- | ---: | ---: | ---: |
| Scaled 5x3 | 190 | 179 | −5.79% |
| Activation replay and ACC, 9x8 | 477 | 467 | −2.10% |
| PAIR, 7x2 | 166 | 153 | −7.83% |

The format-import test independently verifies E2M1 codes and E4M3 scales
against exact dequantized values. The RTL gate verifies the device's
repacked format, including per-row metadata. This establishes small-component
format and timing evidence; it does not establish full-model quantization
accuracy, a compressed MLA backend, a GLM ISA adapter or FPGA placement.
