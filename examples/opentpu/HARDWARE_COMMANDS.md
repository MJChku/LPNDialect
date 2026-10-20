# Actual openTPU LPN: hardware commands as input

The supported input is the accelerator's executable program: `input.json`
and `prog_N.hex`, with eight 32-bit words per instruction. These are the same
files the Verilator harness consumes. The LPN does not take PyTorch operations,
tensor shapes or a model-specific inference schedule as its command interface.

The functional frontend resolves real scalar control, addresses and
data-dependent instruction paths. The existing `CompactOpenTPU` then models
the command window, sequencer, DMA, quantizer, activation buffers, weight
prefetch FIFO, matrix pipelines/retirement, VPU and shared memory ports.
Instruction memory footprints establish data dependencies; LPN resources
establish contention and event timing. Scheduler, guard synthesis and
heartbeat behavior are unchanged.

`hardware_replay.py` is a direct entry point to this existing component model.
It reads the canonical encoded hardware files, resolves static scalar control,
and does not load weight images or perform numerical matrix computations.
It rejects tensor graphs, memory beyond the ISA's 32-bit address space,
unsupported control such as data-dependent RLD, malformed words, and programs
that exceed the instruction memory. Numerical/control resolution remains in
`workload.execute` when static resolution is insufficient; no routing values
are invented to bypass that step.

```sh
PYTHONPATH=. python -m examples.opentpu.hardware_replay \
  --input /path/to/retained/rtl/kernel \
  --opentpu-root ../FeSens-openTPU-packed-mxu \
  --compiler /path/to/heartbeat-enabled/lpn-compile \
  --clock-mhz 200 --runs 3 --output /tmp/hardware-replay
```

Use `--timing timing.json` for explicit `Timing` parameters. Where the RTL
microarchitecture overrides its logical configuration, specify
`--physical-matrix-columns` and `--physical-vector-lanes` explicitly. The
report preserves both the original input configuration and physical overrides.
New parameters are architecture proposals until the corresponding RTL passes
validation. No recorded RTL cycle count is used as an LPN timing input.

For the existing MLIR-only emission entry point, `examples.opentpu.run` also
accepts `--timing-only`: the input remains the canonical hardware program.

## Operation accounting

`CompactOpenTPU.statistics()['hardware_instructions']` counts the resolved
hardware instructions, including hardware-loop iterations:

* An MM with N output channels, M activation rows and KB blocks issues
  `N * M * KB * D` dot MACs, or twice that many dot operations. Encoded K padding
  is included. PAIR changes issue service, not the mathematical work count.
* Weight formats are counted separately from the actual MM flags. The existing
  openTPU MXU computes quantized integer dots with FP32 epilogues. Its dot TOPS
  must not be labeled native BF16 TFLOP/s.
* VPU ADD/SUB/MUL and mathematical sum/dot/sum-of-squares operations are counted
  as logical FP32 FLOPs. Special-function evaluations and function elements
  are reported separately; physical reduction padding is excluded.
* Matrix scaling/accumulation epilogues, quantizer arithmetic and unclassified
  vector functions are not included in that FP32 FLOP subtotal. This is not a
  whole-chip physical FLOP counter.

The replay report divides instruction-derived work by modeled hardware time
for achieved dot TOPS and logical VPU GFLOP/s. Host simulation time, preparation
and compilation are separate fields. The DSE path uses the same counters.

## Validation and current GLM limit

The new input test checks that a canonical hex program with scalar loops and
NVFP4/PAIR instructions resolves to exactly the same PCs, encoded instructions,
arithmetic counters and native LPN cycles as independent numerical ISA
execution. Negative tests reject tensor records, invalid instruction words,
manifest/IMEM mismatches, unresolved dynamic control and wide-address claims.

GLM-4.7-Flash has no complete hardware program yet. The earlier tensor-graph
experiment is excluded from this path. Building real GLM support requires a
compiler/backend that emits hardware instructions for MLA/MoE, real functional
control resolution, and a memory interface that can address or stream the
model. Existing NVFP4 matrix component validation does not complete those
steps. Real retained SmolLM2 inference programs remain the full-model workload
for command-driven architecture experiments until that compiler path exists.

The [completed hardware-input replay and full architecture study](rtl_evolution/validation/hardware-commands/README.md)
preserve the direct hex-file controls, 33 passing tests, all 48-kernel runs,
operation counts and the fresh hardware-input audit. No GLM timing is inferred
from those results.
