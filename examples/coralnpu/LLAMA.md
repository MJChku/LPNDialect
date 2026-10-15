# Llama 3.2 3B timing workload

`llama.py` constructs the complete model's operator dataflow for batch-one
generation: embeddings, 28 decoder layers, Q/K/V projections, RoPE, grouped
query attention, KV appends, masked attention/softmax, output projections,
RMS normalization, SwiGLU, residuals, the tied LM head, and argmax.
Dimensions come from Meta's [pinned architecture definitions](https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/sku_list.py):
3072 hidden features, 8192 intermediate features, 24 query heads, 8 KV heads,
128 features per head, and 128256 vocabulary entries. Tied embeddings give
3,212,749,824 parameters.

This runs a timing workload without downloading weights or executing numerical
inference. Matrix kernels use native BF16 inputs with FP32 accumulation.
The first token comes from prefill logits; `--generate-tokens 4` subsequently
executes three decode passes, extending the KV cache at each pass.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m examples.coralnpu.llama \
  --prompt-tokens 128 --generate-tokens 4 \
  --memory-latency 64 --memory-bandwidth 16 \
  --emit /tmp/llama-3b.mlir \
  --lpn-compiler build/tools/lpn-compile/lpn-compile \
  --output /tmp/llama-3b.json
```

Time is in model cycles; memory bandwidth is bytes/cycle. Defaults are
experiment parameters, not measured DRAM specifications for a particular chip.
`--emit` alone writes MLIR. Full-size execution uses the native compiler;
the Python reference is used on smaller shapes for regression checks.

## Synchronization and speed

Every operand edge is a place. Producers emit readiness tokens for each
consumer, which takes its inputs before issuing. Q/K/V and gate/up have
independent edges; attention waits separately for keys and values. Shared
matrix/vector engine tokens, a DRAM bandwidth token, outstanding-stream credits,
and finite staging credits control contention. Completion delays enable
downstream operators. There is no global time-stepping loop.

Identical decoder layers reuse the same graph. Feedback tokens carry the layer
index and renew both the normalization and residual input edges. A new decode
pass resets the index. This changes graph size without changing scheduling.
The expanded graph is available through `LlamaWorkload(..., fold_layers=False)`
for checks; the compact and expanded versions agree on phase completion times.

The recorded 128-prompt/four-token run on `icdslab2` executes 2,144 logical
operators through 92 templates. Both graph forms execute 6,436 transition
bodies and finish at 7,191,927,436 model cycles. Native wall time fell from
7.16 seconds to 0.160 seconds, and native compilation from 16.42 seconds to
0.829 seconds. Both runs had zero rollbacks. These are individual runs,
including process startup and observation output, not a statistical speedup
measurement against Verilator. Reports are in `validation/llama`.

## Accuracy target and outstanding validation

The target is at most 5% relative timing error on measured workload windows.
Full Llama execution has not been measured in Verilator, so reports explicitly
set `accuracy.status` to `unmeasured` and
`end_to_end_verilator_validated` to `false`.

The matrix baseline includes 64-command BF16 streams for both a one-row decode
shape and a full prefill tile. Their array windows match the native LPN and
Python reference exactly at 260 and 264 cycles, including accumulator hazards.
These kernels check architectural results in the full VME RTL. They establish
array behavior for those streams, not complete Llama accuracy.

The compact operator model currently assumes:

- M-major, output-stationary tiling, with explicit repeated input/weight panel
  traffic. Whole weight matrices are not cached in local SRAM.
- A stream reserves DRAM bandwidth until its payload drains. Compute starts
  after the first panel, with its completion bounded by the remaining stream.
  Staging credits represent in-flight streams, not whole tensors in SRAM.
- Grouped attention reuses streamed K/V panels across query heads and
  materializes masked score tensors. Vector pass counts and operator-level
  overlap still need calibration against an actual mapped implementation.

These assumptions are included in each report. Different tiling or fusion
changes memory traffic and timing; choose the mapping before judging accuracy.
The BF16 checkpoint exceeds the core's 32-bit address space, so a real
whole-model RTL workload also needs a host/windowed weight-streaming scheme.
That requires a defined software mapping, not an AXI beat-level LPN model.

Use Verilator for representative GEMM/GEMV tiles, vector operators, memory
stalls, and small decoder-block fragments. Compare prefill and individual
decode windows separately on held-out cases; cumulative timing can hide a
bad decode estimate. The scalar/RVV validation CLI fails when its configured
5% threshold is exceeded, and matrix validation currently requires exact
array-window agreement.
