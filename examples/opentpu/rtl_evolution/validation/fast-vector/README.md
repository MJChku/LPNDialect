# Dedicated fill/copy engine proposal

This opt-in LPN proposal separates `V_FILL` and `V_COPY` from a single original
FP engine. Both engines have bounded private result storage and share one
scratchpad result write port. The auxiliary engine needs one private read copy,
but no additional FP arithmetic or reciprocal ROM. The model retains data
dependencies and all existing FP tap-order constraints within each engine.

Three new native/Python tests verify independent overlap, true read/write
hazards, bounded capacity and shared output bandwidth. The four previous
vector-pool tests and 30 general integration tests also pass. The RTL replay
validator explicitly rejects this option because it has no implementation.

All 48 unchanged normal-inference programs were evaluated for each point:

| Model | Window | Predicted cycles | Speedup vs original RTL |
| --- | ---: | ---: | ---: |
| Current compact architecture, original VPU | 32 | 8,477,788 | 15.6678x |
| One FP engine with buffered results | 32 | 8,809,178 | 15.0784x |
| One FP engine with buffered results | 128 | 8,672,224 | 15.3165x |
| FP plus dedicated fill/copy engine | 32 | 8,468,768 | 15.6845x |
| FP plus dedicated fill/copy engine | 128 | 8,468,436 | 15.6851x |

The actual control reproduces its earlier complete prediction exactly. The
auxiliary engine recovers much of the buffered model's extra delay, but the
overall improvement over the current design is only about **0.11%**. This
particular proposal is not selected for RTL implementation. The conservative
buffered model drains each full command after the original VPU latency; a
different implementation that streams results directly would need its own
storage/backpressure model and validation.

Each engine has 1,024 result chunks (256 KiB). The auxiliary engine adds one
256 KiB scratchpad read copy and another 256 KiB result store. The counted
FP32 multiply sites remain 272 for the single FP engine. FPGA mapping, routing
and Fmax are unverified. Window 128 also requires a larger hardware scoreboard;
its comparator/selection logic is not covered by the storage estimate.

These are timing proposals, not measured hardware speedups. The event-driven
scheduler, heartbeat and guard synthesis are unchanged. `full-predictions.json`
retains the 240 results, source identities and explicit resource estimates.
