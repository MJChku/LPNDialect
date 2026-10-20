# Cache-aware LPN timing

`Timing(scale_cache_entries=16)` models the existing bounded private scale
cache. Zero preserves prior timing. Int8 requests are reconstructed from the
original resolved instructions; decoder transactions still occur on hits,
while only misses charge 32 bytes to the shared HBM port. The direct-mapped
cache has sixteen entries per physical row engine. Pending hits share fills;
slot-reader lifetimes derive from the configured response latency.

A separate pacing credit bounds replacement issue spacing. It leaves HBM
available to other components while a conflicting slot waits for its readers.
The model supports one array, unified physical traffic and one-wave batches.
Other weight formats are rejected by the opt-in model until their request
sequence is implemented. Defaults retain support for their existing models.

Coherence follows canonical instruction order; actual writes can overlap old
prefetches, so the exact miss count is approximate. **No RTL cycle count or
cache counter enters the predictor.** Its only inputs are original commands
and explicit cache/port parameters. The scheduler, guards and heartbeats are
unchanged.

Four real retained kernels passed a fresh full-memory/source/binary/log audit
before prediction. Every original scale-request count matches exactly:

| Kernel | Phase | Cached RTL cycles | LPN cycles | Error |
| --- | --- | ---: | ---: | ---: |
| 0 | Prefill | 314,371 | 302,421 | −3.8012% |
| 15 | Prefill | 343,399 | 334,735 | −2.5230% |
| 16 | Decode | 99,737 | 98,122 | −1.6193% |
| 47 | Decode | 100,468 | 98,422 | −2.0365% |

Miss counts are 1.3–3.8% below RTL, consistent with the asynchronous coherence
limitation. These were the first four kernels. FPGA mapping/clock remain
unproven; the subsequent full-inference result is recorded below.

## Completed 128-row cached RTL full inference

`full-rtl.json` and `full-audit.json` preserve all 48 kernels of the same
normal upstream SmolLM2-135M 128-input / 32-output inference. The completed
cached RTL takes **8,410,568 cycles**, versus original RTL's 132,828,263:
**15.7930x fewer cycles**. Native LPN predicts **8,142,924 cycles**, −3.1822%
overall and **4.7960% worst kernel**. Every full DRAM/TMEM state matches; the
fresh audit verifies input/program/binary/source hashes, raw logs, instruction
counts and preservation of the upstream schedule/generated IDs.

The interrupted original run was confirmed dead before recovery. Its 32
completed simulations were freshly audited and reused; only 16 missing RTL
simulations ran in the resumed directory. Completed evidence was preserved.
This is the 128x1 physical point, which exceeds the provisional FPGA integer
DSP reserve budget; its timing validation does not establish FPGA fit.

Four focused tests verify pending hits, invalidation, exact int8 request
sequences, bounded replacement and independent HBM/decoder service. The
combined weight+scale-cache test consumes no HBM credit for fully cached
reads; Python and native timing agree. All 30 prior integration tests pass.

Resource accounting now includes column-end integer DSP cells separately
from dot cells and lists matrix FP multiply sites. Packing halves dot cells;
it does not halve column ends or FP pipelines. The old 128x1 array exceeds
the provisional 80%-DSP integer budget after including its column ends.
The paired 96x2 point has 6144 dot and 1728 column-end cells, plus 192 FP32
matrix multiply sites and the VPU/quantizer. It passes that source-level
capacity screen, but DSP58 mapping, whole-design resource use, placement and
Fmax are unproven. Its sixteen-entry cache has 49,152 data bytes plus 10,176
counted control bytes. The 128-core equivalent totals 77.25 KiB.
