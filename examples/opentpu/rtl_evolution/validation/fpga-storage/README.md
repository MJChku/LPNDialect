# Compact physical storage and vocabulary cache placement

All 48 original upstream 128-input/32-output programs were checked, including
their unrolled loop commands, canonical instruction hashes and arguments.
`envelope.json` establishes maximum requirements of **8 ACT rows**, **64 ACT
blocks**, and **9 chunks per MM weight row**. Full QACT writes require all
64 blocks even though individual MMs consume at most nine. Reducing block
capacity to 32 was rejected by the analysis before any hardware claim.

The compact architecture keeps all 64 blocks, halves the physical ACT rows
from 16 to 8 and reduces each weight FIFO from 128 to 64 chunks. Its counted
base storage falls from 14,994,432 to **9,751,552 bytes**: **5 MiB reclaimed**.
Across all 48 programs LPN predicts 8,477,788 cycles, only **0.0124%** above
the 8,476,740-cycle control. A 32-chunk FIFO saves another 256 KiB but costs
about 0.58% in predicted cycles, so 64 is the preferred depth.

`boundary.json` numerically checks the actual smaller RTL allocation against
the original larger logical ISA configuration. Both complete DRAM and
scratchpad states match. Whole and partial QACT loads exercise physical row
7 and block 63, followed by two queued nine-block matrix commands. These are
boundary gates, not a full inference replay. The replay driver accepts
`--act-rows 8 --fifo-depth 64` to validate unchanged retained workload inputs
on this hardware configuration, with corresponding LPN storage and timing.
The new driver path was first checked at the original 16-row/128-chunk
capacities by reusing the four retained RTL pilot outputs. It reproduces
the corrected control cycles exactly and passes a fresh full-state/source/
input/binary audit; see `driver-control*.json`. The actual smaller-capacity
inference pilot is building/running separately, with no reused RTL outputs.

The corrected prefill profile is compute-heavy: matrix execution occupies
73.5% of elapsed cycles and VPU issue 54.5%, while the memory link occupies
21.6%. Decode instead has about 74.9% memory-link occupancy (see the sibling
physical-cache study). These are overlapping intervals, not additive costs.

The final vocabulary projection reads six contiguous weight matrices starting
at byte address **114,436,352**, each 8,192 rows by 576 int8 values. Its memory
traffic is more exposed than the earliest weight region. The bounded cache
study was repeated at this address, with cold state, per-kernel host-change
and write invalidation, and separately hashed per-row hit profiles.

| Compact architecture | Full predicted cycles | Predicted speedup vs original | Counted storage |
| --- | ---: | ---: | ---: |
| No weight cache | 8,477,788 | 15.6678x | 9,751,552 B |
| 24 MiB vocabulary cache | 8,199,603 | 16.1994x | 34,966,528 B |
| 28 MiB vocabulary cache | 8,164,115 | 16.2698x | 39,169,024 B |
| 32 MiB vocabulary cache | 8,164,115 | 16.2698x | 43,371,520 B |

The 28 MiB capacity covers the useful weight region; 32 MiB adds no hits or
cycle benefit. With compact ACT/FIFO storage, the 28 MiB proposal passes the
provisional V80 50%-SRAM capacity screen. It is **not implemented in RTL**;
these are forecasts, not hardware speedups or FPGA-fit proof. The scale-line
buffer is separate and is not assumed by these predictions. All traffic,
source, profile and native timing evidence is retained in the JSON files.
Neither this work nor the earlier study establishes the 100x goal.
