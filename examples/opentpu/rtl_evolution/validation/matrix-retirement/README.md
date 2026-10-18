# Physical scale traffic and matrix retirement

These opt-in LPN details describe the existing parallel RTL. They do not
change the instruction stream, tensor arithmetic, scheduler or hardware.
Historical timing options remain available to reproduce earlier reports.

The 128-engine, two-cycle FP-adder pilot failed the 10% per-kernel timing
target with the older timing equations: decode was underestimated by 13.4%.
The RTL itself already passed full DRAM and scratchpad comparisons.

`Timing(vector_command_queue=2, matrix_retirement=True,
physical_scale_lines=True)` models three structures explicitly:

* The existing two-entry VPU command acceptance FIFO.
* Two matrix command contexts with an ordered, single FP retirement head.
  The registered head switch costs two clocks; ACC also crosses the three
  read stages before its multiply/add stages. Integer work can overlap.
* A 32-byte memory transaction for **each** physical scale-port request.
  `otpu_mxu.sv` requests A on each advance, including repeated reads of the
  same line. `otpu_top.sv` charges 32 link bytes per accepted A request.
  Compact metadata storage size is not its physical traffic volume.

For traced kernel 16, the instruction-derived request count is **2,113,002**,
exactly the summed RTL `MXS0 a=` counters across all 128 engines. This is
**67,616,064 bytes**, versus the old storage-size proxy of 4,151,208 bytes.
The count includes RSC repeated reads and separate Sw reads where needed;
parallel RTL clears PAIR for packed formats. The trace was rerun with full
DRAM and scratchpad checks, identical canonical inputs and simulator hash.
See `scale-line-counts.json` and `check_scale_lines.py`.

| Kernel | RTL cycles | Corrected LPN | Relative error |
| --- | ---: | ---: | ---: |
| 0 | 314,414 | 302,421 | -3.8144% |
| 15 | 345,005 | 339,175 | -1.6898% |
| 16 | 109,349 | 108,415 | -0.8541% |
| 47 | 109,618 | 108,715 | -0.8238% |

This pilot table covers four kernels.
The selected RTL total is 878,386 cycles, or 14.3047x fewer than the same
selected original kernels. The completed full replay is recorded below.
No empirical timing multiplier was added.

The supported replay driver independently rechecked these four retained RTL
outputs with the new timing details. `physical-pilot-replay.json` and
`physical-pilot-audit.json` retain the result and fresh audit. All inputs,
raw logs, instruction counts, sources, simulator and complete numerical
outputs passed. The corrected prediction over **all 48 original programs**
is **8,476,740 cycles**, or **15.6697x** fewer than original RTL; see
`physical-full-prediction.json`. It now passes the complete changed-RTL
validation below. Its 8,192 matrix MAC lanes and roughly
14.3 MiB of accounted storage pass the provisional V80 capacity screen,
which is only a capacity estimate, not an implementation or FPGA-fit claim.

`physical-pilot-diagnostic.json` retains controls reproducing old cycles.
`retirement-only-pilot.json` isolates the retirement change, which alone
still fails 10%. Its source snapshot is reproducible by applying
`retirement-only.patch` to commit `61d6a9e`. Source hashes identify each
diagnostic independently. Four new native/Python tests and the existing
30 integration tests passed; their logs are retained here.

The `physical_128x1_add2` DSE candidate enables all three details. The replay
driver accepts `--vector-command-queue 2 --matrix-retirement
--physical-scale-lines`. Add `--reuse-rtl <completed-replay>` to recalculate
timing without rerunning RTL: it requires matching hardware source hashes,
source/parameter/flags cache key, binary hash, inputs and ROM tables, then
freshly checks the complete retained numerical outputs. Keep source and
output roots separate from earlier evidence.

A [small scale-line reuse buffer](../scale-cache/README.md) is now a separate
RTL prototype with numerical gates; its real-workload measurement is pending.
These corrected predictions assume no such cache. FPGA fit, clock frequency
and vendor HBM integration remain unproven. The 100x goal has not been reached.

## Complete inference validation

The unchanged upstream SmolLM2-135M **128-input/32-output** inference completed
all 48 kernels on the actual 128-engine RTL with two-cycle FP adders, streamed
ACT, lookup quantizer scale and the exact small-column reciprocal unit.

| Phase | Original RTL | Changed RTL | Cycle speedup | Corrected LPN |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 62,013,323 | 5,207,344 | 11.9088x | 5,002,660 |
| Decode | 70,814,940 | 3,501,478 | 20.2243x | 3,474,080 |
| Full inference | 132,828,263 | **8,708,822** | **15.2522x** | **8,476,740** |

Total LPN error is **-2.6649%**, and worst absolute kernel error is
**4.8019%**. Every kernel passes the 10% target. There are zero rollbacks.
The fresh audit checks all canonical input and decoded-program hashes,
arguments/instruction counts, complete DRAM and scratchpad dumps, raw logs,
source files, initialization tables and simulator hash. Full boundary-state
equivalence preserves all generated token IDs. No tensor prepass was repeated
and no inference schedule changed.

`128-add2-full-rtl.json` preserves the original hardware run, which completed
in 3,987 host seconds using 16 validation workers. Its frozen launcher recorded
the old timing equations, so that report's old LPN error still fails. Do not
use it as the corrected-model result. `128-add2-full-corrected.json` rechecks
the exact same RTL output files and binary with the new timing details;
`128-add2-full-audit.json` independently audits that complete result.
Validation workers execute separate HALT-drained kernels; hardware latency
is the sum of their cycle counts, not their parallel host wall time.

The scale cache, packed QST stores, compact ACT/FIFO capacities and pinned
weight-cache proposals are **not included** in this measured architecture.
Their gates and forecasts are separate. FPGA mapping, Fmax, a physical HBM
controller and the 100x target remain unproven.

