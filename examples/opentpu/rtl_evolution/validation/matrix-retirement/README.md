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

This is a four-kernel diagnostic, not a completed 48-kernel validation.
The selected RTL total is 878,386 cycles, or 14.3047x fewer than the same
selected original kernels. A full changed-RTL replay is still running.
No empirical timing multiplier was added.

The supported replay driver independently rechecked these four retained RTL
outputs with the new timing details. `physical-pilot-replay.json` and
`physical-pilot-audit.json` retain the result and fresh audit. All inputs,
raw logs, instruction counts, sources, simulator and complete numerical
outputs passed. The corrected prediction over **all 48 original programs**
is **8,476,740 cycles**, or **15.6697x** fewer than original RTL; see
`physical-full-prediction.json`. This full prediction has not yet passed
the complete changed-RTL validation. Its 8,192 matrix MAC lanes and roughly
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

A small scale-line reuse buffer is a possible FPGA optimization. It has
not been implemented or measured, and these corrected predictions assume
no such cache. FPGA fit, clock frequency and vendor HBM integration remain
unproven. The 100x architecture goal has not been reached.
