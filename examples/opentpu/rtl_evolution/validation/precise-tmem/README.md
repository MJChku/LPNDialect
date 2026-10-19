# Strided scratchpad dependency proposal

The original sequencer stores bounding intervals for each instruction. For
example, three rows of eight words at stride 32 produce one range [0,72).
Another tile at column offset eight overlaps that hull despite sharing no
words. `precise_tmem.py` expands and merges actual row intervals for MM, VOP,
QACT and QST; DRAM beat and ACT dependencies retain the original rules.

This changes dependency metadata only. All original commands, numerical
operations and upstream inference scheduling remain unchanged. It is opt-in
in DSE and **has no matching RTL scoreboard implementation yet**. It must
not be used to claim accuracy against the original conservative scoreboard.
Single-engine defaults and the guard/heartbeat scheduler are unchanged.

Independent ISA instrumentation checks actual tget/tput addresses for all
vector functions and broadcast modes, OUTER flag variants, quantization,
and matrix accumulation/max/scale cases. Additional tests retain RAW/WAR/WAW
hazards, allow a disjoint strided pair to overlap, and check Python/native
cycle agreement for both disjoint and dependent cases. Refined access ranges
are checked against the original access hulls (including RMW writes).

The FPGA proposals retain 200 MHz and 2816 B/cycle assumptions. Existing
resource screens remain provisional: no place-and-route or Fmax is claimed.
Separate larger ASIC cluster points are labeled as such and have no RTL.

All three tests pass, including native/Python agreement and the expanded
OUTER flag variants. The full 48-kernel sweep completed with zero failures:

| Proposal | Full predicted cycles | Speedup versus original RTL |
| --- | ---: | ---: |
| proposal_precise_128x1_v1_w32 | 8,477,788 | 15.6678x |
| proposal_precise_96x2_v1_w32 | 8,341,033 | 15.9247x |
| proposal_precise_64x1_v4_w128 | 8,754,695 | 15.1722x |
| proposal_precise_64x1_v8_w128 | 8,685,943 | 15.2923x |
| proposal_window_64_128x1 | 8,462,252 | 15.6966x |
| proposal_window_128_128x1 | 8,396,400 | 15.8197x |
| proposal_window_256_128x1 | 8,340,619 | 15.9255x |
| physical_compact_act_128x1_add2 | 8,477,788 | 15.6678x |

Exact row ranges produce **no cycle change** for either one-array geometry.
The control reproduces its earlier prediction exactly. The more expensive
four/eight-vector-engine proposals with 64 matrix rows are slower than that
control. Increasing the window alone to 256 saves about 1.62%, with no RTL
implementation/fit proof. No precise-scoreboard RTL is justified by these
results on this workload.

The separate four-kernel ASIC pilot reaches at most 31.09x extrapolated with
sixteen independent 128x8 arrays and sixteen vector engines. Precise ranges
improve that pilot slightly. This huge proposal is not an FPGA candidate,
not all 48 kernels, not RTL validated, and still does not establish 100x.

`resource-demand-pilot.json` independently accounts the model's required
service. DRAM, scale decode, quantizer, vector and matrix retirement totals
match the existing observed transaction holds exactly on kernels 0 and 16.
Matrix compute excludes time waiting for ACT input, explaining why its service
floor is lower than observed credit occupancy. These are bounds for current
units/ports, not limits on a redesigned accelerator.

The full 48-kernel service accounting is also complete
(`resource-demand-full.json`). With the existing single-unit designs, the
sum of the per-kernel resource floors is **5,407,545 cycles**, corresponding
to a **24.5635x** ceiling even if all additional waits disappear. The vector
unit requires 3,772,939 issue cycles, the quantizer 2,087,004, and matrix
compute alone (excluding ACT waits) 2,845,978. These demands overlap.

The unchanged programs request **6,212,192,256 weight bytes** and
**3,053,011,968 scale bytes**. At 2816 B/cycle, weight traffic alone bounds
speedup to **60.2113x** without caching or compression, even with free scales
and compute. For a 100x target at that bandwidth, at least 39.8% of those
weight bytes must avoid the link before allowing for any other traffic.
The source-level FPGA experiments therefore need coordinated changes to
memory reuse and vector/quantizer throughput; array enlargement alone cannot
satisfy that target with the present unit/port schedule. This is not a claim
that 100x is universally impossible on a different accelerator.
