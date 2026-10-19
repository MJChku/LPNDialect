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
