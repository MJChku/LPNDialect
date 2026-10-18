# Independent matrix-engine proposal

The single-array implementation remains the validated RTL. This experiment
splits the same 8192 integer MAC lanes over 2/4/8/16 independent arrays, each
with private activation and weight FIFO storage. All arrays share DRAM and
one 64-bank MXU scratchpad output port. FPGA resource estimates multiply
compute, activation and FIFO storage by the engine count. No DSP packing is
assumed. The VPU keeps its separate existing scratchpad write port.

A batch reserves its aggregate bank write service at compute start (at its
first ready activation group for streaming execution). This enforces shared
throughput but is an approximation of output-wave arbitration; it is not a
cycle-accurate RTL implementation. Single-engine timing is unchanged.

The test submits sixteen independent 64x8 unit-scaled commands, each with
512 output words. At 64 banks they require 128 total busy port cycles, even
with sixteen compute engines and effectively unlimited DRAM bandwidth.
Native and Python runs agree for streaming and nonstreaming execution, both
with independent outputs and with WAW dependencies. An RTL replay rejects
multiple-engine timing until corresponding hardware exists.

The studies retain the original upstream 128-input/32-output programs and
do not split token positions, change inference scheduling, or edit the LPN
scheduler, guards or heartbeats. Predictions are architecture proposals;
FPGA mapping, routing, clock and numerical/cycle RTL validation are pending.
