# Bounded vector concurrency experiments

These are LPN architecture proposals, with no vector-pool or compatible-select
RTL implementation yet. They preserve the upstream programs, arguments and
memory footprints. They do not establish an inference speedup or FPGA fit.

A pool engine reserves a finite, private output store for a complete command
before it issues. Every engine has private operand read ports and shares one
result write port with the pool. The timing model charges a result drain after
its original VPU latency. This is deliberately conservative; hardware would
need the reserved storage and address generators. Each extra engine also adds
two TMEM input copies to the resource lower bound. DSP mapping, pipeline state,
control and fanout remain outside that capacity proof.

The compatible-select alternative chooses the oldest dependency-ready command
whose rank fits the current pipeline and whose result fits the available
store. It may skip an older instruction that cannot currently start. It retains
all address hazards, reduction isolation, rank ordering and bounded credits.
The current event scheduler and guard synthesis are unchanged.

Four original inference kernels (0, 15, 16, 47) were evaluated for each proposal.
The following are pilot extrapolations, not complete 48-kernel predictions:

| Proposal | Extrapolated cycle reduction vs original RTL |
| --- | ---: |
| 128 matrix engines, bounded single VPU | 12.653x |
| 128 matrix engines, two VPUs | 15.695x |
| 128 matrix engines, four VPUs | 16.026x |
| 128 matrix engines, eight VPUs | 16.085x |
| Compatible selection, existing single VPU | 13.551x |
| Compatible selection, two VPUs | 15.814x |
| Two VPUs, 512-slot command window | 15.968x |
| Four VPUs, shorter adder | 17.520x |
| Oversized ASIC diagnostic: 1024 matrix engines, 16 VPUs, 512 slots | 24.823x |

The plateau argues against extensive VPU duplication or a much larger command
window as the next FPGA change. The oversized ASIC case uses an unimplemented
65,536-byte/cycle link and short arithmetic stages; it is not an FPGA candidate.
The results direct the next experiment toward scalar math latency, described
in [RECIP_ROM.md](RECIP_ROM.md). The 100x target is still unachieved.

All 30 existing integration tests pass. Four new tests check native/Python
agreement, independent long/short operations, finite result-store backpressure,
a shared output bandwidth bound, and compatible selection. Reports and source
fingerprints are in [validation/vector-proposals](rtl_evolution/validation/vector-proposals).
