# Wider elementwise vector issue proposal

`Timing(vector_issue_groups=2/4/8)` proposes 128/256/512 elementwise lanes on
the existing 64-lane base. Independent elementwise groups issue together;
sum, square-sum, dot, min/max and argmax reductions retain their original
64-lane timing and ordered accumulation. Scalar reciprocal lookup retains
its existing narrow path. Original programs, dependencies, inference order,
event scheduling, guards and heartbeats are unchanged.

This is a timing proposal with **no matching RTL implementation**; replay
rejects the opt-in parameter. It assumes two scratchpad reads and one write
per elementwise lane per clock. A 256-lane proposal therefore requires
512 read words and 256 write words per cycle, plus banking/address/control
changes. It lists 1088 vector FP multiply sites versus 272 in the control,
conservatively including replicated functions that reductions do not use.
These are sites rather than mapped DSP totals; source-level storage bounds
omit bank-granularity waste, new pipeline state and physical routing.

Three focused tests verify elementwise issue/tails, unchanged reduction
latency, dependency chains, Python/native equality and rejection as existing
RTL. Four scale-cache regression tests and all 30 general integration tests
pass with the default issue width (`cache-regression.log`, `general-regression.log`).

The real-kernel pilot (`pilot.json`, kernels 0/15/16/47) predicts the following
cycle speedups for paired 96x2 with a sixteen-entry scale cache:

| Elementwise lanes | Pilot extrapolation |
| --- | ---: |
| 64 control | 16.9376x |
| 128 | 18.4793x |
| 256 | 19.3917x |
| 512 | 19.5218x |

Intermediate attention positions are extrapolated. These do not supersede
the full validated 15.4406x RTL result, establish FPGA fit or meet 100x.
The small benefit from 512 lanes does not justify prioritizing it.

The **complete 48-kernel** native prediction (`full-prediction.json`) is now
finished with identical canonical programs and upstream inference order:

| Elementwise lanes | Predicted cycles | Speedup versus original RTL |
| --- | ---: | ---: |
| 64 control | 8,001,603 | 16.6002x |
| 128 | 7,273,465 | 18.2620x |
| 256 | 6,896,771 | 19.2595x |

The wider proposals reduce cycles by 9.10% and 13.81% versus their own
cache-enabled control. This is full-workload LPN evidence, still **not RTL
validation** of a wider datapath. Every reported prediction excludes neural
tensor arithmetic and uses the unchanged resolved instruction stream.
