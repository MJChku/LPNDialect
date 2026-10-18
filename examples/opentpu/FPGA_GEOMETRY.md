# FPGA proposal: reuse weights across matrix columns

The current parallel RTL assigns the matrix DSPs to independent weight rows.
This study holds the total at 8,192 integer MAC lanes and redistributes them
between weight rows and simultaneous activation rows:

| Geometry | Weight rows | Activation columns | MAC lanes, D = 64 |
| --- | ---: | ---: | ---: |
| 128 × 1 | 128 | 1 | 8,192 |
| 64 × 2 | 64 | 2 | 8,192 |
| 32 × 4 | 32 | 4 | 8,192 |
| 16 × 8 | 16 | 8 | 8,192 |

Each matrix instruction already specifies its activation-row count. Processing
several of those rows together can reuse the fetched weights. This changes
hardware execution, not the upstream inference schedule or instruction stream.
Single-row decode instructions can leave activation columns idle, so a prefill
gain need not improve complete inference.

The study uses the existing LPN matrix-column model, the small reciprocal unit,
quantizer lookup, interleaved reductions and two-stage adder. ACT streaming is
disabled for **all four** cases: the implemented streaming path is restricted
to one matrix column. Its 128 × 1 point is the matching control. Memory service
remains 2,816 bytes per clock under the same provisional FPGA budget.

The multi-column parallel wrapper is a proposal: the currently implemented
`otpu_mxu_parallel` instantiates one column per core. Weight broadcast, activation
ports, scratchpad arbitration and any restored streaming path need RTL changes
and numerical/timing validation. Raw SRAM/DSP capacity checks do not establish
physical fit, Fmax or HBM-controller behavior.

All points consume all 48 original retained programs, with the decoded words
checked against canonical RTL inputs. Reproduce with `tpu_dse --indices all`
and architecture names `proposal_geometry_128x1_add2`,
`proposal_geometry_64x2_add2`, `proposal_geometry_32x4_add2` and
`proposal_geometry_16x8_add2`. These are LPN forecasts until implemented and
measured. The 100× target remains open.

## Completed full-workload forecasts

| Geometry | Prefill cycles | Decode cycles | Total cycles | Reduction vs original RTL |
| --- | ---: | ---: | ---: | ---: |
| 128 × 1 control | 5,243,213 | 3,056,832 | 8,300,045 | 16.003× |
| 64 × 2 | 5,187,748 | 3,155,072 | 8,342,820 | 15.921× |
| 32 × 4 | 5,134,766 | 3,556,612 | 8,691,378 | 15.283× |
| 16 × 8 | 5,523,559 | 5,412,704 | 10,936,263 | 12.146× |

Every point uses identical upstream inputs, passes the provisional capacity
screen and completes all 48 LPN kernels. The 32 × 4 candidate saves only 2.07%
of prefill cycles and loses substantially in decode. Even optimistically
combining its prefill total with the 128 × 1 decode total saves just 1.31% of
control cycles, before accounting for any hardware to switch geometry. This
does not justify a reconfigurable matrix implementation as the next change.
The result points back toward dependent vector/quantizer/matrix handoffs rather
than more arithmetic parallelism alone.

Evidence and source hashes are in
[validation/fpga-geometry](rtl_evolution/validation/fpga-geometry).
