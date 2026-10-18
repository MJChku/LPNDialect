# FPGA proposal: wider exact reciprocal

This is a timing and resource proposal, without a corresponding wide-reciprocal
VPU RTL implementation. The [small-row RTL variant](SMALL_RECIP.md) remains a
separate candidate. The new proposal tests whether accelerating the wide
reciprocal commands is worth replicating the exact tables.

`Timing(reciprocal_lookup_lanes=2|4|8|16)` models independently addressed lookup
lanes. Each pair shares one dual-port table copy and the VPU's common enable.
Every lookup lane also needs its predictor multiplier and exponent handling.
Inputs still use the existing 16 composite operand lanes; result masks, retirement
and writeback must be implemented and checked before RTL timing is claimed.

For lookup width `w`, the proposed issue span is `rows * ceil(cols / w)` clocks.
The hardware path would be selected only when its added issues, relative to
the original 16-wide reciprocal path, cost no more than the pipeline clocks
saved. Unprofitable shapes fall back. Latency assumes padding the existing
four-clock exact lookup to the same early retirement tap used by the small-row
variant; this assumption has not been validated for multiple lanes.

| Lookup lanes | Dual-port table copies | Raw table bytes |
| --- | ---: | ---: |
| 2 | 1 | 2,123,264 |
| 4 | 2 | 4,246,528 |
| 8 | 4 | 8,493,056 |
| 16 | 8 | 16,986,112 |

These replace the small-row VPU table rather than adding to it. The quantizer
keeps its separate 2,123,264-byte copy. Capacity screening also includes matrix
activation/FIFO and scratchpad storage, under the same provisional V80 budgets.
It is a storage lower bound, not a physical mapping or clock-frequency result.
RTL replay rejects this proposal flag so it cannot be silently compared against
the earlier single-lane hardware.

The model tests cover profitable geometry, bulk fallback, invalid configurations,
dual-port replication cost, rejection by the RTL replay path, and native/Python
agreement for a dependent bulk/small reciprocal chain. Instructions, upstream
inference, LPN scheduling, guards and heartbeat caching are unchanged.

Run all retained kernels with:

```sh
PYTHONPATH=. python -m examples.opentpu.tpu_dse \
  --case /path/to/evolved-pipeline-final \
  --opentpu-root /path/to/FeSens-openTPU-small-recip \
  --compiler build/tools/lpn-compile/lpn-compile \
  --architectures proposal_recip2_128x1_add2,proposal_recip4_128x1_add2,proposal_recip8_128x1_add2,proposal_recip16_128x1_add2 \
  --indices all --workers 8 --output /tmp/wide-recip-predictions
```

These proposals retain 128 matrix engines, ACT streaming, quantizer lookup and
the two-stage adder. The latest audited full RTL result remains
[12.7717×](FPGA_REDUCTIONS.md), and the 100× objective remains open.
