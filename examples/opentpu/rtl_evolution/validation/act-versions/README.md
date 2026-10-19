# Activation-storage version screening

`Timing(act_versions_unbounded=True)` is an optimistic screening proposal,
not a finite FPGA design or a rigorous runtime bound. Each QACT writes fresh
storage for every affected logical ACT row/block, including scale/offset
metadata. Every MM snapshots the most recent preceding version of each page.
Partial writes retain untouched pages; initial pages retain initial storage.
Old readers therefore do not block new activation writers. TMEM/DRAM hazards,
barriers, original instructions and inference scheduling remain unchanged.

The original logical footprints are used first for streaming-producer
analysis. Only dependency construction uses versioned ACT ranges. A late
reader can conservatively wait for producer completion if the shared progress
latch has moved on. No numerical values, future-use oracle or RTL timing
measurements enter the transformation. The event scheduler, guard synthesis
and place heartbeat logic are unchanged.

There is no limit on retained versions, allocation delay or counted physical
bank/port cost. All FPGA capacity screens explicitly fail for these points,
the DSE requires `ceiling_only=True`, and RTL replay rejects the timing flag.
The ordinary default model has this flag disabled. A positive screen needs
finite page banks, reader references captured at dispatch, bounded version
tags and explicit read/write ports before it can become a hardware claim.

Six tests (`tests.log`) cover partial writes, initial pages, DUP/PAIR coverage,
barriers, unknown-access rejection, non-ACT hazards, rejection as RTL/FPGA,
and an overlap example with identical Python/native event-driven timing.

```sh
python test/test_opentpu_act_versions.py --opentpu-root ../FeSens-openTPU \
  --tool build/tools/lpn-compile/lpn-compile
```
