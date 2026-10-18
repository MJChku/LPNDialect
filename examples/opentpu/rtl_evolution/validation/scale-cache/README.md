# FPGA scale-line buffer prototype

`scale-cache.patch` applies after the nine patches through `small-recip.patch`
on FeSens revision `f237dab2f196855a59a083b1454786df8f01291f`.
`patch-proof.json` verifies all 28 compiled source/harness files against a
clean application. Earlier RTL checkouts and measurements are unchanged.

Set `SCALE_CACHE_ENTRIES=16` on the simulation top to enable a private
direct-mapped 32-byte scale-line buffer per physical matrix engine. Zero
preserves the old direct memory path. This is actual RTL data storage and
response routing, not a bandwidth discount applied to the timing model.

* A repeated read can share a pending fill, including before the data arrives.
* References keep an entry alive until every accepted reader retires. A
  conflicting address waits rather than overwriting an outstanding snapshot.
* All DRAM writes conservatively invalidate all scale tags. In-flight readers
  retain their earlier data; newer readers wait for the old references to drain.
  Core reset, system reset and loader start cancel the buffer state.
* Hits preserve fixed read latency and order. The one fill write and registered
  data read have explicit same-cycle forwarding. Tags and reference counts are
  small control arrays; FPGA RAM inference and mapping are not yet measured.
* The shared arbiter charges 32 HBM-link bytes only on misses, still charging
  every actual weight read and write. Idle includes outstanding cache hits.
  The A address is available before grant to avoid a lookup/arbitration loop.

With 128 cores, 16 entries and eight-cycle responses, data capacity is **64 KiB**.
Tags, valid bits, references, response tags and registered hit data add another
**13.25 KiB** in the source-level accounting. This is not a placed resource
estimate. The parallel memory path still uses the experimental behavioral
multi-port harness; it is not wired to a vendor HBM controller.

`unit.json` checks 2-, 8- and 16-cycle memories. Across the three tests, all
accepted non-cancelled reads return the exact expected data and latency.
The tests exercise pending hits, replacement conflicts, invalidation with
older responses in flight, and reset cancellation. `matrix.json` retains nine
full-accelerator numerical gates, including packed weights, partial engine
groups, ACC, ASCALE, RMAX and scatter stores. `coherence.json` adds real DMA
scale overwrites, unaligned quantizer scale overwrites and disjoint writes
overlapping matrix reads. Complete DRAM and scratchpad outputs match the
independent ISA model in all 12 accelerator cases. Each coherence case makes
285 A requests but only 38 backing-memory reads; writes change outputs exactly
where expected.

Run from the LPN project with the patched FeSens checkout and Verilator 5.052:

```sh
python -m examples.opentpu.rtl_evolution.check_scale_cache \
  --opentpu-root ../FeSens-openTPU-scale-cache --output /tmp/scale-unit
python -m examples.opentpu.rtl_evolution.check_matrix \
  --opentpu-root ../FeSens-openTPU-scale-cache --rows 8 \
  --scale-cache-entries 16 --output /tmp/scale-matrix
python -m examples.opentpu.rtl_evolution.check_scale_cache_coherence \
  --opentpu-root ../FeSens-openTPU-scale-cache --output /tmp/scale-coherence
python -m examples.opentpu.rtl_evolution.check_scale_cache_workload \
  --opentpu-root ../FeSens-openTPU-scale-cache \
  --case /path/to/completed-original-input-case --kernel 16 --entries 16 \
  --recip-table /path/to/recip-two-bit-table --output /tmp/scale-decode
```

The real retained decode-kernel run is still building/running. No workload
speedup has been established for this buffer. The current LPN equations model
the uncached RTL; the workload tool explicitly records that it does **not**
validate cached-RTL LPN timing. Conflicts and conservative invalidation can
reduce the traffic benefit, so a hit-rate result alone is not a cycle-speedup
claim. The [full uncached 48-kernel replay and corrected timing audit](../matrix-retirement/README.md#complete-inference-validation)
have completed: 15.2522x fewer cycles, bit-exact states and 4.8019% worst LPN
kernel error. This cache is not part of that measured result. The 100x goal
has not been reached.
