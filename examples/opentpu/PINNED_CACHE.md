# LPN proposal: bounded persistent weight-region cache

This proposal connects the [cold/coherent traffic prepass](WEIGHT_STORAGE.md)
to the existing matrix prefetch component. It consumes the same original
instruction words and canonical input images. Inference remains upstream.
There is no matching cache RTL implementation yet, and RTL replay/audit reject
this timing flag rather than silently using the uncached hardware.

## Ports, storage and ordering

`Timing(pinned_cache_bytes=..., cache_read_bytes_per_cycle=8192)` requires an
explicit per-command, per-weight-row hit profile. A prefix table permits a
runtime batch of rows to read its cached byte count without per-byte events.
The timing model validates row counts, bounded/aligned hit bytes and capacity.
The DSE runner also verifies profile hashes against the completed 48-kernel
manifest and matches canonical program/DRAM hashes with the retained inputs.

HBM and cache reads have independent resource places. Cache hits occupy the
SRAM port; misses and all existing scale metadata occupy HBM. The original
scale decoder remains separately bounded. A batch arrives after the slowest
of those three streams plus the original memory response latency. Later
batches wait for all streams to preserve FIFO order, including a hit following
a miss. SRAM latency is conservatively charged the original memory latency.

The proposed SRAM read and fill-write ports each sustain 8,192 bytes per clock.
The separate fill port exceeds the 2,816-byte/clock HBM link, so fills cannot
exceed its aggregate bandwidth. This does not prove bank-conflict behavior,
physical RAM mapping, routing or Fmax; those remain hardware work. A single
shared cache-read credit bounds all matrix engines' aggregate SRAM traffic.

The fixed region needs data plus one valid bit per 64-byte line. With the
128-engine small-reciprocal baseline, the raw lower bounds are:

| Cache | Total on-chip bytes including current model storage | Capacity screen |
| --- | ---: | --- |
| 24 MiB | 40,209,408 | Within provisional 50% SRAM allocation |
| 48 MiB | 65,424,384 | Requires provisional 80% SRAM allocation |

Both retain 8,192 matrix MAC lanes and the same HBM budget. RAM width rounding,
banking and control may increase storage. An 80% SRAM screen is explicitly more
aggressive than the earlier 50% screen and is not a fit result.

## State continuity and validation scope

The region starts cold in kernel 0. The prepass invalidates every changed host
input line and every line potentially written during a kernel. A cold line is
not counted as a hit until the next kernel. This is conservative within a
kernel and independent of detailed fill ordering. It executes no neural-network
arithmetic. Its cost is separate from native timing execution.

Future RTL validation must preserve or restore and audit the cache's data and
valid bits across kernel boundaries, validate host/core write coherence and
include the cold first kernel. Starting fresh RTL processes with free warm
hits would not validate this proposal.

Tests check native/Python agreement, a shared SRAM bandwidth lower bound, the
effect of finite cache bandwidth, all-miss equality with the original timing,
missing/malformed metadata rejection, cold/warm state, host input changes,
partial-write invalidation and exact per-command profile export. The event
scheduler, guard synthesis and heartbeat mechanism are unchanged.

## Reproduce

```sh
PYTHONPATH=. python -m examples.opentpu.pinned_cache_study \
  --case /path/to/evolved-pipeline-final --mib 24,48 \
  --profile-root /tmp/pinned-profiles --profile-mib 24,48 \
  --opentpu-root /path/to/FeSens-openTPU --output /tmp/pinned-traffic.json

PYTHONPATH=. python -m examples.opentpu.tpu_dse \
  --case /path/to/evolved-pipeline-final \
  --opentpu-root /path/to/FeSens-openTPU \
  --compiler build/tools/lpn-compile/lpn-compile \
  --memory-profile-root /tmp/pinned-profiles \
  --architectures proposal_pinned24m_128x1_add2,proposal_pinned48m_128x1_add2 \
  --indices all --workers 8 --output /tmp/pinned-predictions
```

Native timing binds these functional memory descriptors through the existing
input-table ABI. It does not rerun model arithmetic or serialize token tensors.
The 100× target remains open; a traffic reduction alone is not a speedup proof.

## Completed full-workload forecasts

| Variant | LPN cycles | Predicted reduction vs original RTL |
| --- | ---: | ---: |
| Earlier uncached small-reciprocal candidate | 8,156,090 | 16.286× |
| 24 MiB pinned region | 8,161,626 | 16.275× |
| 48 MiB pinned region | 8,165,994 | 16.266× |

Both cache candidates complete all 48 kernels with canonical program identity
and verified per-row cache-profile hashes. Neither improves this timing model;
the tiny regressions are below 0.13%. Extensive cache RTL is not selected on
these results. Evidence is in
[validation/pinned-cache](rtl_evolution/validation/pinned-cache).

This variant keeps the original memory response latency and leaves scale
metadata in HBM. Its result does not establish the behavior of a lower-latency
cache that also serves metadata. The separate 128-engine real-inference RTL
pilot currently shows 13.4% underprediction on its two completed decode kernels,
so these candidate forecasts are not validated to the 10% timing target.
