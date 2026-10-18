# Lossless weight storage and bounded persistent-cache traffic

These studies read the exact canonical program and DRAM inputs. They execute
no neural-network arithmetic and make no LPN or RTL speedup claim.

## Real-weight compressibility

Decode kernel 16 requests 136,338,048 weight bytes, covering 136,328,832 unique
bytes. Thus almost every requested byte is distinct within that kernel. After
excluding every 64-byte block potentially written by the kernel, 135,228,672
immutable weight bytes remain for the codec study.

| Diagnostic | Estimated or measured stored bytes |
| --- | ---: |
| Raw immutable bytes | 135,228,672 |
| Global Huffman payload, no codebook/indexing | 122,378,596 |
| 64-byte Rice blocks with mode/escape bits | 123,492,831 |
| Software zlib, 4 KiB blocks and descriptors | 124,729,280 |
| Software zlib, 64 KiB blocks and descriptors | 122,714,446 |

Byte entropy is 7.202 bits. The software zlib blocks all round-trip bit-exactly;
Rice is a size estimate only. The roughly 9–10% reduction is too small to solve
the [uncached bandwidth requirement](FPGA_PROFILE.md) on its own. No hardware
codec, directory, decoder throughput or resource mapping is implemented here.

## Persistent pinned-region traffic

A fixed region beginning at logical byte address 2 MiB starts cold at kernel 0.
The study replays all 48 unchanged programs in upstream inference order. Before
each kernel it compares every candidate line with the current canonical input
image and invalidates any changed line. It also invalidates and excludes every
line possibly written by that kernel. Cold reads become available only in the
next kernel, conservatively avoiding assumptions about fill timing.

| Cache data capacity | Weight bytes served from cache | Remaining HBM weight bytes | Optimistic HBM-only speedup ceiling |
| --- | ---: | ---: | ---: |
| 8 MiB | 377,270,272 | 5,834,921,984 | 64.10× |
| 16 MiB | 753,963,008 | 5,458,229,248 | 68.53× |
| 24 MiB | 1,130,222,592 | 5,081,969,664 | 73.60× |
| 32 MiB | 1,503,760,384 | 4,708,431,872 | 79.44× |
| 44 MiB | 2,069,377,024 | 4,142,815,232 | 90.29× |
| 48 MiB | 2,257,145,856 | 3,955,046,400 | 94.57× |
| 64 MiB | 3,010,086,144 | 3,202,106,112 | 116.81× |

Ceilings charge 2,816 HBM bytes per clock for the remaining weight bytes only.
They exclude metadata, DMA/KV traffic, bank conflicts, dependent operations and
all other costs. Cached reads still need physical SRAM bandwidth; a separate
8,192-byte/clock SRAM floor is reported. The first kernel gets no free warm data.
No previously valid candidate line changed in these retained inputs, but tests
exercise both changed-host-input and partial-write invalidation.

Storage includes the data plus one valid bit per 64-byte line. A 24 MiB cache
with the candidate's roughly 15 MB baseline storage fits the provisional half-
SRAM capacity screen; 44/48 MiB require a more aggressive SRAM allocation.
64 MiB leaves very little capacity after existing storage and has no fit proof.
RAM width rounding, ports, control and Fmax remain unproven. The studies do not
assume a selected FPGA board or promote a traffic ceiling into a cycle result.

## Reproduce

```sh
PYTHONPATH=. python -m examples.opentpu.weight_storage \
  --case /path/to/evolved-pipeline-final --kernel 16 \
  --opentpu-root /path/to/FeSens-openTPU --output /tmp/weight-storage.json

PYTHONPATH=. python -m examples.opentpu.pinned_cache_study \
  --case /path/to/evolved-pipeline-final --base 0x200000 \
  --mib 8,16,24,32,44,48,64 --opentpu-root /path/to/FeSens-openTPU \
  --output /tmp/pinned-cache-traffic.json
```

Evidence, input/source hashes and tests are in
[validation/weight-storage](rtl_evolution/validation/weight-storage).
