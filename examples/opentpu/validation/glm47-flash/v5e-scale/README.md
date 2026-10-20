# GLM-4.7-Flash on proposed v5e-scale hardware, LPN first

The screen is complete, with ten architecture/routing points and three native
LPN samples per point. **This is an uncalibrated hardware proposal, not completed
RTL, a measured TPU benchmark, or numerical GLM inference.** The full input is
the previously audited upstream 1024-input / 8-output tensor DAG. Standard
upstream generation owns the batched prefill and seven subsequent decode calls.
Routing remains explicitly annotated as balanced or four-expert hotspot.

## Hardware assumptions

Google lists four MXUs, 197 BF16 TFLOP/s, 800 GiB/s HBM and 16 GB per v5e chip:
<https://docs.cloud.google.com/tpu/docs/v5e?hl=en>. Its architecture documentation
describes pre-v6e MXUs as 128-by-128 BF16 MAC arrays with FP32 accumulation:
<https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm?hl=en>.

Our proposed four 128-by-128 arrays have 65,536 MAC sites. At the explicitly
assumed 1.5 GHz they provide **196.608 trillion matrix FLOPs/s**, counting a
multiply-add as two operations. The clock is a study assumption, not a measured
Fmax. The alternative 64 arrays of 32-by-32 or 16-by-64 preserve exactly that
site count. Eight 128-by-128 arrays double it. No point establishes FPGA fit.

NVFP4 weight arithmetic and BF16 attention share the matrix cluster, as does
the slower FP32 router path. Their precision-specific peaks are **not additive**.
NVFP4 gets no automatic compute multiplier: the proposal assumes sustained
dequantization into the BF16 path, which still needs implementation/calibration.
FP32 throughput is an assumed 1024 MACs/cycle, vector throughput 1024 elements
per cycle. Neither is an assertion about Google's undocumented implementation.

Baseline HBM is exactly 800 GiB/s (858.9934592 decimal GB/s), about 573 bytes
per cycle. The 2x-HBM point doubles bandwidth while preserving compute.
All points propose **32 decimal GB**, explicitly exceeding v5e's listed 16 GB:
our proposed GLM weights alone occupy 16,865,359,940 bytes, before expanded K/V
and workspace. Allocation/liveness and a real wider-than-32-bit memory interface
remain unimplemented; this is not a full capacity proof.

## Results

| Array organization | Peak matrix TFLOP/s | HBM GiB/s | Balanced modeled ms | Hotspot modeled ms |
| --- | ---: | ---: | ---: | ---: |
| 4 x 128 x 128 | 196.608 | 800 | 202.416 | 150.875 |
| 64 x 32 x 32 | 196.608 | 800 | 175.614 | 123.964 |
| 64 x 16 x 64 | 196.608 | 800 | 175.730 | 124.193 |
| 8 x 128 x 128 | 393.216 | 800 | 173.152 | 115.979 |
| 64 x 16 x 64 | 196.608 | 1600 | 101.587 | 84.193 |

The graph has **7,747,810,312,192 useful matrix FLOPs**, identical at every point:

| Matrix arithmetic | Useful FLOPs |
| --- | ---: |
| NVFP4-weight mixed arithmetic | 6,719,133,974,528 |
| BF16 attention matrices | 1,016,243,896,320 |
| FP32 router matrices | 12,432,441,344 |

This count includes no padded array work. Vector, softmax, normalization,
nonlinear functions, and routing-selection work are **excluded from FLOPs**,
not treated as zero latency. Their element-service timing remains in LPN.
`arithmetic.matrix_flops_by_call` separates upstream prefill/decode work.

Achieved useful matrix rates are 38.28 / 51.35 TFLOP/s for the four-array point
(balanced / hotspot), 44.12 / 62.50 for the same-compute 32-by-32 partition,
and 76.27 / 92.02 for the 16-by-64 partition with twice HBM bandwidth.
These are useful matrix operations divided by *modeled hardware seconds*;
they are not the host's simulation throughput or a measured silicon result.

For balanced routing the modeled HBM traffic alone requires at least 163.594 ms
at baseline bandwidth. Thus the 175.614-ms partition point is already near
that screen's bandwidth floor. Doubling arithmetic alone reaches 173.152 ms.
Improving data movement and MLA storage is a higher-priority next investigation
than adding more large arrays. No comparison with old FeSens RTL cycles is made:
the command interface, arithmetic and model differ.

## LPN speed and validation

Each native event-driven run took **1.11–1.84 seconds**, depending on graph and
point. Model/table construction took 0.28–0.43 seconds; compilation 0.24–0.27.
The complete ten-point study with three samples each took **55.19 seconds**,
reusing existing graph files. Separately, original upstream shape capture took
17.14 seconds for balanced routing and 11.01 for hotspot, once per graph.
Every repeated native run returned identical cycles and zero rollbacks.
Each LPN is single threaded; these architecture points were run sequentially.

Six tests pass in Python and compiled native execution. Tests cover shared
matrix-resource contention between precisions, padded-array timing, throughput
floors, analytical partition utilization, dependency forks, shared HBM,
completion flag/window wrap, and invalid inputs. The audit independently
reconstructs matrix FLOPs from every shape, checks all upstream model-work
formulas and dependencies, verifies source/binary/table/log hashes, and checks
that no result violates aggregate shared-HBM service capacity.

The architecture source is frozen at `4a08ebd` in
`~/LPNDialect-glm47-v5e-replay` on icdslab2. The audit and final tests use
`272ef64` in `~/LPNDialect-glm47-v5e-audit`. Full generated graphs and native
binaries/logs remain under `~/lpn-runs/glm47-flash/v5e-scale-1024x8-v2`.
The compiler is the heartbeat-enabled compiler recorded by the launch and audit.

The array service model reserves the cluster for one whole operation. A padded
output tile streams K and pays R+C-2 fill/drain cycles; tiles are distributed in
waves across arrays. Interoperation tile packing is absent. Aggregate memory
and compute overlap optimistically. Each modeled execution path owns a 1-MiB
buffer credit, but this does not establish detailed K tiling, refill/reuse
traffic, SRAM ports, accumulator feedback, scale decode, SFU or controller
timing. All recorded intermediate accesses spill to HBM; compressed/absorbed
MLA is not substituted for the expanded-K/V upstream reference. **The usual
10% RTL timing target has not been demonstrated for this new accelerator.**

## Reproduce

After producing the existing `glm47_screen` traces, use a compiler with
`lpn.input_array` and heartbeat support:

```sh
PYTHONPATH=. python -m examples.opentpu.glm47_v5e_screen \
  --traces /path/to/screen-1024x8 \
  --compiler /path/to/lpn-compile --output /tmp/glm-v5e --runs 3
PYTHONPATH=. python -m examples.opentpu.glm47_v5e_audit \
  --case /tmp/glm-v5e --traces /path/to/screen-1024x8 \
  --compiler /path/to/lpn-compile --output /tmp/glm-v5e-audit.json
PYTHONPATH=. python test/test_tensor_dag.py --tool /path/to/lpn-compile
```

`screen-1024x8.json`, `audit.json`, and `native-tests.log` preserve the result.
The generic timing model contains no model/inference scheduling. Existing
scheduler, guard synthesis, heartbeat cache, and FeSens RTL remain unchanged.
