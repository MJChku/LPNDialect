---
name: lpn-modeling
description: Build and extend LPNDialect's event-driven Petri-net models, guard synthesis, and native compilation, including FeSens/openTPU executable-input validation against Verilator. Use for this repository's compiler, network, accelerator, and memory timing models.
---

# LPNDialect modeling workflow

Run commands from the repository root. Read [README.md](README.md) for the
supported dialect and [examples/opentpu/README.md](examples/opentpu/README.md)
for the active accelerator target and baseline setup. On `icdslab2.epfl.ch`,
the checkouts are `~/LPNDialect` and `~/FeSens-openTPU`. The older `~/OpenTPU`
is UCSBarchlab's design, a different project.

## Build and run

Use Python 3, CMake, a C++ compiler, and LLVM/MLIR 20 development packages.
Native compilation also requires `opt-20`, `llc-20`, and `clang++-20` on `PATH`.

```sh
export LLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
export MLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
CMAKE_BUILD_TYPE=Release ./build.sh -j8
ctest --test-dir build --output-on-failure
```

Smoke-test the all-to-all network:

```sh
./run.sh simulate --devices 4 --messages 2
./run.sh emit --devices 4 --messages 2 > /tmp/network.mlir
build/tools/lpn-compile/lpn-compile /tmp/network.mlir -o /tmp/network
/tmp/network --max-time=20000000 --tokens
```

Alternatively, `./run.sh compile /tmp/network --devices 4 --messages 2`
emits and compiles in one command. Native compilation lowers MLIR to LLVM IR,
optimizes with `opt-20 -O3`, generates an object with `llc-20`, and links the token/event runtime. There is
no generated C++ model or interpreted MLIR in the executable's transitions.
Executables default to a limit of 100 million model time units; pass
`--max-time=NUMBER` explicitly for longer workloads.

## Construct models

Use `NetBuilder` from `lpnlang_mlir`. Instantiate component classes with a
shared builder; pass `PlaceHandle` objects between components and prefix names
per instance. Classes can contain other component classes. Existing examples:
`NetDevice`/`NetUniDirLink` in `examples/network_net.py`, and the units inside
`examples/coralnpu/model.py` and `matrix.py`.

For openTPU architecture changes, follow
[examples/opentpu/ARCHITECTURE.md](examples/opentpu/ARCHITECTURE.md).
`CompactOpenTPU` composes independent execution components; use
`Timing(matrix_engines=N)` for existing-ISA matrix-engine experiments. Keep
shared-memory hazards explicit across engines and retain FIFO command
admission to avoid dependency deadlocks. Use `model.statistics()` to count net
objects. Hardware concurrency does not imply that host transition functions
can run concurrently without scheduler synchronization.

For native functional execution and reusable timing binaries, read
[examples/opentpu/NATIVE_INTERFACE.md](examples/opentpu/NATIVE_INTERFACE.md).
Use immutable named input tables for the native boundary. The shared-library
ABI borrows producer buffers for a whole run; keep them alive and serialize
calls to one runtime. `e2e --reuse-timing` avoids compiling per command stream.
Full trained-model numerical execution remains Python/NumPy; do not describe
the native scalar subset or linking demonstration as a complete neural model.

The supported accelerator input is canonical hardware `input.json` plus
`prog_N.hex`, shared with RTL. Read [HARDWARE_COMMANDS.md](examples/opentpu/HARDWARE_COMMANDS.md)
for static replay and numerical control resolution; reject unresolved
data-dependent instructions in weight-free replay. For GLM, use the actual
[hardware compiler and retained-input validation](examples/opentpu/validation/glm47-flash/hardware-commands/README.md).
Its MLA/router/expert programs and boundary kernels are component gates;
complete 47-layer generation and whole-model memory mapping remain unfinished.
Historical GLM tensor-DAG timings do not establish execution on this LPN.
Use `glm47_target.component_timing()` for the two-column correctness profile:
physical RTL defaults to one epilogue lane, whereas generic `Timing` defaults
to two. Preserve old reports when correcting a physical profile. On icdslab2,
fresh builds need Verilator 5.052 on `PATH` and its prefix in `VERILATOR_ROOT`.

For weight-free FPGA architecture exploration, use
`examples.opentpu.fpga_dse` and read
[FPGA_DSE.md](examples/opentpu/FPGA_DSE.md). This compiles dense Qwen shapes and
resolves static scalar control without creating weight/DRAM images or executing
tensor arithmetic. It rejects data-dependent control. This differs from
`replay_e2e --timing-only`, which retains numerical execution. Prescribed
sequence lengths do not imply generated text or EOS prediction. Distinguish
hardware-cycle reductions, assumed-clock latency and simulator wall speed.
Board bandwidth, MAC/DSP mapping and reserves are explicit assumptions;
resource screening is not synthesis proof or modified-RTL accuracy validation.
The new Timing knobs default to the original RTL behavior. Preserve shared
memory contention, scale readiness, output-bank limits and dependencies when
adding parallel lanes. Whole-image addresses above 4 GiB require a different
mapping; do not silently wrap addresses for a larger model.
Correction: the existing FPGA sweeps and sequence runner use a historical custom
token-by-token prefill stream, including logits at every prompt position. They
are microbenchmarks, not normal inference results; do not quote their 11x/8.23x
cycle reductions as inference gains. LPN must model hardware only. Inference
batching/tiling/decode choices belong to the upstream runner/compiler. Use
`examples.opentpu.standard_inference` for the unmodified `Engine.generate()`
entrypoint shared by planning and real ISA/LPN/Verilator validation. Planning
uses explicit numerical placeholders; actual validation uses real weights and
checks identical inputs/device memories. Architecture experiments must consume
the resulting commands rather than inventing their own inference schedule.
For historical token-by-token microbenchmarks only, use `examples.opentpu.sequence_timing`.
It compares original/32-row/64-row architectures across every position without
weights. Independent HALT-drained kernels run in separate host processes; the
LPN scheduler remains sequential and hardware sequence cycles are summed.
Read `progress.json` and the owning PID before starting duplicate jobs. Resume
the same output directory with unchanged source/parameter fingerprints; worker
count may change. Per-position JSON checkpoints retain completed work. Use
`--prompt-tokens 1024 --output-tokens 8 --workers 16` for the long example.
Prefill is token-by-token; larger-capacity predictions are not new RTL validation.

Create places with `net.place(name, initial_tokens=N, observable=True)` as
needed, and register transitions with `@net.transition`. Transition primitives
such as `take`, `peek`, `count`, `guard`, `create`, and `emit` are supplied by
the DSL. Define decorated functions in real Python files: the frontend reads
their source using AST inspection, so stdin, `exec`, and dynamically generated
source without a backing file are unsuitable.

Use `net.build()` to obtain MLIR or `net.emit_to_file(path)` to write it.
For the Python reference, use `sim = net.python_simulator()`,
`sim.run(max_time=...)`, and `sim.place_dicts(place_name)`.
The C++ reference interpreter is
`build/tools/lpn-opt/lpn-opt --lpn-simulate model.mlir`.

Represent resource capacity with initial credit tokens, acquire them at issue,
and emit them back at the resource's initiation interval. Schedule result
tokens at completion latency. Distinguish occupancy, initiation interval, and
result latency; they need not be equal. Preserve event-driven execution and
use tokens per useful work group instead of one transition per clock or MAC.

### Atomic takes, peeks, and guards

All operations in a firing are atomic. Takes are provisional: if a later
take, peek, or guard blocks, the entire firing rolls back. Emits become visible
only when the firing succeeds, after their specified delays.

`peek(place)` reads the next ready FIFO token without consuming it. It blocks
on an empty place; it does not read the most recently appended token.
Token updates are immutable: keep the result of `token.set(key, value)`.
Do not consume and re-emit a losing arbitration token, which changes FIFO
order. Instead inspect with `peek`, then take the selected input.

Use value-dependent `if` inside decorated transition source. Avoid Python
`and`/`or` on DSL SSA values; express conjunction with separate `guard(...)`
calls. Freeze per-instance loop bindings with default arguments when creating
multiple transitions in a Python loop.

Guard synthesis is automatic where its take-ID representation is sufficient:

```sh
build/tools/lpn-opt/lpn-opt --lpn-synthesize-guards model.mlir
```

It rejects token flow in data-dependent branches, repeating/dynamic loops,
and effectful helpers. For those cases, supply explicit enabling predicates
with `guard(...)` at transition entry, before takes, emits, and effectful helper calls.
Helpers that only read state and compute immutable values can participate in
manual predicates. Unknown or recursive helpers remain conservatively effectful.
Guard counts before peeking where necessary. The pass preserves these manual
predicates and marks the transition `lpn.guard_mode = "manual"`.
See the arbitration example in [README.md](README.md).

Native compilation always synthesizes a separate executable enabling function,
automatically counting token requirements for static control flow. It supports
static repeated takes beyond the reduction pass's take-ID representation.
Dynamic token flow and helper calls require manual entry predicates; otherwise
compilation fails. Enabling functions run before firing transactions are
allocated and before bodies execute. Use `--stats` to check for unexpected
rollbacks. The atomic runtime remains a safeguard for incomplete annotations.
Observable reduction currently rejects manual guards;
do not discard the predicates to force reduction through.

Native predicates borrow peeked tokens while checking state. Scalar helpers
that only read state can do the same; transition-body peeks retain snapshots
so a later take cannot invalidate an earlier peek. Empty predicate peeks
disable the transition before its body runs.

Native predicates cache both true and false results using per-place heartbeats.
Visible token arrivals, takes and replacements invalidate observed dependencies;
unchanged token counts do not imply unchanged state. Actual runtime reads track
helper and dynamic-place dependencies, including empty reads. `--stats` counts
actual evaluations separately from cache hits; `--no-guard-cache` disables this
optimization for differential checks. Preserve firing order and event times
when changing the cache. It does not provide parallel execution or safe bounds
for components to advance independently in simulated time.

Native empty-only queues use counters. Data-bearing tokens have immutable,
reference-counted payloads: movement, clone and peek share handles; `.set()`
copies the payload for an independent update. Mixed queues and rollback keep
FIFO order. Use `--stats` to inspect payload allocations and field copies on
updates. For compact completion state, use `andi`/`ori`/`shli` and 64-bit masks;
keep shift counts in [0,63] and account for the signed high bit.

## FeSens/openTPU

Read [examples/opentpu/README.md](examples/opentpu/README.md) for the pinned
revision, observer patch, architecture, input format, and measured scope.
Use Verilator 5.052 or a compatible newer version and a Python environment
with NumPy. The model reads the same hex programs and big-endian DRAM files
as the RTL harness. The upstream ISA simulator provides numerical execution
and addresses; it supplies no baseline timing. Native LPN then predicts time
from dataflow/resource tokens.

```sh
PYTHONPATH=. ../FeSens-openTPU/.venv/bin/python -m examples.opentpu.benchmark \
  --opentpu-root ../FeSens-openTPU --output /tmp/opentpu-gemm \
  --batch-rows 8 --runs 5 --lpn-compiler build/tools/lpn-compile/lpn-compile
```

Add `--workload mlp` to validate an upstream-compiled RMSNorm/SwiGLU/residual
kernel. `examples.opentpu.run` emits or compiles an existing input bundle.
Use `--batch-rows 1` as the fine-grained reference; larger groups are bounded
by shape and FIFO capacity. Preserve separate initiation intervals and result
latencies, early matrix prefetch, and dependency tokens. Compare actual cycles
and numerical memory state against RTL after changing those assumptions.
Report cycles/second, core wall medians, and preparation/compilation costs
separately. Do not extrapolate kernel accuracy to whole Llama inference.

Use `examples.opentpu.standard_inference` for normal upstream inference.
`Engine.generate()` owns batched prompt prefill and autoregressive decode;
both RTL and LPN receive its programs and persistent device state. The backend
adapter must not decide inference batching, tiling, or token order. The small
synthetic numerical fixture is not full-checkpoint accuracy validation.
Completed normal-inference cases are under `examples/opentpu/validation/standard_inference`:
SmolLM2-135M (`smollm2_16in8out`, two eight-row prefill kernels) and Qwen3-1.7B
(`qwen17_8in2out`, two four-row prefill kernels). Their numerical checks are
bit-exact; worst per-kernel timing errors are 2.79% and 1.08%, respectively. The accompanying
`audit_checkpoint.py` verifies identities and independently replays the final
retained kernel; use frozen project revision `c82de17` for its source checks.
Keep long validation sources frozen. Case manifests and progress for these
cases are under `~/lpn-runs/standard-checkpoint-validation`; both jobs completed
with exit code zero.
Read their job status and child PID before starting another validation; a
completed manifest is historical evidence, not a reason to restart it.

To reproduce the historical custom token loop, `examples.opentpu.e2e` retains
real trained-model numerical execution with persistent KV state and per-step RTL
checks. Follow the pinned public-checkpoint command
in the openTPU README; install its additional CPU Torch/safetensors/tokenizer
dependencies. `--max-steps 3` provides a bring-up checkpoint and `--resume`
continues with identical sources/configuration and verified snapshot hashes.
Keep long-run artifacts under `~/lpn-runs`, not the source tree. The runner
keeps two bounded snapshots and one current input/output bundle. Do not change
hashed model/compiler sources while a resumable validation is running.

Larger Qwen3-1.7B runs need the behavioral memory storage patch and isolated
FeSens worktree described in the openTPU README. Use the installed Verilator
5.052 with both `VERILATOR_ROOT` and `PATH` set. The completed parallel campaign's
manifest is `~/lpn-runs/qwen3-1.7b-parallel/batch.json`; inspect it and the case
logs before launching duplicate jobs.
The batch expanded from four to six concurrent cases using a verified scheduler
handoff. Read the current supervisor PID and parallelism from `batch.json`;
the original four model/RTL workers were preserved. Do not restart a worker
because its supervising scheduler changed. Adopted process completion is
validated from its terminal report and snapshot, with unknown original exit
status explicitly recorded rather than invented as zero.
After four completed generations, a second verified handoff added `hello12`
and `why12`, each requesting twelve generated tokens. The batch now has ten
cases with a limit of six concurrent workers. See
`validation/qwen17/queue_extension.json` under `examples/opentpu` for preserved
worker identities; `supervisor_v3.py` is the recorded successor. Continue to
read the live manifest instead of assuming either historical scheduler owns it.
All ten baselines and complete LPN replays have now passed: 143 token steps,
88 generated tokens. Terminal `batch.json`, `overnight-summary.json` and
`lpn-replays.json` record completion; missing historical controller PIDs are
expected after completion and must not trigger a restart. Results, raw CSV,
an independent archive audit and measured functional overhead are under
`examples/opentpu/validation/qwen17`. Recheck disk snapshots, final logits,
source fingerprints and all paired/replay observations with
`audit_completed.py --root RUN_ROOT --project-root FROZEN_PROJECT_ROOT --opentpu-root FESENS_ROOT
--expected-cases 10 --output /tmp/qwen-completion-audit.json`, using the NumPy
environment. Use a worktree at `cea0c77` for `FROZEN_PROJECT_ROOT` when auditing
the original campaign: the strict audit compares recorded replay source hashes,
which later source changes must not overwrite. The audit is read-only except
for its output and rejects an
unfinished campaign. Report whole-process speedup separately from native
timing speedup; functional frontend time does not isolate pure matrix compute.
RAM scratch holds active large snapshots; the separate overnight monitor
backs up completed-token snapshots to
`recovery/<case>` and publishes `overnight-summary.json`/`overnight-records.csv`.
Check the monitor PID and verified recovery manifest rather than assuming that
a RAM checkpoint is durable. Finished snapshots are archived to `results`.
The separate full-case replay driver is recorded in `lpn-replays.json`; verify
its PID before starting another. `examples.opentpu.replay_e2e` regenerates real
numerics from a completed report and verifies every input/logit/cycle. It
reports session wall time separately from the summed functional/timing stages.
For LPN-only timing experiments, add `replay_e2e --timing-only` to skip full
RTL DRAM/program file serialization and per-step input hashing. Functional
execution, KV state, greedy decoding and logit/cycle checks remain active.
Skipped input checks are recorded as null, not true. Checkpoint/compiler
fingerprints and small timing tables still exist; this option does not disable
all hashing or skip numerical execution. Default mode retains input validation.
Only use `--allow-partial` for an explicitly labeled unfinished snapshot.
Freeze the RTL/observer source
hashes in run identities; the pinned commit alone does not identify local
simulation patches. Do not modify the separate frozen SmolLM2 baseline tree.

Use the compact model for long traces: reusable transitions, immutable flag
banks and precomputed dependency metadata. Compare against the expanded model
when changing the compact representation. Do not replace the event-driven
scheduler. Full trained inference speedups include functional preprocessing
and native preparation separately; GEMM throughput is not representative of
the whole transformer. The checkpoint's real EOS can end a requested long run
early. Report completed tokens/time separately from the estimated full run.

Optional integration checks need the NumPy environment:

```sh
PYTHONPATH=. ../FeSens-openTPU/.venv/bin/python test/test_opentpu.py \
  --opentpu-root ../FeSens-openTPU --tool build/tools/lpn-compile/lpn-compile
```

The backend rejects unsupported host synchronization, chaining, stream-engine
instructions, WAITC, and multi-slice collectives. Extend these deliberately;
do not silently assign arbitrary delays to unsupported instructions.

## Previous CoralNPU and matrix models

The model predicts timing for a dynamic instruction stream; it does not
execute arbitrary ISA arithmetic or predict new control flow. Trace adapters
discard observed timestamps when building the model. Use the original MLIR
for timed validation.

For real executable workloads, use the shared-ELF frontend and compact
scoreboard described in [REAL_WORKLOAD.md](examples/coralnpu/REAL_WORKLOAD.md):

```sh
python3 -m examples.coralnpu.benchmark_workload \
  --coral-root ../coralnpu --output /tmp/coral-real-100k \
  --jobs 160 --runs 5 --lpn-compiler build/tools/lpn-compile/lpn-compile
```

This executes supported RV32/VME instructions independently to obtain the
path, numerical results, and effective addresses from the same ELF as RTL.
Unsupported instructions and external-memory workloads fail explicitly.
The native LPN then predicts timing, using resource and dependency tokens.
Build the RTL baseline with `-c opt --copt=-O3`; default fastbuild does not
optimize generated simulator C++. Report ROI wall medians separately from
frontend execution, graph construction, and native compilation. The command
also reports the combined cost for a single use of the ELF. Disable dispatch,
writeback, and transition traces during speed samples; keep the passive ROI
timers. Native timers use `--wall-start=NAME --wall-end=NAME`.

```sh
python3 -m examples.coralnpu.run \
  --trace examples/coralnpu/validation/rep32/alu_chain.csv \
  --emit /tmp/coral.mlir
build/tools/lpn-compile/lpn-compile /tmp/coral.mlir -o /tmp/coral
/tmp/coral

python3 -m examples.coralnpu.run \
  --gemm 16 16 64 --dtype int8 \
  --memory-latency 20 --memory-bandwidth 32 --emit /tmp/gemm.mlir
build/tools/lpn-compile/lpn-compile /tmp/gemm.mlir -o /tmp/gemm
/tmp/gemm
```

CoralNPU model units are cycles, including memory latency; bandwidth is bytes
per cycle. The generic native output still labels time `ns`.
`MatrixArray` models four staggered PE blocks, packed INT8/BF16/FP32 pipelines,
and accumulator hazards. `GemmModel` adds tiled commands, operand buffers,
accumulator ownership, and memory transfers. Memory service time is
`ceil(bytes / bytes_per_cycle)` plus latency, with bandwidth and outstanding
request resources. This is a simple timing endpoint, not a complete AXI/DRAM
protocol simulator. Reject unknown runtime matrix configuration instead of
guessing shapes or types.

For fresh RTL validation, check out CoralNPU revision
`517f4c09bd5aabb4e8a16436d8853c912a24790b`, apply
`examples/coralnpu/verilator-observer.patch` once, and build the RVV and VME
Verilator targets using the detailed model README. Then run:

```sh
python3 -m examples.coralnpu.validate \
  --coral-root ../coralnpu --clang clang-20 \
  --output /tmp/coral-validation \
  --lpn-compiler build/tools/lpn-compile/lpn-compile
python3 -m examples.coralnpu.validate_matrix \
  --coral-root ../coralnpu --clang clang-20 \
  --output /tmp/matrix-validation \
  --lpn-compiler build/tools/lpn-compile/lpn-compile
```

Recorded CSV observations and comparison reports are under
`examples/coralnpu/validation`. The matrix baseline checks architectural output
elements and array timing. Matching microkernels does not establish complete
core cycle equivalence. Compare equivalent timed regions and report startup,
RTL numerical execution, and readback overhead when discussing speed.

For Llama 3.2 3B, follow [examples/coralnpu/LLAMA.md](examples/coralnpu/LLAMA.md).
Run:

```sh
python3 -m examples.coralnpu.llama --prompt-tokens 128 --generate-tokens 4 \
  --emit /tmp/llama.mlir --lpn-compiler build/tools/lpn-compile/lpn-compile
```

The model reuses layer graphs through feedback tokens. Keep residual fanout
edges and decode-pass counter resets when modifying it. Check compact versus
expanded dataflow on a small shape. Full-size execution is native; Python is a
reference for small regressions. Numerical inference and whole-model 5%
accuracy are not established. Validate representative kernels in Verilator
before claiming that target; AXI beat-level modeling is outside this scope.

## Validate changes and respect current limits

For runtime/compiler changes, run CTest and compare Python/native final token
fields, counts, and completion time on affected models. Guard changes need
coverage for nested takes, rollback, explicit predicates, and rejection cases.
For matrix timing changes, use recorded RTL observations and rerun the actual
baseline when pipeline assumptions or observers change.

The native compiler supports one `lpn.net`, structured `scf.if`/`scf.for`,
arrays, arithmetic, and single-block helpers/transitions. Unsupported operations
fail explicitly. Native compilation rejects place capacities; the interpreters
ignore them, so use credit tokens to enforce bounded resources.

`--lpn-normalize-delays` sets nonzero delays to zero. Use it only for structural
analysis. Observable reduction is experimental and can alter behavior; do not
insert either pass into the native timing workflow without validation.
Keep binaries, build directories, and Python caches out of commits.
