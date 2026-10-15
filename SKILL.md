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
with `guard(...)` at transition entry, before takes, emits, and helper calls.
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
