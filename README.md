# LPNDialect

LPNDialect is a Python modeling DSL and an MLIR dialect for labeled Petri nets.
Transitions take tokens from places, update token fields, and emit tokens after
delays. The native compiler lowers the model's MLIR operations to LLVM IR,
generates an object file, and links it with a token/event runtime. The resulting
executable runs compiled transitions without an MLIR interpreter.

Agents working on models or the compiler should follow [SKILL.md](SKILL.md)
for build commands, modeling semantics, and validation workflows.

For native functional prepasses, [lpnfunc](lpnfunc/README.md) compiles a typed
Python subset to a native library. The [functional/timing example](examples/native_functional/README.md)
links both stages in C++. [Runtime input tables](examples/opentpu/NATIVE_INTERFACE.md)
let one compiled openTPU timing architecture consume changing command streams.

## Model an all-to-all network

From the repository root:

```sh
PYTHONPATH=. python3 examples/network_net.py --devices 4 --messages 2 --simulate
PYTHONPATH=. python3 examples/network_net.py --devices 4 --messages 2 > network.mlir
PYTHONPATH=. python3 test/test_model_runtime.py
```

The example creates one flow from each device to every other device, with
`--messages` packets per flow. A single injection transition per source keeps
the generated IR compact. The switch and links model serialization, propagation,
and switch delay. `--devices` must be at least 2, and `--messages` must be
positive. The default is two devices and one packet per flow.

Install LLVM/MLIR development packages and Clang 20, then set `LLVM_DIR` and
`MLIR_DIR` to their CMake package directories. On Ubuntu 24.04 with the version
20 packages, the following builds the compiler and runs a native network:

```sh
export LLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
export MLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
./build.sh -j8
./run.sh compile network-native --devices 4 --messages 2
./network-native --max-time=20000000 --tokens
ctest --test-dir build --output-on-failure
```

For an existing MLIR file, run
`build/tools/lpn-compile/lpn-compile model.mlir -o model-native`. The native
executable accepts `--max-time=NUMBER`, `--tokens`, `--stats`, and `--trace`. Its default time limit is
100 million model time units; `--tokens` prints token fields as well as place
counts. `lpn-opt --lpn-simulate model.mlir` remains available as an interpreting
C++ reference implementation.

Use `--wall-start=TRANSITION --wall-end=TRANSITION` to measure host elapsed
time between the first successful firings of two named transitions, excluding
initialization and output after the selected region.

The native compiler currently accepts one `lpn.net` with place references,
token operations, arrays, integer and floating-point arithmetic, `scf.if`,
`scf.for`, and `func.func` helpers. It reports unsupported operations instead
of silently interpreting them. Place capacities are rejected by the native
compiler because bounded-place semantics are not implemented. The Python and
C++ interpreters still ignore capacities.

## Guard synthesis

The native compiler synthesizes a separate enabling function for each
transition. The runtime checks it before allocating a firing transaction or
running the transition body. Automatic inference counts required ready FIFO
tokens, including repeated takes in statically evaluable loops and branches.
Peeks require a token without consuming it. Data-dependent token flow and
helper calls currently require an explicit enabling annotation; compilation
fails with a diagnostic when inference cannot establish the requirements.
`--stats` reports guard checks, body executions, and rollback counts.
`--trace` reports successful transition firings with model timestamps.

For observable reduction, `--lpn-synthesize-guards` computes separate take-ID
metadata. Its representation is more restrictive: it rejects repeating or
dynamic token-flow loops, effectful helpers, and data-dependent branches.
Compile the original all-to-all MLIR directly for timed simulation; the native
guard inference handles its static repeated takes.

For transitions the analysis cannot infer, annotate enabling conditions with
`guard(...)` before consuming tokens. For example:

```python
@net.transition
def arbitrate():
    guard(count(control) > 0)
    guard(count(a) > 0)
    guard(count(b) > 0)
    left = peek(a)
    right = peek(b)
    take(control)
    if left.get("priority") >= right.get("priority"):
        emit(output, take(a))
    else:
        emit(output, take(b))
```

`peek` reads the next ready FIFO token without removing it and blocks on an
empty place. Token field updates return new token values. `guard` blocks a
firing when false. The Python simulator, C++ interpreter, and native executable
all implement these operations with the existing atomic firing semantics.
In native executables, manual predicates are compiled into the enabling
function and run before the body. Predicates must precede takes, emits, and
effectful helper calls; helpers that only read state may participate in the
predicate. Earlier peeks are checked for availability before they execute.
The synthesis pass preserves explicit predicates as `lpn.guard` and marks
the transition with `lpn.guard_mode = "manual"`. It does not invent take-ID
paths for these predicates. Observable reduction rejects manual predicates
because it cannot yet preserve their value dependencies. These annotations
are trusted enabling conditions; incomplete annotations can still cause a
failed take, which rolls back as a correctness safeguard. Such rollbacks should
be investigated rather than relied on in a performance model.

## FeSens/openTPU model

The active accelerator target is [FeSens/openTPU](examples/opentpu/README.md).
Its LPN model consumes the same encoded programs and operand files as the
Verilator baseline. It models bounded matrix prefetch, systolic row groups,
quantization, vector operations, command dependencies, and simple DRAM
latency/bandwidth. The existing event-driven scheduler is unchanged.

The validation command checks real quantized GEMM and an upstream-compiled
SwiGLU MLP, compares complete numerical memory state, and reports absolute
cycles/second as well as speedup and timing error. See the model README for
the pinned upstream revision, reproduction commands, and current limits.

The model also includes a compact representation and a resumable trained-model
generation runner. Real checkpoint inputs traverse all transformer layers and
update the KV cache, with LPN/RTL memory and timing checks after every token.
The public SmolLM2 bring-up fixture and hour-scale run commands are documented
in the model README. Real Llama 3.2 3B inference remains unvalidated.

## Previous CoralNPU model

The composable timing model and Verilator comparison workflow are documented
in [examples/coralnpu/README.md](examples/coralnpu/README.md). It uses the existing
event-driven simulator and native compiler. It models resource occupancy and
register dependencies from a dynamic instruction stream; measured Verilator
cycle timestamps are used only for comparisons.

The [executable GEMM benchmark](examples/coralnpu/REAL_WORKLOAD.md) reads the
same RISC-V ELF as Verilator, independently executes its supported instructions
to obtain control flow and addresses, and compiles that path into a compact
LPN scoreboard. It checks real BF16 matrix results and compares a region of
over 100,000 RTL cycles. Reports separate the timing run from functional
preprocessing and compilation.

The matrix-enabled model includes CoralNPU's four staggered PE blocks, packed
INT8/BF16/FP32 multiply-accumulate pipelines, accumulator reuse, and a tiled
GEMM model with operand buffers and latency/bandwidth memory transfers. Emit
a native-compilable GEMM with
`python3 -m examples.coralnpu.run --gemm 16 16 64 --dtype int8 --emit gemm.mlir`.
Thirteen full-core Verilator matrix kernels check array timing and 4,096 output
elements; recorded observations are under `examples/coralnpu/validation/matrix`.

The [Llama 3.2 3B timing workload](examples/coralnpu/LLAMA.md) includes all 28
layers, prefill/decode dataflow, KV-cache traffic, and shared execution/memory
resources. Repeated layers reuse the graph through feedback tokens. Whole-model
timing accuracy remains unmeasured; the target is at most 5% error against
representative Verilator workloads.

`--lpn-normalize-delays` replaces every nonzero emit delay with zero. It is for
timing-insensitive structural analysis; do not apply it to IR used for a timed
simulation. The observable-reduction passes are experimental and can change
model behavior, so use the original IR when validating timing and token counts.
