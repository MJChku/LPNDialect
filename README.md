# LPNDialect

LPNDialect is a Python modeling DSL and an MLIR dialect for labeled Petri nets.
Transitions take tokens from places, update token fields, and emit tokens after
delays. The native compiler lowers the model's MLIR operations to LLVM IR,
generates an object file, and links it with a token/event runtime. The resulting
executable runs compiled transitions without an MLIR interpreter.

Agents working on models or the compiler should follow [SKILL.md](SKILL.md)
for build commands, modeling semantics, and validation workflows.

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
executable accepts `--max-time=NUMBER` and `--tokens`. Its default time limit is
100 million model time units; `--tokens` prints token fields as well as place
counts. `lpn-opt --lpn-simulate model.mlir` remains available as an interpreting
C++ reference implementation.

The native compiler currently accepts one `lpn.net` with place references,
token operations, arrays, integer and floating-point arithmetic, `scf.if`,
`scf.for`, and `func.func` helpers. It reports unsupported operations instead
of silently interpreting them. Place capacities are rejected by the native
compiler because bounded-place semantics are not implemented. The Python and
C++ interpreters still ignore capacities.

## Guard synthesis

`take` operations themselves determine whether a native transition can fire;
the native compiler needs no separate guard pass. For observable-reduction
passes, `--lpn-synthesize-guards` infers guard metadata automatically. It
handles nested one-shot takes under compile-time constant conditions. It
rejects token flow in repeating or dynamic loops and inside helper functions:
the current guard ID format cannot represent those operations' execution
counts. The all-to-all example uses repeated takes, so compile its original
MLIR directly.
It also rejects token flow inside data-dependent `scf.if` branches because
take IDs alone cannot encode a branch predicate derived from a token field.
The native compiler executes such branches and their nested takes directly.

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
The synthesis pass preserves explicit predicates as `lpn.guard` and marks
the transition with `lpn.guard_mode = "manual"`. It does not invent take-ID
paths for these predicates. Observable reduction rejects manual predicates
because it cannot yet preserve their value dependencies. These annotations
are trusted enabling conditions; failed takes still roll back the firing.

## CoralNPU model

The composable timing model and Verilator comparison workflow are documented
in [examples/coralnpu/README.md](examples/coralnpu/README.md). It uses the existing
event-driven simulator and native compiler. It models resource occupancy and
register dependencies from a dynamic instruction stream; measured Verilator
cycle timestamps are used only for comparisons.

The matrix-enabled model includes CoralNPU's four staggered PE blocks, packed
INT8/BF16/FP32 multiply-accumulate pipelines, accumulator reuse, and a tiled
GEMM model with operand buffers and latency/bandwidth memory transfers. Emit
a native-compilable GEMM with
`python3 -m examples.coralnpu.run --gemm 16 16 64 --dtype int8 --emit gemm.mlir`.
Eleven full-core Verilator matrix kernels check array timing and 3,584 output
elements; recorded observations are under `examples/coralnpu/validation/matrix`.

`--lpn-normalize-delays` replaces every nonzero emit delay with zero. It is for
timing-insensitive structural analysis; do not apply it to IR used for a timed
simulation. The observable-reduction passes are experimental and can change
model behavior, so use the original IR when validating timing and token counts.
