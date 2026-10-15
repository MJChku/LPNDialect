# Native functional models

`lpnfunc` is a small typed Python authoring frontend for native functional
prepasses. A prepass computes values, control flow, effective addresses and
the metadata consumed by the LPN timing model. The LPN remains event driven
and synchronizes components through dataflow tokens.

The compilation path is:

```text
annotated Python -> func/scf/arith/memref MLIR -> LLVM IR -> native shared library
```

Standard MLIR operations suffice for this first subset; there is no new
custom dialect or C++ source generation. Execution has native loops and
arithmetic, with one Python/ctypes call per kernel and no Python instruction
callbacks. NumPy buffers are passed without copying. The library can also be
called from C++ using the emitted MLIR C interface and ranked memref
descriptors; see the [LLVM lowering documentation](https://mlir.llvm.org/docs/TargetLLVMIR/).

## Author and compile

Run from the repository root with NumPy and `mlir-opt-20`,
`mlir-translate-20`, and `clang++-20` installed. Define functions in a source
file: the frontend uses `inspect.getsource`, so interactive definitions
without available source do not work.

```python
import numpy as np
from lpnfunc import functional, Buffer, u32, i64

@functional
def copy_words(source: Buffer[u32], destination: Buffer[u32], n: i64) -> i64:
    for j in range(n):
        destination[j] = source[j]
    return n

source = np.arange(32, dtype=np.uint32)
destination = np.zeros_like(source)
native = copy_words.compile()  # Compiles once; subsequent builds use a cache.
native(source, destination, len(source))
print(copy_words.emit())
```

Calling `copy_words(...)` executes its Python body as a reference. Calling the
compiled object executes native code. Python local annotations alone do not
round or wrap intermediate values: use explicit `u32(...)`, `i32(...)`,
`i64(...)`, and `f32(...)` casts where reference arithmetic must reproduce
fixed-width native arithmetic. Validate numerical kernels against their
independent reference before using them for real workloads.

Supported types are `u32`, `i32`, `i64`, `f32`, `boolean`, and contiguous,
aligned, writable one-dimensional `Buffer[T]` arrays. Buffers may alias;
execution preserves sequential reads and writes. All scalar arguments and
locals need types; `range` variables default to `i64`. Supported constructs
include `if/else`, `while`, positive-step `range`, arithmetic and bit
operations, short-circuit boolean `and/or`, buffer loads/stores, `len`,
explicit scalar casts, and 32-bit float/integer `bitcast`.

The subset requires a single final return. Classes, arbitrary Python calls,
early returns, `break`, negative range steps, signed modulo, and numerical
float-to-integer casts are not supported. Unsupported syntax fails during
compilation; there is no interpreter fallback. Boolean `and/or` require
boolean operands, avoiding Python's value-returning integer semantics.
Buffer indices must be nonnegative; Python negative-index shorthand is not
implemented. Native integers wrap at their annotated width; invalid shifts,
division by zero, and signed division overflow raise `ValueError`.

Buffer accesses are bounds checked. An invalid access stops further buffer
writes and loop iterations and raises `IndexError` when the kernel returns.
Earlier writes remain visible. Float operations use `f32` without fast-math
flags or contraction; this alone does not establish bit equivalence for
arbitrary Python/NumPy or RTL numerical routines.

The compilation cache defaults to `~/.cache/lpn-functional`; pass a directory
to `.compile(cache_dir)` to override it. Its key includes emitted IR, tool
versions and compile settings. A lock and atomic library replacement protect
concurrent compilation. The cached MLIR and LLVM IR are retained for review.

## FeSens scalar prototype and validation

[`native_scalar.py`](../examples/opentpu/native_scalar.py) ports one-slice
NOP, HALT, LI, ADDI, LOOP, S1BAR, RLD, LD and ST execution. It supports nested
hardware loops, data-dependent loop counts, register-relative addresses and
bit-exact RLD conversion. It emits packed resolved instruction records, then
adapts them to the existing LPN `Workload` format. **That adapter still creates
Python Command/footprint objects per record.** Matrix, quantizer and vector
operations explicitly fail; this prototype does not yet run SmolLM2's full
functional prepass and is not enabled in the existing trained-model runner.

```sh
PYTHONPATH=. ../FeSens-openTPU/.venv/bin/python test/test_functional.py
PYTHONPATH=. ../FeSens-openTPU/.venv/bin/python \
  test/test_native_functional_opentpu.py --opentpu-root ../FeSens-openTPU
PYTHONPATH=. ../FeSens-openTPU/.venv/bin/python \
  -m examples.opentpu.benchmark_scalar --opentpu-root ../FeSens-openTPU \
  --jobs 8000 --runs 5 --output /tmp/native-scalar.json
```

The benchmark compares identical inputs, all DRAM/TMEM, registers, executed
PCs, resolved instructions and timing footprints against the independent
upstream ISA simulator. It times the full reference prepass, the native
prepass plus Python adapter, and raw kernel execution separately. The raw
kernel region excludes allocation/reset, assembly, adaptation and compilation.
It is a scalar/control microbenchmark, not a transformer or RTL speedup claim.

On `icdslab2`, the [recorded 32,005-instruction run](../examples/opentpu/validation/native_scalar/comparison.json)
took median **1.093 s** in the reference prepass and **0.371 s** in the native
prepass with its adapter: **2.95x**. The raw kernel took **0.933 ms**, showing
that Python trace adaptation dominates the new path. Initial compilation
took 0.487 s, outside execution samples. Measurements ran alongside the
existing long RTL validation and are indicative rather than isolated-host
performance measurements.

For the current SmolLM2 path, an illustrative step is about 1.8 s functional
execution, 1.3 s graph construction, 1.2 s compilation and 2.3 s timing
execution. If the *entire* functional prepass became 3–10x faster, total time
would fall from 6.6 s to 5.4–5.0 s: **1.22–1.33x end to end**. Removing that
prepass cost entirely caps this change at 1.38x. These are projections, not
measurements of the scalar prototype on SmolLM2. Some NumPy numerical work
already runs natively; further gains need native numerical kernels and
metadata consumption, plus separate work on graph/build overhead.
