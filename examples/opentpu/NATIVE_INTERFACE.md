# Reusable native timing and functional interfaces

The functional prepass and LPN timing remain separate stages. Immutable named
arrays form a general interface between them. Producers can be C++, compiled
Python-authoring code, or Python/NumPy; timing execution makes no Python
callbacks. Inputs contain resolved data and dependency metadata, never observed
RTL completion times.

## Compile a timing model once

`InputArray` supplies reference values to Python execution and emits
`lpn.input_array "name" : !lpn.array<i64>` for native compilation:

```python
from lpnlang_mlir import InputArray

sizes = InputArray("transfer_sizes", resolved_sizes)
# Inside a decorated transition:
# bytes_to_transfer = array(sizes)[command_id]
```

Native input arrays are immutable during a simulation. Missing names and
out-of-bounds accesses fail explicitly. `array_set` creates a copy and never
modifies the producer's binding. The Python reference uses `InputArray.values`;
the C++ MLIR interpreter currently rejects `lpn.input_array`.

Compile a standalone executable as usual, or a library with:

```sh
build/tools/lpn-compile/lpn-compile timing.mlir -o timing.so --shared
```

The shared model exports `lpn_simulate` and the binding API from
[NativeInterface.h](../../include/LPN/Runtime/NativeInterface.h). A C++ producer
can execute its functional kernel and bind the resulting immutable buffers:

```cpp
std::vector<uint64_t> sizes = resolve_commands();
lpn_bind_input_array("transfer_sizes", sizes.size(), sizes.data());
lpn_simulate(argc, argv);
double cycles = lpn_simulated_time();
lpn_clear_input_arrays();
```

Bindings borrow the buffer; its storage must remain alive and unchanged until
simulation returns. Binding/clear operations return 0 on success and 2 for an
invalid argument or active run; callers should check these results. Bindings
persist across sequential runs until replaced or cleared. Simulation marking,
static caches and time reset for each run. Calls to one model/runtime must be
serialized; this interface does not make the scheduler thread safe. Invalid
runtime inputs retain the standalone runtime's fatal-error behavior.

The [native functional example](../native_functional/README.md) compiles a typed
Python producer and a timing net, then links both into one C++ executable. Its
test runs that executable with no Python interpreter in `PATH`.

## Standalone input-table files

Python hosts can write the same arrays in a versioned binary file:

```python
from lpnlang_mlir.native_inputs import write_input_tables
write_input_tables("commands.tables", {"transfer_sizes": resolved_sizes})
```

```sh
./timing.native --inputs=commands.tables --stats
```

The format is little endian: eight magic bytes `LPNTBL1\0`, a u64 table count,
then for each table a u64 UTF-8 name length, u64 element count, name bytes, and
u64 raw element bits. Names are nonempty, at most 1024 bytes, contain no NUL,
and must be unique. There are at most 10,000 tables; the native file reader
limits total input size to 1 GiB and rejects truncation/trailing data. Integer
values retain their low 64 bits. Noninteger Python values are rejected.
The Python reference interprets those bits as signed i64, matching native code.
Use separate input paths for concurrently running simulations.

## Reusable openTPU timing

`CompactOpenTPU(..., input_capacity=N)` externalizes its command timing,
dispatch and CSR dependency tables. The capacity fixes the completion-bank
count, allowing the same compiled architecture to accept different command
counts and values. Overflow is an explicit preparation error. Matrix buffer
credits are initialized from runtime metadata; they are not baked in from the
first trace. Engine count, command-window size and other embedded architecture
changes produce a different compilation-cache key.

```python
from examples.opentpu.native_timing import ReusableTiming
from examples.opentpu.model import Timing

runner = ReusableTiming("build/tools/lpn-compile/lpn-compile",
                        "/tmp/timing-cache", capacity=8192)
prepared = runner.prepare(workload, Timing(batch_rows=8,
                          dram_bytes_per_cycle=64), "/tmp/commands.tables")
cycles, core_seconds, process_seconds, log = prepared.run()
```

The cache key includes normalized timing MLIR, compiler and runtime hashes.
Compilation is locked and the finished binary is published atomically. SSA
builder counters are normalized without changing their function-local meaning.
Preparation still computes metadata in Python, but emits a small architecture
instead of embedding thousands of per-instruction constant arrays. The packed
tables are runtime data; repeated streams reuse the timing executable.

The trained-model runner supports this path with `--reuse-timing` and
`--command-capacity` (default 8192). For example, add those arguments to the
existing SmolLM2 command, using a compiler built from this branch. Use a new
output directory; a frozen run cannot resume with changed sources/options.

The current FeSens adapter still uses Python/NumPy numerical execution and
Python command/dependency preparation. The native scalar frontend covers only
scalar/control and LD/ST. Matrix, quantization and vector numerical kernels
need native ports before the *entire* trained-model functional prepass runs
without Python. This interface provides their connection to timing; it does
not claim that those ports are complete.

## Validation and measurements

The tests reuse one executable across changed array lengths/values, exercise a
C++ producer against the shared model over repeated runs, reject malformed
tables, and compare reusable openTPU predictions across command-count changes
and completion-bank boundaries. Existing native/runtime/timing suites remain
part of CTest.

To repeat the full-path measurement on a frozen input bundle:

```sh
PYTHONPATH=. ../FeSens-openTPU/.venv/bin/python \
  -m examples.opentpu.benchmark_reuse --opentpu-root ../FeSens-openTPU \
  --input /path/to/frozen-input --output /tmp/reuse-comparison \
  --compiler build/tools/lpn-compile/lpn-compile \
  --reference-root /path/to/frozen-old-LPNDialect \
  --reference-compiler /path/to/frozen-old-LPNDialect/build/tools/lpn-compile/lpn-compile
```

It alternates engines after warmup and checks identical cycles, input hashes
and every numerical DRAM/TMEM byte via hashes. The timing scope includes
resident-input functional execution, model/metadata preparation, compilation
or cache lookup, and the native timing process. Initial file loading/package
imports are outside that scope, matching the trained runner's measurements.
The cache warmup cost is recorded separately. Measurements on a frozen step
are not a completed long trained-model run.

On the saved real SmolLM2-135M prefill input, three alternating samples after
warmup gave these medians on `icdslab2`:

| Cost | Original path | Reusable/component path |
| --- | ---: | ---: |
| Functional execution | 1.830 s | 1.853 s |
| Model/metadata preparation | 1.252 s | 0.334 s |
| Compilation/cache lookup | 1.189 s | 0.000075 s |
| Native timing process | 2.366 s | 1.423 s |
| Full LPN path | 6.643 s | 3.608 s |

The median full-path improvement is **1.84x**; full-path medians are computed
from each complete sample, not by adding component medians. All numerical
memory hashes and the prediction of **2,238,884 cycles** match. The identical
input was previously validated in RTL at 2,194,095 cycles (+2.041% LPN error).
These host timings ran alongside RTL validation. The
[report](validation/native_reuse/comparison.json) records warmups, samples,
input/numerical hashes and compiler/binary/IR hashes.

A separate [trained-model pilot](validation/native_reuse_pilot/report.json)
validated three steps, including a process restart/resume, at revision
`1f2067b`. It produced " you have" from "Once". All DRAM/TMEM matched RTL bit
for bit, cycle error stayed within 1.251%, and the second/third steps reused
the timing executable, including after restart. The pilot is intentionally
bounded; the original long validation continues on its frozen revision.
