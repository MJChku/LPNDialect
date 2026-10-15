# Native functional producer linked to LPN timing

Both stages are authored in Python. `lpnfunc` compiles the functional subset
through standard MLIR and LLVM into a shared library. `lpn-compile --shared`
compiles the timing net into another library. A C++ driver calls the producer,
binds its output array without copying, and runs the timing model. The resulting
executable requires no Python interpreter at runtime.

```sh
PYTHONPATH=. python3 -m examples.native_functional.build \
  --lpn-compiler build/tools/lpn-compile/lpn-compile \
  --output /tmp/native-functional
/tmp/native-functional/native-functional-timing --stats
```

This small interface example resolves three instructions into transfer sizes
of 4, 32 and 64 bytes. The timing net applies 32 bytes/cycle bandwidth and an
8-cycle latency, predicting 28 cycles total. It demonstrates the linking
mechanism, not a full accelerator or neural-network functional model.

See [model.py](model.py) for the Python authoring and [driver.cpp](driver.cpp)
for the C++ call/ownership sequence. The functional entry uses MLIR's ranked
memref C ABI. Its final hidden memref reports bounds/arithmetic errors. The
timing entry uses [NativeInterface.h](../../include/LPN/Runtime/NativeInterface.h).

For general runtime inputs and cached openTPU timing models, see
[NATIVE_INTERFACE.md](../opentpu/NATIVE_INTERFACE.md). For the supported
functional language subset and numerical restrictions, see
[lpnfunc/README.md](../../lpnfunc/README.md).
