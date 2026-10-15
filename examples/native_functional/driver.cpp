// Both kernels are Python-authored, compiled native libraries. This executable
// runs their entire functional/timing path without a Python interpreter.
#include "LPN/Runtime/NativeInterface.h"
#include <cstdint>
#include <vector>

template <typename T> struct Memref1D {
  T *allocated, *aligned;
  int64_t offset;
  int64_t sizes[1], strides[1];
};

extern "C" int64_t _mlir_ciface_resolve_transfers(
    Memref1D<uint32_t> *, Memref1D<int64_t> *, int64_t, Memref1D<int64_t> *);

int main(int argc, char **argv) {
  std::vector<uint32_t> instructions{1,8,16};
  std::vector<int64_t> resolved(instructions.size());
  int64_t error = 0;
  Memref1D<uint32_t> program{instructions.data(),instructions.data(),0,
                           {int64_t(instructions.size())},{1}};
  Memref1D<int64_t> sizes{resolved.data(),resolved.data(),0,
                         {int64_t(resolved.size())},{1}};
  Memref1D<int64_t> status{&error,&error,0,{1},{1}};
  int64_t count = _mlir_ciface_resolve_transfers(&program,&sizes,resolved.size(),&status);
  if (error || count!=int64_t(resolved.size())) return 3;
  if (lpn_bind_input_array("transfer_sizes",count,
                          reinterpret_cast<const uint64_t *>(resolved.data()))) return 4;
  int result = lpn_simulate(argc,argv);
  if (!result && lpn_simulated_time()!=28) result = 5;
  lpn_clear_input_arrays();
  return result;
}
