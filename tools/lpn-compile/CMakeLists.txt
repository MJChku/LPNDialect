add_library(LPNNativeRuntime STATIC NativeRuntime.cpp)
set_target_properties(LPNNativeRuntime PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(LPNNativeRuntime PRIVATE cxx_std_17)

add_llvm_executable(lpn-compile PARTIAL_SOURCES_INTENDED lpn-compile.cpp)
add_dependencies(lpn-compile LPNNativeRuntime)
target_compile_options(lpn-compile PRIVATE -fexceptions)
target_compile_definitions(lpn-compile PRIVATE
  LPN_NATIVE_RUNTIME_PATH="$<TARGET_FILE:LPNNativeRuntime>")
target_link_libraries(lpn-compile PRIVATE
  MLIRLPN
  MLIRParser
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  LLVMCore
  LLVMSupport
)
