find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_test(
  NAME lpn-elf-workload
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}
          PYTHONDONTWRITEBYTECODE=1 ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/test_elf_workload.py
          --tool $<TARGET_FILE:lpn-compile>
)

add_test(
  NAME lpn-llama-model
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}
          PYTHONDONTWRITEBYTECODE=1 ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/test_llama_model.py
          --tool $<TARGET_FILE:lpn-compile>
)

add_test(
  NAME lpn-coralnpu-matrix
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}
          PYTHONDONTWRITEBYTECODE=1 ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/test_matrix_model.py
)

add_test(
  NAME lpn-model-runtime
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}
          PYTHONDONTWRITEBYTECODE=1
          ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_model_runtime.py
)

add_test(
  NAME lpn-coralnpu-model
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}
          PYTHONDONTWRITEBYTECODE=1 ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/test_coralnpu.py
)

add_test(
  NAME lpn-native-compiler
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}
          PYTHONDONTWRITEBYTECODE=1 ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/test_native_compiler.py
          --tool $<TARGET_FILE:lpn-compile>
)

add_test(
  NAME lpn-guard-rejection
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}
          PYTHONDONTWRITEBYTECODE=1 ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/test_guard_rejection.py
          --tool $<TARGET_FILE:lpn-opt>
)

find_program(FILECHECK_EXE
  NAMES FileCheck
  HINTS ${LLVM_TOOLS_BINARY_DIR}
)
if (NOT FILECHECK_EXE)
  find_program(FILECHECK_EXE FileCheck)
endif()
if (NOT FILECHECK_EXE)
  message(WARNING "FileCheck executable not found; LPN tests disabled")
  return()
endif()

set(LPN_OPT_EXE $<TARGET_FILE:lpn-opt>)
set(LPN_TEST_DRIVER ${CMAKE_CURRENT_SOURCE_DIR}/run_filecheck.py)

function(add_lpn_filecheck_test name input)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs TOOL_ARGS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_test(
    NAME ${name}
    COMMAND ${Python3_EXECUTABLE} ${LPN_TEST_DRIVER}
            --tool ${LPN_OPT_EXE}
            --filecheck ${FILECHECK_EXE}
            --input ${input}
            ${ARG_TOOL_ARGS}
  )
endfunction()

add_lpn_filecheck_test(
  lpn-normalize-delays
  ${CMAKE_CURRENT_SOURCE_DIR}/Transforms/normalize-delays.mlir
  TOOL_ARGS --tool-arg=--lpn-normalize-delays
)

add_lpn_filecheck_test(
  lpn-synthesize-guards
  ${CMAKE_CURRENT_SOURCE_DIR}/Transforms/synthesize-guards.mlir
  TOOL_ARGS --tool-arg=--lpn-synthesize-guards
)
