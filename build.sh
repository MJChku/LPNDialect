#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${LLVM_DIR:-}" || -z "${MLIR_DIR:-}" ]]; then
	echo "LLVM_DIR and MLIR_DIR must be set before running build.sh" >&2
	exit 1
fi

CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-RelWithDebInfo}

cmake -S . -B build \
	-DLLVM_DIR="${LLVM_DIR}" \
	-DMLIR_DIR="${MLIR_DIR}" \
	-DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"

cmake --build build "$@"