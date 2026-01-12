//===- AnalysisCommon.h - Shared analysis utilities -------------*- C++ -*-===//
//
// Common types shared by LPN analysis passes.
//
//===----------------------------------------------------------------------===//

#ifndef LPN_ANALYSIS_ANALYSISCOMMON_H
#define LPN_ANALYSIS_ANALYSISCOMMON_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"

namespace mlir {
class Operation;
}

namespace mlir::lpn {

using ObservableSet =
    llvm::DenseSet<StringAttr, llvm::DenseMapInfo<StringAttr>>;

// Key for emit metadata: (emit operation, path index, target place)
struct EmitPathKey {
  mlir::Operation *emit;
  unsigned pathIndex;
  StringAttr target;
};

struct EmitPathKeyInfo : llvm::DenseMapInfo<EmitPathKey> {
  static inline EmitPathKey getEmptyKey() {
    return {llvm::DenseMapInfo<mlir::Operation *>::getEmptyKey(), 0,
            StringAttr()};
  }
  static inline EmitPathKey getTombstoneKey() {
    return {llvm::DenseMapInfo<mlir::Operation *>::getTombstoneKey(), 0,
            StringAttr()};
  }
  static unsigned getHashValue(const EmitPathKey &key) {
    llvm::hash_code targetHash =
        (key.target != StringAttr()) ? llvm::hash_value(key.target)
                                      : llvm::hash_code(0);
    return llvm::hash_combine(
        llvm::DenseMapInfo<mlir::Operation *>::getHashValue(key.emit),
        key.pathIndex, targetHash);
  }
  static bool isEqual(const EmitPathKey &a, const EmitPathKey &b) {
    return a.emit == b.emit && a.pathIndex == b.pathIndex &&
           a.target == b.target;
  }
};

} // namespace mlir::lpn

#endif // LPN_ANALYSIS_ANALYSISCOMMON_H
