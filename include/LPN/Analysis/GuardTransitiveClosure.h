//===- GuardTransitiveClosure.h - Guard closure for LPN ------*- C++ -*-===//
//
// Compute transitive closure of guard IDs so hidden places can be expressed
// in terms of observable takes.
//
//===----------------------------------------------------------------------===//

#ifndef LPN_ANALYSIS_GUARDTRANSITIVECLOSURE_H
#define LPN_ANALYSIS_GUARDTRANSITIVECLOSURE_H

#include "LPN/Analysis/AnalysisCommon.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::lpn {

class NetOp;

/// Describes a single emit-path segment that produces a guard take.
struct GuardPathSegment {
  Operation *emit = nullptr;
  unsigned pathIndex = 0;
  StringAttr target;
  Value producedTake;
};

/// Represents one concrete way to satisfy a guard take.
struct GuardPathVariant {
  llvm::SmallVector<Value, 4> observables;
  llvm::SmallVector<GuardPathSegment, 4> segments;
};

/// Records the observable guard combinations (with path context) that can
/// satisfy a guard id.
struct GuardTransitiveClosureResult {
  llvm::DenseMap<Value, llvm::SmallVector<GuardPathVariant, 4>> guardPaths;

  /// Returns the path variants for `takeValue`. If none exist, returns an
  /// empty array reference.
  llvm::ArrayRef<GuardPathVariant> getPaths(Value takeValue) const;
};

/// Run the guard transitive closure analysis.
LogicalResult runGuardTransitiveClosureAnalysis(
    NetOp net, const ObservableSet &observables,
    GuardTransitiveClosureResult &result);

} // namespace mlir::lpn

#endif // LPN_ANALYSIS_GUARDTRANSITIVECLOSURE_H
