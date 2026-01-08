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
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::lpn {

class NetOp;

/// Records the observable guard combinations that can satisfy a guard id.
struct GuardTransitiveClosureResult {
  using GuardCombination = llvm::SmallVector<Value, 4>;

  /// Closure for each guard-producing take value.
  llvm::DenseMap<Value, llvm::SmallVector<GuardCombination, 4>> guardClosures;

  /// Returns the closures for `takeValue`. If none exist, returns an empty
  /// array reference.
  llvm::ArrayRef<GuardCombination> getClosures(Value takeValue) const;
};

/// Run the guard transitive closure analysis.
LogicalResult runGuardTransitiveClosureAnalysis(
    NetOp net, const ObservableSet &observables,
    GuardTransitiveClosureResult &result);

} // namespace mlir::lpn

#endif // LPN_ANALYSIS_GUARDTRANSITIVECLOSURE_H
