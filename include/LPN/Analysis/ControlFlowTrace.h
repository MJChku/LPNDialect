//===- ControlFlowTrace.h - Analysis for LPN control flow paths -*- C++ -*-===//
//
// Analysis to recover control flow decisions (if/choice branches) that lead
// to operations within an LPN transition.
//
//===----------------------------------------------------------------------===//

#ifndef LPN_ANALYSIS_CONTROLFLOWTRACE_H
#define LPN_ANALYSIS_CONTROLFLOWTRACE_H

#include "LPN/Dialect/LPNOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::lpn {

/// Represents a single control decision point.
struct ControlContext {
  Operation *op = nullptr;
  enum class Kind { IfOp, ChoiceOp, ForOp, Unknown } kind = Kind::Unknown;
  bool inThen = false; // Relevant for IfOp and ChoiceOp

  bool operator==(const ControlContext &other) const {
    return op == other.op && kind == other.kind && inThen == other.inThen;
  }
  bool operator!=(const ControlContext &other) const {
    return !(*this == other);
  }
};

/// Analysis that provides the control flow path for operations within a Transition.
class ControlFlowTrace {
public:
  /// Initialize the analysis for a given transition.
  /// Currently this performs on-demand tracing, so initialization is cheap.
  explicit ControlFlowTrace(TransitionOp trans) : transition(trans) {}

  /// Returns the sequence of control decisions enclosing this operation,
  /// ordered from outermost to innermost.
  /// Result is cached for subsequent queries on the same operation.
  llvm::ArrayRef<ControlContext> getTrace(Operation *op);

private:
  TransitionOp transition;
  llvm::DenseMap<Operation *, llvm::SmallVector<ControlContext, 4>> cache;
};

} // namespace mlir::lpn

#endif // LPN_ANALYSIS_CONTROLFLOWTRACE_H
