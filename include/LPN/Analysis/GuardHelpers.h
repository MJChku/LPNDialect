#pragma once

#include "LPN/Dialect/LPNOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::lpn {
// A predicate may call a helper that only inspects state and computes immutable
// values. Unknown, external and recursive calls remain conservatively effectful.
inline bool isReadOnlyGuardHelper(func::CallOp call,
                                 llvm::DenseSet<Operation *> &visiting);

inline bool isReadOnlyGuardFunction(func::FuncOp function,
                                   llvm::DenseSet<Operation *> &visiting) {
  if (!function || function.isExternal() || !visiting.insert(function).second)
    return false;
  bool safe = true;
  function.walk([&](Operation *op) {
    if (auto nested = dyn_cast<func::CallOp>(op)) {
      safe &= isReadOnlyGuardHelper(nested, visiting);
      return;
    }
    if (isa<func::FuncOp,func::ReturnOp,scf::ForOp,scf::IfOp,scf::YieldOp,
            PlaceRefOp,CountOp,PeekOp,ArrayOp,InputArrayOp,ArrayGetOp,ArraySetOp,ArrayLenOp,
            ArrayAllocOp,TokenCreateOp,TokenCloneOp,TokenGetOp,TokenSetOp,
            KeyLiteralOp,KeyRegOp>(op))
      return;
    if (op->getDialect() && op->getDialect()->getNamespace()=="arith")
      return;
    safe = false;
  });
  visiting.erase(function);
  return safe;
}

inline bool isReadOnlyGuardHelper(func::CallOp call,
                                 llvm::DenseSet<Operation *> &visiting) {
  return isReadOnlyGuardFunction(SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call,call.getCalleeAttr()),visiting);
}

inline bool isReadOnlyGuardHelper(func::CallOp call) {
  llvm::DenseSet<Operation *> visiting;
  return isReadOnlyGuardHelper(call, visiting);
}
} // namespace mlir::lpn
