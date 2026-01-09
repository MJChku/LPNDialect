//===- ControlFlowTrace.cpp - Analysis for LPN control flow paths ---------===//
//
// Implementation of ControlFlowTrace analysis.
//
//===----------------------------------------------------------------------===//

#include "LPN/Analysis/ControlFlowTrace.h"
#include "LPN/Dialect/LPNOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"

using namespace mlir;
using namespace mlir::lpn;

static bool blockInRegion(Block *block, Region &region) {
  for (Region *current = block ? block->getParent() : nullptr; current;
       current = current->getParentRegion())
    if (current == &region)
      return true;
  return false;
}

ArrayRef<ControlContext> ControlFlowTrace::getTrace(Operation *op) {
  if (auto it = cache.find(op); it != cache.end())
    return it->second;

  SmallVector<ControlContext, 4> trace;
  Operation *originalOp = op;
  Operation *parent = op->getParentOp();
  
  // Walk up the parent chain until we hit the transition or run out of parents.
  while (parent && parent != transition) {
    Block *block = op->getBlock();
    
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      Region &thenRegion = ifOp.getThenRegion();
      bool inThen = blockInRegion(block, thenRegion);
      trace.push_back({ifOp.getOperation(), ControlContext::Kind::IfOp, inThen});
    } else if (auto choice = dyn_cast<ChoiceOp>(parent)) {
      Region &thenRegion = choice.getThenRegion();
      bool inThen = blockInRegion(block, thenRegion);
      trace.push_back({choice.getOperation(), ControlContext::Kind::ChoiceOp, inThen});
    } else if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      trace.push_back({forOp.getOperation(), ControlContext::Kind::ForOp, true});
    }

    op = parent;
    parent = parent->getParentOp();
  }

  // Parents are visited innermost to outermost, reverse to get outer -> inner.
  std::reverse(trace.begin(), trace.end());
  
  // Store in cache
  return cache[originalOp] = std::move(trace);
}
