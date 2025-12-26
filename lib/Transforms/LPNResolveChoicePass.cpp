#include "LPN/Conversion/LPNPasses.h"
#include "LPN/Dialect/LPNOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

namespace mlir::lpn {
namespace {

static bool regionIsTrivial(Region &region) {
  if (region.empty())
    return true;
  Block &block = region.front();
  auto ops = block.without_terminator();
  return ops.empty();
}

static void spliceRegionBefore(Operation *insertionPoint, Region &region) {
  if (region.empty())
    return;
  Block &block = region.front();
  auto &ops = block.getOperations();
  if (ops.empty())
    return;
  auto begin = ops.begin();
  auto end = block.without_terminator().end();
  if (begin == end)
    return;
  insertionPoint->getBlock()->getOperations().splice(
      insertionPoint->getIterator(), ops, begin, end);
}

struct LPNResolveChoicePass
    : PassWrapper<LPNResolveChoicePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNResolveChoicePass)

  StringRef getArgument() const final { return "lpn-resolve-choices"; }
  StringRef getDescription() const final {
    return "Collapse lpn.choice regions by assuming hidden-state branches "
           "always progress.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<ChoiceOp, 8> choices;
    module.walk([&](ChoiceOp op) { choices.push_back(op); });
    for (ChoiceOp choice : choices) {
      bool thenEmpty = regionIsTrivial(choice.getThenRegion());
      bool elseEmpty = regionIsTrivial(choice.getElseRegion());
      if (thenEmpty && elseEmpty) {
        choice.erase();
        continue;
      }
      if (thenEmpty == elseEmpty)
        continue;
      Region &keep = thenEmpty ? choice.getElseRegion()
                               : choice.getThenRegion();
      spliceRegionBefore(choice, keep);
      choice.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLPNResolveChoicePass() {
  return std::make_unique<LPNResolveChoicePass>();
}

} // namespace mlir::lpn
