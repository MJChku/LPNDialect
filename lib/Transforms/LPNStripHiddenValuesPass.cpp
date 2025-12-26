#include "LPN/Conversion/LPNPasses.h"
#include "LPN/Dialect/LPNOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "mlir/Pass/Pass.h"

namespace mlir::lpn {
namespace {

static Value buildUnknownValue(Type type, Location loc,
                               OpBuilder &builder) {
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    auto attr = builder.getIntegerAttr(intTy, 0);
    return builder.create<arith::ConstantOp>(loc, attr).getResult();
  }
  if (auto floatTy = dyn_cast<FloatType>(type)) {
    auto attr = builder.getFloatAttr(floatTy, 0.0);
    return builder.create<arith::ConstantOp>(loc, attr).getResult();
  }
  if (isa<IndexType>(type)) {
    auto attr = builder.getIndexAttr(0);
    return builder.create<arith::ConstantOp>(loc, attr).getResult();
  }
  return Value();
}

struct LPNStripHiddenValuesPass
    : PassWrapper<LPNStripHiddenValuesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNStripHiddenValuesPass)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect>();
  }

  StringRef getArgument() const final {
    return "lpn-strip-hidden-values";
  }

  StringRef getDescription() const final {
    return "Replace values depending on hidden tokens with symbolic unknowns.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    WalkResult walkResult = module.walk([&](TokenGetOp get) -> WalkResult {
      if (!get->hasAttr("lpn.hidden_value"))
        return WalkResult::advance();
      OpBuilder builder(get);
      Value replacement =
          buildUnknownValue(get.getResult().getType(), get.getLoc(), builder);
      if (!replacement) {
        get.emitError("unsupported hidden value type for stripping");
        return WalkResult::interrupt();
      }
      get.getResult().replaceAllUsesWith(replacement);
      get.erase();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLPNStripHiddenValuesPass() {
  return std::make_unique<LPNStripHiddenValuesPass>();
}

} // namespace mlir::lpn
