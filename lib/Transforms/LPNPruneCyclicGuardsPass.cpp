#include "LPN/Analysis/GuardTransitiveClosure.h"
#include "LPN/Conversion/LPNPasses.h"
#include "LPN/Dialect/LPNOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::lpn {
namespace {

static constexpr StringLiteral kGuardIdAttr = "lpn.guard_id";
static constexpr StringLiteral kGuardPathsAttr = "lpn.guard_paths";

struct LPNPruneCyclicGuardsPass
    : PassWrapper<LPNPruneCyclicGuardsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNPruneCyclicGuardsPass)

  StringRef getArgument() const final { return "lpn-prune-cyclic-guards"; }
  StringRef getDescription() const final {
    return "Prune guard IDs that only resolve through cyclic hidden paths.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](NetOp net) {
      if (failed(processNet(net)))
        signalPassFailure();
    });
  }

  LogicalResult processNet(NetOp net) {
    ObservableSet observables;
    for (PlaceOp place : net.getOps<PlaceOp>()) {
      if (place.getObservableAttr())
        observables.insert(place.getSymNameAttr());
    }

    if (observables.empty())
      return success();

    llvm::DenseMap<Value, StringAttr> takePlaces;
    net.walk([&](TakeOp take) {
      if (auto ref = take.getPlace().getDefiningOp<PlaceRefOp>())
        takePlaces[take.getResult()] = ref.getPlaceAttr().getAttr();
    });
    Builder builder(net.getContext());
    bool changed = true;
    while (changed) {
      changed = false;
      GuardTransitiveClosureResult guardPaths;
      if (failed(
              runGuardTransitiveClosureAnalysis(net, observables, guardPaths)))
        return failure();

      for (TransitionOp trans : net.getOps<TransitionOp>()) {
        llvm::DenseMap<unsigned, Value> guardIdToTake;
        trans.walk([&](TakeOp take) {
          auto guardAttr = take->getAttrOfType<IntegerAttr>(kGuardIdAttr);
          if (!guardAttr)
            return;
          guardIdToTake[guardAttr.getInt()] = take.getResult();
        });

        trans.walk([&](EmitOp emit) {
          auto guardsAttr = emit->getAttrOfType<ArrayAttr>(kGuardPathsAttr);
          if (!guardsAttr || guardsAttr.empty())
            return;

          bool emitChanged = false;
          llvm::SmallVector<Attribute, 4> newPaths;
          newPaths.reserve(guardsAttr.size());

          for (Attribute attr : guardsAttr) {
            auto pathAttr = llvm::dyn_cast<ArrayAttr>(attr);
            if (!pathAttr) {
              newPaths.push_back(attr);
              continue;
            }

            bool pathChanged = false;
            llvm::SmallVector<Attribute, 4> newGuards;
            for (Attribute elem : pathAttr) {
              auto intAttr = llvm::dyn_cast<IntegerAttr>(elem);
              if (!intAttr)
                continue;
              auto it = guardIdToTake.find(intAttr.getInt());
              if (it == guardIdToTake.end())
                continue;

              Value takeValue = it->second;
              auto placeIt = takePlaces.find(takeValue);
              if (placeIt == takePlaces.end()) {
                newGuards.push_back(elem);
                continue;
              }
              if (placeIt != takePlaces.end() &&
                  observables.contains(placeIt->second)) {
                newGuards.push_back(elem);
                continue;
              }

              if (!guardPaths.hasCycle(takeValue)) {
                newGuards.push_back(elem);
                continue;
              }

              pathChanged = true;
            }

            // Avoid erasing an entire guard list; only prune individual guards.
            if (newGuards.empty() && !pathAttr.empty()) {
              newPaths.push_back(pathAttr);
              pathChanged = false;
            } else {
              newPaths.push_back(builder.getArrayAttr(newGuards));
            }
            if (pathChanged)
              emitChanged = true;
          }

          if (emitChanged) {
            emit->setAttr(kGuardPathsAttr, builder.getArrayAttr(newPaths));
            changed = true;
          }
        });
      }
    }

    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createLPNPruneCyclicGuardsPass() {
  return std::make_unique<LPNPruneCyclicGuardsPass>();
}

} // namespace mlir::lpn
