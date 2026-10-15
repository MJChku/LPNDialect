#pragma once
#include "LPN/Analysis/GuardHelpers.h"

// Executable enabling predicates are separate from reduction's take-ID paths:
// repeated constant takes require a count, not merely an operation-site ID.
struct GuardPlan {
  mlir::Operation *lastManualGuard = nullptr;
  std::map<std::string, int64_t> required;
};

class GuardInference {
public:
  GuardPlan analyze(mlir::lpn::TransitionOp transition) {
    name = transition.getSymName().str();
    bool effect = false;
    for (mlir::Operation &op : transition.getBody().front()) {
      if (mlir::isa<mlir::lpn::GuardOp>(op)) {
        if (effect)
          reject("manual guards must precede all takes, emits, and helper calls");
        plan.lastManualGuard = &op;
      }
      op.walk([&](mlir::Operation *nested) {
        if (mlir::isa<mlir::lpn::TakeOp, mlir::lpn::EmitOp>(nested))
          effect = true;
        if (auto call = mlir::dyn_cast<mlir::func::CallOp>(nested))
          effect |= !mlir::lpn::isReadOnlyGuardHelper(call);
        if (mlir::isa<mlir::lpn::GuardOp>(nested) && nested != &op)
          reject("manual guards must be top-level");
      });
    }
    if (!plan.lastManualGuard)
      walk(transition.getBody().front());
    return plan;
  }

private:
  [[noreturn]] void reject(const char *reason) {
    throw std::runtime_error("transition '" + name + "': " + reason +
        "; add an explicit guard(...) enabling predicate before token operations");
  }

  mlir::Attribute attribute(mlir::Value value) {
    auto found = constants.find(value);
    return found == constants.end() ? mlir::Attribute{} : found->second;
  }

  std::optional<int64_t> integerValue(mlir::Value value) {
    auto attr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attribute(value));
    return attr ? std::optional<int64_t>(attr.getValue().getSExtValue())
                : std::nullopt;
  }

  bool hasReads(mlir::Operation &op) {
    bool reads = false;
    op.walk([&](mlir::Operation *nested) {
      if (mlir::isa<mlir::lpn::TakeOp, mlir::lpn::PeekOp,
                    mlir::func::CallOp>(nested))
        reads = true;
    });
    return reads;
  }

  void assignResults(mlir::ValueRange results, mlir::ValueRange yielded) {
    for (auto [result, input] : llvm::zip(results, yielded)) {
      constants[result] = attribute(input);
      if (auto it = places.find(input); it != places.end())
        places[result] = it->second;
    }
  }

  mlir::ValueRange walk(mlir::Block &block) {
    for (mlir::Operation &op : block) {
      if (++visited > 100000)
        reject("automatic guard expansion exceeds its static analysis limit");
      for (mlir::Value result : op.getResults()) {
        constants.erase(result);
        places.erase(result);
      }
      if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(op))
        return yield.getResults();
      if (auto ref = mlir::dyn_cast<mlir::lpn::PlaceRefOp>(op)) {
        places[ref.getResult()] = ref.getPlaceAttr().getValue().str();
      } else if (auto take = mlir::dyn_cast<mlir::lpn::TakeOp>(op)) {
        require(take.getPlace(), true);
      } else if (auto peek = mlir::dyn_cast<mlir::lpn::PeekOp>(op)) {
        require(peek.getPlace(), false);
      } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
        auto condition = integerValue(branch.getCondition());
        if (!condition) {
          if (hasReads(op))
            reject("cannot infer token requirements for a data-dependent branch");
          continue;
        }
        auto &region = *condition ? branch.getThenRegion() : branch.getElseRegion();
        if (!region.empty())
          assignResults(branch.getResults(), walk(region.front()));
      } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        if (!hasReads(op))
          continue;
        auto lower = integerValue(loop.getLowerBound());
        auto upper = integerValue(loop.getUpperBound());
        auto step = integerValue(loop.getStep());
        if (!lower || !upper || !step || *step <= 0)
          reject("cannot infer token requirements for a dynamic loop");
        assignResults(loop.getRegionIterArgs(), loop.getInitArgs());
        for (int64_t iv = *lower; iv < *upper;) {
          constants[loop.getInductionVar()] = mlir::IntegerAttr::get(
              loop.getInductionVar().getType(), iv);
          auto yielded = walk(*loop.getBody());
          assignResults(loop.getRegionIterArgs(), yielded);
          if (iv > std::numeric_limits<int64_t>::max() - *step)
            reject("static loop induction overflows");
          iv += *step;
        }
        assignResults(loop.getResults(), loop.getRegionIterArgs());
      } else if (mlir::isa<mlir::func::CallOp>(op)) {
        // Conservatively require annotations for helper calls. The helper
        // can consume a runtime number of tokens through place arguments.
        reject("helper token requirements need a manual annotation");
      } else if (op.getDialect() && op.getDialect()->getNamespace() == "arith") {
        llvm::SmallVector<mlir::Attribute> inputs;
        for (mlir::Value input : op.getOperands())
          inputs.push_back(attribute(input));
        llvm::SmallVector<mlir::OpFoldResult> folded;
        if (mlir::succeeded(op.fold(inputs, folded))) {
          for (auto [result, foldedResult] : llvm::zip(op.getResults(), folded)) {
            if (auto attr = mlir::dyn_cast<mlir::Attribute>(foldedResult))
              constants[result] = attr;
            else
              constants[result] = attribute(mlir::cast<mlir::Value>(foldedResult));
          }
        } else {
          for (mlir::Value result : op.getResults())
            constants.erase(result);
        }
      }
    }
    return {};
  }

  void require(mlir::Value place, bool consume) {
    auto it = places.find(place);
    if (it == places.end())
      reject("cannot resolve a token operation's place statically");
    int64_t &taken = consumed[it->second];
    plan.required[it->second] = std::max(plan.required[it->second], taken + 1);
    if (consume)
      ++taken;
  }

  std::string name;
  GuardPlan plan;
  unsigned visited = 0;
  llvm::DenseMap<mlir::Value, mlir::Attribute> constants;
  llvm::DenseMap<mlir::Value, std::string> places;
  std::map<std::string, int64_t> consumed;
};
