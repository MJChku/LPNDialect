#include "LPN/Conversion/LPNPasses.h"
#include "LPN/Analysis/GuardHelpers.h"
#include "LPN/Dialect/LPNOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>

namespace mlir::lpn {
namespace {

static constexpr StringLiteral kGuardIdAttr = "lpn.guard_id";
static constexpr StringLiteral kGuardPathsAttr = "lpn.guard_paths";

using GuardSet = SmallVector<unsigned, 4>;
struct GuardState {
  GuardSet guards;
  SmallVector<Operation *, 4> emits;

  bool operator==(const GuardState &other) const {
    return guards == other.guards && emits == other.emits;
  }
};
using GuardStateVector = SmallVector<GuardState, 4>;

enum class LoopTripCount { Zero, One, ManyOrUnknown };

static std::optional<int64_t> constantInt(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto attr = dyn_cast<IntegerAttr>(constant.getValue());
  if (!attr)
    return std::nullopt;
  return attr.getValue().getSExtValue();
}

static std::optional<bool> constantBool(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant || !constant.getType().isInteger(1))
    return std::nullopt;
  auto attr = dyn_cast<IntegerAttr>(constant.getValue());
  return attr ? std::optional<bool>(attr.getValue().getBoolValue())
              : std::nullopt;
}

static LoopTripCount classifyLoop(scf::ForOp loop) {
  auto lower = constantInt(loop.getLowerBound());
  auto upper = constantInt(loop.getUpperBound());
  auto step = constantInt(loop.getStep());
  if (!lower || !upper || !step || *step <= 0)
    return LoopTripCount::ManyOrUnknown;
  if (*lower >= *upper)
    return LoopTripCount::Zero;
  uint64_t span = static_cast<uint64_t>(*upper) -
                  static_cast<uint64_t>(*lower);
  return span <= static_cast<uint64_t>(*step)
             ? LoopTripCount::One : LoopTripCount::ManyOrUnknown;
}

static void insertGuard(GuardSet &set, unsigned guardId) {
  auto it = llvm::lower_bound(set, guardId);
  if (it == set.end() || *it != guardId)
    set.insert(it, guardId);
}

static void dedupGuardStates(GuardStateVector &states) {
  llvm::sort(states, [](const GuardState &a, const GuardState &b) {
    if (a.guards != b.guards)
      return std::lexicographical_compare(a.guards.begin(), a.guards.end(),
                                          b.guards.begin(), b.guards.end());
    return std::lexicographical_compare(a.emits.begin(), a.emits.end(),
                                        b.emits.begin(), b.emits.end(),
                                        std::less<Operation *>{});
  });
  auto newEnd = std::unique(states.begin(), states.end(),
                            [](const GuardState &lhs, const GuardState &rhs) {
                              return lhs == rhs;
                            });
  states.erase(newEnd, states.end());
}

static void dedupGuardSets(SmallVector<GuardSet, 4> &sets) {
  llvm::sort(sets, [](const GuardSet &a, const GuardSet &b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(),
                                        b.end());
  });
  sets.erase(std::unique(sets.begin(), sets.end()), sets.end());
}

static GuardStateVector walkBlock(Block &block, GuardStateVector states,
                                  DenseMap<Value, unsigned> &takeIds);

static bool mergeGuardStateVectors(GuardStateVector &dest,
                                   const GuardStateVector &source) {
  bool changed = false;
  for (const GuardState &set : source) {
    bool exists = llvm::any_of(dest, [&](const GuardState &other) {
      return other == set;
    });
    if (!exists) {
      dest.push_back(set);
      changed = true;
    }
  }
  if (changed)
    dedupGuardStates(dest);
  return changed;
}

static GuardStateVector walkRegion(Region &region, GuardStateVector states,
                                   DenseMap<Value, unsigned> &takeIds) {
  if (region.empty())
    return states;
  Block &entry = region.front();
  DenseMap<Block *, GuardStateVector> incoming;
  SmallVector<Block *, 4> worklist;
  incoming[&entry] = std::move(states);
  worklist.push_back(&entry);
  GuardStateVector exitStates;

  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();
    GuardStateVector blockStates = incoming[block];
    GuardStateVector outgoing =
        walkBlock(*block, std::move(blockStates), takeIds);
    Operation *terminator = block->getTerminator();
    if (terminator->getNumSuccessors() == 0) {
      mergeGuardStateVectors(exitStates, outgoing);
      continue;
    }
    for (unsigned idx = 0, e = terminator->getNumSuccessors(); idx < e; ++idx) {
      Block *succ = terminator->getSuccessor(idx);
      GuardStateVector &succStates = incoming[succ];
      if (mergeGuardStateVectors(succStates, outgoing))
        worklist.push_back(succ);
    }
  }

  return exitStates.empty() ? GuardStateVector{} : exitStates;
}

static GuardStateVector
walkChoice(ChoiceOp choice, const GuardStateVector &incoming,
           DenseMap<Value, unsigned> &takeIds) {
  GuardStateVector thenStates =
      walkRegion(choice.getThenRegion(), incoming, takeIds);
  GuardStateVector elseStates =
      walkRegion(choice.getElseRegion(), incoming, takeIds);
  GuardStateVector merged = std::move(thenStates);
  merged.append(elseStates.begin(), elseStates.end());
  dedupGuardStates(merged);
  return merged;
}

static GuardStateVector
walkIf(scf::IfOp ifOp, const GuardStateVector &incoming,
       DenseMap<Value, unsigned> &takeIds) {
  if (auto condition = constantBool(ifOp.getCondition())) {
    if (*condition)
      return walkRegion(ifOp.getThenRegion(), incoming, takeIds);
    return ifOp.getElseRegion().empty()
               ? incoming
               : walkRegion(ifOp.getElseRegion(), incoming, takeIds);
  }
  GuardStateVector thenStates =
      walkRegion(ifOp.getThenRegion(), incoming, takeIds);
  GuardStateVector elseStates;
  if (!ifOp.getElseRegion().empty())
    elseStates =
        walkRegion(ifOp.getElseRegion(), incoming, takeIds);
  else
    elseStates = incoming;
  GuardStateVector merged = std::move(thenStates);
  merged.append(elseStates.begin(), elseStates.end());
  dedupGuardStates(merged);
  return merged;
}

static GuardStateVector walkBlock(Block &block, GuardStateVector states,
                                  DenseMap<Value, unsigned> &takeIds) {

  for (Operation &op : block.without_terminator()) {
    if (auto take = dyn_cast<TakeOp>(&op)) {
      auto idIt = takeIds.find(take.getResult());
      if (idIt == takeIds.end())
        continue;
      for (GuardState &state : states)
        insertGuard(state.guards, idIt->second);
      continue;
    }
    if (auto emit = dyn_cast<EmitOp>(&op)) {
      for (GuardState &state : states)
        state.emits.push_back(emit.getOperation());
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
      states = walkIf(ifOp, states, takeIds);
      continue;
    }
    if (auto choice = dyn_cast<ChoiceOp>(&op)) {
      states = walkChoice(choice, states, takeIds);
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      // Token-flow operations in repeating loops are rejected before this
      // walk: guard paths identify operation sites, not execution counts.
      if (classifyLoop(forOp) == LoopTripCount::One)
        states = walkRegion(forOp.getRegion(), states, takeIds);
      continue;
    }
    if (op.getNumRegions()) {
      for (Region &region : op.getRegions())
        (void)walkRegion(region, states, takeIds);
      continue;
    }
  }

  return states;
}

class LPNSynthesizeGuardPass
    : public PassWrapper<LPNSynthesizeGuardPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNSynthesizeGuardPass)

  StringRef getArgument() const override { return "lpn-synthesize-guards"; }
  StringRef getDescription() const override {
    return "Compute per-transition guard metadata for retain passes.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](TransitionOp trans) {
      if (failed(processTransition(trans)))
        signalPassFailure();
    });
  }

private:
  LogicalResult processTransition(TransitionOp trans) {
    bool explicitGuard = false;
    bool invalidGuard = false;
    bool seenTakeOrEmit = false;
    for (Operation &op : trans.getBody().front()) {
      if (isa<GuardOp>(op)) {
        explicitGuard = true;
        invalidGuard |= seenTakeOrEmit;
      }
      op.walk([&](Operation *nested) {
        if (isa<TakeOp, EmitOp>(nested))
          seenTakeOrEmit = true;
        if (auto call = dyn_cast<func::CallOp>(nested))
          seenTakeOrEmit |= !isReadOnlyGuardHelper(call);
        if (isa<GuardOp>(nested) && nested != &op)
          invalidGuard = true;
      });
    }
    if (invalidGuard)
      return trans.emitError("manual lpn.guard must be top-level and precede "
                             "all takes, emits, and helper calls");
    if (explicitGuard) {
      // Keep the executable predicate. Take-ID paths cannot encode its value
      // dependencies or dynamic multiplicities; reduction must reject it.
      trans->setAttr("lpn.guard_mode", StringAttr::get(trans.getContext(),
                                                     "manual"));
      trans.walk([&](Operation *op) {
        op->removeAttr(kGuardIdAttr);
        op->removeAttr(kGuardPathsAttr);
      });
      return success();
    }
    bool hasPeek = false;
    trans.walk([&](PeekOp) { hasPeek = true; });
    if (hasPeek)
      return trans.emitError("cannot infer read-only token guards; add an "
                             "explicit lpn.guard predicate");
    llvm::StringMap<func::FuncOp> helpers;
    trans->getParentOfType<ModuleOp>().walk([&](func::FuncOp helper) {
      helpers.try_emplace(helper.getSymName(), helper);
    });
    auto helperHasTokenFlow = [&](auto &self, StringRef name,
                                  llvm::DenseSet<Operation *> &visiting) -> bool {
      auto it = helpers.find(name);
      if (it == helpers.end() || !visiting.insert(it->second).second)
        return true;
      bool effect = false;
      it->second.walk([&](Operation *nested) {
        if (isa<TakeOp, EmitOp>(nested))
          effect = true;
        if (auto call = dyn_cast<func::CallOp>(nested))
          effect |= self(self, call.getCallee(), visiting);
      });
      visiting.erase(it->second);
      return effect;
    };
    bool effectfulHelper = false;
    trans.walk([&](func::CallOp call) {
      llvm::DenseSet<Operation *> visiting;
      if (helperHasTokenFlow(helperHasTokenFlow, call.getCallee(), visiting))
        effectfulHelper = true;
    });
    if (effectfulHelper)
      return trans.emitError(
          "cannot synthesize exact guards for token flow inside a helper "
          "function; add an explicit lpn.guard predicate");

    bool unsupportedLoop = false;
    trans.walk([&](scf::ForOp loop) {
      if (classifyLoop(loop) != LoopTripCount::ManyOrUnknown)
        return;
      loop.walk([&](Operation *nested) {
        if (isa<TakeOp, EmitOp>(nested))
          unsupportedLoop = true;
      });
    });
    if (unsupportedLoop)
      return trans.emitError(
          "cannot synthesize exact guards for token flow in a repeating "
          "or dynamic scf.for loop; add an explicit lpn.guard predicate");

    bool unsupportedConditional = false;
    trans.walk([&](scf::IfOp ifOp) {
      if (constantBool(ifOp.getCondition()))
        return;
      ifOp.walk([&](Operation *nested) {
        if (isa<TakeOp, EmitOp>(nested))
          unsupportedConditional = true;
      });
    });
    if (unsupportedConditional)
      return trans.emitError(
          "cannot synthesize exact guards for token flow in a "
          "data-dependent scf.if; add an explicit lpn.guard predicate");

    DenseMap<Value, unsigned> takeIds;
    unsigned nextGuardId = 0;
    OpBuilder builder(trans.getContext());
    trans.walk([&](TakeOp take) {
      unsigned guardId = nextGuardId++;
      take->setAttr(kGuardIdAttr, builder.getI64IntegerAttr(guardId));
      takeIds[take.getResult()] = guardId;
    });

    GuardStateVector initialStates;
    initialStates.push_back(GuardState{});
    GuardStateVector finalStates =
        walkRegion(trans.getBody(), std::move(initialStates), takeIds);

    DenseMap<Operation *, SmallVector<GuardSet, 4>> emitGuards;
    for (const GuardState &state : finalStates)
      for (Operation *emit : state.emits)
        emitGuards[emit].push_back(state.guards);

    for (auto &entry : emitGuards) {
      SmallVector<GuardSet, 4> sets = std::move(entry.second);
      dedupGuardSets(sets);
      SmallVector<Attribute, 4> attrSets;
      for (GuardSet &set : sets) {
        SmallVector<Attribute, 4> ints;
        ints.reserve(set.size());
        for (unsigned guardId : set)
          ints.push_back(builder.getI64IntegerAttr(guardId));
        attrSets.push_back(builder.getArrayAttr(ints));
      }
      entry.first->setAttr(kGuardPathsAttr, builder.getArrayAttr(attrSets));
    }
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createLPNSynthesizeGuardPass() {
  return std::make_unique<LPNSynthesizeGuardPass>();
}

} // namespace mlir::lpn
