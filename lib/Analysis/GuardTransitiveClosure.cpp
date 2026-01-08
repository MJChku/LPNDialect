#include "LPN/Analysis/GuardTransitiveClosure.h"
#include "LPN/Dialect/LPNOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

namespace mlir::lpn {
namespace {

static constexpr StringLiteral kGuardIdAttr = "lpn.guard_id";
static constexpr StringLiteral kGuardPathsAttr = "lpn.guard_paths";

struct GuardNode {
  Value takeValue;
  StringAttr place;
  bool isObservable = false;
  GuardTransitiveClosureResult::GuardCombination baseCombo;
  llvm::SmallVector<GuardTransitiveClosureResult::GuardCombination, 4> combos;
  enum class State { Unvisited, Visiting, Done } state = State::Unvisited;
};

struct GuardPathInfo {
  llvm::SmallVector<Value, 4> guards;
};

struct ProducerInfo {
  llvm::SmallVector<GuardPathInfo, 4> guardPaths;
};

static bool hasInitialTokens(StringAttr place,
                             llvm::DenseMap<StringAttr, PlaceOp> &places) {
  auto it = places.find(place);
  if (it == places.end())
    return false;
  if (auto attr = it->second.getInitialTokens())
    return static_cast<int64_t>(*attr) > 0;
  return false;
}

static LogicalResult resolvePlaceSymbol(Value handle, StringAttr &symbol) {
  if (auto ref = handle.getDefiningOp<PlaceRefOp>()) {
    symbol = ref.getPlaceAttr().getAttr();
    return success();
  }
  return failure();
}

static LogicalResult collectEmitPlaces(
  Value placeValue, llvm::SmallVectorImpl<StringAttr> &places) {
  if (auto ref = placeValue.getDefiningOp<PlaceRefOp>()) {
    places.push_back(ref.getPlaceAttr().getAttr());
    return success();
  }
  if (auto get = placeValue.getDefiningOp<ArrayGetOp>()) {
    auto list = get.getArray().getDefiningOp<ArrayOp>();
    if (!list)
      return failure();
    auto elements = list.getElements();
    if (auto idxConst = get.getIndex().getDefiningOp<arith::ConstantOp>()) {
      auto intAttr = llvm::dyn_cast<IntegerAttr>(idxConst.getValue());
      if (!intAttr)
        return failure();
      int64_t idx = intAttr.getInt();
      if (idx < 0 || idx >= static_cast<int64_t>(elements.size()))
        return failure();
      auto ref = elements[idx].getDefiningOp<PlaceRefOp>();
      if (!ref)
        return failure();
      places.push_back(ref.getPlaceAttr().getAttr());
      return success();
    }
    for (Value element : elements)
      if (auto ref = element.getDefiningOp<PlaceRefOp>())
        places.push_back(ref.getPlaceAttr().getAttr());
    return success();
  }
  return failure();
}

static void dedupCombinationSet(
    llvm::SmallVector<GuardTransitiveClosureResult::GuardCombination, 4>
        &combos) {
  for (auto &combo : combos) {
    llvm::sort(combo.begin(), combo.end(), [](Value a, Value b) {
      return a.getAsOpaquePointer() < b.getAsOpaquePointer();
    });
  }
  llvm::sort(combos, [](const auto &lhs, const auto &rhs) {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                        rhs.end(), [](Value a, Value b) {
                                          return a.getAsOpaquePointer() <
                                                 b.getAsOpaquePointer();
                                        });
  });
  combos.erase(std::unique(combos.begin(), combos.end(),
                           [](const auto &lhs, const auto &rhs) {
                             if (lhs.size() != rhs.size())
                               return false;
                             for (auto [a, b] : llvm::zip(lhs, rhs))
                               if (a != b)
                                 return false;
                             return true;
                           }),
               combos.end());
}

static llvm::SmallVector<GuardTransitiveClosureResult::GuardCombination, 4>
cartProduct(llvm::ArrayRef<GuardTransitiveClosureResult::GuardCombination> lhs,
            llvm::ArrayRef<GuardTransitiveClosureResult::GuardCombination> rhs) {
  llvm::SmallVector<GuardTransitiveClosureResult::GuardCombination, 4> result;
  if (lhs.empty()) {
    result.append(rhs.begin(), rhs.end());
    return result;
  }
  if (rhs.empty()) {
    result.append(lhs.begin(), lhs.end());
    return result;
  }
  for (const auto &left : lhs)
    for (const auto &right : rhs) {
      auto &combo = result.emplace_back();
      combo.append(left.begin(), left.end());
      combo.append(right.begin(), right.end());
    }
  return result;
}

struct GuardClosureContext {
  NetOp net;
  const ObservableSet &observables;
  llvm::DenseMap<Value, GuardNode> nodes;
  llvm::DenseMap<StringAttr, llvm::SmallVector<ProducerInfo>> producers;
  llvm::DenseMap<StringAttr, PlaceOp> placeLookup;

  GuardClosureContext(NetOp net, const ObservableSet &observables)
      : net(net), observables(observables) {}

  LogicalResult initialize();
  LogicalResult compute(Value takeValue);
};

LogicalResult GuardClosureContext::initialize() {
  for (PlaceOp place : net.getOps<PlaceOp>())
    placeLookup[place.getSymNameAttr()] = place;

  for (TransitionOp trans : net.getOps<TransitionOp>()) {
    llvm::DenseMap<unsigned, Value> guardIdToTake;
    for (TakeOp take : trans.getBody().getOps<TakeOp>()) {
      auto guardAttr = take->getAttrOfType<IntegerAttr>(kGuardIdAttr);
      if (!guardAttr)
        continue;
      unsigned guardId = guardAttr.getInt();
      guardIdToTake[guardId] = take.getResult();
      GuardNode &node = nodes[take.getResult()];
      node.takeValue = take.getResult();
      if (failed(resolvePlaceSymbol(take.getPlace(), node.place)))
        return take.emitError("failed to resolve place symbol");
      node.isObservable = observables.contains(node.place);
    }

    for (EmitOp emit : trans.getBody().getOps<EmitOp>()) {
      auto guardsAttr = emit->getAttrOfType<ArrayAttr>(kGuardPathsAttr);
      if (!guardsAttr || guardsAttr.empty())
        continue;
      llvm::SmallVector<StringAttr, 4> targets;
      if (failed(collectEmitPlaces(emit.getPlace(), targets)))
        return emit.emitError("failed to resolve emit targets");
      llvm::SmallVector<GuardPathInfo, 4> guardPaths;
      for (Attribute attr : guardsAttr) {
        auto pathAttr = llvm::dyn_cast<ArrayAttr>(attr);
        if (!pathAttr)
          continue;
        GuardPathInfo path;
        for (Attribute elem : pathAttr) {
          auto intAttr = llvm::dyn_cast<IntegerAttr>(elem);
          if (!intAttr)
            continue;
          unsigned guardId = intAttr.getInt();
          auto it = guardIdToTake.find(guardId);
          if (it == guardIdToTake.end())
            continue;
          path.guards.push_back(it->second);
        }
        guardPaths.push_back(std::move(path));
      }
      if (guardPaths.empty())
        continue;
      for (StringAttr place : targets)
        producers[place].push_back(ProducerInfo{guardPaths});
    }
  }

  return success();
}

LogicalResult GuardClosureContext::compute(Value takeValue) {
  auto nodeIt = nodes.find(takeValue);
  if (nodeIt == nodes.end())
    return failure();
  GuardNode &node = nodeIt->second;
  if (node.state == GuardNode::State::Done)
    return success();
  if (node.state == GuardNode::State::Visiting) {
    takeValue.getDefiningOp()->emitWarning(
        "detected cycle while resolving guard closure; ignoring guard");
    node.combos.clear();
    node.state = GuardNode::State::Done;
    return success();
  }

  node.state = GuardNode::State::Visiting;
  if (node.isObservable) {
    GuardTransitiveClosureResult::GuardCombination combo;
    combo.push_back(takeValue);
    node.combos.push_back(std::move(combo));
  } else {
    auto prodIt = producers.find(node.place);
    if (prodIt == producers.end()) {
      if (hasInitialTokens(node.place, placeLookup))
        node.combos.push_back({});
    } else {
      for (const ProducerInfo &producer : prodIt->second) {
        for (const GuardPathInfo &path : producer.guardPaths) {
      llvm::SmallVector<GuardTransitiveClosureResult::GuardCombination, 4>
        combos;
          combos.push_back({});
          bool feasible = true;
          for (Value guardTake : path.guards) {
            if (failed(compute(guardTake)))
              return failure();
            auto &childCombos = nodes[guardTake].combos;
            if (childCombos.empty()) {
              feasible = false;
              break;
            }
            combos = cartProduct(combos, childCombos);
          }
          if (feasible)
            node.combos.append(combos.begin(), combos.end());
        }
      }
    }
  }

  if (!node.combos.empty())
    dedupCombinationSet(node.combos);
  node.state = GuardNode::State::Done;
  return success();
}

} // namespace

ArrayRef<GuardTransitiveClosureResult::GuardCombination>
GuardTransitiveClosureResult::getClosures(Value takeValue) const {
  auto it = guardClosures.find(takeValue);
  if (it == guardClosures.end())
    return {};
  return it->second;
}

LogicalResult runGuardTransitiveClosureAnalysis(
  NetOp net, const ObservableSet &observables,
  GuardTransitiveClosureResult &result) {
  GuardClosureContext ctx(net, observables);
  if (failed(ctx.initialize()))
    return failure();
  for (auto &entry : ctx.nodes)
    if (failed(ctx.compute(entry.first)))
      return failure();
  for (auto &entry : ctx.nodes)
    result.guardClosures[entry.first] = entry.second.combos;
  return success();
}

} // namespace mlir::lpn
