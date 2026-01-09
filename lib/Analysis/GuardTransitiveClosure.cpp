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
  llvm::SmallVector<GuardPathVariant, 4> paths;
  enum class State { Unvisited, Visiting, Done } state = State::Unvisited;
};

struct ProducerPath {
  Operation *emitOp = nullptr;
  unsigned pathIndex = 0;
  StringAttr target;
  llvm::SmallVector<Value, 4> guardTakes;
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

static void canonicalizeVariant(GuardPathVariant &variant) {
  llvm::sort(variant.observables.begin(), variant.observables.end(),
             [](Value a, Value b) {
               return a.getAsOpaquePointer() < b.getAsOpaquePointer();
             });
}

static bool segmentsLess(const GuardPathSegment &lhs,
                         const GuardPathSegment &rhs) {
  if (lhs.emit != rhs.emit)
    return lhs.emit < rhs.emit;
  if (lhs.pathIndex != rhs.pathIndex)
    return lhs.pathIndex < rhs.pathIndex;
  if (lhs.target != rhs.target)
    return lhs.target.getAsOpaquePointer() < rhs.target.getAsOpaquePointer();
  return lhs.producedTake.getAsOpaquePointer() <
         rhs.producedTake.getAsOpaquePointer();
}

static bool variantsLess(const GuardPathVariant &lhs,
                         const GuardPathVariant &rhs) {
  if (lhs.segments.size() != rhs.segments.size())
    return lhs.segments.size() < rhs.segments.size();
  for (auto [lSeg, rSeg] : llvm::zip(lhs.segments, rhs.segments)) {
    if (segmentsLess(lSeg, rSeg))
      return true;
    if (segmentsLess(rSeg, lSeg))
      return false;
  }
  if (lhs.observables.size() != rhs.observables.size())
    return lhs.observables.size() < rhs.observables.size();
  for (auto [lObs, rObs] : llvm::zip(lhs.observables, rhs.observables)) {
    if (lObs.getAsOpaquePointer() < rObs.getAsOpaquePointer())
      return true;
    if (rObs.getAsOpaquePointer() < lObs.getAsOpaquePointer())
      return false;
  }
  return false;
}

static bool variantsEqual(const GuardPathVariant &lhs,
                          const GuardPathVariant &rhs) {
  if (lhs.segments.size() != rhs.segments.size() ||
      lhs.observables.size() != rhs.observables.size())
    return false;
  for (auto [lSeg, rSeg] : llvm::zip(lhs.segments, rhs.segments)) {
    if (lSeg.emit != rSeg.emit || lSeg.pathIndex != rSeg.pathIndex ||
        lSeg.target != rSeg.target || lSeg.producedTake != rSeg.producedTake)
      return false;
  }
  for (auto [lObs, rObs] : llvm::zip(lhs.observables, rhs.observables))
    if (lObs != rObs)
      return false;
  return true;
}

static void dedupVariantSet(llvm::SmallVectorImpl<GuardPathVariant> &variants) {
  for (auto &variant : variants)
    canonicalizeVariant(variant);
  llvm::sort(variants.begin(), variants.end(), variantsLess);
  variants.erase(std::unique(variants.begin(), variants.end(), variantsEqual),
                 variants.end());
}

static llvm::SmallVector<GuardPathVariant, 4>
cartProductVariants(llvm::ArrayRef<GuardPathVariant> lhs,
                    llvm::ArrayRef<GuardPathVariant> rhs) {
  llvm::SmallVector<GuardPathVariant, 4> result;
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
      GuardPathVariant combo;
      combo.observables.append(left.observables.begin(),
                               left.observables.end());
      combo.observables.append(right.observables.begin(),
                               right.observables.end());
      combo.segments.append(left.segments.begin(), left.segments.end());
      combo.segments.append(right.segments.begin(), right.segments.end());
      result.push_back(std::move(combo));
    }
  return result;
}

struct GuardClosureContext {
  NetOp net;
  const ObservableSet &observables;
  llvm::DenseMap<Value, GuardNode> nodes;
  llvm::DenseMap<StringAttr, llvm::SmallVector<ProducerPath, 4>> producers;
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
      unsigned pathIndex = 0;
      for (Attribute attr : guardsAttr) {
        auto pathAttr = llvm::dyn_cast<ArrayAttr>(attr);
        if (!pathAttr) {
          ++pathIndex;
          continue;
        }
  ProducerPath base;
  base.emitOp = emit.getOperation();
        base.pathIndex = pathIndex;
        for (Attribute elem : pathAttr) {
          auto intAttr = llvm::dyn_cast<IntegerAttr>(elem);
          if (!intAttr)
            continue;
          unsigned guardId = intAttr.getInt();
          auto it = guardIdToTake.find(guardId);
          if (it == guardIdToTake.end())
            continue;
          base.guardTakes.push_back(it->second);
        }
        for (StringAttr place : targets) {
          ProducerPath entry = base;
          entry.target = place;
          producers[place].push_back(std::move(entry));
        }
        ++pathIndex;
      }
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
  // Cycle means the guard depends only on itself or other non-observable
  // guards; drop it quietly so downstream analyses treat it as unguarded.
  if (node.state == GuardNode::State::Visiting) {
    node.paths.clear();
    node.state = GuardNode::State::Done;
    return success();
  }

  node.state = GuardNode::State::Visiting;
  if (node.isObservable) {
    GuardPathVariant variant;
    variant.observables.push_back(takeValue);
    node.paths.push_back(std::move(variant));
  } else {
    auto prodIt = producers.find(node.place);
    if (prodIt == producers.end()) {
      if (hasInitialTokens(node.place, placeLookup))
        node.paths.emplace_back();
    } else {
      for (const ProducerPath &producer : prodIt->second) {
        llvm::SmallVector<GuardPathVariant, 4> combos;
        combos.emplace_back();
        bool feasible = true;
        for (Value guardTake : producer.guardTakes) {
          if (failed(compute(guardTake)))
            return failure();
          auto &childPaths = nodes[guardTake].paths;
          if (childPaths.empty()) {
            feasible = false;
            break;
          }
          combos = cartProductVariants(combos, childPaths);
          if (combos.empty()) {
            feasible = false;
            break;
          }
        }
        if (!feasible)
          continue;
        for (auto &variant : combos) {
          GuardPathSegment segment;
          segment.emit = producer.emitOp;
          segment.pathIndex = producer.pathIndex;
          segment.target = producer.target;
          segment.producedTake = node.takeValue;
          variant.segments.push_back(segment);
          node.paths.push_back(std::move(variant));
        }
      }
    }
  }

  if (!node.paths.empty())
    dedupVariantSet(node.paths);
  node.state = GuardNode::State::Done;
  return success();
}

} // namespace

ArrayRef<GuardPathVariant>
GuardTransitiveClosureResult::getPaths(Value takeValue) const {
  auto it = guardPaths.find(takeValue);
  if (it == guardPaths.end())
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
    result.guardPaths[entry.first] = entry.second.paths;
  return success();
}

} // namespace mlir::lpn
