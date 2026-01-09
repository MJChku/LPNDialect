//===- TokenFlowAnalysis.cpp - Token flow analysis for LPN ---------------===//
//
// Emit-centric token flow analysis shared by retain passes.
//
//===----------------------------------------------------------------------===//

#include "LPN/Analysis/ControlFlowTrace.h"
#include "LPN/Analysis/GuardTransitiveClosure.h"
#include "LPN/Analysis/TokenFlowAnalysis.h"
#include "LPN/Dialect/LPNTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <optional>

namespace mlir::lpn {
namespace {

static constexpr StringLiteral kGuardIdAttr = "lpn.guard_id";
static constexpr StringLiteral kGuardPathsAttr = "lpn.guard_paths";

static llvm::hash_code
hashValueExpr(Value value, llvm::DenseMap<Value, llvm::hash_code> &cache);
static llvm::hash_code
hashOptionalValue(Value value, llvm::DenseMap<Value, llvm::hash_code> &cache);



static LogicalResult resolvePlaceSymbol(Value handle, StringAttr &symbol) {
  auto ref = handle.getDefiningOp<PlaceRefOp>();
  if (!ref)
    return failure();
  symbol = ref.getPlaceAttr().getAttr();
  return success();
}

static Value stripIndexCasts(Value value) {
  Value current = value;
  while (Operation *def = current.getDefiningOp()) {
    if (auto cast = dyn_cast<arith::IndexCastOp>(def)) {
      current = cast.getIn();
      continue;
    }
    break;
  }
  return current;
}

static std::optional<int64_t> getConstI64(Value value) {
  if (!value)
    return std::nullopt;
  if (Operation *def = value.getDefiningOp())
    if (auto constOp = dyn_cast<arith::ConstantOp>(def))
      if (auto attr = dyn_cast<IntegerAttr>(constOp.getValue()))
        return attr.getValue().getSExtValue();
  return std::nullopt;
}

static std::optional<TokenGuard> matchListIndexGuard(Value index,
                                                     int64_t slot) {
  llvm::DenseMap<Value, llvm::hash_code> guardHashCache;
  Value current = stripIndexCasts(index);

  auto buildGuard = [&](TokenGetOp get, int64_t offset,
                        bool add) -> std::optional<TokenGuard> {
    TokenGuard guard;
    guard.key = get.getKey();
    guard.keyHash = hashValueExpr(guard.key, guardHashCache);
    guard.equalsValue = add ? slot - offset : slot + offset;
    return guard;
  };

  if (Operation *def = current.getDefiningOp())
    if (auto get = dyn_cast<TokenGetOp>(def))
      return buildGuard(get, /*offset=*/0, /*add=*/true);

  if (Operation *def = current.getDefiningOp())
    if (auto subi = dyn_cast<arith::SubIOp>(def)) {
      if (Operation *lhsDef = subi.getLhs().getDefiningOp())
        if (auto lhs = dyn_cast<TokenGetOp>(lhsDef))
          if (auto rhsConst = getConstI64(subi.getRhs()))
            return buildGuard(lhs, *rhsConst, /*add=*/false);
      if (Operation *rhsDef = subi.getRhs().getDefiningOp())
        if (auto rhs = dyn_cast<TokenGetOp>(rhsDef))
          if (auto lhsConst = getConstI64(subi.getLhs()))
            return buildGuard(rhs, *lhsConst, /*add=*/true);
    }

  if (Operation *def = current.getDefiningOp())
    if (auto addi = dyn_cast<arith::AddIOp>(def)) {
      if (Operation *lhsDef = addi.getLhs().getDefiningOp())
        if (auto lhs = dyn_cast<TokenGetOp>(lhsDef))
          if (auto rhsConst = getConstI64(addi.getRhs()))
            return buildGuard(lhs, *rhsConst, /*add=*/true);
      if (Operation *rhsDef = addi.getRhs().getDefiningOp())
        if (auto rhs = dyn_cast<TokenGetOp>(rhsDef))
          if (auto lhsConst = getConstI64(addi.getLhs()))
            return buildGuard(rhs, *lhsConst, /*add=*/true);
    }

  if (auto constIdx = getConstI64(current)) {
    TokenGuard guard;
    guard.equalsValue = constIdx.value();
    guard.keyHash = llvm::hash_value(static_cast<void *>(nullptr));
    return guard;
  }

  return std::nullopt;
}

static LogicalResult
resolveEmitTargets(Value placeValue,
                   llvm::SmallVectorImpl<TargetInfo> &targets) {
  if (auto ref = placeValue.getDefiningOp<PlaceRefOp>()) {
    targets.push_back(TargetInfo{ref.getPlaceAttr().getAttr(), {}});
    return success();
  }

  if (auto get = placeValue.getDefiningOp<ArrayGetOp>()) {
    auto list = get.getArray().getDefiningOp<ArrayOp>();
    if (!list)
      return failure();
    auto elements = list.getElements();

    Value baseIndex = stripIndexCasts(get.getIndex());
    if (auto constIdx = getConstI64(baseIndex)) {
      if (*constIdx < 0 || *constIdx >= static_cast<int64_t>(elements.size()))
        return failure();

      Value element = elements[*constIdx];
      if (auto ref = element.getDefiningOp<PlaceRefOp>()) {
        targets.push_back(TargetInfo{ref.getPlaceAttr().getAttr(), {}});
        return success();
      }
      return failure();
    }

    for (auto [slot, val] : llvm::enumerate(elements)) {
      if (auto ref = val.getDefiningOp<PlaceRefOp>()) {
        auto sym = ref.getPlaceAttr();
        TargetInfo info;
        info.symbol = sym.getAttr();
        if (auto guard = matchListIndexGuard(get.getIndex(), slot))
          if (guard->key)
            info.guards.push_back(*guard);
        targets.push_back(std::move(info));
      }
    }
    return success();
  }

  return failure();
}

static llvm::hash_code
hashValueExpr(Value value, llvm::DenseMap<Value, llvm::hash_code> &cache) {
  if (auto it = cache.find(value); it != cache.end())
    return it->second;

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    llvm::hash_code h = llvm::hash_combine(
        arg.getArgNumber(), reinterpret_cast<uintptr_t>(arg.getOwner()));
    cache[value] = h;
    return h;
  }

  Operation *def = value.getDefiningOp();
  if (!def) {
    llvm::hash_code h = llvm::hash_value(value.getAsOpaquePointer());
    cache[value] = h;
    return h;
  }

  unsigned resultNumber = 0;
  if (auto res = dyn_cast<OpResult>(value))
    resultNumber = res.getResultNumber();
  llvm::hash_code h = llvm::hash_combine(
      llvm::hash_value(def->getName().getStringRef()), resultNumber);
  for (NamedAttribute attr : def->getAttrs())
    h = llvm::hash_combine(
        h, llvm::hash_value(attr.getName()),
        llvm::hash_value(attr.getValue().getAsOpaquePointer()));
  for (Value operand : def->getOperands())
    h = llvm::hash_combine(h, hashValueExpr(operand, cache));
  cache[value] = h;
  return h;
}

static llvm::hash_code
hashOptionalValue(Value value, llvm::DenseMap<Value, llvm::hash_code> &cache) {
  if (!value)
    return llvm::hash_value(static_cast<void *>(nullptr));
  return hashValueExpr(value, cache);
}

static void recordSourceRefs(Value root, TokenEditSignature &sig,
                             ArrayRef<ObservableSource> sources,
                             llvm::SmallPtrSetImpl<Value> &visited) {
  if (!root || !visited.insert(root).second)
    return;
  if (auto get = root.getDefiningOp<TokenGetOp>()) {
    for (auto [idx, src] : llvm::enumerate(sources))
      if (src.takeValue == get.getToken())
        sig.sourceRefs.push_back(idx);
  }
  if (Operation *producer = root.getDefiningOp())
    for (Value operand : producer->getOperands())
      recordSourceRefs(operand, sig, sources, visited);
}

static LogicalResult
summarizeTokenEdits(Value current,
                    llvm::SmallVectorImpl<TokenEditSignature> &edits,
                    llvm::DenseMap<Value, llvm::hash_code> &hashCache,
                    ArrayRef<ObservableSource> sources) {
  Operation *def = current.getDefiningOp();
  if (!def)
    return failure();
  // Stop at token sources (TakeOp or CreateOp)
  if (isa<TakeOp, TokenCreateOp>(def))
    return success();
  if (auto set = dyn_cast<TokenSetOp>(def)) {
    if (failed(summarizeTokenEdits(set.getToken(), edits, hashCache, sources)))
      return failure();
    TokenEditSignature sig;
    sig.keyHash = hashValueExpr(set.getKey(), hashCache);
    sig.valueHash = hashValueExpr(set.getValue(), hashCache);
    llvm::SmallPtrSet<Value, 8> visited;
    recordSourceRefs(set.getValue(), sig, sources, visited);
    recordSourceRefs(set.getKey(), sig, sources, visited);
    edits.push_back(std::move(sig));
    return success();
  }
  if (auto clone = dyn_cast<TokenCloneOp>(def))
    return summarizeTokenEdits(clone.getToken(), edits, hashCache, sources);
  return def->emitError("token flow includes unsupported op while summarizing");
}

static bool lessSource(const ObservableSource &a, const ObservableSource &b) {
  if (a.place != b.place)
    return a.place.getValue() < b.place.getValue();
  return a.takeValue.getAsOpaquePointer() < b.takeValue.getAsOpaquePointer();
}

static void canonicalizeSources(
    llvm::SmallVectorImpl<ObservableSource> &sources) {
  llvm::sort(sources.begin(), sources.end(), lessSource);
  sources.erase(std::unique(sources.begin(), sources.end(),
                            [](const ObservableSource &lhs,
                               const ObservableSource &rhs) {
                              return lhs.takeValue == rhs.takeValue;
                            }),
                sources.end());
}

static void dedupSegments(llvm::SmallVectorImpl<GuardPathSegment> &segments) {
  llvm::DenseSet<Value> seen;
  llvm::SmallVector<GuardPathSegment, 8> filtered;
  filtered.reserve(segments.size());
  for (const GuardPathSegment &segment : segments) {
    if (segment.producedTake && !seen.insert(segment.producedTake).second)
      continue;
    filtered.push_back(segment);
  }
  segments.swap(filtered);
}

static llvm::SmallVector<GuardPathVariant, 4>
combineVariants(llvm::ArrayRef<GuardPathVariant> lhs,
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
      GuardPathVariant variant;
      variant.observables.append(left.observables.begin(),
                                 left.observables.end());
      variant.observables.append(right.observables.begin(),
                                 right.observables.end());
      variant.segments.append(left.segments.begin(), left.segments.end());
      variant.segments.append(right.segments.begin(), right.segments.end());
      result.push_back(std::move(variant));
    }
  return result;
}

struct TokenFlowAnalysisImpl {
  TokenFlowAnalysisImpl(NetOp net, const ObservableSet &observables,
                        const GuardTransitiveClosureResult &guardClosure,
                        TokenFlowAnalysisResult &result)
      : net(net), observables(observables), guardClosure(guardClosure),
        result(result) {}

  LogicalResult run() {
    takePlaces.clear();
    result.emitMetadata.clear();
    result.observablePaths.clear();
    result.totalHyperedges = 0;
    result.guardHyperedges = 0;
    if (failed(collectTakePlaces()))
      return failure();
    for (TransitionOp trans : net.getOps<TransitionOp>())
      if (failed(processTransition(trans)))
        return failure();
    return resolveObservablePaths();
  }

private:
  LogicalResult collectTakePlaces() {
    for (TransitionOp trans : net.getOps<TransitionOp>()) {
      for (TakeOp take : trans.getBody().getOps<TakeOp>()) {
        StringAttr place;
        if (failed(resolvePlaceSymbol(take.getPlace(), place)))
          return take.emitError("failed to resolve place symbol");
        takePlaces[take.getResult()] = place;
      }
    }
    return success();
  }

  LogicalResult processTransition(TransitionOp trans) {
    ControlFlowTrace trace(trans);
    llvm::DenseMap<unsigned, Value> guardIdToTake;
    for (TakeOp take : trans.getBody().getOps<TakeOp>())
      if (auto guardAttr = take->getAttrOfType<IntegerAttr>(kGuardIdAttr))
        guardIdToTake[guardAttr.getInt()] = take.getResult();

    WalkResult walkResult = trans->walk([&](EmitOp emit) -> WalkResult {
      if (failed(processEmit(trans, emit, trace, guardIdToTake)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return failure();
    return success();
  }

  LogicalResult processEmit(TransitionOp trans, EmitOp emit,
                            ControlFlowTrace &trace,
                            const llvm::DenseMap<unsigned, Value> &guardIdToTake);
  LogicalResult resolveObservablePaths();

  bool isSatisfiedByInitialTokens(Value takeValue) const {
    auto variants = guardClosure.getPaths(takeValue);
    return llvm::any_of(variants, [](const GuardPathVariant &variant) {
      return variant.segments.empty() && variant.observables.empty();
    });
  }

  NetOp net;
  const ObservableSet &observables;
  const GuardTransitiveClosureResult &guardClosure;
  TokenFlowAnalysisResult &result;
  llvm::DenseMap<Value, StringAttr> takePlaces;
};

LogicalResult TokenFlowAnalysisImpl::processEmit(
    TransitionOp trans, EmitOp emit, ControlFlowTrace &trace,
    const llvm::DenseMap<unsigned, Value> &guardIdToTake) {
  llvm::ArrayRef<ControlContext> contexts = trace.getTrace(emit);

  llvm::SmallVector<TargetInfo, 4> targets;
  if (failed(resolveEmitTargets(emit.getPlace(), targets)))
    return emit.emitError("failed to resolve emit targets"), failure();

  auto guardAttr = emit->getAttrOfType<ArrayAttr>(kGuardPathsAttr);
  if (!guardAttr || guardAttr.empty())
    return success();

  unsigned pathIndex = 0;
  for (Attribute attr : guardAttr) {
    auto pathAttr = dyn_cast<ArrayAttr>(attr);
    if (!pathAttr) {
      ++pathIndex;
      continue;
    }

    llvm::SmallVector<Value, 4> guardTakes;
    llvm::SmallVector<ObservableSource, 4> guardSources;
    bool referencedGuard = false;
    for (Attribute elem : pathAttr) {
      auto intAttr = dyn_cast<IntegerAttr>(elem);
      if (!intAttr)
        continue;
      auto it = guardIdToTake.find(intAttr.getInt());
      if (it == guardIdToTake.end())
        continue;
      referencedGuard = true;
      Value takeValue = it->second;
      // Guards that can be satisfied entirely by initial tokens do not need
      // to be replayed here. TODO: require explicit initial-token counts when
      // the retain pipeline begins modeling depletion.
      if (isSatisfiedByInitialTokens(takeValue))
        continue;
      guardTakes.push_back(takeValue);
      if (auto placeIt = takePlaces.find(takeValue); placeIt != takePlaces.end())
        guardSources.push_back({placeIt->second, takeValue});
    }

    if (!referencedGuard) {
      ++pathIndex;
      continue;
    }

    for (const TargetInfo &target : targets) {
      EmitMetadata metadata;
      metadata.transition = trans;
      metadata.emit = emit;
      metadata.pathIndex = pathIndex;
      metadata.targetPlace = target.symbol;
      metadata.tokenValue = emit.getToken();
      metadata.delayValue = emit.getDelay();
      metadata.contexts.assign(contexts.begin(), contexts.end());
      metadata.target = target;
      metadata.guardTakes = guardTakes;
      metadata.hasGuardPath = true;

      llvm::DenseMap<Value, llvm::hash_code> hashCache;
      if (failed(summarizeTokenEdits(emit.getToken(), metadata.edits, hashCache,
                                     guardSources)))
        return failure();
      metadata.tokenHash = hashOptionalValue(emit.getToken(), hashCache);
      metadata.delayHash = hashOptionalValue(emit.getDelay(), hashCache);

      EmitPathKey key{emit.getOperation(), pathIndex, target.symbol};
      result.emitMetadata[key] = std::move(metadata);
    }

    ++pathIndex;
  }

  return success();
}

LogicalResult TokenFlowAnalysisImpl::resolveObservablePaths() {
  for (const auto &entry : result.emitMetadata) {
    const EmitMetadata &metadata = entry.second;
    if (!metadata.hasGuardPath)
      continue;
    if (!observables.contains(metadata.target.symbol))
      continue;

    llvm::SmallVector<GuardPathVariant, 4> combos;
    combos.emplace_back();
    bool feasible = true;
    for (Value guardTake : metadata.guardTakes) {
      llvm::ArrayRef<GuardPathVariant> variants =
          guardClosure.getPaths(guardTake);
      if (variants.empty()) {
        feasible = false;
        break;
      }
      combos = combineVariants(combos, variants);
      if (combos.empty()) {
        feasible = false;
        break;
      }
    }
    if (!feasible || combos.empty())
      continue;

    for (GuardPathVariant &variant : combos) {
      llvm::DenseSet<Value> satisfied;
      for (const GuardPathSegment &segment : variant.segments)
        if (segment.producedTake)
          satisfied.insert(segment.producedTake);
      for (Value obs : variant.observables)
        satisfied.insert(obs);
      bool missingGuard = llvm::any_of(
          metadata.guardTakes,
          [&](Value take) { return !satisfied.contains(take); });
      if (missingGuard)
        continue;

      dedupSegments(variant.segments);
      ObservablePath path;
      path.terminalKey = entry.first;
      path.prefixSegments = std::move(variant.segments);
      for (Value obs : variant.observables) {
        auto placeIt = takePlaces.find(obs);
        if (placeIt == takePlaces.end())
          return net.emitError(
                     "failed to resolve observable source place during guard closure"),
                 failure();
        path.sources.push_back({placeIt->second, obs});
      }
      canonicalizeSources(path.sources);
      result.observablePaths[metadata.target.symbol].push_back(
          std::move(path));
      ++result.totalHyperedges;
      ++result.guardHyperedges;
    }
  }

  return success();
}


} // namespace

LogicalResult runTokenFlowAnalysis(NetOp net, const ObservableSet &observables,
                                   TokenFlowAnalysisResult &result) {
  GuardTransitiveClosureResult guardClosure;
  if (failed(
          runGuardTransitiveClosureAnalysis(net, observables, guardClosure)))
    return failure();
  TokenFlowAnalysisImpl impl(net, observables, guardClosure, result);
  return impl.run();
}

} // namespace mlir::lpn
