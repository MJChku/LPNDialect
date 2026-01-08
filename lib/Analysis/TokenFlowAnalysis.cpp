//===- TokenFlowAnalysis.cpp - Token flow analysis for LPN ---------------===//
//
// Emit-centric token flow analysis shared by retain passes.
//
//===----------------------------------------------------------------------===//

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

static bool blockInRegion(Block *block, Region &region) {
  for (Region *current = block ? block->getParent() : nullptr; current;
       current = current->getParentRegion())
    if (current == &region)
      return true;
  return false;
}

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
summarizeTokenEdits(Value current, Value source,
                    llvm::SmallVectorImpl<TokenEditSignature> &edits,
                    llvm::DenseMap<Value, llvm::hash_code> &hashCache,
                    ArrayRef<ObservableSource> sources) {
  if (current == source)
    return success();
  Operation *def = current.getDefiningOp();
  if (!def)
    return failure();
  if (auto set = dyn_cast<TokenSetOp>(def)) {
    if (failed(summarizeTokenEdits(set.getToken(), source, edits, hashCache,
                                   sources)))
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
    return summarizeTokenEdits(clone.getToken(), source, edits, hashCache,
                               sources);
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

static void appendUniqueSources(ArrayRef<ObservableSource> extras,
                                llvm::SmallVectorImpl<ObservableSource> &dst) {
  for (const ObservableSource &src : extras) {
    bool exists = llvm::any_of(dst, [&](const ObservableSource &seen) {
      return seen.takeValue == src.takeValue;
    });
    if (!exists)
      dst.push_back(src);
  }
}

static void addDriverVariants(
    ArrayRef<ObservableSource> base,
    llvm::SmallVectorImpl<llvm::SmallVector<ObservableSource, 4>> &out) {
  for (const ObservableSource &driver : base) {
    llvm::SmallVector<ObservableSource, 4> variant;
    variant.push_back(driver);
    for (const ObservableSource &src : base) {
      if (src.takeValue == driver.takeValue)
        continue;
      variant.push_back(src);
    }
    if (variant.size() > 1)
      llvm::sort(variant.begin() + 1, variant.end(), lessSource);
    out.push_back(std::move(variant));
  }
}

static LogicalResult
collectTokenSources(Value token, llvm::SmallVectorImpl<TakeOp> &sources,
                    llvm::SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(token).second)
    return success();
  if (auto take = token.getDefiningOp<TakeOp>()) {
    sources.push_back(take);
    return success();
  }
  if (auto set = token.getDefiningOp<TokenSetOp>())
    return collectTokenSources(set.getToken(), sources, visited);
  if (auto clone = token.getDefiningOp<TokenCloneOp>())
    return collectTokenSources(clone.getToken(), sources, visited);
  if (token.getDefiningOp<TokenCreateOp>())
    return success();
  if (auto arg = dyn_cast<BlockArgument>(token))
    return arg.getOwner()->getParentOp()->emitError(
               "unsupported block argument token in retain analysis"),
           failure();
  if (Operation *def = token.getDefiningOp())
    return def->emitError(
               "unsupported token producer while tracing emit sources"),
           failure();
  return failure();
}

static LogicalResult
traceEmitSources(EmitOp emit, llvm::DenseMap<Value, StringAttr> &takePlaces,
                 llvm::SmallVectorImpl<ObservableSource> &sources) {
  llvm::SmallVector<TakeOp, 4> takes;
  llvm::SmallPtrSet<Value, 16> visited;
  if (failed(collectTokenSources(emit.getToken(), takes, visited)))
    return failure();
  llvm::DenseSet<Value> seen;
  for (TakeOp take : takes) {
    Value tok = take.getResult();
    if (!seen.insert(tok).second)
      continue;
    auto placeIt = takePlaces.find(tok);
    if (placeIt == takePlaces.end())
      continue;
    sources.push_back({placeIt->second, tok});
  }
  if (sources.size() > 1)
    llvm::sort(sources, lessSource);
  return success();
}

static void
collectControlContexts(EmitOp emit, TransitionOp trans,
                       llvm::SmallVectorImpl<ControlContext> &contexts) {
  Operation *parent = emit->getParentOp();
  while (parent && parent != trans) {
    Block *emitBlock = emit->getBlock();
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      Region &thenRegion = ifOp.getThenRegion();
      bool inThen = blockInRegion(emitBlock, thenRegion);
      bool hidden = ifOp->hasAttr("lpn.hidden_choice");
      contexts.push_back({ifOp.getOperation(),
                          hidden ? ContextKind::ChoiceOp : ContextKind::IfOp,
                          inThen});
    } else if (auto choice = dyn_cast<ChoiceOp>(parent)) {
      Region &thenRegion = choice.getThenRegion();
      bool inThen = blockInRegion(emitBlock, thenRegion);
      contexts.push_back(
          {choice.getOperation(), ContextKind::ChoiceOp, inThen});
    } else if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      contexts.push_back({forOp.getOperation(), ContextKind::ForOp, true});
    }
    parent = parent->getParentOp();
  }
  std::reverse(contexts.begin(), contexts.end());
}

static bool guardsEqual(ArrayRef<TokenGuard> lhs, ArrayRef<TokenGuard> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [a, b] : llvm::zip(lhs, rhs)) {
    if (a.equalsValue != b.equalsValue)
      return false;
    if ((a.key && !b.key) || (!a.key && b.key))
      return false;
    if (a.key && a.keyHash != b.keyHash)
      return false;
  }
  return true;
}

static bool editsEqual(ArrayRef<TokenEditSignature> lhs,
             ArrayRef<TokenEditSignature> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [a, b] : llvm::zip(lhs, rhs))
    if (a.keyHash != b.keyHash || a.valueHash != b.valueHash)
      return false;
  return true;
}

static bool contextsEqual(ArrayRef<ControlContext> lhs,
               ArrayRef<ControlContext> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [a, b] : llvm::zip(lhs, rhs))
    if (a.op != b.op || a.isThen != b.isThen)
      return false;
  return true;
}

static bool equivalentTemplate(const EdgeTemplate *lhs,
                    const EdgeTemplate *rhs) {
  if (lhs == rhs)
    return true;
  if (!lhs || !rhs)
    return false;
  if (lhs->driver != rhs->driver)
    return false;
  if (lhs->sources.size() != rhs->sources.size())
    return false;
  for (auto [a, b] : llvm::zip(lhs->sources, rhs->sources))
    if (a.place != b.place)
      return false;
  if (lhs->target.symbol != rhs->target.symbol)
    return false;
  if (!guardsEqual(lhs->target.guards, rhs->target.guards))
    return false;
  if (!editsEqual(lhs->editSummary, rhs->editSummary))
    return false;
  if (!contextsEqual(lhs->contexts, rhs->contexts))
    return false;
  if (lhs->tokenHash != rhs->tokenHash || lhs->delayHash != rhs->delayHash)
    return false;
  return true;
}

static bool equivalentPath(const EdgePath &lhs, const EdgePath &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [a, b] : llvm::zip(lhs, rhs))
    if (!equivalentTemplate(a, b))
      return false;
  return true;
}

static void clusterHyperedges(TokenFlowAnalysisResult &result) {
  result.clusteredHyperedges = 0;
  for (auto &entry : result.adjacency) {
    llvm::SmallVector<const EdgeTemplate *> unique;
    for (const EdgeTemplate *templ : entry.second) {
      bool duplicate = llvm::any_of(unique, [&](const EdgeTemplate *seen) {
        return equivalentTemplate(templ, seen);
      });
      if (!duplicate)
        unique.push_back(templ);
    }
    result.clusteredHyperedges += unique.size();
    entry.second.swap(unique);
  }
}

static void dfsPaths(StringAttr root, StringAttr current,
           DenseSet<StringAttr> &visited, EdgePath &prefix,
           const ObservableSet &observables,
           TokenFlowAnalysisResult &result) {
  if (observables.contains(current) && current != root) {
    result.observablePaths[root].push_back(prefix);
    ++result.rawPaths;
    return;
  }
  auto it = result.adjacency.find(current);
  if (it == result.adjacency.end())
    return;
  for (const EdgeTemplate *templ : it->second) {
    StringAttr next = templ->target.symbol;
    if (!visited.insert(next).second)
      continue;
    prefix.push_back(templ);
    dfsPaths(root, next, visited, prefix, observables, result);
    prefix.pop_back();
    visited.erase(next);
  }
}

static void dedupObservablePaths(TokenFlowAnalysisResult &result) {
  result.retainedPaths = 0;
  for (auto &entry : result.observablePaths) {
    llvm::SmallVector<EdgePath> unique;
    for (EdgePath &path : entry.second) {
      bool duplicate = llvm::any_of(unique, [&](const EdgePath &seen) {
        return equivalentPath(path, seen);
      });
      if (!duplicate)
        unique.push_back(path);
    }
    result.retainedPaths += unique.size();
    entry.second.swap(unique);
  }
}

static void enumerateObservablePaths(const ObservableSet &observables,
                      TokenFlowAnalysisResult &result) {
  result.observablePaths.clear();
  result.rawPaths = 0;
  result.retainedPaths = 0;
  if (result.adjacency.empty())
    return;
  for (StringAttr root : observables) {
    DenseSet<StringAttr> visited;
    visited.insert(root);
    EdgePath prefix;
    dfsPaths(root, root, visited, prefix, observables, result);
  }
  if (result.rawPaths == 0)
    return;
  dedupObservablePaths(result);
}

static bool collectGuardClosureSources(
    EmitOp emit,
    const llvm::DenseMap<unsigned, llvm::SmallVector<llvm::SmallVector<ObservableSource, 4>, 4>>
        &guardIdClosures,
    llvm::SmallVectorImpl<llvm::SmallVector<ObservableSource, 4>> &out) {
  auto guardAttr = emit->getAttrOfType<ArrayAttr>(kGuardPathsAttr);
  if (!guardAttr)
    return false;
  bool used = false;
  for (Attribute attr : guardAttr) {
    auto pathAttr = dyn_cast<ArrayAttr>(attr);
    if (!pathAttr)
      continue;
    llvm::SmallVector<llvm::SmallVector<ObservableSource, 4>, 4> combos;
    combos.push_back({});
    for (Attribute elem : pathAttr) {
      auto intAttr = dyn_cast<IntegerAttr>(elem);
      if (!intAttr)
        continue;
      unsigned guardId = intAttr.getInt();
      auto it = guardIdClosures.find(guardId);
      if (it == guardIdClosures.end() || it->second.empty())
        continue;
      llvm::SmallVector<llvm::SmallVector<ObservableSource, 4>, 4> next;
      for (const auto &base : combos) {
        for (const auto &addition : it->second) {
          llvm::SmallVector<ObservableSource, 4> merged(base.begin(),
                                                       base.end());
          appendUniqueSources(addition, merged);
          canonicalizeSources(merged);
          next.push_back(std::move(merged));
        }
      }
      if (!next.empty())
        combos.swap(next);
    }
    for (auto &combo : combos) {
      if (combo.empty())
        continue;
      out.push_back(std::move(combo));
      used = true;
    }
  }
  return used;
}

struct TokenFlowAnalysisImpl {
  TokenFlowAnalysisImpl(NetOp net, const ObservableSet &observables,
                        const GuardTransitiveClosureResult &guardClosure,
                        TokenFlowAnalysisResult &result)
      : net(net), observables(observables), guardClosure(guardClosure),
        result(result) {}

  LogicalResult run() {
    takePlaces.clear();
    guardIdToTake.clear();
    guardIdClosures.clear();
    if (failed(collectTakeMetadata()))
      return failure();
    for (TransitionOp trans : net.getOps<TransitionOp>())
      if (failed(processTransition(trans)))
        return failure();
    return success();
  }

private:
  LogicalResult collectTakeMetadata();
  LogicalResult populateGuardClosures(unsigned guardId, Value takeValue);
  LogicalResult processTransition(TransitionOp trans) {
    WalkResult walkResult = trans->walk([&](EmitOp emit) -> WalkResult {
      if (failed(processEmit(trans, emit)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return failure();
    return success();
  }

  LogicalResult processEmit(TransitionOp trans, EmitOp emit) {
    llvm::SmallVector<ControlContext, 4> contexts;
    collectControlContexts(emit, trans, contexts);

    llvm::SmallVector<TargetInfo, 4> targets;
    if (failed(resolveEmitTargets(emit.getPlace(), targets)))
      return emit.emitError("failed to resolve emit targets"), failure();

    llvm::SmallVector<ObservableSource, 4> ssaSources;
    if (failed(traceEmitSources(emit, takePlaces, ssaSources)))
      return failure();

    llvm::SmallVector<llvm::SmallVector<ObservableSource, 4>, 4> baseCandidates;
    if (!ssaSources.empty())
      addDriverVariants(ssaSources, baseCandidates);

  llvm::SmallVector<llvm::SmallVector<ObservableSource, 4>, 4> guardSources;
  bool guardDerived =
    collectGuardClosureSources(emit, guardIdClosures, guardSources);

    llvm::SmallVector<llvm::SmallVector<ObservableSource, 4>, 4>
        candidateSources;
    if (baseCandidates.empty())
      return success();
    if (!guardSources.empty()) {
      for (const auto &base : baseCandidates) {
        for (const auto &extras : guardSources) {
          llvm::SmallVector<ObservableSource, 4> merged(base.begin(),
                                                        base.end());
          for (const ObservableSource &src : extras) {
            bool exists = llvm::any_of(merged, [&](const ObservableSource &it) {
              return it.takeValue == src.takeValue;
            });
            if (!exists)
              merged.push_back(src);
          }
          if (merged.size() > 1)
            llvm::sort(merged.begin() + 1, merged.end(), lessSource);
          candidateSources.push_back(std::move(merged));
        }
      }
    } else {
      candidateSources = baseCandidates;
      guardDerived = false;
    }

    if (candidateSources.empty())
      return success();

    for (const TargetInfo &target : targets) {
      for (const auto &sources : candidateSources) {
        if (sources.empty())
          continue;
        auto templ = std::make_unique<EdgeTemplate>();
        templ->driver = sources.front().place;
        templ->sources.assign(sources.begin(), sources.end());
        templ->target = target;
        templ->tokenValue = emit.getToken();
        templ->delayValue = emit.getDelay();
        templ->contexts.assign(contexts.begin(), contexts.end());

        llvm::DenseMap<Value, llvm::hash_code> hashCache;
        if (failed(summarizeTokenEdits(emit.getToken(),
                                       sources.front().takeValue,
                                       templ->editSummary, hashCache, sources)))
          return failure();
        templ->tokenHash = hashOptionalValue(emit.getToken(), hashCache);
        templ->delayHash = hashOptionalValue(emit.getDelay(), hashCache);

        const EdgeTemplate *ptr = templ.get();
        result.adjacency[templ->driver].push_back(ptr);
        result.templates.push_back(std::move(templ));
        ++result.totalHyperedges;
        if (guardDerived)
          ++result.guardHyperedges;
      }
    }
    return success();
  }

  NetOp net;
  const ObservableSet &observables;
  const GuardTransitiveClosureResult &guardClosure;
  TokenFlowAnalysisResult &result;
  llvm::DenseMap<Value, StringAttr> takePlaces;
  llvm::DenseMap<unsigned, Value> guardIdToTake;
  llvm::DenseMap<unsigned, llvm::SmallVector<llvm::SmallVector<ObservableSource, 4>, 4>>
      guardIdClosures;
};

LogicalResult TokenFlowAnalysisImpl::collectTakeMetadata() {
  for (TransitionOp trans : net.getOps<TransitionOp>()) {
    for (TakeOp take : trans.getBody().getOps<TakeOp>()) {
      StringAttr place;
      if (failed(resolvePlaceSymbol(take.getPlace(), place)))
        return take.emitError("failed to resolve place symbol");
      takePlaces[take.getResult()] = place;
      if (auto guardAttr = take->getAttrOfType<IntegerAttr>(kGuardIdAttr)) {
        unsigned guardId = guardAttr.getInt();
        guardIdToTake[guardId] = take.getResult();
      }
    }
  }
  for (const auto &entry : guardIdToTake)
    if (failed(populateGuardClosures(entry.first, entry.second)))
      return failure();
  return success();
}

LogicalResult TokenFlowAnalysisImpl::populateGuardClosures(unsigned guardId,
                                                           Value takeValue) {
  llvm::SmallVector<llvm::SmallVector<ObservableSource, 4>, 4> converted;
  ArrayRef<GuardTransitiveClosureResult::GuardCombination> closures =
      guardClosure.getClosures(takeValue);
  if (closures.empty()) {
    auto placeIt = takePlaces.find(takeValue);
    if (placeIt != takePlaces.end() &&
        observables.contains(placeIt->second)) {
      ObservableSource src{placeIt->second, takeValue};
      converted.push_back({src});
    }
    guardIdClosures[guardId] = std::move(converted);
    return success();
  }

  for (const auto &combo : closures) {
    if (combo.empty()) {
      converted.push_back({});
      continue;
    }
    llvm::SmallVector<ObservableSource, 4> sources;
    for (Value guardTake : combo) {
      auto placeIt = takePlaces.find(guardTake);
      if (placeIt == takePlaces.end())
        continue;
      if (!observables.contains(placeIt->second))
        continue;
      sources.push_back({placeIt->second, guardTake});
    }
    if (sources.empty())
      continue;
    canonicalizeSources(sources);
    converted.push_back(std::move(sources));
  }
  guardIdClosures[guardId] = std::move(converted);
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
  if (failed(impl.run()))
    return failure();
  clusterHyperedges(result);
  enumerateObservablePaths(observables, result);
  return success();
}

} // namespace mlir::lpn
