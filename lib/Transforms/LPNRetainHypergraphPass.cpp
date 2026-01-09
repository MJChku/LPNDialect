#include "LPN/Analysis/TokenFlowAnalysis.h"
#include "LPN/Conversion/LPNPasses.h"
#include "LPN/Dialect/LPNOps.h"
#include "LPN/Dialect/LPNTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <iterator>
#include <memory>

namespace mlir::lpn {
namespace {

//===----------------------------------------------------------------------===//
// Pass overview
//===----------------------------------------------------------------------===//
//
// The hypergraph retain pass collapses an arbitrary LPN network down to the
// user-marked observable places without making structural assumptions such as
// “each place has at most one producer and consumer”.  Instead, it treats the
// network as a hypergraph whose vertices are observable places and whose
// hyperedges capture the exact set of observable tokens consumed together:
//
//   1. For every `lpn.take` result we enumerate all reachable `lpn.emit`
//      operations by walking the SSA use-def chain.  The SSA slice from the
//      take to each emit is summarized as an EdgeTemplate that records the
//      canonical driver observable, the multiset of observable takes used,
//      the cloned token/delay expressions, and metadata (guards, edit log,
//      control context, and expression fingerprints).
//
//   2. We cluster identical EdgeTemplates per driver place so structurally
//      equivalent hyperedges are stored only once.  Equivalence requires the
//      same driver, observable source set, guard predicates, edit signatures,
//      SSA fingerprints, and nesting of control operations.
//
//   3. The hyperedges induce a directed graph between observables.  We walk
//      this graph via DFS to enumerate every observable-to-observable path.
//      Paths that revisit an observable terminate, ensuring cycles do not
//      explode the search space.
//
//   4. For each observable root with outgoing paths we synthesize a retained
//      transition.  The transition takes from the root once, then replays each
//      path by rematerializing the recorded slices.  Additional observables
//      required by a hyperedge are taken exactly once per synthesized firing;
//      guards become `scf.if`, arbitrary control constructs (e.g., nested ifs)
//      are cloned, and per-edge delays are accumulated.
//
// Because hyperedges track multi-source consumption explicitly, the pass can
// retain nets where transitions require multiple observable tokens at once,
// where emissions are routed via data-dependent control flow, and where
// different latent paths share large sub-slices.  The resulting retained net
// is minimal with respect to the chosen observables yet faithful to both the
// token edits and the causal relationships encoded in the original MLIR.
//
//   * Before enumerating observable paths we cluster identical hyperedges per
//     driver place.  Hyperedges with the same driver, observable source set,
//     control contexts, token edits, guard predicates, and delay/token SSA
//     slices collapse to a single template.  This ensures the retained network
//     does not duplicate transitions just because several paths in the original
//     net were structurally identical.

//===----------------------------------------------------------------------===//

/// Utilities to simplify redundant lpn.choice ladders.
static bool regionIsTriviallyEmpty(Region &region) {
  if (region.empty())
    return true;
  return region.front().getOperations().size() == 1;
}

static ChoiceOp getSingletonChoice(Region &region) {
  if (!region.hasOneBlock())
    return nullptr;
  auto range = region.front().without_terminator();
  if (!llvm::hasSingleElement(range))
    return nullptr;
  return dyn_cast<ChoiceOp>(&*range.begin());
}

static bool spliceBranchIntoParent(Region &from, Block &destBlock) {
  if (from.empty())
    return false;
  Block &srcBlock = from.front();
  auto &srcOps = srcBlock.getOperations();
  if (srcOps.empty())
    return false;
  auto srcBegin = srcOps.begin();
  auto srcEnd = srcOps.end();
  --srcEnd;
  if (srcBegin == srcEnd)
    return false;
  auto &destOps = destBlock.getOperations();
  auto insertIt = destBlock.getTerminator()->getIterator();
  destOps.splice(insertIt, srcOps, srcBegin, srcEnd);
  return true;
}

static bool simplifyChoiceOnce(ChoiceOp op) {
  bool thenEmpty = regionIsTriviallyEmpty(op.getThenRegion());
  bool elseEmpty = regionIsTriviallyEmpty(op.getElseRegion());
  if (thenEmpty && elseEmpty) {
    op.erase();
    return true;
  }
  if (thenEmpty == elseEmpty)
    return false;

  Region &candidateRegion = thenEmpty ? op.getElseRegion() : op.getThenRegion();
  ChoiceOp inner = getSingletonChoice(candidateRegion);
  if (!inner)
    return false;

  bool innerThenEmpty = regionIsTriviallyEmpty(inner.getThenRegion());
  bool innerElseEmpty = regionIsTriviallyEmpty(inner.getElseRegion());
  if (innerThenEmpty == innerElseEmpty)
    return false;

  Region &nonEmptyRegion =
      innerThenEmpty ? inner.getElseRegion() : inner.getThenRegion();
  Block &destBlock = candidateRegion.front();
  if (!spliceBranchIntoParent(nonEmptyRegion, destBlock))
    return false;
  inner.erase();
  return true;
}

static void simplifyChoiceLadders(NetOp net) {
  bool changed = true;
  while (changed) {
    changed = false;
    net.walk([&](ChoiceOp op) -> WalkResult {
      if (simplifyChoiceOnce(op)) {
        changed = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
}

/// Observable-to-observable chain of edges.
using EdgePath = SmallVector<const EdgeTemplate *>;

struct PathCursor {
  const EdgePath *path;
  size_t edgeIndex;
  size_t contextIndex;
};

/// Recursively clone the SSA slice for `value`.
static Value cloneValueInto(Value value, IRMapping &mapping,
                            ImplicitLocOpBuilder &builder) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (Value mapped = mapping.lookupOrNull(arg))
      return mapped;
    arg.getOwner()->getParentOp()->emitError(
        "unmapped block argument while cloning hypergraph slice");
    return {};
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return {};

  for (Value operand : def->getOperands())
    if (!cloneValueInto(operand, mapping, builder))
      return {};

  Operation *clone = builder.clone(*def, mapping);
  auto result = cast<OpResult>(value);
  return clone->getResult(result.getResultNumber());
}

/// Helper to ensure a valid accumulated delay value exists.
static Value ensureDelay(Value delay, ImplicitLocOpBuilder &builder) {
  if (delay)
    return delay;
  return builder.create<arith::ConstantOp>(builder.getF64FloatAttr(0.0));
}

using TokenEnv = DenseMap<StringAttr, SmallVector<Value>>;
using TakeEnv = DenseMap<Value, Value>;
using SSAEnv = DenseMap<Value, Value>;

struct CursorState {
  PathCursor cursor;
  Value token;
  Value delay;
  TokenEnv tokens;
  TakeEnv takes;
  SSAEnv ssa;
};

static void mapContextValues(const SSAEnv &env, IRMapping &mapping) {
  for (const auto &entry : env)
    mapping.map(entry.first, entry.second);
}

static void mapTemplateSources(const EdgeTemplate *templ, IRMapping &mapping,
                               const TakeEnv &takes) {
  for (const ObservableSource &src : templ->sources)
    if (Value mapped = takes.lookup(src.takeValue))
      mapping.map(src.takeValue, mapped);
}

static LogicalResult ensureTemplateSources(const EdgeTemplate *templ,
                                           Value driverToken, TokenEnv &tokens,
                                           TakeEnv &takes,
                                           ImplicitLocOpBuilder &builder) {
  if (templ->sources.empty())
    return failure();
  Value driverTake = templ->sources.front().takeValue;
  auto placeType = PlaceType::get(builder.getContext());
  auto tokenType = TokenType::get(builder.getContext());
  for (const ObservableSource &src : templ->sources) {
    if (takes.contains(src.takeValue))
      continue;
    if (src.takeValue == driverTake) {
      takes[src.takeValue] = driverToken;
      continue;
    }
    SmallVector<Value> &queue = tokens[src.place];
    Value token;
    if (!queue.empty()) {
      token = queue.pop_back_val();
    } else {
      Value ref = builder.create<PlaceRefOp>(placeType,
                                             FlatSymbolRefAttr::get(src.place));
      token = builder.create<TakeOp>(tokenType, ref);
    }
    takes[src.takeValue] = token;
  }
  return success();
}

static Value buildGuardCondition(const SmallVectorImpl<TokenGuard> &guards,
                                 const EdgeTemplate *templ, CursorState &state,
                                 ImplicitLocOpBuilder &builder) {
  if (guards.empty())
    return {};
  auto i64Ty = IntegerType::get(builder.getContext(), 64);
  IRMapping mapping;
  mapContextValues(state.ssa, mapping);
  mapTemplateSources(templ, mapping, state.takes);
  if (templ->sources.empty())
    return {};
  mapping.map(templ->sources.front().takeValue, state.token);
  Value condition;
  for (const TokenGuard &guard : guards) {
    if (!guard.key)
      continue;
    Value keyValue = cloneValueInto(guard.key, mapping, builder);
    if (!keyValue)
      return {};
    Value lhs = builder.create<TokenGetOp>(i64Ty, state.token, keyValue);
    Value rhs = builder.create<arith::ConstantOp>(
        builder.getI64IntegerAttr(guard.equalsValue));
    Value eq =
        builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lhs, rhs);
    condition = condition ? builder.create<arith::AndIOp>(condition, eq) : eq;
  }
  return condition;
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

struct LPNRetainHypergraphPass
    : PassWrapper<LPNRetainHypergraphPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNRetainHypergraphPass)

  struct NetStats {
    uint64_t totalHyperedges = 0;
    uint64_t guardHyperedges = 0;
    uint64_t clusteredHyperedges = 0;
    uint64_t rawPaths = 0;
    uint64_t retainedPaths = 0;
    uint64_t synthesizedTransitions = 0;
  };

  StringRef getArgument() const final { return "lpn-retain-hypergraph"; }
  StringRef getDescription() const final {
    return "EXPERIMENTAL: hypergraph-based observable reduction.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](NetOp net) {
      if (failed(processNet(net)))
        signalPassFailure();
    });
  }

  LogicalResult processNet(NetOp net) {
    NetStats stats;
    SmallVector<PlaceOp> observablePlaces;
    DenseSet<StringAttr> observableNames;
    for (PlaceOp place : net.getOps<PlaceOp>())
      if (place.getObservableAttr()) {
        observablePlaces.push_back(place);
        observableNames.insert(place.getSymNameAttr());
      }
    if (observablePlaces.size() < 2)
      return success();

    TokenFlowAnalysisResult analysisResult;
    if (failed(runTokenFlowAnalysis(net, observableNames, analysisResult)))
      return failure();

    auto &adjacency = analysisResult.adjacency;
    stats.totalHyperedges = analysisResult.totalHyperedges;
    stats.guardHyperedges = analysisResult.guardHyperedges;

    stats.clusteredHyperedges = analysisResult.clusteredHyperedges;
    stats.rawPaths = analysisResult.rawPaths;
    stats.retainedPaths = analysisResult.retainedPaths;

    if (adjacency.empty()) {
      reportNetStats(net, stats);
      return success();
    }

    auto &observablePaths = analysisResult.observablePaths;

    if (observablePaths.empty()) {
      reportNetStats(net, stats);
      return success();
    }

    HaltOp halt = nullptr;
    for (Operation &op : net.getBody().getOps())
      if ((halt = dyn_cast<HaltOp>(&op)))
        break;
    if (!halt)
      return net.emitError("net missing lpn.halt terminator");

    SmallVector<Operation *> originalTransitions;
    for (TransitionOp trans : net.getOps<TransitionOp>())
      originalTransitions.push_back(trans);
    SmallVector<Operation *> removablePlaces;
    for (PlaceOp place : net.getOps<PlaceOp>())
      if (!place.getObservableAttr())
        removablePlaces.push_back(place);

    Block &body = net.getBody().front();
    OpBuilder topBuilder(&body, Block::iterator(halt));
    MLIRContext *ctx = net.getContext();
    for (auto &entry : observablePaths) {
      if (entry.second.empty())
        continue;
      std::string transName = (entry.first.getValue() + "_retain").str();
      auto trans = topBuilder.create<TransitionOp>(
          net.getLoc(), topBuilder.getStringAttr(transName));
      Region &region = trans.getBody();
      auto *block = new Block();
      region.push_back(block);
      ImplicitLocOpBuilder builder(net.getLoc(), ctx);
      builder.setInsertionPointToStart(block);
      auto placeRef = builder.create<PlaceRefOp>(
          PlaceType::get(ctx), FlatSymbolRefAttr::get(entry.first));
      Value seed = builder.create<TakeOp>(TokenType::get(ctx), placeRef);
      SmallVector<CursorState, 8> states;
      TokenEnv seedTokens;
      seedTokens[entry.first] = SmallVector<Value>{seed};
      for (const EdgePath &path : entry.second) {
        CursorState state;
        state.cursor = PathCursor{&path, 0, 0};
        state.token = seed;
        state.delay = Value();
        state.tokens = seedTokens;
        states.push_back(std::move(state));
      }
      if (failed(emitCursorSet(std::move(states), builder)))
        return failure();
      builder.create<ScheduleReturnOp>();
      ++stats.synthesizedTransitions;
    }

    for (Operation *op : originalTransitions)
      op->erase();
    for (Operation *op : removablePlaces)
      op->erase();

    simplifyChoiceLadders(net);
    reportNetStats(net, stats);
    return success();
  }

  LogicalResult emitCursorSet(SmallVector<CursorState> states,
                              ImplicitLocOpBuilder &builder) const;
  LogicalResult emitIfContext(const ControlContext &ctx, CursorState state,
                              ImplicitLocOpBuilder &builder) const;
  LogicalResult emitChoiceContext(const ControlContext &ctx, CursorState state,
                                  ImplicitLocOpBuilder &builder) const;
  LogicalResult emitForContext(const ControlContext &ctx, CursorState state,
                               ImplicitLocOpBuilder &builder) const;
  LogicalResult emitLeaf(CursorState state, const EdgeTemplate *templ,
                         ImplicitLocOpBuilder &builder) const;
  void reportNetStats(NetOp net, const NetStats &stats) const;

  const EdgeTemplate *getTemplate(const PathCursor &cursor) const {
    if (!cursor.path || cursor.edgeIndex >= cursor.path->size())
      return nullptr;
    return (*(cursor.path))[cursor.edgeIndex];
  }

  void
  dfsPaths(StringAttr root, StringAttr current, DenseSet<StringAttr> &visited,
           EdgePath &prefix, const DenseSet<StringAttr> &observables,
           DenseMap<StringAttr, SmallVector<const EdgeTemplate *>> &adjacency,
           DenseMap<StringAttr, SmallVector<EdgePath>> &paths) const {
    if (observables.contains(current) && current != root) {
      paths[root].push_back(prefix);
      return;
    }
    auto it = adjacency.find(current);
    if (it == adjacency.end())
      return;
    for (const EdgeTemplate *templ : it->second) {
      StringAttr next = templ->target.symbol;
      if (!visited.insert(next).second)
        continue;
      prefix.push_back(templ);
      dfsPaths(root, next, visited, prefix, observables, adjacency, paths);
      prefix.pop_back();
      visited.erase(next);
    }
  }
};

} // namespace

LogicalResult
LPNRetainHypergraphPass::emitCursorSet(SmallVector<CursorState> states,
                                       ImplicitLocOpBuilder &builder) const {
  if (states.empty())
    return success();

  for (CursorState &stateRef : states) {
    CursorState state = std::move(stateRef);
    const EdgeTemplate *templ = getTemplate(state.cursor);
    if (!templ)
      continue;
    if (state.cursor.contextIndex < templ->contexts.size()) {
      const ControlContext &ctx = templ->contexts[state.cursor.contextIndex];
      switch (ctx.kind) {
      case ControlContext::Kind::IfOp:
        if (failed(emitIfContext(ctx, std::move(state), builder)))
          return failure();
        break;
      case ControlContext::Kind::ChoiceOp:
        if (failed(emitChoiceContext(ctx, std::move(state), builder)))
          return failure();
        break;
      case ControlContext::Kind::ForOp:
        if (failed(emitForContext(ctx, std::move(state), builder)))
          return failure();
        break;
      case ControlContext::Kind::Unknown:
        break;
      }
      continue;
    }
    if (failed(emitLeaf(std::move(state), templ, builder)))
      return failure();
  }
  return success();
}

LogicalResult
LPNRetainHypergraphPass::emitIfContext(const ControlContext &ctx,
                                       CursorState state,
                                       ImplicitLocOpBuilder &builder) const {
  auto ifOp = dyn_cast<scf::IfOp>(ctx.op);
  if (!ifOp)
    return builder.getInsertionBlock()->getParentOp()->emitError(
        "unsupported control context");
  const EdgeTemplate *templ = getTemplate(state.cursor);
  if (!templ)
    return success();
  TokenEnv condTokens = state.tokens;
  TakeEnv condTakes = state.takes;
  if (failed(ensureTemplateSources(templ, state.token, condTokens, condTakes,
                                   builder)))
    return failure();
  IRMapping mapping;
  mapContextValues(state.ssa, mapping);
  mapTemplateSources(templ, mapping, condTakes);
  Value cond = cloneValueInto(ifOp.getCondition(), mapping, builder);
  if (!cond)
    return failure();
  bool hasElse = !ifOp.getElseRegion().empty() || !ctx.inThen;
  scf::IfOp cloned = builder.create<scf::IfOp>(TypeRange(), cond, hasElse);
  auto populateEmpty = [&](Region &region) -> LogicalResult {
    region.getBlocks().clear();
    Block &block = region.emplaceBlock();
    ImplicitLocOpBuilder branchBuilder(builder.getLoc(), builder.getContext());
    branchBuilder.setInsertionPointToStart(&block);
    branchBuilder.create<scf::YieldOp>();
    return success();
  };
  auto populateActive = [&](Region &region,
                            CursorState branchState) -> LogicalResult {
    region.getBlocks().clear();
    Block &block = region.emplaceBlock();
    ImplicitLocOpBuilder branchBuilder(builder.getLoc(), builder.getContext());
    branchBuilder.setInsertionPointToStart(&block);
    CursorState next = std::move(branchState);
    next.cursor.contextIndex++;
    SmallVector<CursorState, 1> advanced;
    advanced.push_back(std::move(next));
    if (failed(emitCursorSet(std::move(advanced), branchBuilder)))
      return failure();
    branchBuilder.create<scf::YieldOp>();
    return success();
  };
  if (ctx.inThen) {
    if (failed(populateActive(cloned.getThenRegion(), std::move(state))))
      return failure();
    if (hasElse)
      if (failed(populateEmpty(cloned.getElseRegion())))
        return failure();
  } else {
    if (failed(populateEmpty(cloned.getThenRegion())))
      return failure();
    if (failed(populateActive(cloned.getElseRegion(), std::move(state))))
      return failure();
  }
  builder.setInsertionPointAfter(cloned);
  return success();
}

LogicalResult
LPNRetainHypergraphPass::emitChoiceContext(const ControlContext &ctx,
                                           CursorState state,
                                           ImplicitLocOpBuilder &builder) const {
  auto choice = dyn_cast<ChoiceOp>(ctx.op);
  if (!choice)
    return builder.getInsertionBlock()->getParentOp()->emitError(
        "unsupported choice context");
  ChoiceOp cloned = builder.create<ChoiceOp>();
  auto populateEmpty = [&](Region &region) -> LogicalResult {
    region.getBlocks().clear();
    Block &block = region.emplaceBlock();
    ImplicitLocOpBuilder branchBuilder(builder.getLoc(), builder.getContext());
    branchBuilder.setInsertionPointToStart(&block);
    branchBuilder.create<ChoiceYieldOp>();
    return success();
  };
  auto populateActive = [&](Region &region,
                            CursorState branchState) -> LogicalResult {
    region.getBlocks().clear();
    Block &block = region.emplaceBlock();
    ImplicitLocOpBuilder branchBuilder(builder.getLoc(), builder.getContext());
    branchBuilder.setInsertionPointToStart(&block);
    CursorState next = std::move(branchState);
    next.cursor.contextIndex++;
    SmallVector<CursorState, 1> advanced;
    advanced.push_back(std::move(next));
    if (failed(emitCursorSet(std::move(advanced), branchBuilder)))
      return failure();
    branchBuilder.create<ChoiceYieldOp>();
    return success();
  };
  if (ctx.inThen) {
    if (failed(populateActive(cloned.getThenRegion(), std::move(state))))
      return failure();
    if (failed(populateEmpty(cloned.getElseRegion())))
      return failure();
  } else {
    if (failed(populateEmpty(cloned.getThenRegion())))
      return failure();
    if (failed(populateActive(cloned.getElseRegion(), std::move(state))))
      return failure();
  }
  builder.setInsertionPointAfter(cloned);
  return success();
}

LogicalResult
LPNRetainHypergraphPass::emitForContext(const ControlContext &ctx,
                                        CursorState state,
                                        ImplicitLocOpBuilder &builder) const {
  auto forOp = dyn_cast<scf::ForOp>(ctx.op);
  if (!forOp)
    return builder.getInsertionBlock()->getParentOp()->emitError(
        "unsupported loop context");
  const EdgeTemplate *templ = getTemplate(state.cursor);
  if (!templ)
    return success();
  TokenEnv loopTokens = state.tokens;
  TakeEnv loopTakes = state.takes;
  if (failed(ensureTemplateSources(templ, state.token, loopTokens, loopTakes,
                                   builder)))
    return failure();
  IRMapping mapping;
  mapContextValues(state.ssa, mapping);
  mapTemplateSources(templ, mapping, loopTakes);
  Value lower = cloneValueInto(forOp.getLowerBound(), mapping, builder);
  Value upper = cloneValueInto(forOp.getUpperBound(), mapping, builder);
  Value step = cloneValueInto(forOp.getStep(), mapping, builder);
  if (!lower || !upper || !step)
    return failure();
  scf::ForOp cloned = builder.create<scf::ForOp>(lower, upper, step);
  Block &body = cloned.getRegion().front();
  body.clear();
  ImplicitLocOpBuilder inner(builder.getLoc(), builder.getContext());
  inner.setInsertionPointToStart(&body);
  CursorState next = std::move(state);
  next.cursor.contextIndex++;
  next.ssa[forOp.getBody()->getArgument(0)] = cloned.getInductionVar();
  SmallVector<CursorState, 1> advanced;
  advanced.push_back(std::move(next));
  if (failed(emitCursorSet(std::move(advanced), inner)))
    return failure();
  inner.create<scf::YieldOp>();
  builder.setInsertionPointAfter(cloned);
  return success();
}

LogicalResult
LPNRetainHypergraphPass::emitLeaf(CursorState state, const EdgeTemplate *templ,
                                  ImplicitLocOpBuilder &builder) const {
  if (failed(ensureTemplateSources(templ, state.token, state.tokens,
                                   state.takes, builder)))
    return failure();
  auto emitBody = [&](CursorState innerState,
                      ImplicitLocOpBuilder &inner) -> LogicalResult {
    IRMapping mapping;
    mapContextValues(innerState.ssa, mapping);
    mapTemplateSources(templ, mapping, innerState.takes);
    Value newToken = cloneValueInto(templ->tokenValue, mapping, inner);
    if (!newToken)
      return failure();
    Value stepDelay = cloneValueInto(templ->delayValue, mapping, inner);
    if (!stepDelay)
      return failure();
    Value totalDelay = ensureDelay(innerState.delay, inner);
    totalDelay = inner.create<arith::AddFOp>(totalDelay, stepDelay).getResult();
    innerState.tokens[templ->target.symbol].push_back(newToken);
    bool last = innerState.cursor.path && (innerState.cursor.edgeIndex + 1 ==
                                           innerState.cursor.path->size());
    if (last) {
      auto placeType = PlaceType::get(inner.getContext());
      auto placeAttr = FlatSymbolRefAttr::get(templ->target.symbol);
      Value place = inner.create<PlaceRefOp>(placeType, placeAttr);
      inner.create<EmitOp>(place, newToken, totalDelay);
      return success();
    }
    CursorState next = std::move(innerState);
    next.cursor.edgeIndex++;
    next.cursor.contextIndex = 0;
    next.token = newToken;
    next.delay = totalDelay;
    SmallVector<CursorState, 1> children;
    children.push_back(std::move(next));
    return emitCursorSet(std::move(children), inner);
  };

  Value cond = buildGuardCondition(templ->target.guards, templ, state, builder);
  if (!cond)
    return emitBody(std::move(state), builder);

  scf::IfOp guardIf =
      builder.create<scf::IfOp>(TypeRange(), cond, /*withElse=*/false);
  auto &guardBlock = guardIf.getThenRegion().front();
  guardBlock.clear();
  ImplicitLocOpBuilder inner(builder.getLoc(), builder.getContext());
  inner.setInsertionPointToStart(&guardBlock);
  if (failed(emitBody(std::move(state), inner)))
    return failure();
  inner.create<scf::YieldOp>();
  builder.setInsertionPointAfter(guardIf);
  return success();
}

void LPNRetainHypergraphPass::reportNetStats(NetOp net,
                                             const NetStats &stats) const {
  uint64_t fallback = 0;
  if (stats.totalHyperedges >= stats.guardHyperedges)
    fallback = stats.totalHyperedges - stats.guardHyperedges;
  auto diag = net.emitRemark();
  diag << "[lpn-retain-hypergraph] hyperedges(before cluster)="
       << stats.totalHyperedges << " (guard=" << stats.guardHyperedges
       << ", fallback=" << fallback << "), unique=" << stats.clusteredHyperedges
       << "; paths=" << stats.retainedPaths << "/" << stats.rawPaths
       << "; transitions=" << stats.synthesizedTransitions;
}

std::unique_ptr<Pass> createLPNRetainHypergraphPass() {
  return std::make_unique<LPNRetainHypergraphPass>();
}

} // namespace mlir::lpn
