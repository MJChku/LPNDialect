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
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::lpn {
namespace {

//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

/// Recursively clone SSA slice for value
static Value cloneValueInto(Value value, IRMapping &mapping,
                            ImplicitLocOpBuilder &builder) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    arg.getOwner()->getParentOp()->emitError(
        "unmapped block argument while cloning");
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

using ContextBodyFn =
    llvm::function_ref<LogicalResult(ImplicitLocOpBuilder &, IRMapping &)>;

static LogicalResult runContextChain(ArrayRef<ControlContext> contexts,
                                    unsigned depth, IRMapping &mapping,
                                    ImplicitLocOpBuilder &builder,
                                    ContextBodyFn body);

static LogicalResult runWithContexts(ArrayRef<ControlContext> contexts,
                                     IRMapping &mapping,
                                     ImplicitLocOpBuilder &builder,
                                     ContextBodyFn body) {
  return runContextChain(contexts, /*depth=*/0, mapping, builder, body);
}

static LogicalResult emitIfContext(const ControlContext &ctx,
                                   ArrayRef<ControlContext> contexts,
                                   unsigned depth, IRMapping &mapping,
                                   ImplicitLocOpBuilder &builder,
                                   ContextBodyFn body) {
  auto ifOp = dyn_cast<scf::IfOp>(ctx.op);
  if (!ifOp)
    return builder.getInsertionBlock()->getParentOp()->emitError(
        "control context mismatch: expected scf.if");
  Value cond = cloneValueInto(ifOp.getCondition(), mapping, builder);
  if (!cond)
    return failure();
  bool needElse = ctx.inThen ? !ifOp.getElseRegion().empty() : true;
  scf::IfOp cloned = builder.create<scf::IfOp>(cond, needElse);
  auto populateBranch = [&](Region &region, bool active) -> LogicalResult {
    region.getBlocks().clear();
    Block &block = region.emplaceBlock();
    ImplicitLocOpBuilder branchBuilder(builder.getLoc(), builder.getContext());
    branchBuilder.setInsertionPointToStart(&block);
    if (active) {
      if (failed(runContextChain(contexts, depth + 1, mapping, branchBuilder,
                                 body)))
        return failure();
    }
    branchBuilder.create<scf::YieldOp>();
    return success();
  };
  if (ctx.inThen) {
    if (failed(populateBranch(cloned.getThenRegion(), /*active=*/true)))
      return failure();
    if (needElse &&
        failed(populateBranch(cloned.getElseRegion(), /*active=*/false)))
      return failure();
  } else {
    if (failed(populateBranch(cloned.getThenRegion(), /*active=*/false)))
      return failure();
    if (failed(populateBranch(cloned.getElseRegion(), /*active=*/true)))
      return failure();
  }
  builder.setInsertionPointAfter(cloned);
  return success();
}

static LogicalResult emitChoiceContext(const ControlContext &ctx,
                                       ArrayRef<ControlContext> contexts,
                                       unsigned depth, IRMapping &mapping,
                                       ImplicitLocOpBuilder &builder,
                                       ContextBodyFn body) {
  auto choice = dyn_cast<ChoiceOp>(ctx.op);
  if (!choice)
    return builder.getInsertionBlock()->getParentOp()->emitError(
        "control context mismatch: expected lpn.choice");
  ChoiceOp cloned = builder.create<ChoiceOp>();
  auto populateBranch = [&](Region &region, bool active) -> LogicalResult {
    region.getBlocks().clear();
    Block &block = region.emplaceBlock();
    ImplicitLocOpBuilder branchBuilder(builder.getLoc(), builder.getContext());
    branchBuilder.setInsertionPointToStart(&block);
    if (active) {
      if (failed(runContextChain(contexts, depth + 1, mapping, branchBuilder,
                                 body)))
        return failure();
    }
    branchBuilder.create<ChoiceYieldOp>();
    return success();
  };
  if (ctx.inThen) {
    if (failed(populateBranch(cloned.getThenRegion(), /*active=*/true)))
      return failure();
    if (failed(populateBranch(cloned.getElseRegion(), /*active=*/false)))
      return failure();
  } else {
    if (failed(populateBranch(cloned.getThenRegion(), /*active=*/false)))
      return failure();
    if (failed(populateBranch(cloned.getElseRegion(), /*active=*/true)))
      return failure();
  }
  builder.setInsertionPointAfter(cloned);
  return success();
}

static LogicalResult emitForContext(const ControlContext &ctx,
                                    ArrayRef<ControlContext> contexts,
                                    unsigned depth, IRMapping &mapping,
                                    ImplicitLocOpBuilder &builder,
                                    ContextBodyFn body) {
  auto forOp = dyn_cast<scf::ForOp>(ctx.op);
  if (!forOp)
    return builder.getInsertionBlock()->getParentOp()->emitError(
        "control context mismatch: expected scf.for");
  Value lower = cloneValueInto(forOp.getLowerBound(), mapping, builder);
  if (!lower)
    return failure();
  Value upper = cloneValueInto(forOp.getUpperBound(), mapping, builder);
  if (!upper)
    return failure();
  Value step = cloneValueInto(forOp.getStep(), mapping, builder);
  if (!step)
    return failure();
  SmallVector<Value> initArgs;
  initArgs.reserve(forOp.getInitArgs().size());
  for (Value arg : forOp.getInitArgs()) {
    Value cloned = cloneValueInto(arg, mapping, builder);
    if (!cloned)
      return failure();
    initArgs.push_back(cloned);
  }
  scf::ForOp cloned = builder.create<scf::ForOp>(lower, upper, step, initArgs);
  Block &origBlock = *forOp.getBody();
  Block &newBlock = *cloned.getBody();
  ImplicitLocOpBuilder loopBuilder(builder.getLoc(), builder.getContext());
  loopBuilder.setInsertionPointToStart(&newBlock);

  SmallVector<Value> mappedValues;
  for (auto [origArg, newArg] : llvm::zip(origBlock.getArguments(),
                                          newBlock.getArguments())) {
    mapping.map(origArg, newArg);
    mappedValues.push_back(origArg);
  }

  if (failed(runContextChain(contexts, depth + 1, mapping, loopBuilder, body)))
    return failure();

  if (forOp.getNumRegionIterArgs() > 0) {
    SmallVector<Value> forwarded;
    for (Value arg : newBlock.getArguments().drop_front())
      forwarded.push_back(arg);
    loopBuilder.create<scf::YieldOp>(forwarded);
  } else {
    loopBuilder.create<scf::YieldOp>();
  }

  for (Value orig : mappedValues)
    mapping.erase(orig);

  builder.setInsertionPointAfter(cloned);
  return success();
}

static LogicalResult runContextChain(ArrayRef<ControlContext> contexts,
                                    unsigned depth, IRMapping &mapping,
                                    ImplicitLocOpBuilder &builder,
                                    ContextBodyFn body) {
  if (depth >= contexts.size())
    return body(builder, mapping);
  const ControlContext &ctx = contexts[depth];
  switch (ctx.kind) {
  case ControlContext::Kind::IfOp:
    return emitIfContext(ctx, contexts, depth, mapping, builder, body);
  case ControlContext::Kind::ChoiceOp:
    return emitChoiceContext(ctx, contexts, depth, mapping, builder, body);
  case ControlContext::Kind::ForOp:
    return emitForContext(ctx, contexts, depth, mapping, builder, body);
  case ControlContext::Kind::Unknown:
    return runContextChain(contexts, depth + 1, mapping, builder, body);
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

struct LPNRetainHypergraphPass
    : PassWrapper<LPNRetainHypergraphPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNRetainHypergraphPass)

  StringRef getArgument() const final { return "lpn-retain-hypergraph"; }
  StringRef getDescription() const final {
    return "EXPERIMENTAL: guard-closure-based observable reduction.";
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
    // Collect observables
    llvm::DenseSet<StringAttr> observableNames;
    for (PlaceOp place : net.getOps<PlaceOp>())
      if (place.getObservableAttr())
        observableNames.insert(place.getSymNameAttr());

    if (observableNames.size() < 2)
      return success();

    // Run analysis
    TokenFlowAnalysisResult analysisResult;
    if (failed(runTokenFlowAnalysis(net, observableNames, analysisResult)))
      return failure();

    if (analysisResult.observablePaths.empty()) {
      reportStats(net, analysisResult);
      return success();
    }

    // Find halt terminator
    HaltOp halt = nullptr;
    for (Operation &op : net.getBody().getOps())
      if ((halt = dyn_cast<HaltOp>(&op)))
        break;
    if (!halt)
      return net.emitError("net missing lpn.halt terminator");

    // Save original transitions and hidden places for removal
    llvm::SmallVector<Operation *> originalTransitions;
    for (TransitionOp trans : net.getOps<TransitionOp>())
      originalTransitions.push_back(trans);
    llvm::SmallVector<Operation *> removablePlaces;
    for (PlaceOp place : net.getOps<PlaceOp>())
      if (!place.getObservableAttr())
        removablePlaces.push_back(place);

    // Generate retained transitions
    Block &body = net.getBody().front();
    OpBuilder topBuilder(&body, Block::iterator(halt));
    MLIRContext *ctx = net.getContext();
    
    for (const auto &entry : analysisResult.observablePaths) {
      StringAttr targetPlace = entry.first;
      const auto &paths = entry.second;
      if (paths.empty())
        continue;

      std::string transName = (targetPlace.getValue() + "_retain").str();
      auto trans = topBuilder.create<TransitionOp>(
          net.getLoc(), topBuilder.getStringAttr(transName));
      Region &region = trans.getBody();
      auto *block = new Block();
      region.push_back(block);
      ImplicitLocOpBuilder builder(net.getLoc(), ctx);
      builder.setInsertionPointToStart(block);

      llvm::DenseMap<Value, Value> takeMapping;
      auto placeType = PlaceType::get(ctx);
      auto tokenType = TokenType::get(ctx);

      for (const ObservablePath &path : paths) {
        IRMapping mapping;
        bool skipPath = false;

        auto ensureGuardMapping = [&](ArrayRef<Value> guardTakes) -> LogicalResult {
          for (Value guardTake : guardTakes) {
            if (mapping.lookupOrNull(guardTake))
              continue;
            skipPath = true;
            return success();
          }
          return success();
        };

        auto getOrCreateTake = [&](const ObservableSource &src) -> Value {
          if (auto it = takeMapping.find(src.takeValue);
              it != takeMapping.end())
            return it->second;
          Value ref = builder.create<PlaceRefOp>(placeType,
                                                 FlatSymbolRefAttr::get(src.place));
          Value token = builder.create<TakeOp>(tokenType, ref);
          takeMapping[src.takeValue] = token;
          return token;
        };

        for (const ObservableSource &src : path.sources) {
          Value token = getOrCreateTake(src);
          mapping.map(src.takeValue, token);
        }

        auto materializeSegment = [&](const GuardPathSegment &segment)
            -> LogicalResult {
          EmitPathKey key{segment.emit, segment.pathIndex, segment.target};
          auto it = analysisResult.emitMetadata.find(key);
          if (it == analysisResult.emitMetadata.end())
            return net.emitError("missing metadata for guard path segment"),
                   failure();
          const EmitMetadata &metadata = it->second;
          if (failed(ensureGuardMapping(metadata.guardTakes)))
            return failure();
          if (skipPath)
            return success();
          auto body = [&](ImplicitLocOpBuilder &ctxBuilder,
                          IRMapping &ctxMapping) -> LogicalResult {
            Value tokenValue =
                cloneValueInto(metadata.tokenValue, ctxMapping, ctxBuilder);
            if (!tokenValue)
              return failure();
            ctxMapping.map(segment.producedTake, tokenValue);
            return success();
          };
          return runWithContexts(metadata.contexts, mapping, builder, body);
        };

        for (const GuardPathSegment &segment : path.prefixSegments) {
          if (failed(materializeSegment(segment))) {
            if (skipPath)
              break;
            return failure();
          }
        }
        if (skipPath)
          continue;

        auto termIt = analysisResult.emitMetadata.find(path.terminalKey);
        if (termIt == analysisResult.emitMetadata.end())
          return net.emitError("missing metadata for terminal emit"), failure();
        const EmitMetadata &terminalMeta = termIt->second;
        if (failed(ensureGuardMapping(terminalMeta.guardTakes)))
          return failure();
        if (skipPath)
          continue;
        auto emitBody = [&](ImplicitLocOpBuilder &ctxBuilder,
                            IRMapping &ctxMapping) -> LogicalResult {
          Value newToken =
              cloneValueInto(terminalMeta.tokenValue, ctxMapping, ctxBuilder);
          if (!newToken)
            return failure();

          Value delay = Value();
          if (terminalMeta.delayValue) {
            Value newDelay = cloneValueInto(terminalMeta.delayValue, ctxMapping,
                                            ctxBuilder);
            if (!newDelay)
              return failure();
            delay = newDelay;
          } else {
            delay = ctxBuilder.create<arith::ConstantOp>(
                ctxBuilder.getF64FloatAttr(0.0));
          }

          Value targetRef = ctxBuilder.create<PlaceRefOp>(
              placeType, FlatSymbolRefAttr::get(targetPlace));
          ctxBuilder.create<EmitOp>(targetRef, newToken, delay);
          return success();
        };
        if (failed(runWithContexts(terminalMeta.contexts, mapping, builder,
                                   emitBody)))
          return failure();
      }

      builder.create<ScheduleReturnOp>();
    }

    // Remove original transitions and hidden places
    for (Operation *op : originalTransitions)
      op->erase();
    for (Operation *op : removablePlaces)
      op->erase();

    reportStats(net, analysisResult);
    return success();
  }

  void reportStats(NetOp net, const TokenFlowAnalysisResult &result) const {
    auto diag = net.emitRemark();
    diag << "[lpn-retain-hypergraph] hyperedges=" << result.totalHyperedges
         << " (guard=" << result.guardHyperedges << ")";
  }
};

} // namespace

std::unique_ptr<Pass> createLPNRetainHypergraphPass() {
  return std::make_unique<LPNRetainHypergraphPass>();
}

} // namespace mlir::lpn
