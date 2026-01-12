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
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <string>

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
  // The builder inserts a default scf.yield; remove it so we can rebuild the
  // body and terminator based on retained contexts.
  if (Operation *terminator = newBlock.getTerminator())
    terminator->erase();
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

struct GuardConditionSpec {
  Value tokenValue;
  TokenGuard guard;
};

static LogicalResult buildGuardCondition(ArrayRef<GuardConditionSpec> specs,
                                         IRMapping &mapping,
                                         ImplicitLocOpBuilder &builder,
                                         Value &condition) {
  condition = Value();
  if (specs.empty())
    return success();
  auto i64Ty = IntegerType::get(builder.getContext(), 64);
  for (const GuardConditionSpec &spec : specs) {
    if (!spec.guard.key)
      continue;
    Value token = mapping.lookupOrNull(spec.tokenValue);
    if (!token)
      token = cloneValueInto(spec.tokenValue, mapping, builder);
    if (!token)
      return failure();
    Value keyValue = cloneValueInto(spec.guard.key, mapping, builder);
    if (!keyValue)
      return failure();
    Value lhs = builder.create<TokenGetOp>(i64Ty, token, keyValue);
    Value rhs = builder.create<arith::ConstantOp>(
        builder.getI64IntegerAttr(spec.guard.equalsValue));
    Value eq =
        builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lhs, rhs);
    condition = condition ? builder.create<arith::AndIOp>(condition, eq) : eq;
  }
  return success();
}

struct SourceSetKey {
  llvm::SmallVector<const void *, 4> takes;
};

struct SourceSetKeyInfo : llvm::DenseMapInfo<SourceSetKey> {
  static inline SourceSetKey getEmptyKey() {
    SourceSetKey key;
    key.takes.push_back(
        llvm::DenseMapInfo<const void *>::getEmptyKey());
    return key;
  }
  static inline SourceSetKey getTombstoneKey() {
    SourceSetKey key;
    key.takes.push_back(
        llvm::DenseMapInfo<const void *>::getTombstoneKey());
    return key;
  }
  static unsigned getHashValue(const SourceSetKey &key) {
    return llvm::hash_combine_range(key.takes.begin(), key.takes.end());
  }
  static bool isEqual(const SourceSetKey &lhs, const SourceSetKey &rhs) {
    return lhs.takes == rhs.takes;
  }
};

struct GroupedPath {
  StringAttr target;
  const ObservablePath *path = nullptr;
};

struct TransitionGroup {
  llvm::SmallVector<GroupedPath, 8> paths;
  llvm::SmallVector<StringAttr, 4> sourcePlaces;
};

static SourceSetKey buildSourceKey(ArrayRef<ObservableSource> sources) {
  SourceSetKey key;
  for (const ObservableSource &src : sources)
    key.takes.push_back(src.takeValue.getAsOpaquePointer());
  llvm::sort(key.takes);
  key.takes.erase(std::unique(key.takes.begin(), key.takes.end()),
                  key.takes.end());
  return key;
}

static llvm::SmallVector<StringAttr, 4>
collectSourcePlaces(ArrayRef<ObservableSource> sources) {
  llvm::DenseSet<StringAttr> seen;
  llvm::SmallVector<StringAttr, 4> result;
  for (const ObservableSource &src : sources)
    if (seen.insert(src.place).second)
      result.push_back(src.place);
  llvm::sort(result, [](StringAttr lhs, StringAttr rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  return result;
}

static std::string
buildGroupBaseName(ArrayRef<StringAttr> sourcePlaces) {
  if (sourcePlaces.empty())
    return "retain";
  std::string name;
  for (StringAttr place : sourcePlaces) {
    if (!name.empty())
      name.append("_");
    name.append(place.getValue().str());
  }
  name.append("_retain");
  return name;
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
    registry.insert<arith::ArithDialect, scf::SCFDialect>();
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
    
    // Group sinks that share the same observable takes so each transition
    // consumes shared tokens only once.
    llvm::DenseMap<SourceSetKey, TransitionGroup, SourceSetKeyInfo> groups;
    llvm::SmallVector<StringAttr, 8> sinks;
    sinks.reserve(analysisResult.observablePaths.size());
    for (const auto &entry : analysisResult.observablePaths)
      sinks.push_back(entry.first);
    llvm::sort(sinks, [](StringAttr lhs, StringAttr rhs) {
      return lhs.getValue() < rhs.getValue();
    });

    for (StringAttr sink : sinks) {
      auto it = analysisResult.observablePaths.find(sink);
      if (it == analysisResult.observablePaths.end())
        continue;
      const auto &paths = it->second;
      if (paths.empty())
        continue;
      for (const ObservablePath &path : paths) {
        SourceSetKey key = buildSourceKey(path.sources);
        TransitionGroup &group = groups[key];
        if (group.sourcePlaces.empty())
          group.sourcePlaces = collectSourcePlaces(path.sources);
        group.paths.push_back({sink, &path});
      }
    }

    struct GroupRecord {
      std::string baseName;
      TransitionGroup *group = nullptr;
    };
    llvm::SmallVector<GroupRecord, 8> groupRecords;
    groupRecords.reserve(groups.size());
    for (auto &entry : groups)
      groupRecords.push_back(
          {buildGroupBaseName(entry.second.sourcePlaces), &entry.second});
    llvm::sort(groupRecords, [](const GroupRecord &lhs,
                                const GroupRecord &rhs) {
      return lhs.baseName < rhs.baseName;
    });

    std::string lastBaseName;
    unsigned nameSuffix = 0;
    for (const GroupRecord &record : groupRecords) {
      if (!record.group || record.group->paths.empty())
        continue;
      std::string transName = record.baseName;
      if (record.baseName == lastBaseName) {
        ++nameSuffix;
        transName =
            record.baseName + "_" + std::to_string(nameSuffix);
      } else {
        lastBaseName = record.baseName;
        nameSuffix = 0;
      }

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

      for (const GroupedPath &grouped : record.group->paths) {
        if (!grouped.path)
          continue;
        const ObservablePath &path = *grouped.path;
        StringAttr targetPlace = grouped.target;
        IRMapping mapping;
        bool skipPath = false;
        llvm::SmallVector<GuardConditionSpec, 4> pathGuards;

        auto appendContexts = [](llvm::SmallVectorImpl<ControlContext> &dst,
                                 ArrayRef<ControlContext> src) {
          for (const ControlContext &ctx : src) {
            if (!dst.empty() && dst.back() == ctx)
              continue;
            dst.push_back(ctx);
          }
        };

        auto appendGuards = [&](const EmitMetadata &metadata) {
          for (const TokenGuard &guard : metadata.target.guards)
            pathGuards.push_back({metadata.tokenValue, guard});
        };

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

        llvm::SmallVector<ControlContext, 8> pathContexts;
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
          appendContexts(pathContexts, metadata.contexts);
          appendGuards(metadata);
          // If the token value is already mapped, avoid re-emitting the
          // control-context shell that would otherwise be empty.
          if (Value mappedToken = mapping.lookupOrNull(metadata.tokenValue)) {
            mapping.map(segment.producedTake, mappedToken);
            return success();
          }
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
        appendGuards(terminalMeta);
        llvm::SmallVector<ControlContext, 8> terminalContexts = pathContexts;
        appendContexts(terminalContexts, terminalMeta.contexts);
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
          Value guardCondition;
          if (failed(buildGuardCondition(pathGuards, ctxMapping, ctxBuilder,
                                         guardCondition)))
            return failure();
          if (guardCondition) {
            scf::IfOp ifOp = ctxBuilder.create<scf::IfOp>(guardCondition,
                                                         /*withElse=*/false);
            Block &thenBlock = ifOp.getThenRegion().front();
            thenBlock.getOperations().clear();
            ImplicitLocOpBuilder thenBuilder(ctxBuilder.getLoc(),
                                             ctxBuilder.getContext());
            thenBuilder.setInsertionPointToStart(&thenBlock);
            thenBuilder.create<EmitOp>(targetRef, newToken, delay);
            thenBuilder.create<scf::YieldOp>();
            ctxBuilder.setInsertionPointAfter(ifOp);
          } else {
            ctxBuilder.create<EmitOp>(targetRef, newToken, delay);
          }
          return success();
        };
        if (failed(runWithContexts(terminalContexts, mapping, builder,
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

    if (result.observablePaths.empty())
      return;

    llvm::SmallVector<StringAttr, 8> sinks;
    sinks.reserve(result.observablePaths.size());
    for (const auto &entry : result.observablePaths)
      sinks.push_back(entry.first);
    llvm::sort(sinks, [](StringAttr lhs, StringAttr rhs) {
      return lhs.getValue() < rhs.getValue();
    });

    for (StringAttr sink : sinks) {
      auto it = result.observablePaths.find(sink);
      if (it == result.observablePaths.end())
        continue;
      net.emitRemark() << "[lpn-guard-paths] sink=" << sink
                       << " paths=" << it->second.size();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLPNRetainHypergraphPass() {
  return std::make_unique<LPNRetainHypergraphPass>();
}

} // namespace mlir::lpn
