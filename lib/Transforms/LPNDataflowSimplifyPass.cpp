#include "LPN/Conversion/LPNPasses.h"
#include "LPN/Dialect/LPNOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Hashing.h"
#include <tuple>
#include <algorithm>

namespace mlir::lpn {
namespace {

static bool regionIsTrivial(Region &region) {
  if (region.empty())
    return true;
  Block &block = region.front();
  return block.without_terminator().empty();
}

static bool valuesEquivalent(Value lhs, Value rhs, IRMapping &mapping) {
  if (lhs == rhs)
    return true;
  if (Value mapped = mapping.lookupOrNull(lhs))
    return mapped == rhs;
  return false;
}

static bool regionsStructurallyEqual(Region &lhs, Region &rhs,
                                     IRMapping mapping);

static bool blocksStructurallyEqual(Block &lhs, Block &rhs,
                                    IRMapping mapping);

static bool opsStructurallyEqual(Operation *lhs, Operation *rhs,
                                 IRMapping &mapping) {
  if (lhs->getName() != rhs->getName())
    return false;
  if (lhs->getAttrs() != rhs->getAttrs())
    return false;
  if (lhs->getNumOperands() != rhs->getNumOperands())
    return false;
  for (auto [a, b] : llvm::zip(lhs->getOperands(), rhs->getOperands()))
    if (!valuesEquivalent(a, b, mapping))
      return false;
  if (lhs->getNumRegions() != rhs->getNumRegions())
    return false;
  for (auto [leftRegion, rightRegion] :
       llvm::zip(lhs->getRegions(), rhs->getRegions()))
    if (!regionsStructurallyEqual(const_cast<Region &>(leftRegion),
                                  const_cast<Region &>(rightRegion), mapping))
      return false;
  if (lhs->getNumResults() != rhs->getNumResults())
    return false;
  for (auto [a, b] : llvm::zip(lhs->getResults(), rhs->getResults()))
    mapping.map(a, b);
  return true;
}

static bool blocksStructurallyEqual(Block &lhs, Block &rhs,
                                    IRMapping mapping) {
  if (lhs.getNumArguments() != rhs.getNumArguments())
    return false;
  for (auto [a, b] : llvm::zip(lhs.getArguments(), rhs.getArguments()))
    mapping.map(a, b);
  auto itL = lhs.begin();
  auto itR = rhs.begin();
  for (; itL != lhs.end() && itR != rhs.end(); ++itL, ++itR)
    if (!opsStructurallyEqual(&*itL, &*itR, mapping))
      return false;
  return itL == lhs.end() && itR == rhs.end();
}

static bool regionsStructurallyEqual(Region &lhs, Region &rhs,
                                     IRMapping mapping) {
  if (lhs.getBlocks().size() != rhs.getBlocks().size())
    return false;
  auto itL = lhs.begin();
  auto itR = rhs.begin();
  for (; itL != lhs.end() && itR != rhs.end(); ++itL, ++itR)
    if (!blocksStructurallyEqual(*itL, *itR, mapping))
      return false;
  return true;
}

static bool branchesStructurallyEqual(scf::IfOp ifOp) {
  if (ifOp.getNumResults() != 0)
    return false;
  Region &thenRegion = ifOp.getThenRegion();
  Region &elseRegion = ifOp.getElseRegion();
  if (thenRegion.empty() || elseRegion.empty())
    return false;
  IRMapping mapping;
  return regionsStructurallyEqual(thenRegion, elseRegion, mapping);
}

static void inlineBranchAndErase(scf::IfOp ifOp, Region &region) {
  if (region.empty()) {
    ifOp.erase();
    return;
  }
  Block &thenBlock = region.front();
  OpBuilder builder(ifOp);
  IRMapping mapping;
  for (Operation &op : llvm::make_early_inc_range(
           thenBlock.without_terminator()))
    builder.clone(op, mapping);
  ifOp.erase();
}

/// Hash SSA expressions so we can summarize observable effects.
static llvm::hash_code
hashValueExpr(Value value, DenseMap<Value, llvm::hash_code> &cache) {
  if (!value)
    return llvm::hash_value(0);
  if (auto it = cache.find(value); it != cache.end())
    return it->second;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    llvm::hash_code h = llvm::hash_combine(
        reinterpret_cast<uintptr_t>(arg.getOwner()), arg.getArgNumber());
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
  if (auto placeRef = dyn_cast<PlaceRefOp>(def)) {
    llvm::hash_code h = llvm::hash_combine(
        llvm::hash_value("PlaceRef"), resultNumber,
        llvm::hash_value(placeRef.getPlaceAttr().getAttr()));
    cache[value] = h;
    return h;
  }
  llvm::hash_code h =
      llvm::hash_combine(llvm::hash_value(def->getName().getStringRef()),
                         resultNumber);
  for (NamedAttribute attr : def->getAttrs())
    h = llvm::hash_combine(
        h, llvm::hash_value(attr.getName()),
        llvm::hash_value(attr.getValue().getAsOpaquePointer()));
  for (Value operand : def->getOperands())
    h = llvm::hash_combine(h, hashValueExpr(operand, cache));
  cache[value] = h;
  return h;
}

struct EmitSignature {
  llvm::hash_code placeHash;
  llvm::hash_code tokenHash;
  llvm::hash_code delayHash;

  bool operator==(const EmitSignature &rhs) const {
    return placeHash == rhs.placeHash && tokenHash == rhs.tokenHash &&
           delayHash == rhs.delayHash;
  }

  bool operator<(const EmitSignature &rhs) const {
    return std::tie(placeHash, tokenHash, delayHash) <
           std::tie(rhs.placeHash, rhs.tokenHash, rhs.delayHash);
  }
};

static void collectEmitSignatures(Operation *op,
                                  SmallVectorImpl<EmitSignature> &signatures,
                                  DenseMap<Value, llvm::hash_code> &cache) {
  if (auto emit = dyn_cast<EmitOp>(op)) {
    EmitSignature sig{hashValueExpr(emit.getPlace(), cache),
                      hashValueExpr(emit.getToken(), cache),
                      hashValueExpr(emit.getDelay(), cache)};
    signatures.push_back(sig);
    return;
  }
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &nested :
           llvm::make_early_inc_range(block.without_terminator()))
        collectEmitSignatures(&nested, signatures, cache);
}

static void summarizeBranchEffects(Region &region,
                                   SmallVectorImpl<EmitSignature> &summary) {
  DenseMap<Value, llvm::hash_code> cache;
  for (Block &block : region)
    for (Operation &op :
         llvm::make_early_inc_range(block.without_terminator()))
      collectEmitSignatures(&op, summary, cache);
  llvm::sort(summary);
  summary.erase(std::unique(summary.begin(), summary.end()),
                summary.end());
}

static bool branchesEffectivelyEqual(scf::IfOp ifOp) {
  if (ifOp.getNumResults() != 0)
    return false;
  SmallVector<EmitSignature> thenSummary;
  SmallVector<EmitSignature> elseSummary;
  summarizeBranchEffects(ifOp.getThenRegion(), thenSummary);
  summarizeBranchEffects(ifOp.getElseRegion(), elseSummary);
  return thenSummary == elseSummary;
}

struct LPNDataflowSimplifyPass
    : PassWrapper<LPNDataflowSimplifyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNDataflowSimplifyPass)

  StringRef getArgument() const final { return "lpn-dataflow-simplify"; }
  StringRef getDescription() const final {
    return "Fold structurally or effect-wise equivalent control flow.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<scf::IfOp, 16> guards;
    module.walk([&](scf::IfOp ifOp) { guards.push_back(ifOp); });
    for (scf::IfOp ifOp : guards) {
      if (ifOp->getParentOp() == nullptr || ifOp->use_empty() == false)
        continue;
      if (ifOp.getNumResults() != 0)
        continue;
      bool thenTrivial = regionIsTrivial(ifOp.getThenRegion());
      bool elseTrivial = regionIsTrivial(ifOp.getElseRegion());
      if (thenTrivial && elseTrivial) {
        ifOp.erase();
        continue;
      }
      auto constCond =
          ifOp.getCondition().getDefiningOp<arith::ConstantOp>();
      if (!constCond)
        continue;
      auto condAttr = dyn_cast<IntegerAttr>(constCond.getValue());
      if (!condAttr)
        continue;
      bool condTrue = !condAttr.getValue().isZero();
      Region &keep = condTrue ? ifOp.getThenRegion() : ifOp.getElseRegion();
      if (keep.empty())
        continue;
      Block &block = keep.front();
      if (!block.getArguments().empty())
        continue;
      auto ops = block.without_terminator();
      if (ops.empty()) {
        ifOp.erase();
        continue;
      }
      auto &parentOps = ifOp->getBlock()->getOperations();
      parentOps.splice(Block::iterator(ifOp), block.getOperations(),
                       ops.begin(), ops.end());
      ifOp.erase();
    }

    SmallVector<scf::IfOp, 8> foldCandidates;
    module.walk([&](scf::IfOp ifOp) {
      if (ifOp.getNumResults() != 0)
        return;
      if (branchesStructurallyEqual(ifOp) ||
          branchesEffectivelyEqual(ifOp))
        foldCandidates.push_back(ifOp);
    });
    for (scf::IfOp ifOp : foldCandidates)
      inlineBranchAndErase(ifOp, ifOp.getThenRegion());

    SmallVector<Block *, 16> blocks;
    module.walk([&](Block *block) { blocks.push_back(block); });
    for (Block *block : blocks) {
      DenseMap<Value, llvm::hash_code> hashCache;
      SmallVector<EmitSignature, 4> seen;
      for (Operation &op :
           llvm::make_early_inc_range(block->getOperations())) {
        if (auto emit = dyn_cast<EmitOp>(&op)) {
          EmitSignature sig{hashValueExpr(emit.getPlace(), hashCache),
                            hashValueExpr(emit.getToken(), hashCache),
                            hashValueExpr(emit.getDelay(), hashCache)};
          bool duplicate = llvm::any_of(seen, [&](const EmitSignature &other) {
            return sig == other;
          });
          if (duplicate) {
            emit.erase();
            continue;
          }
          seen.push_back(sig);
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLPNDataflowSimplifyPass() {
  return std::make_unique<LPNDataflowSimplifyPass>();
}

} // namespace mlir::lpn
