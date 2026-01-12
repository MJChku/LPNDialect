#include "LPN/Analysis/GuardTransitiveClosure.h"
#include "LPN/Dialect/LPNOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

namespace mlir::lpn {
namespace {

static constexpr StringLiteral kGuardIdAttr = "lpn.guard_id";
static constexpr StringLiteral kGuardPathsAttr = "lpn.guard_paths";

enum class GuardFeasibility {
  Feasible,
  Blocking,
  Cyclic,
};

struct GuardNode {
  Value takeValue;
  StringAttr place;
  bool isObservable = false;
  GuardFeasibility feasibility = GuardFeasibility::Blocking;
  bool hasCycle = false;
  struct PathRecord {
    GuardPathVariant variant;
    llvm::DenseMap<StringAttr, llvm::SmallVector<StringAttr, 4>> edges;
    llvm::DenseSet<StringAttr> nodes;
  };
  llvm::SmallVector<PathRecord, 4> paths;
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

static bool isReachable(const GuardNode::PathRecord &record, StringAttr start,
                        StringAttr target) {
  if (start == target)
    return true;
  auto it = record.edges.find(start);
  if (it == record.edges.end())
    return false;
  llvm::SmallVector<StringAttr, 8> stack(it->second.begin(),
                                         it->second.end());
  llvm::DenseSet<StringAttr> visited;
  visited.insert(start);
  while (!stack.empty()) {
    StringAttr current = stack.pop_back_val();
    if (current == target)
      return true;
    if (!visited.insert(current).second)
      continue;
    auto nextIt = record.edges.find(current);
    if (nextIt == record.edges.end())
      continue;
    for (StringAttr next : nextIt->second)
      if (!visited.contains(next))
        stack.push_back(next);
  }
  return false;
}

static bool addEdgeIfAcyclic(GuardNode::PathRecord &record, StringAttr from,
                             StringAttr to) {
  if (!from || !to || from == to)
    return false;
  if (isReachable(record, to, from))
    return false;
  record.nodes.insert(from);
  record.nodes.insert(to);
  record.edges[from].push_back(to);
  return true;
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

static bool recordsLess(const GuardNode::PathRecord &lhs,
                        const GuardNode::PathRecord &rhs) {
  return variantsLess(lhs.variant, rhs.variant);
}

static bool recordsEqual(const GuardNode::PathRecord &lhs,
                         const GuardNode::PathRecord &rhs) {
  return variantsEqual(lhs.variant, rhs.variant);
}

static void dedupRecordSet(
    llvm::SmallVectorImpl<GuardNode::PathRecord> &records) {
  for (auto &record : records)
    canonicalizeVariant(record.variant);
  llvm::sort(records.begin(), records.end(), recordsLess);
  records.erase(std::unique(records.begin(), records.end(), recordsEqual),
                records.end());
}

static bool intersectsNodeSets(const llvm::DenseSet<StringAttr> &lhs,
                               const llvm::DenseSet<StringAttr> &rhs) {
  if (lhs.empty() || rhs.empty())
    return false;
  if (lhs.size() <= rhs.size()) {
    for (StringAttr node : lhs)
      if (rhs.contains(node))
        return true;
    return false;
  }
  for (StringAttr node : rhs)
    if (lhs.contains(node))
      return true;
  return false;
}

static bool hasDirectedCycle(
    const llvm::DenseMap<StringAttr, llvm::SmallVector<StringAttr, 4>> &edges,
    const llvm::DenseSet<StringAttr> &nodes) {
  llvm::DenseMap<StringAttr, unsigned> indegree;
  indegree.reserve(nodes.size());
  unsigned nodeCount = 0;
  for (StringAttr node : nodes)
    if (indegree.insert({node, 0}).second)
      ++nodeCount;
  for (const auto &entry : edges) {
    if (indegree.insert({entry.first, 0}).second)
      ++nodeCount;
    for (StringAttr dst : entry.second) {
      if (indegree.insert({dst, 0}).second)
        ++nodeCount;
      ++indegree[dst];
    }
  }

  llvm::SmallVector<StringAttr, 16> queue;
  queue.reserve(nodes.size());
  for (const auto &entry : indegree)
    if (entry.second == 0)
      queue.push_back(entry.first);

  unsigned processed = 0;
  while (!queue.empty()) {
    StringAttr node = queue.pop_back_val();
    ++processed;
    auto it = edges.find(node);
    if (it == edges.end())
      continue;
    for (StringAttr dst : it->second) {
      auto degIt = indegree.find(dst);
      if (degIt == indegree.end())
        continue;
      if (--degIt->second == 0)
        queue.push_back(dst);
    }
  }
  return processed != nodeCount;
}

static llvm::SmallVector<GuardNode::PathRecord, 4>
cartProductVariants(llvm::ArrayRef<GuardNode::PathRecord> lhs,
                    llvm::ArrayRef<GuardNode::PathRecord> rhs) {
  llvm::SmallVector<GuardNode::PathRecord, 4> result;
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
      GuardNode::PathRecord combo;
      combo.variant.observables.append(left.variant.observables.begin(),
                                       left.variant.observables.end());
      combo.variant.observables.append(right.variant.observables.begin(),
                                       right.variant.observables.end());
      combo.variant.segments.append(left.variant.segments.begin(),
                                    left.variant.segments.end());
      combo.variant.segments.append(right.variant.segments.begin(),
                                    right.variant.segments.end());

      combo.nodes = left.nodes;
      for (StringAttr node : right.nodes)
        combo.nodes.insert(node);
      combo.edges = left.edges;
      for (const auto &entry : right.edges)
        combo.edges[entry.first].append(entry.second.begin(),
                                        entry.second.end());

      if (intersectsNodeSets(left.nodes, right.nodes) &&
          hasDirectedCycle(combo.edges, combo.nodes))
        continue;
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

  struct ProducerCombos {
    llvm::SmallVector<GuardNode::PathRecord, 4> combos;
    llvm::SmallVector<Value, 4> usedGuards;
    bool feasible = true;
    bool blocked = false;
    bool sawCycle = false;
  };

  LogicalResult initialize();
  LogicalResult compute(Value takeValue, llvm::DenseSet<Value> &visiting);
  LogicalResult collectProducerCombos(GuardNode &node,
                                      const ProducerPath &producer,
                                      llvm::DenseSet<Value> &visiting,
                                      ProducerCombos &out);
  void appendProducerCombos(GuardNode &node, const ProducerPath &producer,
                            ProducerCombos &combos, bool &sawCycle);
};

LogicalResult GuardClosureContext::initialize() {
  for (PlaceOp place : net.getOps<PlaceOp>())
    placeLookup[place.getSymNameAttr()] = place;

  for (TransitionOp trans : net.getOps<TransitionOp>()) {
    llvm::DenseMap<unsigned, Value> guardIdToTake;
    LogicalResult takeStatus = success();
    auto takeWalk = trans.walk([&](TakeOp take) -> WalkResult {
      auto guardAttr = take->getAttrOfType<IntegerAttr>(kGuardIdAttr);
      if (!guardAttr)
        return WalkResult::advance();
      unsigned guardId = guardAttr.getInt();
      guardIdToTake[guardId] = take.getResult();
      GuardNode &node = nodes[take.getResult()];
      node.takeValue = take.getResult();
      if (failed(resolvePlaceSymbol(take.getPlace(), node.place))) {
        take.emitError("failed to resolve place symbol");
        takeStatus = failure();
        return WalkResult::interrupt();
      }
      node.isObservable = observables.contains(node.place);
      return WalkResult::advance();
    });
    if (takeWalk.wasInterrupted())
      return takeStatus;

    LogicalResult emitStatus = success();
    auto emitWalk = trans.walk([&](EmitOp emit) -> WalkResult {
      auto guardsAttr = emit->getAttrOfType<ArrayAttr>(kGuardPathsAttr);
      if (!guardsAttr || guardsAttr.empty())
        return WalkResult::advance();
      llvm::SmallVector<StringAttr, 4> targets;
      if (failed(collectEmitPlaces(emit.getPlace(), targets)))
        return emitStatus = emit.emitError("failed to resolve emit targets"),
               WalkResult::interrupt();
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
      return WalkResult::advance();
    });
    if (emitWalk.wasInterrupted())
      return emitStatus;
  }

  return success();
}

LogicalResult GuardClosureContext::collectProducerCombos(
    GuardNode &node, const ProducerPath &producer,
    llvm::DenseSet<Value> &visiting, ProducerCombos &out) {
  // Phase 1: traverse guard takes for this producer path, drop cyclic guards,
  // and build the cartesian product of non-cyclic child variants.
  out.combos.clear();
  out.combos.emplace_back();
  out.usedGuards.clear();
  out.feasible = true;
  out.blocked = false;
  out.sawCycle = false;

  for (Value guardTake : producer.guardTakes) {
    auto guardNodeIt = nodes.find(guardTake);
    if (guardNodeIt != nodes.end() &&
        guardNodeIt->second.place == node.place) {
      // Same-place dependency implies a hidden cycle in place space.
      out.feasible = false;
      out.sawCycle = true;
      break;
    }
    if (visiting.contains(guardTake)) {
      // Depth-first recursion hit a cycle in the guard closure graph.
      out.feasible = false;
      out.sawCycle = true;
      break;
    }
    if (failed(compute(guardTake, visiting)))
      return failure();

    auto &childNode = nodes[guardTake];
    // Cyclic guard dependencies are skipped so that non-cyclic guards can
    // still contribute observable paths.
    if (childNode.hasCycle) {
      // This guard is cyclic; drop it so other guards can still be resolved.
      out.sawCycle = true;
      continue;
    }

    auto &childPaths = childNode.paths;
    if (childPaths.empty()) {
      if (childNode.feasibility == GuardFeasibility::Blocking) {
        // Blocking guard: no producers and no initial tokens.
        out.feasible = false;
        out.blocked = true;
        break;
      }
      // Non-blocking but cyclic/empty: skip this guard for this producer path.
      out.sawCycle = true;
      continue;
    }

    out.usedGuards.push_back(guardTake);
    out.combos = cartProductVariants(out.combos, childPaths);
    if (out.combos.empty()) {
      // No compatible cartesian product left after combining guard variants.
      out.feasible = false;
      out.sawCycle = true;
      break;
    }
  }

  if (out.feasible && !producer.guardTakes.empty() && out.usedGuards.empty()) {
    // If every guard is cyclic, this producer path is effectively unusable.
    out.feasible = false;
    out.sawCycle = true;
  }

  return success();
}

void GuardClosureContext::appendProducerCombos(GuardNode &node,
                                               const ProducerPath &producer,
                                               ProducerCombos &combos,
                                               bool &sawCycle) {
  // Phase 2: merge guard variants with this producer path and re-check cycles
  // introduced by combining multiple guards that were individually acyclic.
  bool addedVariant = false;
  for (auto &variant : combos.combos) {
    bool acyclic = true;
    for (Value guardTake : combos.usedGuards) {
      auto guardIt = nodes.find(guardTake);
      if (guardIt == nodes.end()) {
        acyclic = false;
        break;
      }
      if (!node.isObservable && !guardIt->second.isObservable) {
        // The combined variant may form a new hidden cycle (A->B, B->A) even if
        // each guard was individually acyclic; check after merging edges.
        if (!addEdgeIfAcyclic(variant, node.place, guardIt->second.place)) {
          acyclic = false;
          break;
        }
      }
    }
    if (!acyclic) {
      sawCycle = true;
      continue;
    }
    GuardPathSegment segment;
    segment.emit = producer.emitOp;
    segment.pathIndex = producer.pathIndex;
    segment.target = producer.target;
    segment.producedTake = node.takeValue;
    variant.variant.segments.push_back(segment);
    node.paths.push_back(std::move(variant));
    addedVariant = true;
  }
  if (!addedVariant)
    sawCycle = true;
}

LogicalResult GuardClosureContext::compute(Value takeValue,
                                           llvm::DenseSet<Value> &visiting) {
  auto nodeIt = nodes.find(takeValue);
  if (nodeIt == nodes.end())
    return failure();
  GuardNode &node = nodeIt->second;
  if (node.state == GuardNode::State::Done)
    return success();
  if (visiting.contains(takeValue))
    return success();

  node.state = GuardNode::State::Visiting;
  visiting.insert(takeValue);
  bool sawBlocking = false;
  bool sawCycle = false;
  if (node.isObservable) {
    GuardNode::PathRecord record;
    record.variant.observables.push_back(takeValue);
    node.paths.push_back(std::move(record));
  } else {
    bool hasInitTokens = hasInitialTokens(node.place, placeLookup);
    auto prodIt = producers.find(node.place);
    if (prodIt == producers.end()) {
      if (hasInitTokens)
        node.paths.emplace_back();
      else
        sawBlocking = true;
    } else {
      for (const ProducerPath &producer : prodIt->second) {
        ProducerCombos combos;
        if (failed(collectProducerCombos(node, producer, visiting, combos)))
          return failure();
        if (combos.sawCycle)
          sawCycle = true;
        if (!combos.feasible) {
          if (combos.blocked)
            sawBlocking = true;
          continue;
        }
        appendProducerCombos(node, producer, combos, sawCycle);
      }
    }
  }

  if (!node.paths.empty()) {
    dedupRecordSet(node.paths);
    node.feasibility = GuardFeasibility::Feasible;
  } else if (sawBlocking) {
    node.feasibility = GuardFeasibility::Blocking;
  } else if (sawCycle) {
    node.feasibility = GuardFeasibility::Cyclic;
  } else {
    node.feasibility = GuardFeasibility::Blocking;
  }
  // Keep track of cyclic guard dependencies for downstream pruning.
  node.hasCycle = sawCycle;
  node.state = GuardNode::State::Done;
  visiting.erase(takeValue);
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

bool GuardTransitiveClosureResult::hasCycle(Value takeValue) const {
  auto it = guardHasCycle.find(takeValue);
  if (it == guardHasCycle.end())
    return false;
  return it->second;
}

LogicalResult runGuardTransitiveClosureAnalysis(
  NetOp net, const ObservableSet &observables,
  GuardTransitiveClosureResult &result) {
  GuardClosureContext ctx(net, observables);
  if (failed(ctx.initialize()))
    return failure();
  for (auto &entry : ctx.nodes) {
    llvm::DenseSet<Value> visiting;
    if (failed(ctx.compute(entry.first, visiting)))
      return failure();
  }
  for (auto &entry : ctx.nodes)
    for (auto &record : entry.second.paths)
      result.guardPaths[entry.first].push_back(record.variant);
  for (auto &entry : ctx.nodes)
    result.guardHasCycle[entry.first] = entry.second.hasCycle;
  return success();
}

} // namespace mlir::lpn
