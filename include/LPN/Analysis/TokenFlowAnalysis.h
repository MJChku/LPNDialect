//===- TokenFlowAnalysis.h - Token flow analysis for LPN ---------*- C++ -*-===//
//
// Emit-centric token flow analysis shared by retain passes.
//
//===----------------------------------------------------------------------===//

#ifndef LPN_ANALYSIS_TOKENFLOWANALYSIS_H
#define LPN_ANALYSIS_TOKENFLOWANALYSIS_H

#include "LPN/Analysis/AnalysisCommon.h"
#include "LPN/Analysis/ControlFlowTrace.h"
#include "LPN/Analysis/GuardTransitiveClosure.h"
#include "LPN/Dialect/LPNOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <vector>

namespace mlir {
namespace lpn {

class NetOp;

struct TokenGuard {
	Value key;
	llvm::hash_code keyHash = {};
	int64_t equalsValue = 0;
};

struct TargetInfo {
	StringAttr symbol;
	llvm::SmallVector<TokenGuard, 4> guards;
};

struct TokenEditSignature {
	llvm::hash_code keyHash = {};
	llvm::hash_code valueHash = {};
	llvm::SmallVector<unsigned, 4> sourceRefs;
};



struct ObservableSource {
	StringAttr place;
	Value takeValue;
};

// Key for emit metadata: (emit operation, path index, target place)
struct EmitPathKey {
	Operation* emit;
	unsigned pathIndex;
	StringAttr target;
};

struct EmitPathKeyInfo : llvm::DenseMapInfo<EmitPathKey> {
	static inline EmitPathKey getEmptyKey() {
		return {llvm::DenseMapInfo<Operation*>::getEmptyKey(), 0, StringAttr()};
	}
	static inline EmitPathKey getTombstoneKey() {
		return {llvm::DenseMapInfo<Operation*>::getTombstoneKey(), 0, StringAttr()};
	}
	static unsigned getHashValue(const EmitPathKey &key) {
		llvm::hash_code targetHash =
			(key.target != StringAttr()) ? llvm::hash_value(key.target)
			                               : llvm::hash_code(0);
		return llvm::hash_combine(
			llvm::DenseMapInfo<Operation*>::getHashValue(key.emit),
			key.pathIndex, targetHash);
	}
	static bool isEqual(const EmitPathKey &a, const EmitPathKey &b) {
		return a.emit == b.emit && a.pathIndex == b.pathIndex &&
		       a.target == b.target;
	}
};

// Per-emit metadata: transformation info without path resolution
struct EmitMetadata {
	TransitionOp transition;  // Which transition this emit is in
	EmitOp emit;              // The emit operation
	unsigned pathIndex;       // Index in guard_paths attribute
	StringAttr targetPlace;   // Target place for this emit path
	Value tokenValue;         // SSA value for token
	Value delayValue;         // SSA value for delay
	llvm::SmallVector<ControlContext, 4> contexts;  // Control flow path
	llvm::SmallVector<TokenEditSignature, 4> edits; // Token transformations
	TargetInfo target;        // Where emit goes
	llvm::SmallVector<Value, 4> guardTakes;         // Guard takes for this path
	bool hasGuardPath = false;
	llvm::hash_code tokenHash = {};
	llvm::hash_code delayHash = {};
};

struct ObservablePath {
	EmitPathKey terminalKey;
	llvm::SmallVector<GuardPathSegment, 4> prefixSegments;
	llvm::SmallVector<ObservableSource, 4> sources;
};

struct TokenFlowAnalysisResult {
	// Per-emit-path metadata: keyed by (emit, pathIndex)
	llvm::DenseMap<EmitPathKey, EmitMetadata, EmitPathKeyInfo> emitMetadata;
	// Map: observable place -> resolved guard paths
	llvm::DenseMap<StringAttr, llvm::SmallVector<ObservablePath, 4>,
	              llvm::DenseMapInfo<StringAttr>> observablePaths;
	unsigned totalHyperedges = 0;
	unsigned guardHyperedges = 0;
};

LogicalResult runTokenFlowAnalysis(NetOp net,
								   const ObservableSet &observables,
								   TokenFlowAnalysisResult &result);

} // namespace lpn
} // namespace mlir

#endif // LPN_ANALYSIS_TOKENFLOWANALYSIS_H
