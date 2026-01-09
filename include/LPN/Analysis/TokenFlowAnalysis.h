//===- TokenFlowAnalysis.h - Token flow analysis for LPN ---------*- C++ -*-===//
//
// Emit-centric token flow analysis shared by retain passes.
//
//===----------------------------------------------------------------------===//

#ifndef LPN_ANALYSIS_TOKENFLOWANALYSIS_H
#define LPN_ANALYSIS_TOKENFLOWANALYSIS_H

#include "LPN/Analysis/AnalysisCommon.h"
#include "LPN/Analysis/ControlFlowTrace.h"
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

struct EdgeTemplate {
	StringAttr driver;
	llvm::SmallVector<ObservableSource, 4> sources;
	TargetInfo target;
	Value tokenValue;
	Value delayValue;
	llvm::SmallVector<ControlContext, 4> contexts;
	llvm::SmallVector<TokenEditSignature, 4> editSummary;
	llvm::hash_code tokenHash = {};
	llvm::hash_code delayHash = {};
};

using EdgePath = llvm::SmallVector<const EdgeTemplate *>;

struct TokenFlowAnalysisResult {
	std::vector<std::unique_ptr<EdgeTemplate>> templates;
	llvm::DenseMap<StringAttr, llvm::SmallVector<const EdgeTemplate *>> adjacency;
	llvm::DenseMap<StringAttr, llvm::SmallVector<EdgePath>> observablePaths;
	uint64_t totalHyperedges = 0;
	uint64_t guardHyperedges = 0;
	uint64_t clusteredHyperedges = 0;
	uint64_t rawPaths = 0;
	uint64_t retainedPaths = 0;
};

LogicalResult runTokenFlowAnalysis(NetOp net,
								   const ObservableSet &observables,
								   TokenFlowAnalysisResult &result);

} // namespace lpn
} // namespace mlir

#endif // LPN_ANALYSIS_TOKENFLOWANALYSIS_H
