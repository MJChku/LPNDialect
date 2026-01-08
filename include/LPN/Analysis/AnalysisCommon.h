//===- AnalysisCommon.h - Shared analysis utilities -------------*- C++ -*-===//
//
// Common types shared by LPN analysis passes.
//
//===----------------------------------------------------------------------===//

#ifndef LPN_ANALYSIS_ANALYSISCOMMON_H
#define LPN_ANALYSIS_ANALYSISCOMMON_H

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::lpn {

using ObservableSet =
    llvm::DenseSet<StringAttr, llvm::DenseMapInfo<StringAttr>>;

} // namespace mlir::lpn

#endif // LPN_ANALYSIS_ANALYSISCOMMON_H
