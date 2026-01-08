add_mlir_library(MLIRLPNAnalysis
  GuardTransitiveClosure.cpp
  TokenFlowAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRLPN
  MLIRSCFDialect
  MLIRArithDialect
)