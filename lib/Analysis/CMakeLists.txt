add_mlir_library(MLIRLPNAnalysis
  ControlFlowTrace.cpp
  GuardTransitiveClosure.cpp
  TokenFlowAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRLPN
  MLIRSCFDialect
  MLIRArithDialect
)