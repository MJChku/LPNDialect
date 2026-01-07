add_mlir_library(MLIRLPNAnalysis
  TokenFlowAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRLPN
  MLIRSCFDialect
  MLIRArithDialect
)