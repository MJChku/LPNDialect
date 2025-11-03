#ifndef LPNTYPES_H
#define LPNTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "LPN/Dialect/LPNTypes.h.inc"

#endif // LPNTYPES_H