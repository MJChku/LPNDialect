#ifndef LPN_DIALECT_TYPES_H
#define LPN_DIALECT_TYPES_H

#include "mlir/IR/Types.h"
#include <tuple>

namespace mlir {
namespace lpn {
namespace detail {

struct ArrayTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::mlir::Type>;

  ArrayTypeStorage(::mlir::Type elementType)
      : elementType(std::move(elementType)) {}

  KeyTy getAsKey() const { return KeyTy(elementType); }

  bool operator==(const KeyTy &tblgenKey) const {
    return elementType == std::get<0>(tblgenKey);
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ArrayTypeStorage *construct(::mlir::TypeStorageAllocator &allocator,
                                     KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ArrayTypeStorage>())
        ArrayTypeStorage(std::move(elementType));
  }

  ::mlir::Type elementType;
};

} // namespace detail
} // namespace lpn
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "LPN/Dialect/LPNTypes.h.inc"

#endif // LPN_DIALECT_TYPES_H
