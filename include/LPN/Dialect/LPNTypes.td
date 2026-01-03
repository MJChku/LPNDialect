#ifndef LPN_TYPES_TD
#define LPN_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "LPNDialect.td"

def TokenType : TypeDef<LPN_Dialect, "Token"> {
  let summary = "Opaque handle representing a token consumed/produced by schedules";

  let description = [{
    Tokens are abstract values that flow through schedule functions. They can
    only be materialized by `lpn.take` and consumed by `lpn.emit` or the other
    token-manipulation operations provided by this dialect.
  }];

  let mnemonic = "token";
}

def KeyType : TypeDef<LPN_Dialect, "Key"> {
  let summary = "Runtime handle representing a token metadata key";

  let description = [{
    Keys identify entries inside the append-only token log.  Literal keys are
    constructed via `lpn.key.literal` while fully dynamic handles can be
    materialized from runtime register identifiers with `lpn.key.reg`.  Keeping
    keys explicit in the IR makes downstream analyses resilient to dynamic
    metadata schemas.
  }];

  let mnemonic = "key";
}

def PlaceHandleType : TypeDef<LPN_Dialect, "Place"> {
  let summary = "Handle to a place";

  let description = [{
    Represents a runtime reference to a place. Obtain handles via
    `lpn.place_ref` or by indexing into place lists.
  }];

  let mnemonic = "place";
}

def ArrayType : TypeDef<LPN_Dialect, "Array"> {
  let summary = "Homogeneous SSA array";

  let description = [{
    Represents a compile-time array of SSA values that all share the same
    element type. Arrays can hold place handles, scalar values, or any other
    dialect-level type, and are primarily used to model runtime lookups such as
    switch egress tables.
  }];

  let parameters = (ins TypeParameter<"::mlir::Type",
                                     "Type stored by each array slot">:$elementType);
  let genStorageClass = 0;
  let storageClass = "ArrayTypeStorage";
  let storageNamespace = "::mlir::lpn::detail";

  let assemblyFormat = "`<` $elementType `>`";

  let mnemonic = "array";
}

#endif // LPN_TYPES_TD
