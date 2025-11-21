#ifndef LPN_ATTRIBUTES_TD
#define LPN_ATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "LPNDialect.td"

def TokenPropertySetAttr : AttrDef<LPN_Dialect, "TokenPropertySet"> {
  let summary = "schema of token properties (field names)";

  let description = [{
    Represents the logical "shape" of tokens stored in a place or
    represented by a token type. It is a list of field names such as
    ["addr", "size", "prio"].

    In the Python LPN library, this corresponds to:
      - Token.prop_list
      - Place.type_annotations
  }];

  let mnemonic = "token_props";

  let parameters = (ins ArrayAttr:$properties);
  let assemblyFormat = "`<` $properties `>`";
  // let genVerifyDecl = 1;

  let builders = [
    AttrBuilder<(ins "::llvm::ArrayRef<::llvm::StringRef>":$properties), [{
      SmallVector<Attribute> stringAttrs;
      stringAttrs.reserve(properties.size());
      for (auto name : properties)
        stringAttrs.push_back(::mlir::StringAttr::get($_ctxt, name));
      return $_get($_ctxt, ::mlir::ArrayAttr::get($_ctxt, stringAttrs));
    }]>
  ];
};

def TokenPropertySet : Attr<"::mlir::lpn::TokenPropertySetAttr">;

def IntSymbolAttr : AttrDef<LPN_Dialect, "IntSymbol"> {
  let summary = "integer with optional bounds and id (from IntSymbol)";

  let description = [{
    Models the IntSymbol class in the Python LPN library:

      IntSymbol(value, lower_bound=None, upper_bound=None, id=None)

    - `value`       : concrete integer value (or default).
    - `lower_bound` : optional lower bound, if present.
    - `upper_bound` : optional upper bound, if present.
    - `id`          : optional symbolic identifier (e.g. "N", "K").

    This attribute can be attached to operations that use symbolic
    integers, such as delays, guards, or other constraints.
  }];

  let mnemonic = "int_symbol";

  let parameters = (ins
    "int64_t":$value,
    OptionalParameter<"std::optional<int64_t>",
                      "closed lower bound of the interval">:$lower_bound,
    OptionalParameter<"std::optional<int64_t>",
                      "closed upper bound of the interval">:$upper_bound,
    OptionalParameter<"::mlir::StringAttr",
                      "symbolic identifier for the integer">:$id
  );

  let assemblyFormat = "`<` $value (`,` `lower` `=` $lower_bound^)? (`,` `upper` `=` $upper_bound^)? (`,` `id` `=` $id^)? `>`";
  // let genVerifyDecl = 1;
};

def IntSymbol : Attr<"::mlir::lpn::IntSymbolAttr">;


def PlaceMode : I32EnumAttr<"PlaceMode", "place simulation mode", [
    I32EnumAttrCase<"Normal",   0>,
    I32EnumAttrCase<"Symbolic", 1>
  ]> {
  let cppNamespace = "::mlir::lpn";
  let genSpecializedAttr = 0;
}

#endif // LPN_ATTRIBUTES_TD
