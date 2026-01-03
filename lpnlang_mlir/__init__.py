from __future__ import annotations

import ast
import hashlib
import inspect
import operator
import re
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union


_AST_OP_MAP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
    ast.MatMult: operator.matmul,
}


_FNV_OFFSET_BASIS = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def _to_signed_i64(value: int) -> int:
  mask = (1 << 64) - 1
  value &= mask
  if value >= (1 << 63):
    value -= (1 << 64)
  return value


_FNV_OFFSET_BASIS_I64 = _to_signed_i64(_FNV_OFFSET_BASIS)


def _hash_literal_segment_to_i64(text: str) -> int:
  if not text:
    return 0
  digest = hashlib.sha256(text.encode("utf-8")).digest()
  raw = int.from_bytes(digest[:8], byteorder="little", signed=False)
  return _to_signed_i64(raw)


class Statement:
  def render(self, indent: str) -> str:
    raise NotImplementedError


class RawStatement(Statement):
  def __init__(self, text: str):
    self.text = text

  def render(self, indent: str) -> str:
    return f"{indent}{self.text}"


class IfStatement(Statement):
  def __init__(self,
               cond: str,
               then_stmts: List[Statement],
               else_stmts: Optional[List[Statement]] = None):
    self.cond = cond
    self.then_stmts = then_stmts
    self.else_stmts = else_stmts

  def _render_block(self, stmts: List[Statement], indent: str) -> List[str]:
    rendered = []
    inner = indent + "  "
    for stmt in stmts:
      rendered.append(stmt.render(inner))
    rendered.append(f"{inner}scf.yield")
    return rendered

  def render(self, indent: str) -> str:
    lines = [f"{indent}scf.if {self.cond} {{"]
    lines.extend(self._render_block(self.then_stmts, indent))
    lines.append(f"{indent}}}")
    if self.else_stmts is not None:
      lines.append(f"{indent}else {{")
      lines.extend(self._render_block(self.else_stmts, indent))
      lines.append(f"{indent}}}")
    return "\n".join(lines)


class ForStatement(Statement):
  def __init__(self, iv: str, lower: str, upper: str, step: str,
               body: List[Statement]):
    self.iv = iv
    self.lower = lower
    self.upper = upper
    self.step = step
    self.body = body

  def render(self, indent: str) -> str:
    lines = [f"{indent}scf.for {self.iv} = {self.lower} to {self.upper} step {self.step} {{"]
    inner = indent + "  "
    for stmt in self.body:
      lines.append(stmt.render(inner))
    lines.append(f"{inner}scf.yield")
    lines.append(f"{indent}}}")
    return "\n".join(lines)


class TransitionScriptExecutor:
  """Very small AST executor that rewrites Python if-statements to scf.if."""

  def __init__(self,
               fn: Callable[..., None],
               *,
               require_builder_arg: bool = True):
    self.fn = fn
    self.require_builder_arg = require_builder_arg
    self.filename = inspect.getsourcefile(fn) or fn.__code__.co_filename
    if not self.filename:
      raise ValueError("unable to locate source file for transition script")
    try:
      with open(self.filename, "r", encoding="utf-8") as f:
        file_source = f.read()
    except OSError as exc:
      raise ValueError("unable to read transition source") from exc

    module = ast.parse(file_source, filename=self.filename)
    first_line = fn.__code__.co_firstlineno

    def matches_function(node: ast.FunctionDef) -> bool:
      deco_lines = len(getattr(node, "decorator_list", []))
      start_line = node.lineno - deco_lines if deco_lines else node.lineno
      return start_line <= first_line <= node.lineno

    func_defs = [
        node for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__
        and matches_function(node)
    ]
    if not func_defs:
      func_defs = [
          node for node in ast.walk(module)
          if isinstance(node, ast.FunctionDef) and node.name == fn.__name__
      ]
    if not func_defs:
      raise ValueError("transition script could not find the function body")
    self.func_def = func_defs[0]
    self.signature = inspect.signature(fn)

  def __call__(self, builder: "TransitionBuilder", *args: Any,
               **kwargs: Any) -> None:
    if self.require_builder_arg:
      try:
        bound = self.signature.bind_partial(builder, *args, **kwargs)
      except TypeError as exc:
        raise TypeError(
            f"transition script '{self.fn.__name__}' must accept a builder argument"
        ) from exc
    else:
      params = list(self.signature.parameters.values())
      if params:
        first = params[0]
        if (first.kind in (inspect.Parameter.POSITIONAL_ONLY,
                           inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and first.default is inspect._empty):
          raise TypeError(
              f"jit transition '{self.fn.__name__}' should not declare an explicit builder argument"
          )
      bound = self.signature.bind_partial(*args, **kwargs)

    bound.apply_defaults()
    self.builder = builder
    self.locals: Dict[str, Any] = dict(bound.arguments)
    self.locals.setdefault("builder", builder)
    closure_vars = inspect.getclosurevars(self.fn)
    self.locals.update(closure_vars.nonlocals)
    for attr in dir(builder):
      if attr.startswith("_"):
        continue
      value = getattr(builder, attr)
      if callable(value) and attr not in self.locals:
        self.locals[attr] = value
    self.globals: Dict[str, Any] = dict(self.fn.__globals__)
    self.globals.setdefault("__builtins__", __builtins__)
    self._exec_block(self.func_def.body)

  def _exec_block(self, statements: List[ast.stmt]) -> None:
    for stmt in statements:
      self._exec_stmt(stmt)

  def _exec_stmt(self, stmt: ast.stmt) -> None:
    if isinstance(stmt, ast.If):
      self._exec_if(stmt)
      return
    if isinstance(stmt, ast.For) and self._is_range_loop(stmt):
      self._exec_range_loop(stmt)
      return
    if isinstance(stmt, ast.Assign) and self._try_exec_assign(stmt):
      return
    if isinstance(stmt, ast.AugAssign) and self._try_exec_aug_assign(stmt):
      return

    mod = ast.Module(body=[stmt], type_ignores=[])
    ast.fix_missing_locations(mod)
    code = compile(mod, self.filename, "exec")
    exec(code, self.globals, self.locals)

  def _try_exec_assign(self, stmt: ast.Assign) -> bool:
    if len(stmt.targets) != 1:
      return False
    target = stmt.targets[0]
    if not isinstance(target, ast.Subscript):
      return False
    if not isinstance(target.value, ast.Name):
      return False

    var_name = target.value.id
    if var_name not in self.locals:
      return False

    obj = self.locals[var_name]
    if not isinstance(obj, Value):
      return False

    idx_node = target.slice
    if sys.version_info < (3, 9) and isinstance(idx_node, ast.Index):
      idx_node = idx_node.value

    try:
      index = self._eval_expr(idx_node)
      value = self._eval_expr(stmt.value)
    except Exception:
      return False

    new_obj = obj.set(index, value)
    self.locals[var_name] = new_obj
    return True

  def _try_exec_aug_assign(self, stmt: ast.AugAssign) -> bool:
    target = stmt.target
    if not isinstance(target, ast.Subscript):
      return False
    if not isinstance(target.value, ast.Name):
      return False

    var_name = target.value.id
    if var_name not in self.locals:
      return False

    obj = self.locals[var_name]
    if not isinstance(obj, Value):
      return False

    idx_node = target.slice
    if sys.version_info < (3, 9) and isinstance(idx_node, ast.Index):
      idx_node = idx_node.value

    op_type = type(stmt.op)
    op_func = _AST_OP_MAP.get(op_type)
    if not op_func:
      return False

    try:
      index = self._eval_expr(idx_node)
      operand = self._eval_expr(stmt.value)
    except Exception:
      return False

    current_val = obj[index]
    new_val = op_func(current_val, operand)
    new_obj = obj.set(index, new_val)
    self.locals[var_name] = new_obj
    return True

  def _eval_expr(self, expr: ast.expr) -> Any:
    node = ast.Expression(expr)
    ast.fix_missing_locations(node)
    code = compile(node, self.filename, "eval")
    return eval(code, self.globals, self.locals)

  def _exec_branch(self, statements: List[ast.stmt]) -> None:
    saved_locals = self.locals
    branch_locals = dict(saved_locals)
    self.locals = branch_locals
    try:
      self._exec_block(statements)
    finally:
      self.locals = saved_locals

  def _exec_if(self, node: ast.If) -> None:
    cond_value = self._eval_expr(node.test)
    if not isinstance(cond_value, Value):
      raise TypeError("transition conditions must evaluate to LPN SSA values")

    def then_fn(_builder: "TransitionBuilder") -> None:
      self._exec_branch(node.body)

    def else_fn(_builder: "TransitionBuilder") -> None:
      self._exec_branch(node.orelse)

    false_fn = else_fn if node.orelse else None
    self.builder.if_op(cond_value, then_fn, false_fn)

  def _is_range_loop(self, node: ast.For) -> bool:
    call = node.iter
    return isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "range"

  def _exec_range_loop(self, node: ast.For) -> None:
    if node.orelse:
      raise TypeError("lpn jit for-loops do not support 'else' blocks")
    if not isinstance(node.target, ast.Name):
      raise TypeError("for-loop target must be a simple variable name")
    call = node.iter
    if call.keywords:
      raise TypeError("range(...) with keyword arguments is not supported")
    arg_count = len(call.args)
    if arg_count == 1:
      lower = 0
      upper = self._eval_expr(call.args[0])
      step = 1
    elif arg_count == 2:
      lower = self._eval_expr(call.args[0])
      upper = self._eval_expr(call.args[1])
      step = 1
    elif arg_count == 3:
      lower = self._eval_expr(call.args[0])
      upper = self._eval_expr(call.args[1])
      step = self._eval_expr(call.args[2])
      if isinstance(step, int) and step == 0:
        raise ValueError("range() step cannot be zero in jit loops")
    else:
      raise TypeError("range() expects 1-3 positional arguments in jit loops")

    def loop_body(_builder: "TransitionBuilder", iv_value: Value) -> None:
      saved_locals = self.locals
      loop_locals = dict(saved_locals)
      loop_locals[node.target.id] = iv_value
      self.locals = loop_locals
      try:
        self._exec_block(node.body)
      finally:
        self.locals = saved_locals

    self.builder.for_range(lower, upper, step=step, body=loop_body)


@dataclass(frozen=True)
class Value:
  builder: "TransitionBuilder"
  name: str
  typ: str

  def __str__(self) -> str:
    return self.name

  def __bool__(self) -> bool:
    raise TypeError("LPN SSA values cannot be used as Python booleans")

  def _require_numeric(self) -> None:
    if self.typ not in ("i64", "f64"):
      raise TypeError(f"operation not supported on values of type {self.typ}")

  def _require_integer_like(self) -> None:
    if self.typ not in ("i64", "index"):
      raise TypeError("comparison is only defined on integer/index values")

  def _cmp_int(self, predicate: str, other: Union["Value", int]) -> "Value":
    self._require_integer_like()
    return self.builder.cmpi(predicate, self, other, typ=self.typ)

  def __add__(self, other: Union["Value", int, float]) -> "Value":
    self._require_numeric()
    if self.typ == "f64":
      return self.builder.addf(self, other)
    return self.builder.addi(self, other, typ=self.typ)

  def __radd__(self, other: Union["Value", int, float]) -> "Value":
    return self.__add__(other)

  def __sub__(self, other: Union["Value", int, float]) -> "Value":
    self._require_numeric()
    if self.typ == "f64":
      return self.builder.subf(self, other)
    return self.builder.subi(self, other, typ=self.typ)

  def __rsub__(self, other: Union["Value", int, float]) -> "Value":
    self._require_numeric()
    if self.typ == "f64":
      return self.builder.subf(other, self)
    return self.builder.subi(other, self, typ=self.typ)

  def __truediv__(self, other: Union["Value", int, float]) -> "Value":
    if self.typ != "f64":
      raise TypeError("division only supported for f64 values")
    return self.builder.divf(self, other)

  def __eq__(self, other: object) -> "Value":  # type: ignore[override]
    if isinstance(other, (Value, int)):
      return self._cmp_int("eq", other)
    return NotImplemented

  def __ne__(self, other: object) -> "Value":  # type: ignore[override]
    if isinstance(other, (Value, int)):
      return self._cmp_int("ne", other)
    return NotImplemented

  def __lt__(self, other: Union["Value", int]) -> "Value":
    return self._cmp_int("slt", other)

  def __le__(self, other: Union["Value", int]) -> "Value":
    return self._cmp_int("sle", other)

  def __gt__(self, other: Union["Value", int]) -> "Value":
    return self._cmp_int("sgt", other)

  def __ge__(self, other: Union["Value", int]) -> "Value":
    return self._cmp_int("sge", other)

  def __getitem__(self, index: Union["Value", int]) -> "Value":
    return self.builder.array_get(self, index)

  def __setitem__(self, index: Union["Value", int], value: Union["Value", int, float]) -> None:
    raise TypeError("LPN arrays are immutable SSA values; use 'new_arr = arr.set(index, value)' instead of 'arr[index] = value'.")

  def set(self, index: Union["Value", int], value: Union["Value", int, float]) -> "Value":
    return self.builder.array_set(self, index, value)

  def len(self) -> "Value":
    return self.builder.array_len(self)

  def __format__(self, format_spec: str) -> str:
    if format_spec not in ("", "s"):
      raise ValueError(
          f"LPN SSA values only support empty format specifiers, got '{format_spec}'"
      )
    return self.builder._register_format_placeholder(self)

  def eq(self, other: Union["Value", int, float]) -> "Value":
    return self.__eq__(other)

  def ne(self, other: Union["Value", int, float]) -> "Value":
    return self.__ne__(other)


@dataclass(frozen=True)
class TokenValue:
  builder: "TransitionBuilder"
  name: str

  def __str__(self) -> str:
    return self.name

  def get(self, key: Union[str, "KeyValue", Value, int]) -> Value:
    return self.builder.token_get(self, key)

  def set(self,
          key: Union[str, "KeyValue", Value, int],
          value: Union[Value, int]) -> "TokenValue":
    return self.builder.token_set(self, key, value)

  def clone(self) -> "TokenValue":
    return self.builder.token_clone(self)


@dataclass(frozen=True)
class KeyValue:
  builder: "TransitionBuilder"
  name: str

  def __str__(self) -> str:
    return self.name

  def as_value(self) -> Value:
    return self.builder._ssa_values[self.name]


@dataclass(frozen=True)
class PlaceHandle:
  name: str


class TransitionBuilder:
  _global_id = 0
  _value_name_re = re.compile(r"%[A-Za-z0-9_$.]+")

  def __init__(self, name: str):
    self.name = name
    self._ops: List[Statement] = []
    self._value_id = 0
    self._place_handles: dict[str, Value] = {}
    self._literal_keys: dict[str, KeyValue] = {}
    self._ssa_values: dict[str, Value] = {}
    self._local_prefix = TransitionBuilder._global_id
    TransitionBuilder._global_id += 1
    self._literal_segment_hashes: Dict[str, int] = {}
    self._literal_segment_values: Dict[str, Value] = {}
    self._hash_offset_value: Optional[Value] = None
    self._hash_prime_value: Optional[Value] = None

  def _next_value(self) -> str:
    name = f"%t{self._local_prefix}_{self._value_id}"
    self._value_id += 1
    return name

  def _register_format_placeholder(self, value: Value) -> str:
    return value.name

  def _append(self, text: str) -> None:
    self._ops.append(RawStatement(text))

  def _wrap_value(self, name: str, typ: str) -> Value:
    value = Value(self, name, typ)
    self._ssa_values[name] = value
    return value

  def _wrap_token(self, name: str) -> TokenValue:
    return TokenValue(self, name)

  def _wrap_key(self, name: str) -> KeyValue:
    self._wrap_value(name, "!lpn.key")
    return KeyValue(self, name)

  def _coerce_value(self, value: Union[Value, int, float], typ: str) -> Value:
    if isinstance(value, Value):
      if value.typ == typ:
        return value
      if value.typ == "index" and typ == "i64":
        return self.index_cast(value, src_type="index", dst_type="i64")
      if value.typ == "i64" and typ == "index":
        return self.index_cast(value, src_type="i64", dst_type="index")
      raise TypeError(f"expected value of type {typ}, but got {value.typ}")
    if typ == "i64":
      return self.const_i64(int(value))
    if typ == "f64":
      return self.const_f64(float(value))
    if typ == "index":
      if isinstance(value, int):
        return self.const_index(value)
    raise TypeError(f"cannot coerce value of type {type(value)} to {typ}")

  def _as_value(self, value: Union[Value, PlaceHandle, int, float]) -> Value:
    if isinstance(value, Value):
      return value
    if isinstance(value, KeyValue):
      return value.as_value()
    if isinstance(value, PlaceHandle):
      return self._ensure_place_handle(value)
    if isinstance(value, float):
      return self.const_f64(value)
    if isinstance(value, int):
      return self.const_i64(value)
    raise TypeError(f"unsupported element type {type(value)}")

  def _resolve_place_operand(self,
                             place: Union[PlaceHandle, Value]) -> Value:
    if isinstance(place, PlaceHandle):
      return self._ensure_place_handle(place)
    if isinstance(place, Value) and place.typ == "!lpn.place":
      return place
    raise TypeError("expected a place handle or SSA place reference")

  def _ensure_place_handle(self, place: PlaceHandle) -> Value:
    handle = self._place_handles.get(place.name)
    if handle is None:
      name = self._next_value()
      self._ops.append(
          RawStatement(f"{name} = lpn.place_ref @{place.name} : !lpn.place"))
      handle = self._wrap_value(name, "!lpn.place")
      self._place_handles[place.name] = handle
    return handle

  def _array_element_type(self, array_type: str) -> str:
    prefix = "!lpn.array<"
    if array_type.startswith(prefix) and array_type.endswith(">"):
      return array_type[len(prefix):-1]
    raise TypeError("expected an !lpn.array value")

  def _coerce_array_element(self,
                            element: Union[Value, KeyValue, PlaceHandle, int, float],
                            expected_type: Optional[str] = None) -> Value:
    if isinstance(element, (Value, KeyValue, PlaceHandle)):
      value = self._as_value(element)
    elif isinstance(element, float):
      target = expected_type if expected_type == "f64" else "f64"
      value = self._coerce_value(element, target)
    elif isinstance(element, int):
      target = expected_type if expected_type in ("i64", "index") else "i64"
      if target == "index":
        value = self.const_index(element)
      else:
        value = self.const_i64(element)
    else:
      value = self._as_value(element)
    if expected_type is not None and value.typ != expected_type:
      raise TypeError(
          f"array elements must all have type {expected_type}, saw {value.typ}")
    return value

  def array(self,
            *elements: Union[Value, PlaceHandle, int, float,
                             Sequence[Union[Value, PlaceHandle, int, float]]]
            ) -> Value:
    if len(elements) == 1 and isinstance(elements[0], (list, tuple)):
      elements = tuple(elements[0])
    if not elements:
      raise ValueError("array requires at least one element")
    coerced: List[Value] = []
    element_type: Optional[str] = None
    for element in elements:
      value = self._coerce_array_element(element, element_type)
      if element_type is None:
        element_type = value.typ
      coerced.append(value)
    assert element_type is not None
    name = self._next_value()
    operand_names = ", ".join(value.name for value in coerced)
    array_type = f"!lpn.array<{element_type}>"
    operand_types = ", ".join(value.typ for value in coerced)
    self._append(f"{name} = lpn.array {operand_names} : {operand_types} -> {array_type}")
    return self._wrap_value(name, array_type)

  def array_get(self,
                array_value: Value,
                index: Union[Value, int]) -> Value:
    if (not isinstance(array_value, Value)
        or not array_value.typ.startswith("!lpn.array<")):
      raise TypeError("array_get expects an !lpn.array value")
    idx = self._coerce_value(index, "index")
    element_type = self._array_element_type(array_value.typ)
    name = self._next_value()
    self._append(
        f"{name} = lpn.array.get {array_value.name}, {idx.name} : ({array_value.typ}, index) -> {element_type}")
    return self._wrap_value(name, element_type)

  def key_literal(self, name: str) -> KeyValue:
    existing = self._literal_keys.get(name)
    if existing is not None:
      return existing
    value = self._next_value()
    self._append(
        f"{value} = lpn.key.literal \"{name}\" : !lpn.key")
    key = self._wrap_key(value)
    self._literal_keys[name] = key
    return key

  def key_reg(self, identifier: Union[Value, int]) -> KeyValue:
    key_id = identifier
    if not isinstance(key_id, Value) or key_id.typ != "i64":
      key_id = self._coerce_value(key_id, "i64")
    name = self._next_value()
    self._append(
        f"{name} = lpn.key.reg {key_id.name} : i64 -> !lpn.key")
    return self._wrap_key(name)

  def _hash_prime_const(self) -> Value:
    if self._hash_prime_value is None:
      self._hash_prime_value = self.const_i64(_FNV_PRIME)
    return self._hash_prime_value

  def _hash_offset_const(self) -> Value:
    if self._hash_offset_value is None:
      self._hash_offset_value = self.const_i64(_FNV_OFFSET_BASIS_I64)
    return self._hash_offset_value

  def _get_literal_segment_value(self, literal: str) -> Optional[Value]:
    if not literal:
      return None
    existing = self._literal_segment_values.get(literal)
    if existing is not None:
      return existing
    hashed = self._literal_segment_hashes.get(literal)
    if hashed is None:
      hashed = _hash_literal_segment_to_i64(literal)
      self._literal_segment_hashes[literal] = hashed
    value = self.const_i64(hashed)
    self._literal_segment_values[literal] = value
    return value

  def _mix_hash_chunk(self,
                      seed: Value,
                      chunk: Value) -> Value:
    mixed = self.xori(seed, chunk)
    return self.muli(mixed, self._hash_prime_const())

  def _materialize_dynamic_key(self,
                               segments: Sequence[Union[str, Value, int]]
                               ) -> KeyValue:
    current = self._hash_offset_const()
    for segment in segments:
      if isinstance(segment, str):
        literal_value = self._get_literal_segment_value(segment)
        if literal_value is None:
          continue
        current = self._mix_hash_chunk(current, literal_value)
        continue
      segment_value = self._coerce_value(segment, "i64")
      current = self._mix_hash_chunk(current, segment_value)
    return self.key_reg(current)

  def _ensure_key(self,
                  key: Union[str, KeyValue, Value, int]) -> KeyValue:
    if isinstance(key, KeyValue):
      return key
    if isinstance(key, Value):
      if key.typ == "!lpn.key":
        return KeyValue(self, key.name)
      return self.key_reg(key)
    if isinstance(key, int):
      return self.key_reg(key)
    if isinstance(key, str):
      dynamic_segments = self._match_dynamic_key_literal(key)
      if dynamic_segments is not None:
        return self._materialize_dynamic_key(dynamic_segments)
      return self.key_literal(key)
    raise TypeError("expected a key literal string, KeyValue, Value, or int")

  def _match_dynamic_key_literal(
      self, literal: str) -> Optional[List[Union[str, Value]]]:
    """Detect literal strings that embed SSA value names via f-strings."""
    matches = list(TransitionBuilder._value_name_re.finditer(literal))
    if not matches:
      return None
    segments: List[Union[str, Value]] = []
    cursor = 0
    for match in matches:
      start, end = match.span()
      if start > cursor:
        prefix = literal[cursor:start]
        if prefix:
          segments.append(prefix)
      value_name = match.group(0)
      value = self._ssa_values.get(value_name)
      if value is None:
        return None
      segments.append(value)
      cursor = end
    if cursor < len(literal):
      suffix = literal[cursor:]
      if suffix:
        segments.append(suffix)
    if not any(isinstance(segment, Value) for segment in segments):
      return None
    return segments

  def const_f64(self, value: float) -> Value:
    literal = f"{float(value):.6f}"
    name = self._next_value()
    self._append(f"{name} = arith.constant {literal} : f64")
    return self._wrap_value(name, "f64")

  def f64(self, value: float) -> Value:
    return self.const_f64(value)

  def const_i64(self, value: int) -> Value:
    name = self._next_value()
    self._append(f"{name} = arith.constant {int(value)} : i64")
    return self._wrap_value(name, "i64")

  def i64(self, value: int) -> Value:
    return self.const_i64(value)

  def const_index(self, value: int) -> Value:
    name = self._next_value()
    self._append(f"{name} = arith.constant {int(value)} : index")
    return self._wrap_value(name, "index")

  def index(self, value: int) -> Value:
    return self.const_index(value)

  def addi(self,
           lhs: Union[Value, int],
           rhs: Union[Value, int],
           *,
           typ: str = "i64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.addi {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

  def subi(self,
           lhs: Union[Value, int],
           rhs: Union[Value, int],
           *,
           typ: str = "i64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.subi {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

  def muli(self,
           lhs: Union[Value, int],
           rhs: Union[Value, int],
           *,
           typ: str = "i64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.muli {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

  def xori(self,
           lhs: Union[Value, int],
           rhs: Union[Value, int],
           *,
           typ: str = "i64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.xori {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

  def addf(self,
           lhs: Union[Value, float],
           rhs: Union[Value, float],
           *,
           typ: str = "f64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.addf {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

  def subf(self,
           lhs: Union[Value, float],
           rhs: Union[Value, float],
           *,
           typ: str = "f64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.subf {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

  def divf(self,
           lhs: Union[Value, float],
           rhs: Union[Value, float],
           *,
           typ: str = "f64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.divf {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

  def sitofp(self,
             value: Union[Value, int],
             *,
             src_type: str = "i64",
             dst_type: str = "f64") -> Value:
    src_val = self._coerce_value(value, src_type)
    name = self._next_value()
    self._append(f"{name} = arith.sitofp {src_val} : {src_type} to {dst_type}")
    return self._wrap_value(name, dst_type)

  def index_cast(self,
                 value: Union[Value, int],
                 *,
                 src_type: str,
                 dst_type: str = "index") -> Value:
    src_val = self._coerce_value(value, src_type)
    name = self._next_value()
    self._append(
        f"{name} = arith.index_cast {src_val} : {src_type} to {dst_type}")
    return self._wrap_value(name, dst_type)

  def cmpi(self,
           predicate: str,
           lhs: Union[Value, int],
           rhs: Union[Value, int],
           *,
           typ: str = "i64") -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    self._append(f"{name} = arith.cmpi {predicate}, {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, "i1")

  def select(self,
             cond: Value,
             true_value: Union[Value, int, float],
             false_value: Union[Value, int, float],
             *,
             typ: Optional[str] = None) -> Value:
    if not isinstance(cond, Value) or cond.typ != "i1":
      raise TypeError("select condition must be an i1 Value")
    value_type = typ
    if value_type is None:
      if isinstance(true_value, Value):
        value_type = true_value.typ
      elif isinstance(false_value, Value):
        value_type = false_value.typ
      else:
        value_type = "i64"
    true_val = self._coerce_value(true_value, value_type)
    false_val = self._coerce_value(false_value, value_type)
    name = self._next_value()
    self._append(
        f"{name} = arith.select {cond.name}, {true_val.name}, {false_val.name} : {value_type}")
    return self._wrap_value(name, value_type)

  def _capture_ops(
      self,
      fn: Optional[Callable[..., None]],
      *fn_args: Any
  ) -> List[Statement]:
    if fn is None:
      return []
    saved_ops = self._ops
    branch_ops: List[Statement] = []
    self._ops = branch_ops
    try:
      fn(self, *fn_args)
    finally:
      self._ops = saved_ops
    return branch_ops

  def if_op(
      self,
      cond: Union[Value, str],
      true_fn: Callable[['TransitionBuilder'], None],
      false_fn: Optional[Callable[['TransitionBuilder'], None]] = None) -> None:
    cond_name = cond.name if isinstance(cond, Value) else cond
    true_ops = self._capture_ops(true_fn)
    false_ops = self._capture_ops(false_fn) if false_fn else None
    self._ops.append(IfStatement(cond_name, true_ops, false_ops))

  def _ensure_delay(self, delay: Optional[Union[Value, float, int]]) -> Value:
    if delay is None:
      return self.const_f64(0.0)
    if isinstance(delay, Value):
      if delay.typ != "f64":
        raise TypeError("delay must be an f64 value")
      return delay
    return self.const_f64(float(delay))

  def divi(self,
           lhs: Union[Value, int],
           rhs: Union[Value, int],
           *,
           typ: str = "i64",
           signed: bool = True) -> Value:
    lhs_val = self._coerce_value(lhs, typ)
    rhs_val = self._coerce_value(rhs, typ)
    name = self._next_value()
    op = "arith.divsi" if signed else "arith.divui"
    self._append(f"{name} = {op} {lhs_val}, {rhs_val} : {typ}")
    return self._wrap_value(name, typ)

  def take(self, place: Union[PlaceHandle, Value]) -> TokenValue:
    handle = self._resolve_place_operand(place)
    name = self._next_value()
    self._append(
        f"{name} = lpn.take {handle} : !lpn.place -> !lpn.token")
    return self._wrap_token(name)

  def take_handle(self, handle: Value) -> TokenValue:
    if handle.typ != "!lpn.place":
      raise TypeError("take_handle expects a !lpn.place value")
    name = self._next_value()
    self._append(
        f"{name} = lpn.take {handle} : !lpn.place -> !lpn.token")
    return self._wrap_token(name)

  def emit(self,
           place: Union[PlaceHandle, Value],
           token: TokenValue,
           delay: Optional[Union[Value, float, int]] = None) -> None:
    handle = self._resolve_place_operand(place)
    delay_value = self._ensure_delay(delay)
    self._append(
        f"lpn.emit {handle}, {token.name}, {delay_value} : !lpn.place, !lpn.token, f64")

  def emit_handle(self,
                  handle: Value,
                  token: TokenValue,
                  delay: Optional[Union[Value, float, int]] = None) -> None:
    if handle.typ != "!lpn.place":
      raise TypeError("emit_handle expects a !lpn.place SSA value")
    delay_value = self._ensure_delay(delay)
    self._append(
        f"lpn.emit {handle}, {token.name}, {delay_value} : !lpn.place, !lpn.token, f64")

  def count(self, place: Union[PlaceHandle, Value]) -> Value:
    handle = self._resolve_place_operand(place)
    name = self._next_value()
    self._append(f"{name} = lpn.count {handle} : !lpn.place -> i64")
    return self._wrap_value(name, "i64")

  def token_get(self,
                token: TokenValue,
                key: Union[str, KeyValue, Value, int]
                ) -> Value:
    key_value = self._ensure_key(key)
    name = self._next_value()
    self._append(
        f"{name} = lpn.token.get {token.name}, {key_value.name} : !lpn.token, !lpn.key -> i64")
    return self._wrap_value(name, "i64")

  def token_set(self,
                token: TokenValue,
                key: Union[str, KeyValue, Value, int],
                value_ssa: Union[Value, int]) -> TokenValue:
    key_value = self._ensure_key(key)
    value = self._coerce_value(value_ssa, "i64")
    name = self._next_value()
    self._append(
        f"{name} = lpn.token.set {token.name}, {key_value.name}, {value.name} : !lpn.token, !lpn.key, i64 -> !lpn.token")
    return self._wrap_token(name)

  def token_clone(self, token: TokenValue) -> TokenValue:
    name = self._next_value()
    self._append(
        f"{name} = \"lpn.token.clone\"({token.name}) : (!lpn.token) -> !lpn.token")
    return self._wrap_token(name)

  def token_create(self, properties: Optional[Dict[str, int]] = None) -> TokenValue:
    props_dict = properties or {}
    props = ", ".join(
        f"{key} = {value} : i64" for key, value in sorted(props_dict.items()))
    attr = f"{{{props}}}" if props else "{}"
    name = self._next_value()
    self._append(
        f"{name} = \"lpn.token.create\"() {{log_prefix = {attr}}} : () -> !lpn.token")
    return self._wrap_token(name)

  def reg(self, key: Union[str, KeyValue, Value, int]) -> KeyValue:
    """Creates a key handle from literal strings, register ids, or dynamic expressions."""
    return self._ensure_key(key)

  def array_set(self,
                array_value: Value,
                index: Union[Value, int],
                value: Union[Value, int, float]) -> Value:
    if (not isinstance(array_value, Value)
        or not array_value.typ.startswith("!lpn.array<")):
      raise TypeError("array_set expects an !lpn.array value")
    idx = self._coerce_value(index, "index")
    element_type = self._array_element_type(array_value.typ)
    val = self._coerce_array_element(value, element_type)
    name = self._next_value()
    self._append(
        f"{name} = lpn.array.set {array_value.name}, {idx.name}, {val.name} : ({array_value.typ}, index, {element_type}) -> {array_value.typ}")
    return self._wrap_value(name, array_value.typ)

  def array_len(self, array_value: Value) -> Value:
    if (not isinstance(array_value, Value)
        or not array_value.typ.startswith("!lpn.array<")):
      raise TypeError("array_len expects an !lpn.array value")
    name = self._next_value()
    self._append(
        f"{name} = lpn.array.len {array_value.name} : {array_value.typ} -> index")
    return self._wrap_value(name, "index")

  def materialize_place(self, place: PlaceHandle) -> None:
    self._ensure_place_handle(place)

  def for_range(self,
                lower: Union[Value, int],
                upper: Union[Value, int],
                *,
                step: Union[Value, int] = 1,
                body: Callable[['TransitionBuilder', Value], None]) -> None:
    lb = self._coerce_value(lower, "index")
    ub = self._coerce_value(upper, "index")
    st = self._coerce_value(step, "index")
    iv_name = self._next_value()
    iv_value = self._wrap_value(iv_name, "index")
    body_ops = self._capture_ops(body, iv_value)
    self._ops.append(
        ForStatement(iv_name, lb.name, ub.name, st.name, body_ops))

  def render(self) -> List[str]:
    ops = []
    if self._ops:
      for op in self._ops:
        ops.append(op.render("        "))
    ops.append("        lpn.schedule.return")
    return ops


class NetBuilder:
  """Tiny DSL for emitting the new MLIR dialect."""

  def __init__(self, name: str = "net"):
    self.name = name
    self._places: List[tuple[PlaceHandle, Optional[int], Optional[int], bool]] = []
    self._transitions: List[TransitionBuilder] = []

  def place(self,
            name: str,
            *,
            capacity: Optional[int] = None,
            initial_tokens: Optional[int] = None,
            observable: bool = False) -> PlaceHandle:
    handle = PlaceHandle(name)
    self._places.append((handle, capacity, initial_tokens, observable))
    return handle

  def transition(self,
                 name: str,
                 *,
                 script: bool = False,
                 jit: bool = False
                 ) -> Callable[[Callable[[TransitionBuilder], None]], Callable[[TransitionBuilder], None]]:
    if jit:
      script = True

    def decorator(fn: Callable[[TransitionBuilder], None]):
      builder = TransitionBuilder(name)
      if script:
        executor = TransitionScriptExecutor(
            fn, require_builder_arg=not jit)
        executor(builder)
      else:
        fn(builder)
      self._transitions.append(builder)
      return fn
    return decorator

  def jit(self, name: str):
    """Convenience decorator mirroring Triton-style @jit syntax."""
    return self.transition(name, script=True, jit=True)

  def build(self) -> str:
    lines = ["module {", "  lpn.net {"]

    for place, capacity, initial, observable in self._places:
      attrs = []
      if capacity is not None:
        attrs.append(f"capacity = {capacity} : i64")
      if initial is not None:
        attrs.append(f"initial_tokens = {initial} : i64")
      if observable:
        attrs.append("observable")
      attr_text = ""
      if attrs:
        attr_text = " {" + ", ".join(attrs) + "}"
      lines.append(f"    lpn.place @{place.name}{attr_text}")

    for transition in self._transitions:
      lines.append(f"    lpn.transition @{transition.name} {{")
      lines.append("      ^bb0:")
      lines.extend(transition.render())
      lines.append("    }")

    lines.append("    lpn.halt")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)

  def emit_to_file(self, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
      handle.write(self.build())
