"""Typed Python functional models compiled through MLIR to native libraries.

Python is the authoring/reference language. Compiled calls execute native
arithmetic, mutable buffer accesses and control flow, without Python callbacks.
"""

import ast
import ctypes as C
from dataclasses import dataclass
import fcntl
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import textwrap


@dataclass(frozen=True)
class Scalar:
  name: str
  mlir: str
  numpy: str
  ctype: object
  width: int
  unsigned: bool = False

  def __call__(self,value):
    if self.width==1:
      return bool(value)
    if self.name=="f32":
      return struct.unpack("f",struct.pack("f",float(value)))[0]
    bits = int(value)&((1<<self.width)-1)
    return bits if self.unsigned or self.width==1 or bits<(1<<(self.width-1)) else bits-(1<<self.width)


i64 = Scalar("i64","i64","int64",C.c_int64,64)
i32 = Scalar("i32","i32","int32",C.c_int32,32)
u32 = Scalar("u32","i32","uint32",C.c_uint32,32,True)
f32 = Scalar("f32","f32","float32",C.c_float,32)
boolean = Scalar("boolean","i1","bool",C.c_bool,1,True)


@dataclass(frozen=True)
class BufferType:
  element: Scalar

  @property
  def mlir(self):
    return f"memref<?x{self.element.mlir}>"


class Buffer:
  def __class_getitem__(cls,element):
    if not isinstance(element,Scalar):
      raise TypeError("Buffer needs a scalar element type")
    return BufferType(element)


def bitcast(value,target):
  if target==f32:
    return struct.unpack("f",struct.pack("I",int(value)&0xffffffff))[0]
  if target in (u32,i32):
    return target(struct.unpack("I",struct.pack("f",float(value)))[0])
  raise TypeError("bitcast supports f32 and 32-bit integers")


@dataclass
class Value:
  name: str
  typ: object
  literal: bool = False


class Frontend:
  def __init__(self,model):
    self.model = model
    self.lines,self.counter,self.indent = [],0,2
    self.types = dict(model.arguments)
    self.buffers = {n:Value("%arg_"+n,t) for n,t in self.types.items() if isinstance(t,BufferType)}
    for node in ast.walk(model.node):
      if isinstance(node,ast.AnnAssign):
        if not isinstance(node.target,ast.Name):
          self.fail(node,"annotate scalar local names, not buffer elements")
        typ = self.annotation(node.annotation)
        if not isinstance(typ,Scalar):
          self.fail(node,"locals must be scalars")
        if node.target.id in self.types and self.types[node.target.id]!=typ:
          self.fail(node,"local type changes")
        self.types[node.target.id] = typ
      if isinstance(node,ast.For) and isinstance(node.target,ast.Name):
        self.types.setdefault(node.target.id,i64)
    self.locals = {n:Value("%local_"+n,t) for n,t in self.types.items() if isinstance(t,Scalar)}
    self.iterators = {id(n):Value(f"%iterator_{j}",i64)
                      for j,n in enumerate(n for n in ast.walk(model.node) if isinstance(n,ast.For))}
    self.definite(model.node.body,set(model.arguments))

  def definite(self,statements,assigned):
    assigned = set(assigned)
    def read(node):
      if node is not None:
        for name in ast.walk(node):
          if isinstance(name,ast.Name) and isinstance(name.ctx,ast.Load) and name.id in self.types and name.id not in assigned:
            self.fail(name,"local may be read before assignment")
    for node in statements:
      if isinstance(node,(ast.Assign,ast.AnnAssign)):
        read(node.value)
        targets = node.targets if isinstance(node,ast.Assign) else [node.target]
        for target in targets:
          if isinstance(target,ast.Name) and node.value is not None:
            assigned.add(target.id)
          elif isinstance(target,ast.Subscript):
            read(target.value)
            read(target.slice)
      elif isinstance(node,ast.AugAssign):
        if isinstance(node.target,ast.Name) and node.target.id not in assigned:
          self.fail(node,"local may be read before assignment")
        read(node.target)
        read(node.value)
      elif isinstance(node,ast.If):
        read(node.test)
        assigned = self.definite(node.body,assigned)&self.definite(node.orelse,assigned)
      elif isinstance(node,ast.While):
        read(node.test)
        self.definite(node.body,assigned)
      elif isinstance(node,ast.For):
        read(node.iter)
        self.definite(node.body,assigned|({node.target.id} if isinstance(node.target,ast.Name) else set()))
      elif isinstance(node,ast.Return):
        read(node.value)
      elif isinstance(node,ast.Expr):
        read(node.value)
    return assigned

  def fail(self,node,message):
    raise TypeError(f"{self.model.name}:{getattr(node,'lineno',0)}: {message}")

  def annotation(self,node):
    if isinstance(node,ast.Name):
      value = self.model.environment.get(node.id)
      if isinstance(value,(Scalar,BufferType)):
        return value
    self.fail(node,"use an explicit scalar annotation such as i64, u32 or f32")

  def line(self,text):
    self.lines.append("  "*self.indent+text)

  def value(self,operation,typ):
    name = f"%v{self.counter}"
    self.counter += 1
    self.line(f"{name} = {operation}")
    return Value(name,typ)

  def constant(self,value,typ=None):
    typ = typ or (boolean if isinstance(value,bool) else f32 if isinstance(value,float) else i64)
    if typ==f32:
      if not math.isfinite(float(value)):
        raise ValueError("nonfinite float literals are not supported")
      literal = format(float(value),".17e")
    elif typ==boolean:
      literal = "true" if value else "false"
    else:
      literal = str(typ(value))
    result = self.value(f"arith.constant {literal}"+("" if typ==boolean else f" : {typ.mlir}"),typ)
    result.literal = True
    return result

  def cast(self,value,typ):
    if value.typ==typ:
      return value
    if isinstance(value.typ,BufferType) or isinstance(typ,BufferType):
      raise TypeError("buffers cannot be cast or rebound")
    source = value.typ
    if typ==boolean:
      return self.truth(value)
    if source.mlir==typ.mlir:
      return Value(value.name,typ,value.literal)
    if source==f32:
      raise TypeError("float-to-integer casts need checked numerical kernels; use bitcast for raw bits")
    elif typ==f32:
      op = "arith.uitofp" if source.unsigned else "arith.sitofp"
    elif source.width<typ.width:
      op = "arith.extui" if source.unsigned else "arith.extsi"
    else:
      op = "arith.trunci"
    return self.value(f"{op} {value.name} : {source.mlir} to {typ.mlir}",typ)

  def index(self,value):
    value = self.cast(value,i64)
    return self.value(f"arith.index_cast {value.name} : i64 to index",Scalar("index","index","int64",C.c_int64,64))

  def error(self,condition,code):
    condition = self.both(condition,self.healthy())
    self.line(f"scf.if {condition.name} {{")
    self.indent += 1
    value = self.constant(code)
    self.line(f"memref.store {value.name}, %error[%zero] : memref<1xi64>")
    self.indent -= 1
    self.line("}")

  def healthy(self):
    status = self.value("memref.load %error[%zero] : memref<1xi64>",i64)
    zero = self.constant(0)
    return self.value(f"arith.cmpi eq, {status.name}, {zero.name} : i64",boolean)

  def compare(self,predicate,left,right):
    if left.typ==f32:
      pred = {"eq":"oeq","ne":"une","lt":"olt","le":"ole","gt":"ogt","ge":"oge"}[predicate]
      return self.value(f"arith.cmpf {pred}, {left.name}, {right.name} : f32",boolean)
    pred = predicate if predicate in ("eq","ne") else ("u" if left.typ.unsigned else "s")+predicate
    return self.value(f"arith.cmpi {pred}, {left.name}, {right.name} : {left.typ.mlir}",boolean)

  def both(self,left,right):
    return self.value(f"arith.andi {left.name}, {right.name} : i1",boolean)

  def access(self,node):
    buffer = self.expr(node.value)
    if not isinstance(buffer.typ,BufferType):
      self.fail(node,"index a typed Buffer")
    idx = self.cast(self.expr(node.slice),i64)
    dimension = self.value(f"memref.dim {buffer.name}, %zero : {buffer.typ.mlir}",Scalar("index","index","int64",C.c_int64,64))
    size = self.value(f"arith.index_cast {dimension.name} : index to i64",i64)
    valid = self.both(self.compare("ge",idx,self.constant(0)),self.compare("lt",idx,size))
    valid = self.both(valid,self.healthy())
    return buffer,self.index(idx),valid

  def region_value(self,fn):
    saved,self.lines = self.lines,[]
    self.indent += 1
    result = fn()
    self.line(f"scf.yield {result.name} : {result.typ.mlir}")
    lines = self.lines
    self.lines = saved
    self.indent -= 1
    return result,lines

  def conditional(self,condition,true_fn,false_fn):
    yes,yes_lines = self.region_value(true_fn)
    no,no_lines = self.region_value(lambda:self.cast(false_fn(),yes.typ))
    result = self.value(f"scf.if {condition.name} -> ({yes.typ.mlir}) {{",yes.typ)
    self.lines.extend(yes_lines)
    self.line("} else {")
    self.lines.extend(no_lines)
    self.line("}")
    return result

  def pair(self,left_node,right_node):
    left = self.expr(left_node)
    right = self.expr(right_node,None if left.literal else left.typ)
    if left.typ!=right.typ:
      if left.literal:
        left = self.cast(left,right.typ)
      elif right.literal:
        right = self.cast(right,left.typ)
      else:
        self.fail(right_node,"mixed types require an explicit cast")
    return left,right

  def expr(self,node,expected=None):
    if isinstance(node,ast.Constant) and isinstance(node.value,(int,float,bool)):
      return self.constant(node.value,expected)
    if isinstance(node,ast.Name):
      if node.id in self.buffers:
        return self.buffers[node.id]
      if node.id in self.locals:
        local = self.locals[node.id]
        return self.value(f"memref.load {local.name}[%zero] : memref<1x{local.typ.mlir}>",local.typ)
      value = self.model.environment.get(node.id)
      if isinstance(value,(int,float,bool)):
        return self.constant(value,expected)
      self.fail(node,"unknown name or unsupported global value")
    if isinstance(node,ast.Subscript):
      buffer,idx,valid = self.access(node)
      def read():
        return self.value(f"memref.load {buffer.name}[{idx.name}] : {buffer.typ.mlir}",buffer.typ.element)
      def failed():
        self.error(self.constant(True),1)
        return self.constant(0,buffer.typ.element)
      return self.conditional(valid,read,failed)
    if isinstance(node,ast.BinOp):
      left,right = self.pair(node.left,node.right)
      if left.typ==boolean and not isinstance(node.op,(ast.BitAnd,ast.BitOr,ast.BitXor)):
        self.fail(node,"boolean arithmetic requires an explicit integer cast")
      float_ops = {ast.Add:"addf",ast.Sub:"subf",ast.Mult:"mulf",ast.Div:"divf"}
      int_ops = {ast.Add:"addi",ast.Sub:"subi",ast.Mult:"muli",ast.BitAnd:"andi",ast.BitOr:"ori",ast.BitXor:"xori",ast.LShift:"shli"}
      op = (float_ops if left.typ==f32 else int_ops).get(type(node.op))
      if isinstance(node.op,(ast.RShift,ast.FloorDiv,ast.Mod)) and left.typ!=f32:
        if isinstance(node.op,ast.RShift):
          op = "shrui" if left.typ.unsigned else "shrsi"
        elif isinstance(node.op,ast.FloorDiv):
          op = "divui" if left.typ.unsigned else "floordivsi"
        else:
          # Signed Python modulo also depends on the divisor's sign. Keep
          # this first subset explicit rather than silently using LLVM srem.
          if not left.typ.unsigned:
            self.fail(node,"modulo currently requires an unsigned operand")
          op = "remui"
      if op is None:
        self.fail(node,"unsupported arithmetic operation")
      if isinstance(node.op,(ast.LShift,ast.RShift)):
        valid = self.both(self.compare("ge",right,self.constant(0,right.typ)),
                          self.compare("lt",right,self.constant(left.typ.width,right.typ)))
        self.error(self.value(f"arith.xori {valid.name}, {self.constant(True).name} : i1",boolean),2)
        right = self.value(f"arith.select {valid.name}, {right.name}, {self.constant(0,right.typ).name} : {right.typ.mlir}",right.typ)
      if isinstance(node.op,(ast.FloorDiv,ast.Mod)):
        valid = self.compare("ne",right,self.constant(0,right.typ))
        if not left.typ.unsigned:
          minimum = self.constant(-(1<<(left.typ.width-1)),left.typ)
          overflow = self.both(self.compare("eq",left,minimum),self.compare("eq",right,self.constant(-1,right.typ)))
          valid = self.both(valid,self.value(f"arith.xori {overflow.name}, {self.constant(True).name} : i1",boolean))
        self.error(self.value(f"arith.xori {valid.name}, {self.constant(True).name} : i1",boolean),2)
        right = self.value(f"arith.select {valid.name}, {right.name}, {self.constant(1,right.typ).name} : {right.typ.mlir}",right.typ)
      return self.value(f"arith.{op} {left.name}, {right.name} : {left.typ.mlir}",left.typ)
    if isinstance(node,ast.Compare) and len(node.ops)==1:
      left,right = self.pair(node.left,node.comparators[0])
      predicate = {ast.Eq:"eq",ast.NotEq:"ne",ast.Lt:"lt",ast.LtE:"le",ast.Gt:"gt",ast.GtE:"ge"}.get(type(node.ops[0]))
      if predicate:
        return self.compare(predicate,left,right)
    if isinstance(node,ast.UnaryOp):
      value = self.expr(node.operand,expected)
      if isinstance(node.op,ast.Not):
        value = self.truth(value)
        return self.value(f"arith.xori {value.name}, {self.constant(True).name} : i1",boolean)
      if isinstance(node.op,ast.USub):
        if value.typ==boolean:
          self.fail(node,"boolean arithmetic requires an explicit integer cast")
        op = "subf" if value.typ==f32 else "subi"
        return self.value(f"arith.{op} {self.constant(0,value.typ).name}, {value.name} : {value.typ.mlir}",value.typ)
      if isinstance(node.op,ast.Invert) and value.typ not in (f32,boolean):
        return self.value(f"arith.xori {value.name}, {self.constant(-1,value.typ).name} : {value.typ.mlir}",value.typ)
    if isinstance(node,ast.BoolOp):
      def operand(other):
        value = self.expr(other)
        if value.typ!=boolean:
          self.fail(other,"and/or require boolean operands; cast with boolean() explicitly")
        return value
      value = operand(node.values[0])
      for other in node.values[1:]:
        if isinstance(node.op,ast.And):
          value = self.conditional(value,lambda:operand(other),lambda:self.constant(False))
        else:
          value = self.conditional(value,lambda:self.constant(True),lambda:operand(other))
      return value
    if isinstance(node,ast.IfExp):
      return self.conditional(self.truth(self.expr(node.test)),lambda:self.expr(node.body,expected),lambda:self.expr(node.orelse,expected))
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
      if node.func.id in self.types:
        self.fail(node,"indirect calls are unsupported")
      callee = self.model.environment.get(node.func.id)
      if isinstance(callee,Scalar) and len(node.args)==1:
        return self.cast(self.expr(node.args[0]),callee)
      if node.func.id=="len" and len(node.args)==1:
        buffer = self.expr(node.args[0])
        if isinstance(buffer.typ,BufferType):
          size = self.value(f"memref.dim {buffer.name}, %zero : {buffer.typ.mlir}",Scalar("index","index","int64",C.c_int64,64))
          return self.value(f"arith.index_cast {size.name} : index to i64",i64)
      if callee is bitcast and len(node.args)==2:
        value,target = self.expr(node.args[0]),self.annotation(node.args[1])
        if value.typ.width==target.width==32 and (value.typ==f32 or target==f32):
          return self.value(f"arith.bitcast {value.name} : {value.typ.mlir} to {target.mlir}",target)
        self.fail(node,"bitcast requires matching 32-bit float/integer types")
    self.fail(node,"unsupported expression")

  def truth(self,value):
    if value.typ==boolean:
      return value
    return self.compare("ne",value,self.constant(0,value.typ))

  def assign(self,target,value):
    if isinstance(target,ast.Name) and target.id in self.locals:
      local = self.locals[target.id]
      value = self.cast(value,local.typ)
      self.line(f"memref.store {value.name}, {local.name}[%zero] : memref<1x{local.typ.mlir}>")
      return
    if isinstance(target,ast.Subscript):
      buffer,idx,valid = self.access(target)
      value = self.cast(value,buffer.typ.element)
      self.line(f"scf.if {valid.name} {{")
      self.indent += 1
      self.line(f"memref.store {value.name}, {buffer.name}[{idx.name}] : {buffer.typ.mlir}")
      self.indent -= 1
      self.line("} else {")
      self.indent += 1
      self.error(self.constant(True),1)
      self.indent -= 1
      self.line("}")
      return
    self.fail(target,"scalar assignments need an annotation; buffer rebinding is unsupported")

  def loop(self,condition,body):
    self.line("scf.while : () -> () {")
    self.indent += 1
    cond = self.both(self.truth(condition()),self.healthy())
    self.line(f"scf.condition({cond.name})")
    self.indent -= 1
    self.line("} do {")
    self.indent += 1
    body()
    self.line("scf.yield")
    self.indent -= 1
    self.line("}")

  def block(self,statements):
    for node in statements:
      if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
        continue
      if isinstance(node,(ast.AnnAssign,ast.Assign)):
        target = node.target if isinstance(node,ast.AnnAssign) else node.targets[0]
        if isinstance(node,ast.Assign) and len(node.targets)!=1:
          self.fail(node,"multiple assignment is unsupported")
        if node.value is not None:
          typ = self.locals[target.id].typ if isinstance(target,ast.Name) and target.id in self.locals else None
          self.assign(target,self.expr(node.value,typ))
      elif isinstance(node,ast.AugAssign):
        self.assign(node.target,self.expr(ast.BinOp(left=node.target,op=node.op,right=node.value)))
      elif isinstance(node,ast.If):
        cond = self.truth(self.expr(node.test))
        self.line(f"scf.if {cond.name} {{")
        self.indent += 1
        self.block(node.body)
        self.indent -= 1
        if node.orelse:
          self.line("} else {")
          self.indent += 1
          self.block(node.orelse)
          self.indent -= 1
        self.line("}")
      elif isinstance(node,ast.While) and not node.orelse:
        self.loop(lambda:self.expr(node.test),lambda:self.block(node.body))
      elif isinstance(node,ast.For) and not node.orelse and isinstance(node.target,ast.Name):
        call = node.iter
        if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!="range" or call.keywords or not 1<=len(call.args)<=3:
          self.fail(node,"for loops require range(lower, upper, positive_step)")
        args = [self.cast(self.expr(a),i64) for a in call.args]
        lower,upper,step = (self.constant(0),args[0],self.constant(1)) if len(args)==1 else (args[0],args[1],args[2] if len(args)==3 else self.constant(1))
        if self.locals[node.target.id].typ!=i64:
          self.fail(node,"range loop variables must have type i64")
        self.error(self.compare("le",step,self.constant(0)),2)
        # Keep iterator state separate: assignments to the Python loop target
        # do not change range's next value, and an empty loop leaves it alone.
        iterator = self.iterators[id(node)]
        self.line(f"memref.store {lower.name}, {iterator.name}[%zero] : memref<1xi64>")
        def current():
          return self.value(f"memref.load {iterator.name}[%zero] : memref<1xi64>",i64)
        def condition():
          return self.compare("lt",current(),upper)
        def body():
          value = current()
          self.assign(node.target,value)
          self.block(node.body)
          next_value = self.value(f"arith.addi {value.name}, {step.name} : i64",i64)
          increasing = self.compare("gt",next_value,value)
          # A positive increment overflowing i64 has passed every possible
          # upper bound; end the loop rather than wrapping indefinitely.
          next_value = self.value(f"arith.select {increasing.name}, {next_value.name}, {upper.name} : i64",i64)
          self.line(f"memref.store {next_value.name}, {iterator.name}[%zero] : memref<1xi64>")
        self.loop(condition,body)
      elif isinstance(node,ast.Pass):
        pass
      else:
        self.fail(node,"unsupported statement; use a single final return and explicit loop state")

  def emit(self):
    self.line("%zero = arith.constant 0 : index")
    zero = self.constant(0)
    self.line(f"memref.store {zero.name}, %error[%zero] : memref<1xi64>")
    for iterator in self.iterators.values():
      self.line(f"{iterator.name} = memref.alloca() : memref<1xi64>")
    for name,local in self.locals.items():
      self.line(f"{local.name} = memref.alloca() : memref<1x{local.typ.mlir}>")
      initial = Value("%arg_"+name,local.typ) if name in self.model.arguments else self.constant(0,local.typ)
      self.line(f"memref.store {initial.name}, {local.name}[%zero] : memref<1x{local.typ.mlir}>")
    body = self.model.node.body
    if not body or not isinstance(body[-1],ast.Return):
      self.fail(self.model.node,"use a single final return")
    self.block(body[:-1])
    if self.model.result is None:
      if body[-1].value is not None:
        self.fail(body[-1],"void model must return without a value")
      self.line("func.return")
    else:
      result = self.cast(self.expr(body[-1].value,self.model.result),self.model.result)
      self.line(f"func.return {result.name} : {result.typ.mlir}")
    args = [f"%arg_{n}: {t.mlir}" for n,t in self.model.arguments.items()]+["%error: memref<1xi64>"]
    result = f" -> {self.model.result.mlir}" if self.model.result else ""
    return "module {\n  func.func @"+self.model.name+"("+", ".join(args)+")"+result+" attributes {llvm.emit_c_interface} {\n"+"\n".join(self.lines)+"\n  }\n}\n"


class Descriptor(C.Structure):
  _fields_ = [("allocated",C.c_void_p),("aligned",C.c_void_p),("offset",C.c_int64),
              ("sizes",C.c_int64*1),("strides",C.c_int64*1)]


class NativeFunction:
  def __init__(self,model,path,cache_hit,compile_seconds):
    self.model,self.path,self.cache_hit,self.compile_seconds = model,path,cache_hit,compile_seconds
    self.library = C.CDLL(str(path))
    self.entry = getattr(self.library,"_mlir_ciface_"+model.name)
    self.entry.argtypes = [C.POINTER(Descriptor) if isinstance(t,BufferType) else t.ctype
                          for t in model.arguments.values()]+[C.POINTER(Descriptor)]
    self.entry.restype = model.result.ctype if model.result else None

  def __call__(self,*args,**kwargs):
    import numpy as np
    bound = self.model.signature.bind(*args,**kwargs)
    bound.apply_defaults()
    call_args,keepalive = [],[]
    for name,typ in self.model.arguments.items():
      value = bound.arguments[name]
      if isinstance(typ,BufferType):
        if not isinstance(value,np.ndarray) or value.ndim!=1 or value.dtype!=np.dtype(typ.element.numpy) or not value.flags.c_contiguous or not value.flags.writeable or not value.flags.aligned:
          raise TypeError(f"{name} needs a writable contiguous 1D {typ.element.numpy} numpy array")
        descriptor = Descriptor(value.ctypes.data,value.ctypes.data,0,(len(value),),(1,))
        keepalive.append(descriptor)
        call_args.append(C.byref(descriptor))
      else:
        call_args.append(typ(value))
    error = np.zeros(1,np.int64)
    descriptor = Descriptor(error.ctypes.data,error.ctypes.data,0,(1,),(1,))
    result = self.entry(*call_args,C.byref(descriptor))
    if error[0]==1:
      raise IndexError("functional model buffer index out of bounds")
    if error[0]:
      raise ValueError("invalid integer division, shift, or range step in functional model")
    return result


class FunctionalModel:
  def __init__(self,fn):
    self.fn,self.name,self.signature = fn,fn.__name__,inspect.signature(fn)
    self.node = next(n for n in ast.parse(textwrap.dedent(inspect.getsource(fn))).body if isinstance(n,ast.FunctionDef))
    closure = inspect.getclosurevars(fn)
    self.environment = {**fn.__globals__,**closure.nonlocals}
    annotations = inspect.get_annotations(fn,eval_str=True)
    self.arguments = {n:annotations.get(n) for n in self.signature.parameters}
    self.result = annotations.get("return")
    if any(not isinstance(t,(Scalar,BufferType)) for t in self.arguments.values()) or self.result is not None and not isinstance(self.result,Scalar):
      raise TypeError("annotate every argument with a scalar or Buffer, and the scalar return")
    if any(p.kind not in (p.POSITIONAL_OR_KEYWORD,p.KEYWORD_ONLY) for p in self.signature.parameters.values()):
      raise TypeError("variadic and positional-only parameters are not supported")
    self.__name__,self.__doc__ = fn.__name__,fn.__doc__

  def __call__(self,*args,**kwargs):
    bound = self.signature.bind(*args,**kwargs)
    bound.apply_defaults()
    for name,typ in self.arguments.items():
      if isinstance(typ,Scalar):
        bound.arguments[name] = typ(bound.arguments[name])
    value = self.fn(*bound.args,**bound.kwargs)
    return self.result(value) if self.result else value

  def emit(self):
    return Frontend(self).emit()

  def compile(self,cache_dir=None):
    import time
    before = time.perf_counter()
    source = self.emit()
    names = ("mlir-opt-20","mlir-translate-20","clang++-20")
    tools = {n:shutil.which(n) for n in names}
    if not all(tools.values()):
      raise RuntimeError("mlir-opt-20, mlir-translate-20 and clang++-20 are required")
    versions = {n:subprocess.check_output([p,"--version"],text=True) for n,p in tools.items()}
    key = hashlib.sha256((source+json.dumps(versions,sort_keys=True)+"functional-abi-v1-O3-strict-fp").encode()).hexdigest()
    root = Path(cache_dir or Path.home()/".cache/lpn-functional")/key
    root.mkdir(parents=True,exist_ok=True)
    library = root/"model.so"
    with (root/"build.lock").open("a") as lock:
      fcntl.flock(lock,fcntl.LOCK_EX)
      hit = library.exists()
      if not hit:
        with tempfile.TemporaryDirectory(dir=root) as temporary:
          tmp = Path(temporary)
          ir,lowered,llvm = tmp/"model.mlir",tmp/"llvm.mlir",tmp/"model.ll"
          ir.write_text(source)
          commands = [
            [tools["mlir-opt-20"],str(ir),"--canonicalize","--convert-scf-to-cf",
             "--convert-arith-to-llvm","--convert-index-to-llvm","--convert-cf-to-llvm",
             "--finalize-memref-to-llvm","--convert-func-to-llvm","--reconcile-unrealized-casts","-o",str(lowered)],
            [tools["mlir-translate-20"],"--mlir-to-llvmir",str(lowered),"-o",str(llvm)],
            [tools["clang++-20"],"-O3","-shared","-fPIC","-ffp-contract=off",str(llvm),"-o",str(tmp/"model.so")]]
          for command in commands:
            r = subprocess.run(command,capture_output=True,text=True)
            if r.returncode:
              (root/"failed.mlir").write_text(source)
              raise RuntimeError(f"functional compilation failed: {r.stderr}")
          (root/"model.mlir").write_text(source)
          (root/"model.ll").write_text(llvm.read_text())
          (tmp/"model.so").replace(library)
    return NativeFunction(self,library,hit,time.perf_counter()-before)


def functional(fn):
  return FunctionalModel(fn)
