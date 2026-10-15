"""Configuration-aware VME decode from instruction operands, never timestamps.

Scalar constant propagation recovers static mset and tile selectors. Unknown
runtime configuration values fail explicitly instead of inventing a shape.
"""

from dataclasses import replace

from .matrix import MatrixCommand
from .model import Instruction


def signed(value, bits):
  return value - (1 << bits) if value & (1 << (bits - 1)) else value


class MatrixTraceState:
  def __init__(self):
    self.regs = [None] * 32
    self.regs[0] = 0
    self.sew, self.lmul, self.vl = 32, 1, 4
    self.altfmt = False
    self.m, self.n, self.k, self.widen = None, None, None, 0

  def constant(self, reg):
    value = self.regs[reg]
    if value is None:
      raise ValueError(f"VME configuration/selector x{reg} is data-dependent; supply a specialized workload")
    return value

  def vector_group(self, reg, size=None):
    size = self.lmul if size is None else size
    if size < 1 or reg + size > 32:
      raise ValueError("unsupported vector register group")
    return tuple(f"v{i}" for i in range(reg, reg + size))

  def tile_group(self, tile):
    if tile not in (0, 4, 8, 12):
      raise ValueError("VME decoder currently requires aligned 32-bit tile groups")
    return tuple(f"mt{i}" for i in range(tile, tile + 4))

  def configure(self, vtype, matrix=False):
    self.altfmt = bool(vtype & 0x100)
    self.sew = 8 << ((vtype >> 3) & 7)
    if self.sew not in (8, 16, 32):
      raise ValueError("unsupported vector element width")
    lmul_bits = vtype & 7
    if lmul_bits > 3:
      raise ValueError("fractional LMUL needs an explicit workload")
    self.lmul = self.sew // 8 if matrix else 1 << lmul_bits

  def decode(self, pc, bits, next_pc, base_decode):
    op, rd, f3 = bits & 127, (bits >> 7) & 31, (bits >> 12) & 7
    rs1, rs2, f7, f6 = (bits >> 15) & 31, (bits >> 20) & 31, bits >> 25, bits >> 26
    x = lambda reg: (f"x{reg}",) if reg else ()
    result = None
    if op == 0x57 and f3 == 7:
      if f7 == 0x41:  # msetmtype
        mtype = self.constant(rs1)
        self.widen = mtype & 3
        self.m, self.k = (mtype >> 10) & 0x3fff, (mtype >> 5) & 7
        self.configure(self.constant(rs2), matrix=bool(self.widen))
        self.n = self.vl = 0
      elif f7 == 0x42:
        sub = rs2 & 7
        if sub == 0:
          self.n = self.vl = min(self.constant(rs1), 16, self.lmul * 128 // self.sew)
        elif sub == 1:
          self.m = min(self.constant(rs1), 16)
        elif sub == 2:
          self.k = min(self.constant(rs1), 32 // self.sew)
        elif sub == 3:
          self.widen = rs1 & 3
          self.configure(((rs2 >> 3) & 3) << 3, matrix=bool(self.widen))
          self.m = self.k = self.n = self.vl = 0
        else:
          raise ValueError("unmodeled mset subtype")
      elif bits >> 30 == 0:  # vsetvli
        self.configure(bits >> 20)
        self.vl = min(self.constant(rs1), self.lmul * 128 // self.sew) if rs1 else self.lmul * 128 // self.sew
      elif bits >> 30 == 3:  # vsetivli
        self.configure((bits >> 20) & 0x3ff)
        self.vl = min(rs1, self.lmul * 128 // self.sew)
      else:
        self.configure(self.constant(rs2))
        self.vl = min(self.constant(rs1), self.lmul * 128 // self.sew)
      if rd:
        self.regs[rd] = self.vl
      return Instruction("vconfig", x(rs1) + (x(rs2) if f7 in (0x40, 0x41) else ()),
                         x(rd) + ("vtype", "mtype"), pc, bits)

    if op == 0x77:
      if f6 != 0x3c or f3 not in (0, 1) or not ((bits >> 25) & 1):
        raise ValueError("unsupported VME matrix encoding")
      if None in (self.m, self.n, self.k):
        raise ValueError("matrix instruction requires preceding known mset configuration")
      if f3 == 0 and (self.sew != 8 or self.widen != 3 or self.altfmt):
        raise ValueError("integer matrix decode supports the validated INT8/UINT8 to INT32 format")
      if f3 == 1 and (self.sew, self.widen) not in ((16, 2), (32, 1)):
        raise ValueError("floating matrix decode supports BF16/FP32 to FP32")
      if f3 == 1 and self.sew == 32 and rd & 1:
        raise ValueError("mixed-format FP32 matrix operands need an explicit workload")
      dtype = ("int8" if rd & 1 else "uint8") if f3 == 0 else "bf16" if self.sew == 16 else "fp32"
      command = MatrixCommand(rd >> 1, self.m, self.n, self.k, dtype)
      # Four operand uops: byte rows use v8/10/12/14, BF16 uses
      # v8/9/12/13, FP32 v8..11. Inactive K rows may still be read by RTL.
      groups = {8: (0, 2, 4, 6), 16: (0, 1, 4, 5), 32: (0, 1, 2, 3)}[self.sew]
      vectors = tuple(f"v{base + offset}" for base in (rs2, rs1) for offset in groups)
      tiles = self.tile_group(command.tile)
      return Instruction("matrix", vectors + tiles + ("vtype", "mtype"), tiles,
                         pc, bits, matrix=command)

    if op == 0x57 and f3 == 6 and f6 == 0x10 and rs2 in (28, 30, 31):
      if rs2 == 30:  # vtzero
        result = Instruction("tile", ("vtype", "mtype"), self.tile_group(rd >> 1), pc, bits)
      elif rs2 == 28:  # discard
        result = Instruction("tile", (), tuple(f"mt{i}" for i in range(16)), pc, bits)
      else:  # tile -> vector
        tile = (self.constant(rs1) >> 27) & 15
        result = Instruction("tile", x(rs1) + self.tile_group(tile) + ("vtype", "mtype"),
                             self.vector_group(rd), pc, bits)
    elif op == 0x57 and f3 == 6 and f6 == 0x17:
      tile = (self.constant(rs1) >> 27) & 15
      result = Instruction("tile", x(rs1) + self.vector_group(rs2) + ("vtype", "mtype"),
                           self.tile_group(tile), pc, bits)
    elif op in (0x07, 0x27) and f3 == 7 and (bits >> 28) & 1:
      tile = (self.constant(rs2) >> 27) & 15
      tiles = self.tile_group(tile)
      width = 8 << (bits >> 29)
      if width not in (8, 16, 32):
        raise ValueError("unsupported tile memory width")
      reads = x(rs1) + x(rs2) + ("vtype", "mtype") + (tiles if op == 0x27 else ())
      result = Instruction("tile", reads, tiles if op == 0x07 else (), pc, bits,
                           bytes=self.vl * width // 8, uops=max(1, self.vl * width // 128))
    else:
      result = base_decode(pc, bits, next_pc)
      if op in (0x07, 0x27) and f3 in (0, 5, 6):
        size = self.vl * {0: 1, 5: 2, 6: 4}[f3]
        vectors = self.vector_group(rd)
        result = replace(result, reads=x(rs1) + (vectors if op == 0x27 else ()),
                         writes=vectors if op == 0x07 else (), bytes=size,
                         uops=max(1, (size + 15) // 16))

    # Propagate only integer constants used as configuration/selectors. Loads
    # and unimplemented operations invalidate destination constants.
    value = None
    if op == 0x37:
      value = bits & 0xfffff000
    elif op == 0x17:
      value = pc + (bits & 0xfffff000)
    elif op == 0x13 and self.regs[rs1] is not None:
      left, imm = self.regs[rs1], signed(bits >> 20, 12)
      if f3 == 0: value = left + imm
      elif f3 == 6: value = left | imm
      elif f3 == 7: value = left & imm
      elif f3 == 4: value = left ^ imm
      elif f3 == 1: value = left << (rs2 & 31)
      elif f3 == 5: value = left >> (rs2 & 31)
    if rd and f"x{rd}" in result.writes:
      self.regs[rd] = None if value is None else value & 0xffffffff
    return result
