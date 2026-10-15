"""Decode observed RV32/RVV instructions without importing measured timing."""

import csv
from pathlib import Path

from .model import Instruction


def decode(pc: int, bits: int, next_pc=None) -> Instruction:
  if bits & 3 != 3:
    raise ValueError(f"compressed instructions are unsupported at 0x{pc:x}")
  op, rd, f3 = bits & 127, (bits >> 7) & 31, (bits >> 12) & 7
  rs1, rs2, f7 = (bits >> 15) & 31, (bits >> 20) & 31, bits >> 25
  x = lambda reg: (f"x{reg}",) if reg else ()
  f = lambda reg: (f"f{reg}",)
  v = lambda reg: (f"v{reg}",)
  unit, reads, writes, size = "alu", (), (), 0
  redirect = False
  if op in (0x37, 0x17):  # LUI/AUIPC
    writes = x(rd)
  elif op == 0x13:
    reads, writes = x(rs1), x(rd)
  elif op == 0x33:
    reads, writes = x(rs1) + x(rs2), x(rd)
    if f7 == 1:
      unit = "mul" if f3 < 4 else "div"
    elif f7 not in (0, 0x20):
      raise ValueError(f"unmodeled integer extension at 0x{pc:x}: 0x{bits:08x}")
  elif op in (0x63, 0x67, 0x6f):
    unit = "branch"
    reads = x(rs1) + x(rs2) if op == 0x63 else x(rs1) if op == 0x67 else ()
    writes = x(rd) if op != 0x63 else ()
    redirect = next_pc is not None and next_pc != pc + 4
  elif op in (0x03, 0x23):
    if f3 not in (0, 1, 2, 4, 5):
      raise ValueError("unsupported scalar load/store width")
    unit, size = "lsu", 1 << (f3 & 3)
    reads = x(rs1) + (x(rs2) if op == 0x23 else ())
    writes = x(rd) if op == 0x03 else ()
  elif op in (0x07, 0x27):
    unit = "lsu"
    if f3 == 2:  # scalar FP32
      size = 4
      reads = x(rs1) + (f(rs2) if op == 0x27 else ())
      writes = f(rd) if op == 0x07 else ()
    elif f3 in (0, 5, 6):  # vector loads/stores, LMUL=1 baseline
      if (bits >> 29) != 0 or ((bits >> 26) & 3) != 0:
        raise ValueError("segmented/strided/indexed vector memory needs explicit uops")
      size = 16
      reads = x(rs1) + (v(rd) if op == 0x27 else ())
      writes = v(rd) if op == 0x07 else ()
    else:
      raise ValueError("unsupported float/vector memory width")
  elif op == 0x53:
    unit, reads, writes = "fpu", f(rs1) + f(rs2), f(rd)
    if f7 in (0x60, 0x70):
      reads, writes = f(rs1), x(rd)
    elif f7 in (0x68, 0x78):
      reads, writes = x(rs1), f(rd)
    elif f7 == 0x50:
      writes = x(rd)
    elif f7 in (0x0c, 0x2c):
      raise ValueError("FP divide/sqrt latency needs explicit modeling")
  elif op in (0x43, 0x47, 0x4b, 0x4f):
    unit = "fpu"
    reads, writes = f(rs1) + f(rs2) + f((bits >> 27) & 31), f(rd)
  elif op == 0x57:
    if f3 == 7:
      unit = "vconfig"
      reads, writes = x(rs1), x(rd) + ("vtype",)
    else:
      f6 = bits >> 26
      unit = "vfpu" if f3 in (1, 5) else "vmul" if f3 in (2, 6) and f6 >= 0x24 else "valu"
      if f3 in (2, 6) and f6 in (0x20, 0x21, 0x22, 0x23):
        unit = "vdiv"
      reads = v(rs2) + (v(rs1) if f3 in (0, 1, 2) else
                          f(rs1) if f3 == 5 else x(rs1) if f3 in (4, 6) else ())
      if (bits >> 25) & 1 == 0:
        reads += v(0)
      if f6 >= 0x28:  # accumulate forms also read vd
        reads += v(rd)
      reads += ("vtype",)
      writes = v(rd)
  elif op == 0x73:
    unit = "branch"
    reads = x(rs1) if f3 in (1, 2, 3) else ()
    writes = x(rd) if f3 else ()
  elif op == 0x0f:
    unit = "branch"
  else:
    raise ValueError(f"unmodeled opcode at 0x{pc:x}: 0x{bits:08x}")
  return Instruction(unit, reads, writes, pc, bits, size, redirect=redirect)


def load_dispatch_trace(path):
  from .trace_matrix import MatrixTraceState
  with Path(path).open() as source:
    rows = list(csv.DictReader(source))
  if not rows:
    raise ValueError("empty Verilator dispatch trace")
  pcs = [int(row["pc"], 0) for row in rows]
  state = MatrixTraceState()
  instructions = [state.decode(pc, int(row["inst"], 0),
                         pcs[idx + 1] if idx + 1 < len(pcs) else None, decode)
                  for idx, (row, pc) in enumerate(zip(rows, pcs))]
  # Baseline timestamps are returned separately for validation only.
  cycles = [int(row["cycle"]) for row in rows]
  if any(right < left for left, right in zip(cycles, cycles[1:])):
    raise ValueError("dispatch trace cycles are not monotonic")
  return instructions, {"first_dispatch": cycles[0],
                         "last_dispatch": cycles[-1],
                         "dispatch_span": cycles[-1] - cycles[0] + 1}
