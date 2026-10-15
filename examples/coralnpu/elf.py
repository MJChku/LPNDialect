"""Execute the supported RV32/VME subset to obtain a timestamp-free ELF workload.

The functional frontend chooses the instruction path from ELF operands. It is
not timed and does not read Verilator observations. Unsupported encodings fail.
"""

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import struct

from .trace import decode
from .trace_matrix import MatrixTraceState, signed


class ELFImage:
  def __init__(self, path):
    self.data = data = Path(path).read_bytes()
    if data[:7] != b"\x7fELF\x01\x01\x01":
      raise ValueError("expected a little-endian ELF32 executable")
    header = struct.unpack_from("<16sHHIIIIIHHHHHH", data)
    if header[2] != 243:
      raise ValueError("expected a RISC-V ELF")
    self.entry, self.memory, self.symbols = header[4], {}, {}
    phoff, shoff = header[5], header[6]
    for idx in range(header[10]):
      kind, offset, va, _, size, memsize, _, _ = struct.unpack_from(
          "<IIIIIIII", data, phoff+idx*header[9])
      if kind == 1:
        for pos in range(memsize):
          self.memory[va+pos] = data[offset+pos] if pos < size else 0
    sections = [struct.unpack_from("<IIIIIIIIII", data, shoff+i*header[11])
                for i in range(header[12])]
    for section in sections:
      if section[1] != 2:
        continue
      strings = sections[section[6]]
      strings = data[strings[4]:strings[4]+strings[5]]
      for offset in range(section[4],section[4]+section[5],section[9]):
        name,value,_,_,_,_ = struct.unpack_from("<IIIBBH",data,offset)
        end = strings.find(b"\0",name)
        self.symbols[strings[name:end].decode()] = value
    self.sha256 = hashlib.sha256(data).hexdigest()

  def read(self, address, size):
    try:
      return bytes(self.memory[address+i] for i in range(size))
    except KeyError as error:
      raise ValueError(f"ELF accesses unmapped memory at 0x{address:x}") from error

  def write(self, address, data):
    for idx,value in enumerate(data):
      self.memory[address+idx] = value


@dataclass
class ELFWorkload:
  instructions: list
  pcs: list
  scalar_outputs: list
  image: ELFImage
  roi: tuple
  matrices: int
  macs: int
  addresses: list


def execute_elf(path, max_instructions=2000000):
  image = ELFImage(path)
  state = MatrixTraceState()
  state.regs = [0]*32
  vectors = bytearray(32*16)
  tiles = {tile:[[0.0]*16 for _ in range(16)] for tile in (0,4,8,12)}
  pc, instructions, pcs, outputs, addresses = image.entry, [], [], [], []
  matrices = macs = 0
  start = image.symbols.get("roi_start",image.entry)
  end = image.symbols.get("roi_end")
  roi_start = roi_end = None
  for _ in range(max_instructions):
    bits = int.from_bytes(image.read(pc,4),"little")
    if bits == 0x08000073:
      instructions.append(state.decode(pc,bits,pc+4,decode))
      pcs.append(pc)
      addresses.append(None)
      break
    op,rd,f3 = bits&127,(bits>>7)&31,(bits>>12)&7
    rs1,rs2,f7,f6 = (bits>>15)&31,(bits>>20)&31,bits>>25,bits>>26
    left,right = state.regs[rs1],state.regs[rs2]
    next_pc,value,address = pc+4,None,None
    if op == 0x37:
      value = bits&0xfffff000
    elif op == 0x17:
      value = pc+(bits&0xfffff000)
    elif op == 0x13:
      imm = signed(bits>>20,12)
      if f3 == 0: value = left+imm
      elif f3 == 1 and f7 == 0: value = left<<(rs2&31)
      elif f3 == 4: value = left^imm
      elif f3 == 6: value = left|imm
      elif f3 == 7: value = left&imm
      elif f3 == 5 and f7 in (0,0x20):
        value = (signed(left,32) if f7==0x20 else left)>>(rs2&31)
      else: raise ValueError("unimplemented immediate arithmetic")
    elif op == 0x33:
      if f7 == 1 and f3 == 0: value = left*right
      elif f3 == 0 and f7 in (0,0x20): value = left-right if f7 else left+right
      elif f3 == 4 and f7 == 0: value = left^right
      elif f3 == 6 and f7 == 0: value = left|right
      elif f3 == 7 and f7 == 0: value = left&right
      else: raise ValueError("unimplemented register arithmetic")
    elif op == 0x63:
      imm = ((bits>>31)<<12)|(((bits>>7)&1)<<11)|(((bits>>25)&63)<<5)|(((bits>>8)&15)<<1)
      conditions = {0:left==right,1:left!=right,4:signed(left,32)<signed(right,32),
                    5:signed(left,32)>=signed(right,32),6:left<right,7:left>=right}
      if f3 not in conditions: raise ValueError("unsupported branch")
      if conditions[f3]: next_pc = pc+signed(imm,13)
    elif op == 0x6f:
      imm = ((bits>>31)<<20)|(((bits>>12)&255)<<12)|(((bits>>20)&1)<<11)|(((bits>>21)&1023)<<1)
      value,next_pc = pc+4,pc+signed(imm,21)
    elif op == 0x67 and f3 == 0:
      value,next_pc = pc+4,(left+signed(bits>>20,12))&~1
    elif op == 0x03:
      width = {0:1,1:2,2:4,4:1,5:2}.get(f3)
      if width is None: raise ValueError("unsupported load")
      address = (left+signed(bits>>20,12))&0xffffffff
      value = int.from_bytes(image.read(address,width),"little",signed=f3<4)
    elif op == 0x23:
      width = {0:1,1:2,2:4}.get(f3)
      if width is None: raise ValueError("unsupported store")
      imm = ((bits>>25)<<5)|((bits>>7)&31)
      address = (left+signed(imm,12))&0xffffffff
      image.write(address,(right&((1<<(8*width))-1)).to_bytes(width,"little"))
    elif op in (0x07,0x27) and f3 in (0,5,6):
      if bits>>28 or (bits>>26)&3 or not (bits>>25)&1 or rs2:
        raise ValueError("only unmasked unit-stride vector memory is supported")
      width = {0:1,5:2,6:4}[f3]
      size,address = state.vl*width,left
      if rd*16+size > len(vectors): raise ValueError("vector group overflows")
      if op == 0x07: vectors[rd*16:rd*16+size] = image.read(address,size)
      else: image.write(address,vectors[rd*16:rd*16+size])
    elif op == 0x57 and f3 == 7:
      pass  # MatrixTraceState performs the configuration operation below.
    elif op == 0x57 and f3 == 6 and f6 == 0x10 and rs2 == 30:
      tiles[rd>>1] = [[0.0]*16 for _ in range(16)]
    elif op == 0x57 and f3 == 6 and f6 == 0x10 and rs2 == 31:
      tile,row = left>>27,left&15
      data = b"".join(struct.pack("<f",x) for x in tiles[tile][row])
      vectors[rd*16:rd*16+len(data)] = data
    elif op == 0x77:
      # Decode and execute the same packed arithmetic as the instruction.
      command = state.decode(pc,bits,next_pc,decode).matrix
      if command.dtype not in ("bf16","fp32"):
        raise ValueError("ELF functional frontend currently supports BF16/FP32 matrices")
      def operand(base,k):
        stride = 4 if command.dtype=="bf16" else 1
        offset = (base+k*stride)*16
        raw = vectors[offset:offset+(32 if command.dtype=="bf16" else 64)]
        if command.dtype=="bf16":
          return [struct.unpack("<f",b"\0\0"+raw[i:i+2])[0] for i in range(0,32,2)]
        return list(struct.unpack("<16f",raw))
      for inner in range(command.k):
        a,b = operand(rs2,inner),operand(rs1,inner)
        for row in range(command.m):
          for col in range(command.n):
            tiles[command.tile][row][col] = struct.unpack("<f",struct.pack("<f",
                tiles[command.tile][row][col]+a[row]*b[col]))[0]
      matrices += 1
      macs += command.macs
    elif op == 0x73 and f3 in (1,2,3,5,6,7):
      if rd: raise ValueError("CSR reads are not implemented")
      if bits>>20 != 0x300:
        raise ValueError("only mstatus CSR setup is supported")
    elif op == 0x0f and f3 in (0,1):
      pass
    else:
      raise ValueError(f"ELF frontend cannot execute 0x{bits:08x} at 0x{pc:x}")
    instruction = state.decode(pc,bits,next_pc,decode)
    if address is not None:
      # The workload record retains addresses separately from timing metadata.
      instruction = replace(instruction,
          external=not (0x10000 <= address and address+instruction.bytes <= 0x18000))
    if value is not None and rd:
      state.regs[rd] = value&0xffffffff
      if rd == 31: outputs.append(state.regs[rd])
    state.regs[0] = 0
    if pc == start and roi_start is None: roi_start = len(instructions)
    instructions.append(instruction)
    pcs.append(pc)
    addresses.append(address)
    if pc == end: roi_end = len(instructions)
    pc = next_pc&0xffffffff
  else:
    raise ValueError("ELF did not halt within the instruction budget")
  if roi_start is None or roi_end is None:
    raise ValueError("ELF must contain executable roi_start and roi_end symbols")
  return ELFWorkload(instructions,pcs,outputs,image,(roi_start,roi_end),matrices,macs,addresses)
