"""Native functional execution for FeSens scalar/control and LD/ST instructions.

This is the first port to the typed functional frontend. Matrix, quantizer and
vector kernels deliberately fail until their bit-exact arithmetic is ported.
The long trained-model baseline continues to use the independent ISA model.
"""

from types import SimpleNamespace

from lpnfunc import Buffer,functional,i64,u32,boolean
from .workload import Command,Workload


@functional
def scalar_isa(program:Buffer[u32],dram:Buffer[u32],tmem:Buffer[u32],
               regs:Buffer[u32],stack:Buffer[i64],trace:Buffer[u32],max_steps:i64)->i64:
  pc:i64 = 0
  steps:i64 = 0
  depth:i64 = 0
  status:i64 = 0
  halted:boolean = False
  regs[0] = 0
  while status==0 and not halted and steps<max_steps:
    if pc<0 or pc>=len(program)//8:
      status = -1
    elif steps>=len(trace)//9:
      status = -5
    else:
      word:u32 = program[pc*8]
      op:u32 = word&255
      ra:i64 = i64((word>>8)&15)
      rb:i64 = i64((word>>12)&15)
      rc:i64 = i64((word>>16)&15)
      rd:i64 = i64((word>>20)&15)
      flags:u32 = word>>24
      wa:u32 = u32(program[pc*8+1]+regs[ra])
      wb:u32 = u32(program[pc*8+2]+regs[rb])
      wc:u32 = program[pc*8+3]
      if op!=0x10 and op!=0x11:
        wc = u32(wc+regs[rc])
      trace[steps*9] = u32(pc)
      trace[steps*9+1] = word
      trace[steps*9+2] = wa
      trace[steps*9+3] = wb
      trace[steps*9+4] = wc
      for j in range(4):
        trace[steps*9+5+j] = program[pc*8+4+j]
      steps += 1
      jump:boolean = False
      if op==0x01:
        halted = True
      elif op==0x02:
        if rd!=0:
          regs[rd] = program[pc*8+1]
      elif op==0x03:
        if rd!=0:
          regs[rd] = wa
      elif op==0x04:
        body:i64 = i64(program[pc*8+1])
        count:u32 = u32(regs[ra]+program[pc*8+2])
        end:i64 = pc+body
        if body<1:
          status = -3
        else:
          for frame in range(depth):
            if stack[frame*3+1]==end:
              status = -2
          if status==0:
            if count==0:
              pc += 1+body
              jump = True
            elif depth>=4:
              status = -2
            else:
              stack[depth*3] = pc+1
              stack[depth*3+1] = end
              stack[depth*3+2] = i64(count)
              depth += 1
              pc += 1
              jump = True
      elif op==0x06:
        if i64(wa)>=len(tmem):
          status = -4
        else:
          value:u32 = tmem[i64(wa)]
          if (flags&1)==0:
            bits:u32 = value
            exponent:u32 = (bits>>23)&255
            if exponent<127:
              value = 0
            elif exponent>=158:
              value = u32(0x80000000)
            else:
              mantissa:u32 = (bits&0x7fffff)|0x800000
              if exponent>=150:
                value = u32(mantissa<<(exponent-150))
              else:
                value = mantissa>>(150-exponent)
              if (bits>>31)!=0:
                value = u32(0-value)
          if (flags&2)!=0:
            value = u32(value*wb)
          if (flags&4)!=0:
            if value<program[pc*8+3] or value>program[pc*8+4]:
              status = -6
          if status==0 and rd!=0:
            regs[rd] = value
      elif op==0x10 or op==0x11:
        n:i64 = i64(program[pc*8+3])
        d:i64 = i64(wa>>2)
        t:i64 = i64(wb)
        if n>0:
          if (wa&3)!=0 or d+n>len(dram) or t+n>len(tmem):
            status = -4
          else:
            for element in range(n):
              if op==0x10:
                tmem[t+element] = dram[d+element]
              else:
                dram[d+element] = tmem[t+element]
      elif op!=0x00 and op!=0x05:
        status = -8
      if status==0 and not halted and not jump:
        if depth>0 and stack[(depth-1)*3+1]==pc:
          if stack[(depth-1)*3+2]>1:
            stack[(depth-1)*3+2] -= 1
            pc = stack[(depth-1)*3]
          else:
            depth -= 1
            pc += 1
        else:
          pc += 1
  if status==0 and not halted:
    status = -7
  return steps if status==0 else status


_kernels = {}


def execute_scalar(cfg,programs,images,args=None,hashes=None,cache_dir=None,
                   max_steps=1_000_000):
  """Execute supported instructions natively and adapt their resolved trace.

  The shared-library kernel has no Python instruction callbacks. Converting
  its packed records into existing LPN Command objects still runs in Python.
  """
  import numpy as np
  from opentpu import isa as I,waits as W
  if cfg.S!=1 or len(programs)!=1 or len(images)!=1:
    raise ValueError("native scalar frontend currently supports one slice")
  if not 0<max_steps<=100_000_000:
    raise ValueError("max_steps must be 1..100,000,000")
  args = list(args or [])
  if len(args)>8:
    raise ValueError("at most eight initial argument registers")
  allowed = {I.NOP,I.HALT,I.LI,I.ADDI,I.LOOP,I.BAR,I.RLD,I.LD,I.ST}
  for ins in programs[0]:
    if ins.op not in allowed or ins.op==I.HALT and ins.flags:
      raise ValueError(f"{I.OPNAMES.get(ins.op,ins.op)} is not ported to the native scalar frontend")
  if len(programs[0])*8>cfg.IMEM_WORDS:
    raise ValueError("program exceeds IMEM")
  image = np.asarray(images[0],np.uint8).reshape(-1)
  if len(image)>cfg.DRAM_BYTES or cfg.DRAM_BYTES%4:
    raise ValueError("image must fit word-aligned DRAM")
  dram = np.zeros(cfg.DRAM_BYTES,np.uint8)
  dram[:len(image)] = image
  tmem = np.zeros(cfg.TMEM_WORDS,np.uint32)
  regs = np.zeros(16,np.uint32)
  regs[8:8+len(args)] = [int(v)&0xffffffff for v in args]
  stack = np.zeros(12,np.int64)
  trace = np.empty((max_steps,9),np.uint32)
  key = str(cache_dir)
  if key not in _kernels:
    _kernels[key] = scalar_isa.compile(cache_dir)
  kernel = _kernels[key]
  count = kernel(I.assemble(programs[0]),dram.view(np.uint32),tmem,regs,stack,
                 trace.reshape(-1),max_steps)
  errors = {-1:"pc past program",-2:"invalid/overflowing loop stack",-3:"empty loop body",
            -4:"invalid memory address",-5:"trace capacity exceeded",-6:"RLD bound failed",
            -7:"step limit reached",-8:"unsupported opcode"}
  if count<0:
    raise ValueError("native scalar frontend: "+errors.get(count,str(count)))
  commands = []
  zero_regs = [(0,0)]*16
  for record in trace[:count]:
    word = int(record[1])
    ins = I.Instr(word&255,flags=word>>24,w=record[2:].tolist())
    commands.append(Command(int(record[0]),ins,W.footprint(ins,zero_regs,cfg.D,cfg.S)))
  machine = SimpleNamespace(slices=[SimpleNamespace(dram=dram,tmem=tmem,R=regs.tolist(),
      icount=int(count),halted=True)],native_kernel=kernel)
  return Workload(cfg,programs,images,[commands],machine,hashes or {},args)
