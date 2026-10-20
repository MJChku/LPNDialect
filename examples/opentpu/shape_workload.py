"""Weight-free static command traces from the upstream dense Qwen compiler.

Only scalar address/control instructions execute. Tensor operations contribute
commands and memory footprints, never numerical arrays or matrix arithmetic.
Data-dependent control is rejected instead of inventing values or dependencies.
"""

import hashlib

from .workload import Command, Workload


def qwen17_spec():
  from opentpu.llm.qwen3 import Spec
  return Spec(hidden=2048,layers=28,n_q=16,n_kv=8,head_dim=128,
              ffn=6144,vocab=151936)


def resolve_static(cfg,programs,args=None,max_steps=1_000_000):
  """Resolve static loops/register addresses without allocating device memory."""
  from opentpu import isa as I, waits as W
  args=list(args or [])
  if len(args)>8 or len(programs)!=cfg.S or max_steps<1:
    raise ValueError('invalid static trace configuration')
  commands=[]
  allowed={I.NOP,I.LI,I.ADDI,I.LOOP,I.BAR,I.GATHER,I.LD,I.ST,I.MM,I.QACT,I.QST,I.VOP,I.HALT}
  for program in programs:
    regs=[0]*16
    regs[8:8+len(args)]=[int(x)&0xffffffff for x in args]
    pc,stack,trace=0,[],[]
    while True:
      if not 0<=pc<len(program):
        raise ValueError('static program counter is out of range')
      if len(trace)>=max_steps:
        raise ValueError('static trace exceeds instruction limit')
      ins=program[pc]
      if ins.op not in allowed or (ins.op==I.HALT and ins.flags):
        raise ValueError(f'{I.OPNAMES.get(ins.op,ins.op)} needs numerical/control data or is unsupported')
      values=[(x,x) for x in regs]
      resolved=I.Instr(ins.op,flags=ins.flags,w=[x[0] for x in W.resolve(ins,values)])
      fp=W.footprint(ins,values,cfg.D,cfg.S)
      for space,lo,hi in fp.rd+fp.wr:
        limit={W.SP_DRAM:cfg.DRAM_BYTES,W.SP_TMEM:cfg.TMEM_WORDS}.get(space)
        if limit is not None and not 0<=lo<=hi<=limit:
          raise ValueError(f'static footprint outside memory at pc {pc}')
      trace.append(Command(pc,resolved,fp))
      if ins.op==I.HALT:
        break
      if ins.op==I.LI and ins.rd:
        regs[ins.rd]=ins.w[0]&0xffffffff
      elif ins.op==I.ADDI and ins.rd:
        regs[ins.rd]=(regs[ins.ra]+ins.w[0])&0xffffffff
      elif ins.op==I.LOOP:
        body,count=ins.w[0],(regs[ins.ra]+ins.w[1])&0xffffffff
        end=pc+body
        if body<1 or end>=len(program) or any(x[1]==end for x in stack):
          raise ValueError('invalid static loop body')
        if not count:
          pc+=1+body
          continue
        if len(stack)>=4:
          raise ValueError('static loop stack overflow')
        stack.append([pc+1,end,count])
        pc+=1
        continue
      if stack and stack[-1][1]==pc:
        if stack[-1][2]>1:
          stack[-1][2]-=1
          pc=stack[-1][0]
          continue
        stack.pop()
      pc+=1
    commands.append(trace)
  return Workload(cfg,programs,[],commands,None,{},args)


def compile_shape(spec,position,cap=128,weight_format='int8'):
  from opentpu.llm.qwen3 import Image, device_config, PREFILL_ROWS
  if not 0<=position<cap:
    raise ValueError('position must fit KV capacity')
  cfg=device_config(spec,cap,S=1,D=128,MCOLS=4,ACT_ROWS=16,
                    ACT_BLOCKS=64,LANES=8,TMEM_WORDS=65536,IMEM_WORDS=32768,
                    wformat=weight_format)
  image=Image(spec,cfg,cap,rows=PREFILL_ROWS,wformat=weight_format)
  if image.nbytes>1<<32:
    raise ValueError('whole-image trace exceeds 32-bit addresses; use layer streaming or multiple slices')
  programs=image.compile_step(position)
  return resolve_static(cfg,programs),image


def program_hash(program):
  """Match the RTL hex encoding using only the small instruction words."""
  from opentpu import isa as I
  words=I.assemble(program)
  end=next((i+1 for i in range(len(words)-1,-1,-1) if words[i]),1)
  data='\n'.join(f'{int(w):08x}' for w in words[:end])+'\n'
  return hashlib.sha256(data.encode()).hexdigest()
