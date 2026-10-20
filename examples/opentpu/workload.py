"""Read exactly the instruction/memory files used by openTPU's RTL harness."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import subprocess
from types import MethodType

from . import REVISION


def use_checkout(root):
  root = Path(root).resolve()
  if not (root/"opentpu/isa.py").is_file():
    raise ValueError(f"not a FeSens/openTPU checkout: {root}")
  revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
  if revision!=REVISION:
    raise ValueError(f"check out pinned FeSens revision {REVISION}")
  sys.path.insert(0,str(root))
  import opentpu
  if not Path(opentpu.__file__).resolve().is_relative_to(root):
    raise ValueError("another opentpu package was already imported")
  return root


@dataclass
class Command:
  pc: int
  instruction: object
  footprint: object


@dataclass
class Workload:
  config: object
  programs: list
  images: list
  commands: list
  machine: object
  hashes: dict
  args: list


def load(directory):
  import numpy as np
  from opentpu import Config, isa as I, waits as W
  from opentpu.isasim import Machine

  directory = Path(directory)
  metadata = json.loads((directory/"input.json").read_text())
  if metadata["revision"] != REVISION:
    raise ValueError("unsupported openTPU revision")
  cfg = Config(**metadata["config"])
  programs,images,hashes = [],[],{}
  for s in range(cfg.S):
    program,memory = directory/f"prog_{s}.hex",directory/f"dram_{s}.bin"
    words = [int(x,16) for x in program.read_text().split()]
    if any(not 0<=word<=0xffffffff for word in words):
      raise ValueError("instruction words must fit in 32 bits")
    count = metadata.get("instruction_counts",[])
    length = count[s]*8 if count else ((len(words)+7)//8)*8
    if not words or not length-8<len(words)<=length:
      raise ValueError("instruction file length differs from its manifest")
    words.extend([0]*(length-len(words)))
    programs.append([I.Instr.decode(words[i:i+8]) for i in range(0,len(words),8)])
    if memory.stat().st_size%4 or memory.stat().st_size>cfg.DRAM_BYTES:
      raise ValueError("DRAM image must contain aligned words and fit the configured memory")
    # $fread loads big-endian words; the simulated memory itself is little-endian.
    images.append(np.fromfile(memory,dtype=">u4").astype("<u4").view(np.uint8))
    for path in (program,memory):
      hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
  args = metadata.get("args",[])
  return execute(cfg,programs,images,args,hashes)


def execute(cfg,programs,images,args=None,hashes=None):
  """Numerical execution and concrete control/address trace from shared inputs."""
  from opentpu.isasim import Machine
  args = list(args or [])
  machine = Machine(cfg,programs,images,args=args)
  return _record_execution(machine,programs,images,args,hashes)


def execute_on_machine(machine,programs,args=None,hashes=None):
  """Launch commands on resident DRAM/TMEM/ACT, using the real Machine.load ABI.

  No memory image is copied or serialized. The result borrows the machine:
  inspect its state before the next launch. Command/footprint records remain
  independent; images=[] explicitly indicates that no input snapshot exists.
  """
  args=list(args or [])
  machine.load(programs,args=args)
  return _record_execution(machine,programs,[],args,hashes)


def _record_execution(machine,programs,images,args,hashes):
  from opentpu import isa as I,waits as W
  cfg=machine.cfg
  commands = [[] for _ in range(cfg.S)]

  def record_step(self):
    ins = self.prog[self.pc]
    if ins.op in (I.WAITW,I.STREAM,I.DSTEP) or (ins.op==I.HALT and ins.flags):
      raise ValueError(f"{I.OPNAMES[ins.op]} is not supported by the timing frontend yet")
    regs = [(self.reg(r),)*2 for r in range(16)]
    resolved = I.Instr(ins.op,flags=ins.flags,w=[v[0] for v in W.resolve(ins,regs)])
    commands[self.sid].append(Command(self.pc,resolved,W.footprint(ins,regs,cfg.D,cfg.S)))
    original[self.sid]()

  original = [s.step for s in machine.slices]
  overridden = [('step' in s.__dict__,s.__dict__.get('step')) for s in machine.slices]
  for s in machine.slices:
    s.step = MethodType(record_step,s)
  try:
    machine.run(max_steps=1<<40)
  finally:
    for s,(had,value) in zip(machine.slices,overridden):
      if had:s.step=value
      else:del s.step
  return Workload(cfg,programs,images,commands,machine,hashes or {},args)
