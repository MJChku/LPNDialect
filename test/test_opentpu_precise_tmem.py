"""Compare proposed dependency ranges to actual independent ISA memory accesses."""
import argparse
from pathlib import Path
import unittest
import tempfile
from dataclasses import replace

from examples.opentpu.precise_tmem import footprint,refine
from examples.opentpu.workload import Command,use_checkout


class PreciseTMEMTest(unittest.TestCase):
  tool=None
  def test_ranges_match_actual_isa_accesses(self):
    import numpy as np
    from opentpu import Config,isa as I,waits as W
    from opentpu.isasim import Slice
    cfg=Config(D=64,MCOLS=8,ACT_ROWS=8,TMEM_WORDS=4096,DRAM_BYTES=65536)
    cases=[]
    for f in I.VFUNCS:
      if f==I.V_OUTER:continue
      for bmode in [I.B_FULL,I.B_ROW,I.B_COL,I.B_SCALAR]:
        cases.append(I.vop(f,2048,0,768,3,7,16,32,19,bmode,imm=1))
    for flags in range(4):
      cases.append(I.qact(0,3,0,1,80,cscale=768 if flags&1 else None,rscale=1000 if flags&2 else None))
    cases.append(I.qst(0,0,32768,3,1,80,64,1))
    for acc,rmax,ascale in [(False,False,None),(True,False,None),(True,True,768)]:
      cases.append(I.mm(0,0,2048,5,1,64,16,3,0,4,unit=True,acc=acc,rmax=rmax,ascale=ascale))
    outer=I.vop(I.V_OUTER,2048,0,768,3,7,16,0,1,I.B_ROW)
    outer.w[6]=1024
    for flags in [0,I.F_DONE,I.F_DSCALAR,I.F_DONE|I.F_DSCALAR]:
      cases.append(I.Instr(outer.op,flags=flags,w=list(outer.w)))
    for ins in cases:
      with self.subTest(op=ins.op,flags=ins.flags,w=ins.w):
        machine=Slice(cfg,0,[ins],np.zeros(cfg.DRAM_BYTES,np.uint8))
        machine.tmem[:]=np.float32(1).view(np.uint32);machine.ascale[:]=1
        actual=[set(),set()]
        get,put=machine.tget,machine.tput
        def capture_get(idx):actual[0].update(np.asarray(idx).reshape(-1).tolist());return get(idx)
        def capture_put(idx,values):actual[1].update(np.asarray(idx).reshape(-1).tolist());return put(idx,values)
        machine.tget,machine.tput=capture_get,capture_put
        old=W.footprint(ins,[(0,0)]*16,cfg.D,cfg.S)
        proposed=footprint(Command(0,ins,old),cfg,I,W)
        machine.step()
        for ranges,observed in zip([proposed.rd,proposed.wr],actual):
          addresses={a for space,lo,hi in ranges if space==W.SP_TMEM for a in range(lo,hi)}
          self.assertEqual(addresses,observed)

  def test_disjoint_rows_overlap_hulls_but_true_hazards_remain(self):
    from opentpu import Config,isa as I,waits as W
    cfg=Config(D=64,MCOLS=8,ACT_ROWS=8)
    a=I.vop(I.V_FILL,0,0,0,3,8,32,0,0,I.B_SCALAR)
    def fp(ins):return footprint(Command(0,ins,W.footprint(ins,[(0,0)]*16,cfg.D,cfg.S)),cfg,I,W)
    b=I.vop(I.V_COPY,2048,8,0,3,8,32,32,0,I.B_SCALAR)
    self.assertTrue(W.conflict(W.footprint(a,[(0,0)]*16,cfg.D),W.footprint(b,[(0,0)]*16,cfg.D)))
    self.assertFalse(W.conflict(fp(a),fp(b)))
    for function,source,dest in [(I.V_COPY,0,2048),(I.V_COPY,2048,0),(I.V_FILL,0,0)]:
      c=I.vop(function,dest,source,0,3,8,32,32,0,I.B_SCALAR)
      self.assertTrue(W.conflict(fp(a),fp(c)))

  def test_precise_dependencies_release_disjoint_work_and_native_agrees(self):
    from opentpu import Config,isa as I
    from examples.opentpu.shape_workload import resolve_static
    from examples.opentpu.compact import CompactOpenTPU
    from examples.opentpu.model import Timing
    from examples.opentpu.native_timing import ReusableTiming
    cfg=Config(D=64,LANES=64)
    for dependent in (False,True):
      program=[I.vop(I.V_FILL,0,0,0,3,8,32,0,0,I.B_SCALAR),
               I.vop(I.V_COPY,1024,0 if dependent else 8,0,3,8,32,32,0,I.B_SCALAR),I.halt()]
      original=resolve_static(cfg,[program]);timing=Timing(dram_bytes_per_cycle=64)
      ticks=[]
      for work in [original,refine(original)]:
        sim=CompactOpenTPU(work,timing).net.python_simulator()
        value=sim.run(max_time=10000);ticks.append(value)
        self.assertEqual(sim.place_dicts('program_completed'),[{}])
        if self.tool:
          with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=ReusableTiming(self.tool,root/'cache',capacity=128).prepare(work,timing,root/'tables')
            self.assertEqual(p.run()[0],value)
      if dependent:self.assertEqual(*ticks)
      else:self.assertLess(ticks[1],ticks[0])


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True)
  p.add_argument('--tool',type=Path);a=p.parse_args();use_checkout(a.opentpu_root);PreciseTMEMTest.tool=a.tool;unittest.main(argv=[__file__],verbosity=2)
