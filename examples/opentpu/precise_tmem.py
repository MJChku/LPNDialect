"""Proposed exact TMEM row ranges for a sequencer with strided dependency checks.

Original RTL keeps bounding intervals. This opt-in architecture proposal changes
only dependency metadata; instructions and numerical operations are unchanged.
DRAM beats and ACT ranges retain the original conservative rules.
"""
from dataclasses import replace


def merged(ranges):
    result=[]
    for space,lo,hi in sorted(ranges):
        if hi<=lo:continue
        if result and result[-1][0]==space and lo<=result[-1][2]:
            result[-1]=(space,result[-1][1],max(hi,result[-1][2]))
        else:result.append((space,lo,hi))
    return tuple(result)


def footprint(command,cfg,I,W):
    old=command.footprint;ins=command.instruction;w=ins.w;flags=ins.flags
    if old.all or ins.op not in (I.MM,I.VOP,I.QACT,I.QST):return old
    rd=[];wr=[]
    def rows(target,base,count,width,stride):
        if not count or not width:return
        if base<0 or base+(count-1)*stride+width>cfg.TMEM_WORDS:
            raise ValueError('strided TMEM footprint out of bounds')
        target.extend((W.SP_TMEM,base+r*stride,base+r*stride+width) for r in range(count))
    if ins.op==I.MM:
        n,kb,m=w[3]&65535,(w[3]>>16)&255,(w[5]>>16)&255
        if n and kb and m:
            rows(wr,w[2],m,n,w[5]&65535)
            if flags&I.F_ACC:rows(rd,w[2],m,n,w[5]&65535)
            if flags&I.F_RMAX:rows(wr,w[2]+m*(w[5]&65535),1,m,0)
            if flags&I.F_ASCALE:rows(rd,w[1],1,m,0)
    elif ins.op==I.QACT:
        n,kb=w[1]&255,w[1]>>16
        if n and kb:
            rows(rd,w[0],n,kb*cfg.D,w[2])
            if flags&I.F_CSCALE:rows(rd,w[3],1,kb*cfg.D,0)
            if flags&I.F_RSCALE:rows(rd,w[4],1,n,0)
    elif ins.op==I.QST:
        n,kb=w[3]&65535,w[3]>>16
        if n and kb:rows(rd,w[0],n,kb*cfg.D,w[4])
    else:
        n,k=w[3]&65535,w[3]>>16
        f,bmode=(w[5]>>16)&255,(w[5]>>24)&3
        if n and k:
            width=2 if f==I.V_ARGMAX else 1 if f in I.REDUCE else k
            rows(wr,w[0],n,width,w[4]&65535)
            if f==I.V_OUTER:
                rows(rd,w[0],n,k,w[4]&65535)
                if not flags&I.F_DONE:rows(rd,w[1],1,1 if flags&I.F_DSCALAR else k,0)
                rows(rd,w[6],1,k,0)
            elif f!=I.V_FILL:rows(rd,w[1],n,k,w[4]>>16)
            if f in I.READS_B or f==I.V_OUTER:
                if bmode==I.B_FULL:rows(rd,w[2],n,k,w[5]&65535)
                elif bmode==I.B_ROW:rows(rd,w[2],n,1,w[5]&65535)
                elif bmode==I.B_COL:rows(rd,w[2],1,k,0)
    result=replace(old,rd=merged([r for r in old.rd if r[0]!=W.SP_TMEM]+rd),
                       wr=merged([r for r in old.wr if r[0]!=W.SP_TMEM]+wr))
    # RMW reads may have been represented only by a write in the old contract.
    # The refined sets must stay inside the corresponding original access hulls.
    for ranges,original in [(result.rd,old.rd+old.wr),(result.wr,old.wr)]:
        union=merged(original)
        for space,lo,hi in ranges:
            if not any(sp==space and a<=lo and hi<=b for sp,a,b in union):
                raise ValueError('refinement escaped original footprint')
    return result


def refine(workload):
    from opentpu import isa as I,waits as W
    return replace(workload,commands=[[replace(c,footprint=footprint(c,workload.config,I,W))
                                      for c in commands] for commands in workload.commands])
