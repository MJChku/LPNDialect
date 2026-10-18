"""Bounded ACT row availability for an optional quantizer-to-MXU streaming link."""
from .model import quant_scale_setup


def quant_row_first_pops(rows,chunks,capacity,banks,mul_latency=2,add_latency=4,scale_lookup=False):
  """Finite FIFO/bank schedule, before output conversion and ACT RAM landing.

  A scale follows a complete buffered row. Its pop stream has no TMEM stalls
  once it starts, as required by the streaming RTL writer.
  """
  if capacity<chunks or banks<2 or rows<1 or chunks<1:
    raise ValueError('streaming requires a complete row in the finite buffer')
  popped,ends,firsts=[],[],[]
  read=0
  for row in range(rows):
    bank_free=ends[row-banks]+1 if row>=banks else 1
    for chunk in range(chunks):
      index=row*chunks+chunk
      free=popped[index-capacity]+1 if index>=capacity else 1
      read=max(read+1,free,bank_free)
    first=max(read+quant_scale_setup(mul_latency,add_latency,scale_lookup),popped[-1]+2 if popped else 1)
    firsts.append(first)
    popped.extend(range(first,first+chunks))
    ends.append(popped[-1])
  return firsts


def stream_producers(commands,I,W,timing):
  """Use row progress only for a fully covering, latest, ACT-only QACT writer.

  Partial coverage retains command completion dependencies conservatively. No
  completion ancestry may be inferred from these partial progress edges.
  """
  writers=[]
  result=[-1]*len(commands)
  for idx,c in enumerate(commands):
    ins=c.instruction
    if ins.op==I.MM and not c.footprint.all:
      act=[r for r in c.footprint.rd if r[0]==W.SP_ACT]
      prior=next((j for j in reversed(writers) if any(
          lo<whi and wlo<hi for _,lo,hi in act
          for sp,wlo,whi in commands[j].footprint.wr if sp==W.SP_ACT)),None)
      if prior is not None:
        p=commands[prior];q=p.instruction
        if q.op==I.QACT and not p.footprint.all:
          qw,mw=q.w,ins.w
          rows=qw[1]&255;ab=(qw[1]>>8)&255;kb=qw[1]>>16
          mr=(mw[5]>>16)&255;mb=(mw[5]>>24)&255;mk=(mw[3]>>16)&255
          written=rows*(2 if q.flags&I.F_DUP else 1)
          # Check every non-ACT space using the same independent footprint
          # conflict routine, including MM output vs QACT input WAR.
          fp=type(p.footprint)(rd=tuple(r for r in p.footprint.rd if r[0]!=W.SP_ACT),
                              wr=tuple(r for r in p.footprint.wr if r[0]!=W.SP_ACT),all=False)
          if (rows and kb and kb<=timing.quant_buffer_chunks and mr<=written
              and ab<=mb and mb+mk<=ab+kb and not W.conflict(c.footprint,fp)):
            result[idx]=prior
    if any(r[0]==W.SP_ACT for r in c.footprint.wr):writers.append(idx)
  return result


class ActivationStream:
  """One quantizer's current producer and bounded row-first availability events.

  There are no per-command places or per-clock events. Completed producers use
  the existing completion flags; only the active producer needs a progress latch.
  """
  def __init__(self,net,prefix,commands,I,timing,table,is_done):
    offsets,delays,rows,dup,tail=[0],[],[],[],[]
    for c in commands:
      ins=c.instruction
      n=ins.w[1]&255 if ins.op==I.QACT else 0
      kb=ins.w[1]>>16 if ins.op==I.QACT else 0
      buffered=bool(n and kb and kb<=timing.quant_buffer_chunks)
      rows.append(n if buffered else 0)
      dup.append(int(bool(ins.flags&I.F_DUP)) if buffered else 0)
      tail.append(2 if buffered else 0)
      if buffered:
        firsts=quant_row_first_pops(n,kb,timing.quant_buffer_chunks,timing.quant_row_banks,
                                   timing.fp_mul_latency,timing.fp_add_latency,timing.quant_scale_lookup)
        # Output conversion plus three ACT RAM write registers. A synchronous
        # reader can consume the block on the following edge.
        delays.extend(f+timing.fp_mul_latency+10 for f in firsts)
      offsets.append(len(delays))
    offsets=table(prefix+'_offsets',offsets)
    delays=table(prefix+'_delays',delays or [0])
    rows=table(prefix+'_rows',rows)
    dup=table(prefix+'_dup',dup)
    self.tail=table(prefix+'_tail',tail)
    self.state=state=net.place(prefix+'_state',initial_tokens=1)
    self.arrivals=arrivals=net.place(prefix+'_arrivals')

    @net.func(name=prefix+'_start',args=('i64',),results=())
    def start(idx):
      if array(rows)[idx]>0:
        old=take(state)
        emit(state,old.set('id',idx).set('visible',0))
        first=array(offsets)[idx]
        for r in range(array(rows)[idx]):
          emit(arrivals,create().set('id',idx).set('visible',r+1),
               delay=sitofp(array(delays)[first+r]))
      func_return()
    self.start=start

    @net.func(name=prefix+'_can_read',args=('i64','i64'),results=('i1',))
    def can_read(producer,row):
      # Negative means this command retains ordinary completion dependencies.
      safe_idx=select(producer>=0,producer,0)
      logical_row=select(array(dup)[safe_idx]>0,remi(row,select(array(rows)[safe_idx]>0,array(rows)[safe_idx],1)),row)
      current=peek(state)
      partial=select(current.get('id')==producer,current.get('visible')>logical_row,False,typ='i1')
      func_return(select(producer<0,True,select(is_done(safe_idx),True,partial,typ='i1'),typ='i1'))
    self.can_read=can_read

    @net.transition(prefix+'_arrive')
    def arrive():
      guard(count(arrivals)>0)
      token=take(arrivals)
      old=take(state)
      # A late block from the previous command may land after the quantizer
      # accepts its next command. It cannot replace that new command's state.
      visible=select(old.get('id')==token.get('id'),token.get('visible'),old.get('visible'))
      emit(state,old.set('visible',visible))
