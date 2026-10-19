"""Bounded private scale-cache metadata derived from the original instructions.

The request sequence is exact for int8 parallel row engines. Cache state follows
canonical command order; overlaps with asynchronous invalidations are an explicit
approximation, checked against RTL rather than supplied by RTL counters.
"""
from math import ceil


def scale_addresses(ins,I,row,parallel):
    if ins.flags&I.F_UNIT:return []
    if ((ins.flags>>I.WF_SHIFT)&3)!=I.W8:
        raise ValueError('scale-cache timing currently supports int8 weights')
    w=ins.w;kb=(w[3]>>16)&255;k0=w[3]>>24;base=w[1]+row*w[6]
    if ins.flags&I.F_RSC:return [base&~31]*kb
    blocks=[(base+4+k0+k)&~31 for k in range(kb)]
    previous=((w[1]+(row-parallel)*w[6]+4+k0+kb-1)&~31) if row>=parallel else -1
    return ([base&~31] if (base&~31) not in (blocks[0],previous) else [])+blocks


def cache_metadata(commands,I,W,parallel,entries,latency):
    if entries!=16 or parallel<1 or latency<1:raise ValueError('invalid scale-cache shape')
    # One tag and outstanding-reader lifetime per physical direct-mapped slot.
    tags=[[-1]*entries for _ in range(parallel)]
    valid=[[False]*entries for _ in range(parallel)]
    until=[[0]*entries for _ in range(parallel)]
    clocks=[0]*parallel;last=[None]*parallel
    rows=[[] for _ in commands];requests=misses=0
    for index,c in enumerate(commands):
        if any(space==W.SP_DRAM for space,lo,hi in c.footprint.wr):
            valid=[[False]*entries for _ in range(parallel)]
        ins=c.instruction
        if ins.op!=I.MM:continue
        # New command starts together on the physical cores. The timestamps
        # below represent minimum issue spacing, not the actual LPN clock.
        begin=max(clocks);clocks=[begin]*parallel
        for row in range(ins.w[3]&65535):
            p=row%parallel;addresses=scale_addresses(ins,I,row,parallel)
            item=dict(misses=0,span=0,gap=0);rows[index].append(item)
            if not addresses:continue
            first=None
            for address in addresses:
                slot=(address//32)%entries
                if not valid[p][slot] or tags[p][slot]!=address:
                    clocks[p]=max(clocks[p],until[p][slot])
                    valid[p][slot]=True;tags[p][slot]=address;item['misses']+=1;misses+=1
                if first is None:first=clocks[p]
                until[p][slot]=clocks[p]+latency
                clocks[p]+=1;requests+=1
            item['span']=clocks[p]-first
            if last[p] is not None:
                prior,start=last[p];prior['gap']=first-start
            last[p]=(item,first)
    for previous in last:
        if previous is not None:
            item,first=previous;item['gap']=item['span']
    miss_rows=[];pace_rows=[]
    for items in rows:
        miss_rows.append([r['misses'] for r in items])
        pace=[0]*len(items)
        for begin in range(0,len(items),parallel):
            # One LPN batch is a physical wave. Its pacing credit limits issue
            # spacing while the independent DRAM credit charges only misses.
            pace[begin]=max((r['gap'] for r in items[begin:begin+parallel]),default=0)
        pace_rows.append(pace)
    return dict(misses=miss_rows,pacing=pace_rows,requests=requests,backing_reads=misses,
                data_bytes=parallel*entries*32,
                coherence_scope='Canonical-order invalidation; overlapping write timing is approximate')
