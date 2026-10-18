"""Numerical gates for the actual parallel RTL; no timestamps enter the ISA model."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--opentpu-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rows',type=int,default=32)
    ap.add_argument('--vector-lanes',type=int,default=64)
    ap.add_argument('--stores',type=int,default=64)
    ap.add_argument('--depth',type=int,default=64)
    ap.add_argument('--lanes',type=int,default=64)
    ap.add_argument('--host-opt',choices=('0','3'),default='0')
    ap.add_argument('--tree-source',type=Path,help='Explicit alternate reduction-tree source for portability gates')
    a=ap.parse_args()
    sys.path.insert(0,str(a.opentpu_root.resolve()))
    import numpy as np
    from opentpu import Config,isa as I,quant as Q,rtlsim
    from opentpu.isasim import Machine
    if a.tree_source:
        rtlsim.RTL_SOURCES=[str(a.tree_source.resolve()) if s=='vpu/otpu_vtree.sv' else s for s in rtlsim.RTL_SOURCES]
    cfg=Config(D=a.depth,MCOLS=4,ACT_ROWS=16,LANES=a.lanes)
    ua=dict(MXU_ROWS=a.rows,VPU_LANES=a.vector_lanes,QST_LANES=a.stores)
    a.output.mkdir(parents=True,exist_ok=True)
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='12'
    rtlsim.VFLAGS+=['-CFLAGS','-O'+a.host_opt]
    results=[]
    # Partial engine groups, bank-wrapping output strides, activation replay,
    # scale dependencies, fused accumulation, matrix max, and packed weights.
    cases=[dict(name=f'unit_{2*a.rows+3}x8',n=2*a.rows+3,m=8,kb=3,unit=True),
           dict(name='unit_5x3',n=5,m=3,kb=1,unit=True),
           dict(name=f'scaled_{2*a.rows+1}x8',n=2*a.rows+1,m=8,kb=5),
           dict(name='acc_ascale_rmax',n=2*a.rows+3,m=3,kb=3,unit=True,acc=True,ascale=True,rmax=True),
           dict(name='fp4_pair',n=a.rows+3,m=2,kb=3,wf=I.W4F,pair=True),
           dict(name=f'unit_{2*a.rows}x1',n=2*a.rows,m=1,kb=7,unit=True),
           *[dict(name=f'qst_stride_{es}',n=35,m=3,kb=3,unit=True,qst_stride=es)
             for es in (1,4,17)]]
    for c in cases:
        rng=np.random.default_rng(900+len(results)); n,m,kb=c['n'],c['m'],c['kb']; k=kb*cfg.D
        x=rng.normal(0,1,(m,k)).astype(np.float32)
        image=np.zeros(cfg.DRAM_BYTES,np.uint8); image[:x.nbytes]=x.view(np.uint8).ravel()
        wbase,sbase=1<<16,1<<18
        wf=c.get('wf',I.W8)
        if wf==I.W8:
            data=rng.integers(-127,128,(n,k),dtype=np.int8).view(np.uint8)
            srs=4+4*((kb+3)//4)
            scales=np.zeros((n,srs),np.uint8)
            scales[:,:4]=rng.uniform(.001,.02,(n,1)).astype(np.float32).view(np.uint8)
            scales[:,4:4+kb]=rng.integers(0,4,(n,kb),dtype=np.uint8)
        else:
            data=Q.pack4(rng.integers(0,16,(n,k)))
            srs=8+4*kb+(4*kb)%8
            scales=np.zeros((n,srs),np.uint8)
            scales[:,:4]=rng.uniform(.001,.02,(n,1)).astype(np.float32).view(np.uint8)
            scale_words=np.full((n,kb),0x41041000,dtype=np.uint32)
            scales[:,8:8+4*kb]=scale_words.view(np.uint8)
        pitch=((data.shape[1]+cfg.D-1)//cfg.D)*cfg.D
        for r in range(n):image[wbase+r*pitch:wbase+r*pitch+data.shape[1]]=data[r]
        image[sbase:sbase+scales.size]=scales.ravel()
        out=16384; ors=n+3
        prog=[I.ld(0,0,x.size),I.qact(0,m,0,kb,k,dup=c.get('pair',False))]
        if c.get('acc'):
            old=rng.normal(size=(m,ors)).astype(np.float32)
            image[32768:32768+old.nbytes]=old.view(np.uint8).ravel()
            alpha=rng.normal(size=m).astype(np.float32)
            image[49152:49152+alpha.nbytes]=alpha.view(np.uint8)
            prog += [I.ld(32768,out,old.size),I.ld(49152,8192,m)]
        prog += [I.mm(wbase,sbase,out,n,kb,pitch,ors,m,0,srs,
                     unit=c.get('unit',False),acc=c.get('acc',False),
                     ascale=8192 if c.get('ascale') else None,
                     rmax=c.get('rmax',False),wf=wf,pair=c.get('pair',False))]
        if 'qst_stride' in c:
            es=c['qst_stride']
            prog += [I.qst(0,(1<<19)+1,3<<18,m,kb,k,k*es+3,es)]
        prog += [I.halt()]
        golden=Machine(cfg,[prog],[image.copy()]).run().slices[0]
        before=time.perf_counter()
        try:
            drams,tmems,stats=rtlsim.run(cfg,[prog],[image],uarch=ua,
                                      max_cycles=500000,keep=a.output/c['name'],trace=True)
            if not np.array_equal(drams[0],golden.dram) or not np.array_equal(tmems[0],golden.tmem):
                bad=np.flatnonzero(tmems[0]!=golden.tmem)
                raise RuntimeError(f'numerical mismatch: TMEM indices {bad[:16].tolist()}, '
                                   f'actual {tmems[0][bad[:8]].tolist()}, expected {golden.tmem[bad[:8]].tolist()}, '
                                   f'DRAM equal {np.array_equal(drams[0],golden.dram)}')
        except Exception as e:
            (a.output/'failure.json').write_text(json.dumps(dict(case=c,error=str(e)),indent=2)+'\n')
            raise
        result=dict(case=c,rtl_cycles=stats['cycles'],seconds=time.perf_counter()-before,
                    all_dram_and_tmem_bit_exact=True)
        results.append(result)
        (a.output/'report.json').write_text(json.dumps(dict(status='running',config=cfg.__dict__,uarch=ua,records=results),indent=2)+'\n')
        print(json.dumps(result),flush=True)
    (a.output/'report.json').write_text(json.dumps(dict(status='complete',config=cfg.__dict__,uarch=ua,records=results),indent=2)+'\n')

if __name__=='__main__':main()
