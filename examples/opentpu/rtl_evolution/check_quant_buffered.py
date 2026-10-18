"""Compare real quantizer outputs against independent ISA after each command."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opentpu-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chunks',type=int,default=32)
    p.add_argument('--stall',type=int,choices=(0,1,2),default=0)
    p.add_argument('--stream-act',action='store_true')
    p.add_argument('--lookup-scale',action='store_true')
    p.add_argument('--recip-table',type=Path)
    p.add_argument('--fp-mul-latency',type=int,choices=(1,2),default=2)
    p.add_argument('--fp-add-latency',type=int,choices=(1,2,3,4),default=4)
    a=p.parse_args()
    if a.lookup_scale and a.recip_table is None:p.error('--lookup-scale requires --recip-table')
    if a.recip_table:a.recip_table=a.recip_table.resolve()
    sys.path.insert(0,str(a.opentpu_root.resolve()))
    import numpy as np
    from opentpu import Config,isa as I,rtlsim
    from opentpu.isasim import Machine
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=Config(D=64,LANES=64,MCOLS=16,ACT_ROWS=16,ACT_BLOCKS=64,
               TMEM_WORDS=32768,DRAM_BYTES=1<<20)
    rng=np.random.default_rng(672)
    x=np.ldexp(rng.normal(size=cfg.TMEM_WORDS).astype(np.float32),rng.integers(-15,16,cfg.TMEM_WORDS))
    # Zero, infinity, NaN, denormals and ties; chunk and row bank wraparound.
    for row,base in enumerate((64,192,576)):
        x[base:base+64]=0
        x[2*base+3]=np.inf
        x[3*base:3*base+64]=np.float32(1e-40)
        x[4*base+7]=np.nan
    x[22000:24000]=np.resize(np.arange(-127.5,128,.5,dtype=np.float32),2000)
    x[31000:32000]=rng.uniform(.25,4,1000).astype(np.float32)
    specs=[]
    for rows,kb in ((1,1),(8,1),(9,3),(8,9),(9,33)):
        for row,half,es in ((False,False,1),(False,False,17),(True,False,4),(True,True,1)):
            specs.append((f'qst_r{rows}_kb{kb}_row{int(row)}_half{int(half)}_es{es}',
                          I.qst(0,65537,700000,rows,kb,kb*64,kb*64*es+3,es,row=row,half=half)))
    for rows,kb in ((1,1),(8,1),(9,3),(8,9),(9,33)):
        specs.append((f'qact_r{rows}_kb{kb}',I.qact(0,rows,2,kb,kb*64)))
    specs.append(('qact_prescaled',I.qact(0,8,2,9,576,cscale=31000,rscale=31900)))
    specs.append(('qact_dup',I.qact(0,8,2,3,192,dup=True)))
    specs.append(('qst_ties',I.qst(22000,65537,700000,8,3,192,193,1,row=True)))
    (a.output/'tmem.hex').write_text('\n'.join(f'{int(v):08x}' for v in x.view(np.uint32))+'\n')
    words=[]
    for _,ins in specs:
        packed=(ins.op<<232)|(ins.flags<<224)
        for k,w in enumerate(ins.w):packed|=int(w)<<(32*(6-k))
        words.append(f'{packed:060x}')
    (a.output/'commands.hex').write_text('\n'.join(words)+'\n')
    sources=[rtlsim.RTL/s for s in ('vpu/otpu_fp.sv','vpu/otpu_fpipe.sv','top/otpu_pkg.sv',
                                  'tech/generic/otpu_tech_ram.sv','vpu/otpu_quant.sv')]
    sources.append(Path(__file__).with_name('tb_quant_buffered.sv'))
    table_files=[];table_args=[]
    if a.lookup_scale:
        sources += [rtlsim.RTL/'vpu/otpu_recip_rom.sv',rtlsim.RTL/'vpu/otpu_qscale_lookup.sv']
        calibration=json.loads((a.recip_table/'report.json').read_text())
        if calibration['status']!='complete' or calibration['correction_bits']!=2:raise RuntimeError('unverified reciprocal table')
        table_files=[a.recip_table/'coeff.hex',a.recip_table/'correction.hex']
        for f in table_files:
            if hashlib.sha256(f.read_bytes()).hexdigest()!=calibration['files'][f.name]['sha256']:raise RuntimeError('reciprocal table changed')
        table_args=[f'+recip_coeff={table_files[0]}',f'+recip_correction={table_files[1]}']
    files=sources+table_files
    source_hashes={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in files}
    quant_source=(rtlsim.RTL/'vpu/otpu_quant.sv').read_text()
    for parameter,requested in (('STREAM_ACT',a.stream_act),('LOOKUP_SCALE',a.lookup_scale)):
        supported=bool(re.search(r'parameter\s+bit\s+'+parameter+r'\b',quant_source))
        if requested and not supported:raise RuntimeError(f'quantizer does not support {parameter}')
        if supported:rtlsim.VFLAGS.append('-DOTPU_HAS_'+parameter)
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='12'
    rtlsim.VFLAGS += [f'-DOTPU_FP_MUL_LAT={a.fp_mul_latency}',f'-DOTPU_FP_ADD_LAT={a.fp_add_latency}']
    exe=rtlsim.build('tb_quant_buffered',sources,dict(XD=a.chunks,STREAM_ACT=int(a.stream_act),LOOKUP_SCALE=int(a.lookup_scale)))
    run=subprocess.run([str(exe),f'+dir={a.output}',f'+ncmd={len(specs)}',f'+stall={a.stall}']+table_args,
                       capture_output=True,text=True,timeout=120)
    (a.output/'rtl.log').write_text(run.stdout+run.stderr)
    run.check_returncode()
    if source_hashes!={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in files}:
        raise RuntimeError('quantizer sources changed during the gate')
    observed=[int(c) for _,c in re.findall(r'QCMD index=(\d+) cycles=(\d+)',run.stdout)]
    if len(observed)!=len(specs):raise RuntimeError('missing command measurements')
    records=[]
    # Fresh ISA run per prefix proves command-boundary visibility, not only final state.
    for index,(name,_) in enumerate(specs):
        m=Machine(cfg,[[ins for _,ins in specs[:index+1]]+[I.halt()]],[np.zeros(cfg.DRAM_BYTES,np.uint8)])
        m.slices[0].tmem[:]=x.view(np.uint32)
        s=m.run().slices[0]
        expected_scales=(s.ascale.view(np.uint32).astype(np.uint64) |
            (s.adoff[:,:,0].astype(np.uint64)<<32) | (s.adoff[:,:,1].astype(np.uint64)<<40) |
            (s.adoff[:,:,2].astype(np.uint64)<<48) | (s.adoff[:,:,3].astype(np.uint64)<<56))
        for field,expected,dtype in (('dram',s.dram,'u1'),('act',s.act.view(np.uint8),'u1'),
                                     ('scales',expected_scales,'<u8')):
            actual=np.fromfile(a.output/f'{field}_{index}.bin',dtype=dtype)
            bad=np.flatnonzero(actual!=expected.reshape(-1))
            if len(bad):raise RuntimeError(f'{name}: {field} differs at {bad[:16].tolist()}')
        records.append(dict(name=name,rtl_cycles=observed[index],all_state_bit_exact=True))
    report=dict(status='complete',stream_act=a.stream_act,lookup_scale=a.lookup_scale,records=records,stall=a.stall,chunks=a.chunks,
                fp_mul_latency=a.fp_mul_latency,fp_add_latency=a.fp_add_latency,
                simulator=str(exe),source_sha256=source_hashes,rtl_compile_flags=list(rtlsim.VFLAGS))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
