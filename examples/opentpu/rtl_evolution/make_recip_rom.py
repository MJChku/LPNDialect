"""Generate a compact exact table for openTPU's normalized reciprocal sequence.

A 4096-segment fixed-point linear predictor supplies the significand. A signed
2-bit correction indexed by all 23 input fraction bits restores the original
Newton sequence bit for bit. This is arithmetic calibration, not workload data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    a.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    n=1<<23;span=1<<11;shift=12;scale=1<<(47+shift)
    bases=[];slopes=[]
    for segment in range(1<<12):
        lo=(1<<23)+segment*span;hi=lo+span
        bases.append((scale+lo//2)//lo)
        slopes.append(-((scale+(lo*hi)//2)//(lo*hi)))
    bases=np.array(bases,dtype=np.int64);slopes=np.array(slopes,dtype=np.int64)
    assert np.all((slopes>=-(1<<13)) & (slopes<(1<<13)))
    assert np.all((bases>=0)&(bases<(1<<37)))
    (a.output/'coeff.hex').write_text(''.join(f'{(int(b)<<14)|(int(s)&16383):013x}\n' for b,s in zip(bases,slopes)))
    minimum,maximum=100,-100;wrong_exp=0
    reference_hash=hashlib.sha256()
    with (a.output/'correction.hex').open('w') as hexfile,(a.output/'correction.packed').open('wb') as packed:
        for start in range(0,n,1<<18):
            frac=np.arange(start,min(start+(1<<18),n),dtype=np.uint32)
            bits=frac|np.uint32(0x3f800000)
            x=bits.view(np.float32)
            y=(np.uint32(0x7ef311c3)-bits).view(np.float32)
            # Every NumPy operation materializes its float32 result, preserving
            # the separate roundings of the independent ISA and RTL functions.
            for _ in range(2):
                prod=np.multiply(x,y,dtype=np.float32)
                t=np.subtract(np.float32(2),prod,dtype=np.float32)
                y=np.multiply(y,t,dtype=np.float32)
            prod=np.multiply(x,y,dtype=np.float32)
            err=np.subtract(np.float32(1),prod,dtype=np.float32)
            correction=np.multiply(y,err,dtype=np.float32)
            y=np.add(correction,y,dtype=np.float32)
            ybits=y.view(np.uint32)
            reference_hash.update(ybits.astype('<u4').tobytes())
            wrong_exp+=int(np.count_nonzero((frac!=0)&((ybits>>23)!=126)))
            exact=(ybits&np.uint32(0x7fffff)).astype(np.int64)+(1<<23)
            exact=np.where(frac==0,1<<24,exact)
            segment=frac>>11;low=(frac&2047).astype(np.int64)
            predicted=(bases[segment]+slopes[segment]*low+(1<<(shift-1)))>>shift
            delta=exact-predicted
            minimum=min(minimum,int(delta.min()));maximum=max(maximum,int(delta.max()))
            if np.any((delta< -2)|(delta>1)):raise RuntimeError(f'correction exceeds signed pair: {minimum}..{maximum}')
            restored=predicted+delta
            if np.any(restored!=exact):raise RuntimeError('significand reconstruction failed')
            nibble=(delta&3).astype(np.uint8)
            packed.write((nibble[0::4]|(nibble[1::4]<<2)|(nibble[2::4]<<4)|(nibble[3::4]<<6)).tobytes())
            hexfile.write(''.join(f'{int(v):x}\n' for v in nibble))
    if wrong_exp:raise RuntimeError('normalized reference exponent is not 126')
    files={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
           for f in a.output.iterdir() if f.is_file()}
    result=dict(status='complete',scope=__doc__,inputs=n,coarse_segments=len(bases),
        fraction_bits=23,segment_low_bits=11,fixed_point_fraction_bits=shift,
        correction_bits=2,min_correction=minimum,max_correction=maximum,
        expected_physical_table_bits=n*2+len(bases)*51,
        normalized_sequence_reconstructed_all_inputs=True,reference_float32_sha256=reference_hash.hexdigest(),
        normalized_reference='Separate float32 multiplies/adds: seed 0x7ef311c3, two Newton steps and correction; x=1.fraction',
        limitations=['SV primitive gate is separate','Scaled/exceptional reciprocal and QACT/VPU integration not yet validated','FPGA synthesis, inference and clock are not proven'],
        numpy_version=np.__version__,generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seconds=time.perf_counter()-started,files=files)
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
