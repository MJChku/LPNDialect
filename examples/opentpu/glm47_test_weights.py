"""Fast deterministic native-NVFP4 test weights; no trained-model quality claim.

Generate actual E2M1 codes, E4M3 per-16 scales and one FP32 tensor scale, then
use the already validated hardware importer. This avoids pointless Gaussian
FP32 weight generation and SEARCH quantization when no checkpoint is used.
The numerical ISA still computes every operation needed for real routing.
"""
import hashlib


def rng_for(name,seed=470047):
    import numpy as np
    key=hashlib.sha256(f'{seed}:{name}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:8],'little'))


def native_matrix(name,rows,columns,seed=470047):
    import numpy as np
    from opentpu import quant as Q
    rng=rng_for(name,seed)
    codes=rng.integers(0,16,(rows,columns),dtype=np.uint8)
    scales=rng.integers(48,65,(rows,columns//16),dtype=np.uint8)
    tensor_scale=np.float32(1/(3.5*columns**.5))
    return Q.import_nvfp4(codes,scales,tensor_scale,D=128)


def native_expert(name,fmt,seed=470047):
    import numpy as np
    from opentpu import quant as Q
    if fmt.F!=fmt.F0:raise ValueError('native test factory requires an unpadded expert width')
    out=np.zeros(fmt.nbytes,np.uint8)
    def put(pair,values):
        for at,value in zip(pair,values):
            raw=np.ascontiguousarray(value).view(np.uint8).ravel();out[at:at+raw.size]=raw
    put(fmt.wg,native_matrix(name+'.gate',fmt.F,fmt.H,seed))
    put(fmt.wu,native_matrix(name+'.up',fmt.F,fmt.H,seed))
    q,s=native_matrix(name+'.down',fmt.H,fmt.F,seed)
    for pair,values in zip(fmt.wd,Q.column_parts(q,s,'nvfp4',fmt.F,fmt.C,fmt.D)):put(pair,values)
    return out
