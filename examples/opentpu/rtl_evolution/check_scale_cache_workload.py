"""Run one unchanged retained inference kernel on scale-cache RTL.

This is a numerical and hardware-cycle gate. The existing LPN equations do
not yet model the cache, so this tool makes no LPN accuracy claim.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('case','opentpu-root','recip-table','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--kernel',type=int,default=16)
    p.add_argument('--entries',type=int,choices=(0,16),default=16)
    p.add_argument('--rows',type=int,choices=(32,128),default=128)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    from examples.opentpu.e2e import atomic_json,digest
    from examples.opentpu.workload import use_checkout
    from examples.opentpu.retained_inputs import verified_program
    from .parallel_inference import files_equal,configure_rtlsim
    from .rom_inputs import reciprocal_inputs
    use_checkout(a.opentpu_root.resolve())
    import numpy as np
    from opentpu import Config,rtlsim
    reference_path=a.case/'report.json';reference=json.loads(reference_path.read_text())
    if (reference['status']!='complete' or reference['entrypoint']!='unmodified upstream Engine.generate' or
        not reference['all_dram_and_tmem_bit_exact'] or not reference['all_inputs_identical_to_baseline'] or
        reference['prompt_tokens']!=128 or reference['output_tokens']!=32 or a.kernel not in range(48)):
        raise ValueError('requires completed identical-input normal 128/32 inference')
    record=reference['kernels'][a.kernel];cfg=Config(**reference['config'])
    source=a.case/'kernels'/f'{a.kernel:04d}';identities={}
    for sid in range(cfg.S):
        for name in (f'prog_{sid}.hex',f'dram_{sid}.bin'):
            if digest(source/name)!=record['input_sha256'][name]:raise ValueError('canonical input changed')
        _,identities[f'program_{sid}.json']=verified_program(source,sid)
        if not files_equal(source/f'golden_dram_{sid}.bin',source/f'dram_out_{sid}.bin'):
            raise ValueError('retained independent ISA state differs from reference RTL')
        for name in (f'prog_{sid}.hex',f'dram_{sid}.bin',f'program_{sid}.json'):
            os.link(source/name,a.output/name)
    configure_rtlsim(rtlsim,2,2)
    # More C++ translation units reduce build latency; hardware cycles are
    # unchanged. Include the setting in the simulator key and saved flags.
    rtlsim.VFLAGS+=['--output-groups','0']
    ua=dict(reference['uarch'],MXU_ROWS=a.rows,ACT_STREAMING=1,QUANT_LOOKUP=1,
            VPU_SCALAR_RECIP=1,SCALE_CACHE_ENTRIES=a.entries)
    table_files,table_hashes,table_args=reciprocal_inputs(a.recip_table)
    files=[rtlsim.RTL/n for n in rtlsim.RTL_SOURCES]+[rtlsim.TB/'tb_top.sv',rtlsim.TB/'lpn_wall.cpp',
            a.opentpu_root/'opentpu/rtlsim.py',Path(__file__),reference_path,*table_files]
    hashes={str(f):digest(f) for f in files}
    def save(status,**fields):
        atomic_json(a.output/'report.json',dict(status=status,pid=os.getpid(),scope=__doc__,kernel=a.kernel,
            config=asdict(cfg),args=record['args'],uarch=ua,source_sha256=hashes,
            input_sha256=record['input_sha256'],program_json_sha256=identities,
            lookup_table_sha256=table_hashes,rtl_compile_flags=list(rtlsim.VFLAGS),
            lpn_scale_cache_modeled=False,**fields))
    save('building');before=time.perf_counter();exe=rtlsim.build_top(cfg,uarch=ua,dram_lat=8)
    build=time.perf_counter()-before
    command=[str(exe),f'+dir={a.output}',f'+max_cycles={record["baseline_rtl_cycles"]*2}',
             '+lpn_wall']+[f'+arg{k}={int(v)&0xffffffff}' for k,v in enumerate(record['args'])]+table_args
    save('simulating',simulator=str(exe),simulator_sha256=digest(exe),command=command,build_seconds=build)
    before=time.perf_counter();run=rtlsim.run_sim(command,timeout=7200);elapsed=time.perf_counter()-before
    log=run.stdout+run.stderr;(a.output/'rtl.log').write_text(log);run.check_returncode()
    result=re.search(r'RESULT cycles=(\d+) halted=(\d+) error=(\d+)',log)
    wall=re.search(r'LPN_WALL seconds=([0-9.]+) cycles=(\d+)',log)
    if not result or (result[2],result[3])!=('1','0') or not wall or int(wall[2])!=int(result[1]):
        raise RuntimeError('RTL did not halt cleanly or timer differs')
    if [int(x) for x in re.findall(r'SLICE \d+ icount=(\d+)',log)]!=record['instructions']:
        raise RuntimeError('RTL instruction counts differ')
    for sid in range(cfg.S):
        if not files_equal(a.output/f'dram_out_{sid}.bin',source/f'golden_dram_{sid}.bin'):
            raise RuntimeError('full DRAM differs from independent ISA')
        tmem=np.array([int(v,16) for v in (a.output/f'tmem_{sid}.hex').read_text().split()],dtype='<u4')
        if len(tmem)!=cfg.TMEM_WORDS or tmem.tobytes()!=(source/f'golden_tmem_{sid}.bin').read_bytes():
            raise RuntimeError('full scratchpad differs from independent ISA')
        for name in (f'prog_{sid}.hex',f'dram_{sid}.bin'):
            if digest(a.output/name)!=record['input_sha256'][name]:raise RuntimeError('input changed')
        _,identity=verified_program(a.output,sid)
        if identity!=identities[f'program_{sid}.json']:raise RuntimeError('decoded program changed')
    if hashes!={str(f):digest(f) for f in files}:raise RuntimeError('gate sources changed')
    counts=re.findall(r'SCACHE s=(\d+) p=(\d+) req=(\d+) miss=(\d+)',log)
    if a.entries and (len(counts)!=cfg.S*a.rows or len({(s,p) for s,p,_,_ in counts})!=len(counts)):
        raise RuntimeError('missing per-core cache counters')
    save('complete',simulator=str(exe),simulator_sha256=digest(exe),command=command,build_seconds=build,
         rtl_process_seconds=elapsed,rtl_core_seconds=float(wall[1]),rtl_cycles=int(result[1]),
         original_rtl_cycles=record['baseline_rtl_cycles'],all_full_dram_and_tmem_bit_exact=True,
         identical_inputs=True,upstream_schedule_unchanged=True,numerical_prepass_repeated=False,
         cache_requests=sum(int(n) for _,_,n,_ in counts),cache_misses=sum(int(n) for _,_,_,n in counts),
         raw_log_sha256=digest(a.output/'rtl.log'))
    print((a.output/'report.json').read_text(),flush=True)


if __name__=='__main__':main()
