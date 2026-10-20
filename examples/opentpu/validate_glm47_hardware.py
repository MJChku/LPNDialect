"""Validate retained GLM hardware launches against RTL without regenerating inputs.

Parallel workers validate independent snapshots; they do not parallelize the
accelerator or change the workload's stage ordering. Complete ISA/RTL states
and exact inter-stage DRAM continuity are checked before summing kernel cycles.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import time

from .e2e import atomic_json,digest
from .workload import load,use_checkout


def check_stage(task):
    import numpy as np
    from .native_timing import ReusableTiming
    from .hardware_commands import instruction_statistics
    from .glm47_target import component_timing
    root=use_checkout(Path(task['opentpu_root']))
    from opentpu import rtlsim
    rtlsim.VFLAGS[:]=task['rtl_compile_flags']
    source=Path(task['source']);directory=Path(task['directory']);directory.mkdir()
    atomic_json(directory/'worker.json',dict(status='functional',pid=os.getpid()))
    for name,h in task['expected']['input_sha256'].items():
        if digest(source/name)!=h:raise ValueError('retained hardware input changed')
        os.link(source/name,directory/name)
    work=load(directory);gold=work.machine.slices[0]
    stats=instruction_statistics(work)
    if stats!=task['expected']['hardware_instructions']:raise ValueError('actual instruction accounting changed')
    final_dram_hash=hashlib.sha256(gold.dram.view('<u4').astype('>u4').tobytes()).hexdigest()
    timing=component_timing();native=ReusableTiming(Path(task['compiler']),directory/'native-cache',capacity=32768).prepare(
        replace(work,config=replace(work.config,MCOLS=2)),timing,directory/'timing.tables')
    cycles,seconds,native_process,log=native.run();(directory/'lpn.log').write_text(log)
    atomic_json(directory/'worker.json',dict(status='building',pid=os.getpid()))
    before=time.perf_counter();binary=rtlsim.build_top(work.config,uarch=task['uarch']);build_seconds=time.perf_counter()-before
    cmd=[str(binary),f'+dir={directory}','+max_cycles=5000000','+trace','+lpn_wall']
    cmd += [f'+arg{k}={int(v)&0xffffffff}' for k,v in enumerate(work.args)]
    atomic_json(directory/'worker.json',dict(status='simulating',pid=os.getpid(),command=cmd))
    before=time.perf_counter();run=rtlsim.run_sim(cmd);process_seconds=time.perf_counter()-before
    text=run.stdout+run.stderr;(directory/'rtl.log').write_text(text);run.check_returncode()
    matched=re.search(r'RESULT cycles=(\d+) halted=1 error=0',text)
    if not matched:raise ValueError('RTL did not halt cleanly')
    if [int(n) for n in re.findall(r'SLICE \d+ icount=(\d+)',text)]!=[len(work.commands[0])]:
        raise ValueError('RTL dynamic instruction count differs')
    # Binary outputs contain device-endian bytes, unlike big-endian inputs.
    if (directory/'dram_out_0.bin').read_bytes()!=gold.dram.tobytes():raise ValueError('complete RTL DRAM differs')
    words=np.asarray([int(w,16) for w in (directory/'tmem_0.hex').read_text().split()],dtype=np.uint32)
    if not np.array_equal(words,gold.tmem):raise ValueError('complete RTL scratchpad differs')
    for name,h in task['expected']['input_sha256'].items():
        if digest(directory/name)!=h:raise ValueError('RTL hardware input changed')
    observed=int(matched[1]);error=cycles/observed-1
    wall=re.search(r'LPN_WALL seconds=([0-9.]+) cycles=(\d+)',text)
    if not wall or int(wall[2])!=observed:raise ValueError('RTL clock-window timer missing or inconsistent')
    result=dict(name=source.name,status='complete',static_instructions=len(work.programs[0]),
        hardware_instructions=stats,input_sha256=task['expected']['input_sha256'],
        lpn_cycles=cycles,lpn_seconds=seconds,lpn_process_seconds=native_process,
        rtl_cycles=observed,relative_cycle_error=error,rtl_cycle_window_seconds=float(wall[1]),
        timing_target_met=abs(error)<=.1,full_dram_and_tmem_bit_exact=True,
        final_dram_as_input_sha256=final_dram_hash,
        final_dram_sha256=digest(directory/'dram_out_0.bin'),final_tmem_sha256=digest(directory/'tmem_0.hex'),
        simulator=str(binary),simulator_sha256=digest(binary),native_binary_sha256=digest(native.binary),
        rtl_build_seconds=build_seconds,rtl_process_seconds=process_seconds)
    atomic_json(directory/'result.json',result)
    atomic_json(directory/'worker.json',dict(status='complete',pid=os.getpid()))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('opentpu-root','compiler','case','base-rtl-gate','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--cxx-opt',choices=('0','3'),default='0',help='Verilator C++ optimization; use 3 for performance comparisons')
    a=p.parse_args()
    if not 1<=a.workers<=3:p.error('workers must be between one and three')
    a.output.mkdir(parents=True,exist_ok=False);root=use_checkout(a.opentpu_root)
    source=json.loads((a.case/'report.json').read_text());base=json.loads((a.base_rtl_gate/'report.json').read_text())
    flags=list(base['rtl_compile_flags'])
    if flags[flags.index('-CFLAGS')+1]!='-O0':raise ValueError('unexpected baseline C++ build flags')
    flags[flags.index('-CFLAGS')+1]='-O'+a.cxx_opt
    if source['status']!='complete' or not source['all_stages_numerically_executed'] or source['routing_annotations_used']:
        raise ValueError('completed instruction-driven numerical layer required')
    files={**source['source_sha256'],**base['source_sha256']}
    for f,h in files.items():
        if digest(Path(f))!=h:raise ValueError('retained compiler or RTL source changed')
    if base['uarch']!=dict(MXU_ROWS=2,MXU_COLS=2,MXU_PAIR_DOT=1,MXU_CL=8,VPU_LANES=64,QST_LANES=64):
        raise ValueError('physical target differs')
    from .glm47_target import component_timing
    from .rtl_evolution.parallel_inference import validate_variant
    validate_variant(asdict(component_timing()),base['uarch'])
    local=[Path(__file__),*[Path(__file__).with_name(n) for n in
        ('glm47_target.py','workload.py','model.py','compact.py','components.py','native_timing.py','hardware_commands.py')]]
    files.update({str(f):digest(f) for f in local});records=[];started=time.perf_counter()
    def report(status):
        ordered=sorted(records,key=lambda r:[s['name'] for s in source['records']].index(r['name']))
        return dict(status=status,scope=__doc__,records=ordered,timing=asdict(component_timing()),
            source_sha256=files,retained_case=str(a.case),retained_report_sha256=digest(a.case/'report.json'),
            rtl_compile_flags=flags,uarch=base['uarch'],
            shared_physical_dram=True,routing_annotations_used=False,all_stages_numerically_executed=True,
            weights=source['weights'],full_glm_model=False,rtl_validated=status=='complete',
            host_workers=a.workers,device_kernel_order=[r['name'] for r in source['records']],
            host_seconds=time.perf_counter()-started)
    atomic_json(a.output/'report.json',report('running'))
    try:
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(check_stage,dict(opentpu_root=str(root),compiler=str(a.compiler),
                source=str(a.case/r['name']),directory=str(a.output/r['name']),expected=r,
                rtl_compile_flags=flags,uarch=base['uarch'])) for r in source['records']]
            for future in as_completed(futures):
                records.append(future.result());atomic_json(a.output/'report.json',report('running'))
        result=report('complete')
        for first,second in zip(result['records'],result['records'][1:]):
            if first['final_dram_as_input_sha256']!=second['input_sha256']['dram_0.bin']:
                raise ValueError('complete physical DRAM continuity differs between stages')
        for f,h in files.items():
            if digest(Path(f))!=h:raise ValueError('validation source changed during execution')
        result.update(shared_dram_stage_continuity_verified=True,
            rtl_cycles_sum=sum(r['rtl_cycles'] for r in records),lpn_cycles_sum=sum(r['lpn_cycles'] for r in records),
            lpn_seconds_sum=sum(r['lpn_seconds'] for r in records),
            worst_relative_cycle_error=max(abs(r['relative_cycle_error']) for r in records),
            all_full_dram_and_tmem_bit_exact=True,all_timing_targets_met=all(r['timing_target_met'] for r in records))
        atomic_json(a.output/'report.json',result)
    except BaseException as exc:
        result=report('failed');result['error']=str(exc);atomic_json(a.output/'report.json',result);raise
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
