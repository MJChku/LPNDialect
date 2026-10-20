"""Fresh actual-input audit and timing replay of retained collective RTL gates."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import re

from .e2e import atomic_json,digest
from .workload import load,use_checkout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('opentpu-root','compiler','case','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();root=use_checkout(a.opentpu_root);a.output.mkdir(parents=True,exist_ok=False)
    from opentpu import rtlsim
    import numpy as np
    from .compact import CompactOpenTPU
    from .glm47_target import component_timing
    from .native_timing import ReusableTiming
    from .rtl_evolution.parallel_inference import simulator_cache_key
    old=json.loads((a.case/'report.json').read_text())
    if old['status']!='complete':raise ValueError('completed collective RTL report required')
    for f,h in old['source_sha256'].items():
        if digest(Path(f))!=h:raise ValueError('original collective sampler or source changed')
    rtlsim.VFLAGS[:]=old['rtl_compile_flags']
    timing=replace(component_timing(),dma_store_runs=True)
    files=[Path(__file__),*[Path(__file__).with_name(n) for n in ('collective.py','compact.py','model.py',
        'components.py','shape_workload.py','workload.py','native_timing.py','glm47_target.py')],
        *[rtlsim.RTL/n for n in rtlsim.RTL_SOURCES],rtlsim.TB/'tb_top.sv',rtlsim.TB/'lpn_wall.cpp']
    sources={str(f):digest(f) for f in files};records=[]
    runner=ReusableTiming(a.compiler,a.output/'native-cache',capacity=64)
    for prior in old['records']:
        source=a.case/prior['name'];directory=a.output/prior['name'];directory.mkdir()
        if not prior['rtl_validated'] or not prior['all_full_dram_and_tmem_bit_exact']:
            raise ValueError('retained numerical RTL completion required')
        for name,h in prior['input_sha256'].items():
            if digest(source/name)!=h:raise ValueError('canonical collective input changed')
        work=load(source);text=(source/'rtl.log').read_text()
        match=re.search(r'RESULT cycles=(\d+) halted=1 error=0',text)
        if not match or int(match[1])!=prior['rtl_cycles']:raise ValueError('raw RTL completion differs')
        counts=[int(v) for v in re.findall(r'SLICE \d+ icount=(\d+)',text)]
        if counts!=[len(c) for c in work.commands]:raise ValueError('dynamic RTL instruction counts differ')
        for sid,gold in enumerate(work.machine.slices):
            for name in (f'prog_{sid}.hex',f'dram_{sid}.bin'):
                if digest(source/'rtl'/name)!=digest(source/name):raise ValueError('RTL consumed a different input')
            if (source/'rtl'/f'dram_out_{sid}.bin').read_bytes()!=gold.dram.tobytes():
                raise ValueError('complete collective RTL DRAM differs from fresh ISA')
            words=np.asarray([int(v,16) for v in (source/'rtl'/f'tmem_{sid}.hex').read_text().split()],np.uint32)
            if not np.array_equal(words,gold.tmem):raise ValueError('complete collective RTL TMEM differs')
        binary=rtlsim.BUILD/simulator_cache_key(rtlsim,work.config,old['uarch'])/'Vtb_top'
        if not binary.is_file():raise ValueError('matching source/flags/config simulator binary is missing')
        physical=replace(work,config=replace(work.config,MCOLS=2))
        sim=CompactOpenTPU(physical,timing).net.python_simulator();ticks=sim.run(max_time=1000000)
        if sim.place_dicts('program_completed')!=[{}]:raise ValueError('collective model deadlocked')
        native=runner.prepare(physical,timing,directory/'timing.tables')
        cycles,seconds,_,log=native.run();(directory/'lpn.log').write_text(log)
        if cycles!=ticks:raise ValueError('Python/native collective replay differs')
        error=cycles/prior['rtl_cycles']-1
        records.append(dict(name=prior['name'],rtl_cycles=prior['rtl_cycles'],old_lpn_cycles=prior['lpn_cycles'],
            lpn_cycles=cycles,lpn_seconds=seconds,relative_cycle_error=error,timing_target_met=abs(error)<=.1,
            all_full_memories_bit_exact=True,canonical_input_sha256=prior['input_sha256'],
            matching_simulator_sha256=digest(binary),native_binary_sha256=digest(native.binary)))
    if sources!={str(f):digest(f) for f in files}:raise ValueError('audit sources changed')
    atomic_json(a.output/'report.json',dict(status='complete',scope=__doc__,records=records,timing=asdict(timing),
        source_sha256=sources,retained_report_sha256=digest(a.case/'report.json'),
        original_reports_preserved=True,no_rtl_simulation_repeated=True,
        all_timing_targets_met=all(r['timing_target_met'] for r in records)))
    if not all(r['timing_target_met'] for r in records):raise ValueError('collective timing target remains unmet')
    print('fresh collective ISA/native/full-memory audit passed',flush=True)


if __name__=='__main__':main()
