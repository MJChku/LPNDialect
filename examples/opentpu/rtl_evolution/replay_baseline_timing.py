"""Recheck normal-inference LPN predictions against retained original-RTL cycles.

Static dense control is resolved from each original program; tensor arithmetic
and RTL are not rerun. Program hashes and prior numerical/input checks are
required. This cannot replace the separate checkpoint numerical validation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import time


def replay(item):
    from examples.opentpu.workload import use_checkout
    from examples.opentpu.shape_workload import resolve_static
    from examples.opentpu.native_timing import ReusableTiming
    from examples.opentpu.model import Timing
    from examples.opentpu.e2e import digest
    use_checkout(Path(item['opentpu_root']))
    from opentpu import Config,isa as I,rtlsim
    cfg=Config(**item['config'])
    source=Path(item['directory']);target=Path(item['output'])
    target.mkdir(parents=True,exist_ok=False)
    record=item['record']
    programs=[[I.Instr.decode(words) for words in json.loads((source/f'program_{sid}.json').read_text())]
              for sid in range(cfg.S)]
    for sid,program in enumerate(programs):
        path=target/f'prog_{sid}.hex';rtlsim._write_hex(path,I.assemble(program))
        if digest(path)!=record['input_sha256'][path.name]:
            raise RuntimeError(f'kernel {item["index"]}: original program hash differs')
    before=time.perf_counter()
    workload=resolve_static(cfg,programs,record['args'])
    runner=ReusableTiming(Path(item['compiler']),Path(item['cache']),capacity=65536)
    prepared=runner.prepare(workload,Timing(**record['timing']),target/'timing.tables')
    cycles,wall,process,log=prepared.run();(target/'lpn.log').write_text(log)
    result=dict(kernel_index=item['index'],position_before=record['position_before'],
        phase=record['phase'],input_rows=record['input_rows'],rtl_cycles=record['rtl_cycles'],
        old_lpn_cycles=record['lpn_cycles'],new_lpn_cycles=cycles,
        relative_cycle_error=cycles/record['rtl_cycles']-1,
        old_relative_cycle_error=record['relative_cycle_error'],
        host_seconds=time.perf_counter()-before,lpn_core_seconds=wall,
        zero_rollbacks=True,program_hash_verified=True)
    (target/'record.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--opentpu-root',type=Path,required=True)
    ap.add_argument('--case',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--compiler',type=Path,required=True)
    ap.add_argument('--indices',help='Comma-separated kernel indices; default: snapshot all completed kernels')
    ap.add_argument('--workers',type=int,default=1)
    a=ap.parse_args()
    from examples.opentpu.e2e import atomic_json,digest
    if a.workers<1:ap.error('workers must be positive')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    case=a.case.resolve()
    config=json.loads((case/'identity.json').read_text())['config']
    project=Path(__file__).resolve().parents[1]
    files=[project/name for name in ('model.py','components.py','compact.py','shape_workload.py',
                                    'workload.py','native_timing.py','benchmark.py')]+[Path(__file__).resolve()]
    sources={str(p):digest(p) for p in files}
    wanted=set(map(int,a.indices.split(','))) if a.indices else None
    items=[]
    for path in sorted((case/'kernels').glob('*/record.json')):
        record=json.loads(path.read_text());index=record['step']
        if wanted is not None and index not in wanted:continue
        if not record['all_dram_and_tmem_bit_exact'] or not record['identical_input_files']:
            raise ValueError(f'kernel {index} lacks verified original-RTL input/numerical evidence')
        items.append(dict(index=index,record=record,config=config,directory=str(path.parent),
            opentpu_root=str(a.opentpu_root.resolve()),compiler=str(a.compiler.resolve()),
            cache=str(a.output/'cache'),output=str(a.output/f'{index:04d}')))
    if not items or (wanted is not None and wanted!={i['index'] for i in items}):
        raise ValueError('requested completed original-RTL kernels are missing')
    results=[]
    def save(status):
        atomic_json(a.output/'report.json',dict(status=status,pid=os.getpid(),
            scope=__doc__,case=str(case),source_sha256=sources,
            compiler_sha256=digest(a.compiler),
            records=sorted(results,key=lambda r:r['kernel_index']),
            timing_target_met_all_kernels=all(abs(r['relative_cycle_error'])<=.05 for r in results)))
    save('running')
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(replay,item) for item in items]):
            result=future.result();results.append(result)
            print(json.dumps(result),flush=True);save('running')
    if sources!={str(p):digest(p) for p in files}:
        raise RuntimeError('timing sources changed during replay')
    save('complete')

if __name__=='__main__':main()
