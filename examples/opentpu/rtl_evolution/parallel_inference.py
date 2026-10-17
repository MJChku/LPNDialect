"""Validate one upstream inference using parallel, independent RTL kernel runs.

The unmodified Engine generates kernels in inference order using independent
ISA results. Every kernel snapshots its complete input and golden state before
submission. Bit-exact checks for every kernel and matching baseline inputs/IDs
prove state continuity afterwards. Hardware cycles are summed; host parallelism
is only a validation optimization, not additional hardware concurrency.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace


def files_equal(a,b):
    if a.stat().st_size!=b.stat().st_size:return False
    with a.open('rb') as left,b.open('rb') as right:
        while block:=left.read(8<<20):
            if block!=right.read(len(block)):return False
    return True


def check_kernel(task):
    from examples.opentpu.workload import use_checkout
    from examples.opentpu.e2e import atomic_json,digest
    from examples.opentpu.rtl_evolution.baseline_stream import BaselineReference
    use_checkout(Path(task['opentpu_root']))
    import numpy as np
    from opentpu import Config,rtlsim,isa as I
    directory=Path(task['directory']);cfg=Config(**task['config']);index=task['index']
    try:
        reference=BaselineReference(task['baseline'],task['baseline_job'])
        original=reference.kernel(index)
        hashes={name:digest(directory/name) for sid in range(cfg.S)
                for name in (f'prog_{sid}.hex',f'dram_{sid}.bin')}
        if hashes!=task['input_sha256'] or hashes!=original['input_sha256']:
            raise RuntimeError(f'kernel {index}: input hashes differ from original RTL')
        rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='12'
        rtlsim.VFLAGS+=['-CFLAGS','-O3']
        atomic_json(directory/'worker.json',dict(status='building',pid=os.getpid()))
        before=time.perf_counter()
        exe=rtlsim.build_top(cfg,uarch=task['uarch'],dram_lat=8)
        build=time.perf_counter()-before
        # This is the same behavioral-memory harness as rtlsim.run. Files were
        # already serialized once by its canonical writer in the producer.
        cmd=[str(exe),f'+dir={directory}',f'+max_cycles={max(10000,2*original["rtl_cycles"])}',
             '+lpn_wall']+[f'+arg{k}={int(v)&0xffffffff}' for k,v in enumerate(task['args'])]
        if index==0:cmd.append('+trace')
        atomic_json(directory/'worker.json',dict(status='simulating',pid=os.getpid(),command=cmd))
        before=time.perf_counter();run=rtlsim.run_sim(cmd,timeout=task['rtl_timeout'])
        process=time.perf_counter()-before
        log=run.stdout+run.stderr;(directory/'rtl.log').write_text(log)
        run.check_returncode()
        result=re.search(r'RESULT cycles=(\d+) halted=(\d+) error=(\d+)',log)
        wall=re.search(r'LPN_WALL seconds=([0-9.]+) cycles=(\d+)',log)
        counts=[int(v) for v in re.findall(r'SLICE \d+ icount=(\d+)',log)]
        if not result or (result[2],result[3])!=('1','0') or not wall or int(wall[2])!=int(result[1]):
            raise RuntimeError(f'kernel {index}: RTL did not halt cleanly or timer is missing')
        if counts!=task['instructions']:raise RuntimeError('RTL/ISA instruction counts differ')
        for sid in range(cfg.S):
            if not files_equal(directory/f'dram_out_{sid}.bin',directory/f'golden_dram_{sid}.bin'):
                raise RuntimeError(f'kernel {index}: full DRAM differs from independent ISA')
            tmem=np.array([int(v,16) for v in (directory/f'tmem_{sid}.hex').read_text().split()],dtype='<u4')
            if len(tmem)!=cfg.TMEM_WORDS or tmem.tobytes()!=(directory/f'golden_tmem_{sid}.bin').read_bytes():
                raise RuntimeError(f'kernel {index}: full scratchpad differs from independent ISA')
        if hashes!={name:digest(directory/name) for name in hashes}:
            raise RuntimeError('RTL input files changed')
        from examples.opentpu.shape_workload import resolve_static
        from examples.opentpu.native_timing import ReusableTiming
        from examples.opentpu.model import Timing
        programs=[[I.Instr.decode(w) for w in json.loads((directory/f'program_{sid}.json').read_text())]
                  for sid in range(cfg.S)]
        workload=resolve_static(cfg,programs,task['args'])
        physical=replace(workload,config=replace(cfg,MCOLS=1,LANES=64))
        timing=Timing(batch_rows=32,matrix_row_lanes=32,dram_bytes_per_cycle=2816,
            dram_port_bytes=2816,dram_latency=8,fifo_chunks=4096,scale_lanes=32,
            unified_dram=True,scalar_store_bytes=256,epilogue_lanes=1)
        prepared=ReusableTiming(Path(task['compiler']),Path(task['cache']),capacity=65536).prepare(
            physical,timing,directory/'timing.tables')
        cycles,lpn_core,_,native_log=prepared.run();(directory/'lpn.log').write_text(native_log)
        observed=int(result[1])
        record=dict(step=index,input_sha256=hashes,instructions=counts,
            rtl_cycles=observed,baseline_rtl_cycles=original['rtl_cycles'],lpn_cycles=cycles,
            architecture_cycle_speedup=original['rtl_cycles']/observed,
            evolved_lpn_relative_cycle_error=cycles/observed-1,
            lpn_core_seconds=lpn_core,rtl_core_seconds=float(wall[1]),
            rtl_process_seconds=process,rtl_build_seconds=build,
            all_dram_and_tmem_bit_exact=True,identical_input_files=True,zero_rollbacks=True,
            simulator=str(exe),baseline_binary_sha256=digest(exe),timing=asdict(timing))
        atomic_json(directory/'rtl-result.json',record)
        atomic_json(directory/'worker.json',dict(status='complete',pid=os.getpid()))
        return record
    except Exception as error:
        atomic_json(directory/'worker.json',dict(status='failed',pid=os.getpid(),
            error_type=type(error).__name__,error=str(error)))
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--opentpu-root',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--baseline-job',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--compiler',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--rtl-timeout',type=float,default=7200)
    a=ap.parse_args()
    if a.workers<1:ap.error('workers must be positive')
    from examples.opentpu.workload import use_checkout,execute
    from examples.opentpu.rtl_evolution.baseline_stream import BaselineReference
    from examples.opentpu.standard_inference import run_engine
    from examples.opentpu.e2e import CheckedBackend,atomic_json,digest
    from examples.opentpu.benchmark import write_inputs
    a.opentpu_root=use_checkout(a.opentpu_root)
    import torch
    from opentpu import Config,rtlsim
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import load_weights
    if rtlsim.MEMORY['AXI'] or rtlsim.MEMORY['BOOT'] or os.environ.get('OTPU_SIM_PLUSARGS'):
        raise ValueError('parallel validation requires the baseline behavioral-memory harness without extra simulation arguments')
    torch.set_num_threads(1)
    reference=BaselineReference(a.baseline,a.baseline_job);info=reference.info
    cfg=Config(**info['config']);spec=load_spec(Path(info['model_directory']))
    weights=load_weights(Path(info['model_directory']))
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    ua=dict(MXU_ROWS=32,LANES=64,VPU_LANES=64,QST_LANES=64,MEM_BYTES=2816)
    files=[a.opentpu_root/'opentpu/rtlsim.py',rtlsim.TB/'tb_top.sv',rtlsim.TB/'lpn_wall.cpp',
           *[rtlsim.RTL/name for name in rtlsim.RTL_SOURCES]]
    sources={str(p.relative_to(a.opentpu_root)):digest(p) for p in files}
    atomic_json(a.output/'identity.json',dict(pid=os.getpid(),config=asdict(cfg),uarch=ua,
        workers=a.workers,scope=__doc__,rtl_source_sha256=sources,
        project_revision=subprocess.check_output(['git','rev-parse','HEAD'],
            cwd=Path(__file__).resolve().parents[3],text=True).strip()))
    pool=ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'))
    tasks={};observations={};backend=None
    def accept(index,record):
        observations[index]=record
        backend.records[index].update(record)
        backend.records[index]['rtl_verification_pending']=False
        backend.save_previous(backend.engine.pos)
        atomic_json(a.output/'kernels'/f'{index:04d}'/'record.json',backend.records[index])
        print(json.dumps(dict(stage='verified_rtl_kernel',**record)),flush=True)
    def progress(status):
        atomic_json(a.output/'progress.json',dict(status=status,pid=os.getpid(),
            prepared_kernels=len(backend.records),verified_kernels=len(observations),workers=a.workers))
    class Backend(CheckedBackend):
        def __init__(self,config,images):
            super().__init__(config,images,SimpleNamespace(output=a.output));self.engine=None
        def attach(self,engine):self.engine=engine
        def save_previous(self,position_after):
            if self.records:
                record=self.records[-1]
                rows=record.get('rows',position_after-record['position_before'])
                if rows>0:record['input_rows']=rows
                atomic_json(a.output/'kernels'/f"{record['kernel_index']:04d}"/'prepared.json',record)
        def read(self,sid,addr,nbytes):
            if self.records:
                logits=self.engine.image.io['logits']
                if logits<=addr<logits+4*self.engine.spec.vocab*self.engine.rows:
                    self.records[-1]['logit_rows_read']+=nbytes//(4*self.engine.image.v_loc)
            return super().read(sid,addr,nbytes)
        def run(self,programs,args=None):
            index=len(self.records);before=self.engine.pos;self.save_previous(before)
            directory=a.output/'kernels'/f'{index:04d}';directory.mkdir(parents=True)
            write_inputs(directory,self.cfg,programs,self.drams,args)
            hashes={name:digest(directory/name) for sid in range(cfg.S)
                    for name in (f'prog_{sid}.hex',f'dram_{sid}.bin')}
            for sid,program in enumerate(programs):
                (directory/f'program_{sid}.json').write_text(json.dumps([ins.encode() for ins in program])+'\n')
            start=time.perf_counter();workload=execute(self.cfg,programs,self.drams,args,hashes)
            functional=time.perf_counter()-start
            for sid,state in enumerate(workload.machine.slices):
                state.dram.tofile(directory/f'golden_dram_{sid}.bin')
                state.tmem.astype('<u4',copy=False).tofile(directory/f'golden_tmem_{sid}.bin')
            self.drams=[s.dram[:len(im)].copy() for s,im in zip(workload.machine.slices,self.drams)]
            record=dict(kernel_index=index,step=index,position_before=before,
                phase='prefill' if before<128 else 'decode',input_rows=None,
                logit_rows_read=0,functional_seconds=functional,rtl_verification_pending=True,
                input_sha256=hashes,instructions=[len(c) for c in workload.commands],args=list(args or []))
            self.records.append(record)
            task=dict(index=index,directory=str(directory),input_sha256=hashes,
                config=asdict(cfg),uarch=ua,opentpu_root=str(a.opentpu_root),
                baseline=str(a.baseline),baseline_job=str(a.baseline_job) if a.baseline_job else None,
                compiler=str(a.compiler),cache=str(a.output/'timing-cache'),
                instructions=record['instructions'],args=list(args or []),rtl_timeout=a.rtl_timeout)
            tasks[pool.submit(check_kernel,task)]=index
            atomic_json(directory/'prepared.json',record)
            for future,i in list(tasks.items()):
                if future.done():accept(i,future.result());del tasks[future]
            progress('preparing')
            return record
    start=time.perf_counter()
    try:
        # run_engine invokes only the upstream constructor and Engine.generate.
        # The factory records the backend early for progress/result bookkeeping.
        def factory(config,images):
            nonlocal backend
            backend=Backend(config,images);return backend
        engine,generated=run_engine(spec,cfg,256,info['prompt_ids'],32,factory,weights)
        if len(generated)!=32:raise RuntimeError('upstream inference stopped before 32 generated tokens')
        backend.save_previous(engine.pos)
        atomic_json(a.output/'functional-report.json',dict(prompt_ids=info['prompt_ids'],
            generated_ids=generated,final_position=engine.pos,prepared_kernels=len(backend.records)))
        progress('validating')
        for future in as_completed(tasks):
            accept(tasks[future],future.result());progress('validating')
        original=reference.complete()
        if generated!=original['generated_ids'] or len(backend.records)!=len(original['kernels']):
            raise RuntimeError('generated IDs or kernel schedule differ from original RTL')
        if sources!={str(p.relative_to(a.opentpu_root)):digest(p) for p in files}:
            raise RuntimeError('RTL source changed during validation')
        records=backend.records
        from tokenizers import Tokenizer
        text=Tokenizer.from_file(str(Path(info['model_directory'])/'tokenizer.json')).decode(generated)
        for record in records:
            record['rtl_verification_pending']=False
            atomic_json(a.output/'kernels'/f"{record['kernel_index']:04d}"/'record.json',record)
        cycles=sum(r['rtl_cycles'] for r in records);base=sum(r['baseline_rtl_cycles'] for r in records)
        report=dict(status='complete',scope=__doc__,model=asdict(spec),config=asdict(cfg),uarch=ua,
            entrypoint='unmodified upstream Engine.generate',prompt_tokens=128,output_tokens=32,
            prompt_ids=info['prompt_ids'],generated_ids=generated,generated_text=text,final_position=engine.pos,
            baseline_rtl_cycles=base,evolved_rtl_cycles=cycles,architecture_cycle_speedup=base/cycles,
            lpn_cycles=sum(r['lpn_cycles'] for r in records),
            lpn_relative_cycle_error=sum(r['lpn_cycles'] for r in records)/cycles-1,
            worst_lpn_cycle_error=max(abs(r['evolved_lpn_relative_cycle_error']) for r in records),
            all_inputs_identical_to_baseline=True,all_dram_and_tmem_bit_exact=True,
            workers=a.workers,validation_wall_seconds=time.perf_counter()-start,kernels=records,
            rtl_source_sha256=sources,baseline_report_sha256=digest(a.baseline))
        atomic_json(a.output/'report.json',report);progress('complete')
        pool.shutdown()
    except Exception as error:
        atomic_json(a.output/'failure.json',dict(error_type=type(error).__name__,error=str(error)))
        if backend is not None:progress('failed')
        # Stop this validation's children; upstream run_sim's parent watcher
        # cleans up each attached simulator once its worker is reaped.
        children=list((pool._processes or {}).values())
        for child in children:child.terminate()
        for child in children:child.join()
        pool.shutdown(wait=False,cancel_futures=True)
        raise

if __name__=='__main__':main()
