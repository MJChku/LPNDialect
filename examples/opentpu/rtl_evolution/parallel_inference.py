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


def configure_rtlsim(rtlsim,fp_mul_latency=2,fp_add_latency=4):
    # A pool worker handles multiple kernels. Repeated flags alter upstream's
    # cache key even though they do not change the generated simulator.
    if fp_mul_latency not in (1,2) or fp_add_latency not in (1,2,3,4):
        raise ValueError('invalid arithmetic pipeline configuration')
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='12'
    if not any(rtlsim.VFLAGS[i:i+2]==['-CFLAGS','-O3'] for i in range(len(rtlsim.VFLAGS)-1)):
        rtlsim.VFLAGS+=['-CFLAGS','-O3']
    # Reused workers may select another architecture. Remove previous defines
    # before installing this one; defaults preserve the earlier build key.
    rtlsim.VFLAGS[:]=[flag for flag in rtlsim.VFLAGS if not
        flag.startswith(('-DOTPU_FP_MUL_LAT=','-DOTPU_FP_ADD_LAT='))]
    if (fp_mul_latency,fp_add_latency)!=(2,4):
        rtlsim.VFLAGS += [f'-DOTPU_FP_MUL_LAT={fp_mul_latency}',f'-DOTPU_FP_ADD_LAT={fp_add_latency}']



def validate_variant(timing,uarch):
    if timing.get('matrix_engines',1)!=1:
        raise ValueError('multiple matrix engines have no matching RTL implementation')
    if timing.get('pinned_cache_bytes',0):
        raise ValueError('pinned cache proposal has no matching RTL implementation')
    if timing.get('reciprocal_lookup_lanes',0):
        raise ValueError('wide reciprocal proposal has no matching RTL implementation')
    for field,parameter in (('activation_streaming','ACT_STREAMING'),('quant_scale_lookup','QUANT_LOOKUP'),('scalar_recip_lookup','VPU_SCALAR_RECIP')):
        if bool(uarch.get(parameter,0))!=bool(timing.get(field,False)):
            raise ValueError(f'RTL {parameter} setting differs from LPN timing')


def simulator_cache_key(rtlsim,cfg,uarch):
    sources=[rtlsim.RTL/s for s in rtlsim.RTL_SOURCES]+[rtlsim.TB/'lpn_wall.cpp',rtlsim.TB/'tb_top.sv']
    return rtlsim.build_dir('tb_top',sources,rtlsim.top_params(cfg,8,uarch,False,None)).name


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
        if task['args']!=original['args']:
            raise RuntimeError(f'kernel {index}: argument registers differ from original RTL')
        from examples.opentpu.retained_inputs import verified_program
        programs=[];program_json_hashes={}
        for sid in range(cfg.S):
            encoded,identity=verified_program(directory,sid)
            programs.append([I.Instr.decode(word) for word in encoded])
            program_json_hashes[f'program_{sid}.json']=identity
        requested=task.get('timing') or {}
        validate_variant(requested,task['uarch'])
        lookup_hashes={};lookup_args=[]
        if requested.get('quant_scale_lookup',False) or requested.get('scalar_recip_lookup',False):
            from .rom_inputs import reciprocal_inputs
            _,lookup_hashes,lookup_args=reciprocal_inputs(task['recip_table'])
        configure_rtlsim(rtlsim,requested.get('fp_mul_latency',2),requested.get('fp_add_latency',4))
        atomic_json(directory/'worker.json',dict(status='building',pid=os.getpid()))
        reused=None
        if task.get('reuse_directory'):
            source=Path(task['reuse_directory'])/f'{index:04d}'
            previous=source/'rtl-result.json'
            if previous.exists():
                reused=json.loads(previous.read_text())
                identity=source/'input.json'
                # Early retained-input replays omitted input.json. Their
                # launcher can supply identity only after verifying the parent
                # report and its original workload reference hash.
                prior_input=(json.loads(identity.read_text()) if identity.exists()
                             else task['reuse_input_identity'])
                if prior_input['args']!=task['args'] or prior_input['config']!=task['config']:
                    raise RuntimeError('retained input arguments or configuration differ')
                if (reused['input_sha256']!=hashes or not reused['all_dram_and_tmem_bit_exact'] or
                    not reused['identical_input_files'] or reused['step']!=index):
                    raise RuntimeError(f'kernel {index}: retained RTL evidence differs')
                if reused.get('lookup_table_sha256',{})!=lookup_hashes:
                    raise RuntimeError('retained lookup initialization data differs')
                exe=Path(reused['simulator'])
                if exe.parent.name!=simulator_cache_key(rtlsim,cfg,task['uarch']):
                    raise RuntimeError('retained simulator source/parameters/flags cache key differs')
                if digest(exe)!=reused['baseline_binary_sha256']:
                    raise RuntimeError('retained simulator binary changed')
                for sid in range(cfg.S):
                    for name in (f'dram_out_{sid}.bin',f'tmem_{sid}.hex'):
                        os.link(source/name,directory/name)
                build=0
        if reused is None:
            before=time.perf_counter()
            exe=rtlsim.build_top(cfg,uarch=task['uarch'],dram_lat=8)
            build=time.perf_counter()-before
        # This is the same behavioral-memory harness as rtlsim.run. Files were
        # already serialized once by its canonical writer in the producer.
        cmd=[str(exe),f'+dir={directory}',f'+max_cycles={max(10000,2*original["rtl_cycles"])}',
             '+lpn_wall']+[f'+arg{k}={int(v)&0xffffffff}' for k,v in enumerate(task['args'])]
        cmd+=lookup_args
        if index==0:cmd.append('+trace')
        atomic_json(directory/'worker.json',dict(
            status='simulating' if reused is None else 'checking_retained_rtl',pid=os.getpid(),command=cmd))
        if reused is None:
            before=time.perf_counter();run=rtlsim.run_sim(cmd,timeout=task['rtl_timeout'])
            process=time.perf_counter()-before
            log=run.stdout+run.stderr;(directory/'rtl.log').write_text(log)
            run.check_returncode()
        else:
            process=reused['rtl_process_seconds'];log=(source/'rtl.log').read_text()
            (directory/'rtl.log').write_text(log)
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
        workload=resolve_static(cfg,programs,task['args'])
        physical=replace(workload,config=replace(cfg,MCOLS=1,LANES=64,
            ACT_ROWS=task['uarch'].get('ACT_ROWS',cfg.ACT_ROWS),
            ACT_BLOCKS=task['uarch'].get('ACT_BLOCKS',cfg.ACT_BLOCKS)))
        if 'ACT_ROWS' in task['uarch']:
            from examples.opentpu.storage_envelope import command_bounds
            for items in workload.commands:
                bounds=command_bounds(items,I)
                if (bounds['activation_rows']>physical.config.ACT_ROWS or
                    bounds['activation_blocks']>physical.config.ACT_BLOCKS):
                    raise ValueError('program exceeds smaller physical activation storage')
        timing=Timing(**task['timing']) if task.get('timing') else Timing(batch_rows=32,matrix_row_lanes=32,dram_bytes_per_cycle=2816,
            dram_port_bytes=2816,dram_latency=8,fifo_chunks=4096,scale_lanes=32,
            unified_dram=True,scalar_store_bytes=256,epilogue_lanes=1)
        prepared=ReusableTiming(Path(task['compiler']),Path(task['cache']),capacity=65536).prepare(
            physical,timing,directory/'timing.tables')
        cycles,lpn_core,_,native_log=prepared.run();(directory/'lpn.log').write_text(native_log)
        for name,expected in program_json_hashes.items():
            if digest(directory/name)!=expected:raise RuntimeError('decoded timing program changed during validation')
        observed=int(result[1])
        record=dict(step=index,input_sha256=hashes,instructions=counts,canonical_program_json_verified=True,
            program_json_sha256=program_json_hashes,
            rtl_cycles=observed,baseline_rtl_cycles=original['rtl_cycles'],lpn_cycles=cycles,
            architecture_cycle_speedup=original['rtl_cycles']/observed,
            evolved_lpn_relative_cycle_error=cycles/observed-1,
            lpn_core_seconds=lpn_core,rtl_core_seconds=float(wall[1]),
            rtl_process_seconds=process,rtl_build_seconds=build,
            all_dram_and_tmem_bit_exact=True,identical_input_files=True,zero_rollbacks=True,
            simulator=str(exe),baseline_binary_sha256=digest(exe),timing=asdict(timing),
            lookup_table_sha256=lookup_hashes,rtl_compile_flags=list(rtlsim.VFLAGS),rtl_top_parameters=rtlsim.top_params(cfg,8,task['uarch'],False,None))
        if reused is not None:record['rtl_reused_from']=str(previous)
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
    ap.add_argument('--reuse-rtl',type=Path,
        help='Reuse completed identical-source RTL kernels after fresh input and full numerical checks')
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
    if a.reuse_rtl is not None:
        prior=json.loads((a.reuse_rtl/'identity.json').read_text())
        if prior['rtl_source_sha256']!=sources or prior['uarch']!=ua or prior['config']!=asdict(cfg):
            raise ValueError('retained RTL sources or configuration differ')
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
                instructions=record['instructions'],args=list(args or []),rtl_timeout=a.rtl_timeout,
                reuse_directory=str(a.reuse_rtl/'kernels') if a.reuse_rtl is not None else None)
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
