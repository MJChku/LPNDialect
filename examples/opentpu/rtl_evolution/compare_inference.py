"""Execute evolved RTL on the baseline's upstream inference and compare inputs.

Accepts a completed report or completed observations from a verified live
original-RTL job. Every evolved kernel must match its
original program, arguments and numerical input hashes. No inference decisions
are made by the hardware model or this backend.
"""
import argparse
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--opentpu-root',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--baseline-job',type=Path,
        help='Optional live original-RTL job manifest; wait for its completed observations')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--compiler',type=Path,required=True)
    ap.add_argument('--rtl-timeout',type=float,default=28800)
    a=ap.parse_args()
    from examples.opentpu.workload import use_checkout,execute
    from examples.opentpu.standard_inference import make_backend,run_engine
    from examples.opentpu.e2e import CheckedBackend,atomic_json,digest
    from examples.opentpu.benchmark import write_inputs
    from examples.opentpu.native_timing import ReusableTiming
    from examples.opentpu.model import Timing
    from examples.opentpu.rtl_evolution.baseline_stream import BaselineReference
    a.opentpu_root=use_checkout(a.opentpu_root)
    import numpy as np
    import torch
    from opentpu import Config,rtlsim
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import load_weights
    torch.set_num_threads(1)
    reference=BaselineReference(a.baseline,a.baseline_job)
    baseline=reference.info
    output=a.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    ua=dict(MXU_ROWS=32,LANES=64,VPU_LANES=64,QST_LANES=64,MEM_BYTES=2816)
    cfg=Config(**baseline['config'])
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='12'
    rtlsim.VFLAGS+=['-CFLAGS','-O3']
    options=SimpleNamespace(output=output)
    timing=Timing(batch_rows=32,matrix_row_lanes=32,
        dram_bytes_per_cycle=2816,dram_port_bytes=2816,dram_latency=8,
        fifo_chunks=4096,scale_lanes=32,unified_dram=True,scalar_store_bytes=256,
        epilogue_lanes=1)
    runner=ReusableTiming(a.compiler,output/'timing-cache',capacity=65536)
    source_files=[p for folder in ('rtl','sim/verilator')
                  for p in sorted((a.opentpu_root/folder).rglob('*'))
                  if p.is_file() and p.suffix in ('.sv','.svh','.cpp','.h')]
    source_files += [a.opentpu_root/'opentpu/rtlsim.py']
    source={str(p.relative_to(a.opentpu_root)):digest(p) for p in source_files}
    atomic_json(output/'identity.json',dict(pid=os.getpid(),config=asdict(cfg),uarch=ua,
        baseline_report_sha256=digest(a.baseline) if a.baseline.exists() else None,
        baseline_job_sha256=digest(a.baseline_job) if a.baseline_job else None,
        rtl_source_sha256=source))
    class ArchitectureBackend(CheckedBackend):
        def run(self,programs,args=None):
            index=len(self.records); expected=reference.kernel(index)
            current=output/'current'; current.mkdir(exist_ok=True)
            write_inputs(current,self.cfg,programs,self.drams,args)
            hashes={name:digest(current/name) for sid in range(self.cfg.S)
                    for name in (f'prog_{sid}.hex',f'dram_{sid}.bin')}
            if hashes!=expected['input_sha256']:
                raise RuntimeError(f'kernel {index}: inputs differ from original RTL baseline')
            before=time.perf_counter()
            workload=execute(self.cfg,programs,self.drams,args,hashes)
            front=time.perf_counter()-before
            predicted=runner.prepare(replace(workload,
                config=replace(self.cfg,MCOLS=1,LANES=64)),timing,current/'timing.tables')
            lpn_cycles,lpn_core,lpn_process,lpn_log=predicted.run()
            (current/'lpn.log').write_text(lpn_log)
            # The original and evolved RTL receive the same ISA configuration
            # and input files; physical datapath widths are RTL parameters.
            before=time.perf_counter()
            exe=rtlsim.build_top(self.cfg,uarch=ua,dram_lat=8)
            build=time.perf_counter()-before
            original=rtlsim.run_sim; captured=[]
            def run_sim(command,timeout=None):
                r=original(command,timeout=a.rtl_timeout); captured.append(r.stdout+r.stderr); return r
            rtlsim.run_sim=run_sim
            before=time.perf_counter()
            try:
                drams,tmems,stats=rtlsim.run(self.cfg,programs,self.drams,args=args,
                    uarch=ua,dram_lat=8,max_cycles=1<<40,trace=index==0,
                    keep=current,plusargs=['+lpn_wall'])
            finally:rtlsim.run_sim=original
            process=time.perf_counter()-before
            (current/'rtl.log').write_text(captured[-1])
            match=re.search(r'LPN_WALL seconds=([0-9.]+) cycles=(\d+)',captured[-1])
            if not match or int(match[2])!=stats['cycles']:
                raise RuntimeError('RTL timer missing or inconsistent')
            if stats['instructions']!=[len(c) for c in workload.commands]:
                raise RuntimeError('instruction counts differ')
            for sid,state in enumerate(workload.machine.slices):
                if not np.array_equal(drams[sid],state.dram) or not np.array_equal(tmems[sid],state.tmem):
                    bad=np.flatnonzero(tmems[sid]!=state.tmem)
                    raise RuntimeError(f'kernel {index}: numerical RTL mismatch in slice {sid}: {bad[:16].tolist()}')
            if hashes!={name:digest(current/name) for name in hashes}:
                raise RuntimeError('RTL input files changed')
            self.drams=[s.dram[:len(im)].copy() for s,im in zip(workload.machine.slices,self.drams)]
            record=dict(step=index,input_sha256=hashes,instructions=stats['instructions'],
                rtl_cycles=stats['cycles'],baseline_rtl_cycles=expected['rtl_cycles'],
                lpn_cycles=lpn_cycles,lpn_core_seconds=lpn_core,
                evolved_lpn_relative_cycle_error=lpn_cycles/stats['cycles']-1,
                architecture_cycle_speedup=expected['rtl_cycles']/stats['cycles'],
                rtl_core_seconds=float(match[1]),rtl_process_seconds=process,rtl_build_seconds=build,
                functional_seconds=front,all_dram_and_tmem_bit_exact=True,identical_input_files=True,
                simulator=str(exe))
            self.records.append(record)
            print(json.dumps(record),flush=True)
            return record
    spec=load_spec(Path(baseline['model_directory']))
    weights=load_weights(Path(baseline['model_directory']))
    before=time.perf_counter()
    engine,generated=run_engine(spec,cfg,256,baseline['prompt_ids'],32,
        make_backend(output,128,options=options,execution_backend=ArchitectureBackend),weights)
    records=engine.backend.records
    baseline=reference.complete()
    if generated!=baseline['generated_ids'] or len(records)!=len(baseline['kernels']):
        raise RuntimeError('generated tokens or kernel schedule differ from baseline')
    cycles=sum(r['rtl_cycles'] for r in records)
    base=sum(r['baseline_rtl_cycles'] for r in records)
    if source!={str(p.relative_to(a.opentpu_root)):digest(p) for p in source_files}:
        raise RuntimeError('RTL sources changed during the experiment')
    report=dict(status='complete',scope='Actual original/evolved RTL architecture cycle comparison; no FPGA Fmax or resource proof.',
        entrypoint='unmodified upstream Engine.generate',model=asdict(spec),config=asdict(cfg),uarch=ua,
        prompt_tokens=128,output_tokens=len(generated),generated_ids=generated,final_position=engine.pos,
        baseline_rtl_cycles=base,evolved_rtl_cycles=cycles,architecture_cycle_speedup=base/cycles,
        lpn_predicted_cycles=sum(r['lpn_cycles'] for r in records),
        lpn_relative_cycle_error=sum(r['lpn_cycles'] for r in records)/cycles-1,
        host_seconds=time.perf_counter()-before,kernels=records,rtl_source_sha256=source,
        baseline_report_sha256=digest(a.baseline),
        all_inputs_identical_to_baseline=True,all_dram_and_tmem_bit_exact=True)
    atomic_json(output/'report.json',report)
    atomic_json(output/'progress.json',dict(status='complete',pid=os.getpid(),completed_kernels=len(records)))
    print(json.dumps({k:v for k,v in report.items() if k not in ('kernels','rtl_source_sha256')}),flush=True)

if __name__=='__main__':main()
