"""Collect actual original-RTL cycles, retaining predictor misses explicitly.

Reuse is limited to numerically verified prior RTL observations whose input
hashes and simulator binary match. The upstream inference/ISA is replayed to
reconstruct persistent state; no scheduling or numerical result is invented.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--opentpu-root',type=Path,required=True)
    ap.add_argument('--model-dir',type=Path,required=True)
    ap.add_argument('--prompt-json',type=Path,required=True)
    ap.add_argument('--reuse',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--compiler',type=Path,required=True)
    a=ap.parse_args()
    from examples.opentpu.workload import use_checkout,execute
    from examples.opentpu.benchmark import write_inputs
    from examples.opentpu.e2e import CheckedBackend,atomic_json,digest
    from examples.opentpu.standard_inference import make_backend,run_engine
    a.opentpu_root=use_checkout(a.opentpu_root)
    import numpy as np
    import torch
    from opentpu import rtlsim
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import load_weights,device_config
    torch.set_num_threads(1)
    spec=load_spec(a.model_dir); weights=load_weights(a.model_dir)
    prompt=json.loads(a.prompt_json.read_text())['ids']
    if len(prompt)!=128:raise ValueError('expected 128 prompt tokens')
    cfg=device_config(spec,256,S=1,D=64,MCOLS=4,ACT_ROWS=16,
        ACT_BLOCKS=64,LANES=8,IMEM_WORDS=32768,TMEM_WORDS=65536)
    a.output.mkdir(parents=True,exist_ok=False)
    rtlsim.VFLAGS[rtlsim.VFLAGS.index('-j')+1]='4'; rtlsim.VFLAGS+=['-CFLAGS','-O3']
    exe=rtlsim.build_top(cfg,dram_lat=8); binary_sha=digest(exe)
    cached={}
    for path in sorted((a.reuse/'kernels').glob('*/record.json')):
        record=json.loads(path.read_text());cached[record['step']]=(path,record)
    failed=a.reuse/'failed-step.json'
    if failed.exists():
        record=json.loads(failed.read_text())
        if record.get('all_dram_and_tmem_bit_exact') and record.get('identical_input_files'):
            cached.setdefault(record['step'],(failed,record))
    for _,record in cached.values():
        if record['baseline_binary_sha256']!=binary_sha:
            raise ValueError('cached observation uses another RTL executable')
        if not record['all_dram_and_tmem_bit_exact'] or not record['identical_input_files']:
            raise ValueError('cached observation lacks numerical/input verification')
    source_files=[a.opentpu_root/'opentpu/rtlsim.py',*sorted((a.opentpu_root/'rtl').rglob('*.sv')),
                  a.opentpu_root/'sim/verilator/tb_top.sv',a.opentpu_root/'sim/verilator/lpn_wall.cpp']
    sources={str(p.relative_to(a.opentpu_root)):digest(p) for p in source_files}
    options=SimpleNamespace(output=a.output,lpn_compiler=a.compiler,batch_rows=8,
        reuse_timing=True,command_capacity=65536,rtl_timeout=28800,fail_timing_error=False)
    atomic_json(a.output/'identity.json',dict(pid=os.getpid(),config=asdict(cfg),
        reused_records={str(i):str(path) for i,(path,_) in cached.items()},
        binary_sha256=binary_sha,rtl_source_sha256=sources))
    class BaselineBackend(CheckedBackend):
        def run(self,programs,args=None):
            index=len(self.records)
            if index not in cached:return super().run(programs,args)
            path,previous=cached[index]
            current=a.output/'current'; current.mkdir(exist_ok=True)
            write_inputs(current,self.cfg,programs,self.drams,args)
            hashes={name:digest(current/name) for sid in range(self.cfg.S)
                    for name in (f'prog_{sid}.hex',f'dram_{sid}.bin')}
            if hashes!=previous['input_sha256']:
                raise RuntimeError(f'kernel {index}: reuse input hashes differ')
            before=time.perf_counter();workload=execute(self.cfg,programs,self.drams,args,hashes)
            # The final saved bundle permits a fresh full memory comparison as well.
            if path==failed:
                for sid,state in enumerate(workload.machine.slices):
                    old=a.reuse/'current'
                    dram=np.fromfile(old/f'dram_out_{sid}.bin',dtype=np.uint8)
                    tmem=np.array([int(x,16) for x in (old/f'tmem_{sid}.hex').read_text().split()],dtype=np.uint32)
                    if not np.array_equal(dram,state.dram) or not np.array_equal(tmem,state.tmem):
                        raise RuntimeError('retained final cached RTL output differs from fresh ISA execution')
            record=dict(previous,rtl_reused_from=str(path),reuse_input_hashes_reverified=True,
                replay_functional_seconds=time.perf_counter()-before)
            self.drams=[state.dram[:len(im)].copy() for state,im in zip(workload.machine.slices,self.drams)]
            self.records.append(record)
            print(json.dumps(dict(stage='reused_verified_rtl',step=index,rtl_cycles=record['rtl_cycles'])),flush=True)
            return record
    before=time.perf_counter()
    engine,generated=run_engine(spec,cfg,256,prompt,32,
        make_backend(a.output,len(prompt),options=options,execution_backend=BaselineBackend),weights)
    records=engine.backend.records
    if sources!={str(p.relative_to(a.opentpu_root)):digest(p) for p in source_files}:
        raise RuntimeError('RTL sources changed during baseline run')
    report=dict(status='complete',mode='validate-rtl',entrypoint='unmodified upstream Engine.generate',
        scope='Actual original-RTL observations. LPN timing misses are recorded; completion is not a claim that the timing predictor met its 5% target.',
        model=asdict(spec),config=asdict(cfg),model_directory=str(a.model_dir.resolve()),
        prompt_ids=prompt,prompt_tokens=len(prompt),requested_output_tokens=32,
        returned_output_tokens=len(generated),generated_ids=generated,final_position=engine.pos,
        host_seconds=time.perf_counter()-before,kernels=records,rtl_source_sha256=sources,
        all_device_memories_bit_exact=all(r['all_dram_and_tmem_bit_exact'] for r in records),
        all_input_files_identical=all(r['identical_input_files'] for r in records),
        timing_target_met_all_kernels=all(r['accuracy_within_5_percent'] for r in records),
        worst_lpn_cycle_error=max(abs(r['relative_cycle_error']) for r in records),
        rtl_cycles=sum(r['rtl_cycles'] for r in records))
    atomic_json(a.output/'report.json',report)
    atomic_json(a.output/'progress.json',dict(status='complete',pid=os.getpid(),completed_kernels=len(records)))
    print(json.dumps({k:v for k,v in report.items() if k not in ('kernels','rtl_source_sha256')}),flush=True)

if __name__=='__main__':main()
