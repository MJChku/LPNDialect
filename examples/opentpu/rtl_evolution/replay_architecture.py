"""Run changed hardware on retained normal-inference programs and full ISA states.

No tensor prepass is repeated. Every new RTL result must equal the independently
computed retained golden memory at every kernel boundary. This validates a new
hardware implementation on the original upstream inference, not a new schedule.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('case','baseline','opentpu-root','compiler','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--indices',default='0,15,16,47',help='Kernel indices or all')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--rtl-timeout',type=float,default=7200)
    p.add_argument('--reuse-rtl',type=Path,
        help='Recheck completed identical-hardware RTL outputs with new LPN timing details')
    p.add_argument('--matrix-rows',type=int,choices=(16,24,32,48,64,96,112,128),default=128)
    p.add_argument('--matrix-columns',type=int,choices=(1,2,4,8),default=1)
    p.add_argument('--paired-dsp',action='store_true')
    p.add_argument('--act-rows',type=int,choices=(8,16))
    p.add_argument('--fifo-depth',type=int,choices=(32,64,128),default=128)
    p.add_argument('--activation-streaming',action='store_true')
    p.add_argument('--lookup-scale',action='store_true')
    p.add_argument('--scalar-recip-lookup',action='store_true')
    p.add_argument('--recip-table',type=Path)
    p.add_argument('--fp-mul-latency',type=int,choices=(1,2),default=2)
    p.add_argument('--fp-add-latency',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--vector-command-queue',type=int,choices=(0,2),default=0)
    p.add_argument('--matrix-retirement',action='store_true')
    p.add_argument('--physical-scale-lines',action='store_true')
    a=p.parse_args()
    if a.paired_dsp and a.matrix_columns<2:p.error('paired DSP requires multiple columns')
    if (a.lookup_scale or a.scalar_recip_lookup) and a.recip_table is None:p.error('lookup hardware requires --recip-table')
    from examples.opentpu.e2e import atomic_json,digest
    from examples.opentpu.model import Timing
    from examples.opentpu.workload import use_checkout
    from .parallel_inference import check_kernel,files_equal
    case=a.case.resolve()
    reference=json.loads((case/'report.json').read_text())
    if (reference['status']!='complete' or reference['prompt_tokens']!=128 or reference['output_tokens']!=32 or
        reference['entrypoint']!='unmodified upstream Engine.generate' or
        not reference['all_dram_and_tmem_bit_exact'] or not reference['all_inputs_identical_to_baseline']):
        raise ValueError('requires completed, verified upstream 128/32 inference')
    indices=list(range(48)) if a.indices=='all' else sorted(set(map(int,a.indices.split(','))))
    if a.workers<1 or not indices or not set(indices)<=set(range(48)):
        p.error('positive workers and indices in [0,47] required')
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    use_checkout(a.opentpu_root.resolve())
    from opentpu import rtlsim
    sources=[rtlsim.RTL/name for name in rtlsim.RTL_SOURCES]+[rtlsim.TB/'tb_top.sv',rtlsim.TB/'lpn_wall.cpp']
    project=Path(__file__).resolve().parents[1]
    sources += [project/name for name in ('model.py','components.py','compact.py','native_timing.py','shape_workload.py','activation_stream.py','retained_inputs.py')]
    if a.act_rows is not None:sources.append(project/'storage_envelope.py')
    sources += [Path(__file__).resolve(),Path(__file__).with_name('parallel_inference.py')]
    if a.lookup_scale or a.scalar_recip_lookup:
        from .rom_inputs import reciprocal_inputs
        table_files,_,_=reciprocal_inputs(a.recip_table)
        sources += table_files+[a.recip_table.resolve()/'report.json',Path(__file__).with_name('rom_inputs.py')]
    hashes={str(s):digest(s) for s in sources}
    if a.reuse_rtl:
        retained=json.loads((a.reuse_rtl/'report.json').read_text())
        if (retained['status']!='complete' or
            retained['reference_report_sha256']!=digest(case/'report.json') or
            not set(indices)<=set(retained['indices'])):
            raise ValueError('RTL reuse requires a completed replay of the same selected inputs')
        rtl_files=[rtlsim.RTL/name for name in rtlsim.RTL_SOURCES]+[rtlsim.TB/'tb_top.sv',rtlsim.TB/'lpn_wall.cpp']
        if any(retained['rtl_and_model_source_sha256'].get(str(s))!=digest(s) for s in rtl_files):
            raise ValueError('retained RTL source files differ')
        if any(not (a.reuse_rtl/'kernels'/f'{i:04d}'/'rtl-result.json').exists() for i in indices):
            raise ValueError('retained RTL kernel result is missing')
    uarch=dict(reference['uarch'],MXU_ROWS=a.matrix_rows)
    if a.matrix_columns!=1:uarch['MXU_COLS']=a.matrix_columns
    if a.paired_dsp:uarch['MXU_PAIR_DOT']=1
    if a.act_rows is not None:uarch['ACT_ROWS']=a.act_rows
    if a.fifo_depth!=128:uarch['FIFO_DEPTH']=a.fifo_depth
    if a.activation_streaming:uarch['ACT_STREAMING']=1
    if a.lookup_scale:uarch['QUANT_LOOKUP']=1
    if a.scalar_recip_lookup:uarch['VPU_SCALAR_RECIP']=1
    timing=Timing(batch_rows=a.matrix_rows,matrix_row_lanes=a.matrix_rows,dram_bytes_per_cycle=2816,dram_port_bytes=2816,
        dram_latency=8,fifo_chunks=a.fifo_depth*a.matrix_rows,scale_lanes=a.matrix_rows,unified_dram=True,scalar_store_bytes=256,
        epilogue_lanes=max(1,a.matrix_columns//2),quant_row_banks=8,quant_store_buffered=True,vector_row_contexts=4,
        fp_mul_latency=a.fp_mul_latency,fp_add_latency=a.fp_add_latency,activation_streaming=a.activation_streaming,quant_scale_lookup=a.lookup_scale,scalar_recip_lookup=a.scalar_recip_lookup,
        vector_command_queue=a.vector_command_queue,matrix_retirement=a.matrix_retirement,physical_scale_lines=a.physical_scale_lines)
    records=[];failures=[];started=time.perf_counter()
    def save(status):
        result=dict(status=status,pid=os.getpid(),scope=__doc__,case=str(case),
            reference_report_sha256=digest(case/'report.json'),rtl_and_model_source_sha256=hashes,
            project_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),
            timing=asdict(timing),uarch=uarch,indices=indices,workers=a.workers,
            seconds=time.perf_counter()-started,records=sorted(records,key=lambda r:r['step']),failures=failures,
            upstream_schedule_unchanged=True,all_dram_and_tmem_bit_exact=bool(records) and
                all(r['all_dram_and_tmem_bit_exact'] for r in records),
            numerical_prepass_repeated=False)
        if a.reuse_rtl:
            result.update(retained_rtl_report=str(a.reuse_rtl.resolve()/'report.json'),
                retained_rtl_report_sha256=digest(a.reuse_rtl/'report.json'))
        if len(records)==len(indices):
            rtl=sum(r['rtl_cycles'] for r in records);predicted=sum(r['lpn_cycles'] for r in records)
            result.update(selected_rtl_cycles=rtl,selected_lpn_cycles=predicted,
                selected_speedup_vs_original=sum(r['baseline_rtl_cycles'] for r in records)/rtl,
                relative_cycle_error=predicted/rtl-1,
                worst_kernel_relative_cycle_error=max(abs(r['evolved_lpn_relative_cycle_error']) for r in records),
                timing_target_met_all_kernels=all(abs(r['evolved_lpn_relative_cycle_error'])<=.10 for r in records))
            if indices==list(range(48)):
                result.update(full_inference_rtl_cycles=rtl,full_inference_lpn_cycles=predicted,
                    full_cycle_speedup=reference['baseline_rtl_cycles']/rtl,
                    incremental_cycle_speedup=reference['evolved_rtl_cycles']/rtl)
        atomic_json(output/'report.json',result)
    save('preparing')
    tasks=[]
    for index in indices:
        record=reference['kernels'][index]
        if not record['all_dram_and_tmem_bit_exact'] or not record['identical_input_files']:
            raise ValueError('unverified retained kernel')
        source=case/'kernels'/f'{index:04d}';target=output/'kernels'/f'{index:04d}'
        target.mkdir(parents=True,exist_ok=False)
        identity=dict(config=reference['config'],args=record['args'])
        atomic_json(target/'input.json',identity)
        for sid in range(reference['config']['S']):
            if not files_equal(source/f'golden_dram_{sid}.bin',source/f'dram_out_{sid}.bin'):
                raise ValueError('retained ISA golden DRAM no longer matches verified RTL')
            # These links are inputs only; all new RTL outputs have new paths.
            for name in (f'prog_{sid}.hex',f'dram_{sid}.bin',f'program_{sid}.json',
                         f'golden_dram_{sid}.bin',f'golden_tmem_{sid}.bin'):
                os.link(source/name,target/name)
        task=dict(index=index,directory=str(target),config=reference['config'],uarch=uarch,
            opentpu_root=str(a.opentpu_root.resolve()),baseline=str(a.baseline.resolve()),baseline_job=None,
            compiler=str(a.compiler.resolve()),cache=str(output/'timing-cache'),timing=asdict(timing),
            instructions=record['instructions'],args=record['args'],input_sha256=record['input_sha256'],
            rtl_timeout=a.rtl_timeout,reuse_directory=str(a.reuse_rtl.resolve()/'kernels') if a.reuse_rtl else None,
            reuse_input_identity=identity if a.reuse_rtl else None,
            recip_table=str(a.recip_table.resolve()) if a.lookup_scale or a.scalar_recip_lookup else None)
        tasks.append(task)
    save('running')
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(check_kernel,t) for t in tasks]):
            try:
                record=future.result();records.append(record);print(json.dumps(record),flush=True)
            except Exception as error:
                failures.append(repr(error));print(repr(error),flush=True)
            save('running')
    if hashes!={str(s):digest(s) for s in sources}:
        raise RuntimeError('hardware/model sources changed during validation')
    save('failed' if failures else 'complete')
    if failures:raise RuntimeError('hardware validation failures; inspect report')


if __name__=='__main__':main()
