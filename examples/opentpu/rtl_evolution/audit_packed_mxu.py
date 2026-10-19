"""Recheck packed/unpacked MXU gates, full memories, cycle identity and native timing."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import subprocess
import sys


def audit(directory,root,compiler,output):
    from examples.opentpu.workload import use_checkout
    from examples.opentpu.e2e import digest,atomic_json
    from examples.opentpu.weight_storage import read_image
    from examples.opentpu.shape_workload import resolve_static
    from examples.opentpu.model import Timing
    from examples.opentpu.native_timing import ReusableTiming
    from .parallel_inference import simulator_cache_key
    use_checkout(root)
    import numpy as np
    from opentpu import Config,isa as I,rtlsim
    from opentpu.isasim import Machine
    report=json.loads((directory/'report.json').read_text())
    if report['status']!='complete' or len(report['records'])!=11:
        raise ValueError('requires nine primitive and two queued gates')
    vendor_model=report.get('vendor_model')
    if report['uarch'].get('MXU_PAIR_TECH',0):
        if not vendor_model:raise ValueError('vendor gate lacks model identity')
        from .vendor_model import install
        if install(rtlsim,Path(vendor_model['root']))!=vendor_model:
            raise ValueError('vendor model identity changed')
    for name,expected in report['source_sha256'].items():
        if digest(Path(name))!=expected:raise ValueError('gate/RTL sources changed')
    gate=next(Path(n) for n in report['source_sha256'] if n.endswith('/check_matrix.py'))
    if digest(gate.with_name('check_matrix_pipeline.py'))!=report['pipeline_script_sha256']:
        raise ValueError('queued gate script changed')
    cfg=Config(**report['config']);ua=report['uarch'];cols=ua.get('MXU_COLS',1);rows=ua['MXU_ROWS']
    rtlsim.VFLAGS[:]=report['rtl_compile_flags']
    key=simulator_cache_key(rtlsim,cfg,ua);binary=root/'build/verilator'/key/'Vtb_top'
    if not binary.is_file():raise ValueError('missing gate binary')
    binary_hash=digest(binary);results=[]
    output.mkdir(parents=True,exist_ok=False)
    for index,record in enumerate(report['records']):
        source=directory/record['case']['name'];log_path=source/'rtl.log';log=log_path.read_text()
        if record.get('simulator') and (Path(record['simulator'])!=binary or record['simulator_sha256']!=binary_hash):
            raise ValueError('binary identity differs')
        identity={n:digest(source/n) for n in ('prog_0.hex','dram_0.bin')}
        if record.get('input_sha256') and identity!=record['input_sha256']:raise ValueError('input changed')
        if record.get('rtl_log_sha256') and digest(log_path)!=record['rtl_log_sha256']:raise ValueError('raw log changed')
        result=re.search(r'RESULT cycles=(\d+) halted=1 error=0',log)
        if not result or int(result[1])!=record['rtl_cycles']:raise ValueError('raw completion differs')
        words=[int(w,16) for w in (source/'prog_0.hex').read_text().split()]
        words += [0]*((-len(words))%8)
        program=[I.Instr.decode(words[i:i+8]) for i in range(0,len(words),8)]+[I.halt()]
        work=resolve_static(cfg,[program])
        if [int(x) for x in re.findall(r'SLICE \d+ icount=(\d+)',log)]!=[len(work.commands[0])]:
            raise ValueError('instruction count differs')
        golden=Machine(cfg,[program],[read_image(source/'dram_0.bin',cfg.DRAM_BYTES)]).run().slices[0]
        if (source/'dram_out_0.bin').read_bytes()!=golden.dram.tobytes():raise ValueError('full DRAM differs from ISA')
        tmem=np.array([int(w,16) for w in (source/'tmem_0.hex').read_text().split()],dtype='<u4')
        if len(tmem)!=cfg.TMEM_WORDS or not np.array_equal(tmem,golden.tmem):raise ValueError('full TMEM differs from ISA')
        entry=dict(case=record['case'],rtl_cycles=record['rtl_cycles'],input_sha256=identity,
                   output_sha256={n:digest(source/n) for n in ('dram_out_0.bin','tmem_0.hex')},
                   raw_log_sha256=digest(log_path),historical_input_and_log_hashes_recorded=bool(record.get('input_sha256')),
                   all_full_dram_and_tmem_bit_exact=True)
        if compiler:
            def latency(define):
                return int(next(f.split('=')[1] for f in rtlsim.VFLAGS if f.startswith('-D'+define+'=')))
            timing=Timing(batch_rows=rows,matrix_row_lanes=rows,fifo_chunks=128*rows,scale_lanes=rows,
                dram_bytes_per_cycle=ua.get('MEM_BYTES',2816),dram_port_bytes=ua.get('MEM_BYTES',2816),
                unified_dram=True,scalar_store_bytes=256,epilogue_lanes=max(1,cols//2),
                quant_row_banks=8,quant_store_buffered=True,vector_row_contexts=4,
                vector_command_queue=2,matrix_retirement=True,physical_scale_lines=True,
                activation_streaming=bool(ua.get('ACT_STREAMING',0)),
                fp_mul_latency=latency('OTPU_FP_MUL_LAT'),fp_add_latency=latency('OTPU_FP_ADD_LAT'))
            physical=replace(work,config=replace(cfg,MCOLS=cols))
            native=ReusableTiming(compiler,output/'native-cache',capacity=128).prepare(physical,timing,output/f'{index}.tables')
            cycles,_,_,text=native.run();(output/f'{index}.log').write_text(text)
            entry.update(lpn_cycles=cycles,relative_cycle_error=cycles/record['rtl_cycles']-1)
        results.append(entry)
    result=dict(status='complete',config=report['config'],uarch=ua,vendor_model=vendor_model,source_sha256=report['source_sha256'],
        report_sha256=digest(directory/'report.json'),binary_sha256=binary_hash,records=results,
        limitations=['Queued gate inputs/logs are freshly hashed; original gate did not record historical hashes',
                     'Synthetic gates, not full inference or physical FPGA mapping'])
    atomic_json(output/'report.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packed-root','control-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--packed',type=Path,nargs='+',required=True)
    p.add_argument('--controls',type=Path,nargs='+',required=True)
    p.add_argument('--compiler',type=Path,required=True)
    a=p.parse_args()
    from examples.opentpu.e2e import atomic_json,digest
    if len(a.packed)!=len(a.controls):raise ValueError('one control per packed gate required')
    model_root=Path(__file__).resolve().parents[1]
    files=[model_root/n for n in ('model.py','compact.py','components.py','activation_stream.py','shape_workload.py','native_timing.py','weight_storage.py','workload.py')]+[Path(__file__)]
    hashes={str(f):digest(f) for f in files}
    a.output.mkdir(parents=True,exist_ok=False);results=[]
    for index,(paired,control) in enumerate(zip(a.packed,a.controls)):
        outputs=[]
        for name,directory,root,compiler in [('packed',paired,a.packed_root,a.compiler),('control',control,a.control_root,None)]:
            out=a.output/f'{index}-{name}'
            # Each checkout gets its own Python process to isolate opentpu imports.
            code='from pathlib import Path; from examples.opentpu.rtl_evolution.audit_packed_mxu import audit; import sys; audit(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]) if sys.argv[3] else None,Path(sys.argv[4]))'
            subprocess.run([sys.executable,'-c',code,str(directory),str(root),str(compiler) if compiler else '',str(out)],check=True)
            outputs.append(json.loads((out/'report.json').read_text()))
        changed,baseline=outputs
        if changed['config']!=baseline['config'] or changed['uarch']!=baseline['uarch']|{'MXU_PAIR_DOT':1}:
            raise ValueError('configuration changed beyond paired DSP')
        for x,y in zip(changed['records'],baseline['records']):
            for field in ('case','input_sha256','output_sha256','rtl_cycles'):
                if x[field]!=y[field]:raise ValueError('packed/unpacked gate differs: '+field)
            results.append(dict(columns=changed['uarch']['MXU_COLS'],**x))
    if hashes!={str(f):digest(f) for f in files}:raise ValueError('model sources changed')
    result=dict(status='complete',scope=__doc__,records=results,source_sha256=hashes,
        all_input_programs_full_outputs_and_cycles_identical=True,
        timing_target_met_all_cases=all(abs(x['relative_cycle_error'])<=.1 for x in results),
        worst_case_relative_cycle_error=max(abs(x['relative_cycle_error']) for x in results),
        script_sha256=digest(Path(__file__)),
        limitations=['Behavioral packed MXU integration; DSP48E2 component separately vendor-model verified',
                     'No integrated vendor primitive, full inference, DSP58 or FPGA mapping/clock proof'])
    atomic_json(a.output/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
