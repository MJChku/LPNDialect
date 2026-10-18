"""Audit numerical gates and compare multi-column native LPN timing to raw RTL."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('opentpu-root','compiler','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gates',type=Path,nargs='+',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from examples.opentpu.workload import use_checkout
    from examples.opentpu.e2e import digest,atomic_json
    from examples.opentpu.shape_workload import resolve_static
    from examples.opentpu.weight_storage import read_image
    from examples.opentpu.native_timing import ReusableTiming
    from examples.opentpu.model import Timing
    from .parallel_inference import simulator_cache_key
    use_checkout(a.opentpu_root.resolve())
    import numpy as np
    from opentpu import Config,isa as I,rtlsim
    from opentpu.isasim import Machine
    root=Path(__file__).resolve().parents[1]
    files=[root/n for n in ('model.py','compact.py','components.py','activation_stream.py','shape_workload.py','native_timing.py','weight_storage.py')]+[Path(__file__)]
    hashes={str(f):digest(f) for f in files};results=[];reports={}
    for directory in a.gates:
        path=directory/'report.json';report=json.loads(path.read_text());reports[str(path)]=digest(path)
        if report['status']!='complete' or len(report['records'])!=9:raise ValueError('complete nine-case gate required')
        for name,expected in report['source_sha256'].items():
            if digest(Path(name))!=expected:raise ValueError('RTL/gate sources changed')
        cfg=Config(**report['config']);ua=report['uarch'];cols=ua.get('MXU_COLS',1);rows=ua['MXU_ROWS']
        if cfg.S!=1 or cfg.LANES!=64:raise ValueError('comparison supports one 64-lane slice')
        rtlsim.VFLAGS[:]=report['rtl_compile_flags']
        for k,record in enumerate(report['records']):
            source=directory/record['case']['name']
            for name,expected in record['input_sha256'].items():
                if digest(source/name)!=expected:raise ValueError('gate input changed')
            binary=Path(record['simulator'])
            if binary.parent.name!=simulator_cache_key(rtlsim,cfg,ua) or digest(binary)!=record['simulator_sha256']:
                raise ValueError('gate source/parameters/flags binary identity differs')
            log=(source/'rtl.log').read_text()
            if digest(source/'rtl.log')!=record['rtl_log_sha256']:raise ValueError('raw RTL log changed')
            observed=re.search(r'RESULT cycles=(\d+) halted=1 error=0',log)
            if not observed or int(observed[1])!=record['rtl_cycles']:raise ValueError('raw RTL completion differs')
            words=[int(w,16) for w in (source/'prog_0.hex').read_text().split()]
            # The canonical writer removes all-zero HALT and trailing words.
            words += [0]*((-len(words))%8)
            program=[I.Instr.decode(words[i:i+8]) for i in range(0,len(words),8)]+[I.halt()]
            work=resolve_static(cfg,[program])
            if [int(x) for x in re.findall(r'SLICE \d+ icount=(\d+)',log)]!=[len(work.commands[0])]:
                raise ValueError('decoded program instruction count differs')
            golden=Machine(cfg,[program],[read_image(source/'dram_0.bin',cfg.DRAM_BYTES)]).run().slices[0]
            if (source/'dram_out_0.bin').read_bytes()!=golden.dram.tobytes():raise ValueError('full DRAM differs from ISA')
            tmem=np.array([int(w,16) for w in (source/'tmem_0.hex').read_text().split()],dtype='<u4')
            if len(tmem)!=cfg.TMEM_WORDS or not np.array_equal(tmem,golden.tmem):raise ValueError('full scratchpad differs from ISA')
            physical=replace(work,config=replace(cfg,MCOLS=cols))
            def latency(define):
                return int(next(f.split('=')[1] for f in rtlsim.VFLAGS if f.startswith('-D'+define+'=')))
            t=Timing(batch_rows=rows,matrix_row_lanes=rows,fifo_chunks=128*rows,scale_lanes=rows,
                dram_bytes_per_cycle=ua.get('MEM_BYTES',2816),dram_port_bytes=ua.get('MEM_BYTES',2816),
                unified_dram=True,scalar_store_bytes=256,epilogue_lanes=max(1,cols//2),
                quant_row_banks=8,quant_store_buffered=True,vector_row_contexts=4,
                vector_command_queue=2,matrix_retirement=True,physical_scale_lines=True,
                activation_streaming=bool(ua.get('ACT_STREAMING',0)),
                fp_mul_latency=latency('OTPU_FP_MUL_LAT'),fp_add_latency=latency('OTPU_FP_ADD_LAT'))
            if ua.get('SCALE_CACHE_ENTRIES',0) or ua.get('QST_PACKED_WORDS',0):
                raise ValueError('cache and packed-store timing are not modeled here')
            target=a.output/f'c{cols}_{k}'
            prepared=ReusableTiming(a.compiler,a.output/'native-cache',capacity=128).prepare(physical,t,target)
            cycles,_,_,native_log=prepared.run();Path(str(target)+'.log').write_text(native_log)
            results.append(dict(columns=cols,case=record['case']['name'],rtl_cycles=record['rtl_cycles'],
                lpn_cycles=cycles,relative_error=cycles/record['rtl_cycles']-1,all_full_dram_and_tmem_bit_exact=True,
                all_input_log_source_and_binary_hashes_verified=True))
    if hashes!={str(f):digest(f) for f in files}:raise ValueError('model sources changed')
    result=dict(status='complete',scope=__doc__,records=results,source_sha256=hashes,gate_report_sha256=reports,
        timing_target_met_all_cases=all(abs(r['relative_error'])<=.1 for r in results),
        worst_case_relative_error=max(abs(r['relative_error']) for r in results),
        limitations=['Synthetic numerical/timing gates, not full inference','No FPGA resource or clock validation'])
    atomic_json(a.output/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
