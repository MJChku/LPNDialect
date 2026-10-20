"""Fresh full-file, ISA and native-timing audit of retained NVFP4 matrix gates."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('case','opentpu-root','compiler','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from examples.opentpu.e2e import digest,atomic_json
    from examples.opentpu.workload import use_checkout
    from examples.opentpu.shape_workload import resolve_static
    from examples.opentpu.weight_storage import read_image
    from examples.opentpu.model import Timing
    from examples.opentpu.native_timing import ReusableTiming
    from .parallel_inference import simulator_cache_key
    root=use_checkout(a.opentpu_root)
    import numpy as np
    from opentpu import Config,isa as I,rtlsim
    from opentpu.isasim import Machine
    report=json.loads((a.case/'report.json').read_text())
    if report['status']!='complete' or len(report['records'])!=3:raise ValueError('three completed NVFP4 gates required')
    for path,expected in report['source_sha256'].items():
        if digest(Path(path))!=expected:raise ValueError('gate/RTL source changed')
    cfg=Config(**report['config']);ua=report['uarch'];rtlsim.VFLAGS[:]=report['rtl_compile_flags']
    binary=root/'build/verilator'/simulator_cache_key(rtlsim,cfg,ua)/'Vtb_top';binary_hash=digest(binary)
    timing=Timing(batch_rows=2,matrix_row_lanes=2,fifo_chunks=256,scale_lanes=2,chain_length=8,
        dram_bytes_per_cycle=2816,dram_port_bytes=2816,unified_dram=True,scalar_store_bytes=256,
        quant_row_banks=8,quant_store_buffered=True,vector_row_contexts=4,
        vector_command_queue=2,matrix_retirement=True,physical_scale_lines=True)
    project=Path(__file__).resolve().parents[1]
    files=[Path(__file__),root/'opentpu/isasim.py',root/'opentpu/quant.py',
        *[project/n for n in ('model.py','compact.py','components.py','shape_workload.py','native_timing.py')]]
    sources={str(f):digest(f) for f in files};results=[]
    for record in report['records']:
        if Path(record['simulator'])!=binary or record['simulator_sha256']!=binary_hash:raise ValueError('binary differs')
        directory=a.case/record['name']
        for field in ('input_sha256','output_sha256'):
            for name,expected in record[field].items():
                if digest(directory/name)!=expected:raise ValueError('file changed: '+name)
        log=(directory/'rtl.log').read_text()
        if digest(directory/'rtl.log')!=record['raw_log_sha256']:raise ValueError('raw log changed')
        match=re.search(r'RESULT cycles=(\d+) halted=1 error=0',log)
        if not match or int(match[1])!=record['rtl_cycles']:raise ValueError('completion differs')
        words=[int(w,16) for w in (directory/'prog_0.hex').read_text().split()]
        words+=[0]*((-len(words))%8)
        program=[I.Instr.decode(words[i:i+8]) for i in range(0,len(words),8)]+[I.halt()]
        work=resolve_static(cfg,[program])
        if [int(x) for x in re.findall(r'SLICE \d+ icount=(\d+)',log)]!=[len(work.commands[0])]:
            raise ValueError('instruction count differs')
        mm=[c.instruction for c in work.commands[0] if c.instruction.op==I.MM]
        if len(mm)!=1 or (mm[0].flags>>I.WF_SHIFT)&3!=I.WNV:raise ValueError('NVFP4 was not actually selected')
        golden=Machine(cfg,[program],[read_image(directory/'dram_0.bin',cfg.DRAM_BYTES)]).run().slices[0]
        if (directory/'dram_out_0.bin').read_bytes()!=golden.dram.tobytes():raise ValueError('full DRAM differs')
        memory=np.array([int(x,16) for x in (directory/'tmem_0.hex').read_text().split()],dtype='<u4')
        if len(memory)!=cfg.TMEM_WORDS or not np.array_equal(memory,golden.tmem):raise ValueError('full TMEM differs')
        work=replace(work,config=replace(cfg,MCOLS=2))
        native=ReusableTiming(a.compiler,a.output/'native-cache',capacity=128).prepare(work,timing,a.output/(record['name']+'.tables'))
        cycles,_,_,text=native.run();(a.output/(record['name']+'.log')).write_text(text)
        if cycles!=record['lpn_cycles']:raise ValueError('fresh native prediction differs')
        results.append(dict(record,fresh_isa_full_memories_exact=True,fresh_native_cycles=cycles))
    if sources!={str(f):digest(f) for f in files}:raise ValueError('audit/model sources changed')
    atomic_json(a.output/'report.json',dict(status='complete',scope=__doc__,source_sha256=sources,
        gate_report_sha256=digest(a.case/'report.json'),binary_sha256=binary_hash,records=results,
        all_full_memories_exact=True,actual_nvfp4_selected=True,
        worst_relative_cycle_error=max(abs(r['relative_cycle_error']) for r in results),
        timing_target_met=all(abs(r['relative_cycle_error'])<=.1 for r in results),
        full_glm_inference=False,physical_mapping_or_frequency_proven=False))
    print('NVFP4 fresh ISA, native timing and complete-output audit passed',flush=True)


if __name__=='__main__':main()
