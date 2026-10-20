"""Fresh ISA, native-LPN and complete-memory audit of actual GLM programs."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import re

from .e2e import atomic_json,digest
from .workload import load,use_checkout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('opentpu-root','compiler','base-rtl-gate','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cases',type=Path,nargs='+',required=True);a=p.parse_args()
    root=use_checkout(a.opentpu_root);a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from opentpu import rtlsim
    from .model import Timing
    from .native_timing import ReusableTiming
    from .hardware_commands import instruction_statistics
    from .rtl_evolution.parallel_inference import simulator_cache_key
    base=json.loads((a.base_rtl_gate/'report.json').read_text())
    for f,h in base['source_sha256'].items():
        if digest(Path(f))!=h:raise ValueError('original packed RTL source changed')
    rtlsim.VFLAGS[:]=base['rtl_compile_flags'];uarch=base['uarch']
    timing=Timing(batch_rows=2,matrix_row_lanes=2,fifo_chunks=256,scale_lanes=2,chain_length=8,
        dram_bytes_per_cycle=2816,dram_port_bytes=2816,unified_dram=True,scalar_store_bytes=256,
        quant_row_banks=8,quant_store_buffered=True,vector_row_contexts=4,vector_command_queue=2,
        matrix_retirement=True,physical_scale_lines=True)
    project=Path(__file__).parent
    files=[Path(__file__),*[project/n for n in ('model.py','compact.py','components.py','workload.py',
        'hardware_commands.py','shape_workload.py','native_timing.py','benchmark.py','activation_stream.py')],
        *[root/'opentpu'/n for n in ('compiler.py','language.py','runtime.py','isasim.py','fp32.py','quant.py','isa.py','waits.py')]]
    sources={str(f):digest(f) for f in files};records=[]
    for case in a.cases:
        report=json.loads((case/'report.json').read_text())
        if report['status']!='complete':raise ValueError('completed program case required')
        for f,h in report['source_sha256'].items():
            if digest(Path(f))!=h:raise ValueError('compiler-case source changed')
        integrated=bool(report.get('all_stages_numerically_executed'))
        stages=report['records'] if integrated else [report]
        previous=None;subrecords=[]
        for expected in stages:
            directory=case/expected['name'] if integrated else case
            work=load(directory);gold=work.machine.slices[0]
            if previous is not None and previous!=work.images[0].tobytes():raise ValueError('stage DRAM continuity differs')
            previous=gold.dram.tobytes()
            if instruction_statistics(work)!=expected['hardware_instructions']:raise ValueError('actual instruction accounting differs')
            if 'input_sha256' in expected:
                for name,h in expected['input_sha256'].items():
                    if digest(directory/name)!=h:raise ValueError('canonical input changed')
            observed=None;binary=None
            if report.get('rtl_validated'):
                rtl=directory/'rtl';text=(directory/'rtl.log').read_text()
                match=re.search(r'RESULT cycles=(\d+) halted=1 error=0',text)
                if not match or int(match[1])!=expected['rtl_cycles']:raise ValueError('raw RTL completion differs')
                if [int(n) for n in re.findall(r'SLICE \d+ icount=(\d+)',text)]!=[len(work.commands[0])]:
                    raise ValueError('actual RTL instruction count differs')
                for name in ('prog_0.hex','dram_0.bin'):
                    if digest(rtl/name)!=digest(directory/name):raise ValueError('RTL did not consume the same input')
                if (rtl/'dram_out_0.bin').read_bytes()!=previous:raise ValueError('complete RTL DRAM differs from fresh ISA')
                words=np.asarray([int(w,16) for w in (rtl/'tmem_0.hex').read_text().split()],dtype=np.uint32)
                if not np.array_equal(words,gold.tmem):raise ValueError('complete RTL scratchpad differs from fresh ISA')
                binary=root/'build/verilator'/simulator_cache_key(rtlsim,work.config,uarch)/'Vtb_top'
                if not binary.is_file():raise ValueError('matching RTL simulator binary missing')
                observed=expected['rtl_cycles']
            target=a.output/(case.name+'-'+expected.get('name','component'));target.mkdir()
            native=ReusableTiming(a.compiler,a.output/'native-cache',capacity=32768).prepare(
                replace(work,config=replace(work.config,MCOLS=2)),timing,target/'timing.tables')
            cycles,_,_,text=native.run();(target/'lpn.log').write_text(text)
            if cycles!=expected['lpn_cycles']:raise ValueError('fresh actual LPN prediction differs')
            subrecords.append(dict(name=expected.get('name','component'),fresh_lpn_cycles=cycles,
                actual_rtl_cycles=observed,relative_cycle_error=cycles/observed-1 if observed else None,
                actual_hardware_program_sha256=digest(directory/'prog_0.hex'),
                numerical_full_dram_sha256=digest(directory/'rtl/dram_out_0.bin') if observed else None,
                matching_simulator_sha256=digest(binary) if binary else None,
                all_full_memories_bit_exact=observed is not None,
                native_binary_sha256=digest(native.binary),input_tables_sha256=digest(native.inputs)))
        records.append(dict(case=str(case),report_sha256=digest(case/'report.json'),records=subrecords,
            shared_dram_stage_continuity_verified=integrated,routing_annotations_used=False,full_glm_model=False))
    if sources!={str(f):digest(f) for f in files}:raise ValueError('audit inputs changed')
    atomic_json(a.output/'report.json',dict(status='complete',scope=__doc__,source_sha256=sources,
        packed_rtl_gate_sha256=digest(a.base_rtl_gate/'report.json'),timing=asdict(timing),records=records,
        full_glm_model=False,all_numerical_cases_use_actual_hardware_programs=True))
    print('fresh GLM ISA/native/full-memory audit passed',flush=True)


if __name__=='__main__':main()
