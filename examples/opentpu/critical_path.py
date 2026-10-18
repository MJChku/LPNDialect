"""Inspect command dependencies on retained upstream programs, without tensor work.

This is resource-free dependency analysis, not an LPN timing prediction. The
optional renaming analysis removes WAR/WAW edges and assumes unlimited physical
versions. No such renamer exists in the RTL; its storage/ports are not budgeted.
"""
import argparse
from bisect import bisect_left
from collections import Counter
from dataclasses import replace,asdict
import json
from math import ceil,gcd
from pathlib import Path
from types import SimpleNamespace


def renamed_dependencies(commands,I):
    # Most recent writer of each address interval produces its physical version.
    boundaries=[set() for _ in range(3)]
    for c in commands:
        for space,lo,hi in c.footprint.rd+c.footprint.wr:boundaries[space].update((lo,hi))
    bounds=[sorted(b) for b in boundaries]
    writers=[[None for _ in b] for b in bounds]
    def segments(ranges):
        for space,lo,hi in ranges:
            for s in range(bisect_left(bounds[space],lo),bisect_left(bounds[space],hi)):
                yield space,s
    result=[];barrier=None
    for index,c in enumerate(commands):
        if c.footprint.all or c.instruction.op==I.HALT:
            deps=set(range(index));barrier=index
        else:
            deps={writers[space][s] for space,s in segments(c.footprint.rd)}-{None}
            if barrier is not None:deps.add(barrier)
        result.append(sorted(deps))
        for space,s in segments(c.footprint.wr):writers[space][s]=index
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('case','opentpu-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--indices',default='0,15,16,47')
    p.add_argument('--matrix-rows',type=int,default=128)
    p.add_argument('--fp-mul-latency',type=int,choices=(1,2),default=2)
    p.add_argument('--fp-add-latency',type=int,choices=(1,2,3,4),default=4)
    a=p.parse_args()
    from examples.opentpu.workload import use_checkout
    from examples.opentpu.shape_workload import resolve_static
    from examples.opentpu.compact import dependencies
    from examples.opentpu.model import Timing,OpenTPU,bank_write_cycles
    from examples.opentpu.e2e import digest,atomic_json
    use_checkout(a.opentpu_root.resolve())
    from opentpu import Config,isa as I,waits as W
    reference=json.loads((a.case/'report.json').read_text())
    if reference['status']!='complete' or not reference['all_dram_and_tmem_bit_exact']:
        raise ValueError('completed numerical reference required')
    original=Config(**reference['config']);cfg=replace(original,LANES=64,MCOLS=1)
    t=Timing(batch_rows=a.matrix_rows,matrix_row_lanes=a.matrix_rows,dram_bytes_per_cycle=2816,dram_port_bytes=2816,
        fifo_chunks=128*a.matrix_rows,scale_lanes=a.matrix_rows,epilogue_lanes=1,unified_dram=True,scalar_store_bytes=256,
        quant_row_banks=8,quant_store_buffered=True,vector_row_contexts=4,
        fp_mul_latency=a.fp_mul_latency,fp_add_latency=a.fp_add_latency)
    results=[]
    for index in map(int,a.indices.split(',')):
        record=reference['kernels'][index];source=a.case/'kernels'/f'{index:04d}'
        for sid in range(cfg.S):
            if digest(source/f'prog_{sid}.hex')!=record['input_sha256'][f'prog_{sid}.hex']:
                raise ValueError('program changed')
        programs=[[I.Instr.decode(w) for w in json.loads((source/f'program_{sid}.json').read_text())]
                  for sid in range(cfg.S)]
        workload=replace(resolve_static(original,programs,record['args']),config=cfg)
        model=object.__new__(OpenTPU);model.workload=workload;model.timing=t
        memory=SimpleNamespace(b=object(),sw=object())
        for sid,commands in enumerate(workload.commands):
            normal,_=dependencies(commands,I,W)
            for mode,deps in (('existing_address_hazards',normal),('unlimited_renaming_optimism',renamed_dependencies(commands,I))):
                ends=[];parents=[];labels=[];latencies=[];issued=0
                for i,c in enumerate(commands):
                    ins,w=c.instruction,c.instruction.w
                    issued+=2 if ins.op==I.LOOP else 1
                    if ins.op==I.MM:
                        n,kb,m=w[3]&65535,(w[3]>>16)&255,(w[5]>>16)&255
                        pair=bool(ins.flags&I.F_PAIR);wf=(ins.flags>>I.WF_SHIFT)&3
                        chunks=kb if wf==I.W8 else ceil(kb/2)
                        drain=ceil(m/min(t.epilogue_lanes,cfg.MCOLS,cfg.LANES//gcd(w[5]&65535,cfg.LANES)))
                        service=max((chunks if pair else kb)*ceil(m/cfg.MCOLS),drain)
                        tail=min(cfg.D//8,t.chain_length)+cfg.MCOLS+10+2*t.fp_mul_latency+(
                            t.fp_mul_latency+t.fp_add_latency if ins.flags&I.F_ACC else 0)+drain
                        latency=max(ceil(n/t.matrix_row_lanes)*service,
                                    bank_write_cycles(n,m,w[5]&65535,cfg.LANES))+tail
                    elif W.UNIT.get(ins.op,5)==5:latency=0
                    else:latency=model._simple_timing(ins,memory,I)[0]
                    parent=max(deps[i],key=lambda d:ends[d],default=None)
                    ready=ends[parent] if parent is not None else 0
                    earliest=issued+(5 if ins.op>=I.LD else 0)
                    parents.append(parent if ready>=earliest else None)
                    ends.append(max(ready,earliest)+latency)
                    label=I.VFUNCS[(w[5]>>16)&255] if ins.op==I.VOP else I.OPNAMES[ins.op]
                    labels.append(label);latencies.append(latency)
                path=[];cursor=len(commands)-1
                while cursor is not None:
                    path.append(cursor);cursor=parents[cursor]
                demand=Counter()
                for j in path:demand[labels[j]]+=latencies[j]
                results.append(dict(kernel=index,slice=sid,phase=record['phase'],mode=mode,
                    resource_free_command_dependency_cycles=ends[-1],instructions=len(commands),
                    critical_path_commands=len(path),critical_path_unit_latency_sums=dict(demand.most_common()),
                    original_rtl_cycles=record['baseline_rtl_cycles']))
    atomic_json(a.output,dict(scope=__doc__,records=results,
        reference_sha256=digest(a.case/'report.json'),script_sha256=digest(Path(__file__)),
        timing=asdict(t),source_sha256={name:digest(Path(__file__).with_name(name))
            for name in ('model.py','compact.py','workload.py','shape_workload.py')},
        exclusions=['No unit contention or window capacity','MM weights assumed ready',
                    'Renaming mode has unlimited versions and unbudgeted ports/storage'],
        hardware_implemented=False))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
