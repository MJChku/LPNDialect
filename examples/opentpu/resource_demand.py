"""Account resource service in unchanged retained inference without neural arithmetic.

Bounds apply to the specified existing units and port schedule. They are not
universal limits on redesigned hardware; operation waits may overlap.
"""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict,replace
import json
from math import ceil,gcd
from pathlib import Path
from types import SimpleNamespace


def demand(work,timing):
    from opentpu import isa as I
    from .model import OpenTPU,matrix_batch_rows,matrix_scale_requests,bank_write_cycles,vector_rank,VECTOR_DRAIN_RANK
    model=object.__new__(OpenTPU);model.workload=work;model.timing=timing
    memory=SimpleNamespace(b='dram',sw='dram')
    service=Counter();traffic=Counter();shapes=defaultdict(Counter);vectors=defaultdict(Counter)
    for commands in work.commands:
        for command in commands:
            ins=command.instruction;w=ins.w
            service['dispatch']+=2 if ins.op==I.LOOP else 1
            if ins.op==I.MM:
                n,kb,m=w[3]&65535,(w[3]>>16)&255,(w[5]>>16)&255
                chunks=kb if (ins.flags>>I.WF_SHIFT)&3==I.W8 else ceil(kb/2)
                batch=matrix_batch_rows(n,chunks,timing)
                scales=[32*matrix_scale_requests(ins,I,row,timing.matrix_row_lanes) for row in range(n)]
                weights=n*chunks*work.config.D;metadata=sum(scales)
                traffic['matrix_weight_bytes']+=weights;traffic['matrix_scale_bytes']+=metadata
                compute=0;writes=0
                for row in range(0,n,batch):
                    width=min(batch,n-row)
                    core=ceil(width/timing.matrix_row_lanes)*kb*ceil(m/work.config.MCOLS)
                    output=bank_write_cycles(width,m,w[5]&65535,work.config.LANES)
                    compute+=max(core,output);writes+=output
                    transferred=width*chunks*work.config.D+sum(scales[row:row+width])
                    service['dram']+=ceil(transferred/timing.dram_bytes_per_cycle)
                    service['scale_decoder']+=ceil(sum(scales[row:row+width])/32/timing.scale_lanes)
                service['matrix_compute_without_activation_waits']+=compute
                service['matrix_write_port']+=writes
                drain=ceil(m/min(timing.epilogue_lanes,work.config.MCOLS,
                    work.config.LANES//gcd(w[5]&65535,work.config.LANES)))
                service['matrix_retirement']+=5+2*timing.fp_mul_latency+drain+2+(
                    3+timing.fp_mul_latency+timing.fp_add_latency if ins.flags&I.F_ACC else 0)
                shape=(n,kb,m,bool(ins.flags&I.F_UNIT));entry=shapes[shape]
                entry['commands']+=1;entry['compute_floor_cycles']+=compute
                entry['weight_bytes']+=weights;entry['scale_bytes']+=metadata
                entry['integer_macs']+=n*kb*m*work.config.D
            elif ins.op in (I.LD,I.ST,I.QACT,I.QST,I.VOP):
                latency,port,busy=model._simple_timing(ins,memory,I)
                if port=='dram':service['dram']+=busy
                if ins.op in (I.LD,I.ST):
                    service['dma']+=latency;traffic['dma_payload_bytes']+=4*w[2]
                elif ins.op in (I.QACT,I.QST):
                    service['quantizer']+=latency
                    if ins.op==I.QST:
                        rows,kb=w[3]&65535,w[3]>>16
                        traffic['quant_data_bytes']+=rows*kb*work.config.D//(2 if ins.flags&I.F_HALF else 1)
                        traffic['quant_scale_bytes']+=4*rows*(1 if ins.flags&I.F_ROW else 1+ceil(kb/4))
                else:
                    busy=model._vector_interval(ins,I) if vector_rank(ins,I,timing)<VECTOR_DRAIN_RANK else latency
                    service['vector_issue']+=busy
                    entry=vectors[I.VFUNCS[(w[5]>>16)&255]];entry['commands']+=1
                    entry['issue_busy_cycles']+=busy;entry['sum_command_latencies']+=latency
    # Matrix input waits are deliberately omitted, unlike Timeline's compute
    # credit holds. Include only one independent bound per physical resource.
    keys=['dispatch','dram','scale_decoder','matrix_compute_without_activation_waits',
          'matrix_retirement','quantizer','vector_issue']
    return dict(service_cycles=dict(service),traffic_bytes=dict(traffic),
        existing_resource_floor_cycles=max(service[k] for k in keys),
        matrix_shapes=[dict(n=k[0],kb=k[1],m=k[2],unit=k[3],**v) for k,v in
                       sorted(shapes.items(),key=lambda x:-x[1]['compute_floor_cycles'])],
        vector_functions=dict(vectors))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('case','opentpu-root','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--indices',default='all')
    p.add_argument('--timeline-root',type=Path,help='Verify accounting against existing kernel 0/16 observer traces')
    a=p.parse_args()
    from .workload import use_checkout
    from .retained_inputs import verified_program
    from .shape_workload import resolve_static
    from .model import Timing
    from .e2e import digest,atomic_json
    use_checkout(a.opentpu_root)
    from opentpu import Config,isa as I
    reference=json.loads((a.case/'report.json').read_text())
    if reference['status']!='complete' or not reference['all_dram_and_tmem_bit_exact'] or not reference['all_inputs_identical_to_baseline']:
        raise ValueError('completed identical-input numerical validation required')
    indices=list(range(48)) if a.indices=='all' else sorted(set(map(int,a.indices.split(','))))
    if not indices or not set(indices)<=set(range(48)):raise ValueError('invalid kernels')
    cfg=replace(Config(**reference['config']),MCOLS=1,LANES=64)
    timing=Timing(batch_rows=128,matrix_row_lanes=128,fifo_chunks=8192,scale_lanes=128,
        dram_bytes_per_cycle=2816,dram_port_bytes=2816,unified_dram=True,scalar_store_bytes=256,
        quant_row_banks=8,quant_store_buffered=True,quant_scale_lookup=True,scalar_recip_lookup=True,
        vector_row_contexts=4,vector_command_queue=2,matrix_retirement=True,physical_scale_lines=True,
        activation_streaming=True,fp_add_latency=2)
    files=[Path(__file__).with_name(n) for n in ['resource_demand.py','model.py','shape_workload.py','retained_inputs.py']]
    hashes={str(f):digest(f) for f in files};records=[];checks=[]
    for index in indices:
        source=a.case/'kernels'/f'{index:04d}';record=reference['kernels'][index]
        if digest(source/'prog_0.hex')!=record['input_sha256']['prog_0.hex']:raise ValueError('canonical program changed')
        encoded,identity=verified_program(source,0)
        work=resolve_static(cfg,[[I.Instr.decode(w) for w in encoded]],record['args'])
        if [len(c) for c in work.commands]!=record['instructions']:raise ValueError('instruction count differs')
        result=demand(work,timing);result.update(kernel=index,phase=record['phase'],program_json_sha256=identity)
        if a.timeline_root and index in (0,16):
            observed_path=a.timeline_root/f'physical-timeline-{index}'/'report.json'
            observed=json.loads(observed_path.read_text())['resources']
            for key,name in [('dram','dram_port_b'),('scale_decoder','dram_port_a'),('quantizer','unit_2_resource'),
                             ('vector_issue','unit_3_resource'),('matrix_retirement','mxu_0_retirement_resource')]:
                if result['service_cycles'][key]!=observed['s0_'+name]['busy_cycles']:
                    raise ValueError('accounting differs from independent observer: '+key)
            checks.append(dict(kernel=index,timeline_sha256=digest(observed_path),all_five_resource_services_equal=True))
        records.append(result);print(json.dumps(dict(kernel=index,service=result['service_cycles'])),flush=True)
    total=Counter();traffic=Counter()
    for r in records:total.update(r['service_cycles']);traffic.update(r['traffic_bytes'])
    floor=sum(r['existing_resource_floor_cycles'] for r in records)
    original=sum(reference['kernels'][i]['baseline_rtl_cycles'] for i in indices)
    if hashes!={str(f):digest(f) for f in files}:raise ValueError('accounting sources changed')
    atomic_json(a.output,dict(status='complete',scope=__doc__,indices=indices,timing=asdict(timing),
        reference_report_sha256=digest(a.case/'report.json'),source_sha256=hashes,records=records,
        summed_resource_service_cycles=dict(total),summed_traffic_bytes=dict(traffic),
        existing_resource_floor_cycles=floor,existing_resource_speedup_ceiling=original/floor,
        raw_weight_bandwidth_floor_cycles=traffic['matrix_weight_bytes']/timing.dram_bytes_per_cycle,
        raw_weight_only_speedup_ceiling=original/(traffic['matrix_weight_bytes']/timing.dram_bytes_per_cycle),
        observer_checks=checks,numerical_arithmetic_executed=False,
        limitations=['Bounds for current unit throughput, not universal hardware limits',
          'Resource demands overlap; do not sum as elapsed time',
          'Excludes activation waits from matrix service',
          'No cache hits assumed; raw weight bytes alone omit scale/DMA/quantizer traffic']))


if __name__=='__main__':main()
