"""Generic event-driven operation-DAG screen with explicit shared HBM service.

Consumes external tensor commands; contains no inference scheduling or model
dimensions. This uncalibrated proposal is NOT the FeSens ISA/RTL timing model.
All intermediate reads/writes spill to HBM. Operations stream through bounded
tile buffers, with optimistic overlap of compute and aggregate transfer service.
It omits detailed tiling, bank conflicts, quantization/scale decode, FP feedback
and controller timing, so its cycle results have no RTL accuracy guarantee.
"""
import argparse
from dataclasses import asdict,dataclass
import json
from math import ceil,isfinite
from pathlib import Path
import subprocess
import time

from lpnlang_mlir import InputArray,NetBuilder
from lpnlang_mlir.native_inputs import write_input_tables


@dataclass(frozen=True)
class Hardware:
    clock_mhz: int = 1500
    weight_macs_per_cycle: int = 12288
    bf16_macs_per_cycle: int = 1024
    fp32_macs_per_cycle: int = 256
    vector_elements_per_cycle: int = 256
    hbm_gbs: float = 858.9934592
    pipeline_cycles: int = 16
    window: int = 32
    tile_buffer_bytes_per_unit: int = 65536
    hbm_capacity_bytes: int = 32_000_000_000

    def __post_init__(self):
        if not isfinite(self.hbm_gbs):raise ValueError('finite HBM bandwidth required')
        if min(asdict(self).values())<=0:raise ValueError('positive hardware budgets required')
        for name,value in asdict(self).items():
            if name!='hbm_gbs' and type(value) is not int:raise ValueError('integer hardware counts required')


def checked_commands(document):
    if document.get('version')!=1:raise ValueError('unsupported command interface version')
    commands=document['commands']
    if not commands:raise ValueError('nonempty command DAG required')
    for idx,c in enumerate(commands):
        if c['id']!=idx or c['unit'] not in ('matrix','vector'):raise ValueError('invalid command identity/unit')
        if len(set(c['deps']))!=len(c['deps']) or any(type(d) is not int or not 0<=d<idx for d in c['deps']):
            raise ValueError('dependencies must point to distinct preceding commands')
        for name in ('macs','weight_bytes','vector_elements','activation_read_bytes','activation_write_bytes'):
            if type(c[name]) is not int or c[name]<0:raise ValueError('nonnegative integer command service required')
        if c.get('matrix_shape') is not None:
            shape=c['matrix_shape']
            if (any(type(shape[k]) is not int or shape[k]<1 for k in ('batch','m','n','k')) or
                shape['batch']*shape['m']*shape['n']*shape['k']!=c['macs']):
                raise ValueError('matrix shape and MAC count differ')
    return commands


class TensorDAGScreen:
    def __init__(self,document,hardware):
        commands=checked_commands(document);self.hardware=hardware
        self.net=net=NetBuilder();self.tables={}
        def table(name,values):
            self.tables[name]=list(values);return InputArray(name,self.tables[name])
        kinds=[];compute=[];memory=[];offsets=[0];indices=[]
        self.total_hbm_bytes=0
        for c in commands:
            # NVFP4 weight throughput and BF16 activation matmul throughput
            # are independent explicit budgets, with no inferred FPGA mapping.
            precision=c.get('matrix_precision') or ('nvfp4' if c['weight_bytes'] else 'bf16')
            if c['unit']=='matrix' and precision not in ('nvfp4','bf16','fp32'):
                raise ValueError('unknown matrix arithmetic format')
            kind=0 if c['unit']=='matrix' and precision=='nvfp4' else 1 if c['unit']=='matrix' else 2
            kinds.append(kind)
            throughput=(hardware.weight_macs_per_cycle if kind==0 else
                hardware.fp32_macs_per_cycle if precision=='fp32' else hardware.bf16_macs_per_cycle)
            compute.append(max(1,ceil(c['macs']/throughput) if kind<2 else
                ceil(c['vector_elements']/hardware.vector_elements_per_cycle)))
            traffic=c['weight_bytes']+c['activation_read_bytes']+c['activation_write_bytes']
            self.total_hbm_bytes+=traffic
            memory.append(max(1,ceil(traffic/(hardware.hbm_gbs*1000/hardware.clock_mhz))))
            indices.extend(c['deps']);offsets.append(len(indices))
        kinds=table('kinds',kinds);compute=table('compute_cycles',compute)
        memory=table('memory_cycles',memory);offsets=table('dependency_offsets',offsets)
        indices=table('dependency_indices',indices or [0]);n=len(commands)
        # Completion state is compressed into immutable 64-bit flag latches.
        banks=[net.place(f'flags_{i}',initial_tokens=1) for i in range((n+63)//64)]
        slots=[net.place(f'slot_{i}',initial_tokens=1) for i in range(hardware.window)]
        pending=[net.place(f'pending_{i}') for i in range(3)]
        units=[net.place(f'unit_{i}',initial_tokens=1) for i in range(3)]
        bus=net.place('hbm_link',initial_tokens=1)
        # One tile buffer per active unit. Each credit is retained for the
        # whole operation; buffers are never provisioned per model/token.
        buffers=[net.place(f'tile_buffer_{i}',initial_tokens=1) for i in range(3)]
        start=net.place('start',initial_tokens=1)
        cursor=net.place('dispatch_cursor')
        window=net.place('window',initial_tokens=hardware.window)
        signals=net.place('signals');done_count=net.place('done_count',initial_tokens=1)
        finish=net.place('finish')
        completed=net.place('program_completed',observable=True)

        @net.transition('begin')
        def begin():
            take(start);emit(cursor,create())

        @net.transition('complete')
        def complete():
            take(finish);emit(completed,create())

        @net.func(name='is_done',args=('i64',),results=('i1',))
        def is_done(idx):
            token=peek(array(banks)[divi(idx,64)])
            func_return(andi(token.get('bits'),shli(1,remi(idx,64)))!=0)

        @net.func(name='select_ready',args=('i64',),results=('i64',))
        def select_ready(unit):
            chosen=const_i64(-1);oldest=const_i64(n)
            for s in range(hardware.window):
                state=peek(array(slots)[s]);idx=state.get('id')
                ready=select(state.get('valid')>0,array(kinds)[idx]==unit,False,typ='i1')
                for j in range(select(ready,array(offsets)[idx],0),select(ready,array(offsets)[idx+1],0)):
                    ready=select(ready,is_done(array(indices)[j]),False,typ='i1')
                eligible=select(ready,idx<oldest,False,typ='i1')
                chosen=select(eligible,s,chosen,typ='i64')
                oldest=select(eligible,idx,oldest)
            func_return(chosen)

        @net.transition('dispatch')
        def dispatch():
            guard(count(cursor)>0);guard(count(window)>0)
            state=peek(cursor);idx=state.get('id');guard(idx<n)
            slot=array(slots)[remi(idx,hardware.window)]
            old=peek(slot);guard(old.get('valid')==0)
            take(cursor);take(window);take(slot)
            emit(slot,old.set('valid',1).set('id',idx))
            emit(array(pending)[array(kinds)[idx]],create())
            emit(cursor,state.set('id',idx+1),delay=1.0)

        for u in range(3):
            resource=units[u];queue=pending[u];buffer=buffers[u]
            @net.transition(f'execute_{u}')
            def execute(u=u,resource=resource,queue=queue,buffer=buffer):
                guard(count(resource)>0);guard(count(queue)>0);guard(count(bus)>0)
                guard(count(buffer)>0)
                chosen=select_ready(const_i64(u));guard(chosen>=0)
                place=array(slots)[chosen];old=take(place);idx=old.get('id')
                take(queue);emit(place,old.set('valid',0))
                transfer_cycles=array(memory)[idx];compute_cycles=array(compute)[idx]
                transfer=sitofp(transfer_cycles)
                latency=sitofp(select(transfer_cycles>compute_cycles,transfer_cycles,compute_cycles)+hardware.pipeline_cycles)
                port=take(bus);emit(bus,port,delay=transfer)
                credit=take(resource);emit(resource,credit,delay=latency)
                tile=take(buffer);emit(buffer,tile,delay=latency)
                emit(signals,create().set('id',idx),delay=latency)

        @net.transition('publish')
        def publish():
            guard(count(signals)>0)
            signal=take(signals);idx=signal.get('id');bank=array(banks)[divi(idx,64)]
            state=take(bank);emit(bank,state.set('bits',ori(state.get('bits'),shli(1,remi(idx,64)))))
            current=take(done_count);value=current.get('value')+1
            emit(done_count,current.set('value',value));emit(window,create())
            if value==n:emit(finish,create())

    def statistics(self):
        return dict(transitions=len(self.net._transitions),places=len(self.net._places),
            tile_buffer_bytes=3*self.hardware.tile_buffer_bytes_per_unit,
            hbm_transfer_bytes=self.total_hbm_bytes)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commands',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--clock-mhz',type=int,default=1500);p.add_argument('--hbm-gbs',type=float,default=858.9934592)
    p.add_argument('--weight-macs-per-cycle',type=int,default=12288)
    p.add_argument('--bf16-macs-per-cycle',type=int,default=1024)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    from examples.opentpu.e2e import atomic_json,digest
    from examples.opentpu.benchmark import native_observation
    document=json.loads(a.commands.read_text())
    hw=Hardware(clock_mhz=a.clock_mhz,hbm_gbs=a.hbm_gbs,
        weight_macs_per_cycle=a.weight_macs_per_cycle,bf16_macs_per_cycle=a.bf16_macs_per_cycle)
    start=time.perf_counter();model=TensorDAGScreen(document,hw)
    ir=a.output/'model.mlir';ir.write_text(model.net.build());inputs=a.output/'timing.tables'
    write_input_tables(inputs,model.tables);binary=a.output/'timing.native'
    with (a.output/'compile.log').open('w') as log:
        subprocess.run([str(a.compiler.resolve()),str(ir),'-o',str(binary)],stdout=log,stderr=subprocess.STDOUT,check=True)
    cycles,core,process,log=native_observation(binary,inputs=inputs)
    (a.output/'lpn.log').write_text(log)
    atomic_json(a.output/'report.json',dict(status='complete',scope=__doc__,hardware=asdict(hw),
        commands_sha256=digest(a.commands),source_sha256=digest(Path(__file__)),binary_sha256=digest(binary),
        input_tables_sha256=digest(inputs),command_count=len(document['commands']),statistics=model.statistics(),
        cycles=cycles,assumed_hardware_seconds=cycles/(hw.clock_mhz*1e6),native_seconds=core,
        host_total_seconds=time.perf_counter()-start,zero_rollbacks=True,
        rtl_calibrated=False,numerical_validated=False,fpga_mapping_proven=False,
        existing_opentpu_isa=False,implementation='Hypothetical streaming tensor-operation accelerator',
        limitations=['All intermediate activation reads/writes spill to HBM; no SRAM reuse invented',
            'Compute and aggregate HBM service overlap optimistically per operation',
            'Tile-buffer capacity is fixed, but detailed tile timing/refills and port conflicts are unvalidated',
            'NVFP4 scale/dequantization, activation quantization and FP recurrence are not calibrated',
            'No full allocation/liveness capacity proof for out-of-order activation storage',
            'No speedup/accuracy comparison with original openTPU or commercial TPU/GPU is established']))
    print(json.dumps(dict(cycles=cycles,native_seconds=core,statistics=model.statistics())),flush=True)


if __name__=='__main__':main()
