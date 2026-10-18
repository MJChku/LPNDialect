"""Check physical activation/FIFO capacity on all unchanged inference programs."""
import argparse
import json
from pathlib import Path


def command_bounds(commands,I):
    rows=blocks=fifo=matrices=quantizers=0
    for c in commands:
        ins=c.instruction;w=ins.w
        if ins.op==I.QACT:
            nr=w[1]&255;ab=(w[1]>>8)&255;kb=w[1]>>16
            rows=max(rows,nr*(2 if ins.flags&8 else 1));blocks=max(blocks,ab+kb);quantizers+=1
        elif ins.op==I.MM:
            if ((ins.flags>>I.WF_SHIFT)&3)!=I.W8 or ins.flags&I.F_PAIR:
                raise ValueError('storage envelope currently requires non-PAIR int8 matrices')
            kb=(w[3]>>16)&255;m=(w[5]>>16)&255;ab=w[5]>>24
            rows=max(rows,m);blocks=max(blocks,ab+kb);fifo=max(fifo,kb);matrices+=1
    return dict(activation_rows=rows,activation_blocks=blocks,row_fifo_chunks=fifo,
                matrix_commands=matrices,quantizer_commands=quantizers)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('case','opentpu-root','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    from .workload import use_checkout
    from .e2e import atomic_json,digest
    from .shape_workload import resolve_static
    from .retained_inputs import verified_program
    use_checkout(a.opentpu_root.resolve())
    from opentpu import Config,isa as I
    reference=json.loads((a.case/'report.json').read_text())
    if (reference['status']!='complete' or reference['entrypoint']!='unmodified upstream Engine.generate'
        or not reference['all_dram_and_tmem_bit_exact'] or not reference['all_inputs_identical_to_baseline']):
        raise ValueError('requires complete identical-input upstream inference')
    cfg=Config(**reference['config']);records=[]
    files=[Path(__file__),Path(__file__).with_name('shape_workload.py'),Path(__file__).with_name('retained_inputs.py')]
    hashes={str(f):digest(f) for f in files}
    for record in reference['kernels']:
        index=record['step'];source=a.case/'kernels'/f'{index:04d}';programs=[];identities={}
        for sid in range(cfg.S):
            if digest(source/f'prog_{sid}.hex')!=record['input_sha256'][f'prog_{sid}.hex']:
                raise ValueError('canonical program differs')
            enc,identities[str(sid)]=verified_program(source,sid)
            programs.append([I.Instr.decode(x) for x in enc])
        work=resolve_static(cfg,programs,record['args'])
        if [len(c) for c in work.commands]!=record['instructions']:raise ValueError('instruction counts differ')
        bounds=[command_bounds(c,I) for c in work.commands]
        records.append(dict(kernel=index,program_json_sha256=identities,bounds=bounds))
    if [r['kernel'] for r in records]!=list(range(48)):raise ValueError('requires all 48 kernels')
    maxima={k:max(b[k] for r in records for b in r['bounds']) for k in
            ('activation_rows','activation_blocks','row_fifo_chunks')}
    if hashes!={str(f):digest(f) for f in files}:raise RuntimeError('analysis sources changed')
    result=dict(status='complete',scope=__doc__,records=records,maximum_requirements=maxima,
        reference_report_sha256=digest(a.case/'report.json'),source_sha256=hashes,
        original_activation_rows=cfg.ACT_ROWS,original_activation_blocks=cfg.ACT_BLOCKS,
        numerical_arithmetic_executed=False,
        limitations=['Workload-specific capacity proof, not a universal smaller ISA configuration',
                     'Row FIFO bound proves one row fits, not unchanged performance at smaller depth',
                     'Changed physical capacities still require RTL numerical and cycle validation'])
    atomic_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
