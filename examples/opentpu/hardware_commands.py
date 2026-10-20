"""Read encoded openTPU hardware programs and count their resolved instructions.

This interface accepts RTL input manifests and prog_N.hex, never tensor DAGs.
Static resolution executes scalar instructions only. Data-dependent control
requires functional execution; unsupported controls fail rather than inventing
register values. No weights, tensor arithmetic or inference scheduling live here.
"""
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from . import REVISION


def load_commands(directory,max_steps=1_000_000):
    """Load the canonical RTL instruction files without loading DRAM contents."""
    from opentpu import Config,isa as I
    from .shape_workload import resolve_static
    directory=Path(directory)
    if not (directory/'input.json').is_file():
        raise ValueError('hardware commands require input.json and prog_N.hex; tensor graphs are not hardware programs')
    metadata=json.loads((directory/'input.json').read_text())
    if metadata.get('revision')!=REVISION:raise ValueError('unsupported hardware ISA revision')
    cfg=Config(**metadata['config'])
    if not 0<cfg.DRAM_BYTES<=1<<32:
        raise ValueError('this hardware ISA has 32-bit addresses; a larger memory needs a hardware extension')
    counts=metadata.get('instruction_counts')
    if counts is not None and (len(counts)!=cfg.S or any(type(n) is not int or n<1 for n in counts)):
        raise ValueError('invalid hardware instruction counts')
    programs=[];hashes={'input.json':hashlib.sha256((directory/'input.json').read_bytes()).hexdigest()}
    for sid in range(cfg.S):
        path=directory/f'prog_{sid}.hex';raw=path.read_bytes()
        words=[int(x,16) for x in raw.split()]
        if not words or any(not 0<=w<1<<32 for w in words):raise ValueError('hardware instruction words must fit in 32 bits')
        length=counts[sid]*8 if counts is not None else ((len(words)+7)//8)*8
        if not length-8<len(words)<=length or length>cfg.IMEM_WORDS:
            raise ValueError('hardware instruction length differs from manifest or exceeds IMEM')
        words.extend([0]*(length-len(words)))
        programs.append([I.Instr.decode(words[i:i+8]) for i in range(0,length,8)])
        hashes[path.name]=hashlib.sha256(raw).hexdigest()
    workload=resolve_static(cfg,programs,metadata.get('args'),max_steps=max_steps)
    return replace(workload,hashes=hashes)


def instruction_statistics(workload):
    """Count issued arithmetic from the resolved hardware execution path.

    Counts include ISA K padding. They are not logical model FLOPs. openTPU's
    MXU performs quantized integer dot products and FP32 epilogues; two dot
    operations per MAC must not be advertised as native BF16 TFLOP/s.
    """
    from opentpu import isa as I
    opcodes=Counter();formats={};macs=0;matrix_commands=0
    vector_flops=0;sfu=Counter();vector_functions=Counter()
    names={v:k for k,v in I.WFORMATS.items()}
    for commands in workload.commands:
        for command in commands:
            ins=command.instruction;opcodes[I.OPNAMES[ins.op]]+=1;w=ins.w
            if ins.op==I.MM:
                n,kb,m=w[3]&65535,(w[3]>>16)&255,(w[5]>>16)&255
                if min(n,kb,m)<1:raise ValueError('invalid hardware matrix dimensions')
                wf=(ins.flags>>I.WF_SHIFT)&3;count=n*kb*m*workload.config.D
                entry=formats.setdefault(names[wf],dict(commands=0,dot_macs=0,dot_operations=0))
                entry['commands']+=1;entry['dot_macs']+=count;entry['dot_operations']+=2*count
                macs+=count;matrix_commands+=1
            elif ins.op==I.VOP:
                rows,cols=w[3]&65535,w[3]>>16;fn=(w[5]>>16)&255;elements=rows*cols
                vector_functions[I.VFUNCS[fn]]+=elements
                if fn in (I.V_ADD,I.V_SUB,I.V_RSUB,I.V_MUL,I.V_EXP2SUB):vector_flops+=elements
                if fn in (I.V_RSUM,I.V_RSSQ,I.V_RDOT):vector_flops+=rows*max(0,cols-1)
                if fn in (I.V_RSSQ,I.V_RDOT):vector_flops+=elements
                if fn in (I.V_EXP2,I.V_RECIP,I.V_RSQRT,I.V_LOG2,I.V_EXP2SUB):
                    sfu[I.VFUNCS[fn]]+=elements
    return dict(dynamic_instructions=sum(opcodes.values()),opcode_counts=dict(opcodes),
        matrix_commands=matrix_commands,matrix_dot_macs=macs,matrix_dot_operations=2*macs,
        matrix_by_weight_format=formats,logical_fp32_vector_flops=vector_flops,
        vector_function_elements=dict(vector_functions),special_function_evaluations=dict(sfu),
        uncounted_vector_function_elements={k:v for k,v in vector_functions.items() if k=='outer'},
        input_interface='Canonical openTPU hardware instructions, eight 32-bit words each',
        matrix_arithmetic='Quantized integer dots with FP32 epilogues, not a native BF16 matrix datapath',
        counting_scope='Two dot operations per MAC; includes encoded K padding and dynamic loop iterations; no model shape estimates',
        exclusions=['Matrix FP32 scale/accumulation epilogues and quantizer arithmetic are not counted as FLOPs',
            'Vector counts are mathematical operations, excluding physical reduction padding',
            'Special functions, comparisons and copies are not converted into FLOPs'])
