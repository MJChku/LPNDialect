"""Observe upstream GLM generate on meta tensors with explicit routing annotations.

This is shape/control planning, not numerical inference. Tensor math executes
the pinned Transformers implementation on meta tensors. Expert IDs must come
from an external trace or an explicitly labeled sensitivity scenario. Logits
are placeholders so unmodified generate owns batched prefill and decode.
The exported tensor-operation DAG is not the existing FeSens ISA.
"""
import argparse
from collections import Counter
import hashlib
import json
from math import prod
from pathlib import Path
import time
import types

from .glm47 import pinned_config, inventory, nvfp4_bytes

TRANSFORMERS_VERSION='5.12.0'


def checked_routes(rows, experts, topk, supplied=None):
    if len(rows)!=supplied:raise ValueError('routing annotation token count differs from upstream call')
    for row in rows:
        if (len(row)!=topk or len(set(row))!=topk or
            any(type(e) is not int or not 0<=e<experts for e in row)):
            raise ValueError('routing requires distinct in-range expert IDs per token')
    return rows


def routing_scenario(tokens, experts, topk, scenario):
    if scenario not in ('balanced','hotspot'):raise ValueError('unknown routing scenario')
    return [[((t*topk if scenario=='balanced' else 0)+k)%experts for k in range(topk)] for t in range(tokens)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prompt-tokens',type=int,default=1024)
    p.add_argument('--output-tokens',type=int,default=8)
    routes=p.add_mutually_exclusive_group(required=True)
    routes.add_argument('--routing-json',type=Path,help='Actual observed IDs: calls -> layer -> token -> top-k IDs')
    routes.add_argument('--routing-scenario',choices=('balanced','hotspot'),help='Explicit sensitivity assumption, never actual model routing')
    a=p.parse_args()
    if min(a.prompt_tokens,a.output_tokens)<1:p.error('positive lengths required')
    import torch
    import transformers
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    from transformers import Glm4MoeLiteConfig,Glm4MoeLiteForCausalLM
    if transformers.__version__!=TRANSFORMERS_VERSION:raise ValueError('requires pinned transformers '+TRANSFORMERS_VERSION)
    config,identity=pinned_config()
    if a.prompt_tokens+a.output_tokens>config['max_position_embeddings']:
        p.error('requested sequence exceeds model context')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    from .e2e import atomic_json,digest
    supplied=json.loads(a.routing_json.read_text()) if a.routing_json else None
    cfg=Glm4MoeLiteConfig(**config);cfg._attn_implementation='eager';cfg._experts_implementation='eager'
    start=time.perf_counter()
    with torch.device('meta'):model=Glm4MoeLiteForCausalLM(cfg)
    model.to(dtype=torch.bfloat16);model.eval()
    parameter_count=sum(t.numel() for t in model.parameters())
    if parameter_count!=sum(t.elements for t in inventory(config) if t.kind!='router_buffer'):
        raise ValueError('upstream main-model inventory differs')
    parameter_storage={t.untyped_storage()._cdata:n for n,t in model.named_parameters()}
    counter=Counter();commands=[];tensor_info={};retained=[];calls=[];current_call=-1
    observed_routes=[];layer_visits=Counter();call_dependency=-1;route_dependencies={}
    active_route_dependencies=set()
    aliases={'aten.view.default','aten._unsafe_view.default','aten.reshape.default','aten.t.default',
        'aten.transpose.int','aten.permute.default','aten.expand.default','aten.unsqueeze.default',
        'aten.squeeze.dim','aten.squeeze.default','aten.slice.Tensor','aten.select.int',
        'aten.detach.default','aten.alias.default','aten.split.Tensor','aten.split_with_sizes.default',
        'aten.unbind.int'}
    def tensors(value):return [v for v in tree_flatten(value)[0] if isinstance(v,torch.Tensor)]
    def info(t):
        return tensor_info.get(id(t),(set(),parameter_storage.get(t.untyped_storage()._cdata)))
    def remember(t,deps,parameter=None):
        if id(t) not in tensor_info:retained.append(t)
        tensor_info[id(t)]=(set(deps),parameter)
    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs=kwargs or {};name=str(func)
            # Actual IDs stay on CPU so upstream where/nonzero determines the
            # real expert batch shapes for the annotation. Only the final meta
            # scatter's index is mirrored onto meta; its values are not read.
            if name.startswith('aten.index_add') and args[0].device.type=='meta':
                args=tuple(t.to('meta') if isinstance(t,torch.Tensor) and t.device.type!='meta' else t for t in args)
            inputs=tensors((args,kwargs));deps=set().union(*(info(t)[0] for t in inputs)) if inputs else set()
            deps.update(active_route_dependencies)
            if call_dependency>=0:deps.add(call_dependency)
            parameters={info(t)[1] for t in inputs if info(t)[1] is not None}
            result=func(*args,**kwargs);outputs=[t for t in tensors(result) if t.device.type=='meta']
            if not outputs or current_call<0:return result
            counter[name]+=1
            if name in aliases or name in ('aten._to_copy.default','aten.empty.memory_format',
                    'aten.empty_like.default','aten.new_empty.default','aten.empty_strided.default'):
                for out in outputs:remember(out,deps,next(iter(parameters)) if len(parameters)==1 else None)
                return result
            out_elements=sum(t.numel() for t in outputs)
            if not out_elements:
                for out in outputs:remember(out,deps)
                return result
            unit='vector';macs=weight_bytes=0;matrix_shape=None
            if name in ('aten.mm.default','aten.bmm.default'):
                x,w=args[:2];m,k=x.shape[-2:];n=w.shape[-1]
                batches=prod(x.shape[:-2]);macs=batches*m*n*k;unit='matrix'
                matrix_shape=dict(batch=batches,m=m,n=n,k=k)
                weight_parameter=info(w)[1]
                if weight_parameter:
                    if '.gate.weight' in weight_parameter:weight_bytes=w.numel()*4
                    else:weight_bytes=sum(nvfp4_bytes((batches*n,k)).values())
            elif name=='aten.embedding.default':
                weight_bytes=sum(nvfp4_bytes((prod(outputs[0].shape[:-1]),outputs[0].shape[-1])).values())
            index=len(commands)
            activation_read=sum(t.numel()*t.element_size() for t in inputs if t.device.type=='meta' and info(t)[1] is None)
            activation_write=sum(t.numel()*t.element_size() for t in outputs)
            commands.append(dict(id=index,call=current_call,op=name,unit=unit,deps=sorted(deps),
                macs=macs,weight_bytes=weight_bytes,vector_elements=max(out_elements,
                    max((t.numel() for t in inputs if info(t)[1] is None),default=0)),
                activation_read_bytes=activation_read,activation_write_bytes=activation_write,
                matrix_shape=matrix_shape,parameters=sorted(parameters)))
            for out in outputs:remember(out,{index})
            if func._schema.is_mutable:
                for argument,value in zip(func._schema.arguments,args):
                    if argument.alias_info is not None and argument.alias_info.is_write:
                        for t in tensors(value):remember(t,{index})
            return result
    def begin(module,args,kwargs):
        nonlocal current_call,call_dependency
        if current_call>=0:call_dependency=calls[current_call]['end_command']-1
        current_call+=1
        shape=list(kwargs['input_ids'].shape)
        calls.append(dict(index=current_call,input_shape=shape,start_command=len(commands)))
        observed_routes.append({})
    model.register_forward_pre_hook(begin,with_kwargs=True)
    # Position tables are metadata inputs, moved to the same abstract device.
    model.model.rotary_emb.register_forward_hook(lambda m,args,out:tuple(t.to('meta') for t in out))
    for li,layer in enumerate(model.model.layers):
        def visit(module,args,li=li):layer_visits[(current_call,li)]+=1
        layer.register_forward_pre_hook(visit)
        if not hasattr(layer.mlp,'route_tokens_to_experts'):continue
        def route(module,logits,li=li):
            count=logits.shape[0]
            rows=(supplied['calls'][current_call][str(li)] if supplied is not None else
                  routing_scenario(count,cfg.n_routed_experts,cfg.num_experts_per_tok,a.routing_scenario))
            checked_routes(rows,cfg.n_routed_experts,cfg.num_experts_per_tok,count)
            observed_routes[current_call][str(li)]=rows
            deps=info(logits)[0]
            if call_dependency>=0:deps=set(deps)|{call_dependency}
            index=len(commands)
            commands.append(dict(id=index,call=current_call,op='annotation.route_topk',unit='vector',
                deps=sorted(deps),macs=0,weight_bytes=0,
                vector_elements=count*cfg.n_routed_experts,
                activation_read_bytes=count*cfg.n_routed_experts*4,
                activation_write_bytes=count*cfg.num_experts_per_tok*8,
                matrix_shape=None,parameters=[]))
            route_dependencies[li]={index}
            ids=torch.tensor(rows,dtype=torch.long)
            weights=torch.empty((count,cfg.num_experts_per_tok),device='meta')
            remember(ids,{index});remember(weights,{index})
            return ids,weights
        layer.mlp.route_tokens_to_experts=types.MethodType(route,layer.mlp)
        def enter_experts(module,args,li=li):
            nonlocal active_route_dependencies
            active_route_dependencies=route_dependencies[li]
        def leave_experts(module,args,out):
            nonlocal active_route_dependencies
            active_route_dependencies=set()
        layer.mlp.experts.register_forward_pre_hook(enter_experts)
        layer.mlp.experts.register_forward_hook(leave_experts)
    def placeholder_logits(module,args,out):
        calls[current_call]['end_command']=len(commands)
        out.logits=torch.zeros(tuple(out.logits.shape),dtype=torch.float32)
        return out
    model.register_forward_hook(placeholder_logits)
    before=time.perf_counter()
    with torch.no_grad(),Trace():
        ids=model.generate(input_ids=torch.zeros((1,a.prompt_tokens),dtype=torch.long),
            attention_mask=torch.ones((1,a.prompt_tokens),dtype=torch.long),
            max_new_tokens=a.output_tokens,do_sample=False)
    if len(calls)!=a.output_tokens or calls[0]['input_shape']!=[1,a.prompt_tokens] or any(c['input_shape']!=[1,1] for c in calls[1:]):
        raise ValueError('unexpected upstream generation schedule')
    if supplied is not None and len(supplied['calls'])!=len(calls):raise ValueError('routing trace call count differs')
    if any(layer_visits[(i,l)]!=1 for i in range(len(calls)) for l in range(cfg.num_hidden_layers)):
        raise ValueError('upstream did not visit every main-model layer')
    atomic_json(a.output/'commands.json',dict(version=1,commands=commands,calls=calls,
        interface='Tensor DAG, not FeSens ISA',activation_precision_assumption='BF16 tensors with actual FP32 intermediate byte counts; routing cost is an explicit uncalibrated annotation'))
    atomic_json(a.output/'routing.json',dict(calls=observed_routes,
        provenance='external_annotation' if supplied is not None else 'sensitivity_'+a.routing_scenario,
        source_sha256=digest(a.routing_json) if a.routing_json else None))
    model_file=Path(__import__('inspect').getfile(type(model)))
    files=[Path(__file__),Path(__file__).with_name('glm47.py'),model_file,
           Path(transformers.__file__).parent/'generation/utils.py']
    result=dict(status='complete',scope=__doc__,model=identity,parameter_count=parameter_count,
        transformers_version=transformers.__version__,torch_version=torch.__version__,
        prompt_tokens=a.prompt_tokens,output_tokens=a.output_tokens,entrypoint='upstream Transformers model.generate',
        calls=calls,all_47_layers_visited_each_call=True,tensor_commands=len(commands),
        matrix_commands=sum(c['unit']=='matrix' for c in commands),matrix_macs=sum(c['macs'] for c in commands),
        weight_fetch_bytes_without_cache=sum(c['weight_bytes'] for c in commands),operator_counts=dict(counter),
        routing_provenance='external_annotation' if supplied is not None else 'sensitivity_'+a.routing_scenario,
        commands_sha256=digest(a.output/'commands.json'),routing_sha256=digest(a.output/'routing.json'),
        source_sha256={str(f):digest(f) for f in files},seconds=time.perf_counter()-start,
        generation_seconds=time.perf_counter()-before,weights_loaded=False,model_tensor_values_computed=False,
        text_or_accuracy_predicted=False,lpn_executed=False,rtl_validated=False,
        limitations=['Explicit routing annotations replace only data-dependent expert IDs/weights',
            'CPU placeholder logits let upstream generate choose standard prefill/decode; IDs are not predictions',
            'Reference MLA expands K/V before attention; compressed cache and absorbed MLA not substituted',
            'Tensor commands require a new hardware command adapter; existing FeSens compiler cannot execute this stream',
            'No activation quantization, physical SRAM ports, bank service or hardware cycle calibration yet'])
    atomic_json(a.output/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('operator_counts','source_sha256')},indent=2))


if __name__=='__main__':main()
