"""Audit shape work and dependencies independently of upstream GLM trace instrumentation."""
import argparse
import json
from pathlib import Path
import re


def audit(directory):
    from .e2e import digest
    from .glm47 import pinned_config
    from examples.tensor_dag import checked_commands
    report=json.loads((directory/'report.json').read_text())
    if report['status']!='complete':raise ValueError('completed trace required')
    for name,key in [('commands.json','commands_sha256'),('routing.json','routing_sha256')]:
        if digest(directory/name)!=report[key]:raise ValueError('trace artifact changed: '+name)
    for path,expected in report['source_sha256'].items():
        if digest(Path(path))!=expected:raise ValueError('frontend/upstream source changed')
    c,identity=pinned_config()
    if identity!=report['model']:raise ValueError('model metadata identity differs')
    document=json.loads((directory/'commands.json').read_text());commands=checked_commands(document)
    h,l=c['hidden_size'],c['num_hidden_layers'];n=c['num_attention_heads']
    qk=c['qk_nope_head_dim']+c['qk_rope_head_dim'];v=c['v_head_dim'];kr=c['kv_lora_rank'];qr=c['q_lora_rank']
    attention_linear=h*qr+qr*n*qk+h*(kr+c['qk_rope_head_dim'])+kr*n*(c['qk_nope_head_dim']+v)+n*v*h
    dense=c['first_k_dense_replace'];sparse=l-dense
    mlp=dense*3*h*c['intermediate_size']+sparse*(
        3*h*c['moe_intermediate_size']*(c['num_experts_per_tok']+c['n_shared_experts'])+h*c['n_routed_experts'])
    routes=json.loads((directory/'routing.json').read_text())
    records=[];position=0;previous_end=None
    for call in document['calls']:
        index=call['index'];tokens=call['input_shape'][1];position+=tokens
        selected=commands[call['start_command']:call['end_command']]
        if any(x['call']!=index for x in selected):raise ValueError('call bounds changed')
        if previous_end is not None and any(previous_end-1 not in x['deps'] for x in selected):
            raise ValueError('decode can start before previous logits complete')
        previous_end=call['end_command']
        expected_linear=tokens*(l*attention_linear+mlp)+h*c['vocab_size']
        expected_attention=l*n*tokens*position*(qk+v)
        linear=sum(x['macs'] for x in selected if x['op']=='aten.mm.default')
        attention=sum(x['macs'] for x in selected if x['op']=='aten.bmm.default')
        if (linear,attention)!=(expected_linear,expected_attention):
            raise ValueError(f'upstream tensor work differs at call {index}: {(linear,attention)} != {(expected_linear,expected_attention)}')
        route_nodes=[x for x in selected if x['op']=='annotation.route_topk']
        if len(route_nodes)!=sparse:raise ValueError('missing routed layer')
        by_layer={li:node for li,node in zip(range(dense,l),route_nodes)}
        from .glm47_trace import checked_routes
        for li,node in by_layer.items():
            checked_routes(routes['calls'][index][str(li)],c['n_routed_experts'],c['num_experts_per_tok'],tokens)
            if not any(f'model.layers.{li}.mlp.gate.weight' in commands[d]['parameters'] for d in node['deps']):
                raise ValueError('routing annotation lost its router-logit dependency')
        routed_matmuls=0
        for node in selected:
            if node['unit']!='matrix':continue
            for parameter in node['parameters']:
                match=re.match(r'model.layers.(\d+).mlp.experts.',parameter)
                if match:
                    routed_matmuls+=1
                    if by_layer[int(match[1])]['id'] not in node['deps']:
                        raise ValueError('expert matrix can execute before route readiness')
        records.append(dict(call=index,input_rows=tokens,kv_positions=position,linear_macs=linear,
            attention_macs=attention,routed_matmuls=routed_matmuls,
            work_matches_independent_config_formulas=True,route_and_generation_dependencies_preserved=True))
    if sum(r['linear_macs']+r['attention_macs'] for r in records)!=report['matrix_macs']:
        raise ValueError('aggregate work differs')
    return dict(status='complete',scope=__doc__,trace_report_sha256=digest(directory/'report.json'),
        commands_sha256=report['commands_sha256'],routing_sha256=report['routing_sha256'],records=records,
        all_topological_dependencies_valid=True,all_linear_and_attention_work_verified=True,
        all_router_and_decode_readiness_preserved=True,
        routing_provenance=report['routing_provenance'],numerical_or_rtl_validation=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('refusing to replace audit evidence')
    from .e2e import atomic_json
    atomic_json(a.output,audit(a.trace));print('shape/dependency audit passed',flush=True)


if __name__=='__main__':main()
