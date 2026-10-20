"""Pinned GLM-4.7-Flash tensor inventory and large-memory preflight.

This frontend describes the real model, independently of LPN timing. It does
not synthesize a dense substitute, routing decisions, latency or generated text.
Main-model inventory excludes the checkpoint's optional MTP layer 47.
"""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from math import ceil, prod
from pathlib import Path


CONFIG_ROOT = Path(__file__).with_name('model_configs')/'glm47-flash'


def pinned_config(root=CONFIG_ROOT):
    root=Path(root)
    identity=json.loads((root/'provenance.json').read_text())
    for name,expected in identity['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('pinned model metadata changed: '+name)
    return json.loads((root/'config.json').read_text()),identity


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple
    kind: str
    layer: int = -1
    expert: int = -1

    @property
    def elements(self): return prod(self.shape)


def inventory(c):
    if c['model_type']!='glm4_moe_lite' or c['attention_bias'] or c['tie_word_embeddings']:
        raise ValueError('unsupported GLM configuration')
    h,l,q=c['hidden_size'],c['num_hidden_layers'],c['num_attention_heads']
    qr,kr,rope=c['q_lora_rank'],c['kv_lora_rank'],c['qk_rope_head_dim']
    nope,v=c['qk_nope_head_dim'],c['v_head_dim']
    items=[Tensor('model.embed_tokens.weight',(c['vocab_size'],h),'embedding')]
    def add(name,shape,kind,li,expert=-1):items.append(Tensor(name,tuple(shape),kind,li,expert))
    for li in range(l):
        p=f'model.layers.{li}.'
        for name,n,k in [('q_a_proj',qr,h),('q_b_proj',q*(nope+rope),qr),
                         ('kv_a_proj_with_mqa',kr+rope,h),('kv_b_proj',q*(nope+v),kr),
                         ('o_proj',h,q*v)]:
            add(p+'self_attn.'+name+'.weight',(n,k),'attention',li)
        for name,width in [('input_layernorm',h),('post_attention_layernorm',h),
                           ('self_attn.q_a_layernorm',qr),('self_attn.kv_a_layernorm',kr)]:
            add(p+name+'.weight',(width,),'norm',li)
        if li<c['first_k_dense_replace']:
            groups=[('mlp.',c['intermediate_size'],'dense',-1)]
        else:
            add(p+'mlp.gate.weight',(c['n_routed_experts'],h),'router',li)
            add(p+'mlp.gate.e_score_correction_bias',(c['n_routed_experts'],),'router_buffer',li)
            groups=[(f'mlp.experts.{e}.',c['moe_intermediate_size'],'expert',e)
                    for e in range(c['n_routed_experts'])]
            groups.append(('mlp.shared_experts.',c['moe_intermediate_size']*c['n_shared_experts'],'shared',-1))
        for prefix,ffn,kind,e in groups:
            for name,shape in [('gate_proj',(ffn,h)),('up_proj',(ffn,h)),('down_proj',(h,ffn))]:
                add(p+prefix+name+'.weight',shape,kind,li,e)
    items.extend([Tensor('model.norm.weight',(h,),'norm'),
                  Tensor('lm_head.weight',(c['vocab_size'],h),'head')])
    return items


def nvfp4_bytes(shape):
    """1D K-block layout: E2M1 payload, E4M3/16 and one FP32 tensor scale."""
    if len(shape)!=2 or min(shape)<1:raise ValueError('matrix shape required')
    n,k=shape
    return dict(payload=n*ceil(k/2),block_scales=n*ceil(k/16),tensor_scale=4)


def storage(items,embedding_bf16=False):
    totals=Counter();regions=[];offset=0
    for t in items:
        if t.kind in ('norm','router','router_buffer'):
            parts={'fp32':4*t.elements};dtype='fp32'
        elif t.kind=='embedding' and embedding_bf16:
            parts={'bf16':2*t.elements};dtype='bf16'
        else:
            parts=nvfp4_bytes(t.shape);dtype='nvfp4_1d'
        size=sum(parts.values());aligned=(offset+63)//64*64
        totals.update(parts);totals['alignment']+=aligned-offset
        regions.append(dict(name=t.name,shape=t.shape,kind=t.kind,layer=t.layer,expert=t.expert,
            address=aligned,bytes=size,dtype=dtype,parts=parts))
        offset=aligned+size
    return dict(total_bytes=offset,parts=dict(totals),regions=regions,
        minimum_address_bits=max(1,(offset-1).bit_length()),fits_32bit_address_space=offset<=1<<32)


def kv_capacity(c,tokens,batch=1,element_bytes=2):
    if min(tokens,batch,element_bytes)<1:raise ValueError('positive KV sizes required')
    compressed_per_token=c['num_hidden_layers']*(c['kv_lora_rank']+c['qk_rope_head_dim'])*element_bytes
    expanded_per_token=c['num_hidden_layers']*c['num_attention_heads']*(
        c['qk_nope_head_dim']+c['qk_rope_head_dim']+c['v_head_dim'])*element_bytes
    return dict(tokens=tokens,batch=batch,element_bytes=element_bytes,
        compressed_mla_bytes=compressed_per_token*tokens*batch,
        expanded_kv_bytes=expanded_per_token*tokens*batch,
        compressed_bytes_per_token_per_sequence=compressed_per_token,
        expanded_bytes_per_token_per_sequence=expanded_per_token,
        scope='Compressed MLA needs a compressed-cache/absorbed attention backend; expanded reference stores per-head K/V. Neither includes transient workspaces.')


def decode_work(c,items,layout,context):
    """Necessary per-token service; not elapsed time or a cache-hit prediction."""
    if context<1:raise ValueError('positive context required')
    resident=[t for t in items if t.kind not in ('embedding','expert')]
    active_macs=sum(t.elements for t in resident if len(t.shape)==2)
    routed_layers=c['num_hidden_layers']-c['first_k_dense_replace']
    expert_macs=routed_layers*c['num_experts_per_tok']*3*c['hidden_size']*c['moe_intermediate_size']
    regions={r['name']:r for r in layout['regions']}
    # One row lookup is needed, not an entire embedding matrix transfer.
    h=c['hidden_size'];embed=next(t for t in items if t.kind=='embedding')
    embedding_bytes=(2*h if regions[embed.name]['dtype']=='bf16' else sum(nvfp4_bytes((1,h)).values()))
    fixed=sum(regions[t.name]['bytes'] for t in resident)
    expert_bytes=sum(regions[t.name]['bytes'] for t in items if t.kind=='expert' and t.expert<c['num_experts_per_tok'])
    # Shape-only lower service for two possible MLA execution strategies.
    heads=c['num_attention_heads'];qk=c['qk_nope_head_dim']+c['qk_rope_head_dim']
    conventional_attention=2*heads*context*(qk+c['v_head_dim'])*c['num_hidden_layers']
    # Avoid full KV expansion: absorb K into Q, aggregate compressed latent,
    # then project latent into V. Precision/rounding differs unless validated.
    latent=c['kv_lora_rank'];nope=c['qk_nope_head_dim'];v=c['v_head_dim']
    absorbed_extra_macs=c['num_hidden_layers']*heads*(nope*latent+latent*v)
    expanded_kv_projection=c['num_hidden_layers']*latent*heads*(nope+v)
    absorbed_attention=2*heads*context*(2*latent+c['qk_rope_head_dim'])*c['num_hidden_layers']
    return dict(linear_macs=active_macs+expert_macs,
        active_routed_experts_per_layer=c['num_experts_per_tok'],routed_expert_macs=expert_macs,
        uncached_weight_and_scale_bytes=fixed+expert_bytes+embedding_bytes,
        conventional_attention_flops=conventional_attention,
        absorbed_attention_flops=absorbed_attention,
        absorbed_projection_macs=active_macs+expert_macs-expanded_kv_projection+absorbed_extra_macs,
        scope='Selected experts only; no inter-token weight residency, vector/router/scale compute, padding, activation traffic or physical latency. Absorbed MLA is a separate unvalidated implementation, not inferred from the expanded reference.')


def compute_points(work):
    points=[]
    # Equal-clock designs use fixed GB/s, never an unchanged bytes/cycle when
    # increasing frequency. MAC counts do not imply FPGA packing or fit.
    for name,macs,mhz,bw,hbm,kind in [
        ('fpga_current_budget',12288,200,567,32_000_000_000,'fpga_capacity_unproven'),
        ('fpga_more_compute_2x',24576,200,567,32_000_000_000,'fpga_capacity_unproven'),
        ('asic_1x_same_hbm',12288,1500,858.9934592,32_000_000_000,'asic_proposal'),
        ('asic_4x_same_hbm',49152,1500,858.9934592,32_000_000_000,'asic_proposal'),
        ('asic_16x_same_hbm',196608,1500,858.9934592,32_000_000_000,'asic_proposal')]:
        seconds_compute=work['linear_macs']/(macs*mhz*1e6)
        seconds_memory=work['uncached_weight_and_scale_bytes']/(bw*1e9)
        points.append(dict(name=name,kind=kind,macs_per_cycle=macs,clock_mhz=mhz,
            hbm_gbs=bw,hbm_capacity_bytes=hbm,hbm_bytes_per_cycle=bw*1000/mhz,
            logical_mac_ops=2,weight_precision='NVFP4 weight-only; no assumed 2x MAC throughput',
            peak_tops=2*macs*mhz/1e6,linear_compute_floor_ms=seconds_compute*1000,
            uncached_weight_fetch_floor_ms=seconds_memory*1000,
            max_of_two_service_floors_ms=max(seconds_compute,seconds_memory)*1000,
            not_a_latency_prediction=True,rtl_implemented=False,physical_fit_proven=False))
    return points


def report(c,identity,contexts,embedding_bf16=False):
    items=inventory(c);layout=storage(items,embedding_bf16)
    kv=[kv_capacity(c,n) for n in contexts]
    work=decode_work(c,items,layout,contexts[0])
    return dict(status='complete',scope=__doc__,model=identity,config=c,
        parameter_count=sum(t.elements for t in items if t.kind!='router_buffer'),
        optional_mtp_included=False,routing='No expert IDs generated; only fixed top-k active counts for necessary service bounds',
        storage=layout,kv=kv,decode_work=work,compute_points=compute_points(work),
        full_lpn_inference_executed=False,weights_loaded=False,numerical_accuracy_validated=False,
        existing_frontend_support=dict(glm_mla=False,moe_routing=False,whole_image_32bit=layout['fits_32bit_address_space']),
        required_work=['Upstream GLM instruction frontend and real routing trace interface',
            'Wide HBM addresses or explicit bank/window translation without 32-bit wrapping',
            'Bounded MLA/quantization/expert dataflow and scratchpad ports',
            'NVFP4 hardware-format gates and full-kernel timing calibration'],
        sources=dict(config=f"https://huggingface.co/{identity['model']}/blob/{identity['revision']}/config.json",
            nvfp4='https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--contexts',default='1032,32768,131072')
    p.add_argument('--embedding-bf16',action='store_true')
    a=p.parse_args();contexts=list(map(int,a.contexts.split(',')))
    if not contexts or min(contexts)<1:p.error('positive contexts required')
    c,identity=pinned_config();result=report(c,identity,contexts,a.embedding_bf16)
    from .e2e import atomic_json
    if a.output.exists():p.error('refusing to replace existing report')
    a.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(a.output,result)
    print(json.dumps({k:result[k] for k in ('parameter_count','kv','decode_work','compute_points')},indent=2))


if __name__=='__main__':main()
