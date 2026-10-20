"""Exact packed per-slice GLM address layout, without allocating weight images.

The existing ISA has 32-bit addresses local to each slice. Replicated MLA and
partitioned expert ownership can use those existing address spaces, without
wrapping a >4-GiB single-slice image. This file establishes capacity only;
it does not claim complete inference commands or a calibrated multi-slice chip.
"""
import argparse
from dataclasses import asdict,replace
from math import prod
from pathlib import Path

from .e2e import atomic_json,digest
from .glm47 import pinned_config
from .workload import use_checkout


class ResidentLayout:
    def __init__(self,slices=8,rows=16,capacity=1152,prompt_rows=1024):
        from opentpu import Config,quant as Q
        from opentpu.llm.moe import ExpertFormat
        from .glm47_hardware import Layout as MLA,Spec
        if slices not in (1,2,4,8,16,32,64) or not 1<=rows<=16 or prompt_rows<rows:
            raise ValueError('invalid slice count or frontend tile capacity')
        c,_=pinned_config();E=c['n_routed_experts']
        if E%slices:raise ValueError('routed experts must partition evenly')
        H=c['hidden_size'];D=128;self.next=0;self.regions=[];self.layers=[]
        self.slices=slices;self.rows=rows;self.capacity=capacity;self.experts_per_slice=E//slices
        cfg=Config(D=D,MCOLS=1<<max(2,(rows-1).bit_length()),ACT_ROWS=16,LANES=64,
            TMEM_WORDS=262144,IMEM_WORDS=65536,DRAM_BYTES=1<<32)
        def alloc(name,n,kind):
            at=(self.next+D-1)//D*D;self.next=at+n
            self.regions.append(dict(name=name,address=at,bytes=n,kind=kind));return at
        def fp(name,shape):return alloc(name,prod(shape)*4,'fp32')
        self.inputs=fp('activation_ping',(prompt_rows,H));self.outputs=fp('activation_pong',(prompt_rows,H))
        self.tokens=fp('token_ids',(prompt_rows,))
        V=c['vocab_size'];rb=Q.row_bytes(H,'nvfp4',D);sb=Q.scale_bytes(H,'nvfp4',D)
        alloc('embedding.data',V*rb,'nvfp4');alloc('embedding.scales',V*sb,'mxu_scale_rows')
        alloc('embedding.gather_scales',V*Q.gather_scale_bytes(H,'nvfp4',D),'mxu_gather_scales')
        # The existing gather microcode's one-hot operand is independent of
        # the logical row-max capacity; use four operand rows as before.
        fp('embedding.onehot',(4,65*D))
        alloc('untied_head.data',V*rb,'nvfp4');alloc('untied_head.scales',V*sb,'mxu_scale_rows')
        fp('final_norm',(H,));fp('last_logits',(1,V))
        expert=ExpertFormat(H,c['moe_intermediate_size'],D,'nvfp4')
        dense=ExpertFormat(H,c['intermediate_size'],D,'nvfp4')
        for layer in range(c['num_hidden_layers']):
            base=(self.next+D-1)//D*D
            # Every slice has its own checked 4-GiB address space. This
            # capacity plan does not emit inference instructions.
            if base>=1<<32:raise ValueError('resident model exceeds one slice address space')
            mla=MLA(Spec.pinned(),rows,capacity,cfg,base=base)
            self.next=mla.next
            for region in mla.regions:self.regions.append({**region,'name':f'layer.{layer}.mla.'+region['name']})
            fp(f'layer.{layer}.post_norm',(H,))
            entry=dict(layer=layer,mla=mla,kind='dense' if layer<c['first_k_dense_replace'] else 'moe')
            if entry['kind']=='dense':
                entry['dense']=alloc(f'layer.{layer}.dense',dense.nbytes,'nvfp4_expert_format')
            else:
                fp(f'layer.{layer}.router',(E,H));fp(f'layer.{layer}.correction_bias',(E,))
                fp(f'layer.{layer}.router_logits',(rows,E));fp(f'layer.{layer}.router_ids',(rows,4))
                fp(f'layer.{layer}.router_weights',(rows,4))
                entry['experts']=alloc(f'layer.{layer}.local_experts',self.experts_per_slice*expert.nbytes,'nvfp4_expert_format')
                # Reserve the shared slot in every slice for a common layout;
                # only its designated owner needs to populate/read this slot.
                entry['shared']=alloc(f'layer.{layer}.shared_expert',expert.nbytes,'nvfp4_expert_format')
                fp(f'layer.{layer}.local_partial',(rows,H))
            entry['output']=fp(f'layer.{layer}.output',(rows,H));self.layers.append(entry)
        self.cfg=replace(cfg,S=slices);self.expert_format=expert;self.dense_format=dense
        if self.next>1<<32:raise ValueError('resident model exceeds one slice address space')

    def report(self):
        return dict(status='complete',scope=__doc__,slices=self.slices,rows=self.rows,
            kv_capacity=self.capacity,experts_per_slice=self.experts_per_slice,per_slice_bytes=self.next,
            aggregate_reserved_bytes=self.next*self.slices,allocated_hbm_bytes=(1<<32)*self.slices,
            per_slice_address_bits=32,all_regions_fit_32_bit=True,hardware_config=asdict(self.cfg),
            regions=self.regions,layer_count=len(self.layers),
            expert_ownership='Expert e resides in slice floor(e/experts_per_slice), local slot e%experts_per_slice',
            replicated=['MLA weights and expanded KV','embedding and untied head','router and norms','dense first-layer FFN'],
            shared_expert='Reserved everywhere, execute once on designated owner',
            cross_slice_combination='Existing hardware GATHER and FP32 adds required after local expert results',
            weights_allocated=False,full_glm_inference_compiled=False,fpga_fit_or_timing_proven=False,
            source_sha256={str(p):digest(p) for p in (Path(__file__),Path(__file__).with_name('glm47_hardware.py'))})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--slices',type=int,default=8);a=p.parse_args();use_checkout(a.opentpu_root)
    if a.output.exists():p.error('output must be new')
    result=ResidentLayout(a.slices).report();atomic_json(a.output,result)
    print(f"{result['per_slice_bytes']} bytes per slice; {result['aggregate_reserved_bytes']} total",flush=True)


if __name__=='__main__':main()
