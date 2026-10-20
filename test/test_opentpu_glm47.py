"""Independent upstream inventory and quant-format checks for the MLA target."""
import argparse
from pathlib import Path
import tempfile
import unittest

from examples.opentpu.glm47 import inventory,kv_capacity,nvfp4_bytes,pinned_config,storage
from examples.opentpu.glm47_trace import checked_routes,routing_scenario


class GLMTest(unittest.TestCase):
    upstream=False

    def test_real_model_inventory_matches_checkpoint_tensor_names(self):
        import json
        from examples.opentpu.glm47 import CONFIG_ROOT
        c,_=pinned_config();items=inventory(c)
        index=json.loads((CONFIG_ROOT/'model.safetensors.index.json').read_text())['weight_map']
        self.assertTrue(all(t.name in index for t in items))
        self.assertFalse(any(t.layer>=47 for t in items))
        self.assertTrue(any(n.startswith('model.layers.47.') for n in index)) # explicitly excluded MTP
        self.assertEqual(sum(t.elements for t in items if t.kind!='router_buffer'),29943390976)
        self.assertEqual(len([t for t in items if t.kind=='expert']),46*64*3)

    def test_layout_requires_wide_addresses_and_counts_scales(self):
        c,_=pinned_config();layout=storage(inventory(c))
        self.assertFalse(layout['fits_32bit_address_space'])
        self.assertEqual(layout['minimum_address_bits'],34)
        self.assertGreater(layout['parts']['block_scales'],1_800_000_000)
        end=0
        for r in layout['regions']:
            self.assertEqual(r['address']%64,0)
            self.assertGreaterEqual(r['address'],end)
            end=r['address']+r['bytes']
        self.assertEqual(end,layout['total_bytes'])
        # Tails have separate K-block padding for each row.
        self.assertEqual(nvfp4_bytes((3,17)),dict(payload=27,block_scales=6,tensor_scale=4))

    def test_compressed_mla_does_not_silently_replace_reference_cache(self):
        c,_=pinned_config();small=kv_capacity(c,1024);large=kv_capacity(c,2048,batch=3)
        self.assertEqual(small['compressed_bytes_per_token_per_sequence'],47*(512+64)*2)
        self.assertEqual(small['expanded_bytes_per_token_per_sequence'],47*20*(256+256)*2)
        self.assertEqual(large['compressed_mla_bytes'],6*small['compressed_mla_bytes'])
        self.assertGreater(large['expanded_kv_bytes'],large['compressed_mla_bytes']*17)

    def test_annotations_are_explicit_and_topk_is_validated(self):
        balanced=routing_scenario(16,64,4,'balanced');hot=routing_scenario(16,64,4,'hotspot')
        self.assertEqual(len({e for r in balanced for e in r}),64)
        self.assertEqual(len({e for r in hot for e in r}),4)
        self.assertEqual(checked_routes(balanced,64,4,16),balanced)
        for bad in [[[0,0,1,2]],[[0,1,2,64]],[[0,1,2]],[[0,1,2,3.0]]]:
            with self.assertRaises(ValueError):checked_routes(bad,64,4,1)
        with self.assertRaises(ValueError):checked_routes(hot,64,4,8)

    def test_upstream_meta_parameters_and_exact_nvfp4_import(self):
        if not self.upstream:self.skipTest('requires pinned Transformers and FeSens environment')
        import torch
        import numpy as np
        from transformers import Glm4MoeLiteConfig,Glm4MoeLiteForCausalLM
        from opentpu import quant as Q
        c,_=pinned_config();items=inventory(c)
        with torch.device('meta'):model=Glm4MoeLiteForCausalLM(Glm4MoeLiteConfig(**c))
        actual=dict(model.named_parameters())
        expected={t.name:t.shape for t in items if t.kind not in ('expert','router_buffer')}
        for li in range(c['first_k_dense_replace'],c['num_hidden_layers']):
            expected[f'model.layers.{li}.mlp.experts.gate_up_proj']=(64,3072,2048)
            expected[f'model.layers.{li}.mlp.experts.down_proj']=(64,2048,1536)
        self.assertEqual({n:tuple(p.shape) for n,p in actual.items()},expected)
        self.assertEqual(sum(t.numel() for t in actual.values()),sum(t.elements for t in items if t.kind!='router_buffer'))
        codes=np.tile(np.arange(16,dtype=np.uint8),32).reshape(4,128)
        scales=np.tile(np.arange(48,56,dtype=np.uint8),(4,1))
        rows,srows=Q.import_nvfp4(codes,scales,.125,D=128)
        result=Q.dequantize_mxu(rows,srows,'nvfp4',128,D=128)
        # Independent format definition: signed E2M1 and positive normal E4M3.
        levels=np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6])
        e4=(1+(scales&7)/8)*np.exp2((scales>>3).astype(int)-7)
        expected=levels[codes]*np.repeat(e4,16,axis=1)*.125
        np.testing.assert_array_equal(result,expected)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path)
    a=p.parse_args()
    if a.opentpu_root:
        from examples.opentpu.workload import use_checkout
        use_checkout(a.opentpu_root);GLMTest.upstream=True
    unittest.main(argv=[__file__],verbosity=2)
