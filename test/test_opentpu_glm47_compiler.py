"""Actual ISA compiler gates for GLM MLA, causal masking and FP32 routing."""
import argparse
from pathlib import Path
import tempfile
import unittest

from examples.opentpu.workload import use_checkout,execute


class GLMCompilerTest(unittest.TestCase):
    tool=None

    def test_epilogue_profile_tracks_physical_rtl_columns(self):
        from examples.opentpu.glm47_target import component_timing
        for columns,lanes in ((1,1),(2,1),(4,2),(8,4)):
            with self.subTest(columns=columns):
                self.assertEqual(component_timing(columns=columns).epilogue_lanes,lanes)
        with self.assertRaises(ValueError):component_timing(columns=3)

    def test_boundary_kernels_follow_numerical_equations(self):
        import numpy as np
        from examples.opentpu.glm47_boundary_hardware import Layout,numerical_inputs
        for stage in ('embedding','dense','head'):
            with self.subTest(stage=stage):
                layout=Layout(stage,2,128,512,256);tokens=[38,7]
                program=layout.compile(tokens);image,reference=numerical_inputs(layout,tokens)
                work=execute(layout.cfg,[program],[image]);at=layout.m.out.base.static()
                got=work.machine.slices[0].dram[at:at+reference.size*4].view(np.float32).reshape(reference.shape)
                self.assertLess(float(np.linalg.norm(got-reference)/np.linalg.norm(reference)),.02)
                if stage=='embedding':
                    np.testing.assert_allclose(got,reference,atol=1e-7,rtol=1e-6)
                    with self.assertRaises(ValueError):layout.compile([256,7])
                if stage=='head':
                    # Generation consumes only the last input row's logits.
                    image[:128*4]=np.full(128,1e3,np.float32).view(np.uint8)
                    other=execute(layout.cfg,[program],[image]).machine.slices[0].dram[at:at+reference.size*4]
                    np.testing.assert_array_equal(other,got.view(np.uint8).ravel())

    def test_full_dense_and_head_command_arithmetic_matches_dimensions(self):
        from examples.opentpu.glm47_boundary_hardware import Layout
        from examples.opentpu.shape_workload import resolve_static
        from examples.opentpu.hardware_commands import instruction_statistics
        H,F,V=2048,10240,154880
        for stage,macs in (('dense',3*4*H*F),('head',H*V)):
            with self.subTest(stage=stage):
                layout=Layout(stage,4,H,F,V)
                stats=instruction_statistics(resolve_static(layout.cfg,[layout.compile()]))
                self.assertEqual(stats['matrix_dot_macs'],macs)
                self.assertLess(layout.next,1<<32)

    def test_large_mla_tile_requires_real_scratchpad_and_rowmax_capacity(self):
        from dataclasses import replace
        from opentpu import Config,waits as W
        from opentpu.compiler import CompileError
        from examples.opentpu.glm47_hardware import Layout,Spec
        from examples.opentpu.shape_workload import resolve_static
        cfg=Config(D=128,MCOLS=4,ACT_ROWS=16,LANES=64,DRAM_BYTES=64<<20,IMEM_WORDS=65536)
        with self.assertRaisesRegex(CompileError,'TMEM exhausted'):
            Layout(Spec.pinned(),16,1152,cfg).compile(0)
        cfg=replace(cfg,TMEM_WORDS=262144)
        with self.assertRaisesRegex(CompileError,'rowmax'):
            Layout(Spec.pinned(),16,1152,cfg).compile(0)
        cfg=replace(cfg,MCOLS=16)
        program=Layout(Spec.pinned(),16,1152,cfg).compile(0)
        self.assertLess(len(program)*8,cfg.IMEM_WORDS)
        work=resolve_static(cfg,[program])
        for command in work.commands[0]:
            for space,lo,hi in command.footprint.rd+command.footprint.wr:
                if space==W.SP_TMEM:self.assertTrue(0<=lo<=hi<=cfg.TMEM_WORDS)

    def test_expert_dispatch_preserves_every_selected_token_and_rank(self):
        from examples.opentpu.glm47_kernels import expert_batches
        ids=[[7,1,3,0],[1,6,7,3],[7,2,1,0],[2,4,1,7]]
        batches=expert_batches(ids,8,2)
        self.assertTrue(all(len(items)<=2 for e,items in batches))
        self.assertEqual([e for e,items in batches],sorted(e for e,items in batches))
        actual=sorted((token,rank,e) for e,items in batches for token,rank in items)
        self.assertEqual(actual,sorted((token,rank,e) for token,row in enumerate(ids) for rank,e in enumerate(row)))
        with self.assertRaises(ValueError):expert_batches([[0,0]],8,2)
        with self.assertRaises(ValueError):expert_batches([[8]],8,2)

    def test_interleaved_rotary_matches_independent_equations_bit_exact(self):
        import numpy as np
        from opentpu import Config
        from opentpu.isasim import Machine
        from opentpu.runtime import Input,Output,compile_kernel,collect
        from examples.opentpu.glm47_kernels import rotary_test
        rng=np.random.default_rng(211);x=rng.normal(size=(3,64)).astype(np.float32)
        cos=np.cos(rng.normal(size=(3,32))).astype(np.float32)
        sin=np.sin(rng.normal(size=(3,32))).astype(np.float32)
        cfg=Config(D=128,MCOLS=4,ACT_ROWS=16,LANES=64)
        compiled,images=compile_kernel(rotary_test,cfg,x=Input(x),cos=Input(cos),sin=Input(sin),out=Output(x.shape))
        machine=Machine(cfg,compiled.programs,images).run()
        got=collect(compiled,[s.dram for s in machine.slices],[s.tmem for s in machine.slices],{}).outputs['out']
        a,b=x[:,0::2],x[:,1::2]
        expected=np.concatenate((a*cos-b*sin,b*cos+a*sin),axis=-1)
        np.testing.assert_array_equal(got.view(np.uint32),expected.view(np.uint32))

    def test_future_token_cannot_change_previous_mla_output(self):
        import numpy as np
        from opentpu import Config
        from opentpu.isasim import Machine
        from examples.opentpu.glm47_hardware import Layout,Spec,numerical_inputs
        cfg=Config(D=128,MCOLS=4,ACT_ROWS=16,LANES=64)
        layout=Layout(Spec(256,2,256,128,64,64,128),2,128,cfg)
        program=layout.compile(0);image,_=numerical_inputs(layout,0)
        changed=image.copy();start=layout.m.x.base.static()+4*256
        changed[start:start+4*256]=np.linspace(-8,8,256,dtype=np.float32).view(np.uint8)
        a=Machine(cfg,[program],[image]).run().slices[0]
        b=Machine(cfg,[program],[changed]).run().slices[0]
        out=layout.m.out.base.static()
        np.testing.assert_array_equal(a.dram[out:out+4*256],b.dram[out:out+4*256])
        self.assertFalse(np.array_equal(a.dram[out+4*256:out+8*256],b.dram[out+4*256:out+8*256]))

    def test_router_uses_real_data_dependent_instructions_and_unbiased_weights(self):
        import numpy as np
        from opentpu import Config,isa as I
        from examples.opentpu.glm47_router_hardware import compile_router
        from examples.opentpu.hardware_commands import instruction_statistics
        from examples.opentpu.model import Timing
        from examples.opentpu.native_timing import ReusableTiming
        rng=np.random.default_rng(77);R,H,E=2,256,8
        x=rng.normal(size=(R,H)).astype(np.float32);gamma=np.ones(H,np.float32)
        w=(rng.normal(size=(E,H))/H**.5).astype(np.float32);bias=np.zeros(E,np.float32);bias[-1]=2
        cfg=Config(D=128,MCOLS=4,ACT_ROWS=16,LANES=64)
        program,m,image=compile_router(cfg,x,gamma,w,bias)
        work=execute(cfg,[program],[image]);gold=work.machine.slices[0]
        def get(desc):
            at=desc.base.static();return gold.dram[at:at+4*np.prod(desc.shape)].view(np.float32).reshape(desc.shape)
        logits=(x/np.sqrt(np.mean(x*x,axis=1,keepdims=True)+1e-5))@w.T
        scores=1/(1+np.exp(-logits));ids=np.argsort(-(scores+bias),axis=1,kind='stable')[:,:4]
        np.testing.assert_array_equal(get(m.ids).astype(int),ids)
        expected=np.take_along_axis(scores,ids,axis=1);expected=expected/expected.sum(axis=1,keepdims=True)*1.8
        np.testing.assert_allclose(get(m.weights),expected,atol=2e-5,rtol=2e-5)
        stats=instruction_statistics(work)
        self.assertEqual(stats['matrix_commands'],0);self.assertEqual(stats['opcode_counts']['RLD'],R*4)
        self.assertEqual(stats['opcode_counts']['VOP']>0,True)
        if self.tool:
            from examples.opentpu.compact import CompactOpenTPU
            timing=Timing(dram_bytes_per_cycle=128)
            sim=CompactOpenTPU(work,timing).net.python_simulator();ticks=sim.run(max_time=100000)
            self.assertEqual(sim.place_dicts('program_completed'),[{}])
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);native=ReusableTiming(self.tool,root/'cache',capacity=len(work.commands[0])).prepare(work,timing,root/'inputs')
                self.assertEqual(native.run()[0],ticks)

    def test_batched_router_is_bit_exact_to_scalar_reductions(self):
        import numpy as np
        from opentpu import Config
        from examples.opentpu.glm47_router_hardware import compile_router
        rng=np.random.default_rng(1447);R,H,E=4,256,8
        x=rng.normal(size=(R,H)).astype(np.float32);norm=np.ones(H,np.float32)
        weight=(rng.normal(size=(E,H))/H**.5).astype(np.float32);bias=rng.normal(size=E).astype(np.float32)
        cfg=Config(D=128,MCOLS=4,ACT_ROWS=16,LANES=64)
        values=[];counts=[]
        for batched in (False,True):
            program,m,image=compile_router(cfg,x,norm,weight,bias,batched=batched)
            work=execute(cfg,[program],[image]);raw=work.machine.slices[0].dram
            counts.append(len(work.commands[0]));result=[]
            for name in ('logits','ids','weights'):
                desc=getattr(m,name);start=desc.base.static();size=int(np.prod(desc.shape))*4
                result.append(raw[start:start+size].copy())
            values.append(result)
        for before,after in zip(*values):np.testing.assert_array_equal(before,after)
        self.assertLess(counts[1],counts[0])

    def test_real_glm_projection_work_is_in_hardware_commands(self):
        from opentpu import Config
        from examples.opentpu.glm47_hardware import Layout,Spec
        from examples.opentpu.shape_workload import resolve_static
        from examples.opentpu.hardware_commands import instruction_statistics
        s=Spec.pinned();R=4
        cfg=Config(D=128,MCOLS=4,ACT_ROWS=16,LANES=64,DRAM_BYTES=64<<20)
        layout=Layout(s,R,1152,cfg);program=layout.compile(0,stage='project')
        stats=instruction_statistics(resolve_static(cfg,[program]))
        expected=R*(s.hidden*s.q_rank+s.q_rank*s.heads*(s.nope+s.rope)+
            s.hidden*(s.kv_rank+s.rope)+s.kv_rank*s.heads*(s.nope+s.value))
        self.assertEqual(stats['matrix_dot_macs'],expected)
        self.assertEqual(stats['matrix_by_weight_format']['nvfp4']['dot_macs'],expected)
        self.assertEqual(stats['opcode_counts']['QST'],2*s.heads)
        self.assertEqual(stats['matrix_commands'],2+2*s.heads)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--opentpu-root',type=Path,required=True);p.add_argument('--tool',type=Path)
    a=p.parse_args();use_checkout(a.opentpu_root);GLMCompilerTest.tool=a.tool
    unittest.main(argv=[__file__],verbosity=2)
