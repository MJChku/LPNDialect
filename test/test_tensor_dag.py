"""Dependency, shared-bus and native-runtime gates for the generic DAG screen."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from examples.tensor_dag import Hardware,TensorDAGScreen,checked_commands


def command(index,deps=(),unit='matrix',macs=0,traffic=0,elements=0):
    return dict(id=index,deps=list(deps),unit=unit,macs=macs,weight_bytes=traffic,
        vector_elements=elements,activation_read_bytes=0,activation_write_bytes=0)


class DAGTest(unittest.TestCase):
    tool=None

    def run_model(self,commands,hw):
        from lpnlang_mlir.native_inputs import write_input_tables
        model=TensorDAGScreen(dict(version=1,commands=commands),hw)
        sim=model.net.python_simulator();cycles=sim.run(max_time=100000)
        self.assertEqual(sim.place_dicts('program_completed'),[{}])
        if self.tool:
            from examples.opentpu.benchmark import native_observation
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);ir=root/'model.mlir';ir.write_text(model.net.build())
                tables=root/'inputs';write_input_tables(tables,model.tables)
                binary=root/'model.native'
                subprocess.run([str(self.tool),str(ir),'-o',str(binary)],capture_output=True,text=True,check=True)
                self.assertEqual(native_observation(binary,inputs=tables)[0],cycles)
        return cycles

    def test_hbm_service_stays_constant_when_compute_is_increased(self):
        hw=Hardware(clock_mhz=1000,hbm_gbs=1,weight_macs_per_cycle=1,pipeline_cycles=1)
        work=[command(0,macs=2,traffic=100)]
        a=self.run_model(work,hw);b=self.run_model(work,replace(hw,weight_macs_per_cycle=1000))
        self.assertEqual(a,b);self.assertEqual(a,101)

    def test_dependency_fork_and_shared_bus(self):
        hw=Hardware(clock_mhz=1000,hbm_gbs=1,weight_macs_per_cycle=1,
                    bf16_macs_per_cycle=1,vector_elements_per_cycle=1,pipeline_cycles=1)
        independent=[command(0,unit='matrix',macs=100),command(1,unit='vector',elements=100),
                     command(2,(0,1),unit='vector',elements=1)]
        serial=[independent[0],dict(independent[1],deps=[0]),independent[2]]
        self.assertLess(self.run_model(independent,hw),self.run_model(serial,hw))
        a=[command(0,macs=1,traffic=100),command(1,unit='vector',elements=1,traffic=100)]
        self.assertGreaterEqual(self.run_model(a,hw),200)

    def test_completion_flags_and_window_wrap(self):
        hw=Hardware(window=4,pipeline_cycles=1)
        work=[command(i,(i-1,) if i else (),unit='vector',elements=1) for i in range(70)]
        self.assertEqual(self.run_model(work,hw),140)

    def test_bad_graphs_and_budgets_are_rejected(self):
        for cmd in [command(0,[0]),command(0,[-1]),dict(command(0),weight_bytes=-1),dict(command(0),unit='unknown')]:
            with self.assertRaises(ValueError):checked_commands(dict(version=1,commands=[cmd]))
        with self.assertRaises(ValueError):Hardware(hbm_gbs=0)
        with self.assertRaises(ValueError):Hardware(hbm_gbs=float('nan'))
        with self.assertRaises(ValueError):
            checked_commands(dict(version=1,commands=[dict(command(0,macs=2),matrix_shape=dict(batch=1,m=1,n=1,k=1))]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tool',type=Path);a=p.parse_args();DAGTest.tool=a.tool
    unittest.main(argv=[__file__],verbosity=2)
