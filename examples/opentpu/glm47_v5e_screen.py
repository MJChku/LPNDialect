"""Replay audited upstream GLM DAGs on proposed v5e-scale array organizations.

This is an uncalibrated architecture study, not v5e or finished openTPU RTL.
All points share matrix resources across weight, attention and router work.
NVFP4 saves weight bytes but receives no automatic arithmetic speed multiplier.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

from examples.tensor_dag import ArrayGeometry,Hardware
from .e2e import atomic_json,digest


def profiles():
    # 65,536 MAC sites at the explicitly assumed 1.5 GHz gives 196.608 TFLOP/s.
    # Vector/FP32 budgets and 32 GB are our proposals, not Google's published internals.
    return [
        ('four_128x128',ArrayGeometry(128,128,4),858.9934592),
        ('sixtyfour_32x32',ArrayGeometry(32,32,64),858.9934592),
        ('sixtyfour_16x64',ArrayGeometry(16,64,64),858.9934592),
        ('eight_128x128',ArrayGeometry(128,128,8),858.9934592),
        ('sixtyfour_16x64_hbm2x',ArrayGeometry(16,64,64),1717.9869184),
    ]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces',type=Path,required=True,help='Existing audited screen root containing trace-balanced and trace-hotspot')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--runs',type=int,default=3)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),Path(__file__).parents[1]/'tensor_dag.py']
    hashes={str(f):digest(f) for f in sources};records=[];started=time.perf_counter()
    def save(status):
        atomic_json(a.output/'report.json',dict(status=status,scope=__doc__,source_sha256=hashes,
            records=records,host_total_seconds=time.perf_counter()-started,
            baseline_reference=dict(url='https://docs.cloud.google.com/tpu/docs/v5e?hl=en',
                bf16_tflops=197,hbm_gib_per_second=800,hbm_capacity_listed='16 GB',mxus=4),
            assumptions=['1.5 GHz design assumption; no FPGA or ASIC Fmax proof',
                '32 decimal GB proposed HBM capacity, larger than v5e listed 16 GB',
                'NVFP4-weight dequantization assumed sustainable at BF16 array throughput; not calibrated',
                '1024 FP32 MACs/cycle and 1024 vector elements/cycle are proposed budgets',
                'All matrix precisions share one cluster; whole operations reserve it',
                'Output-stationary padded tiles with fill/drain, no interoperation tile packing',
                'All intermediates spill to HBM; reference expanded K/V, not compressed MLA',
                'No detailed tile refill, physical SRAM ports, SFU or controller calibration',
                'Routing is a sensitivity annotation and logits are placeholders; no text inference'],
            rtl_calibrated=False,fpga_mapping_proven=False,commercial_tpu_latency_comparison=False))
    save('running')
    for scenario in ('balanced','hotspot'):
        trace=a.traces/('trace-'+scenario);commands=trace/'commands.json'
        upstream=json.loads((trace/'report.json').read_text())
        if digest(commands)!=upstream['commands_sha256']:raise ValueError('upstream graph hash mismatch')
        for name,geometry,bw in profiles():
            hw=Hardware(weight_macs_per_cycle=geometry.macs_per_cycle,
                bf16_macs_per_cycle=geometry.macs_per_cycle,fp32_macs_per_cycle=1024,
                vector_elements_per_cycle=1024,hbm_gbs=bw,tile_buffer_bytes_per_unit=1048576)
            dest=a.output/(scenario+'-'+name)
            args=[sys.executable,'-m','examples.tensor_dag','--commands',str(commands),
                '--output',str(dest),'--compiler',str(a.compiler.resolve()),'--shared-matrix-unit',
                '--array-rows',str(geometry.rows),'--array-columns',str(geometry.columns),
                '--arrays',str(geometry.arrays),'--runs',str(a.runs)]
            for key,value in asdict(hw).items():
                if key in ('pipeline_cycles','window'):continue
                args.extend(['--'+key.replace('_','-'),str(value)])
            with (a.output/(scenario+'-'+name+'.log')).open('w') as out:
                subprocess.run(args,stdout=out,stderr=subprocess.STDOUT,check=True)
            result=json.loads((dest/'report.json').read_text())
            records.append(dict(scenario=scenario,name=name,**result,
                upstream_trace_sha256=digest(trace/'report.json'),report_sha256=digest(dest/'report.json')))
            save('running')
            print(json.dumps(dict(scenario=scenario,name=name,cycles=result['cycles'],
                hardware_seconds=result['assumed_hardware_seconds'],native_seconds=result['native_seconds_samples'],
                achieved_matrix_tflops=result['arithmetic']['achieved_matrix_tflops'])),flush=True)
    if hashes!={str(f):digest(f) for f in sources}:raise ValueError('sources changed during run')
    save('complete')


if __name__=='__main__':main()
