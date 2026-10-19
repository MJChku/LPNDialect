"""Explore TPU-style hardware on the retained, normal-inference RTL workload.

This consumes unchanged upstream programs, not a new workload schedule. Pilot
extrapolations select candidates only; a 100x result requires all 48 kernels
and subsequent modified-RTL numerical/cycle validation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time


@dataclass(frozen=True)
class Architecture:
    name: str
    rows: int
    columns: int
    bandwidth: int
    latency: int = 8
    ceiling_only: bool = False
    buffered_quant: bool = False
    quant_scale_lookup: bool = False
    scalar_recip_lookup: bool = False
    reciprocal_lookup_lanes: int = 0
    pinned_cache_bytes: int = 0
    cache_read_bytes_per_cycle: int = 8192
    vector_row_contexts: int = 1
    vector_engines: int = 1
    vector_issue_groups: int = 1
    vector_result_chunks: int = 0
    vector_compatible_issue: bool = False
    window: int = 32
    activation_streaming: bool = False
    fp_mul_latency: int = 2
    fp_add_latency: int = 4
    vector_command_queue: int = 0
    matrix_retirement: bool = False
    physical_scale_lines: bool = False
    act_rows: int = 0
    act_blocks: int = 0
    fifo_per_core: int = 128
    matrix_engines: int = 1
    precise_tmem: bool = False
    scale_cache_entries: int = 0
    paired_dsp: bool = False


def architectures():
    return [
        *[Architecture(f'proposal_wide_elementwise_g{groups}_96x2_scale16_add2',96,2,2816,
                       vector_issue_groups=groups,scale_cache_entries=16,paired_dsp=True,
                       buffered_quant=True,vector_row_contexts=4,activation_streaming=True,
                       quant_scale_lookup=True,scalar_recip_lookup=True,fp_add_latency=2,
                       vector_command_queue=2,matrix_retirement=True,physical_scale_lines=True,
                       act_rows=8,fifo_per_core=64,act_blocks=64)
          for groups in (2,4,8)],
        *[Architecture(name,rows,cols,2816,scale_cache_entries=16,paired_dsp=cols>1,
                       buffered_quant=True,vector_row_contexts=4,activation_streaming=True,
                       quant_scale_lookup=True,scalar_recip_lookup=True,fp_add_latency=2,
                       vector_command_queue=2,matrix_retirement=True,physical_scale_lines=True,
                       act_rows=act,fifo_per_core=fifo,act_blocks=64,pinned_cache_bytes=cache)
          for name,rows,cols,act,fifo,cache in [
              ('physical_scale16_128x1_add2',128,1,16,128,0),
              ('proposal_compact_scale16_128x1_add2',128,1,8,64,0),
              ('proposal_paired_scale16_96x2_add2',96,2,8,64,0),
              ('proposal_paired_scale16_head28m_96x2_add2',96,2,8,64,28<<20)]],
        *[Architecture(f'proposal_precise_{rows}x{cols}_v{vectors}_w{window}',rows,cols,2816,
                       precise_tmem=True,buffered_quant=True,vector_row_contexts=4,
                       activation_streaming=True,quant_scale_lookup=True,scalar_recip_lookup=True,
                       fp_add_latency=2,matrix_retirement=True,physical_scale_lines=True,
                       act_rows=8,act_blocks=64,fifo_per_core=64,window=window,
                       vector_command_queue=2 if vectors==1 else 0,vector_engines=vectors,
                       vector_result_chunks=1024 if vectors>1 else 0,vector_compatible_issue=vectors>1)
          for rows,cols,vectors,window in [(128,1,1,32),(96,2,1,32),(64,1,4,128),(64,1,8,128)]],
        *[Architecture(f'proposal_window_{window}_128x1',128,1,2816,
                       buffered_quant=True,vector_row_contexts=4,activation_streaming=True,
                       quant_scale_lookup=True,scalar_recip_lookup=True,fp_add_latency=2,
                       vector_command_queue=2,matrix_retirement=True,physical_scale_lines=True,
                       act_rows=8,act_blocks=64,fifo_per_core=64,window=window)
          for window in (64,128,256)],
        *[Architecture(f'proposal_asic_clusters_{engines}x128x8_v16_precise{int(precise)}',128,8,32768,
                       matrix_engines=engines,precise_tmem=precise,buffered_quant=True,
                       vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,
                       scalar_recip_lookup=True,fp_mul_latency=1,fp_add_latency=1,
                       matrix_retirement=True,physical_scale_lines=True,act_rows=8,act_blocks=64,
                       fifo_per_core=64,vector_engines=16,vector_result_chunks=1024,
                       vector_compatible_issue=True,window=128)
          for engines in (4,16) for precise in (False,True)],
        *[Architecture(f'proposal_clusters_{engines}x{128//engines}x1_add2',128//engines,1,2816,
                       matrix_engines=engines,buffered_quant=True,vector_row_contexts=4,
                       activation_streaming=True,quant_scale_lookup=True,scalar_recip_lookup=True,
                       fp_add_latency=2,vector_command_queue=2,matrix_retirement=True,
                       physical_scale_lines=True,act_rows=8,act_blocks=64,fifo_per_core=64)
          for engines in (2,4,8,16)],
        Architecture('physical_multicol_64x2_add2',64,2,2816,buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,
                     scalar_recip_lookup=True,fp_add_latency=2,vector_command_queue=2,
                     matrix_retirement=True,physical_scale_lines=True,act_rows=8,act_blocks=64,fifo_per_core=64),
        *[Architecture(f'proposal_pair_{rows}x2_add2',rows,2,2816,buffered_quant=True,
                       vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,
                       scalar_recip_lookup=True,fp_add_latency=2,vector_command_queue=2,
                       matrix_retirement=True,physical_scale_lines=True,act_rows=8,act_blocks=64,fifo_per_core=64)
          for rows in (96,112,128)],
        *[Architecture(f'proposal_asic_{rows}x8_vpu16_bw{bw}',rows,8,bw,buffered_quant=True,
                       vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,
                       scalar_recip_lookup=True,fp_mul_latency=1,fp_add_latency=1,
                       matrix_retirement=True,physical_scale_lines=True,act_rows=8,act_blocks=64,
                       fifo_per_core=64,vector_engines=16,vector_result_chunks=1024,
                       vector_compatible_issue=True,window=128)
          for rows,bw in ((256,8192),(1024,8192),(1024,32768))],
        Architecture('physical_compact_act_128x1_add2',128,1,2816,buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,
                     scalar_recip_lookup=True,fp_add_latency=2,vector_command_queue=2,
                     matrix_retirement=True,physical_scale_lines=True,act_rows=8,act_blocks=64,fifo_per_core=64),
        Architecture('physical_compact_fifo32_128x1_add2',128,1,2816,buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,
                     scalar_recip_lookup=True,fp_add_latency=2,vector_command_queue=2,
                     matrix_retirement=True,physical_scale_lines=True,act_rows=8,act_blocks=64,fifo_per_core=32),
        *[Architecture(f'physical_head{mib}m_128x1_add2',128,1,2816,buffered_quant=True,
                       vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,
                       scalar_recip_lookup=True,fp_add_latency=2,vector_command_queue=2,
                       matrix_retirement=True,physical_scale_lines=True,pinned_cache_bytes=mib<<20,
                       act_rows=8,act_blocks=64,fifo_per_core=64) for mib in (24,28,32)],
        *[Architecture(f'physical_pinned{mib}m_128x1_add2',128,1,2816,buffered_quant=True,
                       vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,
                       scalar_recip_lookup=True,fp_add_latency=2,vector_command_queue=2,
                       matrix_retirement=True,physical_scale_lines=True,pinned_cache_bytes=mib<<20)
          for mib in (24,48)],
        Architecture('physical_128x1_add2',128,1,2816,buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,
                     scalar_recip_lookup=True,fp_add_latency=2,vector_command_queue=2,
                     matrix_retirement=True,physical_scale_lines=True),
        *[Architecture(f'proposal_pinned{mib}m_128x1_add2',128,1,2816,buffered_quant=True,
                       vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,
                       scalar_recip_lookup=True,fp_add_latency=2,pinned_cache_bytes=mib<<20) for mib in (24,48)],
        *[Architecture(f'proposal_geometry_{rows}x{cols}_add2',rows,cols,2816,buffered_quant=True,
                       vector_row_contexts=4,quant_scale_lookup=True,scalar_recip_lookup=True,
                       fp_add_latency=2) for rows,cols in ((128,1),(64,2),(32,4),(16,8))],
        *[Architecture(f'proposal_recip{lanes}_128x1_add2',128,1,2816,buffered_quant=True,
                       vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,
                       reciprocal_lookup_lanes=lanes,fp_add_latency=2) for lanes in (2,4,8,16)],
        Architecture('validated_32x1', 32, 1, 2816),
        Architecture('fpga_quant_32x1', 32, 1, 2816, buffered_quant=True),
        Architecture('fpga_quant_128x1', 128, 1, 2816, buffered_quant=True),
        Architecture('fpga_quant_reductions_128x1', 128, 1, 2816, buffered_quant=True,vector_row_contexts=4),
        Architecture('experimental_small_recip_32x1',32,1,2816,buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,scalar_recip_lookup=True),
        Architecture('experimental_small_recip_128x1',128,1,2816,buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,scalar_recip_lookup=True),
        Architecture('experimental_small_recip_128x1_add2',128,1,2816,buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,scalar_recip_lookup=True,fp_add_latency=2),
        Architecture('proposal_asic_small_recip_vector16_1024x1',1024,1,65536,buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,scalar_recip_lookup=True,
                     vector_engines=16,vector_result_chunks=1024,fp_mul_latency=1,fp_add_latency=1),
        Architecture('experimental_lookup_32x1',32,1,2816,buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True),
        Architecture('experimental_lookup_128x1',128,1,2816,buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True),
        Architecture('experimental_lookup_128x1_add2',128,1,2816,buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True,quant_scale_lookup=True,fp_add_latency=2),
        *[Architecture(f'proposal_window{w}_vector2_128x1',128,1,2816,buffered_quant=True,
                       vector_row_contexts=4,activation_streaming=True,vector_compatible_issue=True,
                       vector_engines=2,vector_result_chunks=1024,window=w) for w in (128,512)],
        *[Architecture(f'proposal_asic_window{w}_vector16_1024x1',1024,1,65536,buffered_quant=True,
                       vector_row_contexts=4,activation_streaming=True,fp_add_latency=1,fp_mul_latency=1,
                       vector_engines=16,vector_result_chunks=1024,vector_compatible_issue=True,
                       window=w) for w in (128,512)],
        Architecture('proposal_compatible_128x1',128,1,2816,buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True,vector_compatible_issue=True),
        Architecture('proposal_compatible_vector2_128x1',128,1,2816,buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True,vector_compatible_issue=True,
                     vector_engines=2,vector_result_chunks=1024),
        *[Architecture(f'proposal_vector{v}_128x1',128,1,2816,buffered_quant=True,
                       vector_row_contexts=4,activation_streaming=True,
                       vector_engines=v,vector_result_chunks=1024) for v in (1,2,4,8)],
        *[Architecture(f'proposal_vector{v}_128x1_add2',128,1,2816,buffered_quant=True,
                       vector_row_contexts=4,activation_streaming=True,fp_add_latency=2,
                       vector_engines=v,vector_result_chunks=1024) for v in (2,4)],
        Architecture('proposal_asic_vector16_1024x1',1024,1,65536,buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True,fp_add_latency=1,fp_mul_latency=1,
                     vector_engines=16,vector_result_chunks=1024),
        Architecture('experimental_fpga_stream_32x1', 32, 1, 2816, buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True),
        Architecture('experimental_fpga_stream_128x1', 128, 1, 2816, buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True),
        Architecture('experimental_fpga_stream_128x1_add2', 128, 1, 2816, buffered_quant=True,
                     vector_row_contexts=4,activation_streaming=True,fp_add_latency=2),
        Architecture('experimental_fpga_128x1_add2', 128, 1, 2816, buffered_quant=True,
                     vector_row_contexts=4,fp_add_latency=2),
        Architecture('experimental_128x1_add1_mul1', 128, 1, 2816, buffered_quant=True,
                     vector_row_contexts=4,fp_add_latency=1,fp_mul_latency=1),
        Architecture('asic_256x1_add1_mul1_memory8192', 256, 1, 8192, buffered_quant=True,
                     vector_row_contexts=4,fp_add_latency=1,fp_mul_latency=1),
        Architecture('asic_512x1_add1_mul1_memory16384', 512, 1, 16384, buffered_quant=True,
                     vector_row_contexts=4,fp_add_latency=1,fp_mul_latency=1),
        Architecture('asic_1024x1_add1_mul1_memory65536', 1024, 1, 65536, buffered_quant=True,
                     vector_row_contexts=4,fp_add_latency=1,fp_mul_latency=1),
        Architecture('fpga_quant_256x1', 256, 1, 2816, buffered_quant=True),
        Architecture('fpga_quant_64x4', 64, 4, 2816, buffered_quant=True),
        Architecture('rows128_memory2816', 128, 1, 2816),
        Architecture('rows512_memory2816', 512, 1, 2816),
        Architecture('rows128_cols4_memory2816', 128, 4, 2816),
        Architecture('rows128_cols8_memory2816', 128, 8, 2816),
        Architecture('rows128_cols8_memory8192', 128, 8, 8192),
        Architecture('rows256_cols1_memory8192', 256, 1, 8192),
        Architecture('rows256_cols4_memory8192', 256, 4, 8192),
        Architecture('rows256_cols8_memory8192', 256, 8, 8192),
        Architecture('rows512_cols8_memory8192', 512, 8, 8192),
        Architecture('rows512_cols8_memory16384', 512, 8, 16384),
        Architecture('rows512_cols8_memory16384_latency64', 512, 8, 16384, 64),
        Architecture('oversized_compute_memory_ceiling', 1024, 8, 65536, ceiling_only=True),
    ]


def resources(point, config, clock_mhz):
    # Lower bounds, not placement, routing, DSP-packing or SRAM-port proof.
    macs = config['D'] * point.rows * point.columns * point.matrix_engines
    if point.paired_dsp and (point.columns<2 or point.columns%2):raise ValueError('paired dot cells require even columns')
    dot_cells=macs//2 if point.paired_dsp else macs
    column_end_cells=9*point.rows*point.columns*point.matrix_engines
    matrix_cells=dot_cells+column_end_cells
    activation = point.matrix_engines * point.rows * (point.act_rows or config['ACT_ROWS']) * (point.act_blocks or config['ACT_BLOCKS']) * (config['D'] + 8)
    fifo = point.matrix_engines * point.rows * point.fifo_per_core * config['D']
    vector_results=point.vector_engines*point.vector_result_chunks*64*4
    # Two private input copies for each extra engine, propagated by every write.
    vector_read_copies=2*max(0,point.vector_engines-1)*4*config['TMEM_WORDS']
    # Default TMEM_IMPL=0 has eight read ports, with Q3 and COLL sharing
    # DMA's copy (otpu_slice SH_MASK): six complete data copies, not one.
    tmem_copies=6
    tmem_data=tmem_copies*4*config['TMEM_WORDS']
    instruction_data=4*config['IMEM_WORDS']
    quant_buffer_data=1024*64*4 if point.buffered_quant else 0
    lookup_bytes=(1<<21)+4096*51//8 if point.quant_scale_lookup else 0
    scalar_lookup_bytes=point.vector_engines*((1<<21)+4096*51//8) if point.scalar_recip_lookup else 0
    # Two independently addressed ports share each ROM instance and enable.
    wide_lookup_bytes=point.vector_engines*((point.reciprocal_lookup_lanes+1)//2)*((1<<21)+4096*51//8)
    cache_valid_bytes=point.pinned_cache_bytes//64//8
    scale_data=point.rows*point.scale_cache_entries*32*point.matrix_engines
    scale_control=0
    if point.scale_cache_entries:
        refs=point.latency.bit_length();index=(point.scale_cache_entries-1).bit_length()
        scale_control=point.matrix_engines*point.rows*((point.scale_cache_entries*(30+refs)+256+point.latency*(2+index)+7)//8)
    sram = activation + fifo + tmem_data+instruction_data+quant_buffer_data+vector_results+vector_read_copies+lookup_bytes+scalar_lookup_bytes+wide_lookup_bytes+point.pinned_cache_bytes+cache_valid_bytes+scale_data+scale_control
    limits = dict(matrix_mac_lanes=20000, onchip_bytes=64 << 20, memory_bytes_per_cycle=2816)
    # Board is an illustrative reference, not a user-selected target. Reserve
    # 20% of DSPs and half of embedded memory for shell/other units and mapping.
    # DSP packing is deliberately not assumed in the conservative screen.
    v80=dict(reference='AMD Alveo V80 product brief',
        source='https://www.amd.com/content/dam/amd/en/documents/products/accelerators/alveo/v80/alveo-v80-product-brief.pdf',
        dsp_slices=10848,embedded_memory_mbits=673,peak_hbm_gbs=810,
        matrix_dsp_fraction=.8,onchip_memory_fraction=.5,hbm_efficiency=.7,macs_per_dsp=1)
    v80_limits=dict(matrix_mac_lanes=int(v80['dsp_slices']*.8),
        onchip_bytes=int(673e6/8*.5),memory_gbs=810*.7)
    conservative=(matrix_cells<=v80_limits['matrix_mac_lanes'] and sram<=v80_limits['onchip_bytes'] and
                  point.bandwidth*clock_mhz/1000<=v80_limits['memory_gbs'])
    return dict(precise_tmem_scoreboard_rtl_implemented=False if point.precise_tmem else None,
        matrix_engines=point.matrix_engines,matrix_cluster_rtl_implemented=False if point.matrix_engines>1 else None,
        matrix_write_ports=1,matrix_write_service_scope='Aggregate bank service reserved at batch compute start; output-wave arbitration not validated.',
        matrix_mac_lanes=macs,integer_dot_dsp_cells=dot_cells,column_end_dsp_cells=column_end_cells,
        matrix_integer_dsp_cells=matrix_cells,matrix_fp32_multiply_sites=2*point.rows*max(1,point.columns//2)*point.matrix_engines,
        paired_dot_rtl_integrated=point.paired_dsp,paired_dot_technology='DSP48E2 component verified; DSP58 mapping unverified',
        scale_cache_data_bytes=scale_data,scale_cache_control_bytes=scale_control,
        scale_cache_prediction_scope='Canonical-order invalidation and intrinsic replacement pacing; asynchronous invalidations approximate' if point.scale_cache_entries else None,
 onchip_bytes_lower_bound=sram,
        pinned_cache_data_bytes=point.pinned_cache_bytes,pinned_cache_valid_bytes=cache_valid_bytes,
        pinned_cache_read_bytes_per_cycle=point.cache_read_bytes_per_cycle if point.pinned_cache_bytes else 0,
        pinned_cache_rtl_implemented=False,
        passes_v80_80pct_sram_capacity_screen=(matrix_cells<=v80_limits['matrix_mac_lanes'] and
            sram<=673e6/8*.8 and point.bandwidth*clock_mhz/1000<=v80_limits['memory_gbs']),
        quant_scale_lookup_bytes=lookup_bytes,scalar_reciprocal_lookup_bytes=scalar_lookup_bytes,
        wide_reciprocal_lookup_bytes=wide_lookup_bytes,wide_reciprocal_rtl_implemented=False,
        reciprocal_predictor_integer_multipliers=(int(point.quant_scale_lookup)+point.vector_engines*
            (int(point.scalar_recip_lookup)+point.reciprocal_lookup_lanes)),
        vector_engines=point.vector_engines,vector_result_bytes=vector_results,
        base_tmem_read_copies=tmem_copies,base_tmem_data_bytes=tmem_data,
        instruction_memory_data_bytes=instruction_data,quant_buffer_data_bytes=quant_buffer_data,
        storage_accounting_scope='Default replicated TMEM_IMPL=0 data included; bank granularity, multiwrite mapping, pipeline/control storage and Fmax unproven',
        additional_private_vector_read_copy_bytes=vector_read_copies,
        vector_fp32_multiply_sites=272*point.vector_engines*point.vector_issue_groups,
        vector_issue_groups=point.vector_issue_groups,
        vector_elementwise_issue_lanes=64*point.vector_issue_groups,
        vector_ordered_reduction_lanes=64,
        vector_tmem_read_words_per_cycle=128*point.vector_issue_groups,
        vector_tmem_write_words_per_cycle=64*point.vector_issue_groups,
        vector_scratchpad_scope='Wide proposal requires this aggregate bank throughput and two independent read copies; bank/control mapping and write arbitration are unvalidated' if point.vector_issue_groups>1 else None,
        wide_elementwise_rtl_implemented=False if point.vector_issue_groups>1 else None,
        vector_pool_implemented=False if point.vector_result_chunks else None,
        vector_pool_write_ports=1 if point.vector_result_chunks else None,
        vector_pool_resource_scope='Result data and private input copies included; VPU pipeline state, DSP mapping, address/control and fanout need synthesis.',
        memory_bytes_per_cycle=point.bandwidth, memory_gbs=point.bandwidth * clock_mhz / 1000,
        provisional_v80_reference=v80,provisional_v80_limits=v80_limits,
        passes_conservative_v80_capacity_screen=conservative,
        passes_two_macs_per_dsp_capacity_screen=(macs<=2*v80_limits['matrix_mac_lanes'] and
            sram<=v80_limits['onchip_bytes'] and point.bandwidth*clock_mhz/1000<=v80_limits['memory_gbs']),
        two_macs_per_dsp_status='Hypothetical all-dot packing screen only; use explicit integer cell counts. Column ends and FP sites do not halve.',
        illustrative_fpga_screen_limits=limits,
        passes_illustrative_fpga_screen=macs <= limits['matrix_mac_lanes'] and
            sram <= limits['onchip_bytes'] and point.bandwidth <= limits['memory_bytes_per_cycle'],
        scope='Explicit capacity/throughput assumptions; VPU, non-MXU DSPs, banking, routing and Fmax need implementation.')


def run_kernel(task):
    from examples.opentpu.workload import use_checkout
    from examples.opentpu.shape_workload import resolve_static
    from examples.opentpu.native_timing import ReusableTiming
    from examples.opentpu.model import Timing,reciprocal_lookup_width
    from examples.opentpu.e2e import atomic_json, digest
    use_checkout(Path(task['opentpu_root']))
    from opentpu import Config, isa as I
    point = Architecture(**task['architecture'])
    original = Config(**task['config'])
    directory = Path(task['directory'])
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(task['source'])
    record = task['record']
    for sid in range(original.S):
        if digest(source / f'prog_{sid}.hex') != record['input_sha256'][f'prog_{sid}.hex']:
            raise ValueError('retained canonical program changed')
    from .retained_inputs import verified_program
    programs=[];program_json_hashes={}
    for sid in range(original.S):
        encoded,identity=verified_program(source,sid)
        programs.append([I.Instr.decode(word) for word in encoded])
        program_json_hashes[f'program_{sid}.json']=identity
    start = time.perf_counter()
    workload = resolve_static(original, programs, record['args'])
    if point.precise_tmem:
        from .precise_tmem import refine
        workload=refine(workload)
    if point.act_rows or point.act_blocks:
        from .storage_envelope import command_bounds
        for items in workload.commands:
            bounds=command_bounds(items,I)
            if bounds['activation_rows']>(point.act_rows or original.ACT_ROWS) or bounds['activation_blocks']>(point.act_blocks or original.ACT_BLOCKS):
                raise ValueError('program exceeds proposed physical activation capacity')
    workload = replace(workload, config=replace(original, MCOLS=point.columns, LANES=64,
        ACT_ROWS=point.act_rows or original.ACT_ROWS,ACT_BLOCKS=point.act_blocks or original.ACT_BLOCKS))
    timing = Timing(window=point.window,batch_rows=point.rows,matrix_engines=point.matrix_engines, matrix_row_lanes=point.rows,
        dram_bytes_per_cycle=point.bandwidth, dram_port_bytes=point.bandwidth,
        dram_latency=point.latency, fifo_chunks=point.fifo_per_core * point.rows, scale_lanes=point.rows*point.matrix_engines,
        unified_dram=True, scalar_store_bytes=256, epilogue_lanes=max(1, point.columns // 2),
        quant_row_banks=8 if point.buffered_quant else 0,quant_store_buffered=point.buffered_quant,quant_scale_lookup=point.quant_scale_lookup,scalar_recip_lookup=point.scalar_recip_lookup,reciprocal_lookup_lanes=point.reciprocal_lookup_lanes,pinned_cache_bytes=point.pinned_cache_bytes,cache_read_bytes_per_cycle=point.cache_read_bytes_per_cycle,
        vector_row_contexts=point.vector_row_contexts,vector_issue_groups=point.vector_issue_groups,
        fp_mul_latency=point.fp_mul_latency,fp_add_latency=point.fp_add_latency,activation_streaming=point.activation_streaming,vector_engines=point.vector_engines,vector_result_chunks=point.vector_result_chunks,vector_compatible_issue=point.vector_compatible_issue,
        scale_cache_entries=point.scale_cache_entries,vector_command_queue=point.vector_command_queue,matrix_retirement=point.matrix_retirement,physical_scale_lines=point.physical_scale_lines)
    profile=None;profile_hash=None
    if point.pinned_cache_bytes:
        root=Path(task['memory_profile_root']);manifest=json.loads((root/'manifest.json').read_text())
        profile_path=root/str(point.pinned_cache_bytes)/f'{record["step"]:04d}.json'
        profile_hash=digest(profile_path);profile=json.loads(profile_path.read_text())
        if (manifest['status']!='complete' or manifest['reference_report_sha256']!=task['reference_report_sha256'] or
            manifest['records'][record['step']]['caches'][str(point.pinned_cache_bytes)]['profile_sha256']!=profile_hash or
            profile['kernel']!=record['step'] or profile['input_sha256']!=record['input_sha256'] or
            profile['program_json_sha256']!=program_json_hashes['program_0.json']):
            raise ValueError('cache prepass identity differs from retained workload')
    prepared = ReusableTiming(Path(task['compiler']), Path(task['cache']), capacity=65536).prepare(
        workload, timing, directory / 'timing.tables',prefetch_profile=profile)
    cycles, core, _, log = prepared.run()
    (directory / 'lpn.log').write_text(log)
    accelerated=sum(c.instruction.op==I.VOP and bool(reciprocal_lookup_width(c.instruction,I,timing))
                    for commands in workload.commands for c in commands)
    result = dict(prefetch_profile_sha256=profile_hash,accelerated_reciprocal_commands=accelerated,kernel=record['step'], phase=record['phase'], position_before=record['position_before'],
        cycles=cycles, original_rtl_cycles=record['baseline_rtl_cycles'],
        validated_evolved_rtl_cycles=record['rtl_cycles'],
        predicted_speedup_vs_original=record['baseline_rtl_cycles'] / cycles,
        native_seconds=core, host_seconds=time.perf_counter() - start,
        canonical_program_json_verified=True,program_json_sha256=program_json_hashes,
        zero_rollbacks=True, original_program_hash_verified=True, args=record['args'], timing=asdict(timing))
    if point.name == 'validated_32x1':
        result['control_relative_error'] = cycles / record['rtl_cycles'] - 1
        if abs(result['control_relative_error']) > .10:
            raise ValueError('validated control misses the accepted 10% cycle target')
    atomic_json(directory / 'record.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('case', 'opentpu-root', 'compiler', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cache',type=Path,help='Reuse compiled timing architectures across studies')
    parser.add_argument('--memory-profile-root',type=Path,help='Verified cold/coherent pinned-region prepass profiles')
    parser.add_argument('--indices', default='0,15,16,47',
        help='Comma-separated kernel indices or all; the default is a representative pilot')
    parser.add_argument('--architectures', help='Comma-separated architecture names; default all')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--clock-mhz', type=float, default=200,
        help='Assumed common clock, not synthesized Fmax')
    args = parser.parse_args()
    from examples.opentpu.e2e import atomic_json, digest
    if args.workers < 1 or args.clock_mhz <= 0:
        parser.error('workers and clock must be positive')
    case = args.case.resolve()
    reference = json.loads((case / 'report.json').read_text())
    if (reference['status'] != 'complete' or reference['prompt_tokens'] != 128 or
        reference['output_tokens'] != 32 or reference['entrypoint'] != 'unmodified upstream Engine.generate' or
        not reference['all_dram_and_tmem_bit_exact'] or not reference['all_inputs_identical_to_baseline']):
        raise ValueError('requires the completed, numerically validated normal 128/32 RTL case')
    indices = list(range(48)) if args.indices == 'all' else sorted(set(map(int, args.indices.split(','))))
    if not indices or not set(indices) <= set(range(48)):
        parser.error('indices must select kernels in [0,47]')
    points = architectures()
    if not args.architectures and args.memory_profile_root is None:points=[p for p in points if not p.pinned_cache_bytes]
    if args.architectures:
        wanted = set(args.architectures.split(','))
        points = [p for p in points if p.name in wanted]
        if {p.name for p in points} != wanted:
            parser.error('unknown architecture name')
    if any(p.pinned_cache_bytes for p in points) and args.memory_profile_root is None:
        parser.error('pinned-cache candidates require --memory-profile-root')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parent
    files = [project / name for name in ('tpu_dse.py', 'model.py', 'compact.py', 'components.py',
                                       'workload.py', 'shape_workload.py', 'native_timing.py', 'benchmark.py','activation_stream.py','vector_pool.py','vector_select.py','retained_inputs.py','storage_envelope.py','precise_tmem.py','scale_cache_timing.py')]
    if args.memory_profile_root:files.append(args.memory_profile_root.resolve()/'manifest.json')
    sources = {str(p): digest(p) for p in files}
    observations = {p.name: [] for p in points}
    failures = []
    started = time.perf_counter()

    def save(status):
        reports = []
        for point in points:
            records = sorted(observations[point.name], key=lambda r: r['kernel'])
            item = dict(architecture=asdict(point), resources=resources(point, reference['config'], args.clock_mhz),
                        kernels=records, completed_kernels=len(records))
            if len(records) == len(indices):
                item['selected_kernel_speedup'] = sum(r['original_rtl_cycles'] for r in records) / sum(r['cycles'] for r in records)
                if indices == list(range(48)):
                    item['full_inference_cycles'] = sum(r['cycles'] for r in records)
                    item['predicted_full_speedup'] = reference['baseline_rtl_cycles'] / item['full_inference_cycles']
                elif indices == [0, 15, 16, 47]:
                    by_index = {r['kernel']: r['cycles'] for r in records}
                    estimate = 15 * by_index[0] + by_index[15] + 31 * by_index[16] + by_index[47]
                    item['pilot_extrapolated_inference_cycles'] = estimate
                    item['pilot_extrapolated_speedup'] = reference['baseline_rtl_cycles'] / estimate
                    item['extrapolation_scope'] = 'Candidate selection only; intermediate attention positions not measured.'
            reports.append(item)
        atomic_json(output / 'report.json', dict(status=status, pid=os.getpid(), scope=__doc__,
            reference_report_sha256=digest(case / 'report.json'), source_sha256=sources,
            project_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip(),
            config=reference['config'],reference_cycles=reference['baseline_rtl_cycles'], target_cycles=reference['baseline_rtl_cycles'] / 100,
            indices=indices, clock_mhz=args.clock_mhz, seconds=time.perf_counter() - started,
            numerical_arithmetic_executed=False, modified_rtl_validated=False, points=reports, failures=failures))

    save('running')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {}
        for point in points:
            for index in indices:
                record = reference['kernels'][index]
                task = dict(architecture=asdict(point), config=reference['config'], record=record,
                    source=str(case / 'kernels' / f'{index:04d}'),
                    directory=str(output / point.name / f'{index:04d}'),
                    opentpu_root=str(args.opentpu_root.resolve()), compiler=str(args.compiler.resolve()),
                    cache=str(args.cache.resolve() if args.cache else output / 'native-cache'),
                    memory_profile_root=str(args.memory_profile_root.resolve()) if args.memory_profile_root else None,
                    reference_report_sha256=digest(case/'report.json'))
                pending[pool.submit(run_kernel, task)] = (point.name, index)
        for future in as_completed(pending):
            point, index = pending[future]
            try:
                result = future.result()
                observations[point].append(result)
                print(json.dumps(dict(architecture=point, **result)), flush=True)
            except Exception as error:
                failures.append(dict(architecture=point, kernel=index, error=repr(error)))
                print(json.dumps(failures[-1]), flush=True)
            save('running')
    if sources != {str(p): digest(p) for p in files}:
        raise RuntimeError('model sources changed during the sweep')
    save('complete' if not failures else 'failed')
    if failures:
        raise RuntimeError('some architecture/kernel predictions failed')


if __name__ == '__main__':
    main()
