"""Physical parameters shared by GLM instruction-level component gates."""
from .model import Timing


def component_timing(rows=2,columns=2,bandwidth=2816,latency=8):
    if columns not in (1,2,4,8):raise ValueError('unsupported physical MXU column count')
    # otpu_mxu's default E is max(1, physical MCOLS/2), not Timing's generic
    # two-lane default. In particular MXU_COLS=2 has ONE FP epilogue lane.
    return Timing(batch_rows=rows,matrix_row_lanes=rows,fifo_chunks=128*rows,
        scale_lanes=rows,chain_length=8,epilogue_lanes=max(1,columns//2),
        dram_bytes_per_cycle=bandwidth,dram_port_bytes=bandwidth,dram_latency=latency,
        unified_dram=True,scalar_store_bytes=256,quant_row_banks=8,quant_store_buffered=True,
        vector_row_contexts=4,vector_command_queue=2,matrix_retirement=True,physical_scale_lines=True)
