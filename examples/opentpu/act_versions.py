"""Optimistic ACT-renaming screen, with unlimited retained physical versions.

This is not a bounded FPGA design or a rigorous runtime bound. Every QACT gets
fresh storage; readers keep the latest preceding version of each row/block,
including its scale/offset metadata. Real implementations need allocation,
reader tracking, ports and finite backing banks. Original instructions and
non-ACT hazards are preserved. Streaming-producer analysis must run first on
the original commands, since it describes the original logical ACT addresses.
"""
from dataclasses import replace

from .precise_tmem import merged


def renamed_commands(commands, cfg, I, W):
    pages = cfg.act_rows * cfg.ACT_BLOCKS
    latest = [0] * pages
    result = []
    for index, command in enumerate(commands):
        old = command.footprint
        ins, w = command.instruction, command.instruction.w
        rd = [r for r in old.rd if r[0] != W.SP_ACT]
        wr = [r for r in old.wr if r[0] != W.SP_ACT]
        accesses = any(r[0] == W.SP_ACT for r in old.rd + old.wr)
        if accesses and ins.op not in (I.MM, I.QACT):
            raise ValueError('unimplemented ACT accessor in versioning screen')
        if accesses:
            if ins.op == I.MM:
                rows = ((w[5] >> 16) & 255) * (2 if ins.flags & I.F_PAIR else 1)
                block, chunks = w[5] >> 24, (w[3] >> 16) & 255
            else:
                rows = (w[1] & 255) * (2 if ins.flags & I.F_DUP else 1)
                block, chunks = (w[1] >> 8) & 255, w[1] >> 16
            if not (0 < rows <= cfg.act_rows and 0 <= block < block + chunks <= cfg.ACT_BLOCKS):
                raise ValueError('ACT versioning access exceeds logical storage')
            for row in range(rows):
                for column in range(block, block + chunks):
                    page = row * cfg.ACT_BLOCKS + column
                    if ins.op == I.QACT:
                        latest[page] = index + 1
                    address = latest[page] * pages + page
                    (wr if ins.op == I.QACT else rd).append((W.SP_ACT, address, address + 1))
        result.append(replace(command, footprint=replace(old, rd=merged(rd), wr=merged(wr))))
    return result
