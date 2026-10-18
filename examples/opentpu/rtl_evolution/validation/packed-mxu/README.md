# Paired integer DSPs in the actual MXU

`packed-mxu.patch` applies after `multi-column.patch` and its preceding twelve
patches on FeSens revision f237dab2f196855a59a083b1454786df8f01291f.
`patch-proof.json` verifies all 30 simulated source/harness files against a
clean application. Earlier source roots and measurements are unchanged.

Enable `MXU_PAIR_DOT=1` with `MXU_COLS=2/4/8` and parallel rows. Each pair of
activation columns shares one decoded weight stream and one packed multiply
per chain position. Chains have at most eight terms. The low-field overflow
correction and exact integer extraction are the independently
[exhaustively verified paired component](../paired-dsp/README.md).

The packed weight register changes the even column latency by one cycle.
Explicit activation delays and column-end metadata/deskew compensation keep
the final output latency unchanged. Odd columns retain their previous
latency. The parallel wrapper already clears ISA PAIR and consumes packed
weight halves serially; the packed core explicitly rejects a shared-weight
violation. Scale column ends, floating epilogues, SRAM and control are not
halved. The default remains unpacked.

All **33** gates (nine primitive and two six-command queued cases at each
column count) pass. A fresh audit checks source identities, simulator cache
keys and hashes, raw cycle/instruction logs, canonical input equality,
independent ISA execution and complete DRAM/scratchpad outputs. Every output
and cycle count matches the corresponding unpacked design, including partial
groups, activation replay, int8/FP4, ACC, ASCALE, RMAX, WAW and epoch wrap.

The audit also reruns native LPN timing. One **137-cycle** eight-column 5x3
case is predicted at 123 cycles (**−10.21898%**), just outside the 10% target.
All two-column cases meet 10%; their worst error is 7.9470%. This synthetic
short-command timing limitation is recorded, not hidden or fitted away.
It affects both packed and unpacked RTL, which have identical cycle counts.
This is not a full-inference timing validation.

`MXU_PAIR_TECH=0` is the behavioral datapath tested here.
`MXU_PAIR_TECH=1` explicitly instantiates the one-DSP48E2 component whose
arithmetic was separately verified against the pinned AMD vendor library.
The integrated MXU has not yet been run with that vendor primitive. DSP58
mapping for the provisional V80, actual DSP totals, placement and Fmax remain
unproven. This change halves the integer dot-product cell count for an even
column count; it does not halve whole-design DSP use or establish 100x.

```sh
python -m examples.opentpu.rtl_evolution.check_matrix_pipeline \
  --opentpu-root ../FeSens-openTPU-packed-mxu --rows 8 --columns 2 \
  --paired-dsp --output /tmp/paired-c2
```
