# Exact reciprocal through a compact correction ROM

The standalone RTL primitive reproduces the original openTPU reciprocal
sequence with four enabled clocks. This is a component result: it has not yet
been connected to QACT/VPU, measured on inference, or synthesized for an FPGA.

A 4096-entry table holds a fixed-point piecewise linear approximation of the
normalized significand. One signed 14x12 multiply, integer add/rounding, and a
signed two-bit correction restore the exact original Newton result. The
correction is indexed by all 23 input fraction bits. This preserves the ISA's
specific sequence of separately rounded float32 operations, rather than
substituting an ideal reciprocal with different low bits.

The correction table is 2,097,152 packed bytes; coarse coefficients contain
208,896 bits before physical RAM width rounding. Two independently addressed
read ports share the tables. The primitive has one small multiplier per port.
Its current clock enable freezes both ports together. Separate copies or
independent enables will be needed if QACT and VPU advance differently.
The tables are constants of the arithmetic implementation, independent of
model weights and input values. Their generation, hashes and resource size
must be retained with a build. Initializing their RAM is a hardware setup cost.

Validation against the **original SystemVerilog arithmetic function** passed:

- All **8,388,608** normalized fraction inputs, permuted across both ports.
- **1,048,576** full reciprocal inputs: every exponent/sign across 128 selected
  mantissas, including zero/denormals/infinities/NaNs, then shuffled full words.
- Clock-enable holds and caller tag alignment.

The complete normalized table's required correction is only −1, 0 or +1.
The wrapper preserves the original `fp_recip` exponent, sign and FTZ rules.
This does not exhaust all 2^32 full words; normalized coverage is exhaustive,
while exponent/sign handling has an edge grid plus randomized coverage.
The tests do not prove FPGA RAM inference, placement, frequency, or an end-to-end
cycle reduction. Those remain required before promoting it as an architecture.

Generate tables and reproduce the gates from the LPN root:

```sh
PYTHONPATH=. python -m examples.opentpu.rtl_evolution.make_recip_rom \
  --output /tmp/recip-tables

PYTHONPATH=. python -m examples.opentpu.rtl_evolution.check_recip_rom \
  --opentpu-root /path/to/FeSens-openTPU-act-streaming \
  --table /tmp/recip-tables --output /tmp/recip-normalized

PYTHONPATH=. python -m examples.opentpu.rtl_evolution.check_recip_rom \
  --opentpu-root /path/to/FeSens-openTPU-act-streaming \
  --table /tmp/recip-tables --output /tmp/recip-full \
  --full-reciprocal --limit 1048576
```

Generated tables are not committed: the generator regenerates them in seconds.
Reports, source/table hashes and the original normalized reference digest are
in [validation/recip-rom](rtl_evolution/validation/recip-rom). The RTL module is
[otpu_recip_rom.sv](rtl_evolution/otpu_recip_rom.sv).
