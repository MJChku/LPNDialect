# Exact paired integer DSP component

`otpu_pair_pe.sv` tests two signed-int8 product streams sharing one operand.
The technology branch explicitly instantiates **one DSP48E2**, using its
pre-adder, multiplier and P cascade. This follows the documented
[27-bit pre-adder and 27x18 multiplier](https://docs.amd.com/api/khub/documents/pTysoma4TYgNH95BrY1Sbw/content).
The component supports cascades of at most eight terms, matching the current
D=64 MXU's eight-term groups. Column-end scale multipliers and FP epilogues
are separate and are not packed by this component.

For streams `x`, `y` and shared `a`, the multiplier sees
`(x + (y << 18)) * a`. Over a chain, let `S0=sum(x*a)` and `S1=sum(y*a)`.
The DSP cascade holds `S0 + (S1 << 18)`. For up to eight signed-int8 terms,
`-130048 <= S0 <= 131072`. A negative low sum borrows from the high sum;
the extraction adds that borrow back.

The single exceptional low-field value is **+131072**, which does not fit
signed 18 bits. It requires eight products of `-128 * -128`. A one-bit
all-maximum flag travels beside the cascade; together with the low sign bit,
it distinguishes this positive overflow from a negative sum and corrects
both extracted results. The arithmetic retains every integer bit.

`quick.json` compares both behavioral and vendor-primitive implementations
against two original scalar PEs, cycle for cycle. It checks 65,536 single-PE
clocks, 100,079 eight-position chain clocks, 14,300 enable stalls and the
positive overflow corner. The exhaustive 16,777,216 signed-int8 input-triple
gate is running separately. This component has not been integrated into the
MXU and establishes no inference speedup or FPGA-fit/Fmax result.

The vendor simulation library is the official
[Xilinx Unisim repository](https://github.com/Xilinx/XilinxUnisimLibrary), pinned
at `1c8e05fd1e9a79ceb8b996a0996674122eed086f`. Its Apache-2.0 source remains
an external dependency. Verilator rejects its floating-input `Z` default
checks on tied pins. The driver generates a hashed simulation-only copy
specializing exactly 168 such tests for this gate's explicitly driven binary
inputs. Arithmetic, registers and timing statements are unchanged; original
and derived hashes are retained. This is not a four-state floating-pin test.

```sh
git clone https://github.com/Xilinx/XilinxUnisimLibrary ../XilinxUnisimLibrary
git -C ../XilinxUnisimLibrary checkout 1c8e05fd1e9a79ceb8b996a0996674122eed086f
python -m examples.opentpu.rtl_evolution.check_pair_pe \
  --opentpu-root ../FeSens-openTPU-scalar-recip \
  --unisim-root ../XilinxUnisimLibrary --exhaustive --output /tmp/paired-dsp
```

The same algebra fits the documented DSP58 operand widths, but that device
mapping is untested. More fundamentally, independent asynchronous row engines
do not always share an operand at the same clock. Integrating this component
needs an appropriate array schedule or paired columns; counting all matrix
MACs as two-per-DSP before that integration would be unjustified.
