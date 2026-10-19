# Integrated vendor DSP48E2 matrix-engine gate

The paired **two-row, two-column** matrix engine passes all eleven gates
using `MXU_PAIR_TECH=1`, with the official pinned DSP48E2 simulation model.
Nine primitive cases and two six-command queue cases match behavioral RTL
in original inputs, complete DRAM/TMEM outputs and cycle counts. Fresh
audits independently rerun the ISA and native LPN timing. Worst timing
error is **7.7922%**, within 10% on every case.

Cases include int8 and packed FP4, partial engine groups, activation replay,
accumulation, activation scaling, row maxima, strided quantized stores,
overlapping matrix admissions and WAW ordering. The separate exhaustive
component proof covers all 16,777,216 signed-int8 triples and the exceptional
eight-term positive-overflow case; this integration gate does not replace it.

`c2.json` links the two fresh audit hashes. `vendor-audit.json` and
`behavioral-audit.json` retain source, binary, input, raw-log and complete
output identities. Frozen driver revision is **6197f20**; local subsequent
changes improve path handling and failure recording only.

The vendor library remains external at revision
`1c8e05fd1e9a79ceb8b996a0996674122eed086f`. As in the component proof, exactly
168 floating-input default tests are specialized for explicitly driven
binary pins; arithmetic/register logic is unchanged. An isolated harness
adds the official `glbl` startup reset, a 1ns/1ps clock timescale, and waits
for startup before releasing the measured core. Frozen RTL is unchanged.
The initial two integration attempts exposed missing global scope and an
inherited picosecond clock; their failed evidence is retained remotely.

This closes the **small integrated DSP48E2 simulation** gap. It does not
prove DSP58 mapping, the full inference with vendor primitives, physical
placement, total FPGA resources or Fmax. The full 15.4406x inference result
continues to use behavioral packed arithmetic validated separately.

```sh
python -m examples.opentpu.rtl_evolution.check_vendor_mxu \
  --opentpu-root ../FeSens-openTPU-packed-mxu \
  --unisim-root ../XilinxUnisimLibrary \
  --compiler build/tools/lpn-compile/lpn-compile \
  --rows 2 --columns 2 --output /tmp/vendor-mxu
```
