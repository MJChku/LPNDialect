# Packed quantizer stores

`packed-qst.patch` applies after `scale-cache.patch` and the preceding nine
patches. Set `QST_PACKED_WORDS=1` on the RTL top to enable it. The default
retains the old port behavior. A clean application matches all 29 compiled
source/harness files; hashes are in `patch-proof.json`.

The quantizer presents one byte-enabled word port per output byte. For
consecutive bytes, the new combiner forms actual word payloads and byte masks,
merging up to four ports into one transaction. The arbiter charges four bytes
per merged word, and the behavioral memory receives those merged ports and
data. Thus this is an RTL data-path change, not only a different byte budget.

Packed detection checks wanted addresses and masks before grant. Partial and
unaligned words preserve all byte enables. Scatter and unsupported layouts
retain the original ports; duplicate-address stores retain their old priority.
Grant bubbles and partially accepted inputs preserve byte results. The design
uses no persistent cache or additional FP datapath. FPGA logic/timing mapping
has not been measured.

`unit.json` covers 720 combinations of alignment, tails, mask holes, request
subsets, grant bubbles, stride 1/4/17, duplicate addresses and unsupported byte
masks. Every resulting memory byte matches. The packed full-width case emits
16 word transactions for 64 aligned bytes rather than 64 transactions.
`matrix.json` retains nine bit-exact accelerator gates with both this combiner
and the scale cache enabled, including packed weights, ACC/RMAX and scatter
stores. Their lightly loaded cycle counts are unchanged.

`pressure.json` separately compares identical input files on a synthetic
matrix/QST workload at a shared 256-byte-per-clock link. Both complete DRAM
and scratchpad outputs match the independent ISA model; the packed variant
uses **1.0943x fewer cycles**. This is a contention gate, not a model inference
or a measured benefit at the 2,816-byte-per-clock FPGA-study budget. The LPN
equations have not yet been extended/calibrated for this combiner.

```sh
python -m examples.opentpu.rtl_evolution.check_qst_coalesce \
  --opentpu-root ../FeSens-openTPU-packed-qst --output /tmp/pack-unit
python -m examples.opentpu.rtl_evolution.check_matrix \
  --opentpu-root ../FeSens-openTPU-packed-qst --rows 8 \
  --scale-cache-entries 16 --packed-qst --output /tmp/pack-matrix
python -m examples.opentpu.rtl_evolution.check_qst_pressure \
  --opentpu-root ../FeSens-openTPU-packed-qst --output /tmp/pack-pressure
```
