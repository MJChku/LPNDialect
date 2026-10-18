# FPGA-first architecture exploration

The objective is 100x fewer hardware cycles on the **same upstream SmolLM2-135M
128-input/32-output inference**, compared with the original 132,828,263-cycle
RTL run. **100x has not been achieved.** The latest audited full-run result is
[12.7717x with 128 shared-bank engines](FPGA_REDUCTIONS.md). Earlier controls
are 8.605x and the 8.662x buffered-quantizer follow-up measured below. A
successful capacity screen is not FPGA synthesis, placement, routing or Fmax proof.

LPN consumes the original compiled commands. Upstream `Engine.generate()` owns
prefill batching, tiling, decode and EOS. Independent HALT-drained kernels may
be validated in separate host processes, but their hardware cycles are summed.
The event-driven runtime, automatic/manual guard synthesis and heartbeat cache
are unchanged.

## Measurements so far

The first native LPN geometry pilot varied row/column parallelism and memory
bandwidth. Its best oversized point projects only 12.43x, despite 524,288 matrix
MAC lanes and 65,536 bytes/cycle. That is a pilot extrapolation from four
kernels, not an implemented design or a formal performance ceiling. Larger
arrays alone do not address the remaining quantization/vector dependencies.

The FPGA-focused follow-up uses a bounded, eight-row quantizer and buffered
QST stores. A complete 48-kernel native prediction for **128 single-column
matrix cores** is **12.592x**. It uses 8,192 matrix MAC lanes and 2,816 memory
bytes/cycle; the prediction is 10,548,397 cycles. It is not a new RTL validation.
Its unchanged 32-core control predicts 15,712,791 cycles, reproducing the
previously corrected full inference timing.

The capacity reference is AMD's [Alveo V80 product brief](https://www.amd.com/content/dam/amd/en/documents/products/accelerators/alveo/v80/alveo-v80-product-brief.pdf):
10,848 DSP slices, 673 Mb embedded memory and 810 GB/s peak HBM. The screen
reserves 20% of DSPs and half of embedded memory, assumes one matrix MAC per
DSP, 70% HBM efficiency and a common 200 MHz clock. The resulting limits are
8,678 matrix MACs, 42,062,500 bytes of modeled on-chip storage and 567 GB/s.
The 128-core candidate passes those capacity estimates. It still requires
physical implementation, including all vector/quantizer resources, SRAM port
replication, shell, HBM integration and routing. The board is a planning
reference; the user has not selected a board.

The earlier geometry pilot used a looser illustrative screen of 20,000 MACs
and 64 MiB. That screen is retained in its report for provenance and must not
be described as a V80 fit. The new tool reports the conservative V80 screen
separately. A two-MAC-per-DSP screen is also shown as an explicitly unimplemented
packing possibility; the current RTL has no such packed datapath.

## Implemented quantizer change

Apply [buffered-quant.patch](rtl_evolution/buffered-quant.patch) **after** the
existing [parallel-rtl.patch](rtl_evolution/parallel-rtl.patch), in a separate
checkout at FeSens `f237dab2f196855a59a083b1454786df8f01291f`.
Preserve the original and audited 8.605x checkouts.

The existing prescaled row FIFO now serves full-width QST as well as QACT.
Eight tagged row-state banks allow rows to overlap through the existing scale
pipeline. QST scale records and data use their distinct existing store links,
while the shared byte-budget arbiter still limits aggregate memory traffic.
HALF, narrow stores and rows larger than the FIFO retain the two-pass path.
No arithmetic or software instruction changes are involved.

The production FIFO remains 1,024 chunks. At 64 lanes this is the existing
256 KiB buffer. Increasing four group-state banks to eight adds 15.5 KiB of
group RAM for GR=1,024, plus small scalar row-state registers. These are raw
capacity changes, not a complete mapped FPGA resource report.

The LPN opts in with `Timing(quant_row_banks=8,
quant_store_buffered=True, scalar_store_bytes=256)` and physical `LANES=64`.
The defaults preserve the earlier validated timing. `buffered_quant_cycles`
computes read/pop scheduling from finite FIFO and row-bank credits, the actual
pipeline depth and scale-row store count. It produces command metadata rather
than one simulator event per clock.

The numerical gate compares complete DRAM, activation bytes and activation
scales with independent ISA execution after every command. It covers 28 cases:
row/column scaling, DUP, HALF fallback, unaligned/strided stores, nonfinite
values, ties, row-bank wrap and rows exceeding the small FIFO. Both a 32-chunk
FIFO under stalls and the production FIFO pass. All 26 LPN integration tests
pass, including native/expanded agreement and unchanged earlier RTL fixtures.

With the **same 1,024-chunk FIFO** in both unit-level RTL builds:

| Command | Earlier RTL cycles | Buffered/eight-bank RTL cycles |
| --- | ---: | ---: |
| QST, eight rows of 64 elements | 378 | 88 |
| QST, eight rows of 576 elements | 522 | 176 |
| QACT, eight rows of 64 elements | 104 | 68 |
| QACT, eight rows of 576 elements | 160 | 140 |

The full 48-kernel quantizer-only RTL validation completed in **15,335,479
cycles**, versus the original 132,828,263: **8.6615x fewer hardware cycles**.
Every kernel has identical input files and bit-exact complete DRAM/scratchpad
outputs. This is only a 0.655% cycle speedup over the previous 32-engine RTL;
the unit gains above do not translate directly to full inference.

LPN predicts 15,236,007 cycles: **-0.649% total error**, with **2.958% worst
kernel error**, inside the accepted 10% target for every kernel. This remains a
32-engine control. The later [shared-bank implementation](FPGA_REDUCTIONS.md)
removes the fixed two-lanes-per-engine limitation and is now validated on all
48 kernels at 128 engines.

The [completed report](rtl_evolution/validation/tpu-100x/quant-full-rtl.json) and
[fresh audit](rtl_evolution/validation/tpu-100x/quant-full-audit.json) retain the
evidence. Raw data is at `~/lpn-runs/tpu-100x/quant-normal128x32` on `icdslab2`.

## Remaining dependency work

[critical_path.py](critical_path.py) traces the original address hazards and a
separately labeled unlimited-renaming scenario. It ignores unit contention and
command-window capacity, assumes matrix weights are already available, and uses
the modeled command latencies. It is neither an LPN prediction nor a formal
physical limit. On representative kernels, the 128-core candidate has about
129k–140k cycles of resource-free prefill dependencies and 45.5k cycles of decode
dependencies. Unlimited version storage reduces prefill to about 104k–106k;
decode remains about 45k. Thus adding execution units or removing false memory
dependencies alone does not establish 100x.

The next FPGA work needs to address reduction/normalization and quantizer
pipeline handoffs, while retaining ordered arithmetic, bounded storage and real
memory ports. For example, interleaving independent rows through the existing
reduction adders can hide their feedback latency without duplicating the entire
VPU. Any such change needs its own RTL numerical gates and full inference
measurement. Weight caching must also account for cold loads, host updates and
KV writes; the present results assume no persistent weight cache.

## Reproduce

From the LPN project root, with the pinned FeSens Python environment and native
compiler described in [README.md](README.md):

```sh
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m examples.opentpu.tpu_dse \
  --case /path/to/evolved-pipeline-final \
  --opentpu-root /path/to/FeSens-openTPU-fpga-quant \
  --compiler build/tools/lpn-compile/lpn-compile \
  --architectures validated_32x1,fpga_quant_128x1 --indices all \
  --workers 8 --output /tmp/fpga-full-prediction

python -m examples.opentpu.rtl_evolution.check_quant_buffered \
  --opentpu-root /path/to/FeSens-openTPU-fpga-quant \
  --chunks 1024 --output /tmp/quant-production-gate

python -m examples.opentpu.rtl_evolution.replay_architecture \
  --case /path/to/evolved-pipeline-final \
  --baseline /path/to/baseline-smol128x32-continued/report.json \
  --opentpu-root /path/to/FeSens-openTPU-fpga-quant \
  --compiler build/tools/lpn-compile/lpn-compile \
  --indices all --workers 12 --output /tmp/quant-normal128x32
```

`replay_architecture` uses retained original programs, numerical inputs and
independent ISA golden states. It runs fresh modified RTL and checks every
device-memory byte and scratchpad word. It avoids repeating the tensor prepass.
Use a new output directory. Source hashes are frozen throughout each run;
inspect its report, worker JSON files and owning PID before starting another.

Reports and source/binary hashes are in
[validation/tpu-100x](rtl_evolution/validation/tpu-100x). Raw server data is under
`~/lpn-runs/tpu-100x`. Pilot reports, full predictions and actual RTL results
are distinct artifacts.
