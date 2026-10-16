# Weight-free FPGA architecture exploration

`fpga_dse.py` compiles Qwen3-1.7B dimensions into real upstream openTPU
instructions, resolves their static scalar loops/register addresses, and feeds
the command footprints into the existing event-driven LPN. It loads no
checkpoint, tokenizer, DRAM image or numerical tensor. Matrix, quantizer and
vector instructions contribute timing and dependencies without executing their
arithmetic. This is different from `replay_e2e --timing-only`, which still
executes the numerical model and only skips RTL input-file validation.

The workload includes all 28 transformer layers, attention, KV-cache traffic,
MLP and vocabulary head. Sequence length is prescribed: this mode predicts
timing, not generated text, logits or EOS. Prefill remains token-by-token.
Data-dependent control such as RLD/WAITW is rejected; it requires supplied
control metadata or actual functional execution, not invented readiness.

```mermaid
flowchart LR
  S[Model dimensions and context length] --> C[Upstream instruction compiler]
  C --> R[Static control and address resolution]
  R --> W[LPN command window and dependency guards]
  W --> M[Matrix engines with parallel output-row lanes]
  W --> V[Vector / quantizer / DMA]
  H[Shared memory bandwidth credit] <--> M
  H <--> V
  M --> T[Banked TMEM and completion tokens]
  V --> T
  T -. dependency readiness .-> W
```

## Reproduce

Build LPN using the root README, and install NumPy in the environment for the
pinned FeSens checkout described in [README.md](README.md). Verilator and model
weights are not required for this experiment. Run from the repository root:

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../FeSens-openTPU/.venv/bin/python -m examples.opentpu.fpga_dse \
  --opentpu-root ../FeSens-openTPU \
  --compiler build/tools/lpn-compile/lpn-compile \
  --matrix-columns 1 --output /tmp/qwen17-fpga-dse
```

The default first sweeps a complete forward step at position 15, then compares
the fastest screened candidate with the original architecture for a fixed
16-input/8-output sequence (24 steps). `--matrix-columns 4` retains the original
stationary width in evolved candidates; the original baseline always keeps
four columns. Narrower arrays replay multi-row operands across more groups,
and can spend the saved compute resources on more parallel output rows. The
instruction stream and logical memory layout remain unchanged.

Use `--position 1023 --skip-sequence` for a forward step at a 1,024-token
context. CLI options expose assumed clock, memory latency, HBM efficiency,
resource reserves and MACs per DSP. Every report separates hardware cycles,
assumed-clock hardware seconds and simulator wall time. Reports include source
and compiler hashes; output directories should be separate for each study.

The original 128-token-capacity configuration checks its program hashes and
predicted cycles against the checked-in RTL-validated Qwen baseline. The
16-input/8-output study checks every original-architecture step. Evolved
architecture timing is a prediction; it has not been validated against evolved
RTL. Longer-context original-architecture predictions also extend beyond the
recorded 128-token-capacity validation campaign.

## Provisional hardware envelope

The reference is AMD's
[Alveo V80 product brief](https://www.amd.com/content/dam/amd/en/documents/products/accelerators/alveo/v80/alveo-v80-product-brief.pdf):
32 GB HBM2e, 10,848 DSP slices, 673 Mb embedded memory and 810 GB/s peak
bandwidth. The model uses a common **assumed 200 MHz** clock, **64-cycle**
memory latency, at most **70% of peak HBM bandwidth**, **80% of DSPs** for
the MXU screening estimate, and **50% of embedded memory** for the modeled
storage lower bound. These are explicit planning assumptions, not measured
board behavior or a selected bitstream. One INT8 MAC lane per DSP is assumed;
`--macs-per-dsp` makes that assumption visible rather than treating MAC lanes
as an exact mapped DSP count.

The resource screen charges replicated activation storage, matrix FIFOs,
accumulation pipelines, TMEM capacity and modeled MAC lanes. Passing this screen
does not establish total DSP/LUT use, shell availability, routing, banking
overhead, power or achievable Fmax. Those require synthesis and implementation.
HBM bank conflicts and controller behavior are not modeled; effective bandwidth
and latency should be calibrated later.

## What changes in the LPN

Each matrix engine remains an independently synchronized component. Parallel
output-row lanes replicate its datapath; compute time rounds partial groups
up. Wider memory service shares one credit across matrix prefetch and DMA/QST
traffic. Scale bytes are charged to that bus, and readiness waits for both
weight transfer and scale decoding. No weights are assumed permanently cached
on chip. Banked output service counts writes to each actual modulo-addressed
TMEM bank, including strided collisions and partial batches.

The command-prefetch credit returns only after both the data transfer and scale
decoder finish. Returning it after data alone lets a later unscaled command
overtake an older scaled batch, potentially deadlocking the engine's ready FIFO.
A 1,024-context/four-engine regression reproduced that failure before the fix
and completes with zero rollbacks afterward.

Vector/quantizer width changes `Config.LANES`. Packed QST stores, scale decoding
width, FIFO capacity and physical stationary columns are explicit architecture
choices. Multiple engines preserve inter-engine RAW/WAR/WAW dependencies.
The existing single-engine timing defaults, guards, heartbeat cache and
sequential event-driven scheduler are retained. Eighteen integration tests
passed, including original recorded RTL fixtures, static versus numerical
footprints, rejected dynamic control, bank collisions and expanded/native
agreement for the new datapaths.

## Measured LPN predictions

These are hardware-cycle reductions at the same assumed clock, compared with
the original LPN architecture. They are not measured FPGA or simulator speedups.
Short context means one complete forward step at position 15; long context means
position 1023. The original short-context program/cycles match the saved baseline.

| Physical array | Parallel output rows | Modeled MAC lanes | Assumed memory GB/s | Short-context cycle reduction | 1,024-context cycle reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 x 4 (original) | 1 | 512 | 25.6 | 1.00x | 1.00x |
| 128 x 1 | 8 | 1,024 | 204.8 | 6.03x | 4.67x |
| 128 x 1 | 16 | 2,048 | 409.6 | 9.47x | 6.37x |
| 128 x 1 | 32 | 4,096 | 563.2 | 11.02x | 6.99x |
| 128 x 1 | 64 | 8,192 | 563.2 | 11.04x | 7.00x |

For **16 input + 8 output tokens**, the original architecture takes
**328,851,748 cycles**. The fastest narrow-array candidate takes
**29,675,852 cycles: 11.08x fewer**. The four-column candidate takes
34,627,828 cycles (9.50x fewer). Every one of the 24 original-architecture
steps reproduces the saved program hash and predicted cycles. This checks the
weight-free frontend against the validated model; it does not validate new RTL.

A practical first candidate is **32 output-row lanes of 128 x 1**, with a
64-lane vector/quantizer datapath and 563.2 GB/s effective memory bandwidth.
It uses 4,096 modeled MXU MAC lanes and is only 0.206% slower than doubling that MAC
budget to 8,192 lanes. The fixed clock, effective HBM throughput and all mapped
resource assumptions must be tested in synthesis/implementation. The current
256-byte/cycle packed QST path and scale decode width must also be implemented.

Keeping the original memory bandwidth while adding compute yields little gain.
Keeping the original compute while widening memory also yields little gain.
The balanced points change output-row parallelism, vector/quantizer width,
FIFO storage and memory bandwidth together. With longer context, attention and
other shared/control costs reduce the benefit to about 7x in this study.

[Four-column sweep and full sequence](validation/fpga_dse/qwen17_v80_4cols.json),
[narrow-column sweep and full sequence](validation/fpga_dse/qwen17_v80_1col.json),
[1,024-context sweep](validation/fpga_dse/qwen17_v80_context1024.json),
[test log](validation/fpga_dse/tests.txt), and
[artifact summary](validation/fpga_dse/summary.json) preserve the measurements.

The evolved 24-step simulation took about
56.8 s of preparation/compile/native-process time, plus
5.9 s of shape compilation, with other study processes running. Those
host times exclude the architecture sweep, baseline comparisons and Python
startup. No matrix arithmetic or weight loading occurred.

## Larger models

No weight arrays are necessary to model capacity or timing, but address width
still matters. The current trace compiler rejects a whole model image above
4 GiB per slice: the existing ISA uses 32-bit addresses. A future 27B design
needs layer/weight streaming through address windows, multiple slices, or a
wider/banked address interface, plus explicit transfers in the LPN.

Raw weights alone need about 27 GB at INT8 or 13.5 GB at 4-bit. Quantization
metadata, KV capacity, scratch space and controller/shell reservations must be
added before claiming that the model fits 32 GB. The first implementation here
uses the already validated 1.7B model dimensions; it does not claim a 27B run.
