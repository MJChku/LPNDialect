# Diagnostic: explicit VPU command acceptance FIFO

RTL `otpu_vpu` has two command entries. The sequencer may accept a command into
that FIFO while an older command is issuing or a long operation is still live.
The FIFO head then waits for issue availability, reduction isolation and
pipeline tap compatibility. Acceptance and execution are distinct stages.

The earlier LPN selected the oldest dependency-ready command only when execution
could begin. That can choose an older instruction that became ready after
younger commands had already entered the RTL FIFO, changing order and timing.

`Timing(vector_command_queue=2)` models the FIFO explicitly. Acceptance removes
the ready command from its window slot, reserves one of two queue entries and
advances at most once per cycle. The command reaches the FIFO head a cycle
later. Execution consumes that head, tests the existing rank/reduction guards
and returns its queue credit a cycle later. Window credit and completion still
return only after execution finishes. Scheduler, guard synthesis and heartbeat
behavior are unchanged. The default remains zero until broader timing validation.
This diagnostic currently excludes the proposed vector pool and compatible
selection, which would need their own acceptance queues.

The regression test makes an older command become ready after independent
younger commands were accepted. It verifies FIFO order, the difference from
selection at execution, and native/Python agreement. All 30 existing integration
tests also pass.

## Real-workload diagnostic

The 128-engine shorter-adder RTL remains numerically bit-exact, but queue
modeling explains only part of its timing discrepancy:

| Kernel | RTL cycles | Earlier LPN | FIFO LPN | FIFO error |
| --- | ---: | ---: | ---: | ---: |
| 16 | 109,349 | 94,640 | 95,626 | −12.550% |
| 47 | 109,618 | 94,940 | 95,926 | −12.491% |

These results still fail the accepted 10% tolerance. No constant calibration
factor has been applied, and the configuration is not timing-validated.

Kernel 16 was rerun with trace enabled using the identical executable and
canonical inputs. Complete DRAM/TMEM output comparisons remain bit-exact and
cycles remain 109,349. The observer reproduces the FIFO native result exactly.
All 5,777 dispatched commands align by dynamic index, PC and opcode. The first
VPU acceptance-order difference occurs at position 35; upstream timing already
differs by then, so that difference alone does not identify its cause.

The trace also shows an embedding segment where matrix completions occur at
405 then 420 cycles, whereas FIFO LPN predicts 400 then 404. This points to
matrix drain/retirement overlap as the next diagnostic. Start-to-end durations
alone cannot establish unit latency because RTL accepts some commands into
queues much earlier than the LPN admits them.

Reports and source/trace hashes are in
[validation/vpu-queue](rtl_evolution/validation/vpu-queue). Raw traces remain at
`~/lpn-runs/tpu-100x/small-recip-128-add2-decode-trace` and
`~/lpn-runs/tpu-100x/timeline-vpu-queue2-16` on `icdslab2`.

To compare a checked RTL rerun and an LPN observer result:

```sh
PYTHONPATH=. python -m examples.opentpu.compare_timeline \
  --case /path/to/evolved-pipeline-final --kernel 16 \
  --rtl-dir /path/to/traced-rerun --lpn-dir /path/to/lpn-observer \
  --opentpu-root /path/to/FeSens-openTPU --output /tmp/comparison.json
```

The RTL directory must retain its launch record, canonical inputs, raw log and
complete output dumps. The comparison verifies executable/input/trace identity
and numerical memory states before using timing events.
