# ACT INT8 latency measurement on Alif E8

Checkpoint: `ofcourseistillloveyou/pick_up_red_lid_ac_50`, pinned in `source.json`.
The original two 480x640 camera inputs and 50x6 output are retained.

Measured on E8: **20,900.5 ms median** over four completed measured predictions
(`20900, 20901, 20900, 20901` ms), after one 20,901 ms warm-up. All completed
chunks passed finite-output checks. USB disconnected during the fifth measured
repeat, before final action values and stage timings could be read. The durable
evidence is `board_partial_report.json` and
`board_fixed_latency_allocator_corrected.log`; no completed five-run report is
claimed. Original app restoration is pending reconnection/hardware handoff.

The source checkpoint's training-only VAE encoder is removed. One shared
ResNet-18 program runs once per camera, then the action transformer runs.
Checkpoint tensors load with strict key/shape validation. Exported floating
program outputs are checked against their source modules. The LeRobot
transformer components are copied unchanged; provenance is in `provenance.json`.
The standard ResNet-18 implementation uses the checkpoint's frozen BN statistics.

## Scope

This is a latency smoke test, not a robot success-rate or quantization-quality
evaluation. Calibration uses one synthetic sample. The two camera passes use
the same synthetic normalized float image. This preserves the original tensor
shapes and computation count while requiring only one host image fixture.
There are no motor-control interfaces in the firmware.

Timed: two camera encoder executions, feature copies, transformer action output,
and float/INT8 boundary conversion. Excluded: camera acquisition, image/state
normalization before the network, model transfer/loading, action unnormalization,
and physical execution.

## Build and run

Run from this directory with Python 3.12 and the versions in `requirements.txt`.
CMSIS-Toolbox 2.15.0, GCC 15.3.1, the configured CMSIS packs, and J-Link
9.86a were used. Model weights and generated binaries are intentionally omitted
from Git; `fetch_checkpoint.py` retrieves the pinned weights and reference source
with SHA256 verification.

```sh
python fetch_checkpoint.py
python prepare.py
python verify_fp32.py
python compile_ethosu.py vision
python compile_ethosu.py head
bash build_firmware.sh
python -u run_board.py --flash --upload
```

The runner takes an exclusive advisory lock on
`/tmp/alif-e8-1219599493.lock`, backs up the current HP MRAM, and uses
SRAM-staged transfers with CPU CRC32 verification of each block and entire file.
It verifies the device millisecond timer against host elapsed time, excludes
one warm-up, and targets five measured runs in `board_report.json`.
The recorded run lost USB during the fifth repeat; four completed measured runs
are preserved separately in `board_partial_report.json`.

Workstation defaults are overrideable with `VCPKG_ARTIFACTS_ROOT` for the build,
`ARM_GCC_BIN` for host symbol tools, `CMSIS_PACK_ROOT`, and `JLINK_LIBRARY`.
The hardware runner is deliberately tied to probe serial `1219599493`; edit its
serial and lock together when using another board.

The PSRAM interface is initialized at **25 MHz**. The current candidate programs
fixed read latency on both x16 PSRAM dies (MR0 bit 5), uses eight XIP read wait
cycles, and disables variable-latency RXDS handling. This configuration passed
the other session's three-convolution probe and post-NPU full PTE integrity
check. This is also the setup used for the completed ACT predictions above. A reset-pin recovery
in a separate process was required before the first ACT transfer passed.

The corrected firmware copies the command stream into the existing internal
SRAM staging pool before NPU invocation. External AXI requests are limited to
one outstanding read/write and 64-byte bursts. The loader uses 15 MHz SWD while
the firmware is idle, then switches to 1 MHz for inference monitoring.
Command copying is included in the timed inference scope.
NPU memory attributes match Alif's official normal-memory settings. Each NPU
invocation has an actual 60-second timeout; a timeout is a failed run, not a
latency result. Runtime method metadata has a separate internal 512 KiB pool
per model, while explicit hierarchical allocators keep planned tensors external.

`prepare.py` regenerates the FP32 exports and deterministic synthetic input.
`compile_ethosu.py` regenerates quantized exports, PTE programs, and Vela
intermediates. The committed artifact manifest records hashes of the actual
measured files; a rebuilt ELF can differ due to embedded source paths.

## Memory layout

All offsets are within 64 MiB PSRAM at `0xA0000000`:

| Offset | Purpose | Reserved |
|---|---|---:|
| 0 MiB | Vision PTE | 12 MiB |
| 12 MiB | Transformer PTE | 24 MiB |
| 36 MiB | Shared image input | 4 MiB |
| 40 MiB | Two camera feature buffers | 2 MiB |
| 42 MiB | Vision planned tensors | 5 MiB |
| 47 MiB | Transformer planned tensors | 5 MiB |
| 52 MiB | Shared temporary/NPU scratch | 12 MiB |

Compiler-reported peak scratch is 9,830,400 bytes for vision and 7,405,264
bytes for the transformer. Internal SRAM also supplies 384 KiB fast scratch
and a 1 MiB staging pool, plus the two 512 KiB method metadata pools.

The runner saves a local MRAM backup before its first flash. The original
application is this repository's `alif-e8` live OCR/translation demo. Restore
that application after the measurement, coordinating access to the shared board. Restoring
MRAM alone is insufficient: the original app also needs its six PSRAM assets,
full block CRC verification, and a healthy camera/pipeline check. The
`restore_original.py` workflow handles this; `--resume-assets` avoids reflashing
when the original firmware is already running and waiting for assets.

`restore_original.py` defaults to the enclosing `alif-e8` directory; set
`ALIF_APP_ROOT` for another checkout. Its original ELF and six generated assets
must already exist. Restoration evidence in this commit predates the final
ACT run; restoration after the USB disconnect remains pending.
