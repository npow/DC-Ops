# E8 live translator

The default is a fast English→Spanish **sign demo**: OCR runs on HP/HE U55,
then a small phrase table translates supported signs without autoregressive
decoding. Unknown text shows `No matching sign` with the recognized English.
Set `ocr_sign_mode=0` for the experimental full U85 translator described below.
This lookup mode does not provide arbitrary sentence translation.

Start the debugger-loaded application, restore its model assets, repair any mismatching blocks, and enable live sign overlays:

```sh
translation/.venv/bin/python translation/live/start.py --build
```

Before the first build, generate the mixed driver with
`python3 translation/multi_npu/build_driver.py` and the model assets with the
[staged export workflow](../staging/README.md). The `pack_pipeline.py` command
below generates the required `pipeline_index.h`. Generated weights, driver
archives, environments, and build outputs are excluded from Git.

The screen uses an upright, full-screen camera view with automatic OCR roughly twice per second. Supported signs receive a Spanish text overlay with a transparent background; the camera continues moving without capture presses or display pauses. Green bounds show detected text regions only; the center-stripe fallback has no outline. Set `live_show_ocr_regions=0` to hide detection bounds too. The current detector uses a 128×224 input. Only one selected text region is recognized per pass; multi-line reading order is not implemented.

Use `--button-capture` for optional still captures. Any joystick direction or
center press captures the next frame; another press returns to live preview.
Hardware press latches catch short taps. The reset button restarts the board.
Button captures deliberately bypass automatic stability confirmation. The
image remains frozen through detection, recognition, translation, and viewing
the result. Overlay/status updates redraw that same captured image without
polling or restarting the camera. An empty reading explicitly shows `No text
read`; it does not silently keep scanning.

The optional full-neural mode (`ocr_sign_mode=0`) uses stability confirmation. Automatic
translation requires matching OCR text at an overlapping position for at least
one second across two or more readings. Empty, changed, or moved text resets
that confirmation. During translation, the captured preview is held; afterward
the camera resumes. Successful exact-source translations are reused, so an
unchanged sign does not repeatedly pause the display. This is a frozen-preview
mode, not a text-only screen.

HP U55 runs text detection and HE U55 runs recognition. HG U85 is used only by full-neural translation; the fast sign demo performs its lookup on the CPU. The overlay reports measured stage timings and NPU activity. `speedup n/a` means a CPU baseline has not been measured; it is not an estimate. OCR and translation accuracy remain experimental.

Physical E8 results are collected in `artifacts/translator/validation_summary.json`.
The frozen eight-phrase device run finished 8/8 and matched executable TOSA
tokens 8/8; it matched FP32 baseline tokens 7/8. One integer-lowering wording
difference was “Pulse el botón verde” versus “Presione el botón verde”.
The separate diagnostic inputs `T`, `Hello`, and `STOP` all finished and matched
the pretrained baseline. Baseline parity is not an accuracy claim: this tiny
checkpoint mistranslates uppercase `STOP` as `ESTNDAR` on the host too. No
case normalization or dictionary substitution is enabled.

The same-firmware display comparison measured `Hello` at 3.488 s with live
rendering versus 0.759 s with a held preview; `STOP` took 8.507 s versus
1.914 s. These are translator-only times. OCR is additional; automatic mode also adds steady-text
confirmation. The reference encoder-layer benchmark measured
FP32 CPU execution at 24.971 s versus A16W8 U85 at 80.086 ms (311.8×), or
38.4× including that benchmark's staging/loading. That comparison predates
the display-pause optimization and is one layer, not full-pipeline speedup.

The live model is the 12-stage spatial-FFN pack in `spatial_staging`. Its
17,056,896 bytes fit PSRAM; every stage has one U85 delegate, with CPU boundary
quantization/dequantization. Maximum per-stage PTE/planned/scratch/fast-scratch
sizes are 2,047,504 / 82,048 / 32,832 / 147,456 bytes, respectively, within the
2.5 MiB / 512 KiB / 512 KiB / 384 KiB allocations. Existing E8 hardware has
executed these stages; no simulator license or additional hardware is required.

Factory boot configuration is unchanged. A power cycle starts the factory application and can overwrite external RAM. Run the start command again to restore and verify the translator's assets.

The loader uses a persistent J-Link SDK connection and 32-bit writes. `repair_assets.py` checks every 256-byte block against host CRC32 values, repairs mismatches, and rescans before startup. Boundary samples alone are insufficient: the first full check found sparse single-bit errors inside model data. Source SHA256 values and per-block verification reports are stored under `artifacts/translator/asset_repair`.

A transfer failure while the firmware is still idle before inference allows one
guarded warm restart and resume. This reinitializes the RAM protocol without
reflashing or changing factory boot settings. Full CRC verification remains
mandatory before enabling inference. Known working firmware backups are in
`artifacts/translator/camera_backups`.

Manual/benchmark modes reserve the shared inference buffers. Camera frame preprocessing returns immediately in those modes. Null NPU addresses remain null, and translation checks that its required fast scratch buffer is enabled before execution. Live display servicing remains active.

Inspect without resetting or halting the MCU:

```sh
python3 translation/live/device.py status
```

Reproduce the corrected model exports and pack:

```sh
translation/.venv/bin/python translation/staging/export_staged.py --spatial-ffn --output artifacts/translator/spatial_staging
python3 translation/staging/pack_pipeline.py --directory artifacts/translator/spatial_staging
```

Run the frozen board evaluation while retaining the camera firmware:

```sh
translation/.venv/bin/python translation/live/evaluate_translation.py --pause-ocr --pause-display 1
```

Use `--pause-display 0 --group diagnostic` for the continuous-video comparison.
These inject diagnostic text through the manual service and do not claim OCR
or live camera text evidence. Results include exact token IDs, EOS, errors,
NPU counters, display status, frame advancement, and pause durations.
