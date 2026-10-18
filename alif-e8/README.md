# DC-Ops on Alif E8

The current firmware is a [live English→Spanish sign demo](translation/live/README.md).
It runs text detection on HP U55 and recognition on HE U55, then overlays a
phrase-table translation for 60 common signs while the camera remains live.
The optional full neural translator runs through ExecuTorch on U85. See the
[checked-in validation reports](verification/README.md) for measured results,
numerical comparisons, and limitations. Build tools still use this Mac's SDK paths.

The original DC-Ops object detector port is documented below.

DC-Ops' trained YOLOv8n-seg v3, exported as a bounding-box detector for the
Ethos-U85. The Cortex-M55 captures a real OV5675 camera image, letterboxes it,
executes INT8 inference, decodes YOLO's distributional box regression, applies
class-aware NMS, and draws the live image, boxes, class labels, and confidence
on the 480×800 ILI9806E LCD. Segmentation masks are omitted.

Hardware: Ensemble AppKit-E8 AE822FA0E5597LS0, M55-HP, U85-256, OV5675
camera (chip ID 0x5675 at I²C 0x10), scaled by the ISP to 320×240 planar RGB.
The display uses an RGB565 copy. Model input: 320×320 INT8 with NHWC storage
(NCHW logical dimensions and channels-last strides), with 40-pixel
vertical letterbox margins. Confidence threshold 0.35; NMS IoU threshold 0.45.

## Build and run on this Mac

1. Export: `.venv/bin/python export_detector.py > artifacts/export.log 2>&1`.
2. Build: `./build.sh`.
3. Flash and start: `./flash.sh`.
4. Inspect status: `.venv/bin/python inspect_board.py`.
5. Capture the LCD and camera: `.venv/bin/python inspect_board.py --images`.

The scripts use the existing CMSIS-Toolbox 2.15.0, GCC 15.3.1, Alif Ensemble
2.2.1, and PyTorch ExecuTorch 1.5.1 CMSIS pack. The export environment imports
the matching Python SDK from `~/alif/CMSIS-Executorch/.venv`, and installs
Ultralytics separately. `build.sh` contains the tool paths for this machine.

The board's power-on boot configuration starts M55-HP at **0x80200000**. The
linker uses that address and reserves the existing ATOC/SE data at the top of
MRAM. It does not change Secure Enclave power profiles. Model and firmware
remain in MRAM; DMA-visible camera, display, NPU scratch, and planned tensor
buffers are in SRAM0. The linker also reserves DTCM stack/heap space.

The imported MNIST board layer had the wrong camera reset GPIO (14/4).
This AppKit uses GPIO9/1. Its LCD reset is GPIO13/6, rather than the
DevKit LCD reset at GPIO15/5. Board pin/GPIO definitions come from the
previously working AppKit camera project in `/private/tmp/alif-mlek-yolo`. Its earlier
firmware used OV5675; the first attempts using MT9M114 did not find a sensor.
The ISP uses RAW10 input, planar RGB output, and routes CPI directly into ISP
without storing a full-resolution raw frame. The LCD is initialized after
the camera because both configure MIPI clock hardware.

UART4 logs are available at 115200 baud if SW4 is set to UART4. Inspection over
J-Link does not require changing SW4: firmware keeps a diagnostic ring buffer
in DTCM. Inspecting temporarily halts M55-HP and resumes it afterward; an image
capture can miss its timeout while stopped.

## Artifacts and evidence

`artifacts/export_metadata.json` records the checkpoint hash, exact calibration
image list and hashes, quantization scales, output shapes, model hash, and
planned buffers. Calibration uses 24 images from the same Hugging Face dataset;
these are not a held-out accuracy evaluation.

`artifacts/vela/output/out_summary_Ethos_U85_SRAM_MRAM.csv` contains compiler
memory and performance estimates. Compiler timing is not measured device timing.
`artifacts/board.log` and `artifacts/board_status.json` contain the latest hardware
inspection. `artifacts/lcd.png` is a readback of the board's actual LCD framebuffer.

The original entire 5.5 MiB MRAM contents were saved before flashing as
`artifacts/e8_mram_before.bin`. This backup includes board-specific boot data;
keep it on this machine.

Checkpoint: [dc_ops_yolov8n_seg_v3.pt](https://huggingface.co/datasets/abhijitbetigeri/dc-ops-dataset/blob/main/models/dc_ops_yolov8n_seg_v3.pt).

This port derives its board startup, linker, display initialization, and CMSIS
solution from the existing `~/alif/CMSIS-Executorch-MNIST` project. The original
project is preserved. Generated model arrays, build outputs, and device dumps
are excluded by `.gitignore`.

The CMSIS build context retains the inherited `DevKit-E8` name. Its pin,
GPIO, and board definitions are the AppKit-E8 definitions for the attached
board. `flash.sh` derives the reset entry from this build's ELF, rather than
using the debugger's default entry for the older image at 0x80008000.

## Verified on the E8

The 2,983,120-byte model compiles into one Ethos-U85 delegate with no CPU
operators. It reserves 475,200 bytes of planned tensors and Vela reports
722,464 bytes of NPU scratch. The firmware reserves 1 MiB for scratch and
512 KiB for planned tensors. SRAM0 static allocation is 3,800,064 bytes;
firmware/model MRAM allocation is about 3.17 MB. Inference measured 19 ms
using the M55 cycle counter at 400 MHz; this excludes camera/display work.

`artifacts/label_test.json` records a known-image hardware test: the E8
reported a compute tray at 65.7% confidence and drew a matching green box
and class/confidence label in its actual LCD framebuffer. The test image was
injected through the debugger for one frame; the firmware then resumed the
physical camera. No test image is built into the application.

`artifacts/hardware_validation.json` compares all 168,000 INT8 output elements
from one frozen E8 inference against the host quantized graph using the same
24 audited calibration images. This checks numeric agreement and execution,
not held-out detection accuracy. Input storage must be NHWC: directly copying
NCHW bytes to the NPU produces incorrect results despite successful execution.

The LCD uses two RGB565 framebuffers. Rendering only touches the hidden buffer;
`CDC200_FRAMEBUF_UPDATE_VSYNC` switches frames at vertical blanking. The old
buffer is reused after two scanline interrupts, with a bounded timeout. Camera
frames are rotated 180 degrees before inference and display, so model input,
video, and detection boxes share upright coordinates. Labels stay upright.
`inspect_board.py` reports display presents, vertical blanking events, and timeouts.
