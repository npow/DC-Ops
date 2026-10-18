# Hardware verification

Verified on the attached AppKit-E8 with its OV5675 camera and ILI9806E LCD.
The user confirmed the DC-Ops live feed is visible on the physical screen.

- 550+ real camera frames processed without an inference or LCD update error.
- Ethos-U85 inference: 19 ms, excluding camera, decoding, and display work.
- A known repository image produced a compute-tray detection at 65.7% confidence;
  the actual LCD framebuffer contained its box, class label, and confidence.
- All 168,000 INT8 output values from that inference were compared with the
  matching host quantized graph. Mean absolute differences per output head
  were below 1.5 quantization steps. Exact values and hashes are saved in
  `artifacts/hardware_validation.json` and `artifacts/verification.json`.
- Firmware was left running the physical camera; no test image is embedded.

The model is the trained v3 checkpoint at 320×320, with the segmentation branch
removed. This is functional and numerical verification, not held-out accuracy
evaluation. Full source and build/flash instructions are in README.md.

The board is AppKit-E8: LCD reset is GPIO13/6 and camera reset GPIO9/1.
Its power-on application address is 0x80200000. The inherited CMSIS context
name remains DevKit-E8; its GPIO and pin definitions are for AppKit-E8.

Display correction: the original renderer cleared the actively scanned RGB888
framebuffer, causing visible blinking reported by the user. The replacement uses
two RGB565 buffers and switches at vertical blanking, reusing the old buffer only
after two scanline events. Camera frames are rotated 180° before inference and
display; model input and box coordinates share that upright orientation. The updated firmware built and was flashed successfully.
Hardware display counters are saved in `artifacts/display_validation.json`;
the user confirmed: “Upright and blinking is gone.”
