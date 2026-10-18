The camera module extracts the confirmed working OV5675/ISP capture sequence into a polling API. It does not invoke inference or touch the LCD.

Add `translation/camera.cpp` and `translation/camera_preprocess.cpp` to the final application alongside the display renderer. Exclude the original detector `src/app_main.cpp`, which owns another set of camera buffers and another `app_main`.

```cpp
// After board pin/GPIO/clock initialization and enabling DWT CYCCNT:
int result = dcops::camera_init();
if (result == 0) result = dcops::camera_start();

// Run regularly on the camera/display owner thread:
switch (dcops::camera_poll()) {
case dcops::CaptureState::Ready: {
    auto image = dcops::camera_image();  // Already corrected 180 degrees.
    auto planes = dcops::camera_planes();
    // If scheduling a new OCR job, copy/preprocess its input NOW, before start.
    // dcops::detector_input(planes, detector_float_buffer, 3*480*288);
    // dcops::recognizer_input(planes, source_region, recognizer_float_buffer, 3*48*320);
    result = dcops::camera_start(); // ISP captures next image while LCD scans out.
    if (result == 0)
        result = translator_display_frame(&image, actual_boxes, actual_count, stats);
    break;
}
case dcops::CaptureState::Pending:
    // Service other bounded tasks; do not block this thread in token generation.
    break;
case dcops::CaptureState::Error:
    // Log failure and stop/recover explicitly. No automatic retry loop.
    break;
}
```

Ownership is singleton and explicit: only the camera/display owner thread calls the capture API. `camera_image()` returns a borrowed RGB565 frame valid until the next successful `camera_poll()` conversion; starting the next capture leaves this RGB565 buffer intact because the required configuration disables CPI AXI DMA and sends data through ISP only. Do not let another task call `camera_poll()` while displaying the borrowed image. `camera_planes()` returns the borrowed 8-bit planar R/G/B DMA buffer, valid only until `camera_start()` or `camera_stop()`. Preprocess/copy that input before restarting capture; a worker must not retain this view. Other threads receive separately owned model inputs and immutable translation snapshots. The module does not itself create an RTOS thread or make a blocking model loop concurrent.

Default capture dimensions come from `RTE_ISP_OUTPUT_WIDTH/HEIGHT`: currently **320x240**, with the exact existing camera configuration retained. For the intended native **288x480** portrait crop, compile `camera.cpp` with `DCOPS_CAMERA_WIDTH=640` and `DCOPS_CAMERA_HEIGHT=480`. This allocates larger buffers and calls the vendor `ISP_CONTROL_SET_OUTPUT` before the first capture. Vendor Ensemble 2.2.1 supports this output-size operation while stopped; its configured OV5675 input is 1296x972 and the ISP scaler is enabled. This path has been compiled, but 640x480 capture still requires board validation. No existing RTE settings were edited.

Memory (32-byte aligned in existing bulk-SRAM sections):

| Capture | RGB565 `.bss.camera_frame_buf` | RGB888P `.bss.camera_frame_bayer_to_rgb_buf` | Total |
|---|---:|---:|---:|
| 320x240 | 153,600 bytes | 230,400 bytes | 384,000 bytes |
| 640x480 | 614,400 bytes | 921,600 bytes | 1,536,000 bytes |

Driver state adds 136 BSS bytes. No heap is used. Caller-owned detector float input needs **1,658,880 bytes** (3x480x288); recognizer input needs **184,320 bytes** (3x48x320). These are additional to the capture and 1,536,000-byte LCD buffers, so place model inputs in an appropriate separately validated memory pool. Display/NPU/camera together must fit the linker map.

Preprocessing preserves full 8-bit ISP color data, rather than requantizing RGB565. It reverses the camera orientation during sampling and emits NCHW **BGR**: detector ImageNet mean/std `[.485,.456,.406]/[.229,.224,.225]`, recognizer `pixel/127.5-1`. Detector output matches the exact display center crop (320x240 => x88/y0/144x240; 640x480 => x176/y0/288x480), resized to 288x480. `detector_box_to_source` maps detector pixel coordinates to the upright source pixels expected by the overlay renderer. Recognizer uses an axis-aligned source crop with aspect-preserving resize to height48, maximum width320, and normalized-space zero right padding. Return value is valid width: CTC must ignore padded timesteps using `ceil(logit_steps*valid_width/320)`.

Resizing is bilinear with half-pixel sampling. The existing Python recognition smoke tests use Pillow's default resize filter, so camera preprocessing is not claimed bitwise identical to those synthetic fixtures. Validate recognition quality on captured text and align the offline preprocessing filter if necessary. Axis-aligned crops do not rectify perspective/skew.

`camera_poll` returns immediately while capture is pending, checks driver error events, and aborts after three DWT seconds. Frame completion performs the existing ISP frame-end/dequeue/stop operations and RGB565 conversion; it does not contain an event wait loop. DWT must remain enabled and clocked while capturing, with polling frequent enough to observe the three-second timeout before its 32-bit counter wraps.

Validation: host Clang C++17 with ASan/UBSan and `-Wall -Wextra -Werror` passes source/display geometry, BGR normalization, 180-degree correction, recognizer valid-width/padding, invalid bounds, and output guard bytes. Cortex-M55 GCC15.3.1 object compilation passes for camera, preprocessing, and the 640x480 variant with the translator project's real generated include paths. No camera hardware was accessed or flashed.

```sh
clang++ -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined -g \
  translation/display.cpp translation/camera_preprocess.cpp translation/camera_test.cpp \
  -o /tmp/dcops-camera-test
/tmp/dcops-camera-test
```
