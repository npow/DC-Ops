# OCR postprocessing

Compile `postprocess.cpp` and `characters.cpp` into the camera application. They allocate no heap memory and require no ML runtime. All output derives from supplied detector probabilities and recognizer logits; no canned phrase fallback is used.

```cpp
dcops::DetectionWorkspace workspace;  // Prefer static storage: 33,664 bytes.
dcops::DetectedBox boxes[4];
size_t count = 0;
auto result = dcops::detect_boxes(detector_output, 288, 480,
                                 boxes, 4, count, workspace);
```

The detector uses 8-connected foreground components, default pixel threshold 0.3, component mean score threshold 0.5, and minimum 12 foreground pixels. It expands shrunken DB regions by 0.35 times their height (minimum 2 pixels), clamps to map boundaries, and retains the largest qualifying regions. Coordinates remain in detector pixels. This is an axis-aligned initial implementation; it does not rectify rotated text or implement full DB polygon unclip.

For each box, call the existing `detector_box_to_source(image, box.x, box.y, box.x+box.w, box.y+box.h, nullptr)`. Convert the returned source endpoints to a bounded integer `Viewport` using floor for the starting coordinate and ceil for the endpoint. The existing `recognizer_input(image, region, input, elements)` produces NCHW `[1,3,48,320]` and returns `valid_width`. Pass this exact valid width to the CTC decoder:

```cpp
char recognized[256];
float confidence;
auto status = dcops::decodeCTC(recognizer_output, 40, 97, valid_width,
                              recognized, sizeof(recognized), &confidence);
```

CTC reads only `ceil(timesteps*valid_width/320)` rows, removes repeated consecutive classes, removes blank class 0, and trims outside spaces. An intervening blank allows a repeated letter (as in `Hello`). The dictionary exactly preserves the converted checkpoint's 95 `en_dict.txt` entries, plus the blank and appended space; the source dictionary itself also contains a space. Output is ASCII, and therefore valid UTF-8. Confidence is the mean softmax probability of emitted character classes; it is zero for an empty result. Invalid/nonfinite logits and undersized output buffers return an error with an empty output string.

Use one `TextStability` object for each tracked region. `observe(text, detector_box, now_ms)` requests translation after two matching readings within 5 seconds and box IoU at least 0.3. `source()` returns the pending source text. `setTranslation(text, now_ms)` atomically commits the text and box for display. `visible(now_ms)`, `translation()`, and `box()` expose the committed overlay for up to 5 seconds after it was last observed. A new pending recognition retains the previous committed overlay until the replacement translation is ready or its TTL expires. Track objects need to be associated with regions by the caller; do not associate them solely by largest-area rank when multiple regions move.

Workspace has a bit per possible detector pixel and a bounded flood-fill queue. A pathological foreground topology can exhaust the queue; this returns `WorkspaceExceeded` and zero boxes rather than returning partially processed geometry. Normal dense/full-frame and small-text components are covered by tests. Probability input is read-only.

Host validation:

```sh
c++ -std=c++17 -O2 -Wall -Wextra -Werror -fsanitize=address,undefined \
  translation/ocr_pipeline/{postprocess,characters,host_test}.cpp \
  -o artifacts/translator/ocr_postprocess_test
artifacts/translator/ocr_postprocess_test
```

Tests cover clamped edge geometry, area ordering/capacity, confidence filtering, a dense full-frame component, empty/invalid maps, repeated-letter CTC and blank handling, padding exclusion, output bounds, invalid logits, stability, retained pending overlays, TTL, and timestamp wraparound. Cortex-M55 cross-compilation with Arm GNU 15.3.1 `-Os -fno-exceptions -fno-rtti -fno-fast-math` succeeds; the postprocessor object contains 2,436 bytes of code before link-time changes and C math functions.
