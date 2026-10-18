The renderer is ready for the translator application; it is not wired into the currently flashed benchmark.

1. Add `translation/display.cpp` and `translation/display_board.cpp` to the final camera/translator project. Exclude `board/DevKit-E8/display.c` from that project: both implementations own a complete LCD framebuffer pair. Keep the original detector project unchanged.
2. Keep the working board LCD initialization: AppKit reset pin GPIO13/6, RGB565 panel configuration, board pin/GPIO setup, HFOSC/100 MHz clocks, and MIPI DPHY power. Enable DWT CYCCNT before calling `translator_display_frame`. Call `translator_display_init` once after board setup.
3. Feed the upright RGB565 camera buffer to the API below after the existing planar-ISP conversion/180-degree camera correction. This renderer does no second rotation. Render each new camera frame even when OCR/translation results have not changed. Schedule inference separately; this presenter waits at most 250 ms for scanout. It cannot make a blocking single-thread token loop produce live video.
4. Pass only actual completed translations for their corresponding camera text regions. Expire cached regions when tracking loses the source text or the camera moves. Supply `nullptr, 0` while no translation is available. The API does not generate translations, invent labels, or display sample phrases.
5. Pass measured stage timings and NPU utilization in `DisplayStats`, or pass `nullptr` to hide the HUD. Set `npu_active_percent=-1` for unavailable counters; CPU/NPU speedup is displayed only when both corresponding measured runtimes are nonzero. Those two runtimes must refer to the same workload.

```cpp
#include "translation/translation_display.h"

// After board pins/clocks/MIPI setup and enabling DWT CYCCNT:
int result = translator_display_init();

// For each upright camera frame (320x240 matches the existing camera config):
dcops::Image camera{dc_camera_frame, 320, 240, 320};
// boxes: actual results expressed in pixels of this full upright camera image.
result = translator_display_frame(&camera, boxes, box_count, nullptr);
```

`camera_viewport(camera_width, camera_height, 480, 800)` gives the aspect-preserving center crop. At 320x240 this is x=88,y=0,width=144,height=240; at 640x480 it is x=176,y=0,width=288,height=480. OCR should consume that same crop to recognize exactly what the screen shows. To map a detector coordinate `(dx,dy)` from a `DW x DH` detector image into a source pixel, use `sx=crop.x + dx*crop.width/DW`, `sy=crop.y + dy*crop.height/DH`. Map both box corners before creating `TranslationBox`.

The panel receives camera pixels over the entire 480x800 screen. Translation boxes cover the source text with a solid color sampled just outside the region, select contrasting text, wrap words, and shrink the font to fit. Long translations may expand the patch vertically up to half the screen; remaining overflow is indicated with dots. This is rectangular background replacement, not perspective correction or learned inpainting. Spanish UTF-8 áéíóúüñ/ÁÉÍÓÚÜÑ/¿¡ and typographic quotes/dashes are supported. Other codepoints display `?`.

Memory: the presenter owns **1,536,000 bytes** of 32-byte-aligned `.bss.lcd_frame_buf`, currently mapped by the existing linker to bulk SRAM0. It does not use external PSRAM. Renderer peak stack is approximately 3.5 KB (bounded 256-codepoint text and line arrays); rendering uses no heap. Camera buffers remain owned by the application. Do not overlap source and destination buffers. Ensure the final SRAM map fits all display, camera, NPU and runtime pools together; moving this section to another LCD-visible SRAM bank requires board validation.

A failed swap freezes rendering to avoid writing a buffer that the LCD may still scan. Counters are `translator_lcd_presents`, `translator_lcd_timeouts`, `translator_lcd_errors`. Retry only by explicitly calling `translator_display_init`, which stops and reinitializes the controller before clearing buffers.

Validation completed without accessing hardware:

- Clang C++17 `-Wall -Wextra -Werror` with AddressSanitizer and UndefinedBehaviorSanitizer: crop pixels, center-crop box clipping, Spanish glyph differences, malformed UTF-8, nonfinite/extreme boxes, invalid inputs, framebuffer guard bytes all pass.
- Cortex-M55 GCC 15.3.1 `-Wall -Wextra -Werror -fsyntax-only` passes for renderer and presenter using the translator project's actual generated include paths and device defines.

```sh
clang++ -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined -g \
  translation/display.cpp translation/display_test.cpp -o /tmp/dcops-display-test
/tmp/dcops-display-test
```
