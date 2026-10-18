// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stddef.h>
#include <stdint.h>

namespace dcops {
struct Image {
    const uint16_t *pixels;
    int width, height, stride;
};
struct Surface {
    uint16_t *pixels;
    int width, height, stride;
};
// Coordinates are pixels in the upright source image, before the center crop.
struct TranslationBox {
    float x0, y0, x1, y1;
    const char *translated_utf8;
    uint16_t outline_color = 0;
};
struct Viewport {
    int x, y, width, height;
};
struct DisplayStats {
    unsigned camera_fps_x10;
    unsigned detector_ms, recognizer_ms, translator_ms;
    unsigned cpu_ms, npu_ms;
    int npu_active_percent; // -1 when no measured utilization is available.
};
Viewport camera_viewport(int source_width, int source_height, int output_width, int output_height);
// Source and destination must not overlap. No allocations; bounded to 32 boxes.
bool render_translation(Surface output, Image upright_camera,
                        const TranslationBox *boxes, size_t count,
                        const DisplayStats *stats = nullptr);
}

extern "C" {
// Call once, after the board's display clocks/pins/MIPI power are configured.
int translator_display_init(void);
// A failed present leaves the buffers frozen until init is explicitly retried.
int translator_display_frame(const dcops::Image *image,
    const dcops::TranslationBox *boxes, size_t count, const dcops::DisplayStats *stats);
}
