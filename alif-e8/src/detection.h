// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
typedef struct {
    float x0, y0, x1, y1, score;
    int class_id;
} Detection;
#ifdef __cplusplus
extern "C" {
#endif
extern const char *dc_class_names[16];
int board_display_detection(const uint16_t *image, int width, int height,
    const Detection *detections, int count, unsigned frame, unsigned inference_ms);
int board_display_status(const char *message);
#ifdef __cplusplus
}
#endif
