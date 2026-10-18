// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "translation_display.h"

namespace dcops {
// Three contiguous R,G,B planes; each plane has width*height bytes.
struct CameraPlanes {
    const uint8_t *pixels;
    int width, height;
    bool rotated_180;
};
enum class CaptureState { Error=-1, Pending=0, Ready=1 };
// Singleton driver/buffers. Call only from the camera/display owner thread.
int camera_init();
int camera_start();
CaptureState camera_poll();
int camera_stop();
// RGB565 remains valid until the next successful poll; planes until next start.
// Return an empty view before the first successful capture.
Image camera_image();
CameraPlanes camera_planes();
unsigned camera_sequence();

// NCHW BGR float input using an aspect-preserving center crop.
bool detector_input(CameraPlanes image, float *output, size_t output_elements,
                    int detector_width = 288, int detector_height = 480);
// Axis-aligned upright SOURCE pixel region. Output is NCHW BGR 3*48*320.
// Return resized valid width (1..320), or 0 for invalid inputs. Right padding is
// zero in normalized space; CTC must ignore time steps beyond valid width.
int recognizer_input(CameraPlanes image, Viewport source_region,
                     float *output, size_t output_elements);
// Convert detector coordinates (288x480) back to upright source coordinates.
TranslationBox detector_box_to_source(CameraPlanes image, float x0, float y0,
                                     float x1, float y1, const char *translation,
                                     int detector_width = 288, int detector_height = 480);
}
