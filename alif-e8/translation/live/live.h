// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../camera.h"

extern "C" {
int live_init(void);
// Invoke from the single camera/display owner, including NPU semaphore waits.
int live_service(void);
// Freeze the last LCD frame while the current camera DMA finishes. Owner thread only.
// Returns0 on success, -1 for invalid values or a transition during live_service.
int live_set_display_paused(int paused);
// Caller must use the same owner thread; text is copied before this returns.
typedef void (*live_frame_callback)(const dcops::Image *image,
    const dcops::CameraPlanes *planes,unsigned sequence,void *context);
// Callback runs before restarting DMA. Copy/preprocess only; never run a model here.
void live_set_frame_callback(live_frame_callback callback,void *context);
int live_set_boxes(const dcops::TranslationBox *boxes, size_t count);
int live_set_ocr_region(const dcops::TranslationBox *region, int detected);
int live_refresh_display(void);
int live_button_pressed(void);
void live_set_stats(const dcops::DisplayStats *stats);
uint32_t live_now_ms(void);
extern volatile uint32_t live_status, live_frames, live_error, live_milliseconds;
extern volatile uint32_t live_display_paused, live_display_pause_count;
extern volatile uint32_t live_display_last_pause_ms, live_display_total_pause_ms;
}
