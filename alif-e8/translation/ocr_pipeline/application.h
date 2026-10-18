#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int ocr_app_init(void);
int ocr_app_service(void);
// Owner thread, outside live_service: capture next frame, or return frozen result to live preview.
int ocr_app_button_press(void);
extern volatile uint32_t ocr_single_shot_enabled, ocr_capture_state;
extern volatile uint32_t ocr_capture_requests, ocr_captured_frame;
extern volatile uint32_t ocr_sign_mode, ocr_sign_lookup_ms, ocr_sign_status, ocr_sign_matched;
extern volatile uint32_t ocr_capture_elapsed_ms;
extern volatile uint32_t ocr_capture_command; //1: same owner-thread event as a joystick press.
extern volatile uint32_t ocr_state, ocr_runs, ocr_error;
extern volatile uint32_t ocr_detector_ms, ocr_recognizer_ms, ocr_translator_ms;
extern volatile uint32_t ocr_boxes, ocr_confidence_x1000;
extern volatile uint32_t ocr_pause_display_during_translation;
extern char ocr_english[256], ocr_spanish[512];
#ifdef __cplusplus
}
#endif
