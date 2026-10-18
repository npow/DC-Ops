#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
int manual_translation_service(void);
extern volatile uint32_t manual_translation_command, manual_translation_status;
extern volatile uint32_t manual_translation_error, manual_translation_elapsed_ms;
extern volatile uint32_t manual_translation_pause_display;
extern char manual_translation_input[256], manual_translation_output[512];
#ifdef __cplusplus
}
#endif
