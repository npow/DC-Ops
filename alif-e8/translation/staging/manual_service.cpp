#include "manual_service.h"

#include "live_engine.h"
#include "../live/live.h"
#include "../multi_npu/multi_runtime.h"
#include "../../src/trace.h"

#include <cstring>

extern "C" {
volatile uint32_t manual_translation_command, manual_translation_status;
volatile uint32_t manual_translation_error, manual_translation_elapsed_ms;
volatile uint32_t manual_translation_pause_display = 1;
char manual_translation_input[256] = "Hello";
char manual_translation_output[512];

int manual_translation_service(void) {
    uint32_t command = manual_translation_command;
    if (!command) return 0;
    if (command != 1) {
        manual_translation_error = static_cast<uint32_t>(-1);
        manual_translation_status = 0xe0;
        manual_translation_command = 0;
        return -1;
    }
    manual_translation_status = 1;
    manual_translation_error = manual_translation_elapsed_ms = 0;
    manual_translation_output[0] = 0;
    char input[sizeof(manual_translation_input)];
    std::memcpy(input, manual_translation_input, sizeof(input));
    input[sizeof(input)-1] = 0;
    trace_printf("MANUAL_TRANSLATION injected_input=%s\r\n", input);
    uint32_t started = live_now_ms();
    int result = live_npu_set_fast_scratch_enabled(1);
    int cleanup = 0;
    if (!result) {
        const uint32_t previous_pause = live_display_paused;
        bool restore_display = false;
        if (manual_translation_pause_display) {
            result = live_set_display_paused(1);
            restore_display = result == 0;
        }
        if (!result) result = staged_engine_translate(input, manual_translation_output,
                                                     sizeof(manual_translation_output));
        if (restore_display) {
            int display_error = live_set_display_paused(previous_pause ? 1 : 0);
            if (!result) result = display_error;
        }
        cleanup = live_npu_set_fast_scratch_enabled(0);
        if (!result) result = cleanup;
    }
    manual_translation_output[sizeof(manual_translation_output)-1] = 0;
    manual_translation_elapsed_ms = live_now_ms()-started;
    manual_translation_error = static_cast<uint32_t>(result);
    manual_translation_status = result ? 0xe0 : 2;
    manual_translation_command = 0;
    trace_printf("MANUAL_TRANSLATION result=%d cleanup=%d elapsed_ms=%u pause_display=%u output=%s\r\n",
                 result, cleanup, static_cast<unsigned>(manual_translation_elapsed_ms),
                 static_cast<unsigned>(manual_translation_pause_display),
                 manual_translation_output);
    return result;
}
}
