// SPDX-License-Identifier: Apache-2.0
#include "live.h"
#include "RTE_Components.h"
#include CMSIS_device_header
#include "../../src/trace.h"
#include "external_memory.h"
#include "asset_transfer.h"
#include "integrity.h"
#include "benchmark_service.h"
#include "runner.h"
#include "../staging/live_engine.h"
#include "../staging/manual_service.h"
#include "../ocr_pipeline/application.h"
#include "../multi_npu/boot_he.h"
extern "C" {
volatile uint32_t live_assets_ready,live_pipeline_state,live_pipeline_error;
}
extern "C" int app_main(void) {
    trace_printf("TRANSLATOR_LIVE_BOOT\r\n");
    if (live_init()) for (;;) __WFI();
    live_pipeline_state=1;
    int result=live_memory_initialize();
    if (result) {
        live_pipeline_state=0xe1;
        live_pipeline_error=result;
        trace_printf("LIVE_MEMORY_ERROR code=%d\r\n",result);
    } else {
        live_pipeline_state=2;
        const int he_result=live_boot_he_worker();
        trace_printf("LIVE_HE_BOOT result=%d\r\n",he_result);
    }
    bool initialized=false;
    for (;;) {
        live_service();
        if (live_pipeline_state==2 && !live_assets_ready) {
            live_asset_service();
            live_integrity_service();
        }
        if (live_pipeline_state==2 && live_assets_ready && !initialized) {
            SCB_InvalidateDCache();
            live_pipeline_state=3;
            result=live_npu_initialize();
            if (!result) result=staged_engine_init();
            if (!result) result=ocr_app_init();
            live_pipeline_error=result;
            live_pipeline_state=result ? 0xe2 : 4;
            initialized=!result;
            trace_printf("LIVE_PIPELINE_READY result=%d\r\n",result);
        }
        if (initialized) {
            if (live_button_pressed() && !live_benchmark_enabled) ocr_app_button_press();
            if (live_benchmark_enabled==2) manual_translation_service();
            else if (live_benchmark_enabled) live_benchmark_service();
            else ocr_app_service();
        }
    }
}
