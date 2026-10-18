// SPDX-License-Identifier: Apache-2.0
#include "live.h"
#include "button_input.h"
#include "../camera.h"
#include "RTE_Components.h"
#include CMSIS_device_header
#include "board_config.h"
#include "se_services_port.h"
#include "services_lib_api.h"
#include "../../src/trace.h"
#include <cstring>

extern "C" {
volatile uint32_t live_status,live_frames,live_error,live_milliseconds;
volatile uint32_t live_display_paused,live_display_pause_count;
volatile uint32_t live_display_last_pause_ms,live_display_total_pause_ms;
volatile uint32_t live_button_down,live_button_presses;
volatile uint32_t live_button_pins,live_button_events;
volatile uint32_t live_show_ocr_regions = 1;
uint32_t live_now_ms(void) { return live_milliseconds; }
void SysTick_Handler(void) { ++live_milliseconds; }
}
namespace {
constexpr size_t max_boxes=12,max_text=512;
dcops::TranslationBox boxes[max_boxes];
dcops::TranslationBox ocr_region;
bool show_ocr_region;
char strings[max_boxes][max_text];
size_t box_count;
dcops::DisplayStats measured_stats;
bool show_stats,ready,servicing;
live_frame_callback frame_callback;
void *frame_context;
uint32_t fps_started,fps_frames,pause_started;
void display_status(const char *status) {
    const uint16_t background=0x0841;
    const dcops::Image image{&background,1,1,1};
    const dcops::TranslationBox label{0,.4f,1,.6f,status};
    translator_display_frame(&image,&label,1,nullptr);
}
}
extern "C" int live_button_pressed(void) {
    static dcops::ButtonInput input;
    auto *gpio=reinterpret_cast<volatile GPIO_Type *>(LPGPIO_BASE);
    const uint32_t pins=~gpio->GPIO_EXT_PORTA & 0x1f;
    const uint32_t events=gpio->GPIO_RAW_INTSTATUS & 0x1f;
    if (events) gpio->GPIO_PORTA_EOI=events;
    live_button_down=pins!=0;
    live_button_pins=pins;
    live_button_events|=events;
    if (!input.press(pins,events,live_now_ms())) return 0;
    ++live_button_presses;
    return 1;
}
extern "C" int live_refresh_display(void) {
    if (!ready || servicing) return -1;
    servicing=true;
    const auto image=dcops::camera_image();
    dcops::TranslationBox updates[max_boxes+1];
    for (size_t i=0;i<box_count;++i) updates[i]=boxes[i];
    size_t count=box_count;
    if (show_ocr_region && live_show_ocr_regions) updates[count++]=ocr_region;
    const int result=translator_display_frame(&image,updates,count,show_stats ? &measured_stats : nullptr);
    servicing=false;
    return result;
}
extern "C" int live_set_ocr_region(const dcops::TranslationBox *region,int detected) {
    if (servicing) return -1;
    show_ocr_region=region!=nullptr && detected;
    if (region) {
        ocr_region=*region;
        ocr_region.translated_utf8=nullptr;
        ocr_region.outline_color=detected ? 0x07e0 : 0xffe0;
    }
    return 0;
}
extern "C" int live_set_boxes(const dcops::TranslationBox *updates,size_t count) {
    if ((count && !updates) || count>max_boxes || servicing) return -1;
    for (size_t i=0;i<count;++i) {
        const char *source=updates[i].translated_utf8;
        size_t length=0;
        if (source) while (length<max_text && source[length]) ++length;
        if (length==max_text) return -1;
    }
    for (size_t i=0;i<count;++i) {
        boxes[i]=updates[i];
        const char *source=updates[i].translated_utf8;
        if (source) std::strcpy(strings[i],source);
        else strings[i][0]=0;
        boxes[i].translated_utf8=strings[i];
    }
    box_count=count;
    return 0;
}
extern "C" void live_set_frame_callback(live_frame_callback callback,void *context) {
    frame_callback=callback; frame_context=context;
}
extern "C" void live_set_stats(const dcops::DisplayStats *stats) {
    show_stats=stats!=nullptr;
    if (stats) {
        const unsigned fps=measured_stats.camera_fps_x10;
        measured_stats=*stats;
        measured_stats.camera_fps_x10=fps;
    }
}
extern "C" int live_set_display_paused(int paused) {
    if ((paused!=0 && paused!=1) || servicing) return -1;
    if (live_display_paused==static_cast<uint32_t>(paused)) return 0;
    const uint32_t now=live_now_ms();
    if (paused) {
        pause_started=now;
        ++live_display_pause_count;
    } else {
        live_display_last_pause_ms=now-pause_started;
        live_display_total_pause_ms+=live_display_last_pause_ms;
        fps_started=now;
        fps_frames=0;
        measured_stats.camera_fps_x10=0;
    }
    live_display_paused=paused;
    return 0;
}
extern "C" int live_init(void) {
    ready=false;
    live_display_paused=0;
    live_status=1;
    CoreDebug->DEMCR|=CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL|=DWT_CTRL_CYCCNTENA_Msk;
    SysTick_Config(SystemCoreClock/1000);
    uint32_t se_error=0;
    // Preserve SRAM0/MRAM power while explicitly enabling the display's SRAM1.
    uint32_t result=SERVICES_power_memory_req(se_services_s_handle,
        POWER_MEM_SRAM_0_ENABLE|POWER_MEM_SRAM_1_ENABLE|POWER_MEM_MRAM_ENABLE,&se_error);
    if (result || se_error) { live_error=result ? result : se_error; live_status=0xe1; return -1; }
    board_clocks_config(CLKEN_HFOSC_MASK|CLKEN_CLK_100M_MASK|(1U<<CLKEN_SRAM1));
    result=translator_display_init();
    if (result) { live_error=result; live_status=0xe2; return -1; }
    live_status=2;
    display_status("INITIALIZING CAMERA");
    result=dcops::camera_init();
    if (!result) result=dcops::camera_start();
    if (result) { live_error=result; live_status=0xe3; display_status("CAMERA START ERROR"); return -1; }
    fps_started=live_now_ms();
    ready=true;
    live_status=3;
    trace_printf("LIVE_READY camera=320x240 panel=480x800 upright=1 fullscreen=1\r\n");
    return 0;
}
extern "C" int live_service(void) {
    if (!ready || servicing) return -1;
    if (live_display_paused) return 0;
    servicing=true;
    const auto state=dcops::camera_poll();
    if (state==dcops::CaptureState::Pending) { servicing=false; return 0; }
    if (state==dcops::CaptureState::Error) {
        live_status=0xe4; live_error=1; ready=false; servicing=false;
        display_status("CAMERA CAPTURE ERROR");
        trace_printf("LIVE_CAMERA_ERROR frame=%lu\r\n",live_frames);
        return -1;
    }
    auto image=dcops::camera_image();
    if (frame_callback) {
        auto planes=dcops::camera_planes();
        frame_callback(&image,&planes,dcops::camera_sequence(),frame_context);
    }
    // The next ISP capture owns planar memory only; RGB565 remains our draw source.
    int result=dcops::camera_start();
    dcops::TranslationBox updates[max_boxes+1];
    for (size_t i=0;i<box_count;++i) updates[i]=boxes[i];
    size_t count=box_count;
    if (show_ocr_region && live_show_ocr_regions) updates[count++]=ocr_region;
    if (!result) result=translator_display_frame(&image,updates,count,show_stats ? &measured_stats : nullptr);
    if (result) {
        live_status=0xe5; live_error=result; ready=false; servicing=false;
        trace_printf("LIVE_DISPLAY_ERROR frame=%lu code=%d\r\n",live_frames,result);
        return -1;
    }
    ++live_frames; ++fps_frames;
    const auto elapsed=live_now_ms()-fps_started;
    if (elapsed>=1000) {
        measured_stats.camera_fps_x10=fps_frames*10000/elapsed;
        fps_started=live_now_ms(); fps_frames=0;
    }
    live_status=4;
    if (live_frames%30==1)
        trace_printf("LIVE frame=%lu boxes=%u fps_x10=%u\r\n",live_frames,unsigned(box_count),measured_stats.camera_fps_x10);
    servicing=false;
    return 1;
}
