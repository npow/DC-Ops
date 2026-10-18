// SPDX-License-Identifier: Apache-2.0
#include "translation_display.h"
#include "RTE_Components.h"
#include CMSIS_device_header
#include "RTE_Device.h"
#include "Driver_CDC200.h"

#if RTE_CDC200_PIXEL_FORMAT != 2
#error "The translator display requires RGB565 (RTE_CDC200_PIXEL_FORMAT 2)"
#endif
extern "C" ARM_DRIVER_CDC200 Driver_CDC200;
namespace {
constexpr int width=RTE_PANEL_HACTIVE_TIME, height=RTE_PANEL_VACTIVE_LINE;
alignas(32) uint16_t frames[2][width*height] __attribute__((section(".bss.lcd_frame_buf")));
unsigned front;
volatile bool ready;
bool initialized;
volatile uint32_t vsyncs;
void display_callback(uint32_t event) {
    if (event & ARM_CDC_SCANLINE0_EVENT) ++vsyncs;
    if (event & ARM_CDC_DSI_ERROR_EVENT) ready=false;
}
}
extern "C" {
volatile uint32_t translator_lcd_presents, translator_lcd_timeouts, translator_lcd_errors;
}
extern "C" int translator_display_init(void) {
    ready=false;
    if (initialized) {
        if (Driver_CDC200.Stop()!=ARM_DRIVER_OK) return -1;
        if (Driver_CDC200.PowerControl(ARM_POWER_OFF)!=ARM_DRIVER_OK) return -1;
        if (Driver_CDC200.Uninitialize()!=ARM_DRIVER_OK) return -1;
        initialized=false;
    }
    front=0;
    vsyncs=0;
    for (auto &frame:frames) for (auto &pixel:frame) pixel=0;
    SCB_CleanDCache_by_Addr(reinterpret_cast<uint32_t *>(frames),sizeof(frames));
    if (Driver_CDC200.Initialize(display_callback)!=ARM_DRIVER_OK) return -1;
    initialized=true;
    if (Driver_CDC200.PowerControl(ARM_POWER_FULL)!=ARM_DRIVER_OK ||
        Driver_CDC200.Control(CDC200_CONFIGURE_DISPLAY,reinterpret_cast<uint32_t>(frames[0]))!=ARM_DRIVER_OK ||
        Driver_CDC200.Control(CDC200_SCANLINE0_EVENT,1)!=ARM_DRIVER_OK ||
        Driver_CDC200.Start()!=ARM_DRIVER_OK) return -1;
    ready=true;
    return 0;
}
extern "C" int translator_display_frame(const dcops::Image *image,
    const dcops::TranslationBox *boxes,size_t count,const dcops::DisplayStats *stats) {
    if (!ready || !image) return -1;
    auto *back=frames[front^1];
    if (!dcops::render_translation({back,width,height,width},*image,boxes,count,stats)) return -1;
    SCB_CleanDCache_by_Addr(reinterpret_cast<uint32_t *>(back),sizeof(frames[0]));
    __DSB();
    const uint32_t start=DWT->CYCCNT;
    if (Driver_CDC200.Control(CDC200_FRAMEBUF_UPDATE_VSYNC,reinterpret_cast<uint32_t>(back))!=ARM_DRIVER_OK) {
        ++translator_lcd_errors;
        ready=false;
        return -1;
    }
    const uint32_t before=vsyncs;
    // The second event guarantees that scanout has released the old front buffer.
    while (static_cast<uint32_t>(vsyncs-before)<2) {
        if (!ready || static_cast<uint32_t>(DWT->CYCCNT-start)>SystemCoreClock/4) {
            ++translator_lcd_timeouts;
            ready=false;
            return -1;
        }
    }
    front^=1;
    ++translator_lcd_presents;
    return 0;
}
