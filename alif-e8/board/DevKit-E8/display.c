/*---------------------------------------------------------------------------
 * Copyright (c) 2026 Arm Limited (or its affiliates). All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * AppKit-E8 LCD: render off-screen in RGB565, then swap at vertical blanking.
 *---------------------------------------------------------------------------*/

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "RTE_Components.h"
#include CMSIS_device_header
#include "RTE_Device.h"

#include "Driver_CDC200.h"
#include "../../src/detection.h"
#include "font.h"

#define LCD_WIDTH   RTE_PANEL_HACTIVE_TIME   /* 480 */
#define LCD_HEIGHT  RTE_PANEL_VACTIVE_LINE   /* 800 */

#if RTE_CDC200_PIXEL_FORMAT != 2
#error "display.c expects RTE_CDC200_PIXEL_FORMAT 2 (RGB565) in RTE_Device.h"
#endif
extern ARM_DRIVER_CDC200 Driver_CDC200;
static ARM_DRIVER_CDC200 *CDCdrv = &Driver_CDC200;

static uint16_t lcd_frame[2][LCD_HEIGHT][LCD_WIDTH]
    __attribute__((section(".bss.lcd_frame_buf"), aligned(32)));
static uint16_t (*lcd_draw)[LCD_WIDTH] = lcd_frame[1];
volatile uint32_t dc_lcd_front, dc_lcd_vsyncs, dc_lcd_presents, dc_lcd_timeouts;

static int display_ready = 0;

static void display_callback(uint32_t event)
{
    if (event & ARM_CDC_SCANLINE0_EVENT) ++dc_lcd_vsyncs;
    if (event & ARM_CDC_DSI_ERROR_EVENT) {
        printf("Display: DSI error event\n");
    }
}

/* Bring up the display controller, DSI link and panel; returns 0 on success. */
static int display_init(void)
{
    int32_t ret;

    memset(lcd_frame, 0, sizeof(lcd_frame));
    SCB_CleanDCache_by_Addr((uint32_t *) lcd_frame, sizeof(lcd_frame));

    ret = CDCdrv->Initialize(display_callback);
    if (ret != ARM_DRIVER_OK) {
        printf("Display: CDC200 Initialize failed (%d)\n", (int) ret);
        return -1;
    }
    ret = CDCdrv->PowerControl(ARM_POWER_FULL);
    if (ret != ARM_DRIVER_OK) {
        printf("Display: CDC200 PowerControl failed (%d)\n", (int) ret);
        return -1;
    }
    ret = CDCdrv->Control(CDC200_CONFIGURE_DISPLAY, (uint32_t) lcd_frame);
    if (ret != ARM_DRIVER_OK) {
        printf("Display: CDC200 configure failed (%d)\n", (int) ret);
        return -1;
    }
    ret = CDCdrv->Control(CDC200_SCANLINE0_EVENT, 1);
    if (ret != ARM_DRIVER_OK) return -1;
    ret = CDCdrv->Start();
    if (ret != ARM_DRIVER_OK) {
        printf("Display: CDC200 Start failed (%d)\n", (int) ret);
        return -1;
    }
    printf("Display: %ux%u RGB565 double-buffered panel started\n", (unsigned) LCD_WIDTH, (unsigned) LCD_HEIGHT);
    return 0;
}

static int display_present(void)
{
    SCB_CleanDCache_by_Addr((uint32_t *)lcd_draw, sizeof(lcd_frame[0]));
    int ret = CDCdrv->Control(CDC200_FRAMEBUF_UPDATE_VSYNC, (uint32_t)lcd_draw);
    if (ret != ARM_DRIVER_OK) return -1;
    uint32_t start = DWT->CYCCNT, vsync = dc_lcd_vsyncs;
    /* Two blanking events ensure the old front buffer is no longer scanned. */
    while ((uint32_t)(dc_lcd_vsyncs - vsync) < 2) {
        if ((uint32_t)(DWT->CYCCNT - start) > SystemCoreClock / 4) {
            ++dc_lcd_timeouts;
            return -1;
        }
    }
    dc_lcd_front ^= 1;
    lcd_draw = lcd_frame[dc_lcd_front ^ 1];
    ++dc_lcd_presents;
    return 0;
}

static void fill_rect(int x, int y, int width, int height, unsigned color)
{
    int x0 = x < 0 ? 0 : x, y0 = y < 0 ? 0 : y;
    int x1 = x + width > LCD_WIDTH ? LCD_WIDTH : x + width;
    int y1 = y + height > LCD_HEIGHT ? LCD_HEIGHT : y + height;
    for (int row = y0; row < y1; ++row) {
        for (int col = x0; col < x1; ++col) {
            lcd_draw[row][col] = ((color >> 8) & 0xf800) |
                ((color >> 5) & 0x07e0) | ((color >> 3) & 0x001f);
        }
    }
}
static void text(int x, int y, const char *s, int scale, unsigned color)
{
    for (; *s; ++s, x += 8 * scale) {
        if (*s < 32 || *s > 126) continue;
        for (int row = 0; row < 14; ++row)
            for (int col = 0; col < 8; ++col)
                if (font8x14[(unsigned char)*s-32][row] & (1U << (7-col)))
                    fill_rect(x+col*scale,y+row*scale,scale,scale,color);
    }
}
int board_display_status(const char *message)
{
    if (!display_ready) {
        if (display_init() != 0) return -1;
        display_ready = 1;
    }
    fill_rect(0,0,LCD_WIDTH,LCD_HEIGHT,0x080f1c);
    text(20,20,"DC-OPS",3,0x36efb1);
    text(20,120,message,1,0xe5edf7);
    return display_present();
}
int board_display_detection(const uint16_t *image, int width, int height,
    const Detection *detections, int count, unsigned frame, unsigned inference_ms)
{
    if (!display_ready) {
        if (display_init() != 0) return -1;
        display_ready = 1;
    }
    fill_rect(0,0,LCD_WIDTH,LCD_HEIGHT,0x080f1c);
    text(16,12,"DC-OPS",3,0x36efb1);
    text(296,24,"LIVE / E8",2,0xe5edf7);
    const int image_y = 72, image_h = 360;
    for (int y = 0; y < image_h; ++y) {
        for (int x = 0; x < LCD_WIDTH; ++x) {
            uint16_t pixel = image[(y*height/image_h)*width + x*width/LCD_WIDTH];
            lcd_draw[image_y+y][x] = pixel;
        }
    }
    for (int i = count-1; i >= 0; --i) {
        const Detection *d = detections+i;
        int x0 = (int)(d->x0*(LCD_WIDTH-1)), x1 = (int)(d->x1*(LCD_WIDTH-1));
        int y0 = image_y+(int)(d->y0*(image_h-1)), y1 = image_y+(int)(d->y1*(image_h-1));
        const unsigned color = 0x36efb1;
        fill_rect(x0,y0,x1-x0,2,color); fill_rect(x0,y1,x1-x0,2,color);
        fill_rect(x0,y0,2,y1-y0,color); fill_rect(x1,y0,2,y1-y0,color);
        char label[48];
        snprintf(label,sizeof(label),"%s %u%%",dc_class_names[d->class_id],(unsigned)(d->score*100));
        int label_width = (int)strlen(label)*8+8;
        int lx = x0;
        if (lx+label_width > LCD_WIDTH) lx = LCD_WIDTH-label_width;
        int ly = y0 >= image_y+18 ? y0-18 : y0;
        fill_rect(lx,ly,label_width,18,0x080f1c);
        text(lx+4,ly+2,label,1,color);
    }
    char line[64];
    snprintf(line,sizeof(line),"%d found | NPU %u ms",count,inference_ms);
    text(16,452,line,2,0xe5edf7);
    fill_rect(16,490,448,2,0x23354a);
    if (!count) {
        text(16,528,"No confident detections",2,0xa0b1c6);
        text(16,576,"Point at ports, cables or a rack",1,0xa0b1c6);
    }
    for (int i = 0; i < count && i < 5; ++i) {
        snprintf(line,sizeof(line),"%s",dc_class_names[detections[i].class_id]);
        text(16,512+i*44,line,2,0x36efb1);
        snprintf(line,sizeof(line),"%u%%",(unsigned)(detections[i].score*100));
        text(400,512+i*44,line,2,0xe5edf7);
    }
    snprintf(line,sizeof(line),"FRAME %u | YOLO INT8 | ETHOS-U85",frame);
    text(16,766,line,1,0xa0b1c6);
    return display_present();
}
