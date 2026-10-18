// SPDX-License-Identifier: Apache-2.0
#include "RTE_Components.h"
#include CMSIS_device_header
#include "Driver_CPI.h"
#include "Driver_ISP.h"
#include "vsi_comm_video.h"
#include "detection.h"
#include "executorch_runner.h"
#include "model_params.h"
#include "model_pte.h"
#include "trace.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#define printf trace_printf
extern "C" ARM_DRIVER_CPI Driver_CPI;
extern "C" ARM_DRIVER_ISP Driver_ISP;
static VIDEO_BUF_S camera_buffer;
extern "C" int board_console_poll(void);
constexpr int CW = 320, CH = 240, S = MODEL_INPUT_SIZE;
constexpr int MAX_DETECTIONS = 12, MAX_CANDIDATES = 96;
__attribute__((section(".bss.ai_pool"), aligned(32))) uint16_t dc_camera_frame[CW * CH];
__attribute__((section(".bss.ai_pool"), aligned(32))) uint8_t dc_isp_frame[CW * CH * 3];
__attribute__((section(".bss.ai_pool"), aligned(32))) int8_t dc_input[3 * S * S];
extern "C" {
const char *dc_class_names[16] = {"server rack", "compute tray", "NVLink switch tray",
    "network switch", "power shelf", "cable", "network port", "LED indicator",
    "label", "fan", "cooling manifold", "cable cartridge", "power connector",
    "drive bay", "management port", "DPU"};
volatile unsigned dc_frame_count, dc_detection_count, dc_inference_ms;
volatile int dc_status;
volatile unsigned dc_camera_checksum;
Detection dc_detections[MAX_DETECTIONS];
}
static volatile uint32_t camera_events;
static void camera_callback(uint32_t event) { camera_events |= event; }
static bool camera_init() {
    int ret = Driver_CPI.Initialize(camera_callback);
    printf("CAMERA_INITIALIZE=%d\r\n", ret);
    if (ret == ARM_DRIVER_OK) ret = Driver_CPI.PowerControl(ARM_POWER_FULL);
    printf("CAMERA_POWER=%d\r\n", ret);
    if (ret == ARM_DRIVER_OK) ret = Driver_CPI.Control(CPI_CONFIGURE, 0);
    if (ret == ARM_DRIVER_OK) ret = Driver_CPI.Control(CPI_CAMERA_SENSOR_CONFIGURE, 0);
    if (ret == ARM_DRIVER_OK) ret = Driver_CPI.Control(CPI_EVENTS_CONFIGURE,
        ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED | ARM_CPI_EVENT_ERR_HARDWARE |
        ARM_CPI_EVENT_ERR_CAMERA_INPUT_FIFO_OVERRUN |
        ARM_CPI_EVENT_ERR_CAMERA_OUTPUT_FIFO_OVERRUN | ARM_CPI_EVENT_MIPI_CSI2_ERROR);
    if (ret == ARM_DRIVER_OK) {
        camera_buffer = {};
        camera_buffer.index = 0;
        camera_buffer.numPlanes = 3;
        camera_buffer.imageSize = sizeof(dc_isp_frame);
        camera_buffer.planes[0].dmaPhyAddr = (vsi_dma_t)dc_isp_frame;
        camera_buffer.planes[1].dmaPhyAddr = (vsi_dma_t)(dc_isp_frame + CW * CH);
        camera_buffer.planes[2].dmaPhyAddr = (vsi_dma_t)(dc_isp_frame + 2 * CW * CH);
    }
    printf("CAMERA_CONFIGURE=%d\r\n", ret);
    return ret == ARM_DRIVER_OK;
}
static bool capture() {
    camera_events = 0;
    SCB_CleanInvalidateDCache_by_Addr((uint32_t *)dc_isp_frame, sizeof(dc_isp_frame));
    if (Driver_ISP.Control(ISP_CONTROL_QBUF, (uint32_t)&camera_buffer) != ARM_DRIVER_OK) return false;
    if (Driver_CPI.CaptureFrame((uint8_t *)dc_camera_frame) != ARM_DRIVER_OK) return false;
    const uint32_t start = DWT->CYCCNT;
    while (!(camera_events & ARM_ISP_MI_EVENT_MP_FRAME_END_DETECTED)) {
        if ((uint32_t)(DWT->CYCCNT - start) > 1200000000U) {
            printf("CAMERA_TIMEOUT events=%lx\r\n", camera_events);
            Driver_CPI.Stop();
            return false;
        }
    }
    Driver_ISP.Control(ISP_PROCESS_FRAME_END, 0);
    Driver_ISP.Control(ISP_CONTROL_DQBUF, (uint32_t)&camera_buffer);
    Driver_CPI.Stop();
    SCB_InvalidateDCache_by_Addr((uint32_t *)dc_isp_frame, sizeof(dc_isp_frame));
    for (int i = 0; i < CW * CH; ++i) {
        const unsigned r = dc_isp_frame[i], g = dc_isp_frame[CW*CH+i], b = dc_isp_frame[2*CW*CH+i];
        dc_camera_frame[CW*CH-1-i] = ((r>>3)<<11) | ((g>>2)<<5) | (b>>3);
    }
    unsigned checksum = 0;
    for (int i = 0; i < CW * CH; i += 31) checksum = checksum * 33 + dc_camera_frame[i];
    dc_camera_checksum = checksum;
    return true;
}
static void preprocess() {
    int8_t lut[256];
    for (int i = 0; i < 256; ++i) {
        int value = (int)std::round(i / (255.f * MODEL_INPUT_SCALE)) + MODEL_INPUT_ZERO_POINT;
        lut[i] = (int8_t)std::max(-128, std::min(127, value));
    }
    const int resized_h = S * CH / CW;
    const int top = (S - resized_h) / 2;
    memset(dc_input, lut[114], sizeof(dc_input));
    for (int y = 0; y < resized_h; ++y) {
        const int sy = y * CH / resized_h;
        for (int x = 0; x < S; ++x) {
            const uint16_t pixel = dc_camera_frame[sy * CW + x * CW / S];
            int offset = ((y + top) * S + x) * 3;
            dc_input[offset] = lut[((pixel >> 11) & 31) * 255 / 31];
            dc_input[offset + 1] = lut[((pixel >> 5) & 63) * 255 / 63];
            dc_input[offset + 2] = lut[(pixel & 31) * 255 / 31];
        }
    }
}
static float logit(const int8_t *data, int index, int head) {
    return (data[index] - model_output_zero_points[head]) * model_output_scales[head];
}
static float overlap(const Detection &a, const Detection &b) {
    const float w = std::max(0.f, std::min(a.x1,b.x1) - std::max(a.x0,b.x0));
    const float h = std::max(0.f, std::min(a.y1,b.y1) - std::max(a.y0,b.y0));
    const float intersection = w * h;
    const float area = (a.x1-a.x0)*(a.y1-a.y0) + (b.x1-b.x0)*(b.y1-b.y0) - intersection;
    return area > 0 ? intersection / area : 0;
}
static int decode(const int8_t **outputs) {
    struct Candidate { float score; int head, cell, class_id; };
    Candidate candidates[MAX_CANDIDATES];
    int count = 0;
    for (int head = 0; head < 3; ++head) {
        const int grid = S / (8 << head), cells = grid * grid;
        for (int cell = 0; cell < cells; ++cell) {
            int cls = 0;
            int8_t best = outputs[head][64 * cells + cell];
            for (int c = 1; c < 16; ++c) {
                const int8_t value = outputs[head][(64+c)*cells+cell];
                if (value > best) { cls = c; best = value; }
            }
            float score = 1.f / (1.f + std::exp(-(best - model_output_zero_points[head]) * model_output_scales[head]));
            if (score < .35f) continue;
            int at = count;
            if (at == MAX_CANDIDATES) {
                if (score <= candidates[at-1].score) continue;
                --at;
            } else ++count;
            while (at > 0 && candidates[at-1].score < score) {
                candidates[at] = candidates[at-1]; --at;
            }
            candidates[at] = {score, head, cell, cls};
        }
    }
    int accepted = 0;
    for (int k = 0; k < count && accepted < MAX_DETECTIONS; ++k) {
        const auto &c = candidates[k];
        const int stride = 8 << c.head, grid = S / stride, cells = grid * grid;
        float distance[4];
        for (int side = 0; side < 4; ++side) {
            float maximum = -1e10f;
            for (int bin = 0; bin < 16; ++bin)
                maximum = std::max(maximum, logit(outputs[c.head], (side*16+bin)*cells+c.cell, c.head));
            float sum = 0, weighted = 0;
            for (int bin = 0; bin < 16; ++bin) {
                float e = std::exp(logit(outputs[c.head], (side*16+bin)*cells+c.cell, c.head) - maximum);
                sum += e; weighted += bin * e;
            }
            distance[side] = weighted / sum * stride;
        }
        const float x = ((c.cell % grid) + .5f) * stride;
        const float y = ((c.cell / grid) + .5f) * stride;
        const float top = (S - S * CH / CW) / 2;
        Detection d = {(x-distance[0])/S, (y-distance[1]-top)/(S*CH/CW),
            (x+distance[2])/S, (y+distance[3]-top)/(S*CH/CW), c.score, c.class_id};
        d.x0 = std::clamp(d.x0, 0.f, 1.f); d.x1 = std::clamp(d.x1, 0.f, 1.f);
        d.y0 = std::clamp(d.y0, 0.f, 1.f); d.y1 = std::clamp(d.y1, 0.f, 1.f);
        if (d.x1 <= d.x0 || d.y1 <= d.y0) continue;
        bool suppressed = false;
        for (int i = 0; i < accepted; ++i)
            if (d.class_id == dc_detections[i].class_id && overlap(d,dc_detections[i]) > .45f) suppressed = true;
        if (!suppressed) dc_detections[accepted++] = d;
    }
    return accepted;
}
extern "C" int app_main() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    DWT->CYCCNT = 0;
    GPIO7->GPIO_SWPORTA_DDR |= 1U << 4;
    GPIO7->GPIO_SWPORTA_DR |= 1U << 4;
    printf("DC_OPS_E8_BOOT model=%lu input=%d\r\n", model_pte_size, S);
    dc_status = 1;
    int result = executorch_init(model_pte, model_pte_size);
    if (result != 0) {
        dc_status = -1;
        printf("DETECTOR_INIT_FAILED=%d\r\n", result);
        board_display_status("Detector initialization failed");
        for (;;) __WFI();
    }
    if (!camera_init()) {
        dc_status = -2;
        board_display_status("Camera unavailable - check ribbon");
        for (;;) __WFI();
    }
    board_display_status("Starting live camera...");
    dc_status = 2;
    for (;;) {
        if (!capture()) {
            dc_status = -3;
            board_display_status("Camera capture timed out");
            for (;;) __WFI();
        }
        preprocess();
        const int8_t *outputs[3];
        const uint32_t start = DWT->CYCCNT;
        result = executorch_run_detector(dc_input, outputs);
        dc_inference_ms = (uint32_t)(DWT->CYCCNT-start)/400000;
        if (result != 0) {
            dc_status = -4;
            printf("INFERENCE_FAILED=%d\r\n", result);
            board_display_status("Inference failed");
            for (;;) __WFI();
        }
        int count = decode(outputs);
        dc_detection_count = count;
        ++dc_frame_count;
        if (dc_frame_count % 10 == 0) GPIO7->GPIO_SWPORTA_DR ^= 1U << 4;
        result = board_display_detection(dc_camera_frame, CW, CH, dc_detections,
            count, dc_frame_count, dc_inference_ms);
        dc_status = result == 0 ? 3 : -5;
        if (dc_frame_count % 30 == 1) {
            printf("LIVE frame=%u detections=%d inference_ms=%u camera=%08x lcd=%d\r\n",
                dc_frame_count, count, dc_inference_ms, dc_camera_checksum, result);
            for (int i = 0; i < count && i < 5; ++i)
                printf("  %s %u%%\r\n", dc_class_names[dc_detections[i].class_id],
                    (unsigned)(dc_detections[i].score*100));
        }
    }
}
