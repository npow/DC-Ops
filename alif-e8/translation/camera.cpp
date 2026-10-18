// SPDX-License-Identifier: Apache-2.0
#include "camera.h"
#include "RTE_Components.h"
#include CMSIS_device_header
#include "RTE_Device.h"
#include "Driver_CPI.h"
#include "Driver_ISP.h"
#include "vsi_comm_video.h"
#include "../src/trace.h"

extern "C" {
volatile uint32_t live_camera_events,live_camera_reason,live_camera_cycles;
volatile uint32_t live_camera_control_warnings;
volatile int32_t live_camera_process_result,live_camera_dequeue_result,live_camera_stop_result;
}

#ifndef DCOPS_CAMERA_WIDTH
#define DCOPS_CAMERA_WIDTH RTE_ISP_OUTPUT_WIDTH
#endif
#ifndef DCOPS_CAMERA_HEIGHT
#define DCOPS_CAMERA_HEIGHT RTE_ISP_OUTPUT_HEIGHT
#endif
static_assert(RTE_ISP_OUTPUT_FORMAT==39,"Camera capture requires RGB888 planar ISP output");
static_assert((DCOPS_CAMERA_WIDTH==320 && DCOPS_CAMERA_HEIGHT==240) ||
              (DCOPS_CAMERA_WIDTH==640 && DCOPS_CAMERA_HEIGHT==480),"Unsupported capture dimensions");
static_assert(RTE_CPI_AXI_PORT==0 && RTE_CPI_ISP_PORT==1,
              "Camera ownership requires ISP-only DMA; CPI AXI must be disabled");
extern "C" ARM_DRIVER_CPI Driver_CPI;
extern "C" ARM_DRIVER_ISP Driver_ISP;
namespace dcops {
namespace {
constexpr unsigned width=DCOPS_CAMERA_WIDTH,height=DCOPS_CAMERA_HEIGHT,pixels=width*height;
alignas(32) uint16_t rgb565[pixels] __attribute__((section(".bss.camera_frame_buf")));
alignas(32) uint8_t planar[pixels*3] __attribute__((section(".bss.camera_frame_bayer_to_rgb_buf")));
VIDEO_BUF_S buffer;
volatile uint32_t events;
bool initialized,active,have_image,have_planes;
unsigned sequence;
uint32_t capture_started;
constexpr uint32_t error_events=ARM_CPI_EVENT_ERR_HARDWARE |
    ARM_CPI_EVENT_ERR_CAMERA_INPUT_FIFO_OVERRUN | ARM_CPI_EVENT_ERR_CAMERA_OUTPUT_FIFO_OVERRUN |
    ARM_CPI_EVENT_MIPI_CSI2_ERROR;
void callback(uint32_t event) { events|=event; live_camera_events=events; }
}
int camera_stop() {
    int result=Driver_CPI.Stop();
    active=false;
    have_planes=false;
    return result;
}
int camera_init() {
    if (initialized) return active ? -1 : 0;
    int result=Driver_CPI.Initialize(callback);
    if (result!=ARM_DRIVER_OK) return result;
    result=Driver_CPI.PowerControl(ARM_POWER_FULL);
    if (result==ARM_DRIVER_OK) result=Driver_CPI.Control(CPI_CONFIGURE,0);
    if (result==ARM_DRIVER_OK) result=Driver_CPI.Control(CPI_CAMERA_SENSOR_CONFIGURE,0);
    if (result==ARM_DRIVER_OK && (width!=RTE_ISP_OUTPUT_WIDTH || height!=RTE_ISP_OUTPUT_HEIGHT)) {
        isp_output_info output={};
        output.width=width; output.height=height;
        result=Driver_ISP.Control(ISP_CONTROL_SET_OUTPUT,reinterpret_cast<uint32_t>(&output));
    }
    if (result==ARM_DRIVER_OK)
        result=Driver_CPI.Control(CPI_EVENTS_CONFIGURE,ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED | error_events);
    if (result!=ARM_DRIVER_OK) {
        Driver_CPI.PowerControl(ARM_POWER_OFF);
        Driver_CPI.Uninitialize();
        return result;
    }
    buffer={}; buffer.index=0; buffer.numPlanes=3; buffer.imageSize=sizeof(planar);
    for (unsigned plane=0;plane<3;++plane)
        buffer.planes[plane].dmaPhyAddr=reinterpret_cast<vsi_dma_t>(planar+plane*pixels);
    initialized=true;
    return 0;
}
int camera_start() {
    if (!initialized || active) return -1;
    events=0;
    have_planes=false;
    SCB_CleanInvalidateDCache_by_Addr(reinterpret_cast<uint32_t *>(planar),sizeof(planar));
    int result=Driver_ISP.Control(ISP_CONTROL_QBUF,reinterpret_cast<uint32_t>(&buffer));
    if (result!=ARM_DRIVER_OK) return result;
    result=Driver_CPI.CaptureFrame(reinterpret_cast<uint8_t *>(rgb565));
    if (result!=ARM_DRIVER_OK) { Driver_CPI.Stop(); return result; }
    capture_started=DWT->CYCCNT;
    live_camera_reason=0; live_camera_events=events;
    active=true;
    return 0;
}
CaptureState camera_poll() {
    if (!active) { live_camera_reason=1; return CaptureState::Error; }
    live_camera_cycles=static_cast<uint32_t>(DWT->CYCCNT-capture_started);
    if (!(events & ARM_ISP_MI_EVENT_MP_FRAME_END_DETECTED)) {
        if (live_camera_cycles>SystemCoreClock*3U) {
            live_camera_reason=2;
            trace_printf("LIVE_CAMERA_TIMEOUT events=%lx cycles=%lu dma=%p\r\n",
                events,live_camera_cycles,planar);
            camera_stop();
            return CaptureState::Error;
        }
        // Match the confirmed capture path: CSI startup flags do not override a
        // completed ISP frame. A missing frame still fails at the bounded timeout.
        return CaptureState::Pending;
    }
    int result=Driver_ISP.Control(ISP_PROCESS_FRAME_END,0);
    int dequeue=Driver_ISP.Control(ISP_CONTROL_DQBUF,reinterpret_cast<uint32_t>(&buffer));
    int stop=Driver_CPI.Stop();
    active=false;
    live_camera_process_result=result;
    live_camera_dequeue_result=dequeue;
    live_camera_stop_result=stop;
    if (result!=ARM_DRIVER_OK || dequeue!=ARM_DRIVER_OK || stop!=ARM_DRIVER_OK) {
        ++live_camera_control_warnings;
        if (sequence<3 || (sequence+1)%120==0)
            trace_printf("LIVE_CAMERA_FRAME events=%lx process=%d dequeue=%d stop=%d\r\n",
                events,result,dequeue,stop);
    }
    SCB_InvalidateDCache_by_Addr(reinterpret_cast<uint32_t *>(planar),sizeof(planar));
    for (unsigned i=0;i<pixels;++i) {
        unsigned r=planar[i],g=planar[pixels+i],b=planar[pixels*2+i];
        rgb565[pixels-1-i]=((r>>3)<<11)|((g>>2)<<5)|(b>>3);
    }
    have_image=have_planes=true;
    ++sequence;
    return CaptureState::Ready;
}
Image camera_image() { return have_image ? Image{rgb565,int(width),int(height),int(width)} : Image{}; }
CameraPlanes camera_planes() { return have_planes ? CameraPlanes{planar,int(width),int(height),true} : CameraPlanes{}; }
unsigned camera_sequence() { return sequence; }
}
