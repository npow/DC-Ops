// SPDX-License-Identifier: Apache-2.0
#include "runner.h"
#include "live.h"
#include "RTE_Components.h"
#include CMSIS_device_header
#include "ethosu_driver.h"
#include "sys_utils.h"
#include "../../src/trace.h"
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
namespace {
ethosu_driver npu;
alignas(32) uint8_t fast_pool[384*1024] __attribute__((section(".bss.ai_pool")));
alignas(32) uint8_t pal_pool[64*1024];
size_t pal_offset;
}
extern "C" {
alignas(32) uint8_t live_model_staging[2560*1024] __attribute__((section(".bss.ai_pool")));
alignas(32) uint8_t live_temporary_pool[512*1024] __attribute__((section(".bss.ai_pool")));
unsigned char *ethosu_fast_scratch=fast_pool;
size_t ethosu_fast_scratch_size=sizeof(fast_pool);
int live_npu_initialize(void) {
    executorch::runtime::runtime_init();
    return ethosu_init(&npu,reinterpret_cast<void *>(NPU_HG_BASE),nullptr,0,1,1);
}
uint64_t ethosu_address_remap(uint64_t address,int) {
    return LocalToGlobal(reinterpret_cast<void *>(static_cast<uintptr_t>(address)));
}
unsigned int ethosu_config_select(uint64_t address,int) { return (address&0xa0000000ULL) ? 2 : 0; }
struct semaphore { volatile uint8_t count; };
int ethosu_semaphore_take(void *sem,uint64_t) {
    auto *s=static_cast<semaphore *>(sem);
    const uint32_t started=live_now_ms();
    while (!s->count) {
        if (live_now_ms()-started>10000) return -1;
        if (*reinterpret_cast<volatile uint32_t *>(NPU_HG_BASE+4)&2U) ethosu_irq_handler(&npu);
        // Video continues while the NPU runs; service has a reentrancy guard.
        live_service();
    }
    --s->count;
    return 0;
}
void NPU_HP_IRQHandler(void) {}
void et_pal_init(void) {}
ET_NORETURN void et_pal_abort(void) {
    trace_printf("LIVE_RUNTIME_ABORT\r\n");
    for (;;) { live_service(); __NOP(); }
}
et_timestamp_t et_pal_current_ticks(void) { return DWT->CYCCNT; }
et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) { return {5,2}; }
void et_pal_emit_log_message(et_timestamp_t,et_pal_log_level_t level,
    const char *,const char *,size_t,const char *message,size_t length) {
    trace_printf("ET:%c %.*s\r\n",static_cast<char>(level),int(length),message);
}
void *et_pal_allocate(size_t bytes) {
    size_t offset=(pal_offset+15)&~size_t(15);
    if (offset>sizeof(pal_pool) || bytes>sizeof(pal_pool)-offset) return nullptr;
    pal_offset=offset+bytes;
    return pal_pool+offset;
}
void et_pal_free(void *) {}
}
