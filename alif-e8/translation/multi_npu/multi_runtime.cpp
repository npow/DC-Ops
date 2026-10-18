// SPDX-License-Identifier: Apache-2.0
#include "mixed_driver.h"
#include "multi_runtime.h"
#include "../live/runner.h"
#include "../live/live.h"
#include "RTE_Components.h"
#include CMSIS_device_header
#include "sys_utils.h"
#include "pmu_ethosu.h"
#include "../../src/trace.h"
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
namespace {
ethosu_driver npus[2];
e8_npu_stats stats[2];
uint32_t started[2];
bool ready[2];
alignas(32) uint8_t fast_pool[384*1024] __attribute__((section(".bss.ai_pool")));
alignas(32) uint8_t pal_pool[64*1024];
size_t pal_offset;
int index_of(const ethosu_driver *driver) {
    if (driver==&npus[0]) return 0;
    return driver==&npus[1] ? 1 : -1;
}
}
extern "C" {
alignas(32) uint8_t live_model_staging[2560*1024] __attribute__((section(".bss.ai_pool")));
alignas(32) uint8_t live_temporary_pool[512*1024] __attribute__((section(".bss.ai_pool")));
unsigned char *ethosu_fast_scratch=fast_pool;
size_t ethosu_fast_scratch_size=sizeof(fast_pool);
const uint32_t live_u55_diag_offsets[LIVE_U55_DIAG_COUNT]={
    0x00,0x04,0x08,0x10,0x18,0x1c,0x20,0x24,0x28,0x3c,
    0x40,0x44,0x48,0x4c,0x80,0x88,0x90,0x100,0x104,0x108,0x110,0x114,0x148};
volatile uint32_t live_u55_diag_registers[LIVE_U55_DIAG_COUNT];
volatile uint32_t live_u55_diag_sequence;
const uint32_t live_u85_diag_offsets[LIVE_U85_DIAG_COUNT]={
    0x00,0x04,0x08,0x10,0x18,0x1c,0x20,0x24,0x28,0x30,0x38,0x3c,
    0x40,0x44,0x48,0x4c,0x50,0x54,0x60,0x64,0x80,0x88,0x90,
    0x240,0x250,0x258,0x260,0x268,0x26c,0x274,0x27c,0x280,0x2b8,0x2bc};
volatile uint32_t live_u85_diag_registers[LIVE_U85_DIAG_COUNT];
volatile uint32_t live_u85_diag_sequence;
volatile uint32_t live_u85_diag_regions[6];
int live_npu_fast_scratch_enabled(void) {
    return ethosu_fast_scratch==fast_pool && ethosu_fast_scratch_size==sizeof(fast_pool);
}
int live_npu_set_fast_scratch_enabled(int enabled) {
    if (enabled!=0 && enabled!=1) return -1;
    for (unsigned i=0;i<2;++i)
        if (ready[i] && npus[i].job.state!=ETHOSU_JOB_IDLE) return -2;
    ethosu_fast_scratch=enabled ? fast_pool : nullptr;
    ethosu_fast_scratch_size=enabled ? sizeof(fast_pool) : 0;
    __DSB();
    return 0;
}
void *live_npu_fast_scratch_storage(size_t *bytes) {
    if (bytes) *bytes=sizeof(fast_pool);
    return fast_pool;
}
int live_npu_initialize(void) {
    executorch::runtime::runtime_init();
    const uintptr_t bases[2]={NPU_HP_BASE,NPU_HG_BASE};
    for (unsigned i=0;i<2;++i) {
        int error=ethosu_init(&npus[i],reinterpret_cast<void *>(bases[i]),nullptr,0,1,1);
        if (error) return -int(10+i);
        ready[i]=true;
        error=e8_register_npu(&npus[i]);
        if (error) return -int(20+i);
        ethosu_hw_info info{};
        ethosu_get_hw_info(&npus[i],&info);
        trace_printf("LIVE_NPU index=%u base=%08lx product=%u macs=%u\r\n",i,
            static_cast<unsigned long>(bases[i]),info.version.product_major,1u<<info.cfg.macs_per_cc);
    }
    return 0;
}
uint64_t ethosu_address_remap(uint64_t address,int) {
    if (!address) return 0;
    return LocalToGlobal(reinterpret_cast<void *>(static_cast<uintptr_t>(address)));
}
// U55 AXI1 is read-only. Keep writable scratch on AXI0 even in PSRAM;
// profiles 0/1 match the Alif BSP for scratch and command/weight access.
unsigned int u55_ethosu_config_select(uint64_t,int index) { return index==1 ? 0 : 1; }
unsigned int u85_ethosu_config_select(uint64_t address,int) { return (address&0xa0000000ULL) ? 2 : 0; }
void live_npu_poll(void) {
    const uintptr_t bases[2]={NPU_HP_BASE,NPU_HG_BASE};
    for (unsigned i=0;i<2;++i)
        if (ready[i] && (*reinterpret_cast<volatile uint32_t *>(bases[i]+4)&2U)) ethosu_irq_handler(&npus[i]);
}
struct semaphore { volatile uint8_t count; };
int ethosu_semaphore_take(void *sem,uint64_t) {
    auto *s=static_cast<semaphore *>(sem);
    const uint32_t begin=live_now_ms();
    while (!s->count) {
        if (live_now_ms()-begin>10000) return -1;
        live_npu_poll();
        if (!s->count) live_service();
    }
    --s->count;
    return 0;
}
void NPU_HP_IRQHandler(void) { if (ready[0]) ethosu_irq_handler(&npus[0]); }
void IRQ366_Handler(void) { if (ready[1]) ethosu_irq_handler(&npus[1]); }
void ethosu_inference_begin(ethosu_driver *driver,void *) {
    const int index=index_of(driver);
    if (index<0) return;
    started[index]=live_now_ms();
    ETHOSU_PMU_Enable(driver);
    ETHOSU_PMU_CNTR_Disable(driver,ETHOSU_PMU_CCNT_Msk|7u);
    ETHOSU_PMU_CYCCNT_Reset(driver);
    ETHOSU_PMU_EVCNTR_ALL_Reset(driver);
    ETHOSU_PMU_Set_EVTYPER(driver,0,ETHOSU_PMU_NPU_ACTIVE);
    ETHOSU_PMU_Set_EVTYPER(driver,1,ETHOSU_PMU_MAC_ACTIVE);
    ETHOSU_PMU_Set_EVTYPER(driver,2,ETHOSU_PMU_NPU_IDLE);
    ETHOSU_PMU_CNTR_Enable(driver,ETHOSU_PMU_CCNT_Msk|7u);
}
void ethosu_inference_end(ethosu_driver *driver,void *) {
    const int index=index_of(driver);
    if (index<0) return;
    if (index==0) {
        // Preserve the final state before the driver's timeout path resets it.
        for (unsigned i=0;i<LIVE_U55_DIAG_COUNT;++i)
            live_u55_diag_registers[i]=*reinterpret_cast<volatile uint32_t *>(NPU_HP_BASE+live_u55_diag_offsets[i]);
        ++live_u55_diag_sequence;
    }
    if (index==1) {
        for (unsigned i=0;i<LIVE_U85_DIAG_COUNT;++i)
            live_u85_diag_registers[i]=*reinterpret_cast<volatile uint32_t *>(NPU_HG_BASE+live_u85_diag_offsets[i]);
        for (unsigned i=0;i<3 && i<unsigned(driver->job.num_base_addr);++i) {
            live_u85_diag_regions[i]=driver->job.base_addr[i];
            live_u85_diag_regions[i+3]=driver->job.base_addr_size[i];
        }
        ++live_u85_diag_sequence;
    }
    auto &sample=stats[index];
    sample.last_ms=live_now_ms()-started[index];
    sample.elapsed_ms+=sample.last_ms;
    ++sample.invocations;
    if (driver->job.result!=ETHOSU_JOB_RESULT_OK) ++sample.failures;
    ETHOSU_PMU_CNTR_Disable(driver,ETHOSU_PMU_CCNT_Msk|7u);
    sample.total_cycles+=ETHOSU_PMU_Get_CCNTR(driver);
    sample.active_cycles+=ETHOSU_PMU_Get_EVCNTR(driver,0);
    sample.mac_cycles+=ETHOSU_PMU_Get_EVCNTR(driver,1);
    sample.idle_cycles+=ETHOSU_PMU_Get_EVCNTR(driver,2);
    ETHOSU_PMU_Disable(driver);
}
int live_npu_stats(unsigned index,e8_npu_stats *destination) {
    if (index>=2 || !destination) return -1;
    *destination=stats[index];
    return 0;
}
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
    const size_t offset=(pal_offset+15)&~size_t(15);
    if (offset>sizeof(pal_pool) || bytes>sizeof(pal_pool)-offset) return nullptr;
    pal_offset=offset+bytes;
    return pal_pool+offset;
}
void et_pal_free(void *) {}
}
