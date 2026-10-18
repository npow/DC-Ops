// SPDX-License-Identifier: Apache-2.0
#include "worker_mailbox.h"
#include "../live/runner.h"
#include "RTE_Components.h"
#include CMSIS_device_header
#include "ethosu_driver.h"
#include "pmu_ethosu.h"
#include "sys_utils.h"
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <cstring>
namespace {
ethosu_driver npu;
dcops::Runner runner;
alignas(32) uint8_t method_pool[64*1024];
alignas(32) uint8_t pal_pool[16*1024];
alignas(32) uint8_t planned_pool[256*1024] __attribute__((section(".he_worker")));
alignas(32) uint8_t temporary_pool[768*1024] __attribute__((section(".he_worker")));
size_t pal_offset;
bool model_loaded;
bool npu_ready;
uint64_t npu_cycles;
uint32_t active_cycles,mac_cycles,idle_cycles,begin_ms;
void invalidate(const void *pointer,size_t bytes) {
    if (!bytes) return;
    const uintptr_t first=reinterpret_cast<uintptr_t>(pointer)&~uintptr_t(31);
    const uintptr_t end=(reinterpret_cast<uintptr_t>(pointer)+bytes+31)&~uintptr_t(31);
    SCB_InvalidateDCache_by_Addr(reinterpret_cast<void *>(first),end-first); __DSB();
}
void clean(const void *pointer,size_t bytes) {
    if (!bytes) return;
    const uintptr_t first=reinterpret_cast<uintptr_t>(pointer)&~uintptr_t(31);
    const uintptr_t end=(reinterpret_cast<uintptr_t>(pointer)+bytes+31)&~uintptr_t(31);
    SCB_CleanDCache_by_Addr(reinterpret_cast<void *>(first),end-first); __DSB();
}
}
extern "C" {
volatile uint32_t he_worker_status,he_worker_error,he_worker_milliseconds,he_worker_requests;
e8_mailbox he_worker_mailbox __attribute__((section(".he_mailbox"),aligned(128)));
unsigned char *ethosu_fast_scratch=nullptr;
size_t ethosu_fast_scratch_size=0;
void SysTick_Handler(void) { ++he_worker_milliseconds; }
uint32_t live_now_ms(void) { return he_worker_milliseconds; }
int live_service(void) { return 0; }
uint64_t ethosu_address_remap(uint64_t address,int) {
    return LocalToGlobal(reinterpret_cast<void *>(static_cast<uintptr_t>(address)));
}
unsigned int ethosu_config_select(uint64_t address,int) { return (address&0xa0000000ULL) ? 2 : 0; }
void ethosu_flush_dcache(const uint64_t *address,const size_t *bytes,int count) {
    for (int i=0;i<count;++i) if (address[i] && bytes[i]) clean(reinterpret_cast<void *>(static_cast<uintptr_t>(address[i])),bytes[i]);
}
void ethosu_invalidate_dcache(const uint64_t *address,const size_t *bytes,int count) {
    for (int i=0;i<count;++i) if (address[i] && bytes[i]) invalidate(reinterpret_cast<void *>(static_cast<uintptr_t>(address[i])),bytes[i]);
}
struct semaphore { volatile uint8_t count; };
int ethosu_semaphore_take(void *handle,uint64_t) {
    auto *sem=static_cast<semaphore *>(handle);
    const uint32_t start=live_now_ms();
    while (!sem->count) {
        if (live_now_ms()-start>10000) return -1;
        if (npu_ready && (*reinterpret_cast<volatile uint32_t *>(NPU_HE_BASE+4)&2u)) ethosu_irq_handler(&npu);
    }
    --sem->count;
    return 0;
}
void NPU_HE_IRQHandler(void) { if (npu_ready) ethosu_irq_handler(&npu); }
void ethosu_inference_begin(ethosu_driver *driver,void *) {
    begin_ms=live_now_ms();
    ETHOSU_PMU_Enable(driver);
    ETHOSU_PMU_CNTR_Disable(driver,ETHOSU_PMU_CCNT_Msk|7u);
    ETHOSU_PMU_CYCCNT_Reset(driver); ETHOSU_PMU_EVCNTR_ALL_Reset(driver);
    ETHOSU_PMU_Set_EVTYPER(driver,0,ETHOSU_PMU_NPU_ACTIVE);
    ETHOSU_PMU_Set_EVTYPER(driver,1,ETHOSU_PMU_MAC_ACTIVE);
    ETHOSU_PMU_Set_EVTYPER(driver,2,ETHOSU_PMU_NPU_IDLE);
    ETHOSU_PMU_CNTR_Enable(driver,ETHOSU_PMU_CCNT_Msk|7u);
}
void ethosu_inference_end(ethosu_driver *driver,void *) {
    ETHOSU_PMU_CNTR_Disable(driver,ETHOSU_PMU_CCNT_Msk|7u);
    npu_cycles=ETHOSU_PMU_Get_CCNTR(driver);
    active_cycles=ETHOSU_PMU_Get_EVCNTR(driver,0);
    mac_cycles=ETHOSU_PMU_Get_EVCNTR(driver,1);
    idle_cycles=ETHOSU_PMU_Get_EVCNTR(driver,2);
    ETHOSU_PMU_Disable(driver);
}
void et_pal_init(void) {}
ET_NORETURN void et_pal_abort(void) { he_worker_status=0xe0; for (;;) __WFI(); }
et_timestamp_t et_pal_current_ticks(void) { return DWT->CYCCNT; }
et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) { return {1000000000u,SystemCoreClock}; }
void et_pal_emit_log_message(et_timestamp_t,et_pal_log_level_t,const char *,const char *,size_t,const char *,size_t) {}
void *et_pal_allocate(size_t bytes) {
    const size_t offset=(pal_offset+15)&~size_t(15);
    if (offset>sizeof(pal_pool) || bytes>sizeof(pal_pool)-offset) return nullptr;
    pal_offset=offset+bytes; return pal_pool+offset;
}
void et_pal_free(void *) {}
int stdout_putchar(int character) { return character; }
int stderr_putchar(int character) { return character; }
int stdin_getchar(void) { return -1; }
}
namespace {
int execute(const e8_request *request,e8_response *response) {
    if (request->model!=1 || request->input_count!=1) return -10;
    if (!model_loaded) {
        constexpr uintptr_t model_address=0xa2500000u;
        constexpr size_t model_bytes=2198384;
        invalidate(reinterpret_cast<void *>(model_address),model_bytes);
        int error=runner.load(reinterpret_cast<void *>(model_address),model_bytes,
            {method_pool,sizeof(method_pool),planned_pool,sizeof(planned_pool),temporary_pool,sizeof(temporary_pool)});
        if (error) { he_worker_error=error; return error; }
        model_loaded=true;
    }
    dcops::RunnerInput input{reinterpret_cast<void *>(static_cast<uintptr_t>(request->inputs[0].address)),request->inputs[0].bytes};
    const uint32_t start=live_now_ms();
    int error=runner.execute(&input,1);
    response->elapsed_us=(live_now_ms()-start)*1000u;
    response->npu_cycles=npu_cycles;
    response->npu_active=active_cycles; response->mac_active=mac_cycles; response->npu_idle=idle_cycles;
    if (error) { he_worker_error=error; return error; }
    auto output=runner.output();
    if (!output.data || output.bytes>request->output.bytes) return -11;
    std::memcpy(reinterpret_cast<void *>(static_cast<uintptr_t>(request->output.address)),output.data,output.bytes);
    response->output_bytes=output.bytes;
    ++he_worker_requests;
    return 0;
}
}
int main(void) {
    he_worker_status=1;
    CoreDebug->DEMCR|=CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT=0; DWT->CTRL|=DWT_CTRL_CYCCNTENA_Msk;
    SystemCoreClockUpdate();
    SysTick_Config(SystemCoreClock/1000u);
    std::memset(&he_worker_mailbox,0,sizeof(he_worker_mailbox));
    clean(&he_worker_mailbox,sizeof(he_worker_mailbox));
    executorch::runtime::runtime_init();
    int error=ethosu_init(&npu,reinterpret_cast<void *>(NPU_HE_BASE),nullptr,0,1,1);
    if (error) { he_worker_error=error; he_worker_status=0xe1; for (;;) __WFI(); }
    npu_ready=true;
    he_worker_status=2;
    const e8_worker_ops ops{invalidate,clean,execute};
    for (;;) e8_worker_poll(&he_worker_mailbox,&ops);
}
