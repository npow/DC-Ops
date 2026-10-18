// SPDX-License-Identifier: Apache-2.0
#include "benchmark_service.h"
#include "runner.h"
#include "live.h"
#include "../multi_npu/multi_runtime.h"
#include "../../artifacts/translator/spatial_staging/pipeline_index.h"
#include "RTE_Components.h"
#include CMSIS_device_header
#include "../../src/trace.h"
#include <algorithm>
#include <cmath>
#include <cstring>
namespace {
constexpr uintptr_t cpu_model=0xa3400000u,input_address=0xa3900000u,mask_address=0xa3910000u;
constexpr uintptr_t cpu_output=0x02210000u,npu_output=0x02218000u;
constexpr uintptr_t planned_address=0x025a0000u,method_address=0x02190000u;
constexpr size_t cpu_model_bytes=4219536,input_bytes=32768,mask_bytes=128;
bool running;
void invalidate(uintptr_t address,size_t bytes) {
    SCB_InvalidateDCache_by_Addr(reinterpret_cast<void *>(address&~uintptr_t(31)),(bytes+31)&~size_t(31)); __DSB();
}
uint32_t fixture_hash() {
    uint32_t hash=2166136261u;
    const uintptr_t addresses[2]={input_address,mask_address};
    const size_t bytes[2]={input_bytes,mask_bytes};
    for (unsigned buffer=0;buffer<2;++buffer) {
        const auto *data=reinterpret_cast<const uint8_t *>(addresses[buffer]);
        for (size_t i=0;i<bytes[buffer];++i) hash=(hash^data[i])*16777619u;
    }
    return hash;
}
int fail(int error) {
    live_benchmark_error=static_cast<uint32_t>(error);
    live_benchmark_status=0xe0;
    running=false;
    return error;
}
void compare_outputs() {
    live_benchmark_compare_valid=0;
    live_benchmark_nonfinite=0;
    if (!live_benchmark_cpu_runs || !live_benchmark_npu_runs ||
        live_benchmark_cpu_hash!=live_benchmark_npu_hash) return;
    const auto *cpu=reinterpret_cast<const float *>(cpu_output);
    const auto *npu=reinterpret_cast<const float *>(npu_output);
    double absolute=0,squared=0;
    float maximum=0;
    for (size_t i=0;i<input_bytes/sizeof(float);++i) {
        if (!std::isfinite(cpu[i]) || !std::isfinite(npu[i])) { ++live_benchmark_nonfinite; continue; }
        const float difference=std::fabs(cpu[i]-npu[i]);
        absolute+=difference; squared+=double(difference)*difference;
        maximum=std::max(maximum,difference);
    }
    if (live_benchmark_nonfinite) return;
    live_benchmark_mean_abs=absolute/(input_bytes/sizeof(float));
    live_benchmark_max_abs=maximum;
    live_benchmark_rmse=std::sqrt(squared/(input_bytes/sizeof(float)));
    live_benchmark_stage_speedup=live_benchmark_npu_ms ? float(live_benchmark_cpu_ms)/live_benchmark_npu_ms : 0;
    live_benchmark_compare_valid=1;
}
}
extern "C" {
volatile uint32_t live_benchmark_enabled,live_benchmark_command;
volatile uint32_t live_benchmark_status,live_benchmark_error,live_benchmark_target;
volatile uint32_t live_benchmark_ms,live_benchmark_prepare_ms;
volatile uint64_t live_benchmark_cycles;
volatile uint32_t live_benchmark_output_address,live_benchmark_output_bytes;
volatile uint64_t live_benchmark_pmu_cycles,live_benchmark_pmu_active,live_benchmark_pmu_mac,live_benchmark_pmu_idle;
volatile uint32_t live_benchmark_cpu_ms,live_benchmark_npu_ms,live_benchmark_cpu_runs,live_benchmark_npu_runs;
volatile uint32_t live_benchmark_cpu_hash,live_benchmark_npu_hash,live_benchmark_compare_valid,live_benchmark_nonfinite;
volatile float live_benchmark_mean_abs,live_benchmark_max_abs,live_benchmark_rmse,live_benchmark_stage_speedup;

int live_benchmark_service(void) {
    if (!live_benchmark_enabled || !live_benchmark_command || running) return 0;
    const uint32_t target=live_benchmark_command;
    live_benchmark_command=0;
    running=true;
    live_benchmark_target=target;
    live_benchmark_error=live_benchmark_ms=live_benchmark_prepare_ms=0;
    live_benchmark_compare_valid=0; live_benchmark_stage_speedup=0;
    live_benchmark_output_address=live_benchmark_output_bytes=0;
    live_benchmark_cycles=live_benchmark_pmu_cycles=live_benchmark_pmu_active=live_benchmark_pmu_mac=live_benchmark_pmu_idle=0;
    if (target!=1 && target!=2) return fail(-1);
    live_benchmark_status=1;
    const uint32_t prepare_started=live_now_ms();
    invalidate(input_address,input_bytes); invalidate(mask_address,mask_bytes);
    const uint32_t hash=fixture_hash();
    const void *program=reinterpret_cast<const void *>(cpu_model);
    size_t program_bytes=cpu_model_bytes;
    if (target==2) {
        const auto &stage=dcops_packed_stages[0];
        if (stage.bytes>sizeof(live_model_staging) || stage.input_count!=2 || stage.output_elements!=8192) return fail(-2);
        const uintptr_t source=0xa1300000u+stage.offset;
        invalidate(source,stage.bytes);
        for (size_t offset=0;offset<stage.bytes;offset+=16*1024) {
            const size_t bytes=std::min(size_t(16*1024),size_t(stage.bytes)-offset);
            std::memcpy(live_model_staging+offset,reinterpret_cast<const void *>(source+offset),bytes);
            if (live_service()<0) return fail(-3);
        }
        SCB_CleanDCache_by_Addr(live_model_staging,(stage.bytes+31)&~uint32_t(31)); __DSB();
        program=live_model_staging; program_bytes=stage.bytes;
    } else invalidate(cpu_model,cpu_model_bytes);
    live_benchmark_status=2;
    dcops::Runner runner;
    int error=runner.load(program,program_bytes,
        {reinterpret_cast<uint8_t *>(method_address),128*1024,
         reinterpret_cast<uint8_t *>(planned_address),512*1024,
         live_temporary_pool,sizeof(live_temporary_pool)});
    if (error) return fail(-1000-error);
    if (live_service()<0) return fail(-4);
    live_benchmark_prepare_ms=live_now_ms()-prepare_started;
    const dcops::RunnerInput inputs[2]={{reinterpret_cast<void *>(input_address),input_bytes},
                                       {reinterpret_cast<void *>(mask_address),mask_bytes}};
    e8_npu_stats before{},after{};
    live_npu_stats(1,&before);
    live_benchmark_status=3;
    const uint32_t began_ms=live_now_ms(),began_cycles=DWT->CYCCNT;
    error=runner.execute(inputs,2);
    const uint32_t elapsed_cycles=DWT->CYCCNT-began_cycles;
    live_benchmark_ms=live_now_ms()-began_ms;
    // Recover DWT wraps using the independently interrupt-driven millisecond clock.
    const uint64_t expected=uint64_t(live_benchmark_ms)*SystemCoreClock/1000u;
    const uint64_t wraps=expected>elapsed_cycles ? (expected-elapsed_cycles+(1ull<<31))>>32 : 0;
    live_benchmark_cycles=elapsed_cycles+(wraps<<32);
    live_npu_stats(1,&after);
    live_benchmark_pmu_cycles=after.total_cycles-before.total_cycles;
    live_benchmark_pmu_active=after.active_cycles-before.active_cycles;
    live_benchmark_pmu_mac=after.mac_cycles-before.mac_cycles;
    live_benchmark_pmu_idle=after.idle_cycles-before.idle_cycles;
    if (error) return fail(-2000-error);
    auto result=runner.output();
    if (!result.data || result.bytes!=input_bytes) return fail(-5);
    live_benchmark_status=4;
    const uintptr_t output=target==1 ? cpu_output : npu_output;
    std::memcpy(reinterpret_cast<void *>(output),result.data,result.bytes);
    SCB_CleanDCache_by_Addr(reinterpret_cast<void *>(output),result.bytes); __DSB();
    live_benchmark_output_address=output; live_benchmark_output_bytes=result.bytes;
    if (target==1) { live_benchmark_cpu_ms=live_benchmark_ms; live_benchmark_cpu_hash=hash; ++live_benchmark_cpu_runs; }
    else { live_benchmark_npu_ms=live_benchmark_ms; live_benchmark_npu_hash=hash; ++live_benchmark_npu_runs; }
    compare_outputs();
    live_benchmark_status=5;
    trace_printf("LIVE_BENCHMARK target=%u ms=%u prepare_ms=%u output=%08lx bytes=%u\r\n",
        unsigned(target),unsigned(live_benchmark_ms),unsigned(live_benchmark_prepare_ms),
        static_cast<unsigned long>(output),unsigned(result.bytes));
    live_service();
    running=false;
    return 1;
}
}
