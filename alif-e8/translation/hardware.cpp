// SPDX-License-Identifier: Apache-2.0
#include "RTE_Components.h"
#include CMSIS_device_header
#include "board_defs.h"
#include "board_config.h"
#include "APS512XXN_PSRAM.h"
#include "ospi_psram_xip.h"
#include "sys_utils.h"
#include "ethosu_driver.h"
#include "pmu_ethosu.h"
#include "../src/trace.h"
#include <cstdio>
#include <cstring>
#include <optional>
#include <utility>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#define printf trace_printf
using namespace executorch::runtime;
extern "C" int board_display_status(const char *message);

static int32_t initialize_psram(OSPI_Type *ospi, AES_Type *aes) {
    ospi_disable(ospi);
    // The pack driver ORs the transfer threshold, retaining the previous transfer's length.
    ospi->OSPI_TXFTLR = 0;
    return aps512xxn_psram_init(ospi, aes);
}

extern "C" {
volatile uint32_t tr_status, tr_command, tr_model_index, tr_error;
volatile uint32_t tr_model_address[2], tr_model_size[2];
volatile uint32_t tr_input_address[4], tr_input_bytes[4], tr_output_address, tr_output_bytes;
volatile uint32_t tr_execute_cycles, tr_pmu_active, tr_pmu_mac, tr_pmu_idle;
volatile uint32_t tr_milliseconds, tr_execute_ms;
volatile uint64_t tr_pmu_cycles;
volatile uint32_t tr_npu_product, tr_npu_macs;
}

namespace {
constexpr size_t kMethodBytes = 128 * 1024;
constexpr size_t kPlannedBytes = 512 * 1024;
constexpr size_t kTempBytes = 512 * 1024;
alignas(32) uint8_t method_pool[2][kMethodBytes];
alignas(32) uint8_t pal_pool[64 * 1024];
__attribute__((section(".bss.ai_pool"), aligned(32))) uint8_t fast_pool[393216];
__attribute__((section(".bss.ai_pool"), aligned(32))) uint8_t temp_pool[kTempBytes];
size_t pal_offset;
struct ethosu_driver npu;

class Loader : public DataLoader {
public:
    const uint8_t *data = nullptr;
    size_t length = 0;
    Result<FreeableBuffer> load(size_t offset, size_t size, const SegmentInfo&) const override {
        if (offset > length || size > length - offset) return Error::InvalidArgument;
        return FreeableBuffer(data + offset, size, nullptr);
    }
    Result<size_t> size() const override { return length; }
};

struct Runner {
    Loader loader;
    std::optional<Program> program;
    std::optional<Method> method;
    std::optional<MemoryAllocator> allocator;
    std::optional<MemoryAllocator> temp;
    std::optional<HierarchicalAllocator> planned;
    std::optional<MemoryManager> manager;
    Span<uint8_t> spans[8];
};
Runner runners[2];

int load_model(unsigned index) {
    if (index >= 2) return -1;
    Runner &r = runners[index];
    r.method.reset();
    r.program.reset();
    r.loader.data = reinterpret_cast<uint8_t *>(tr_model_address[index]);
    r.loader.length = tr_model_size[index];
    auto p = Program::load(&r.loader, Program::Verification::Minimal);
    if (!p.ok()) return (int)p.error();
    r.program.emplace(std::move(p.get()));
    auto metadata = r.program->method_meta("forward");
    if (!metadata.ok()) return (int)metadata.error();
    auto meta = metadata.get();
    if (meta.num_memory_planned_buffers() > 8) return -2;
    size_t offset = 0;
    for (size_t i = 0; i < meta.num_memory_planned_buffers(); ++i) {
        auto size = meta.memory_planned_buffer_size(i);
        if (!size.ok()) return -3;
        offset = (offset+31) & ~size_t(31);
        if (offset+size.get() > kPlannedBytes) return -4;
        r.spans[i] = Span<uint8_t>(reinterpret_cast<uint8_t *>(0xa3800000 + index*kPlannedBytes + offset), size.get());
        offset += size.get();
    }
    r.allocator.emplace(kMethodBytes, method_pool[index]);
    r.temp.emplace(kTempBytes, temp_pool);
    r.planned.emplace(Span<Span<uint8_t>>(r.spans, meta.num_memory_planned_buffers()));
    r.manager.emplace(&*r.allocator, &*r.planned, &*r.temp);
    auto method = r.program->load_method("forward", &*r.manager);
    if (!method.ok()) return (int)method.error();
    r.method.emplace(std::move(method.get()));
    printf("TRANSLATOR_LOADED model=%u planned=%u inputs=%u\r\n", index, (unsigned)offset, (unsigned)meta.num_inputs());
    return 0;
}

int execute(unsigned index) {
    using namespace executorch::aten;
    if (index >= 2 || !runners[index].method) return -1;
    Runner &r = runners[index];
    auto metadata = r.program->method_meta("forward");
    if (!metadata.ok()) return -2;
    auto meta = metadata.get();
    if (meta.num_inputs() > 4) return -3;
    TensorImpl::SizesType shapes[4][8];
    TensorImpl::DimOrderType orders[4][8];
    TensorImpl::StridesType strides[4][8];
    std::optional<TensorImpl> impls[4];
    std::optional<Tensor> tensors[4];
    for (size_t i = 0; i < meta.num_inputs(); ++i) {
        auto info = meta.input_tensor_meta(i);
        if (!info.ok() || info->sizes().size() > 8 || info->nbytes() != tr_input_bytes[i]) return -4;
        const auto sizes = info->sizes();
        int step = 1;
        for (int d = (int)sizes.size()-1; d >= 0; --d) {
            shapes[i][d] = sizes[d];
            orders[i][d] = info->dim_order()[d];
            strides[i][d] = step;
            step *= sizes[d];
        }
        impls[i].emplace(info->scalar_type(), sizes.size(), shapes[i],
            reinterpret_cast<void *>(tr_input_address[i]), orders[i], strides[i]);
        tensors[i].emplace(&*impls[i]);
        auto status = r.method->set_input(EValue(*tensors[i]), i);
        if (status != Error::Ok) return (int)status;
    }
    uint32_t start = DWT->CYCCNT, start_ms = tr_milliseconds;
    auto status = r.method->execute();
    tr_execute_cycles = DWT->CYCCNT-start;
    tr_execute_ms = tr_milliseconds-start_ms;
    if (status != Error::Ok) return (int)status;
    auto output = r.method->get_output(0).toTensor();
    tr_output_address = reinterpret_cast<uintptr_t>(output.const_data_ptr());
    tr_output_bytes = output.nbytes();
    SCB_CleanDCache_by_Addr((uint32_t *)tr_output_address, tr_output_bytes);
    printf("TRANSLATOR_RUN model=%u cycles=%u pmu=%llu active=%u mac=%u bytes=%u\r\n", index,
        tr_execute_cycles, tr_pmu_cycles, tr_pmu_active, tr_pmu_mac, tr_output_bytes);
    return 0;
}
}

extern "C" {
void ethosu_inference_begin(struct ethosu_driver *driver, void *) {
    ETHOSU_PMU_Enable(driver);
    ETHOSU_PMU_Set_EVTYPER(driver, 0, ETHOSU_PMU_NPU_ACTIVE);
    ETHOSU_PMU_Set_EVTYPER(driver, 1, ETHOSU_PMU_MAC_ACTIVE);
    ETHOSU_PMU_Set_EVTYPER(driver, 2, ETHOSU_PMU_NPU_IDLE);
    ETHOSU_PMU_CYCCNT_Reset(driver);
    ETHOSU_PMU_EVCNTR_ALL_Reset(driver);
    ETHOSU_PMU_CNTR_Enable(driver, ETHOSU_PMU_CCNT_Msk | 7);
}
void ethosu_inference_end(struct ethosu_driver *driver, void *) {
    tr_pmu_cycles = ETHOSU_PMU_Get_CCNTR(driver);
    tr_pmu_active = ETHOSU_PMU_Get_EVCNTR(driver, 0);
    tr_pmu_mac = ETHOSU_PMU_Get_EVCNTR(driver, 1);
    tr_pmu_idle = ETHOSU_PMU_Get_EVCNTR(driver, 2);
}
uint64_t ethosu_address_remap(uint64_t address, int) {
    return LocalToGlobal(reinterpret_cast<void *>(static_cast<uintptr_t>(address)));
}
unsigned int ethosu_config_select(uint64_t address, int) {
    return (address & 0xa0000000ULL) ? 2 : 0;
}
unsigned char *ethosu_fast_scratch = fast_pool;
size_t ethosu_fast_scratch_size = sizeof(fast_pool);
struct semaphore { volatile uint8_t count; };
int ethosu_semaphore_take(void *sem, uint64_t) {
    auto *s = static_cast<semaphore *>(sem);
    uint32_t started = tr_milliseconds;
    while (!s->count) {
        if (tr_milliseconds-started > 10000) return -1;
        if (*(volatile uint32_t *)(NPU_HG_BASE+4) & 2U) ethosu_irq_handler(&npu);
        __NOP();
    }
    --s->count;
    return 0;
}
void NPU_HP_IRQHandler(void) {}
void SysTick_Handler(void) { ++tr_milliseconds; }
void et_pal_init(void) {}
ET_NORETURN void et_pal_abort(void) {
    tr_status = 0xff;
    printf("TRANSLATOR_ABORT\r\n");
    for (;;) __WFI();
}
et_timestamp_t et_pal_current_ticks(void) { return DWT->CYCCNT; }
et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) { return {5,2}; }
void et_pal_emit_log_message(et_timestamp_t, et_pal_log_level_t level,
    const char *, const char *, size_t, const char *message, size_t length) {
    printf("ET:%c %.*s\r\n", static_cast<char>(level), (int)length, message);
}
void *et_pal_allocate(size_t size) {
    size_t aligned = (pal_offset+15) & ~size_t(15);
    if (aligned+size > sizeof(pal_pool)) return nullptr;
    pal_offset = aligned+size;
    return pal_pool+aligned;
}
void et_pal_free(void *) {}

int app_main(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    DWT->CYCCNT = 0;
    SysTick_Config(SystemCoreClock / 1000);
    tr_status = 1;
    printf("TRANSLATOR_BOOT\r\n");
    board_clocks_config(CLKEN_SYSPLL_MASK | CLKEN_HFOSC_MASK | CLKEN_CLK_100M_MASK | (1U << CLKEN_CLK_400M));
    ospi_psram_xip_config ram = {};
    ram.instance = static_cast<OSPI_INSTANCE>(BOARD_PSRAM_OSPI_INSTANCE);
    ram.ram_init = initialize_psram;
    ram.ram_type = RAM_TYPE_PSRAM;
    ram.config_mode = CONFIG_MODE_USER_PARAMETERS;
    ram.bus_speed = 25000000;
    ram.spi_frf = OSPI_SPI_FRF_DUAL_OCTAL;
    ram.wait_cycles = 4;
    ram.ddr_drive_edge = 1;
    ram.rxds_delay = 6;
    ram.signal_delay = 22;
    ram.dfs = 32;
    int result = ospi_psram_xip_init(&ram);
    tr_error = result;
    if (result) { tr_status = 0xe1; for (;;) __WFI(); }
    tr_status = 2;
    volatile uint32_t *probe = reinterpret_cast<volatile uint32_t *>(0xa3fff000);
    for (int i = 0; i < 64; ++i) probe[i] = 0x12345678U ^ (i*0x01010101U);
    SCB_CleanInvalidateDCache_by_Addr((uint32_t *)probe, 256);
    for (int i = 0; i < 64; ++i) {
        if (probe[i] != (0x12345678U ^ (i*0x01010101U))) {
            tr_error = i+1; tr_status = 0xe2; for (;;) __WFI();
        }
    }
    runtime_init();
    if (ethosu_init(&npu, reinterpret_cast<void *>(NPU_HG_BASE), nullptr, 0, 1, 1)) {
        tr_status = 0xe3; for (;;) __WFI();
    }
    struct ethosu_hw_info info;
    ethosu_get_hw_info(&npu, &info);
    tr_npu_product = info.version.product_major;
    tr_npu_macs = 1U << info.cfg.macs_per_cc;
    ETHOSU_PMU_Enable(&npu);
    ETHOSU_PMU_Set_EVTYPER(&npu, 0, ETHOSU_PMU_NPU_ACTIVE);
    ETHOSU_PMU_Set_EVTYPER(&npu, 1, ETHOSU_PMU_MAC_ACTIVE);
    ETHOSU_PMU_Set_EVTYPER(&npu, 2, ETHOSU_PMU_NPU_IDLE);
    ETHOSU_PMU_CNTR_Enable(&npu, ETHOSU_PMU_CCNT_Msk | 7);
    tr_status = 3;
    printf("TRANSLATOR_READY psram=64MB npu_product=%u macs=%u\r\n", tr_npu_product, tr_npu_macs);
    for (;;) {
        if (!tr_command) continue;
        unsigned command = tr_command, index = tr_model_index;
        SCB_InvalidateDCache();
        tr_status = 4;
        int result = command == 1 ? load_model(index) : command == 2 ? execute(index) : -99;
        tr_error = result;
        tr_command = 0;
        tr_status = result ? 0xe4 : command == 1 ? 5 : 6;
    }
}
}
