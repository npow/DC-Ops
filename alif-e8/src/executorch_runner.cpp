// SPDX-License-Identifier: Apache-2.0
#include "RTE_Components.h"
#include CMSIS_device_header
#include "ethosu_driver.h"
#include "executorch_runner.h"
#include "model_params.h"
#include "trace.h"
#include <cstdio>
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
namespace {
constexpr size_t kMethodSize = 96 * 1024;
constexpr size_t kTempSize = 1024 * 1024;
constexpr size_t kPlannedSize = 512 * 1024;
alignas(32) uint8_t method_pool[kMethodSize];
__attribute__((section(".bss.ai_pool"), aligned(32))) uint8_t temp_pool[kTempSize];
__attribute__((section(".bss.ai_pool"), aligned(32))) uint8_t planned_pool[kPlannedSize];
uint8_t pal_pool[64 * 1024];
size_t pal_offset;
struct ethosu_driver npu;
class Loader : public DataLoader {
public:
    const uint8_t *data;
    size_t length;
    Result<FreeableBuffer> load(size_t offset, size_t size, const SegmentInfo&) const override {
        if (offset > length || size > length - offset) return Error::InvalidArgument;
        return FreeableBuffer(data + offset, size, nullptr);
    }
    Result<size_t> size() const override { return length; }
};
Loader loader;
std::optional<Program> program;
std::optional<Method> method;
Span<uint8_t> planned_spans[16];
std::optional<HierarchicalAllocator> planned;
MemoryAllocator method_allocator(kMethodSize, method_pool);
MemoryAllocator temp_allocator(kTempSize, temp_pool);
std::optional<MemoryManager> manager;
}
// Poll U85 completion: no dependence on the board's legacy U55 IRQ routing.
struct semaphore { volatile uint8_t count; };
extern "C" int ethosu_semaphore_take(void *sem, uint64_t) {
    auto *s = static_cast<semaphore *>(sem);
    while (!s->count) {
        if (*(volatile uint32_t *)(NPU_HG_BASE + 4) & 2U) ethosu_irq_handler(&npu);
        __NOP();
    }
    --s->count;
    return 0;
}
extern "C" void NPU_HP_IRQHandler(void) {}
extern "C" {
void et_pal_init(void) {}
ET_NORETURN void et_pal_abort(void) {
    printf("ET_ABORT\r\n");
    for (;;) __WFI();
}
et_timestamp_t et_pal_current_ticks(void) { return DWT->CYCCNT; }
et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) { return {5, 2}; }
void et_pal_emit_log_message(et_timestamp_t, et_pal_log_level_t level,
    const char *, const char *, size_t, const char *message, size_t length) {
    printf("ET:%c %.*s\r\n", static_cast<char>(level), (int)length, message);
}
void *et_pal_allocate(size_t size) {
    size_t aligned = (pal_offset + 15) & ~size_t(15);
    if (aligned + size > sizeof(pal_pool)) return nullptr;
    pal_offset = aligned + size;
    return pal_pool + aligned;
}
void et_pal_free(void *) {}
int executorch_init(const uint8_t *data, size_t size) {
    runtime_init();
    if (ethosu_init(&npu, reinterpret_cast<void *>(NPU_HG_BASE), nullptr, 0, 1, 1)) return -1;
    printf("NPU_U85_READY model=%u\r\n", (unsigned)size);
    loader.data = data;
    loader.length = size;
    auto p = Program::load(&loader, Program::Verification::Minimal);
    if (!p.ok()) { printf("PROGRAM_LOAD_ERROR=%d\r\n", (int)p.error()); return -2; }
    program.emplace(std::move(p.get()));
    auto metadata = program->method_meta("forward");
    if (!metadata.ok()) return -3;
    auto meta = metadata.get();
    size_t offset = 0;
    const size_t count = meta.num_memory_planned_buffers();
    if (count > 16 || meta.num_inputs() != 1 || meta.num_outputs() != 3) return -4;
    for (size_t i = 0; i < count; ++i) {
        auto result = meta.memory_planned_buffer_size(i);
        if (!result.ok()) return -5;
        size_t bytes = result.get();
        offset = (offset + 31) & ~size_t(31);
        printf("PLANNED[%u]=%u\r\n", (unsigned)i, (unsigned)bytes);
        if (offset + bytes > kPlannedSize) return -6;
        planned_spans[i] = Span<uint8_t>(planned_pool + offset, bytes);
        offset += bytes;
    }
    planned.emplace(Span<Span<uint8_t>>(planned_spans, count));
    manager.emplace(&method_allocator, &*planned, &temp_allocator);
    auto result = program->load_method("forward", &*manager);
    if (!result.ok()) { printf("METHOD_LOAD_ERROR=%d\r\n", (int)result.error()); return -7; }
    method.emplace(std::move(result.get()));
    auto info = meta.input_tensor_meta(0);
    if (!info.ok() || info->scalar_type() != executorch::aten::ScalarType::Char ||
        info->nbytes() != 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE) return -8;
    printf("DETECTOR_READY planned=%u\r\n", (unsigned)offset);
    return 0;
}
int executorch_run_detector(int8_t *input, const int8_t **outputs) {
    using namespace executorch::aten;
    TensorImpl::SizesType sizes[] = {1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE};
    TensorImpl::DimOrderType order[] = {0, 2, 3, 1};
    TensorImpl::StridesType strides[] = {3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE,
        1, MODEL_INPUT_SIZE * 3, 3};
    TensorImpl impl(ScalarType::Char, 4, sizes, input, order, strides);
    Tensor tensor(&impl);
    Error error = method->set_input(EValue(tensor), 0);
    if (error != Error::Ok) return (int)error;
    error = method->execute();
    if (error != Error::Ok) return (int)error;
    for (int i = 0; i < 3; ++i) {
        auto &value = method->get_output(i);
        if (!value.isTensor()) return -9;
        auto output = value.toTensor();
        if (output.scalar_type() != ScalarType::Char) return -10;
        outputs[i] = output.const_data_ptr<int8_t>();
    }
    return 0;
}
}
