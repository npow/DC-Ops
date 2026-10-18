// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <optional>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>

namespace dcops {
struct RunnerMemory {
    uint8_t *method; size_t method_bytes;
    uint8_t *planned; size_t planned_bytes;
    uint8_t *temporary; size_t temporary_bytes;
};
struct RunnerInput { void *data; size_t bytes; };
struct RunnerOutput { const void *data; size_t bytes; };
class Runner {
public:
    int load(const void *pte,size_t bytes,RunnerMemory memory);
    int execute(const RunnerInput *inputs,size_t count);
    RunnerOutput output(size_t index=0);
private:
    class Loader : public executorch::runtime::DataLoader {
    public:
        const uint8_t *data=nullptr; size_t bytes=0;
        executorch::runtime::Result<executorch::runtime::FreeableBuffer> load(
            size_t offset,size_t size,const SegmentInfo&) const override;
        executorch::runtime::Result<size_t> size() const override { return bytes; }
    } loader_;
    std::optional<executorch::runtime::Program> program_;
    std::optional<executorch::runtime::Method> method_;
    std::optional<executorch::runtime::MemoryAllocator> allocator_,temporary_;
    std::optional<executorch::runtime::HierarchicalAllocator> planned_;
    std::optional<executorch::runtime::MemoryManager> manager_;
    executorch::runtime::Span<uint8_t> spans_[8];
    executorch::aten::TensorImpl::SizesType shapes_[4][8];
    executorch::aten::TensorImpl::DimOrderType orders_[4][8];
    executorch::aten::TensorImpl::StridesType strides_[4][8];
    std::optional<executorch::aten::TensorImpl> impls_[4];
    std::optional<executorch::aten::Tensor> tensors_[4];
};
}
extern "C" {
int live_npu_initialize(void);
extern uint8_t live_model_staging[2560*1024];
extern uint8_t live_temporary_pool[512*1024];
}
