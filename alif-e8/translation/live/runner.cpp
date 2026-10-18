// SPDX-License-Identifier: Apache-2.0
#include "runner.h"
#include "live.h"
#include <utility>
using namespace executorch::runtime;
namespace dcops {
Result<FreeableBuffer> Runner::Loader::load(size_t offset,size_t size,const SegmentInfo&) const {
    if (offset>bytes || size>bytes-offset) return Error::InvalidArgument;
    return FreeableBuffer(data+offset,size,nullptr);
}
int Runner::load(const void *pte,size_t bytes,RunnerMemory memory) {
    method_.reset(); program_.reset();
    if (!pte || !bytes || !memory.method || !memory.planned || !memory.temporary ||
        reinterpret_cast<uintptr_t>(memory.planned)%32) return -1;
    loader_.data=static_cast<const uint8_t *>(pte); loader_.bytes=bytes;
    auto program=Program::load(&loader_,Program::Verification::Minimal);
    if (!program.ok()) return int(program.error());
    program_.emplace(std::move(program.get()));
    auto metadata=program_->method_meta("forward");
    if (!metadata.ok()) return int(metadata.error());
    auto meta=metadata.get();
    if (meta.num_memory_planned_buffers()>8) return -2;
    size_t offset=0;
    for (size_t i=0;i<meta.num_memory_planned_buffers();++i) {
        auto bytes=meta.memory_planned_buffer_size(i);
        if (!bytes.ok()) return -3;
        offset=(offset+31)&~size_t(31);
        if (offset>memory.planned_bytes || bytes.get()>memory.planned_bytes-offset) return -4;
        spans_[i]=Span<uint8_t>(memory.planned+offset,bytes.get());
        offset+=bytes.get();
    }
    allocator_.emplace(memory.method_bytes,memory.method);
    temporary_.emplace(memory.temporary_bytes,memory.temporary);
    planned_.emplace(Span<Span<uint8_t>>(spans_,meta.num_memory_planned_buffers()));
    manager_.emplace(&*allocator_,&*planned_,&*temporary_);
    auto method=program_->load_method("forward",&*manager_);
    if (!method.ok()) return int(method.error());
    method_.emplace(std::move(method.get()));
    return 0;
}
int Runner::execute(const RunnerInput *inputs,size_t count) {
    using namespace executorch::aten;
    if (!method_ || !inputs || count>4) return -1;
    auto metadata=program_->method_meta("forward");
    if (!metadata.ok() || metadata->num_inputs()!=count) return -2;
    for (size_t i=0;i<count;++i) {
        auto info=metadata->input_tensor_meta(i);
        if (!info.ok() || info->sizes().size()>8 || info->nbytes()!=inputs[i].bytes || !inputs[i].data) return -3;
        auto sizes=info->sizes();
        for (size_t d=0;d<sizes.size();++d) { shapes_[i][d]=sizes[d]; orders_[i][d]=info->dim_order()[d]; }
        int stride=1;
        for (int order=int(sizes.size())-1;order>=0;--order) {
            int dimension=orders_[i][order];
            strides_[i][dimension]=stride; stride*=shapes_[i][dimension];
        }
        tensors_[i].reset(); impls_[i].reset();
        impls_[i].emplace(info->scalar_type(),sizes.size(),shapes_[i],inputs[i].data,orders_[i],strides_[i]);
        tensors_[i].emplace(&*impls_[i]);
        auto status=method_->set_input(EValue(*tensors_[i]),i);
        if (status!=Error::Ok) return int(status);
    }
    return int(method_->execute());
}
RunnerOutput Runner::output(size_t index) {
    if (!method_ || index>=method_->outputs_size()) return {};
    auto value=method_->get_output(index);
    if (!value.isTensor()) return {};
    auto tensor=value.toTensor();
    return {tensor.const_data_ptr(),tensor.nbytes()};
}
}
