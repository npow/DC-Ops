#include "asset_transfer.h"
#include "live.h"
#include "runner.h"
#include "RTE_Components.h"
#include CMSIS_device_header
#include <algorithm>
#include <cstring>

extern "C" {
volatile uint32_t live_asset_command;
volatile uint32_t live_asset_destination;
volatile uint32_t live_asset_bytes;
volatile uint32_t live_asset_error;
volatile uint32_t live_asset_sequence;
volatile uint32_t live_asset_completed_sequence;
}
namespace {
bool copying;
constexpr uintptr_t kPsramBegin = 0xa0000000u, kPsramEnd = 0xa4000000u;
constexpr size_t kMaximumChunk = 1024 * 1024, kCopyChunk = 16 * 1024;
void invalidate(const void* pointer, size_t bytes) {
    const uintptr_t first = reinterpret_cast<uintptr_t>(pointer) & ~uintptr_t(31);
    const uintptr_t end = (reinterpret_cast<uintptr_t>(pointer) + bytes + 31) & ~uintptr_t(31);
    SCB_InvalidateDCache_by_Addr(reinterpret_cast<void*>(first), end - first);
    __DSB();
}
void clean(const void* pointer, size_t bytes) {
    const uintptr_t first = reinterpret_cast<uintptr_t>(pointer) & ~uintptr_t(31);
    const uintptr_t end = (reinterpret_cast<uintptr_t>(pointer) + bytes + 31) & ~uintptr_t(31);
    SCB_CleanDCache_by_Addr(reinterpret_cast<void*>(first), end - first);
    __DSB();
}
}
extern "C" int live_asset_service(void) {
    if (copying) return -1;
    const uint32_t command = live_asset_command;
    if (!command) return 0;
    copying = true;
    __DMB();
    const uintptr_t destination = live_asset_destination;
    const size_t bytes = live_asset_bytes;
    const uint32_t sequence = live_asset_sequence;
    uint32_t error = 0;
    if (command != 1) error = 1;
    else if (!bytes || bytes > kMaximumChunk || bytes > sizeof(live_model_staging)) error = 2;
    else if (destination < kPsramBegin || destination >= kPsramEnd || bytes > kPsramEnd - destination) error = 3;
    else if (!sequence || sequence == live_asset_completed_sequence) error = 4;
    if (!error) {
        // Debug-probe writes bypass HP's cached view of the staging arena.
        invalidate(live_model_staging, bytes);
        for (size_t offset = 0; offset < bytes; offset += kCopyChunk) {
            const size_t count = std::min(kCopyChunk, bytes - offset);
            auto* target = reinterpret_cast<uint8_t*>(destination + offset);
            std::memcpy(target, live_model_staging + offset, count);
            clean(target, count);
            live_service();
        }
    }
    live_asset_error = error;
    live_asset_completed_sequence = sequence;
    __DSB();
    live_asset_command = 0;
    __DSB();
    copying = false;
    return 1;
}
