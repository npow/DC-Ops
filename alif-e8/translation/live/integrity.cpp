// SPDX-License-Identifier: Apache-2.0
#include "integrity.h"
#include "runner.h"
#include "live.h"
#include "RTE_Components.h"
#include CMSIS_device_header
#include <array>
#include <algorithm>
namespace {
constexpr size_t block_bytes=256,expected_capacity=1024*1024,bad_capacity=262144;
constexpr std::array<uint32_t,256> crc_table() {
    std::array<uint32_t,256> result{};
    for (uint32_t byte=0;byte<256;++byte) {
        uint32_t value=byte;
        for (unsigned bit=0;bit<8;++bit) value=(value>>1)^((value&1u) ? 0xedb88320u : 0u);
        result[byte]=value;
    }
    return result;
}
constexpr auto table=crc_table();
bool running;
uint32_t crc32(const uint8_t *source,size_t bytes) {
    uint32_t crc=0xffffffffu;
    for (size_t index=0;index<bytes;++index) crc=(crc>>8)^table[(crc^source[index])&255u];
    return crc^0xffffffffu;
}
void invalidate(uintptr_t address,size_t bytes) {
    const uintptr_t first=address&~uintptr_t(31);
    const uintptr_t end=(address+bytes+31)&~uintptr_t(31);
    SCB_InvalidateDCache_by_Addr(reinterpret_cast<void *>(first),end-first); __DSB();
}
int fail(int error) {
    live_integrity_error=error;
    live_integrity_status=0xe0;
    running=false;
    return error;
}
}
extern "C" {
extern volatile uint32_t live_assets_ready;
volatile uint32_t live_integrity_command,live_integrity_address,live_integrity_bytes,live_integrity_block_size=256;
volatile uint32_t live_integrity_status,live_integrity_bad_count,live_integrity_stored_count;
volatile int32_t live_integrity_error;
volatile uint32_t live_integrity_blocks_done,live_integrity_block_count,live_integrity_elapsed_ms;
volatile uint32_t live_integrity_expected_address=reinterpret_cast<uintptr_t>(live_model_staging);
volatile uint32_t live_integrity_bad_indices_address=reinterpret_cast<uintptr_t>(live_model_staging)+expected_capacity;
int live_integrity_service(void) {
    if (!live_integrity_command || running) return 0;
    const uint32_t command=live_integrity_command;
    live_integrity_command=0;
    running=true;
    live_integrity_status=1;
    live_integrity_error=0;
    live_integrity_bad_count=live_integrity_stored_count=live_integrity_blocks_done=live_integrity_block_count=live_integrity_elapsed_ms=0;
    if (command!=1) return fail(-1);
    if (live_assets_ready) return fail(-2);
    const uintptr_t address=live_integrity_address;
    const size_t bytes=live_integrity_bytes;
    if (!bytes || address<0xa0000000u || address>=0xa4000000u || bytes>0xa4000000u-address) return fail(-3);
    if (live_integrity_block_size!=block_bytes) return fail(-4);
    const size_t blocks=(bytes+block_bytes-1)/block_bytes;
    if (blocks>expected_capacity/sizeof(uint32_t)) return fail(-5);
    live_integrity_block_count=blocks;
    const uint32_t started=live_now_ms();
    invalidate(reinterpret_cast<uintptr_t>(live_model_staging),blocks*sizeof(uint32_t));
    const auto *expected=reinterpret_cast<const uint32_t *>(live_model_staging);
    auto *bad=reinterpret_cast<uint32_t *>(live_model_staging+expected_capacity);
    const auto *source=reinterpret_cast<const uint8_t *>(address);
    // Invalidate source in chunks so cache maintenance also services video.
    constexpr size_t chunk=16*1024;
    for (size_t offset=0;offset<bytes;offset+=chunk) {
        const size_t count=std::min(chunk,bytes-offset);
        invalidate(address+offset,count);
        for (size_t local=0;local<count;local+=block_bytes) {
            const size_t block=(offset+local)/block_bytes;
            const uint32_t observed=crc32(source+offset+local,std::min(block_bytes,count-local));
            if (observed!=expected[block]) {
                const uint32_t index=live_integrity_bad_count++;
                if (index<bad_capacity) bad[live_integrity_stored_count++]=block;
            }
            live_integrity_blocks_done=block+1;
        }
        live_integrity_elapsed_ms=live_now_ms()-started;
        if (live_assets_ready) return fail(-2);
        if (live_service()<0) return fail(-7);
    }
    if (live_integrity_stored_count) {
        SCB_CleanDCache_by_Addr(bad,(live_integrity_stored_count*sizeof(uint32_t)+31)&~uint32_t(31)); __DSB();
    }
    live_integrity_elapsed_ms=live_now_ms()-started;
    if (live_integrity_bad_count>bad_capacity) return fail(-6);
    live_integrity_status=2;
    running=false;
    return 1;
}
}
