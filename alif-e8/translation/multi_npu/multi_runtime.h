#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
struct e8_npu_stats {
    uint32_t invocations;
    uint32_t failures;
    uint32_t elapsed_ms;
    uint32_t last_ms;
    uint64_t total_cycles;
    uint64_t active_cycles;
    uint64_t mac_cycles;
    uint64_t idle_cycles;
};
// index 0: HP U55, index 1: HG U85. Caller and inference run on one MCU owner thread.
int live_npu_stats(unsigned index, struct e8_npu_stats *stats);
void live_npu_poll(void);
// Change only between invocations. OFF permits a Shared_Sram OCR model while
// the application owns the backing buffer as a snapshot; restore before U85.
int live_npu_fast_scratch_enabled(void);
int live_npu_set_fast_scratch_enabled(int enabled);
void *live_npu_fast_scratch_storage(size_t *bytes);
#define LIVE_U55_DIAG_COUNT 23
extern const uint32_t live_u55_diag_offsets[LIVE_U55_DIAG_COUNT];
extern volatile uint32_t live_u55_diag_registers[LIVE_U55_DIAG_COUNT];
extern volatile uint32_t live_u55_diag_sequence;
#define LIVE_U85_DIAG_COUNT 34
extern const uint32_t live_u85_diag_offsets[LIVE_U85_DIAG_COUNT];
extern volatile uint32_t live_u85_diag_registers[LIVE_U85_DIAG_COUNT];
extern volatile uint32_t live_u85_diag_sequence;
extern volatile uint32_t live_u85_diag_regions[6];
#ifdef __cplusplus
}
#endif
