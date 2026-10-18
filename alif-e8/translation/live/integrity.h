// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Before assets_ready only: stage little-endian zlib CRC32 values, then command=1.
// CRC blocks are256B; the last block hashes its actual remaining byte count.
int live_integrity_service(void);
extern volatile uint32_t live_integrity_command,live_integrity_address,live_integrity_bytes,live_integrity_block_size;
extern volatile uint32_t live_integrity_status,live_integrity_bad_count,live_integrity_stored_count;
extern volatile int32_t live_integrity_error;
extern volatile uint32_t live_integrity_blocks_done,live_integrity_block_count,live_integrity_elapsed_ms;
extern volatile uint32_t live_integrity_expected_address,live_integrity_bad_indices_address;
#ifdef __cplusplus
}
#endif
