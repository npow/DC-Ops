#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int live_boot_he_worker(void);
extern volatile uint32_t live_he_boot_stage,live_he_boot_transport,live_he_boot_service,live_he_seen_status;
#ifdef __cplusplus
}
#endif
