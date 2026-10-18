#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Before assets_ready: write <=1 MiB into live_model_staging, then publish
// destination/bytes/new nonzero sequence and command=1 last. Do not modify
// staging or metadata until command returns to zero with completed_sequence.
extern volatile uint32_t live_asset_command;
extern volatile uint32_t live_asset_destination;
extern volatile uint32_t live_asset_bytes;
extern volatile uint32_t live_asset_error;
extern volatile uint32_t live_asset_sequence;
extern volatile uint32_t live_asset_completed_sequence;
// 0 idle; 1 request completed (inspect error); -1 recursive call ignored.
// Errors: 1 command, 2 size/source bounds, 3 PSRAM bounds, 4 sequence.
// Caller runs on the camera/display owner thread only, before OCR starts.
int live_asset_service(void);
#ifdef __cplusplus
}
#endif
