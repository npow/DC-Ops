// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
int trace_printf(const char *format, ...);
extern volatile unsigned dc_log_position;
extern char dc_log[16384];
#ifdef __cplusplus
}
#endif
