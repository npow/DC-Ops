// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int executorch_init(const uint8_t *data, size_t size);
int executorch_run_detector(int8_t *input, const int8_t **outputs);
#ifdef __cplusplus
}
#endif
