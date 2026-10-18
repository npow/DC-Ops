// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Call only after assets and NPUs are ready, with OCR/translation idle.
// Set enabled=1 and command=1 (FP32 CPU) or 2 (A16W8 U85). One invocation per command.
int live_benchmark_service(void);
extern volatile uint32_t live_benchmark_enabled,live_benchmark_command;
extern volatile uint32_t live_benchmark_status,live_benchmark_error,live_benchmark_target;
extern volatile uint32_t live_benchmark_ms,live_benchmark_prepare_ms;
extern volatile uint64_t live_benchmark_cycles;
extern volatile uint32_t live_benchmark_output_address,live_benchmark_output_bytes;
extern volatile uint64_t live_benchmark_pmu_cycles,live_benchmark_pmu_active,live_benchmark_pmu_mac,live_benchmark_pmu_idle;
extern volatile uint32_t live_benchmark_cpu_ms,live_benchmark_npu_ms,live_benchmark_cpu_runs,live_benchmark_npu_runs;
extern volatile uint32_t live_benchmark_cpu_hash,live_benchmark_npu_hash,live_benchmark_compare_valid,live_benchmark_nonfinite;
extern volatile float live_benchmark_mean_abs,live_benchmark_max_abs,live_benchmark_rmse,live_benchmark_stage_speedup;
#ifdef __cplusplus
}
#endif
