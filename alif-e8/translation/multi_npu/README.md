# E8 three-NPU integration

The mixed driver and HE mailbox execute on the E8 in the live firmware. HP U55
runs detection, HE U55 runs recognition, and the optional neural translator
runs on U85. The default 60-sign demo uses a CPU phrase lookup after OCR.
See [live firmware](../live/README.md) and [validation reports](../../verification/README.md)
for the current application and physical-board measurements.

| MCU | NPU | Model | Actual PTE command-stream configuration |
|---|---|---|---|
| M55 HP | local U55 | text detector | product 0, log2 MACs 8 |
| M55 HE | local U55 | text recognizer | product 0, log2 MACs 7 |
| M55 HP | global U85 | translation encoder and decoder | product 2, log2 MACs 8 |

This is an initial assignment to benchmark. Recognition on HE can overlap translation on HP; detection and translation on HP remain synchronous until a scheduler or asynchronous delegate is implemented. Three NPUs do not imply three CPUs, and the stock synchronous ExecuTorch call blocks its calling MCU until that inference completes.

## Build and verification

```sh
python3 translation/multi_npu/build_driver.py
python3 translation/multi_npu/verify_dispatch.py
```

The builder uses the installed ARM driver 1.26.5 and GCC 15.3.1, writes only under `artifacts/translator/mixed-driver`, and produces `libe8_ethosu_mixed.a`. It compiles common `ethosu_driver.c` once, device and PMU implementations once per architecture with renamed symbols, and dispatches their public APIs using the device's MMIO base. The default PMU API exposes four event counters, the shared safe subset. PMU event names are translated because the two architecture headers assign different enum values to the same events. The shared `ethosu_config_select` hook must handle the architecture it is configuring; the U85 MEM_ATTR index meaning must not be assumed identical to U55 AXI-limit selection. The application using this mixed library must compile with `ETHOSU85` and `ETHOSU_ARCH=u85` so its public PMU enum follows U85 numbering.

Verification compiled all sources for Cortex-M55, partially linked the combined objects to detect duplicate definitions, and ran a host check of routing plus all four real PTE command-stream headers. This verifies linkage and dispatch logic; it does not verify MMIO, board clocks, memory access, or accelerator execution.

## HP integration

1. Replace the stock Generic U85 driver component with the mixed library; keep its include directories and public `ETHOSU85` definitions. Do not link the original device/PMU/driver objects a second time. Keep the existing board cache, semaphore, address remapping, and memory-port selection overrides. The library leaves these weak platform hooks available for the board's strong overrides.
2. Initialize two `ethosu_driver` instances with `ethosu_init`: local U55 at `NPU_HP_BASE` (0x400E1000), then U85 at `NPU_HG_BASE` (0x49042000). Include `mixed_driver.h`; after each successful initialization, call `e8_register_npu(&driver)`. This enables the strong `ethosu_reserve_driver_ex` implementation to select using the product and MAC count parsed from the Vela command stream. It preserves the common driver's semaphore accounting. One inference thread per MCU is required by this adapter.
3. Route each NPU interrupt to `ethosu_irq_handler` for its own instance. Existing polling waits must poll both devices and call the same handler. Do not assume the U85 STATUS register offset applies to U55: use architecture-specific status logic or real interrupts.
4. Load the detector and translator models using the existing ExecuTorch backend. Its strong parser hook reads COP1 optimizer configuration and fails unsupported requests instead of silently using the first device. Allocate separate model arenas and buffers; the backend's global fast-scratch allocation can be shared only while HP invocations remain sequential. Register each driver once before loading/running models.

## HE recognition worker

Build a small M55 HE firmware with the standard Generic U55 driver, `ETHOSU55`, `ETHOSU_ARCH=u55`, and only the recognizer's ExecuTorch operations. The installed HE BSP already maps `NPULOCAL_BASE` to `NPU_HE_BASE` at its local 0x400E1000. Its clock setup and LocalToGlobal mapping must use the HE headers, not the HP translation unit. Keep PSRAM/controller initialization exclusively in HP; HE must wait for the HP memory-ready handshake.

`worker_mailbox.h` defines a single outstanding request containing global input/output addresses, model ID, sequence, and a response with timing/PMU counters. Compile `worker_mailbox.cpp` in the HE worker and call `e8_worker_poll` from its loop. Supply cache-maintenance callbacks including DSB and an execute callback which loads/runs the recognizer, copies output into the bounded caller-owned buffer, and reports the bytes written. This execute callback is deliberately not supplied until the recognizer runner and memory allocation are integrated. Model ID must be checked by that callback.

Reserve a 256-byte, 128-byte-aligned shared SRAM block in both linker maps. Choose the address after inspecting both final maps; these files do not guess an unused address. HP owns the request's first 128 bytes, HE owns the response's second 128 bytes. Use global aliases in every address descriptor; no local TCM addresses may cross the mailbox. Do not share a cache line between the mailbox and unrelated allocations.

For each HP submission: wait for the previous response, clean all input data, write the request with sequence 0, clean all 128 request bytes and DSB, then write the next nonzero sequence and clean its first cache line and DSB. HE uses the same two-phase publication for responses and cleans output before publishing completion. HP invalidates response cache lines before polling and invalidates output after seeing the expected completed sequence. A completed response is not reusable until HP has consumed its output. Keep sequence 0 reserved; restart both sequence states together after wrap/reset. `volatile` alone does not make caches coherent.

Start HE temporarily through the existing debugger or secure-enclave CPU boot service, after its image and vectors are loaded. Do not modify the persistent boot table while the user is evaluating firmware. E8 pack `se_services/include/services_lib_api.h` supplies `SERVICES_boot_cpu`, `SERVICES_boot_set_vtor`, and `SERVICES_boot_release_cpu`; pick the CPU ID from the installed protocol definitions when integrating. No boot service or debugger command is executed by this component.

## Measurements to display

Report each NPU's invocation count, elapsed inference milliseconds, PMU total/active/MAC-active cycles, and failures. Snapshot PMU inside inference begin/end while the NPU is powered, and check overflow/clock frequency before computing a rate. Per-inference active/total is a busy ratio during that invocation, not utilization over the whole video interval. For a HUD window, accumulate active cycles over that window and divide by its duration times that NPU's measured/configured frequency. Keep model time separate from preprocessing, tokenization, transport, and camera/display time.

Report CPU time divided by NPU time for the same frozen inputs and outputs, with warmup and repeated samples. Label a float32 CPU comparison explicitly when the NPU model is quantized. Translation must include encoder latency, decoder latency per token, and complete sentence time with token count; per-token time alone is not end-to-end translation latency. Measure the combined pipeline too: overlapping NPUs share memory bandwidth, so isolated speedups cannot be added.

## Source evidence

Installed sources used for this implementation:

- `ARM/ethos-u-core-driver/1.26.5/ethos_u_core_driver/include/ethosu_types.h`: architecture-independent `ethosu_device` and hardware-info structures.
- Same driver's `src/ethosu_driver.c`: one registered-driver list, generic invocation/control code, COP1 optimizer config structure, semaphore accounting in reserve/release.
- Same driver's `src/ethosu_device.h`: 10 architecture-dependent device entry points; `src/ethosu_device_u55_u65.c` and `src/ethosu_device_u85.c`: distinct register implementations.
- Same driver's `include/pmu_ethosu.h`: different U55/U85 event enum numbering, 24 PMU entry points; interface headers expose 4 U55 versus 8 U85 counters.
- `PyTorch/ExecuTorch/1.5.1/src/backends/arm/runtime/EthosUBackend_Cortex_M.cpp`: weak product parser/reservation hooks and synchronous `ethosu_invoke_v3` call.
- `AlifSemiconductor/Ensemble/2.2.1/Boards/DevKit-e8/Layers/M55_HP/ethos_setup.c` and `M55_HE/ethos_setup.c`: stock compile-time selection between local U55 and global U85.
- E8 device `include/rtss_hp/core_defines.h` and `include/rtss_he/core_defines.h`: different local NPU/IRQ aliases and inter-MCU MHU aliases.
