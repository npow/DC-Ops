// SPDX-License-Identifier: Apache-2.0
#include "boot_he.h"
#include "../live/live.h"
#include "RTE_Components.h"
#include CMSIS_device_header
extern "C" {
#include "se_services_port.h"
}
#include "services_lib_api.h"
#include "../../src/trace.h"

extern "C" {
volatile uint32_t live_he_boot_stage,live_he_boot_transport,live_he_boot_service,live_he_seen_status;
int live_boot_he_worker(void) {
    constexpr uintptr_t vectors_address=0x80100000u;
    constexpr uintptr_t worker_status_address=0x58800150u;
    live_he_boot_stage=1;
    live_he_boot_transport=0; live_he_boot_service=0; live_he_seen_status=0;
    SCB_InvalidateDCache_by_Addr(reinterpret_cast<void *>(vectors_address),32); __DSB();
    const auto *vectors=reinterpret_cast<const volatile uint32_t *>(vectors_address);
    if (vectors[0]!=0x20040000u || !(vectors[1]&1u) ||
        vectors[1]<vectors_address || vectors[1]>=0x80200000u) return -40;
    if (se_services_s_handle==0xffffffffu) return -41;
    uint32_t error=0;
    live_he_boot_stage=2;
    live_he_boot_transport=SERVICES_boot_set_vtor(se_services_s_handle,EXTSYS_1,vectors_address,&error);
    live_he_boot_service=error;
    if (live_he_boot_transport || error) return -42;
    live_he_boot_stage=3;
    error=0;
    live_he_boot_transport=SERVICES_boot_reset_cpu(se_services_s_handle,EXTSYS_1,&error);
    live_he_boot_service=error;
    if (live_he_boot_transport || error) return -43;
    live_he_boot_stage=4;
    error=0;
    live_he_boot_transport=SERVICES_boot_release_cpu(se_services_s_handle,EXTSYS_1,&error);
    live_he_boot_service=error;
    if (live_he_boot_transport || error) return -44;
    live_he_boot_stage=5;
    const uint32_t started=live_now_ms();
    for (;;) {
        SCB_InvalidateDCache_by_Addr(reinterpret_cast<void *>(worker_status_address&~uintptr_t(31)),32); __DSB();
        live_he_seen_status=*reinterpret_cast<const volatile uint32_t *>(worker_status_address);
        if (live_he_seen_status==2) {
            live_he_boot_stage=6;
            trace_printf("LIVE_HE_READY cpu=%u vector=%08lx\r\n",unsigned(EXTSYS_1),static_cast<unsigned long>(vectors_address));
            return 0;
        }
        if (live_he_seen_status>=0xe0 && live_he_seen_status<=0xff) return -45;
        if (live_now_ms()-started>=5000u) return -46;
        live_service();
    }
}
}
