// SPDX-License-Identifier: Apache-2.0
#include "external_memory.h"
#include "live.h"
#include "RTE_Components.h"
#include CMSIS_device_header
#include "board_config.h"
#include "board_defs.h"
#include "APS512XXN_PSRAM.h"
#include "ospi_psram_xip.h"
#include "sys_ctrl_aes.h"
#include "../../src/trace.h"

namespace {
int32_t reset_ram(OSPI_Type *ospi, AES_Type *aes) {
    aes_disable_xip(aes);
    ospi_disable(ospi);
    // The pack's polling sender retains its previous transfer threshold.
    ospi->OSPI_TXFTLR=0;
    return aps512xxn_psram_init(ospi,aes);
}
}
extern "C" int live_memory_initialize(void) {
    int result=board_clocks_config(CLKEN_SYSPLL_MASK|CLKEN_HFOSC_MASK|
        CLKEN_CLK_100M_MASK|(1U<<CLKEN_CLK_400M)|(1U<<CLKEN_SRAM1));
    if (result) return result;
    ospi_psram_xip_config ram{};
    ram.instance=static_cast<OSPI_INSTANCE>(BOARD_PSRAM_OSPI_INSTANCE);
    ram.ram_init=reset_ram;
    ram.ram_type=RAM_TYPE_PSRAM;
    ram.config_mode=CONFIG_MODE_USER_PARAMETERS;
    ram.bus_speed=25000000;
    ram.spi_frf=OSPI_SPI_FRF_DUAL_OCTAL;
    ram.wait_cycles=4;
    ram.ddr_drive_edge=1;
    ram.rxds_delay=6;
    ram.signal_delay=22;
    ram.dfs=32;
    result=ospi_psram_xip_init(&ram);
    if (result) return result;
    // Register reads and dual-octal XIP reads require different sampling delays.
    aes_set_rxds_delay(AES0,11);
    auto *probe=reinterpret_cast<volatile uint32_t *>(0xa3fff000);
    for (unsigned i=0;i<64;++i) probe[i]=0x12345678U^(i*0x01010101U);
    SCB_CleanInvalidateDCache_by_Addr(reinterpret_cast<uint32_t *>(const_cast<uint32_t *>(probe)),256);
    for (unsigned i=0;i<64;++i)
        if (probe[i]!=(0x12345678U^(i*0x01010101U))) return -100-int(i);
    trace_printf("LIVE_PSRAM_READY bytes=67108864 clock=25000000\r\n");
    return 0;
}
