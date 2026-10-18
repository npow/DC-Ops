#include "mixed_driver.h"
#include <stdint.h>
#include <string.h>

// Per-core executor: register only devices initialized by this MCU.
static struct ethosu_driver *devices[2];
static unsigned device_count;

int e8_register_npu(struct ethosu_driver *driver)
{
    if (!driver || device_count == 2) return -1;
    for (unsigned i = 0; i < device_count; ++i) if (devices[i] == driver) return -1;
    devices[device_count++] = driver;
    return 0;
}

int ethosu_get_product_config_from_cop_data(const void *data, const int size,
                                          uint32_t *product, uint32_t *log2_macs)
{
    if (!data || size < 16 || !product || !log2_macs) return -1;
    const uint8_t *bytes = data;
    uint32_t word;
    memcpy(&word, bytes, 4);
    if (word != 0x31504f43u) return -1;
    for (int offset = 4; offset <= size - 4; offset += 4) {
        memcpy(&word, bytes + offset, 4);
        if ((word & 255u) == 5u) continue;
        if ((word & 255u) != 1u || offset > size - 12) return -1;
        memcpy(&word, bytes + offset + 4, 4);
        *product = word >> 28;
        *log2_macs = word & 15u;
        return 0;
    }
    return -1;
}

struct ethosu_driver *ethosu_reserve_driver_ex(uint32_t product, uint32_t log2_macs)
{
    struct ethosu_driver *wanted = 0;
    for (unsigned i = 0; i < device_count; ++i) {
        uint32_t config = *(volatile uint32_t *)((uintptr_t)devices[i]->dev.reg + 0x28u);
        if ((config >> 28) == product && (config & 15u) == log2_macs) wanted = devices[i];
    }
    if (!wanted) return 0;
    // Preserve the common driver's semaphore accounting while selecting a device.
    struct ethosu_driver *held[2];
    unsigned count = 0;
    struct ethosu_driver *selected = 0;
    while (count < device_count) {
        struct ethosu_driver *driver = ethosu_reserve_driver();
        if (!driver) break;
        if (driver == wanted) { selected = driver; break; }
        held[count++] = driver;
    }
    while (count) ethosu_release_driver(held[--count]);
    return selected;
}
