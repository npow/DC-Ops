#ifndef DCOPS_MIXED_DRIVER_H
#define DCOPS_MIXED_DRIVER_H
#include "ethosu_driver.h"
#ifdef __cplusplus
extern "C" {
#endif
int e8_register_npu(struct ethosu_driver *driver);
int ethosu_get_product_config_from_cop_data(const void *data, int size,
                                          uint32_t *product, uint32_t *log2_macs);
struct ethosu_driver *ethosu_reserve_driver_ex(uint32_t product, uint32_t log2_macs);
#ifdef __cplusplus
}
#endif
#endif
