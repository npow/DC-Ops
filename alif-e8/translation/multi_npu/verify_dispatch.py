"""Validate driver routing and real Vela payload headers without touching hardware."""
from pathlib import Path
import subprocess
import tempfile
from build_driver import DRIVER, ROOT

source = r'''
#include "ethosu_driver.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int e8_register_npu(struct ethosu_driver *);
struct ethosu_driver *ethosu_reserve_driver_ex(uint32_t, uint32_t);
int ethosu_get_product_config_from_cop_data(const void *, int, uint32_t *, uint32_t *);
static struct ethosu_driver drivers[2];
static uint32_t regs[2][16];
struct ethosu_driver *ethosu_reserve_driver(void) {
    for (int i=0;i<2;++i) if (!drivers[i].reserved) { drivers[i].reserved=true; return &drivers[i]; }
    return NULL;
}
void ethosu_release_driver(struct ethosu_driver *driver) { driver->reserved=false; }
int main(int argc, char **argv) {
    for (int i=0;i<2;++i) { drivers[i].dev.reg=(void *)regs[i]; assert(e8_register_npu(&drivers[i])==0); }
    regs[0][10]=8; regs[1][10]=0x20000008;
    assert(ethosu_reserve_driver_ex(2,8)==&drivers[1]);
    assert(!drivers[0].reserved); ethosu_release_driver(&drivers[1]);
    assert(ethosu_reserve_driver_ex(0,8)==&drivers[0]); ethosu_release_driver(&drivers[0]);
    assert(ethosu_reserve_driver_ex(0,7)==NULL);
    uint32_t product, macs;
    assert(ethosu_get_product_config_from_cop_data(NULL,16,&product,&macs)<0);
    unsigned checked=0;
    for(int f=1; f<argc; ++f) {
        FILE *file=fopen(argv[f],"rb"); assert(file);
        fseek(file,0,SEEK_END); long length=ftell(file); rewind(file);
        unsigned char *data=malloc(length); assert(fread(data,1,length,file)==(size_t)length); fclose(file);
        int found=0;
        for(long offset=0;offset<=length-16;++offset) {
            if(memcmp(data+offset,"COP1",4)==0 && ethosu_get_product_config_from_cop_data(data+offset,length-offset,&product,&macs)==0) {
                printf("%s product=%u log2_macs=%u\n",argv[f],product,macs); found=1; ++checked; break;
            }
        }
        assert(found); free(data);
    }
    assert(checked==4); puts("routing and all four real model headers passed");
}
'''
with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    test = directory / 'verify.c'
    test.write_text(source)
    executable = directory / 'verify'
    subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-I'+str(DRIVER/'include'), str(test), str(Path(__file__).with_name('dispatch.c')), '-o', str(executable)], check=True)
    models = ['text_detector_ethos-u55-256.pte', 'text_recognizer_ethos-u55-128.pte', 'encoder_u85.pte', 'decoder_u85.pte']
    subprocess.run([str(executable), *[str(ROOT/'artifacts/translator'/model) for model in models]], check=True)
