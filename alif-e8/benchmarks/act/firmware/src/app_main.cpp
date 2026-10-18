#include "RTE_Components.h"
#include CMSIS_device_header
#include "board_config.h"
#include "Driver_IO.h"
#include "ospi_psram_xip.h"
#include "APS512XXN_PSRAM.h"
#include "sys_utils.h"
#include "ethosu_driver.h"
#include "arm_embedded_module.hpp"
#include <executorch/runtime/platform/runtime.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::runtime::EValue;
using executorch::runtime::Error;
using executorch::runtime::MemoryAllocator;
using arm::embedded::EmbeddedModule;

// Match the tested x16 PSRAM setup: both dies use fixed read latency.
// Keep the vendor pack unchanged; wrap its polling register-write helper.
extern "C" void __real_ospi_send_polling(OSPI_Type*, ospi_transfer_t*);
extern "C" void __wrap_ospi_send_polling(OSPI_Type* ospi, ospi_transfer_t* transfer) {
    const uint32_t* original = transfer->tx_buff;
    uint32_t fixed_latency[3];
    const bool mode_zero = transfer->tx_total_cnt == 3 &&
        original[0] == APS512XXN_CMD_MODE_REGISTER_WRITE &&
        original[1] == APS512XXN_MODE_REG0_ADDR;
    if (mode_zero) {
        memcpy(fixed_latency, original, sizeof(fixed_latency));
        fixed_latency[2] |= 0x2020;
        transfer->tx_buff = fixed_latency;
    }
    __real_ospi_send_polling(ospi, transfer);
    if (mode_zero) transfer->tx_buff = original + 3;
    for (volatile unsigned wait=0; wait<10000; ++wait) __NOP();
}

struct Benchmark {
    uint32_t magic, state, command, vision_bytes, head_bytes, error, run, phase;
    uint32_t front_ms[6], side_ms[6], head_ms[6], total_ms[6];
    float actions[300];
    uint32_t upload_destination, upload_bytes, upload_crc, upload_sequence, upload_ack, upload_operation;
};
extern "C" {
volatile Benchmark act_benchmark;
volatile uint32_t act_ticks;
volatile uint32_t act_npu_snapshot[163];
uint32_t act_command_bytes;
char act_last_error[512];
void SysTick_Handler(void) { ++act_ticks; }
// The driver's default bare-metal wait ignores timeout. Bound each NPU job
// using the millisecond SysTick, while preserving unbounded driver acquisition.
int ethosu_semaphore_take(void* semaphore, uint64_t timeout) {
    auto* count = static_cast<volatile uint8_t*>(semaphore);
    const uint32_t start = act_ticks;
    while (*count == 0) {
        if (timeout != ETHOSU_SEMAPHORE_WAIT_FOREVER &&
            static_cast<uint32_t>(act_ticks-start) >= timeout) return -1;
        __WFE();
    }
    --*count;
    return 0;
}
void ethosu_inference_begin(struct ethosu_driver* driver,void*) {
    // Conservative OSPI transaction limits: one outstanding read/write, 64B bursts.
    *reinterpret_cast<volatile uint32_t*>(NPU_HG_BASE+0x54)=0;
    __DSB();
    act_npu_snapshot[0]=act_ticks;
    act_npu_snapshot[1]=0;
    act_command_bytes=driver->job.custom_data_size;
}
void ethosu_inference_end(struct ethosu_driver* driver,void*) {
    act_npu_snapshot[1]=act_ticks;
    act_npu_snapshot[2]=driver->job.result;
    auto* registers=reinterpret_cast<volatile uint32_t*>(NPU_HG_BASE);
    for(unsigned i=0;i<160;++i) act_npu_snapshot[i+3]=registers[i];
}
void et_pal_emit_log_message(et_timestamp_t, et_pal_log_level_t, const char* filename,
                            const char*, size_t line, const char* message, size_t) {
    snprintf(act_last_error,sizeof(act_last_error),"%s:%u %s",filename,unsigned(line),message);
}
}
alignas(32) uint8_t temp_pool[1024*1024] __attribute__((section(".bss.ai_pool")));
alignas(32) uint8_t fast_pool[384*1024] __attribute__((section(".bss.ai_pool")));
alignas(32) uint8_t vision_method_pool[512*1024] __attribute__((section(".bss.ai_pool")));
alignas(32) uint8_t head_method_pool[512*1024] __attribute__((section(".bss.ai_pool")));
extern "C" {
unsigned char* ethosu_fast_scratch = fast_pool;
size_t ethosu_fast_scratch_size = sizeof(fast_pool);
uint64_t ethosu_address_remap(uint64_t address, int index) {
    if(index==-1 && address && act_command_bytes<=sizeof(temp_pool)) {
        memcpy(temp_pool,reinterpret_cast<void*>(static_cast<uintptr_t>(address)),act_command_bytes);
        SCB_CleanDCache_by_Addr(reinterpret_cast<uint32_t*>(temp_pool),act_command_bytes);
        return LocalToGlobal(temp_pool);
    }
    return address ? LocalToGlobal(reinterpret_cast<void*>(static_cast<uintptr_t>(address))) : 0;
}
unsigned int ethosu_config_select(uint64_t address, int) {
    return (address & 0xa0000000ULL) ? 2 : 0;
}
}

// Board PSRAM initialization is copied into this file by configure_act.py.
static int32_t reset_ram(OSPI_Type* ospi, AES_Type* aes) {
    aes_disable_xip(aes);
    ospi_disable(ospi);
    ospi->OSPI_TXFTLR = 0;
    return aps512xxn_psram_init(ospi, aes);
}

static int init_ram() {
    int error = board_clocks_config(CLKEN_SYSPLL_MASK | CLKEN_HFOSC_MASK |
        CLKEN_CLK_100M_MASK | (1U << CLKEN_CLK_400M) | (1U << CLKEN_SRAM1));
    if (error) return error;
    ospi_psram_xip_config config = {};
    config.instance = OSPI_INSTANCE_0;
    config.ram_init = reset_ram;
    config.ram_type = RAM_TYPE_PSRAM;
    config.config_mode = CONFIG_MODE_USER_PARAMETERS;
    config.bus_speed = 25000000;
    config.spi_frf = OSPI_SPI_FRF_DUAL_OCTAL;
    config.wait_cycles = 4;
    config.ddr_drive_edge = 1;
    config.rxds_delay = 6;
    config.signal_delay = 22;
    config.dfs = 32;
    error = ospi_psram_xip_init(&config);
    aes_set_rxds_delay(AES0, 11);
    ospi_disable(OSPI0);
    OSPI0->OSPI_XIP_CTRL = (OSPI0->OSPI_XIP_CTRL &
        ~((0x1fU << XIP_CTRL_WAIT_CYCLES_OFFSET) | (1U << XIP_CTRL_RXDS_VL_EN_OFFSET))) |
        (7U << XIP_CTRL_WAIT_CYCLES_OFFSET);
    ospi_enable(OSPI0);
    return error;
}



struct Input {
    TensorImpl::SizesType sizes[5];
    TensorImpl::DimOrderType order[5];
    TensorImpl::StridesType strides[5];
    std::unique_ptr<TensorImpl> impl;
    Input(std::initializer_list<int> dimensions, void* data) {
        int i=0;
        for (int size:dimensions) { sizes[i]=size; order[i]=i; ++i; }
        int stride=1;
        for (int j=i-1;j>=0;--j) { strides[j]=stride; stride*=sizes[j]; }
        impl=std::make_unique<TensorImpl>(ScalarType::Float,i,sizes,data,order,strides);
    }
    Tensor tensor() { return Tensor(impl.get()); }
};

static uint32_t crc32(const void* data,size_t bytes) {
    static uint32_t table[256];
    static bool initialized=false;
    if(!initialized) {
        for(uint32_t i=0;i<256;++i) {
            uint32_t value=i;
            for(int bit=0;bit<8;++bit) value=(value>>1)^((value&1)?0xedb88320U:0);
            table[i]=value;
        }
        initialized=true;
    }
    uint32_t value=0xffffffffU;
    auto* input=static_cast<const uint8_t*>(data);
    for(size_t i=0;i<bytes;++i) value=(value>>8)^table[(value^input[i])&255];
    return value^0xffffffffU;
}

extern "C" int app_main() {
    act_benchmark.magic=0x41435438;
    act_benchmark.state=1;
    SysTick_Config(SystemCoreClock/1000);
    if (init_ram()!=0) {act_benchmark.state=90; return -1;}
    act_benchmark.state=2;
    while (!act_benchmark.command) {
        if(act_benchmark.upload_sequence!=act_benchmark.upload_ack) {
            uint32_t address=act_benchmark.upload_destination, bytes=act_benchmark.upload_bytes;
            uint32_t operation=act_benchmark.upload_operation;
            if(address<0xa0000000U || bytes>64*1024*1024 || address>0xa4000000U-bytes ||
                (operation==1 && bytes>sizeof(temp_pool))) {
                act_benchmark.upload_crc=0;
            } else {
                auto* destination=reinterpret_cast<uint32_t*>(address);
                if(operation==1) {
                    SCB_InvalidateDCache_by_Addr(reinterpret_cast<uint32_t*>(temp_pool),bytes);
                    memcpy(destination,temp_pool,bytes);
                    SCB_CleanInvalidateDCache_by_Addr(destination,bytes);
                }
                act_benchmark.upload_crc=crc32(destination,bytes);
            }
            __DSB();
            act_benchmark.upload_ack=act_benchmark.upload_sequence;
        }
    }
    SCB_InvalidateDCache();
    act_benchmark.state=3;
    executorch::runtime::runtime_init();
    // PSRAM: model 0..36MiB, image 36..40MiB, features 40..42MiB,
    // planned tensors 42..52MiB, shared NPU scratch 52..64MiB.
    // Method metadata stays in internal SRAM, separate from NPU tensor writes.
    EmbeddedModule vision(reinterpret_cast<uint8_t*>(0xa0000000),act_benchmark.vision_bytes,
        std::make_unique<executorch::extension::BufferDataLoader>(reinterpret_cast<uint8_t*>(0xa0000000),act_benchmark.vision_bytes),
        std::make_unique<MemoryAllocator>(sizeof(vision_method_pool),vision_method_pool),
        std::make_unique<MemoryAllocator>(12*1024*1024,reinterpret_cast<uint8_t*>(0xa3400000)));
    EmbeddedModule head(reinterpret_cast<uint8_t*>(0xa0c00000),act_benchmark.head_bytes,
        std::make_unique<executorch::extension::BufferDataLoader>(reinterpret_cast<uint8_t*>(0xa0c00000),act_benchmark.head_bytes),
        std::make_unique<MemoryAllocator>(sizeof(head_method_pool),head_method_pool),
        std::make_unique<MemoryAllocator>(12*1024*1024,reinterpret_cast<uint8_t*>(0xa3400000)));
    using executorch::runtime::Span;
    using executorch::runtime::HierarchicalAllocator;
    // Runtime MethodMeta omits the export's unused buffer-zero entry.
    Span<uint8_t> vision_span(reinterpret_cast<uint8_t*>(0xa2a00000),5*1024*1024);
    Span<uint8_t> head_span(reinterpret_cast<uint8_t*>(0xa2f00000),5*1024*1024);
    HierarchicalAllocator vision_planned(Span<Span<uint8_t>>(&vision_span,1));
    HierarchicalAllocator head_planned(Span<Span<uint8_t>>(&head_span,1));
    act_benchmark.phase=1;
    auto e=vision.load_method("forward",&vision_planned);
    if(e!=Error::Ok) {act_benchmark.error=static_cast<uint32_t>(e);act_benchmark.state=91;return -1;}
    act_benchmark.phase=2;
    e=head.load_method("forward",&head_planned);
    if(e!=Error::Ok) {act_benchmark.error=static_cast<uint32_t>(e);act_benchmark.state=92;return -1;}
    Input image({1,3,480,640},reinterpret_cast<void*>(0xa2400000));
    Input front({1,512,15,20},reinterpret_cast<void*>(0xa2800000));
    Input side({1,512,15,20},reinterpret_cast<void*>(0xa28a0000));
    float state[6]={0};
    Input joints({1,6},state);
    std::vector<EValue> vision_inputs={image.tensor()};
    std::vector<EValue> head_inputs={front.tensor(),side.tensor(),joints.tensor()};
    act_benchmark.state=4;
    for(unsigned run=0;run<6;++run) {
        act_benchmark.run=run;
        uint32_t start=act_ticks;
        act_benchmark.phase=3;
        auto a=vision.execute("forward",vision_inputs);
        if(!a.ok()) {act_benchmark.error=static_cast<uint32_t>(a.error());act_benchmark.state=93;return -1;}
        memcpy(reinterpret_cast<void*>(0xa2800000),a.get()[0].toTensor().const_data_ptr<float>(),512*15*20*sizeof(float));
        uint32_t t1=act_ticks;
        act_benchmark.front_ms[run]=t1-start;
        act_benchmark.phase=4;
        auto b=vision.execute("forward",vision_inputs);
        if(!b.ok()) {act_benchmark.error=static_cast<uint32_t>(b.error());act_benchmark.state=94;return -1;}
        memcpy(reinterpret_cast<void*>(0xa28a0000),b.get()[0].toTensor().const_data_ptr<float>(),512*15*20*sizeof(float));
        uint32_t t2=act_ticks;
        act_benchmark.side_ms[run]=t2-t1;
        act_benchmark.phase=5;
        auto c=head.execute("forward",head_inputs);
        if(!c.ok()) {act_benchmark.error=static_cast<uint32_t>(c.error());act_benchmark.state=95;return -1;}
        uint32_t t3=act_ticks;
        act_benchmark.head_ms[run]=t3-t2;
        act_benchmark.total_ms[run]=t3-start;
        auto* out=c.get()[0].toTensor().const_data_ptr<float>();
        for(int i=0;i<300;++i) {
            if(!std::isfinite(out[i])) {act_benchmark.state=96;return -1;}
            act_benchmark.actions[i]=out[i];
        }
        printf("ACT run %u: %lu ms\n",run,(unsigned long)act_benchmark.total_ms[run]);
    }
    act_benchmark.state=5;
    while(true) {__WFI();}
}
