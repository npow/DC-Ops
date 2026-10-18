#ifndef DCOPS_WORKER_MAILBOX_H
#define DCOPS_WORKER_MAILBOX_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define E8_WORKER_ABI 1u
#define E8_WORKER_MAGIC 0x45384f43u
#define E8_WORKER_INPUTS 4u

struct e8_buffer { uint32_t address; uint32_t bytes; };
struct e8_request {
    uint32_t magic;
    uint32_t abi;
    uint32_t sequence;
    uint32_t model;
    uint32_t input_count;
    struct e8_buffer inputs[E8_WORKER_INPUTS];
    struct e8_buffer output;
    uint8_t reserved[68];
};
struct e8_response {
    uint32_t sequence;
    int32_t error;
    uint32_t output_bytes;
    uint32_t elapsed_us;
    uint64_t npu_cycles;
    uint32_t npu_active;
    uint32_t mac_active;
    uint32_t npu_idle;
    uint8_t reserved[92];
};
struct __attribute__((aligned(128))) e8_mailbox {
    struct e8_request request;
    struct e8_response response;
};

// Cache callbacks include a DSB after maintenance. Addresses must be global aliases.
struct e8_worker_ops {
    void (*invalidate)(const void *address, size_t bytes);
    void (*clean)(const void *address, size_t bytes);
    int (*execute)(const struct e8_request *, struct e8_response *);
};
int e8_worker_poll(struct e8_mailbox *mailbox, const struct e8_worker_ops *ops);
int e8_worker_submit(struct e8_mailbox *mailbox, const struct e8_worker_ops *ops, const struct e8_request *request);
int e8_worker_result(struct e8_mailbox *mailbox, const struct e8_worker_ops *ops, uint32_t sequence, struct e8_response *response);

#ifdef __cplusplus
}
static_assert(sizeof(e8_request) == 128, "request must occupy separate cache lines");
static_assert(sizeof(e8_response) == 128, "response must occupy separate cache lines");
static_assert(offsetof(e8_mailbox, response) == 128, "cache lines must not overlap");
#endif
#endif
