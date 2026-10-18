#include "worker_mailbox.h"
#include <cassert>
static unsigned executions;
static void cache(const void *, size_t) {}
static int execute(const e8_request *request, e8_response *response) {
    ++executions;
    response->output_bytes = request->model == 99 ? request->output.bytes + 1 : 4;
    return 0;
}
int main() {
    e8_mailbox mailbox{};
    const e8_worker_ops ops{cache, cache, execute};
    assert(e8_worker_poll(&mailbox, &ops) == 0);
    mailbox.request.magic = E8_WORKER_MAGIC;
    mailbox.request.abi = E8_WORKER_ABI;
    mailbox.request.sequence = 1;
    mailbox.request.output = {0x02000000, 16};
    assert(e8_worker_poll(&mailbox, &ops) == 1);
    assert(mailbox.response.sequence == 1 && mailbox.response.error == 0 && executions == 1);
    assert(e8_worker_poll(&mailbox, &ops) == 0 && executions == 1);
    mailbox.request.sequence = 2;
    mailbox.request.input_count = E8_WORKER_INPUTS + 1;
    assert(e8_worker_poll(&mailbox, &ops) == 1 && mailbox.response.error == -2 && executions == 1);
    mailbox.request.sequence = 3;
    mailbox.request.input_count = 0;
    mailbox.request.model = 99;
    assert(e8_worker_poll(&mailbox, &ops) == 1 && mailbox.response.error == -3 && executions == 2);
    assert(mailbox.response.output_bytes == 0);
    e8_request request=mailbox.request;
    request.sequence=4; request.model=1;
    assert(e8_worker_submit(&mailbox,&ops,&request)==0);
    e8_response response{};
    assert(e8_worker_result(&mailbox,&ops,4,&response)==0);
    assert(e8_worker_poll(&mailbox,&ops)==1);
    assert(e8_worker_result(&mailbox,&ops,4,&response)==1 && response.output_bytes==4);
    assert(e8_worker_submit(&mailbox,&ops,&request)==-3);
    request.sequence=5;
    assert(e8_worker_submit(&mailbox,&ops,&request)==0);
    request.sequence=6;
    assert(e8_worker_submit(&mailbox,&ops,&request)==-2);
}
