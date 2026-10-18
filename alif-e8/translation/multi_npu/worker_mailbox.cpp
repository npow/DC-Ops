#include "worker_mailbox.h"
#include <cstring>

int e8_worker_poll(e8_mailbox *mailbox, const e8_worker_ops *ops)
{
    if (!mailbox || !ops || !ops->invalidate || !ops->clean || !ops->execute) return -1;
    ops->invalidate(&mailbox->request, sizeof(mailbox->request));
    const e8_request request = mailbox->request;
    if (!request.sequence || request.sequence == mailbox->response.sequence) return 0;
    e8_response response{};
    response.sequence = request.sequence;
    if (request.magic != E8_WORKER_MAGIC || request.abi != E8_WORKER_ABI ||
        request.input_count > E8_WORKER_INPUTS || !request.output.address || !request.output.bytes) {
        response.error = -2;
    } else {
        response.error = 0;
        for (uint32_t i = 0; i < request.input_count; ++i) {
            if (!request.inputs[i].address || !request.inputs[i].bytes) { response.error = -2; break; }
            ops->invalidate(reinterpret_cast<void *>(static_cast<uintptr_t>(request.inputs[i].address)), request.inputs[i].bytes);
        }
        if (!response.error) {
            response.error = ops->execute(&request, &response);
            if (response.output_bytes > request.output.bytes) { response.error = -3; response.output_bytes = 0; }
            if (!response.error) ops->clean(reinterpret_cast<void *>(static_cast<uintptr_t>(request.output.address)), response.output_bytes);
        }
    }
    response.sequence = 0;
    mailbox->response = response;
    ops->clean(&mailbox->response, sizeof(mailbox->response));
    mailbox->response.sequence = request.sequence;
    ops->clean(&mailbox->response, 32);
    return 1;
}

int e8_worker_submit(e8_mailbox *mailbox,const e8_worker_ops *ops,const e8_request *request)
{
    if (!mailbox || !ops || !ops->invalidate || !ops->clean || !request ||
        !request->sequence || request->input_count>E8_WORKER_INPUTS) return -1;
    ops->invalidate(&mailbox->response,sizeof(mailbox->response));
    if (mailbox->request.sequence && mailbox->request.sequence!=mailbox->response.sequence) return -2;
    if (request->sequence==mailbox->request.sequence) return -3;
    for (unsigned i=0;i<request->input_count;++i) {
        if (!request->inputs[i].address || !request->inputs[i].bytes) return -1;
        ops->clean(reinterpret_cast<void *>(static_cast<uintptr_t>(request->inputs[i].address)),request->inputs[i].bytes);
    }
    e8_request publish=*request;
    publish.sequence=0;
    mailbox->request=publish;
    ops->clean(&mailbox->request,sizeof(mailbox->request));
    mailbox->request.sequence=request->sequence;
    ops->clean(&mailbox->request,32);
    return 0;
}
int e8_worker_result(e8_mailbox *mailbox,const e8_worker_ops *ops,uint32_t sequence,e8_response *response)
{
    if (!mailbox || !ops || !ops->invalidate || !response || !sequence) return -1;
    ops->invalidate(&mailbox->response,sizeof(mailbox->response));
    if (mailbox->response.sequence!=sequence) return 0;
    *response=mailbox->response;
    if (!response->error && response->output_bytes) {
        if (response->output_bytes>mailbox->request.output.bytes) return -2;
        ops->invalidate(reinterpret_cast<void *>(static_cast<uintptr_t>(mailbox->request.output.address)),response->output_bytes);
    }
    return 1;
}
