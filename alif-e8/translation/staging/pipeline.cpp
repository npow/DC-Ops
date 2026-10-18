#include "pipeline.h"

#include <cstring>
#include <limits>

namespace dcops_translation {

bool translate(const float *encoder_embeddings, const float *encoder_mask,
               const Callbacks &callbacks, Workspace &workspace,
               uint32_t *tokens, size_t capacity, size_t *token_count) {
    if (!callbacks.run || !callbacks.embed_decoder || !tokens || !token_count) return false;
    *token_count = 0;
    std::memcpy(workspace.hidden[0], encoder_embeddings, sizeof(workspace.hidden[0]));
    unsigned active = 0;
    for (unsigned stage = 0; stage < 6; ++stage) {
        const float *inputs[] = {workspace.hidden[active], encoder_mask};
        const size_t sizes[] = {kHiddenElements, kLength};
        const float *output = nullptr;
        size_t elements = 0;
        if (!callbacks.run(callbacks.context, stage, inputs, sizes, 2, &output, &elements) ||
            !output || elements != kHiddenElements) return false;
        active ^= 1;
        std::memmove(workspace.hidden[active], output, sizeof(workspace.hidden[active]));
    }
    std::memcpy(workspace.encoded, workspace.hidden[active], sizeof(workspace.encoded));
    for (size_t i = 0; i < kLength; ++i) workspace.prefix[i] = kPad;
    size_t count = capacity < kLength-1 ? capacity : kLength-1;
    for (size_t position = 0; position < count; ++position) {
        active = 0;
        if (!callbacks.embed_decoder(callbacks.context, workspace.prefix, position+1,
                                     workspace.hidden[active], kHiddenElements)) return false;
        for (unsigned stage = 6; stage < 8; ++stage) {
            const float *inputs[] = {workspace.hidden[active], workspace.encoded, encoder_mask};
            const size_t sizes[] = {kHiddenElements, kHiddenElements, kLength};
            const float *output = nullptr;
            size_t elements = 0;
            if (!callbacks.run(callbacks.context, stage, inputs, sizes, 3, &output, &elements) ||
                !output || elements != kHiddenElements) return false;
            active ^= 1;
            std::memmove(workspace.hidden[active], output, sizeof(workspace.hidden[active]));
        }
        for (size_t i = 0; i < kLength; ++i) workspace.last_position[i] = i == position ? 1.0f : 0.0f;
        float best = -std::numeric_limits<float>::infinity();
        uint32_t selected = kEos;
        for (unsigned stage = 8; stage < 12; ++stage) {
            const float *inputs[] = {workspace.hidden[active], workspace.last_position};
            const size_t sizes[] = {kHiddenElements, kLength};
            const float *output = nullptr;
            size_t elements = 0;
            size_t expected = stage == 11 ? 8001 : 8000;
            if (!callbacks.run(callbacks.context, stage, inputs, sizes, 2, &output, &elements) ||
                !output || elements != expected) return false;
            for (size_t i = 0; i < elements; ++i) {
                uint32_t id = (stage-8)*8000+i;
                if (id != kPad && output[i] > best) { best = output[i]; selected = id; }
            }
        }
        tokens[position] = selected;
        workspace.prefix[position+1] = selected;
        *token_count = position+1;
        if (callbacks.token) callbacks.token(callbacks.context, selected, position);
        if (selected == kEos) return true;
    }
    return true;
}

} // namespace dcops_translation
