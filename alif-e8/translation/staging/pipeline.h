#pragma once

#include <cstddef>
#include <cstdint>

namespace dcops_translation {

constexpr size_t kLength = 32;
constexpr size_t kHiddenElements = 8192;
constexpr uint32_t kPad = 32000;
constexpr uint32_t kEos = 0;

struct Workspace {
    float hidden[2][kHiddenElements];
    float encoded[kHiddenElements];
    float last_position[kLength];
    uint32_t prefix[kLength];
};

struct Callbacks {
    void *context;
    bool (*run)(void *context, unsigned stage, const float *const *inputs,
                const size_t *input_elements, size_t input_count,
                const float **output, size_t *output_elements);
    bool (*embed_decoder)(void *context, const uint32_t *prefix, size_t count,
                          float *embeddings, size_t elements);
    void (*token)(void *context, uint32_t token, size_t index);
};

bool translate(const float *encoder_embeddings, const float *encoder_mask,
               const Callbacks &callbacks, Workspace &workspace,
               uint32_t *tokens, size_t capacity, size_t *token_count);

} // namespace dcops_translation
