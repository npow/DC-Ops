#pragma once

#include <cstddef>
#include <cstdint>

struct StagedEngineAssets {
    const void *embeddings;
    size_t embedding_bytes;
    const void *lexicon;
    size_t lexicon_bytes;
    const void *positions;
    size_t position_bytes;
    const void *pipeline;
    size_t pipeline_bytes;
    void *planned;
    size_t planned_bytes;
};

extern "C" {
int staged_engine_init(const StagedEngineAssets *assets = nullptr);
int staged_engine_translate(const char *english, char *spanish, size_t capacity);
extern volatile uint32_t tr_text_state, tr_text_stage, tr_text_tokens, tr_text_error;
extern char tr_text_english[512], tr_text_spanish[512];
}
