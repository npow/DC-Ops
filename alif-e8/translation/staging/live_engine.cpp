#include "live_engine.h"

#include "pipeline.h"
#include "../frontend/frontend.h"
#include "../live/live.h"
#include "../live/runner.h"
#include "../multi_npu/multi_runtime.h"
#include "../../artifacts/translator/spatial_staging/pipeline_index.h"
#include "RTE_Components.h"
#include CMSIS_device_header

#include <algorithm>
#include <cstring>

extern "C" void ocr_translation_progress(unsigned tokens,uint32_t elapsed_ms) __attribute__((weak));
extern "C" {
volatile uint32_t tr_text_truncated;
volatile uint32_t tr_stage_program_address,tr_stage_prepare_ms,tr_stage_execute_ms;
volatile uint32_t tr_stage_copy_ms,tr_stage_service_ms;
}

namespace {
dcops::Frontend frontend;
dcops_translation::Workspace workspace;
alignas(32) uint8_t method_pool[128*1024];
alignas(32) float encoder_embeddings[dcops_translation::kHiddenElements];
alignas(32) float stage_output[dcops_translation::kHiddenElements];
float encoder_mask[dcops_translation::kLength];
float decoder_position[dcops_translation::kLength];
uint32_t source_ids[dcops_translation::kLength], generated[dcops_translation::kLength];
StagedEngineAssets mapping;
bool initialized = false;
int engine_error;
uint32_t translation_started,last_progress;
void progress(bool force=false) {
    const uint32_t now=live_now_ms();
    if (!force && now-last_progress<1000) return;
    last_progress=now;
    if (ocr_translation_progress) ocr_translation_progress(tr_text_tokens,now-translation_started);
}

int fail(int error) {
    engine_error = error;
    tr_text_error = static_cast<uint32_t>(error);
    tr_text_state = 0xe0;
    return error;
}

bool run_stage(void *, unsigned stage, const float *const *inputs,
               const size_t *elements, size_t input_count,
               const float **output, size_t *output_elements) {
    if (stage >= DCOPS_PIPELINE_STAGE_COUNT) { fail(-20); return false; }
    const auto &asset = dcops_packed_stages[stage];
    if (input_count != asset.input_count || asset.bytes>sizeof(live_model_staging) ||
        asset.offset > mapping.pipeline_bytes || asset.bytes > mapping.pipeline_bytes-asset.offset) {
        fail(-21); return false;
    }
    tr_text_stage = stage;
    tr_text_state = stage < 6 ? 4 : 5;
    const auto *source = static_cast<const uint8_t *>(mapping.pipeline)+asset.offset;
    if (reinterpret_cast<uintptr_t>(source)%32) { fail(-21); return false; }
    tr_stage_program_address=reinterpret_cast<uintptr_t>(live_model_staging);
    tr_stage_prepare_ms=tr_stage_execute_ms=tr_stage_copy_ms=tr_stage_service_ms=0;
    auto service_video=[]() {
        const uint32_t started=live_now_ms();
        const int result=live_service();
        tr_stage_service_ms+=live_now_ms()-started;
        return result;
    };
    if (service_video()<0) { fail(-22); return false; }
    uint32_t last_service=live_now_ms();
    for (size_t offset=0;offset<asset.bytes;offset+=128*1024) {
        const size_t bytes=std::min(size_t(128*1024),size_t(asset.bytes)-offset);
        const uint32_t started=live_now_ms();
        auto *destination=reinterpret_cast<volatile uint32_t *>(live_model_staging+offset);
        const auto *words=reinterpret_cast<const volatile uint32_t *>(source+offset);
        for (size_t word=0;word<bytes/sizeof(uint32_t);++word) destination[word]=words[word];
        for (size_t tail=bytes&~size_t(3);tail<bytes;++tail) live_model_staging[offset+tail]=source[offset+tail];
        tr_stage_copy_ms+=live_now_ms()-started;
        if (live_now_ms()-last_service>=25) {
            if (service_video()<0) { fail(-22); return false; }
            last_service=live_now_ms();
        }
    }
    SCB_CleanDCache_by_Addr(live_model_staging,(asset.bytes+31)&~uint32_t(31)); __DSB();
    if (service_video()<0) { fail(-22); return false; }
    const uint32_t prepare_started=live_now_ms();
    dcops::Runner runner;
    dcops::RunnerMemory memory{method_pool, sizeof(method_pool),
        static_cast<uint8_t *>(mapping.planned), mapping.planned_bytes,
        live_temporary_pool, sizeof(live_temporary_pool)};
    int result = runner.load(live_model_staging, asset.bytes, memory);
    if (result) { fail(-1000-result); return false; }
    dcops::RunnerInput arguments[4];
    for (size_t i = 0; i < input_count; ++i) {
        arguments[i] = {const_cast<float *>(inputs[i]), elements[i]*sizeof(float)};
    }
    if (!live_npu_fast_scratch_enabled()) { fail(-25); return false; }
    tr_stage_prepare_ms=live_now_ms()-prepare_started;
    const uint32_t execute_started=live_now_ms();
    result = runner.execute(arguments, input_count);
    tr_stage_execute_ms=live_now_ms()-execute_started;
    if (result) { fail(-2000-result); return false; }
    auto value = runner.output();
    if (!value.data || value.bytes != asset.output_elements*sizeof(float) ||
        value.bytes > sizeof(stage_output)) { fail(-23); return false; }
    std::memcpy(stage_output, value.data, value.bytes);
    *output = stage_output;
    *output_elements = asset.output_elements;
    progress();
    live_service();
    return true;
}

bool embed_decoder(void *, const uint32_t *prefix, size_t count,
                   float *embeddings, size_t elements) {
    if (elements != dcops_translation::kHiddenElements) { fail(-24); return false; }
    auto result = frontend.decoderInputs(prefix, count, embeddings, decoder_position);
    if (result != dcops::Result::Ok) { fail(-3000-static_cast<int>(result)); return false; }
    live_service();
    return true;
}

void token_ready(void *, uint32_t, size_t index) {
    tr_text_tokens = index+1;
    frontend.detokenizeSpanish(workspace.prefix+1, index+1,
                              tr_text_spanish, sizeof(tr_text_spanish));
    tr_text_spanish[sizeof(tr_text_spanish)-1] = 0;
    progress(true);
    live_service();
}
} // namespace

extern "C" {
volatile uint32_t tr_text_state, tr_text_stage, tr_text_tokens, tr_text_error;
char tr_text_english[512], tr_text_spanish[512];

int staged_engine_init(const StagedEngineAssets *assets) {
    initialized = false;
    engine_error = 0;
    tr_text_state = 1;
    tr_text_error = tr_text_tokens = tr_text_truncated = 0;
    tr_text_stage = 0xffffffffU;
    tr_text_english[0] = tr_text_spanish[0] = 0;
    mapping = assets ? *assets : StagedEngineAssets{
        reinterpret_cast<const void *>(0xa0000000), 16384512,
        reinterpret_cast<const void *>(0xa1000000), 2337119,
        reinterpret_cast<const void *>(0xa1240000), 32768,
        reinterpret_cast<const void *>(0xa1300000), DCOPS_PIPELINE_BYTES,
        reinterpret_cast<void *>(0x025a0000), 512*1024};
    if (!mapping.pipeline || reinterpret_cast<uintptr_t>(mapping.pipeline)%32 ||
        mapping.pipeline_bytes < DCOPS_PIPELINE_BYTES ||
        !mapping.planned || reinterpret_cast<uintptr_t>(mapping.planned)%32 ||
        mapping.planned_bytes < 192032) return fail(-1);
    auto result = frontend.initialize(mapping.lexicon, mapping.lexicon_bytes,
        mapping.embeddings, mapping.embedding_bytes, mapping.positions, mapping.position_bytes);
    if (result != dcops::Result::Ok) return fail(-100-static_cast<int>(result));
    initialized = true;
    tr_text_state = 2;
    return 0;
}

int staged_engine_translate(const char *english, char *spanish, size_t capacity) {
    if (!initialized || !english || !spanish || !capacity) return fail(-2);
    spanish[0] = 0;
    translation_started=last_progress=live_now_ms();
    tr_text_state = 3;
    tr_text_error = tr_text_tokens = tr_text_truncated = 0;
    tr_text_spanish[0] = 0;
    engine_error = 0;
    std::strncpy(tr_text_english, english, sizeof(tr_text_english)-1);
    tr_text_english[sizeof(tr_text_english)-1] = 0;
    size_t source_count = 0;
    auto result = frontend.tokenizeEnglish(english, source_ids, source_count);
    if (result != dcops::Result::Ok) return fail(-100-static_cast<int>(result));
    result = frontend.encoderInputs(source_ids, encoder_embeddings, encoder_mask);
    if (result != dcops::Result::Ok) return fail(-200-static_cast<int>(result));
    size_t generated_count = 0;
    dcops_translation::Callbacks callbacks{nullptr, run_stage, embed_decoder, token_ready};
    const size_t token_limit=std::min(dcops_translation::kLength-1,
                                    std::max(size_t(8),source_count*3));
    if (!dcops_translation::translate(encoder_embeddings, encoder_mask, callbacks, workspace,
                                     generated, token_limit, &generated_count)) {
        return engine_error ? engine_error : fail(-3);
    }
    result = frontend.detokenizeSpanish(generated, generated_count, spanish, capacity);
    if (result != dcops::Result::Ok) return fail(-300-static_cast<int>(result));
    std::strncpy(tr_text_spanish, spanish, sizeof(tr_text_spanish)-1);
    tr_text_spanish[sizeof(tr_text_spanish)-1] = 0;
    if (!generated_count || generated[generated_count-1]!=dcops_translation::kEos) {
        tr_text_truncated=1;
        return fail(-26);
    }
    tr_text_state = 6;
    tr_text_stage = 0xffffffffU;
    return 0;
}
}
