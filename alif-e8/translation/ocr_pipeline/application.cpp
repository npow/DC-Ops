#include "application.h"
#include "../../src/trace.h"
#include "postprocess.h"
#include "common_signs.h"
#include "../camera.h"
#include "../live/live.h"
#include "../live/benchmark_service.h"
#include "../live/runner.h"
#include "../multi_npu/multi_runtime.h"
#include "../multi_npu/worker_mailbox.h"
#include "RTE_Components.h"
#include CMSIS_device_header
#include "../staging/live_engine.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>

extern "C" {
extern volatile uint32_t live_benchmark_enabled;
volatile uint32_t ocr_state, ocr_runs, ocr_error;
volatile uint32_t ocr_detector_ms, ocr_recognizer_ms, ocr_translator_ms;
volatile uint32_t ocr_boxes, ocr_confidence_x1000;
volatile uint32_t ocr_skip_detector = 0;
volatile uint32_t ocr_pause_display_during_translation = 1;
volatile uint32_t ocr_single_shot_enabled = 0;
volatile uint32_t ocr_sign_mode = 1;
volatile uint32_t ocr_sign_lookup_ms, ocr_sign_status, ocr_sign_matched;
volatile uint32_t ocr_capture_elapsed_ms;
volatile uint32_t ocr_capture_state, ocr_capture_requests, ocr_captured_frame;
volatile uint32_t ocr_capture_command;
volatile uint32_t ocr_stable_matches, ocr_stable_elapsed_ms, ocr_cache_hits;
volatile uint32_t ocr_stage_copy_ms, ocr_stage_service_ms;
volatile uint32_t ocr_recognizer_valid_width;
volatile uint32_t ocr_source_region[4];
float ocr_detected_box_coordinates[4][5];
char ocr_english[256], ocr_spanish[512];
}
namespace {
constexpr size_t kDetectorBytes = 667232;
constexpr unsigned kDetectorWidth = 128, kDetectorHeight = 224;
constexpr size_t kDetectorElements = 3 * kDetectorWidth * kDetectorHeight, kRecognizerElements = 3 * 48 * 320;
float* const detector_input_buffer = reinterpret_cast<float*>(live_temporary_pool);
float* const recognizer_input_buffer = reinterpret_cast<float*>(live_temporary_pool);
uint8_t* snapshot;
alignas(32) uint8_t method_pool[128 * 1024];
dcops::DetectionWorkspace detection_workspace;
dcops::TextStability text_stability;
bool busy, pending, overlay_visible;
bool capture_requested, capture_active;
uint32_t captured_present_frame, capture_request_started;
char shot_cache_source[256], shot_cache_spanish[512];
uint32_t last_capture, last_overlay;
dcops::CameraPlanes captured{};
dcops::DisplayStats stats{};
dcops::TranslationBox recognized_box{};
char recognized_label[272];
e8_npu_stats pass_npu_before[2]{};
bool pass_stats_valid;
uint32_t stage_started;
e8_mailbox* const he_mailbox = reinterpret_cast<e8_mailbox*>(0x02680000);
float* const he_logits = reinterpret_cast<float*>(0x025a0000);
uint32_t he_sequence;
bool he_inflight;
e8_response he_measurement{};
void invalidate(const void* pointer, size_t bytes) {
    if (!bytes) return;
    const uintptr_t first = reinterpret_cast<uintptr_t>(pointer) & ~uintptr_t(31);
    const uintptr_t end = (reinterpret_cast<uintptr_t>(pointer) + bytes + 31) & ~uintptr_t(31);
    SCB_InvalidateDCache_by_Addr(reinterpret_cast<void*>(first), end - first); __DSB();
}
void clean(const void* pointer, size_t bytes) {
    if (!bytes) return;
    const uintptr_t first = reinterpret_cast<uintptr_t>(pointer) & ~uintptr_t(31);
    const uintptr_t end = (reinterpret_cast<uintptr_t>(pointer) + bytes + 31) & ~uintptr_t(31);
    SCB_CleanDCache_by_Addr(reinterpret_cast<void*>(first), end - first); __DSB();
}
const e8_worker_ops he_ops{invalidate, clean, nullptr};
int waitForHe(e8_response& response) {
    const uint32_t started = live_now_ms();
    for (;;) {
        int result = e8_worker_result(he_mailbox, &he_ops, he_sequence, &response);
        if (result < 0) return result;
        if (result == 1) { he_inflight = false; return response.error; }
        if (live_now_ms() - started > 10000) return -110;
        live_service();
    }
}

void frame(const dcops::Image*, const dcops::CameraPlanes* planes, unsigned, void*) {
    if (live_benchmark_enabled) return;
    if (ocr_single_shot_enabled && !capture_requested) return;
    const uint32_t now = live_now_ms();
    if (busy || pending || he_inflight || (!capture_requested && now - last_capture < 500) || !planes || !planes->pixels ||
        planes->width != 320 || planes->height != 240) return;
    if (live_npu_set_fast_scratch_enabled(0)) { ocr_error = 106; return; }
    std::memcpy(snapshot, planes->pixels, 320 * 240 * 3);
    captured = {snapshot, planes->width, planes->height, planes->rotated_180};
    if (!dcops::detector_input(captured, detector_input_buffer, kDetectorElements, kDetectorWidth, kDetectorHeight)) {
        ocr_error = 1; return;
    }
    last_capture = now; pending = true; ocr_state = 2;
    captured_present_frame = live_frames + 1; ocr_captured_frame = captured_present_frame;
    capture_requested = false;
}
void stage(uintptr_t source, size_t bytes) {
    auto service = [&]() {
        const uint32_t began = live_now_ms();
        live_service();
        ocr_stage_service_ms += live_now_ms() - began;
    };
    service();
    uint32_t last_service = live_now_ms();
    for (size_t offset = 0; offset < bytes; offset += 128 * 1024) {
        const size_t count = std::min(size_t(128 * 1024), bytes - offset);
        const uint32_t began = live_now_ms();
        size_t copied = 0;
        if (((source + offset) | reinterpret_cast<uintptr_t>(live_model_staging + offset)) % 4 == 0) {
            auto* from = reinterpret_cast<const volatile uint32_t*>(source + offset);
            auto* to = reinterpret_cast<volatile uint32_t*>(live_model_staging + offset);
            for (size_t i = 0; i < count / 4; ++i) to[i] = from[i];
            copied = count & ~size_t(3);
        }
        auto* from = reinterpret_cast<const volatile uint8_t*>(source + offset);
        auto* to = reinterpret_cast<volatile uint8_t*>(live_model_staging + offset);
        for (; copied < count; ++copied) to[copied] = from[copied];
        ocr_stage_copy_ms += live_now_ms() - began;
        if (live_now_ms() - last_service >= 25) {
            service(); last_service = live_now_ms();
        }
    }
    service();
}

dcops::RunnerMemory memory() {
    return {method_pool, sizeof(method_pool), reinterpret_cast<uint8_t*>(0x025a0000), 0xe0000,
            live_model_staging + 1024 * 1024, sizeof(live_model_staging) - 1024 * 1024};
}
int recognize(dcops::Viewport region) {
    // A timeout leaves ownership with HE until its completion is observed.
    if (he_inflight) {
        e8_response previous{};
        int result = waitForHe(previous);
        if (result) return result;
    }
    int width = dcops::recognizer_input(captured, region, recognizer_input_buffer, kRecognizerElements);
    if (!width) return -101;
    ocr_recognizer_valid_width = width;
    ocr_source_region[0] = region.x; ocr_source_region[1] = region.y;
    ocr_source_region[2] = region.width; ocr_source_region[3] = region.height;
    trace_printf("OCR_CROP source=%d,%d,%d,%d valid_width=%d\r\n",
        region.x, region.y, region.width, region.height, width);
    ocr_state = 6;
    e8_request request{};
    request.magic = E8_WORKER_MAGIC; request.abi = E8_WORKER_ABI;
    if (++he_sequence == 0) ++he_sequence;
    request.sequence = he_sequence; request.model = 1; request.input_count = 1;
    request.inputs[0] = {static_cast<uint32_t>(reinterpret_cast<uintptr_t>(recognizer_input_buffer)),
                         kRecognizerElements * sizeof(float)};
    request.output = {0x025a0000, 40 * 97 * sizeof(float)};
    // This arena held HP planned tensors; release dirty cache lines before HE owns it.
    clean(he_logits, request.output.bytes);
    invalidate(he_logits, request.output.bytes);
    int result = e8_worker_submit(he_mailbox, &he_ops, &request);
    if (result == -3) {
        if (++he_sequence == 0) ++he_sequence;
        request.sequence = he_sequence;
        result = e8_worker_submit(he_mailbox, &he_ops, &request);
    }
    if (result) return result;
    he_inflight = true; ocr_state = 7;
    result = waitForHe(he_measurement);
    if (result) return result;
    if (he_measurement.output_bytes != 40 * 97 * sizeof(float)) return -102;
    float confidence = 0;
    auto status = dcops::decodeCTC(he_logits, 40, 97, unsigned(width),
                                  ocr_english, sizeof(ocr_english), &confidence);
    ocr_confidence_x1000 = uint32_t(std::clamp(confidence, 0.0f, 1.0f) * 1000);
    return status == dcops::OcrResult::Ok ? 0 : -103;
}
int showFeedback(const char* status, bool show_english) {
    const auto viewport = dcops::camera_viewport(320, 240, 480, 800);
    dcops::TranslationBox updates[2];
    size_t count = 0;
    if (show_english && recognized_label[0]) {
        updates[count] = recognized_box;
        updates[count++].translated_utf8 = recognized_label;
    }
    updates[count++] = {float(viewport.x), 206.0f, float(viewport.x + viewport.width), 239.0f, status};
    return live_set_boxes(updates, count);
}
void finishCapturedResult(const char* status) {
    if (!capture_active) return;
    const auto viewport = dcops::camera_viewport(320, 240, 480, 800);
    dcops::TranslationBox updates[2];
    size_t count = 0;
    if (!ocr_error && ocr_spanish[0]) {
        updates[count] = recognized_box;
        updates[count++].translated_utf8 = ocr_spanish;
    } else if (recognized_label[0]) {
        updates[count] = recognized_box;
        updates[count++].translated_utf8 = recognized_label;
    }
    updates[count++] = {float(viewport.x), 198.0f, float(viewport.x + viewport.width), 239.0f, status};
    live_set_boxes(updates, count);
    live_refresh_display();
    ocr_capture_elapsed_ms = live_now_ms() - capture_request_started;
    ocr_capture_state = 3;
}
void publishStats(const e8_npu_stats before[2]) {
    uint64_t cycles = he_measurement.npu_cycles, active = he_measurement.npu_active;
    unsigned elapsed = he_measurement.elapsed_us / 1000;
    bool measured = false;
    for (unsigned i = 0; i < 2; ++i) {
        e8_npu_stats after{};
        if (live_npu_stats(i, &after)) continue;
        if (after.total_cycles < before[i].total_cycles || after.active_cycles < before[i].active_cycles) continue;
        cycles += after.total_cycles - before[i].total_cycles;
        active += after.active_cycles - before[i].active_cycles;
        elapsed += after.elapsed_ms - before[i].elapsed_ms;
        measured = true;
    }
    stats.detector_ms = ocr_detector_ms;
    stats.recognizer_ms = ocr_recognizer_ms;
    stats.translator_ms = ocr_translator_ms;
    stats.npu_ms = measured ? elapsed : 0;
    stats.cpu_ms = 0;  // No CPU-only baseline has been measured by this application.
    stats.npu_active_percent = cycles ? int(std::min(uint64_t(100), active * 100 / cycles)) : -1;
    live_set_stats(&stats);
}
int fail(int code) {
    const uint32_t state = ocr_state;
    const uint32_t elapsed = live_now_ms() - stage_started;
    if (state == 3 || state == 4) ocr_detector_ms = elapsed;
    else if (state == 6 || state == 7) ocr_recognizer_ms = elapsed;
    else if (state == 8) ocr_translator_ms = elapsed;
    trace_printf("OCR_FAIL run=%lu state=%lu error=%d detector_ms=%lu recognizer_ms=%lu translator_ms=%lu\r\n",
        static_cast<unsigned long>(ocr_runs), static_cast<unsigned long>(state), code,
        static_cast<unsigned long>(ocr_detector_ms), static_cast<unsigned long>(ocr_recognizer_ms),
        static_cast<unsigned long>(ocr_translator_ms));
    text_stability.observe("", {}, live_now_ms());
    ocr_stable_matches = ocr_stable_elapsed_ms = 0;
    if (pass_stats_valid) publishStats(pass_npu_before);
    if (state == 8 && ocr_english[0]) {
        showFeedback("Translation unavailable", true);
        last_overlay = live_now_ms(); overlay_visible = true;
    } else if (!overlay_visible) showFeedback("Recognition unavailable", false);
    ocr_error = static_cast<uint32_t>(code);
    ocr_state = 0xe0;
    finishCapturedResult(state == 8 ? "Translation unavailable\nPress joystick for live" :
                                      "Capture processing failed\nPress joystick for live");
    busy = false;
    return code;
}
}
extern "C" int ocr_app_init(void) {
    size_t snapshot_bytes = 0;
    snapshot = static_cast<uint8_t*>(live_npu_fast_scratch_storage(&snapshot_bytes));
    if (!snapshot || snapshot_bytes < 320 * 240 * 3) return -106;
    int error = live_npu_set_fast_scratch_enabled(0);
    if (error) return error;
    busy = pending = overlay_visible = false;
    capture_requested = capture_active = false; captured_present_frame = 0;
    ocr_capture_state = ocr_capture_requests = ocr_captured_frame = ocr_capture_command = 0;
    ocr_sign_lookup_ms = ocr_sign_status = ocr_sign_matched = 0;
    shot_cache_source[0] = shot_cache_spanish[0] = 0;
    capture_request_started = ocr_capture_elapsed_ms = 0;
    last_capture = last_overlay = 0;
    he_sequence = 0; he_inflight = false; he_measurement = {};
    ocr_state = 1; ocr_runs = ocr_error = 0;
    text_stability.reset(); ocr_stable_matches = ocr_stable_elapsed_ms = ocr_cache_hits = 0;
    ocr_detector_ms = ocr_recognizer_ms = ocr_translator_ms = 0;
    ocr_boxes = ocr_confidence_x1000 = 0;
    ocr_english[0] = ocr_spanish[0] = recognized_label[0] = 0;
    stats = {}; stats.npu_active_percent = -1;
    pass_stats_valid = false; stage_started = 0;
    live_set_stats(&stats);
    if (ocr_single_shot_enabled) {
        const dcops::TranslationBox guide{91, 90, 228, 150, nullptr};
        live_set_ocr_region(&guide, 0);
        showFeedback("Press joystick to capture", false);
    } else showFeedback("Looking for text\nHold text in center", false);
    live_set_frame_callback(frame, nullptr);
    return 0;
}
extern "C" int ocr_app_service(void) {
    if (busy) return 0;
    if (ocr_capture_command) {
        const uint32_t command = ocr_capture_command;
        ocr_capture_command = 0;
        const int result = command == 1 ? ocr_app_button_press() : -119;
        if (result) { ocr_error = static_cast<uint32_t>(result); return result; }
    }
    if (he_inflight) {
        e8_response discarded{};
        int completion = e8_worker_result(he_mailbox, &he_ops, he_sequence, &discarded);
        if (completion != 1) return 0;
        he_inflight = false;
    }
    if (!capture_active && overlay_visible && uint32_t(live_now_ms() - last_overlay) > 10000) {
        showFeedback("Looking for text\nHold text in center", false); overlay_visible = false;
    }
    if (!pending) return 0;
    if (ocr_single_shot_enabled && live_frames < captured_present_frame) return 0;
    if (ocr_single_shot_enabled) {
        int pause_error = live_set_display_paused(1);
        if (pause_error) return fail(pause_error);
        capture_active = true; ocr_capture_state = 2;
        showFeedback("Reading captured image...", false);
        live_refresh_display();
    }
    busy = true; pending = false; ++ocr_runs; ocr_error = 0;
    ocr_boxes = ocr_confidence_x1000 = 0;
    ocr_sign_lookup_ms = ocr_sign_status = ocr_sign_matched = 0;
    ocr_stage_copy_ms = ocr_stage_service_ms = 0;
    trace_printf("OCR_START run=%lu frame=%lu\r\n", static_cast<unsigned long>(ocr_runs),
        static_cast<unsigned long>(live_frames));
    ocr_english[0] = ocr_spanish[0] = recognized_label[0] = 0;
    ocr_detector_ms = ocr_recognizer_ms = ocr_translator_ms = 0;
    he_measurement = {};
    std::memset(pass_npu_before, 0, sizeof(pass_npu_before));
    for (unsigned i = 0; i < 2; ++i) live_npu_stats(i, &pass_npu_before[i]);
    pass_stats_valid = true;
    dcops::DetectedBox boxes[4]; size_t count = 0;
    uint32_t started = live_now_ms();
    stage_started = started;
    const bool skip_detector = ocr_skip_detector != 0;
    if (!skip_detector) {
        ocr_state = 3;
        stage(0xa2400000, kDetectorBytes);
        dcops::Runner runner;
        int result = runner.load(live_model_staging, kDetectorBytes, memory());
        if (result) return fail(result);
        ocr_state = 4;
        dcops::RunnerInput input{detector_input_buffer, kDetectorElements * sizeof(float)};
        result = runner.execute(&input, 1);
        if (result) return fail(result);
        auto output = runner.output();
        if (!output.data || output.bytes != kDetectorWidth * kDetectorHeight * sizeof(float)) return fail(-104);
        auto status = dcops::detect_boxes(static_cast<const float*>(output.data), kDetectorWidth, kDetectorHeight,
                                          boxes, 4, count, detection_workspace);
        if (status != dcops::OcrResult::Ok) return fail(-105);
    }
    ocr_detector_ms = skip_detector ? 0 : live_now_ms() - started;
    ocr_state = 5; ocr_boxes = count;
    std::memset(ocr_detected_box_coordinates, 0, sizeof(ocr_detected_box_coordinates));
    for (size_t i = 0; i < count; ++i) {
        ocr_detected_box_coordinates[i][0] = boxes[i].x;
        ocr_detected_box_coordinates[i][1] = boxes[i].y;
        ocr_detected_box_coordinates[i][2] = boxes[i].w;
        ocr_detected_box_coordinates[i][3] = boxes[i].h;
        ocr_detected_box_coordinates[i][4] = boxes[i].score;
    }
    trace_printf("OCR_DETECTOR run=%lu boxes=%lu ms=%lu skipped=%u copy_ms=%lu service_ms=%lu\r\n", static_cast<unsigned long>(ocr_runs),
        static_cast<unsigned long>(ocr_boxes), static_cast<unsigned long>(ocr_detector_ms), unsigned(skip_detector),
        static_cast<unsigned long>(ocr_stage_copy_ms), static_cast<unsigned long>(ocr_stage_service_ms));
    dcops::TranslationBox source_box;
    if (count) {
        const auto& box = boxes[0];
        source_box = dcops::detector_box_to_source(captured, box.x, box.y, box.x + box.w, box.y + box.h, nullptr, kDetectorWidth, kDetectorHeight);
    } else {
        // Real center-stripe OCR still runs when DB finds no region.
        source_box = dcops::detector_box_to_source(captured, 0, kDetectorHeight * .375f, kDetectorWidth,
                                                    kDetectorHeight * .625f, nullptr, kDetectorWidth, kDetectorHeight);
    }
    const int x0 = std::clamp(int(std::floor(source_box.x0)), 0, captured.width - 1);
    const int y0 = std::clamp(int(std::floor(source_box.y0)), 0, captured.height - 1);
    const int x1 = std::clamp(int(std::ceil(source_box.x1)), x0 + 1, captured.width);
    const int y1 = std::clamp(int(std::ceil(source_box.y1)), y0 + 1, captured.height);
    const dcops::TranslationBox crop_box{float(x0),float(y0),float(x1),float(y1),nullptr};
    live_set_ocr_region(&crop_box,count!=0);
    started = live_now_ms(); stage_started = started;
    int result = recognize({x0, y0, x1 - x0, y1 - y0});
    ocr_recognizer_ms = live_now_ms() - started;
    trace_printf("OCR_RECOGNIZER run=%lu result=%d english=%s confidence_x1000=%lu ms=%lu\r\n",
        static_cast<unsigned long>(ocr_runs), result, ocr_english,
        static_cast<unsigned long>(ocr_confidence_x1000), static_cast<unsigned long>(ocr_recognizer_ms));
    if (result) return fail(result);
    if (!ocr_english[0]) {
        text_stability.observe("", {}, live_now_ms());
        ocr_stable_matches = ocr_stable_elapsed_ms = 0;
        if (!overlay_visible) showFeedback("Looking for text\nHold text in center", false);
        publishStats(pass_npu_before); ocr_state = 1;
        finishCapturedResult("No text read\nPress joystick for live");
        busy = false; return 0;
    }
    recognized_box = source_box;
    std::snprintf(recognized_label, sizeof(recognized_label), "EN: %s", ocr_english);
    if (ocr_sign_mode) {
        const uint32_t lookup_started = live_now_ms();
        const char* spanish = dcops::lookup_common_sign(ocr_english);
        ocr_sign_lookup_ms = live_now_ms() - lookup_started;
        ocr_sign_status = spanish ? 1 : 2; ocr_sign_matched = spanish ? 1 : 0;
        ocr_translator_ms = 0;
        trace_printf("OCR_SIGN input=%s matched=%u lookup_ms=%lu\r\n", ocr_english, unsigned(spanish != nullptr),
                     static_cast<unsigned long>(ocr_sign_lookup_ms));
        publishStats(pass_npu_before);
        if (spanish) {
            std::strcpy(ocr_spanish, spanish);
            source_box.translated_utf8 = ocr_spanish;
            if (capture_active) finishCapturedResult("Common sign match\nPress joystick for live");
            else {
                result = live_set_boxes(&source_box, 1);
                if (result) return fail(result);
            }
            ocr_state = 9;
        } else {
            showFeedback("No matching sign", true);
            finishCapturedResult("No matching sign\nPress joystick for live");
            ocr_state = 12;
        }
        last_overlay = live_now_ms(); overlay_visible = true; busy = false; return 0;
    }
    if (capture_active && shot_cache_source[0] && std::strcmp(shot_cache_source, ocr_english) == 0) {
        std::strcpy(ocr_spanish, shot_cache_spanish); ++ocr_cache_hits;
        publishStats(pass_npu_before); ocr_state = 9;
        finishCapturedResult("Press joystick for live"); busy = false; return 0;
    }
    if (!capture_active) {
    const dcops::DetectedBox observed_box{source_box.x0, source_box.y0,
        source_box.x1 - source_box.x0, source_box.y1 - source_box.y0, float(ocr_confidence_x1000) / 1000};
    const bool translate = text_stability.observe(ocr_english, observed_box, live_now_ms());
    ocr_stable_matches = text_stability.matches(); ocr_stable_elapsed_ms = text_stability.stableElapsed();
    if (!text_stability.confirmed()) {
        result = showFeedback("Hold text steady", true);
        if (result) return fail(result);
        last_overlay = live_now_ms(); overlay_visible = true;
        publishStats(pass_npu_before); ocr_state = 10; busy = false; return 0;
    }
    if (!translate && text_stability.cachedFor(ocr_english)) {
        std::strcpy(ocr_spanish, text_stability.translation());
        source_box.translated_utf8 = ocr_spanish;
        result = live_set_boxes(&source_box, 1);
        if (result) return fail(result);
        ++ocr_cache_hits; last_overlay = live_now_ms(); overlay_visible = true;
        trace_printf("OCR_CACHE_HIT hits=%lu source=%s\r\n", static_cast<unsigned long>(ocr_cache_hits), ocr_english);
        publishStats(pass_npu_before); ocr_state = 9; busy = false; return 0;
    }
    }
    result = showFeedback("Translating...", true);
    if (result) return fail(result);
    last_overlay = live_now_ms(); overlay_visible = true;
    if (capture_active) live_refresh_display();
    const uint32_t feedback_frame = live_frames, feedback_started = live_now_ms();
    while (!live_display_paused && live_frames == feedback_frame && live_now_ms() - feedback_started < 500)
        if (live_service() < 0) break;
    ocr_state = 8; started = live_now_ms(); stage_started = started;
    const uint32_t previous_pause = live_display_paused;
    bool restore_display = false;
    result = live_npu_set_fast_scratch_enabled(1);
    if (!result && ocr_pause_display_during_translation) {
        result = live_set_display_paused(1);
        restore_display = result == 0;
    }
    if (!result) result = staged_engine_translate(ocr_english, ocr_spanish, sizeof(ocr_spanish));
    if (restore_display) {
        int display_error = live_set_display_paused(previous_pause ? 1 : 0);
        if (!result) result = display_error;
    }
    int scratch_error = live_npu_set_fast_scratch_enabled(0);
    if (!result) result = scratch_error;
    ocr_translator_ms = live_now_ms() - started;
    trace_printf("OCR_TRANSLATOR run=%lu result=%d spanish=%s ms=%lu\r\n",
        static_cast<unsigned long>(ocr_runs), result, ocr_spanish, static_cast<unsigned long>(ocr_translator_ms));
    publishStats(pass_npu_before);
    if (result) return fail(result);
    if (ocr_spanish[0]) {
        if (capture_active) {
            std::strcpy(shot_cache_source, ocr_english);
            std::strcpy(shot_cache_spanish, ocr_spanish);
        } else text_stability.setTranslation(ocr_spanish, live_now_ms());
        source_box.translated_utf8 = ocr_spanish;
        result = live_set_boxes(&source_box, 1);
        if (result) return fail(result);
        last_overlay = live_now_ms(); overlay_visible = true; ocr_state = 9;
    } else {
        text_stability.observe("", {}, live_now_ms());
        ocr_stable_matches = ocr_stable_elapsed_ms = 0;
        showFeedback("No translation returned", true);
        last_overlay = live_now_ms(); overlay_visible = true; ocr_state = 1;
    }
    finishCapturedResult(ocr_spanish[0] ? "Press joystick for live" :
                                            "No translation returned\nPress joystick for live");
    busy = false;
    return 0;
}

extern "C" int ocr_app_button_press(void) {
    if (!ocr_single_shot_enabled || live_benchmark_enabled || busy) return -1;
    if (ocr_capture_state == 3 || capture_active || ocr_capture_state == 1) {
        if (he_inflight) return -1;
        int error = live_set_display_paused(0);
        if (error) return error;
        pending = capture_requested = capture_active = overlay_visible = false;
        ocr_capture_state = 0; ocr_state = 1;
        ocr_english[0] = ocr_spanish[0] = recognized_label[0] = 0;
        const dcops::TranslationBox guide{91, 90, 228, 150, nullptr};
        live_set_ocr_region(&guide, 0);
        showFeedback("Press joystick to capture", false);
        return 0;
    }
    if (live_display_paused) return -1;
    pending = false; capture_requested = true; ++ocr_capture_requests;
    capture_request_started = live_now_ms(); ocr_capture_elapsed_ms = 0;
    ocr_capture_state = 1; ocr_state = 11;
    showFeedback("Capturing next frame...", false);
    return 0;
}

extern "C" void ocr_translation_progress(unsigned tokens, uint32_t elapsed_ms) {
    if (!busy || ocr_state != 8) return;
    ocr_translator_ms = elapsed_ms;
    stats.translator_ms = elapsed_ms;
    if (live_display_paused) return;
    char progress[96];
    std::snprintf(progress, sizeof(progress), "Translating: %u tokens\n%lu s elapsed", tokens,
                  static_cast<unsigned long>(elapsed_ms / 1000));
    showFeedback(progress, true);
    live_set_stats(&stats);
    live_service();
}
