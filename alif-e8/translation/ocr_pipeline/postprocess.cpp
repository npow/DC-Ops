#include "postprocess.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace dcops {
OcrResult detect_boxes(const float* probabilities, unsigned width, unsigned height,
                       DetectedBox* output, size_t capacity, size_t& count,
                       DetectionWorkspace& workspace, const DetectionOptions& options) {
    count = 0;
    if (!probabilities || !output || !capacity || !width || !height ||
        width > 288 || height > 480 || !std::isfinite(options.pixel_threshold) ||
        !std::isfinite(options.box_threshold) || !std::isfinite(options.expand_height) ||
        options.pixel_threshold < 0 || options.pixel_threshold > 1 ||
        options.box_threshold < 0 || options.box_threshold > 1 || options.expand_height < 0)
        return OcrResult::InvalidInput;
    const unsigned pixels = width * height;
    std::memset(workspace.visited, 0, (pixels + 7) / 8);
    auto marked = [&](unsigned index) { return workspace.visited[index >> 3] & (1u << (index & 7)); };
    auto mark = [&](unsigned index) { workspace.visited[index >> 3] |= 1u << (index & 7); };
    auto foreground = [&](unsigned index) { return std::isfinite(probabilities[index]) && probabilities[index] >= options.pixel_threshold; };
    for (unsigned seed = 0; seed < pixels; ++seed) {
        if (marked(seed) || !foreground(seed)) continue;
        unsigned head = 0, tail = 1, queued = 1;
        workspace.queue[0] = seed;
        mark(seed);
        unsigned left = seed % width, right = left, top = seed / width, bottom = top, area = 0;
        float total = 0;
        while (queued) {
            unsigned index = workspace.queue[head];
            head = (head + 1) % 4096; --queued;
            unsigned x = index % width, y = index / width;
            left = std::min(left, x); right = std::max(right, x);
            top = std::min(top, y); bottom = std::max(bottom, y);
            ++area; total += probabilities[index];
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                int nx = int(x) + dx, ny = int(y) + dy;
                if ((!dx && !dy) || nx < 0 || ny < 0 || nx >= int(width) || ny >= int(height)) continue;
                unsigned next = unsigned(ny) * width + unsigned(nx);
                if (marked(next) || !foreground(next)) continue;
                if (queued == 4096) { count = 0; return OcrResult::WorkspaceExceeded; }
                mark(next); workspace.queue[tail] = next; tail = (tail + 1) % 4096; ++queued;
            }
        }
        float score = float(total / area);
        if (area < options.minimum_pixels || score < options.box_threshold || right - left < 2 || bottom - top < 2) continue;
        const float margin = std::max(2.0f, float(bottom - top + 1) * options.expand_height);
        float x0 = std::max(0.0f, float(left) - margin), y0 = std::max(0.0f, float(top) - margin);
        float x1 = std::min(float(width), float(right + 1) + margin), y1 = std::min(float(height), float(bottom + 1) + margin);
        DetectedBox box{x0, y0, x1 - x0, y1 - y0, score};
        // DB can split a word into separate character components at low resolution.
        // Join neighboring components on the same text line before choosing a crop.
        for (size_t i = 0; i < count;) {
            const DetectedBox& other = output[i];
            float overlap = std::min(box.y + box.h, other.y + other.h) - std::max(box.y, other.y);
            float gap = std::max(0.0f, std::max(box.x, other.x) - std::min(box.x + box.w, other.x + other.w));
            float maximum_height = std::max(box.h, other.h), minimum_height = std::min(box.h, other.h);
            float center_delta = std::abs(box.y + box.h * .5f - other.y - other.h * .5f);
            if (overlap >= minimum_height * .5f && center_delta <= maximum_height * .5f &&
                maximum_height <= minimum_height * 2.5f && gap <= maximum_height * .65f) {
                float first_area = box.w * box.h, second_area = other.w * other.h;
                float merged_score = (box.score * first_area + other.score * second_area) / (first_area + second_area);
                float merged_x = std::min(box.x, other.x), merged_y = std::min(box.y, other.y);
                box = {merged_x, merged_y, std::max(box.x + box.w, other.x + other.w) - merged_x,
                       std::max(box.y + box.h, other.y + other.h) - merged_y, merged_score};
                for (size_t j = i + 1; j < count; ++j) output[j - 1] = output[j];
                --count;
                i = 0;
            } else ++i;
        }
        size_t position = 0;
        while (position < count && output[position].w * output[position].h >= box.w * box.h) ++position;
        if (position >= capacity) continue;
        if (count < capacity) ++count;
        for (size_t i = count - 1; i > position; --i) output[i] = output[i - 1];
        output[position] = box;
    }
    return OcrResult::Ok;
}
OcrResult decodeCTC(const float* logits, unsigned timesteps, unsigned classes,
                    unsigned valid_width, char* output, size_t capacity, float* confidence) {
    if (confidence) *confidence = 0;
    if (output && capacity) output[0] = 0;
    if (!logits || !output || !capacity || !timesteps || timesteps > 4096 || classes != 97 ||
        !valid_width || valid_width > 320) return OcrResult::InvalidInput;
    unsigned steps = (timesteps * valid_width + 319) / 320, previous = 0, emitted = 0;
    size_t length = 0;
    float probability_sum = 0;
    for (unsigned t = 0; t < steps; ++t) {
        const float* row = logits + t * classes;
        unsigned best = 0;
        for (unsigned c = 0; c < classes; ++c) {
            if (!std::isfinite(row[c])) { output[0] = 0; return OcrResult::InvalidInput; }
            if (row[c] > row[best]) best = c;
        }
        if (best && best != previous) {
            if (length + 1 >= capacity) { output[0] = 0; return OcrResult::OutputTooSmall; }
            output[length++] = char(kOcrCharacters[best]);
            float denominator = 0;
            for (unsigned c = 0; c < classes; ++c) denominator += std::exp(row[c] - row[best]);
            probability_sum += 1.0f / denominator;
            ++emitted;
        }
        previous = best;
    }
    while (length && output[length - 1] == ' ') --length;
    size_t first = 0;
    while (first < length && output[first] == ' ') ++first;
    if (first) { std::memmove(output, output + first, length - first); length -= first; }
    output[length] = 0;
    if (confidence && length && emitted) *confidence = probability_sum / emitted;
    return OcrResult::Ok;
}
namespace {
bool copyText(char* destination, const char* source) {
    if (!source) return false;
    size_t length = 0;
    while (length < 256 && source[length]) ++length;
    if (length == 256) return false;
    std::memcpy(destination, source, length + 1);
    return true;
}
bool nearby(const DetectedBox& a, const DetectedBox& b) {
    float intersection_w = std::max(0.0f, std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x));
    float intersection_h = std::max(0.0f, std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y));
    float intersection = intersection_w * intersection_h;
    float union_area = a.w * a.h + b.w * b.h - intersection;
    return union_area > 0 && intersection / union_area >= 0.5f;
}
}
bool TextStability::observe(const char* text, const DetectedBox& box, uint32_t now_ms) {
    confirmed_ = false;
    if (!text || !text[0] || !std::isfinite(box.x) || !std::isfinite(box.y) ||
        !std::isfinite(box.w) || !std::isfinite(box.h) || box.w <= 0 || box.h <= 0) {
        candidate_[0] = 0; matches_ = 0; pending_ = false;
        return false;
    }
    // Compare against the first ROI, so gradual motion cannot accumulate into a trigger.
    if (std::strcmp(text, candidate_) == 0 && uint32_t(now_ms - candidate_ms_) <= 2500 && nearby(box, candidate_box_)) {
        if (matches_ < 0xffff) ++matches_;
    } else {
        if (!copyText(candidate_, text)) { matches_ = 0; return false; }
        matches_ = 1; candidate_first_ms_ = now_ms; candidate_box_ = box;
    }
    candidate_ms_ = now_ms;
    if (matches_ < 2 || uint32_t(now_ms - candidate_first_ms_) < 1000) return false;
    if (!copyText(stable_, text)) return false;
    confirmed_ = true; stable_box_ = box;
    if (cachedFor(text)) {
        display_ms_ = now_ms; display_box_ = box; pending_ = false;
        return false;
    }
    pending_ = true;
    return true;
}
bool TextStability::cachedFor(const char* source) const {
    return source && translated_[0] && std::strcmp(source, translated_source_) == 0;
}
bool TextStability::setTranslation(const char* translation, uint32_t now_ms) {
    if (!confirmed_ || !pending_ || !stable_[0] || !translation || !translation[0]) return false;
    size_t length = 0;
    while (length < sizeof(translated_) && translation[length]) ++length;
    if (length == sizeof(translated_)) return false;
    std::memcpy(translated_, translation, length + 1);
    copyText(translated_source_, stable_);
    display_ms_ = now_ms; display_box_ = stable_box_; pending_ = false;
    return true;
}
bool TextStability::visible(uint32_t now_ms) const {
    return translated_[0] && uint32_t(now_ms - display_ms_) <= 5000;
}
void TextStability::reset() {
    candidate_[0] = stable_[0] = translated_[0] = translated_source_[0] = 0;
    matches_ = 0; candidate_ms_ = candidate_first_ms_ = display_ms_ = 0;
    pending_ = confirmed_ = false;
    candidate_box_ = stable_box_ = display_box_ = {};
}
}
