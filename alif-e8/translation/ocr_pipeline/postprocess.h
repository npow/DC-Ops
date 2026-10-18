#pragma once
#include <cstddef>
#include <cstdint>

namespace dcops {
struct DetectedBox { float x, y, w, h, score; };
enum class OcrResult { Ok, InvalidInput, WorkspaceExceeded, OutputTooSmall };
struct DetectionOptions {
    float pixel_threshold = 0.3f;
    float box_threshold = 0.5f;
    unsigned minimum_pixels = 12;
    float expand_height = 0.35f;
};
struct DetectionWorkspace {
    uint8_t visited[(288 * 480 + 7) / 8];
    uint32_t queue[4096];
};
// Largest-area qualifying connected regions first, up to capacity (normally 4).
// Coordinates refer to this probability map, not source-camera pixels.
OcrResult detect_boxes(const float* probabilities, unsigned width, unsigned height,
                       DetectedBox* output, size_t capacity, size_t& count,
                       DetectionWorkspace& workspace,
                       const DetectionOptions& options = {});
// Raw logits [timesteps,97]. CTC blank=0. valid_width is preprocessing's 1..320.
// Confidence is mean softmax probability over emitted characters, or zero if empty.
OcrResult decodeCTC(const float* logits, unsigned timesteps, unsigned classes,
                    unsigned valid_width, char* output, size_t capacity,
                    float* confidence = nullptr);
extern const unsigned char kOcrCharacters[97];

// Retain a translation until its recognized text/geometry expires. Translation
// becomes eligible after >=1 second and >=2 matching observations; timestamps are milliseconds.
class TextStability {
 public:
    bool observe(const char* text, const DetectedBox& box, uint32_t now_ms);
    bool setTranslation(const char* translation, uint32_t now_ms);
    bool visible(uint32_t now_ms) const;
    bool confirmed() const { return confirmed_; }
    bool cachedFor(const char* source) const;
    unsigned matches() const { return matches_; }
    uint32_t stableElapsed() const { return matches_ ? candidate_ms_ - candidate_first_ms_ : 0; }
    const char* source() const { return stable_; }
    const char* translation() const { return translated_; }
    const DetectedBox& box() const { return display_box_; }
    void reset();
 private:
    char candidate_[256] = {}, stable_[256] = {}, translated_source_[256] = {};
    char translated_[512] = {};
    DetectedBox candidate_box_ = {}, stable_box_ = {}, display_box_ = {};
    uint32_t candidate_ms_ = 0, candidate_first_ms_ = 0, display_ms_ = 0;
    bool pending_ = false, confirmed_ = false;
    unsigned matches_ = 0;
};
}
