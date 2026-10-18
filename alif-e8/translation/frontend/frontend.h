#pragma once
#include <cstddef>
#include <cstdint>

namespace dcops {
constexpr size_t kLength = 32;
constexpr size_t kWidth = 256;
constexpr uint32_t kPad = 32000;
constexpr uint32_t kEos = 0;
enum class Result { Ok, InvalidAsset, UnsupportedUnicode, InputTooLong, InvalidToken, OutputTooSmall };

// Assets are immutable little-endian buffers; keep them mapped for this object's lifetime.
// One object per concurrent caller. Working storage is fixed; no heap allocation.
class Frontend {
 public:
  Result initialize(const void* lexicon, size_t lexicon_bytes,
                    const void* embeddings, size_t embedding_bytes,
                    const void* positions, size_t position_bytes);
  Result tokenizeEnglish(const char* text, uint32_t ids[kLength], size_t& count);
  Result encoderInputs(const uint32_t ids[kLength], float embeddings[kLength * kWidth],
                       float mask[kLength]) const;
  Result decoderInputs(const uint32_t* prefix, size_t count,
                       float embeddings[kLength * kWidth], float last_position[kLength]) const;
  Result detokenizeSpanish(const uint32_t* ids, size_t count, char* output,
                           size_t capacity) const;
  uint32_t greedyToken(const float logits[32001]) const;

 private:
  struct Node { uint32_t edge_begin; uint16_t edge_count, reserved; uint32_t token; float score; };
  struct Edge { uint32_t child; uint8_t byte, reserved[3]; };
  const Node* nodes_ = nullptr;
  const Edge* edges_ = nullptr;
  const uint32_t* pieces_ = nullptr;
  const char* strings_ = nullptr;
  const uint16_t* embedding_ = nullptr;
  const float* positions_ = nullptr;
  float unknown_score_ = 0;
  char normalized_[1024];
  float scores_[1025];
  uint16_t previous_[1025];
  uint32_t tokens_[1025];
  Result normalize(const char* text, size_t& length);
  Result embed(const uint32_t* ids, float* output) const;
};
}
