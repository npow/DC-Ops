#include "frontend.h"
#include <cmath>
#include <cstring>
#include <limits>

namespace dcops {
namespace {
struct Header { char magic[8]; uint32_t version, nodes, edges, strings; float unknown_score; uint32_t reserved; };
float halfFloat(uint16_t bits) {
  uint32_t sign = (uint32_t(bits) & 0x8000) << 16;
  int exponent = (bits >> 10) & 31;
  uint32_t mantissa = bits & 1023;
  uint32_t value;
  if (exponent == 0) {
    if (mantissa == 0) value = sign;
    else {
      exponent = -14;
      while (!(mantissa & 1024)) { mantissa <<= 1; --exponent; }
      value = sign | uint32_t(exponent + 127) << 23 | (mantissa & 1023) << 13;
    }
  } else if (exponent == 31) value = sign | 0x7f800000 | mantissa << 13;
  else value = sign | uint32_t(exponent + 112) << 23 | mantissa << 13;
  float result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}
}
Result Frontend::initialize(const void* lexicon, size_t bytes, const void* embeddings,
                            size_t embedding_bytes, const void* positions, size_t position_bytes) {
  if (!lexicon || bytes < sizeof(Header) || !embeddings || !positions) return Result::InvalidAsset;
  const auto* header = static_cast<const Header*>(lexicon);
  if (std::memcmp(header->magic, "DCOPSTK1", 8) || header->version != 1 || !header->nodes ||
      embedding_bytes != 32001 * kWidth * sizeof(uint16_t) || position_bytes != kLength * kWidth * sizeof(float)) return Result::InvalidAsset;
  const size_t required = sizeof(Header) + size_t(header->nodes) * sizeof(Node) +
      size_t(header->edges) * sizeof(Edge) + 32002 * sizeof(uint32_t) + header->strings;
  if (required != bytes) return Result::InvalidAsset;
  nodes_ = reinterpret_cast<const Node*>(header + 1);
  edges_ = reinterpret_cast<const Edge*>(nodes_ + header->nodes);
  pieces_ = reinterpret_cast<const uint32_t*>(edges_ + header->edges);
  strings_ = reinterpret_cast<const char*>(pieces_ + 32002);
  embedding_ = static_cast<const uint16_t*>(embeddings);
  positions_ = static_cast<const float*>(positions);
  unknown_score_ = header->unknown_score;
  return Result::Ok;
}
Result Frontend::normalize(const char* text, size_t& length) {
  length = 0;
  bool pending_space = true;
  auto append = [&](unsigned char byte) {
    if (length >= sizeof(normalized_) - 1) return false;
    normalized_[length++] = char(byte);
    return true;
  };
  const auto* cursor = reinterpret_cast<const unsigned char*>(text);
  size_t input_bytes = 0;
  while (*cursor) {
    uint32_t code = *cursor++;
    if (++input_bytes > 512) return Result::InputTooLong;
    if (code >= 128) {
      if (code < 0xc2 || code > 0xc3 || (*cursor & 0xc0) != 0x80) return Result::UnsupportedUnicode;
      code = ((code & 31) << 6) | (*cursor++ & 63);
      ++input_bytes;
    }
    if (code == ' ' || code == '\t' || code == '\n' || code == '\r' || code == 0xa0) { pending_space = true; continue; }
    if (code < 32 || (code >= 127 && code < 160) || code == 0xa8 || code == 0xaf || code == 0xb4 ||
        code == 0xb8 || code == 0xbc || code == 0xbd || code == 0xbe || code == 0xb5) return Result::UnsupportedUnicode;
    if (pending_space) {
      if (!append(0xe2) || !append(0x96) || !append(0x81)) return Result::InputTooLong;
      pending_space = false;
    }
    // These Latin-1 compatibility characters normalize to ASCII under NFKC.
    if (code == 0xaa) code = 'a';
    if (code == 0xba) code = 'o';
    if (code == 0xb2) code = '2';
    if (code == 0xb3) code = '3';
    if (code == 0xb9) code = '1';
    if (code < 128) { if (!append(code)) return Result::InputTooLong; }
    else if (!append(0xc0 | (code >> 6)) || !append(0x80 | (code & 63))) return Result::InputTooLong;
  }
  normalized_[length] = 0;
  return Result::Ok;
}
Result Frontend::tokenizeEnglish(const char* text, uint32_t ids[kLength], size_t& count) {
  if (!nodes_) return Result::InvalidAsset;
  size_t length;
  Result result = normalize(text, length);
  if (result != Result::Ok) return result;
  for (size_t i = 0; i <= length; ++i) scores_[i] = -std::numeric_limits<float>::infinity();
  scores_[0] = 0;
  for (size_t start = 0; start < length; ++start) {
    if (!std::isfinite(scores_[start])) continue;
    size_t char_length = (uint8_t(normalized_[start]) < 128) ? 1 : ((uint8_t(normalized_[start]) < 224) ? 2 : 3);
    bool single = false;
    uint32_t node_index = 0;
    for (size_t end = start; end < length; ++end) {
      const Node& node = nodes_[node_index];
      uint32_t child = 0;
      for (size_t edge = node.edge_begin; edge < node.edge_begin + node.edge_count; ++edge)
        if (edges_[edge].byte == uint8_t(normalized_[end])) { child = edges_[edge].child; break; }
      if (!child) break;
      node_index = child;
      const Node& found = nodes_[node_index];
      if (found.token == 0xffffffff) continue;
      size_t next = end + 1;
      if (next == start + char_length) single = true;
      float score = scores_[start] + found.score;
      if (score > scores_[next]) { scores_[next] = score; previous_[next] = start; tokens_[next] = found.token; }
    }
    if (!single) {
      size_t next = start + char_length;
      float score = scores_[start] + unknown_score_;
      if (score > scores_[next]) { scores_[next] = score; previous_[next] = start; tokens_[next] = 1; }
    }
  }
  // Backtrack in the normalized buffer's otherwise unused token work space.
  uint32_t reversed[512];
  size_t reversed_count = 0;
  for (size_t position = length; position; position = previous_[position]) {
    if (reversed_count == 512) return Result::InputTooLong;
    reversed[reversed_count++] = tokens_[position];
  }
  count = 0;
  while (reversed_count) {
    uint32_t token = reversed[--reversed_count];
    if (token == 1 && count && ids[count - 1] == 1) continue;
    if (count >= kLength - 1) return Result::InputTooLong;
    ids[count++] = token;
  }
  ids[count++] = kEos;
  for (size_t i = count; i < kLength; ++i) ids[i] = kPad;
  return Result::Ok;
}
Result Frontend::embed(const uint32_t* ids, float* output) const {
  if (!embedding_) return Result::InvalidAsset;
  for (size_t i = 0; i < kLength; ++i) {
    if (ids[i] > kPad) return Result::InvalidToken;
    for (size_t j = 0; j < kWidth; ++j)
      output[i * kWidth + j] = halfFloat(embedding_[ids[i] * kWidth + j]) * 16.0f + positions_[i * kWidth + j];
  }
  return Result::Ok;
}
Result Frontend::encoderInputs(const uint32_t ids[kLength], float* embeddings, float* mask) const {
  for (size_t i = 0; i < kLength; ++i) mask[i] = ids[i] == kPad ? -100.0f : -0.0f;
  return embed(ids, embeddings);
}
Result Frontend::decoderInputs(const uint32_t* prefix, size_t count, float* embeddings, float* last_position) const {
  if (!count || count > kLength) return Result::InputTooLong;
  uint32_t ids[kLength];
  for (size_t i = 0; i < kLength; ++i) { ids[i] = i < count ? prefix[i] : kPad; last_position[i] = i == count - 1 ? 1.0f : 0.0f; }
  return embed(ids, embeddings);
}
Result Frontend::detokenizeSpanish(const uint32_t* ids, size_t count, char* output, size_t capacity) const {
  if (!pieces_) return Result::InvalidAsset;
  size_t used = 0;
  for (size_t i = 0; i < count; ++i) {
    if (ids[i] > kPad) return Result::InvalidToken;
    if (ids[i] == kEos || ids[i] == kPad || ids[i] == 1) continue;
    for (size_t j = pieces_[ids[i]]; j < pieces_[ids[i] + 1]; ++j) {
      char c = strings_[j];
      if (uint8_t(c) == 0xe2 && j + 2 < pieces_[ids[i] + 1] && uint8_t(strings_[j + 1]) == 0x96 && uint8_t(strings_[j + 2]) == 0x81) {
        j += 2;
        if (!used) continue;
        c = ' ';
      }
      if (used + 1 >= capacity) return Result::OutputTooSmall;
      output[used++] = c;
    }
  }
  while (used && output[used - 1] == ' ') --used;
  if (used >= capacity) return Result::OutputTooSmall;
  output[used] = 0;
  return Result::Ok;
}
uint32_t Frontend::greedyToken(const float* logits) const {
  uint32_t best = 0;
  for (uint32_t i = 1; i < kPad; ++i) if (logits[i] > logits[best]) best = i;
  return best;
}
}
