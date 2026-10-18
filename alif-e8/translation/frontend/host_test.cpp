#include "frontend.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

std::vector<char> readFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) { std::cerr << "Cannot open " << path << '\n'; std::exit(1); }
  return {std::istreambuf_iterator<char>(input), {}};
}
void require(bool valid, const std::string& message) {
  if (!valid) { std::cerr << message << '\n'; std::exit(1); }
}
struct Fixture {
  std::vector<char> bytes;
  size_t position = 0;
  uint32_t integer() { uint32_t value; block(&value, 4); return value; }
  void block(void* output, size_t count) {
    require(position + count <= bytes.size(), "Truncated fixture");
    std::memcpy(output, bytes.data() + position, count); position += count;
  }
  std::string text() { size_t count = integer(); std::string value(count, '\0'); block(value.data(), count); return value; }
};
int main(int argc, char** argv) {
  require(argc == 2, "Usage: frontend_host_test ASSET_DIRECTORY");
  const std::string root = std::string(argv[1]) + "/";
  auto lexicon = readFile(root + "lexicon.bin"), embeddings = readFile(root + "embeddings_fp16.bin"), positions = readFile(root + "positions_fp32.bin");
  dcops::Frontend frontend;
  require(frontend.initialize(lexicon.data(), lexicon.size(), embeddings.data(), embeddings.size(), positions.data(), positions.size()) == dcops::Result::Ok, "Asset initialization");
  Fixture fixture{readFile(root + "fixtures.bin")};
  uint32_t denominator = fixture.integer();
  float max_error = 0;
  for (uint32_t sample = 0; sample < denominator; ++sample) {
    std::string text = fixture.text(); size_t expected_count = fixture.integer(), actual_count = 0;
    uint32_t expected[32], actual[32]; fixture.block(expected, sizeof(expected));
    auto result = frontend.tokenizeEnglish(text.c_str(), actual, actual_count);
    if (result != dcops::Result::Ok || actual_count != expected_count || std::memcmp(expected, actual, sizeof(expected))) {
      std::cerr << "Tokenizer mismatch: " << text << " expected:";
      for (size_t i = 0; i < expected_count; ++i) std::cerr << ' ' << expected[i];
      std::cerr << " actual:"; for (size_t i = 0; i < actual_count; ++i) std::cerr << ' ' << actual[i];
      std::cerr << '\n'; return 1;
    }
    float reference[8192], reference_mask[32], output[8192], mask[32];
    fixture.block(reference, sizeof(reference)); fixture.block(reference_mask, sizeof(reference_mask));
    require(frontend.encoderInputs(actual, output, mask) == dcops::Result::Ok, "Encoder embedding");
    for (size_t i = 0; i < 8192; ++i) max_error = std::max(max_error, std::abs(reference[i] - output[i]));
    require(!std::memcmp(reference_mask, mask, sizeof(mask)), "Encoder mask mismatch");
  }
  uint32_t decode_count = fixture.integer();
  for (uint32_t sample = 0; sample < decode_count; ++sample) {
    uint32_t count = fixture.integer(); std::vector<uint32_t> ids(count); fixture.block(ids.data(), count * 4);
    std::string expected = fixture.text(); char output[512];
    require(frontend.detokenizeSpanish(ids.data(), count, output, sizeof(output)) == dcops::Result::Ok && expected == output, "Detokenizer mismatch: " + expected + " actual " + output);
  }
  uint32_t prefix_count = fixture.integer(); std::vector<uint32_t> prefix(prefix_count); fixture.block(prefix.data(), prefix_count * 4);
  float reference[8192], reference_last[32], output[8192], last[32];
  fixture.block(reference, sizeof(reference)); fixture.block(reference_last, sizeof(reference_last));
  require(frontend.decoderInputs(prefix.data(), prefix_count, output, last) == dcops::Result::Ok, "Decoder inputs");
  for (size_t i = 0; i < 8192; ++i) max_error = std::max(max_error, std::abs(reference[i] - output[i]));
  require(!std::memcmp(reference_last, last, sizeof(last)), "Decoder position mask mismatch");
  require(max_error == 0, "Embedding mismatch: " + std::to_string(max_error));
  uint32_t ids[32]; size_t count = 0;
  require(frontend.tokenizeEnglish("e\xcc\x81", ids, count) == dcops::Result::UnsupportedUnicode, "Combining marks must be rejected");
  require(frontend.tokenizeEnglish("\xe6\x97\xa5", ids, count) == dcops::Result::UnsupportedUnicode, "Unsupported Unicode must be rejected");
  std::string huge(600, 'a');
  require(frontend.tokenizeEnglish(huge.c_str(), ids, count) == dcops::Result::InputTooLong, "Oversized input must be rejected");
  require(frontend.decoderInputs(prefix.data(), 0, output, last) == dcops::Result::InputTooLong, "Empty prefix must be rejected");
  float logits[32001] = {}; logits[32000] = 100; logits[42] = 2;
  require(frontend.greedyToken(logits) == 42, "Greedy decoding must exclude PAD");
  std::cout << "{\"source_token_matches\":" << denominator << ",\"source_denominator\":" << denominator
            << ",\"spanish_decode_matches\":" << decode_count << ",\"spanish_decode_denominator\":" << decode_count
            << ",\"encoder_decoder_embedding_max_error\":" << max_error
            << ",\"frontend_workspace_bytes\":" << sizeof(frontend) << ",\"boundary_checks\":5}\n";
}
