#pragma once
#include <cstddef>

namespace dcops {
// English common-sign dictionary, not a general translation model. Exact match
// after ASCII case/spacing/basic punctuation normalization. No heap or fuzzy match.
const char* lookup_common_sign(const char* english);
size_t common_sign_count();
}
