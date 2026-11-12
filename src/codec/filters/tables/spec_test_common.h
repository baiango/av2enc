// Shared helpers for the recovered-table CPU reference tests.
//
// The oracles in the *_test.cpp files are independent transcriptions of the
// specification pseudocode.  They share only the normative generated tables
// and this tiny harness, never the implementation under test.

#ifndef AV2ENC_ENCODER_FILTERS_SPEC_TABLES_SPEC_TEST_COMMON_H_
#define AV2ENC_ENCODER_FILTERS_SPEC_TABLES_SPEC_TEST_COMMON_H_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace av2enc::filters::spec::test {

inline int g_checks = 0;
inline int g_failures = 0;

inline void ReportCheck(bool ok, const char* file, int line, const char* what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, what);
  }
}

#define CHECK(cond) ::av2enc::filters::spec::test::ReportCheck((cond), __FILE__, __LINE__, #cond)

inline int Finish(const char* name) {
  if (g_failures == 0) {
    std::printf("%s: %d checks passed\n", name, g_checks);
    return 0;
  }
  std::fprintf(stderr, "%s: %d/%d checks FAILED\n", name, g_failures, g_checks);
  return 1;
}

// Simple deterministic generator (xorshift32); independent of the spec RNG.
class TestRng {
 public:
  explicit TestRng(std::uint32_t seed) : state_(seed ? seed : 1) {}
  std::uint32_t Next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return state_;
  }
  int Range(int lo, int hi) {
    return lo + static_cast<int>(Next() % static_cast<std::uint32_t>(hi - lo + 1));
  }

 private:
  std::uint32_t state_;
};

struct Image {
  int width = 0;
  int height = 0;
  int stride = 0;
  std::vector<std::uint16_t> px;

  Image() = default;
  Image(int w, int h, int s = 0) { Reset(w, h, s); }

  void Reset(int w, int h, int s = 0) {
    width = w;
    height = h;
    stride = s == 0 ? w : s;
    px.assign(static_cast<std::size_t>(stride) * static_cast<std::size_t>(height), 0);
  }
  std::uint16_t& At(int x, int y) {
    return px[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
              static_cast<std::size_t>(x)];
  }
  std::uint16_t At(int x, int y) const {
    return px[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
              static_cast<std::size_t>(x)];
  }
  void Fill(int value) {
    for (auto& v : px) v = static_cast<std::uint16_t>(value);
  }
};

// Independent §7.20.2 get_source_sample(0, x, y).
inline int OracleSample(const Image& curr, const Image& cdef, int start_x,
                        int end_x, int start_y, int end_y, int stripe_start_y,
                        int stripe_end_y, int x, int y) {
  x = x < start_x ? start_x : (x > end_x ? end_x : x);
  y = y < start_y ? start_y : (y > end_y ? end_y : y);
  if (y < stripe_start_y) {
    if (y < stripe_start_y - 2) y = stripe_start_y - 2;
    return curr.At(x, y);
  }
  if (y > stripe_end_y) {
    if (y > stripe_end_y + 2) y = stripe_end_y + 2;
    return curr.At(x, y);
  }
  return cdef.At(x, y);
}

}  // namespace av2enc::filters::spec::test

#endif  // AV2ENC_ENCODER_FILTERS_SPEC_TABLES_SPEC_TEST_COMMON_H_
