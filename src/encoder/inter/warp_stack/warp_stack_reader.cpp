// Hand-written snapshot reader.  See warp_stack_io.h for the format and the
// reason this is a separate translation unit.

#include "encoder/inter/warp_stack/warp_stack_io.h"

#include <cstdlib>
#include <string>

namespace av2enc::inter::warp {
namespace {

class Scanner {
 public:
  explicit Scanner(const std::string& text) : text_(text) {}

  bool token(std::string* out) {
    skip_space();
    if (pos_ >= text_.size()) return false;
    const std::size_t begin = pos_;
    while (pos_ < text_.size() && !is_space(text_[pos_])) ++pos_;
    *out = text_.substr(begin, pos_ - begin);
    return true;
  }

  bool integer(int* out) {
    std::string t;
    if (!token(&t)) return false;
    if (t.empty()) return false;
    std::size_t i = 0;
    bool negative = false;
    if (t[0] == '-') {
      negative = true;
      i = 1;
      if (t.size() == 1) return false;
    }
    long long value = 0;
    for (; i < t.size(); ++i) {
      if (t[i] < '0' || t[i] > '9') return false;
      value = value * 10 + (t[i] - '0');
      if (value > 2147483648LL) return false;
    }
    if (negative) value = -value;
    if (value < -2147483648LL || value > 2147483647LL) return false;
    *out = static_cast<int>(value);
    return true;
  }

  bool at_end() {
    skip_space();
    return pos_ == text_.size();
  }

 private:
  static bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
  }
  void skip_space() {
    while (pos_ < text_.size() && is_space(text_[pos_])) ++pos_;
  }

  const std::string& text_;
  std::size_t pos_ = 0;
};

bool expect(Scanner& scanner, const char* literal, std::string* error) {
  std::string t;
  if (!scanner.token(&t) || t != literal) {
    *error = std::string("expected '") + literal + "', got '" + t + "'";
    return false;
  }
  return true;
}

bool expect_int(Scanner& scanner, int* value, std::string* error) {
  if (!scanner.integer(value)) {
    *error = "expected integer";
    return false;
  }
  return true;
}

}  // namespace

bool read_snapshot(const std::string& text, WarpStack* stack,
                   std::string* error) {
  if (stack == nullptr || error == nullptr) return false;
  error->clear();
  *stack = WarpStack{};
  Scanner scanner(text);

  if (!expect(scanner, "WARPSTACK", error)) return false;
  int version = 0;
  if (!expect_int(scanner, &version, error) || version != 1) {
    *error = "unsupported snapshot version";
    return false;
  }
  if (!expect(scanner, "COUNT", error)) return false;
  int count = 0;
  if (!expect_int(scanner, &count, error) || count < 0 ||
      count > kMaxRefMvStackSize) {
    *error = "COUNT out of range";
    return false;
  }
  if (!expect(scanner, "NEAREST", error)) return false;
  int nearest = 0;
  if (!expect_int(scanner, &nearest, error) || nearest < 0 || nearest > count) {
    *error = "NEAREST out of range";
    return false;
  }
  if (!expect(scanner, "WARP", error)) return false;
  int warp = 0;
  if (!expect_int(scanner, &warp, error) || warp < 0 ||
      warp > kMaxWarpRefCandidates) {
    *error = "WARP out of range";
    return false;
  }
  if (!expect(scanner, "SORTED", error)) return false;
  int sorted = 0;
  if (!expect_int(scanner, &sorted, error) || (sorted != 0 && sorted != 1)) {
    *error = "SORTED out of range";
    return false;
  }

  stack->count = count;
  stack->num_nearest = nearest;
  stack->num_warp_found = warp;
  stack->sorted = sorted != 0;

  for (int i = 0; i < count; ++i) {
    if (!expect(scanner, "C", error)) return false;
    WarpCandidate& c = stack->cand[i];
    int v[8] = {};
    for (int k = 0; k < 8; ++k) {
      if (!expect_int(scanner, &v[k], error)) return false;
    }
    c.mv[0].y = v[0];
    c.mv[0].x = v[1];
    c.mv[1].y = v[2];
    c.mv[1].x = v[3];
    c.weight = v[4];
    c.row_offset = v[5];
    c.col_offset = v[6];
    if (v[7] < -128 || v[7] > 127) {
      *error = "cwp out of range";
      return false;
    }
    c.cwp = static_cast<std::int8_t>(v[7]);
  }
  for (int i = 0; i < warp; ++i) {
    if (!expect(scanner, "W", error)) return false;
    for (int k = 0; k < 6; ++k) {
      if (!expect_int(scanner, &stack->warp_params[i].p[k], error)) return false;
    }
  }
  if (!expect(scanner, "END", error)) return false;
  if (!scanner.at_end()) {
    *error = "trailing tokens after END";
    return false;
  }
  return true;
}

}  // namespace av2enc::inter::warp
