// libFuzzer target for the Y4M header parser.
//
// The Y4M header is the only place untrusted bytes reach av2enc, so this is
// where memory-safety bugs would live: token splitting, integer parsing, the
// dimension budget and the colorspace/interlacing checks.
//
// Drive it with tools/fuzz_smoke.sh, or directly:
//   build-fuzz/fuzz_y4m tests/fuzz_corpus -max_total_time=600
//
// Built only with -DAV2ENC_FUZZ=ON (needs a clang that ships libFuzzer; Homebrew
// LLVM does, Apple Command Line Tools does not).

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "core/io/y4m.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,
                                      std::size_t size) {
  // A Y4M header line is tiny; anything larger is not a header and would only
  // slow the fuzzer down.
  if (size == 0 || size > 8192) return 0;

  const std::string_view input(reinterpret_cast<const char*>(data), size);
  av2enc::Y4mHeader header;
  std::string error;

  // Accepted or rejected, the parser must not read out of bounds, hang, or
  // allocate from an unchecked dimension.
  (void)av2enc::parse_y4m_header(input, &header, &error);

  // Feed the same bytes again with a CRLF terminator: the reader strips '\r'
  // before parsing, so this covers a shape the raw corpus may miss.
  std::string with_crlf(input);
  if (with_crlf.size() < 8000) {
    with_crlf.push_back('\r');
    (void)av2enc::parse_y4m_header(with_crlf, &header, &error);
  }
  return 0;
}
