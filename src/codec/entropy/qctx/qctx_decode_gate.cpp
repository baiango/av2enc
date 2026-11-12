// Coefficient-CDF qctx decode gate (normative §8.2 coder round trip).
//
// The real end-to-end gate for slices 1..3 is "encode a stream at
// base_q_idx > 90 and decode it with dav2d".  That needs the M1/M2 writers to
// stop pinning slice 0 (see README.md, "Core change required"), and this
// module is not allowed to edit them.  What can be gated here, without core
// edits, is the piece that would silently corrupt such a stream: whether the
// generated tables are valid inputs to the *normative decoder* and whether
// the raw-base_q_idx selection resolves encoder and decoder to the same slice.
//
// For every table, every qctx slice and every context distribution:
//   * encode every symbol of the alphabet with a private CDF copy,
//   * decode the payload with the normative Decoder state machine and a
//     second private CDF copy,
//   * require the decoded symbol sequence and exit_symbol() to match.
// The same distributions are reached through tables_for_base_q_idx() at
// representative raw qindex values on both sides.
//
// Standalone: build with run_tests.sh (no CMakeLists edit required).
#include "codec/entropy/qctx/qctx.h"

#include "codec/entropy/av2/av2_entropy.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

using av2enc::still::qctx::kCoeffCdfSliceCount;
using av2enc::still::qctx::kCoeffCdfTableCount;
using av2enc::still::qctx::kCoeffCdfTables;
using av2enc::still::qctx::slice_for_base_q_idx;
using av2enc::still::qctx::table_slice;
using av2enc::still::qctx::tables_for_base_q_idx;

int g_failures = 0;
int g_gates = 0;
int g_symbols = 0;

void fail(const char* file, int line, const std::string& what) {
  ++g_failures;
  std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, what.c_str());
}

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) fail(__FILE__, __LINE__, #cond);                         \
  } while (0)

// One representative raw qindex per slice, plus the exact boundaries.
struct QpCase {
  int base_q_idx;
  int slice;
};

constexpr QpCase kQpCases[] = {
    {-7, 0},  {0, 0},    {45, 0},   {90, 0},   {91, 1},   {120, 1},
    {140, 1}, {141, 2},  {170, 2},  {190, 2},  {191, 3},  {255, 3},
    {303, 3}, {1000, 3},
};

bool round_trip(const std::uint16_t* cdf_source, int symbols,
                const std::string& label) {
  // CDF arrays are N symbols + rate + adaptation count (N+1 entries).
  std::array<std::uint16_t, 9> enc_cdf{};
  std::array<std::uint16_t, 9> dec_cdf{};
  const std::size_t entries = static_cast<std::size_t>(symbols) + 1;
  std::memcpy(enc_cdf.data(), cdf_source,
              entries * sizeof(std::uint16_t));
  std::memcpy(dec_cdf.data(), cdf_source,
              entries * sizeof(std::uint16_t));

  std::array<std::uint8_t, 128> buffer{};
  av2enc::av2::Encoder encoder;
  encoder.disable_cdf_update = 1;
  av2enc::av2::Status status =
      encoder.init_symbol(buffer.data(), buffer.size());
  if (status != av2enc::av2::Status::kOk) {
    fail(__FILE__, __LINE__, label + ": encoder init failed");
    return false;
  }
  for (int symbol = 0; symbol < symbols; ++symbol) {
    status = encoder.encode_symbol(enc_cdf.data(), symbols, symbol);
    if (status != av2enc::av2::Status::kOk) {
      fail(__FILE__, __LINE__,
           label + ": encode symbol " + std::to_string(symbol) +
               " failed (status " + std::to_string(static_cast<int>(status)) +
               ")");
      return false;
    }
    ++g_symbols;
  }
  std::size_t payload_bytes = 0;
  status = encoder.exit_symbol(&payload_bytes);
  if (status != av2enc::av2::Status::kOk) {
    fail(__FILE__, __LINE__, label + ": encoder exit failed");
    return false;
  }

  av2enc::av2::Decoder decoder;
  decoder.disable_cdf_update = 1;
  status = decoder.init_symbol(buffer.data(), buffer.size(), payload_bytes);
  if (status != av2enc::av2::Status::kOk) {
    fail(__FILE__, __LINE__, label + ": decoder init failed");
    return false;
  }
  for (int expected = 0; expected < symbols; ++expected) {
    int decoded = -1;
    status = decoder.decode_symbol(dec_cdf.data(), symbols, &decoded);
    if (status != av2enc::av2::Status::kOk) {
      fail(__FILE__, __LINE__,
           label + ": decode symbol " + std::to_string(expected) +
               " failed (status " + std::to_string(static_cast<int>(status)) +
               ")");
      return false;
    }
    if (decoded != expected) {
      fail(__FILE__, __LINE__,
           label + ": decoded " + std::to_string(decoded) + " expected " +
               std::to_string(expected));
      return false;
    }
  }
  status = decoder.exit_symbol();
  if (status != av2enc::av2::Status::kOk) {
    fail(__FILE__, __LINE__, label + ": decoder exit failed");
    return false;
  }

  if (std::memcmp(enc_cdf.data(), cdf_source,
                  entries * sizeof(std::uint16_t)) != 0 ||
      std::memcmp(dec_cdf.data(), cdf_source,
                  entries * sizeof(std::uint16_t)) != 0) {
    fail(__FILE__, __LINE__,
         label + ": CDF mutated despite disable_cdf_update");
    return false;
  }
  ++g_gates;
  return true;
}

void test_selection_gate() {
  for (const QpCase& qp : kQpCases) {
    const int selected = slice_for_base_q_idx(qp.base_q_idx);
    CHECK(selected == qp.slice);
    // Public API: encoder and decoder both derive the same typed view from
    // the same raw base_q_idx.
    const auto& view = tables_for_base_q_idx(qp.base_q_idx);
    CHECK(view.coeff_base == &av2enc::still::qctx::kCoeffBaseCdf[qp.slice]);
    CHECK(view.txb_skip == &av2enc::still::qctx::kTxbSkipCdf[qp.slice]);
    CHECK(view.eob_pt_64 == &av2enc::still::qctx::kEobPt64Cdf[qp.slice]);
    CHECK(view.dc_sign == &av2enc::still::qctx::kDcSignCdf[qp.slice]);
  }
}

void test_all_slices_all_contexts() {
  for (int slice = 0; slice < kCoeffCdfSliceCount; ++slice) {
    for (int table = 0; table < kCoeffCdfTableCount; ++table) {
      const auto& info = kCoeffCdfTables[table];
      const std::uint16_t* base = table_slice(info, slice);
      const int step = info.symbols + 1;
      for (std::size_t offset = 0; offset < info.slice_stride;
           offset += static_cast<std::size_t>(step)) {
        const std::string label = std::string(info.name) + " slice " +
                                  std::to_string(slice) + " offset " +
                                  std::to_string(offset);
        if (!round_trip(base + offset, info.symbols, label)) {
          return;  // avoid a cascade of identical failures
        }
      }
    }
  }
}

// Explicitly exercise the end-to-end selection path on the tables the M1/M2
// writers copy today: encode from tables_for_base_q_idx(qp), decode from a
// fresh tables_for_base_q_idx(qp) lookup.
void test_selected_table_round_trip() {
  for (const QpCase& qp : kQpCases) {
    const auto& encode_view = tables_for_base_q_idx(qp.base_q_idx);
    const auto& decode_view = tables_for_base_q_idx(qp.base_q_idx);
    CHECK(encode_view.txb_skip != nullptr);
    CHECK(encode_view.coeff_base != nullptr);
    if (!round_trip(&(*encode_view.txb_skip)[0][0][0][0], 2,
                    "selected txb_skip qp=" +
                        std::to_string(qp.base_q_idx))) {
      return;
    }
    if (!round_trip(&(*decode_view.coeff_base)[0][0][0][0], 4,
                    "selected coeff_base qp=" +
                        std::to_string(qp.base_q_idx))) {
      return;
    }
  }
}

}  // namespace

int main() {
  test_selection_gate();
  test_all_slices_all_contexts();
  test_selected_table_round_trip();

  if (g_failures != 0) {
    std::fprintf(stderr, "qctx_decode_gate: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("qctx_decode_gate: PASS (%d distributions, %d symbols)\n",
              g_gates, g_symbols);
  return 0;
}
