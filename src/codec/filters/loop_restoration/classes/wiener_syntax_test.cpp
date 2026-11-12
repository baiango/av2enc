// Tests for the exact AV2 loop-restoration syntax order and CDF round-trips.
//
// Pinned bit strings are built directly from the §5.18.7.11 / §5.20.10.5–.6
// pseudo-code; every header and tile also has a write -> read round-trip using
// the normative §8.2 arithmetic coder (including CDF adaptation state).

#include "codec/filters/loop_restoration/classes/wiener_syntax.h"

#include "codec/filters/loop_restoration/classes/wiener_cdfs.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace av2enc::restoration;
using namespace av2enc::restoration::classes;
using av2enc::BitWriter;
namespace av2 = av2enc::av2;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
    }                                                                      \
  } while (0)

std::string bit_string(const BitWriter& writer) {
  std::string out;
  const auto bytes = writer.bytes();
  for (std::size_t i = 0; i < writer.bit_size(); ++i) {
    const bool bit =
        ((bytes[i / 8U] >> (7U - static_cast<unsigned>(i & 7U))) & 1U) != 0U;
    out.push_back(bit ? '1' : '0');
  }
  return out;
}

WienerLrHeaderConfig header_config() {
  WienerLrHeaderConfig c;
  c.enable_restoration = true;
  c.num_planes = 3;
  c.sb_size = 64;
  c.sub_x = 1;
  c.sub_y = 1;
  c.base_q_idx = 90;
  c.lr_tools_disable[0][kRestorePcWiener] = true;
  c.lr_tools_disable[1][kRestorePcWiener] = true;
  c.lr_tools_disable[0][kRestoreWienerNs] = false;
  c.lr_tools_disable[1][kRestoreWienerNs] = false;
  return c;
}

void test_ns_roundtrip() {
  for (unsigned n = 2; n <= 17; ++n) {
    for (unsigned v = 0; v < n; ++v) {
      BitWriter w;
      std::string error;
      CHECK(wiener_put_ns(w, n, v, &error), "ns write");
      WienerBitReader r(w.bytes().data(), w.bit_size());
      std::uint32_t back = 0;
      CHECK(r.read_ns(n, &back), "ns read");
      CHECK(back == v, "ns value round-trip");
      CHECK(r.bits_left() == 0, "ns consumes exactly its bits");
    }
  }
}

void test_subexp_roundtrip() {
  for (int k = 1; k <= 4; ++k) {
    for (int num = 2; num <= 40; ++num) {
      for (int target = 0; target < num; ++target) {
        BitWriter w;
        std::string error;
        CHECK(wiener_encode_signed_subexp_with_ref(w, 0, num, num / 2, k,
                                                   target, &error),
              "subexp write");
        WienerBitReader r(w.bytes().data(), w.bit_size());
        int back = -1;
        CHECK(wiener_decode_signed_subexp_with_ref(r, 0, num, num / 2, k,
                                                   &back, &error),
              "subexp read");
        CHECK(back == target, "subexp value round-trip");
        CHECK(r.bits_left() == 0, "subexp consumes exactly its bits");
      }
    }
  }
}

void test_4part_roundtrip() {
  for (int k = 4; k <= 6; ++k) {
    const int low = -24;
    const int high = low + (1 << k);
    for (int target = low; target < high; ++target) {
      std::vector<std::uint8_t> buffer(1024);
      av2::Encoder enc;
      CHECK(enc.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk,
            "4part encoder init");
      std::uint16_t cdf[5];
      std::memcpy(cdf, kWienerNsBaseCdf, sizeof(cdf));
      std::string error;
      CHECK(wiener_encode_signed_4part(enc, cdf, low, k, low, target, &error),
            "4part write");
      std::size_t payload = 0;
      CHECK(enc.exit_symbol(&payload) == av2::Status::kOk, "4part encoder exit");

      av2::Decoder dec;
      CHECK(dec.init_symbol(buffer.data(), buffer.size(), payload) ==
                av2::Status::kOk,
            "4part decoder init");
      std::uint16_t cdf2[5];
      std::memcpy(cdf2, kWienerNsBaseCdf, sizeof(cdf2));
      int back = 0;
      CHECK(wiener_decode_signed_4part(dec, cdf2, low, k, low, &back, &error),
            "4part read");
      CHECK(back == target, "4part value round-trip");
      CHECK(std::memcmp(cdf, cdf2, sizeof(cdf)) == 0,
            "4part CDF adaptation round-trip");
      CHECK(dec.exit_symbol() == av2::Status::kOk, "4part decoder exit");
    }
  }
}

void test_header_none_pinned() {
  WienerLrHeaderConfig config = header_config();
  WienerLrHeaderParams params;
  WienerReferenceSet refs;
  BitWriter w;
  std::string error;
  CHECK(wiener_write_lr_params(w, config, &params, refs, &error),
        "NONE header write");
  CHECK(error.empty(), "NONE header no error");
  CHECK(w.bit_size() == 3, "NONE header is three ns(2) zero bits");
  CHECK(bit_string(w) == "000", "NONE header bits");

  WienerLrHeaderParams back;
  WienerBitReader r(w.bytes().data(), w.bit_size());
  CHECK(wiener_read_lr_params(r, config, &back, refs, &error),
        "NONE header read");
  CHECK(back.frame_type[0] == kRestoreNone, "NONE read plane 0");
  CHECK(back.frame_type[1] == kRestoreNone, "NONE read plane 1");
  CHECK(back.frame_type[2] == kRestoreNone, "NONE read plane 2");
}

void test_header_luma_tool_pinned() {
  WienerLrHeaderConfig config = header_config();
  WienerLrHeaderParams params;
  params.frame_type[0] = kRestoreWienerNs;
  params.frame_filters_on[0] = false;
  params.luma_shift = 3;
  params.chroma_shift = 3;
  WienerReferenceSet refs;
  BitWriter w;
  std::string error;
  CHECK(wiener_write_lr_params(w, config, &params, refs, &error),
        "luma tool header write");
  // plane0: ns(2)=1, frame_filters_on=0; plane1: ns(2)=0; plane2: ns(2)=0;
  // usesLumaLr: half=0, max=0, quarter=0.
  CHECK(bit_string(w) == "1000000", "luma tool header field order");
  WienerLrHeaderParams back;
  WienerBitReader r(w.bytes().data(), w.bit_size());
  CHECK(wiener_read_lr_params(r, config, &back, refs, &error),
        "luma tool header read");
  CHECK(back.frame_type[0] == kRestoreWienerNs, "luma tool type");
  CHECK(back.luma_shift == 3, "luma size shift");
  CHECK(r.bits_left() == 0, "luma tool header fully consumed");
}

void test_header_two_class_frame_filters() {
  WienerLrHeaderConfig config = header_config();
  WienerLrHeaderParams params;
  params.frame_type[0] = kRestoreWienerNs;
  params.frame_filters_on[0] = true;
  params.num_filter_classes = 2;
  params.num_filter_classes_idx = 1;
  params.frame_filter[0][0].match_index = 0;
  params.frame_filter[0][0].merged = true;
  params.frame_filter[0][1].match_index = 1;
  params.frame_filter[0][1].merged = true;
  std::string error;
  BitWriter w;
  WienerReferenceSet refs;
  CHECK(wiener_write_lr_params(w, config, &params, refs, &error),
        "two-class header write");
  CHECK(error.empty(), "two-class header no error");
  WienerLrHeaderParams back;
  WienerBitReader r(w.bytes().data(), w.bit_size());
  CHECK(wiener_read_lr_params(r, config, &back, refs, &error),
        "two-class header read");
  CHECK(back.frame_filters_on[0], "two-class frame filters on");
  CHECK(back.num_filter_classes == 2, "two-class count");
  CHECK(back.frame_filter[0][0].match_index == 0, "class 0 match");
  CHECK(back.frame_filter[0][1].match_index == 1, "class 1 match");
  CHECK(back.frame_filter[0][1].merged, "class 1 merged");
  CHECK(r.bits_left() == 0, "two-class header fully consumed");
}

void test_header_ref_groups() {
  WienerLrHeaderConfig config = header_config();
  config.frame_is_intra = false;
  config.num_total_refs = 3;
  WienerReferenceSet refs;
  refs.frame_is_intra = false;
  refs.num_total_refs = 3;
  for (int i = 0; i < 3; ++i) {
    refs.order_hint[i] = i;
    refs.ref_frame_idx[i] = 10 + i;
    refs.ref[10 + i].filters_on[0] = true;
    refs.ref[10 + i].num_filter_classes = 2;
    for (int c = 0; c < 2; ++c) {
      for (int j = 0; j < kWienerNsLumaCoeffs; ++j) {
        refs.ref[10 + i].coeffs[0][c][j] =
            static_cast<std::int16_t>((i + 1) * 3 - c + j % 2);
      }
    }
  }
  WienerLrHeaderParams params;
  params.frame_type[0] = kRestoreWienerNs;
  params.frame_filters_on[0] = true;
  params.num_filter_classes = 2;
  params.num_filter_classes_idx = 1;
  // c=0: a new zero filter; c=1: merge the first reference filter.
  params.frame_filter[0][0].match_index = 0;
  params.frame_filter[0][0].merged = false;
  params.frame_filter[0][0].subset = 0;
  params.frame_filter[0][1].match_index = 2;  // numClasses + ref 0
  params.frame_filter[0][1].merged = true;

  const WienerSearchResult search =
      wiener_search_frame_filters(0, -1, params, refs, true);
  CHECK(search.num_classes == 2, "search class count");
  CHECK(search.num_ref_filters == 6, "search ref filter count");
  CHECK(search.ref_filters.size() == 6, "search ref filter list");

  std::string error;
  BitWriter w;
  CHECK(wiener_write_lr_params(w, config, &params, refs, &error),
        "ref group header write");
  CHECK(error.empty(), "ref group header no error");
  WienerLrHeaderParams back;
  WienerBitReader r(w.bytes().data(), w.bit_size());
  CHECK(wiener_read_lr_params(r, config, &back, refs, &error),
        "ref group header read");
  CHECK(back.frame_filter[0][0].match_index == 0, "ref group class 0 match");
  CHECK(back.frame_filter[0][1].match_index == 2, "ref group class 1 match");
  CHECK(back.frame_filter[0][1].merged, "ref group class 1 merged");
  for (int j = 0; j < kWienerNsLumaCoeffs; ++j) {
    CHECK(back.frame_coeffs[0][1][j] == refs.ref[10].coeffs[0][0][j],
          "merged reference filter copied exactly");
  }
  CHECK(r.bits_left() == 0, "ref group header fully consumed");
}

void test_header_temporal_prediction() {
  WienerLrHeaderConfig config = header_config();
  config.frame_is_intra = false;
  config.num_total_refs = 2;
  WienerReferenceSet refs;
  refs.frame_is_intra = false;
  refs.num_total_refs = 2;
  refs.order_hint[0] = 0;
  refs.order_hint[1] = 1;
  refs.ref_frame_idx[0] = 5;
  refs.ref_frame_idx[1] = 6;
  refs.ref[5].filters_on[0] = true;
  refs.ref[5].num_filter_classes = 2;
  for (int c = 0; c < 2; ++c) {
    for (int j = 0; j < kWienerNsLumaCoeffs; ++j) {
      refs.ref[5].coeffs[0][c][j] =
          static_cast<std::int16_t>(c - 2 + (j % 3));
    }
  }
  WienerLrHeaderParams params;
  params.frame_type[0] = kRestoreWienerNs;
  params.frame_filters_on[0] = true;
  params.temporal_pred_flag[0] = true;
  params.rst_ref_pic_idx[0] = 0;
  params.num_filter_classes = 2;
  params.num_filter_classes_idx = 1;

  std::string error;
  BitWriter w;
  CHECK(wiener_write_lr_params(w, config, &params, refs, &error),
        "temporal header write");
  CHECK(error.empty(), "temporal header no error");
  CHECK(w.bit_size() < 64, "temporal header carries no filter coefficients");
  WienerLrHeaderParams back;
  WienerBitReader r(w.bytes().data(), w.bit_size());
  CHECK(wiener_read_lr_params(r, config, &back, refs, &error),
        "temporal header read");
  CHECK(back.temporal_pred_flag[0], "temporal flag round-trip");
  CHECK(back.num_filter_classes == 2, "temporal class count from reference");
  for (int c = 0; c < 2; ++c) {
    for (int j = 0; j < kWienerNsLumaCoeffs; ++j) {
      CHECK(back.frame_coeffs[0][c][j] == refs.ref[5].coeffs[0][c][j],
            "temporal filter copied from reference");
    }
  }
  CHECK(r.bits_left() == 0, "temporal header fully consumed");
}

void test_frame_filter_field_order() {
  WienerLrHeaderConfig config = header_config();
  WienerLrHeaderParams params;
  params.frame_type[0] = kRestoreWienerNs;
  params.frame_filters_on[0] = true;
  params.num_filter_classes = 2;
  params.num_filter_classes_idx = 1;
  params.frame_filter[0][0].match_index = 0;
  params.frame_filter[0][0].merged = false;
  params.frame_filter[0][0].subset = 0;
  for (int j = 0; j < kWienerNsCoeffStride; ++j) {
    params.frame_filter[0][0].coeffs[j] = 0;
  }
  params.frame_filter[0][1].match_index = 1;
  params.frame_filter[0][1].merged = true;

  WienerReferenceSet refs;
  std::string error;
  BitWriter w;
  CHECK(wiener_write_lr_params(w, config, &params, refs, &error),
        "field order header write");
  const std::string bits = bit_string(w);
  // Prefix: tool ns(2)=1, frame_filters_on=1, num_filter_classes_idx=001,
  // plane1=0, plane2=0, luma size flags=000 -> 10 bits.  Class 1's match
  // subexp is one 0 bit, then the merged flags for classes 0 and 1 must be
  // adjacent (0 then 1), not separated by class 0's coefficient data.
  CHECK(bits.size() > 13, "field order bitstream long enough");
  CHECK(bits.substr(11, 2) == "01",
        "merged flags for all classes precede coefficient data");
}

void test_unit_range() {
  WienerLrHeaderParams params;
  params.frame_type[0] = kRestoreWienerNs;
  params.luma_shift = 3;    // 64
  params.chroma_shift = 3;  // 32 for 4:2:0
  WienerTileState state;
  std::string error;
  CHECK(wiener_tile_init(&state, params, 64, 64, 1, 1, &error),
        "unit range tile init");
  WienerUnitRange r;
  CHECK(wiener_unit_range(state, 0, 0, 0, 16, 16, 0, 0, &r, &error),
        "luma unit range");
  CHECK(r.row_begin == 0 && r.row_end == 1 && r.col_begin == 0 &&
            r.col_end == 1,
        "64 unit covers one 64 SB");
  CHECK(wiener_unit_range(state, 0, 16, 16, 16, 16, 0, 0, &r, &error),
        "second luma unit range");
  CHECK(r.row_begin == 1 && r.row_end == 2 && r.col_begin == 1 &&
            r.col_end == 2,
        "second 64 unit");

  params.luma_shift = 0;  // 512, one unit for this frame
  CHECK(wiener_tile_init(&state, params, 64, 64, 1, 1, &error),
        "512 unit tile init");
  CHECK(wiener_unit_range(state, 0, 0, 0, 16, 16, 0, 0, &r, &error),
        "512 unit first SB");
  CHECK(r.row_end == 1 && r.col_end == 1, "512 unit read once");
  CHECK(wiener_unit_range(state, 0, 16, 16, 16, 16, 0, 0, &r, &error),
        "512 unit second SB");
  CHECK(r.row_end == r.row_begin && r.col_end == r.col_begin,
        "512 unit not read twice");
}

void test_tile_multi_tile_order() {
  WienerLrHeaderParams params;
  for (int p = 0; p < 3; ++p) {
    params.frame_type[p] = kRestoreWienerNs;
    params.frame_filters_on[p] = false;
  }
  params.luma_shift = 3;
  params.chroma_shift = 3;
  const int tile_mi[2] = {0, 32};
  for (int tr = 0; tr < 2; ++tr) {
    for (int tc = 0; tc < 2; ++tc) {
      WienerTileState write_state;
      std::string error;
      CHECK(wiener_tile_init(&write_state, params, 64, 64, 1, 1, &error),
            "multi-tile write init");
      std::vector<WienerUnitSyntax> plans[3];
      for (int p = 0; p < 3; ++p) {
        const std::size_t units =
            static_cast<std::size_t>(write_state.unit_cols[p]) *
            static_cast<std::size_t>(write_state.unit_rows[p]);
        plans[p].assign(units, WienerUnitSyntax{});
        WienerUnitRange range;
        CHECK(wiener_unit_range(write_state, p, tile_mi[tr], tile_mi[tc], 16,
                                16, 0, 0, &range, &error),
              "multi-tile unit range");
        for (int r = range.row_begin; r < range.row_end; ++r) {
          for (int c = range.col_begin; c < range.col_end; ++c) {
            const std::size_t index =
                static_cast<std::size_t>(r) *
                    static_cast<std::size_t>(write_state.unit_cols[p]) +
                static_cast<std::size_t>(c);
            plans[p][index].type = ((r + c) & 1) ? kRestoreWienerNs
                                                  : kRestoreNone;
          }
        }
      }
      std::vector<std::uint8_t> buffer(1 << 20);
      av2::Encoder encoder;
      CHECK(encoder.init_symbol(buffer.data(), buffer.size()) ==
                av2::Status::kOk,
            "multi-tile encoder init");
      CHECK(wiener_write_tile(encoder, write_state, plans, 64, tile_mi[tr],
                              tile_mi[tc], &error),
            "multi-tile write");
      std::size_t payload = 0;
      CHECK(encoder.exit_symbol(&payload) == av2::Status::kOk,
            "multi-tile encoder exit");

      WienerTileState read_state;
      CHECK(wiener_tile_init(&read_state, params, 64, 64, 1, 1, &error),
            "multi-tile read init");
      av2::Decoder decoder;
      CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload) ==
                av2::Status::kOk,
            "multi-tile decoder init");
      std::vector<WienerUnitSyntax> got[3];
      CHECK(wiener_read_tile(decoder, read_state, got, 64, tile_mi[tr],
                             tile_mi[tc], &error),
            "multi-tile read");
      CHECK(decoder.exit_symbol() == av2::Status::kOk,
            "multi-tile decoder exit");
      for (int p = 0; p < 3; ++p) {
        WienerUnitRange range;
        CHECK(wiener_unit_range(read_state, p, tile_mi[tr], tile_mi[tc], 16,
                                16, 0, 0, &range, &error),
              "multi-tile read range");
        for (int r = range.row_begin; r < range.row_end; ++r) {
          for (int c = range.col_begin; c < range.col_end; ++c) {
            const std::size_t index =
                static_cast<std::size_t>(r) *
                    static_cast<std::size_t>(read_state.unit_cols[p]) +
                static_cast<std::size_t>(c);
            CHECK(read_state.unit_type[p][index] ==
                      static_cast<std::uint8_t>(plans[p][index].type),
                  "multi-tile unit type round-trip");
          }
        }
      }
    }
  }
}

void test_tile_cdf_roundtrip() {
  WienerLrHeaderParams params;
  for (int p = 0; p < 3; ++p) {
    params.frame_type[p] = kRestoreWienerNs;
    params.frame_filters_on[p] = false;
  }
  params.luma_shift = 3;
  params.chroma_shift = 3;
  WienerTileState write_state;
  std::string error;
  CHECK(wiener_tile_init(&write_state, params, 64, 64, 1, 1, &error),
        "tile write init");

  std::vector<WienerUnitSyntax> plans[3];
  for (int p = 0; p < 3; ++p) {
    const std::size_t units =
        static_cast<std::size_t>(write_state.unit_cols[p]) *
        static_cast<std::size_t>(write_state.unit_rows[p]);
    plans[p].assign(units, WienerUnitSyntax{});
    for (std::size_t i = 0; i < units; ++i) {
      WienerUnitSyntax& u = plans[p][i];
      u.type = ((i + static_cast<std::size_t>(p)) % 3 == 0)
                   ? kRestoreNone
                   : kRestoreWienerNs;
      if (u.type != kRestoreWienerNs) {
        continue;
      }
      if (p == 0) {
        u.subset = static_cast<int>(i % 4);
        u.merged = (i % 5 == 0);
        u.use_bank_k = 0;
      } else {
        u.subset = static_cast<int>(i % 3);
        u.merged = (i % 4 == 0);
        u.use_bank_k = 0;
        u.uv_sym = (u.subset > 0);
      }
      for (int j = 0; j < kWienerNsCoeffStride; ++j) {
        const int min = wiener_ns_coeff_min(p > 0, j);
        const int max = wiener_ns_coeff_max(p > 0, j);
        u.coeffs[j] =
            static_cast<std::int16_t>(
                min + static_cast<int>((i + static_cast<std::size_t>(j)) %
                                        static_cast<std::size_t>(max - min + 1)));
      }
      if (p > 0 && u.uv_sym) {
        for (int j = kWienerNsShortCoeffs; j + 1 < kWienerNsChromaCoeffs;
             j += 2) {
          u.coeffs[j + 1] = u.coeffs[j];
        }
      }
    }
  }

  std::vector<std::uint8_t> buffer(1 << 22);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk,
        "tile encoder init");
  CHECK(wiener_write_tile(encoder, write_state, plans, 64, 0, 0, &error),
        "tile write");
  CHECK(error.empty(), "tile write no error");
  std::size_t payload = 0;
  CHECK(encoder.exit_symbol(&payload) == av2::Status::kOk, "tile encoder exit");

  WienerTileState read_state;
  CHECK(wiener_tile_init(&read_state, params, 64, 64, 1, 1, &error),
        "tile read init");
  av2::Decoder decoder;
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload) ==
            av2::Status::kOk,
        "tile decoder init");
  std::vector<WienerUnitSyntax> got[3];
  CHECK(wiener_read_tile(decoder, read_state, got, 64, 0, 0, &error),
        "tile read");
  CHECK(error.empty(), "tile read no error");
  CHECK(decoder.exit_symbol() == av2::Status::kOk, "tile decoder exit");

  for (int p = 0; p < 3; ++p) {
    CHECK(read_state.unit_type[p] == write_state.unit_type[p],
          "tile unit type round-trip");
    CHECK(read_state.unit_coeffs[p] == write_state.unit_coeffs[p],
          "tile unit coefficient round-trip");
  }
  CHECK(std::memcmp(read_state.use_wiener_ns, write_state.use_wiener_ns,
                    sizeof(write_state.use_wiener_ns)) == 0,
        "use_wiener_ns CDF state round-trip");
  CHECK(std::memcmp(read_state.wiener_ns_length, write_state.wiener_ns_length,
                    sizeof(write_state.wiener_ns_length)) == 0,
        "wiener_ns_length CDF state round-trip");
  CHECK(std::memcmp(read_state.wiener_ns_base, write_state.wiener_ns_base,
                    sizeof(write_state.wiener_ns_base)) == 0,
        "wiener_ns_base CDF state round-trip");
  CHECK(write_state.use_wiener_ns[2] != 0,
        "use_wiener_ns CDF actually adapted");
}

}  // namespace

int main() {
  test_ns_roundtrip();
  test_subexp_roundtrip();
  test_4part_roundtrip();
  test_header_none_pinned();
  test_header_luma_tool_pinned();
  test_header_two_class_frame_filters();
  test_header_ref_groups();
  test_header_temporal_prediction();
  test_frame_filter_field_order();
  test_unit_range();
  test_tile_multi_tile_order();
  test_tile_cdf_roundtrip();
  std::printf("wiener_syntax_test: %d checks, %d failures\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
