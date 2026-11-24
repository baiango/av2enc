// R19 experiment: range-coder interval-width rate estimate.
//
// AV2ENC_RATE_INTERVAL=1       interval-width estimate (default off)
// AV2ENC_RATE_INTERVAL=exact   ablation: full trial encode of the same symbols
// AV2ENC_RATE_INTERVAL_STATS=1 per-leaf predicted-vs-actual report at exit
// AV2ENC_RATE_INTERVAL_TRACE=1 one stderr line per reported leaf
//
// The decision layer prices a leaf's residual with the historical heuristics
// (`estimate_tx_bits`: 8 + 2.5 * nonzero + 0.5 * eob, plus 4 mode bits).
// That is a static proxy: it cannot see level magnitudes, the EOB symbol, the
// base/BR split, the escape or the sign.  This model walks the *actual* §8.3
// coefficient symbol sequence of a decided TX plan (the same sequence
// `rdoq::av2::Model` charges with `-log2 p`) and charges every symbol through
// the §8.2.6 coder interval arithmetic instead:
//
//     bits += log2(SymbolRange / interval_width)
//     SymbolRange = interval_width << renorm_bits
//
// which is exactly `-log2` of the interval the real coder would end up with,
// without emitting a single bit.  Two properties matter:
//
//   * the §8.2.6 boundaries round to an 8-multiple (`<< 3`) and use the
//     Prob_Inc table, so rare symbols are floored like the real coder rather
//     than priced at a theoretical -log2(p) (R01's model);
//   * the CDF rows are a *scratch copy* of the frame-start §8.2 CDF image
//     (`cdf_seed`, i.e. the carried live state) and adapt inside the leaf
//     when the frame adapts CDFs, so the estimate follows the live model
//     instead of the §9.3 defaults.
//
// `AV2ENC_RATE_INTERVAL=exact` runs the same symbol sequence through a real
// `av2::Encoder` into a scratch buffer and returns the unmaterialized prefix
// bits: the R05 "full trial coding" reference.  It is the accuracy ceiling and
// the cost floor the interval model is compared against; it is an ablation,
// not the shipped path.
//
// The scratch state is frame-start, not leaf-start: the decision layer runs
// before the tile writer (and the wavefront decides many superblocks before
// the serial tile stream exists), so the model cannot read the encoder's
// current interval.  The resulting error is measured, not hidden: the writer
// records every leaf's actual symbol information cost and this header reports
// median / p90 / signed bias per class.
//
// Header-only on purpose: the experiment must not add a translation unit to
// the integrator-owned CMakeLists.txt.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "codec/entropy/av2/av2_entropy.h"
#include "codec/entropy/qctx/qctx.h"
#include "codec/rate/rdoq/rdoq_av2_model.h"
#include "codec/transform/inter_residual/inter_tx_type.h"
#include "codec/transform/tx_partition/tx_partition.h"
#include "codec/transform/variants/tx_variants.h"
#include "pipeline/video/video_syntax.h"
#include "spec_tables/spec_math.h"

namespace av2enc::video::rate_interval {

enum class Mode {
  kOff = 0,
  kInterval = 1,
  kExact = 2,
};

// Defined at the bottom; registered with atexit by stats_enabled().
inline void print_stats();

namespace detail {

inline bool parse_off(const char* value) {
  if (value == nullptr || *value == '\0') return true;
  if (std::strcmp(value, "0") == 0) return true;
  auto equal = [value](const char* word) {
    for (std::size_t i = 0; word[i] != '\0'; ++i) {
      if (std::tolower(static_cast<unsigned char>(value[i])) != word[i]) {
        return false;
      }
    }
    return value[std::strlen(word)] == '\0';
  };
  return equal("false") || equal("off") || equal("no");
}

inline Mode parse_mode() {
  const char* env = std::getenv("AV2ENC_RATE_INTERVAL");
  if (parse_off(env)) return Mode::kOff;
  if (std::strcmp(env, "exact") == 0 || std::strcmp(env, "r05") == 0 ||
      std::strcmp(env, "encode") == 0) {
    return Mode::kExact;
  }
  return Mode::kInterval;
}

inline bool stats_enabled() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_RATE_INTERVAL_STATS");
    return !parse_off(env);
  }();
  static const bool registered = [] {
    const bool on = std::getenv("AV2ENC_RATE_INTERVAL_STATS") != nullptr &&
                    !parse_off(std::getenv("AV2ENC_RATE_INTERVAL_STATS"));
    if (on) std::atexit(&print_stats);
    return true;
  }();
  (void)registered;
  return value;
}

// The compile-time constants are disjoint from the model, so this never
// depends on an include order quirk.
inline constexpr double kImpossibleBits = 1.0e6;

}  // namespace detail

inline Mode mode() {
  static const Mode value = detail::parse_mode();
  return value;
}

inline bool enabled() { return mode() != Mode::kOff; }

// ---------------------------------------------------------------------------
// Coefficient / mode CDF scratch copy.
//
// One copy per frame, built from the §8.2 image the writer starts from
// (`cdf_seed`) or, on the first frame / with the CDF chain off, from the §9.3
// slice selected by the raw frame base_q_idx.  Every estimate reads it; the
// sink copies a row on first use so nothing is mutated here.
// ---------------------------------------------------------------------------
struct CdfStore {
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
  std::uint16_t eob_pt_1024[3][9];
  std::uint16_t eob_pt_512[3][9];
  std::uint16_t eob_pt_256[3][9];
  std::uint16_t eob_pt_128[3][9];
  std::uint16_t eob_pt_64[3][8];
  std::uint16_t eob_pt_32[3][7];
  std::uint16_t eob_pt_16[3][6];
  std::uint16_t eob_extra[3];
  std::uint16_t coeff_base[5][20][2][5];
  std::uint16_t coeff_base_lf[5][33][2][7];
  std::uint16_t coeff_base_eob[5][4][4];
  std::uint16_t coeff_base_lf_eob[5][4][6];
  std::uint16_t coeff_br[7][5];
  std::uint16_t coeff_br_lf[14][5];
  std::uint16_t coeff_base_uv[12][5];
  std::uint16_t coeff_base_lf_uv[12][7];
  std::uint16_t coeff_base_eob_uv[4][4];
  std::uint16_t coeff_base_lf_eob_uv[4][6];
  std::uint16_t coeff_br_uv[4][5];
  std::uint16_t dc_sign[2][2][3][3];
  // Luma inter TX type rows (§5.20.8.2) and the TX partition rows, so the
  // residual estimate covers every symbol the writer emits for a TX block.
  std::uint16_t inter_tx_type_set1[3][2][3];
  std::uint16_t inter_tx_type_index_set1[3][9];
  std::uint16_t inter_tx_type_offset_set1[3][9];
  std::uint16_t inter_tx_type_set2[3][3];
  std::uint16_t inter_tx_type_index_set2[3][9];
  std::uint16_t inter_tx_type_offset_set2[3][5];
  std::uint16_t inter_tx_type_set3[3][4][3];
  std::uint16_t inter_tx_type_long[3][4][5];
  std::uint16_t is_long_side_dct[2][3];
  txpartition::TxPartitionCdfs tx_partition;
  // Inter mode rows (§5.20.7.6/.8/.10/.20).  Only the single-reference
  // profile the nightly aggregate codes is modelled; compound/TIP/AMVD rows
  // are deliberately absent (that path is rejected by the caller).
  std::uint16_t is_inter[4][3];
  std::uint16_t skip[6][3];
  std::uint16_t single_mode[5][4];
  std::uint16_t drl_mode[3][5][3];
  std::uint16_t shell_set[3];
  std::uint16_t shell6_class0[9];
  std::uint16_t shell6_class1[9];
  std::uint16_t shell_last_two[3];
  std::uint16_t offset_low[2][3];
  std::uint16_t offset_class2[3];
  std::uint16_t offset_other[16][3];
  std::uint16_t col_mv_greater[2][3];
  std::uint16_t col_mv_index[4][3];
  std::uint16_t use_bawp[3];
  std::uint16_t explicit_bawp[3][3];
  std::uint16_t explicit_bawp_scale[3];
  std::uint16_t use_bawp_chroma[3];
};

// §9.3 defaults (first frame / CDF chain off).
inline void load_defaults(CdfStore& s, int base_q_idx) {
  const still::qctx::CoeffCdfSlice& q =
      still::qctx::tables_for_base_q_idx(base_q_idx);
  auto copy = [](auto& dst, const auto& src) {
    std::memcpy(&dst, &src, sizeof(dst));
  };
  copy(s.txb_skip, *q.txb_skip);
  copy(s.v_txb_skip, *q.v_txb_skip);
  copy(s.eob_pt_1024, *q.eob_pt_1024);
  copy(s.eob_pt_512, *q.eob_pt_512);
  copy(s.eob_pt_256, *q.eob_pt_256);
  copy(s.eob_pt_128, *q.eob_pt_128);
  copy(s.eob_pt_64, *q.eob_pt_64);
  copy(s.eob_pt_32, *q.eob_pt_32);
  copy(s.eob_pt_16, *q.eob_pt_16);
  copy(s.eob_extra, *q.eob_extra);
  copy(s.coeff_base, *q.coeff_base);
  copy(s.coeff_base_lf, *q.coeff_base_lf);
  copy(s.coeff_base_eob, *q.coeff_base_eob);
  copy(s.coeff_base_lf_eob, *q.coeff_base_lf_eob);
  copy(s.coeff_br, *q.coeff_br);
  copy(s.coeff_br_lf, *q.coeff_br_lf);
  copy(s.coeff_base_uv, *q.coeff_base_uv);
  copy(s.coeff_base_lf_uv, *q.coeff_base_lf_uv);
  copy(s.coeff_base_eob_uv, *q.coeff_base_eob_uv);
  copy(s.coeff_base_lf_eob_uv, *q.coeff_base_lf_eob_uv);
  copy(s.coeff_br_uv, *q.coeff_br_uv);
  copy(s.dc_sign, *q.dc_sign);
  s.tx_partition.reset(/*reduced_tx_part_set=*/false);
}

// Copy from a `syntax::detail::CdfState` (the writer's §8.2 image).  A
// template so this header does not have to include the writer.
template <class CdfStateT>
inline void load_from_state(CdfStore& s, const CdfStateT& c) {
  auto copy = [](auto& dst, const auto& src) {
    std::memcpy(&dst, &src, sizeof(dst));
  };
  copy(s.txb_skip, c.txb_skip);
  copy(s.v_txb_skip, c.v_txb_skip);
  copy(s.eob_pt_1024, c.eob_pt_1024);
  copy(s.eob_pt_512, c.eob_pt_512);
  copy(s.eob_pt_256, c.eob_pt_256);
  copy(s.eob_pt_128, c.eob_pt_128);
  copy(s.eob_pt_64, c.eob_pt_64);
  copy(s.eob_pt_32, c.eob_pt_32);
  copy(s.eob_pt_16, c.eob_pt_16);
  copy(s.eob_extra, c.eob_extra);
  copy(s.coeff_base, c.coeff_base);
  copy(s.coeff_base_lf, c.coeff_base_lf);
  copy(s.coeff_base_eob, c.coeff_base_eob);
  copy(s.coeff_base_lf_eob, c.coeff_base_lf_eob);
  copy(s.coeff_br, c.coeff_br);
  copy(s.coeff_br_lf, c.coeff_br_lf);
  copy(s.coeff_base_uv, c.coeff_base_uv);
  copy(s.coeff_base_lf_uv, c.coeff_base_lf_uv);
  copy(s.coeff_base_eob_uv, c.coeff_base_eob_uv);
  copy(s.coeff_base_lf_eob_uv, c.coeff_base_lf_eob_uv);
  copy(s.coeff_br_uv, c.coeff_br_uv);
  copy(s.dc_sign, c.dc_sign);
  copy(s.inter_tx_type_set1, c.inter_tx_type_set1);
  copy(s.inter_tx_type_index_set1, c.inter_tx_type_index_set1);
  copy(s.inter_tx_type_offset_set1, c.inter_tx_type_offset_set1);
  copy(s.inter_tx_type_set2, c.inter_tx_type_set2);
  copy(s.inter_tx_type_index_set2, c.inter_tx_type_index_set2);
  copy(s.inter_tx_type_offset_set2, c.inter_tx_type_offset_set2);
  copy(s.inter_tx_type_set3, c.inter_tx_type_set3);
  copy(s.inter_tx_type_long, c.inter_tx_type_long);
  copy(s.is_long_side_dct, c.is_long_side_dct);
  copy(s.tx_partition, c.tx_partition);
  copy(s.is_inter, c.is_inter);
  copy(s.skip, c.skip);
  copy(s.single_mode, c.single_mode);
  copy(s.drl_mode, c.drl_mode);
  copy(s.shell_set, c.shell_set);
  copy(s.shell6_class0, c.shell6_class0);
  copy(s.shell6_class1, c.shell6_class1);
  copy(s.shell_last_two, c.shell_last_two);
  copy(s.offset_low, c.offset_low);
  copy(s.offset_class2, c.offset_class2);
  copy(s.offset_other, c.offset_other);
  copy(s.col_mv_greater, c.col_mv_greater);
  copy(s.col_mv_index, c.col_mv_index);
  copy(s.use_bawp, c.use_bawp);
  copy(s.explicit_bawp, c.explicit_bawp);
  copy(s.explicit_bawp_scale, c.explicit_bawp_scale);
  copy(s.use_bawp_chroma, c.use_bawp_chroma);
}

// Fill the store from the packed §8.2 CDF image the tile writer starts from
// (the carried live state).  Implemented in video_syntax.cpp, where the image
// layout lives; an empty image loads the §9.3 defaults for `base_q_idx`.
[[nodiscard]] bool load_store_from_image(std::span<const std::uint16_t> image,
                           int base_q_idx, CdfStore& out);

// A const view of the store in the shape `rdoq::av2::Model` reads.  Arrays the
// model never touches (BOB / IDTX / PH / parity hiding) point at the §9.3
// slice so every pointer is well-formed.
inline still::qctx::CoeffCdfSlice slice_of(const CdfStore& s) {
  const still::qctx::CoeffCdfSlice& d = still::qctx::tables_for_slice(0);
  still::qctx::CoeffCdfSlice v = d;
  v.txb_skip = &s.txb_skip;
  v.v_txb_skip = &s.v_txb_skip;
  v.eob_pt_1024 = &s.eob_pt_1024;
  v.eob_pt_512 = &s.eob_pt_512;
  v.eob_pt_256 = &s.eob_pt_256;
  v.eob_pt_128 = &s.eob_pt_128;
  v.eob_pt_64 = &s.eob_pt_64;
  v.eob_pt_32 = &s.eob_pt_32;
  v.eob_pt_16 = &s.eob_pt_16;
  v.eob_extra = &s.eob_extra;
  v.coeff_base = &s.coeff_base;
  v.coeff_base_lf = &s.coeff_base_lf;
  v.coeff_base_eob = &s.coeff_base_eob;
  v.coeff_base_lf_eob = &s.coeff_base_lf_eob;
  v.coeff_br = &s.coeff_br;
  v.coeff_br_lf = &s.coeff_br_lf;
  v.coeff_base_uv = &s.coeff_base_uv;
  v.coeff_base_lf_uv = &s.coeff_base_lf_uv;
  v.coeff_base_eob_uv = &s.coeff_base_eob_uv;
  v.coeff_base_lf_eob_uv = &s.coeff_base_lf_eob_uv;
  v.coeff_br_uv = &s.coeff_br_uv;
  v.dc_sign = &s.dc_sign;
  return v;
}

// ---------------------------------------------------------------------------
// Sinks
// ---------------------------------------------------------------------------

// Shared CDF-row bank: one mutable copy per CDF row used by the leaf, seeded
// from the frame-start store and adapted after each symbol (when the frame
// adapts).  Both sinks use it, so the symbol sequences are identical.
class RowBank {
 public:
  explicit RowBank(bool adapt) : adapt_(adapt) {}

  std::uint16_t* get(const std::uint16_t* key, int n) {
    if (n <= 1) return nullptr;
    for (int i = 0; i < count_; ++i) {
      if (rows_[i].key == key) return rows_[i].data;
    }
    if (count_ >= kCapacity || n + 1 > kMaxEntries) {
      failed_ = true;
      return nullptr;
    }
    Row& row = rows_[count_++];
    row.key = key;
    row.n = n;
    for (int i = 0; i <= n; ++i) row.data[i] = key[i];
    return row.data;
  }

  void adapt(std::uint16_t* row, int n, int symbol) {
    if (adapt_ && row != nullptr) av2::adapt_cdf(row, n, symbol);
  }

  bool failed() const { return failed_; }

 private:
  static constexpr int kCapacity = 96;
  static constexpr int kMaxEntries = 9;
  struct Row {
    const std::uint16_t* key = nullptr;
    int n = 0;
    std::uint16_t data[kMaxEntries] = {};
  };
  Row rows_[kCapacity];
  int count_ = 0;
  bool adapt_ = false;
  bool failed_ = false;
};

// log2(v) for v in [1, 2^32): exponent plus a 32-segment linear mantissa
// table.  ~1e-4 bit absolute error, which is far below the §8.2.6 rounding it
// measures, and it keeps the interval sink (an integer state machine) from
// calling libm once per symbol.
inline double fast_log2(std::uint32_t v) {
  static const double kFrac[33] = {
      0.000000000, 0.044394119, 0.087462841, 0.129283017, 0.169925001,
      0.209453366, 0.247927513, 0.285402219, 0.321928095, 0.357552005,
      0.392317423, 0.426264755, 0.459431619, 0.491853096, 0.523561956,
      0.554588852, 0.584962501, 0.614709844, 0.643856190, 0.672425342,
      0.700439718, 0.727920455, 0.754887502, 0.781359714, 0.807354922,
      0.832890014, 0.857980995, 0.882643050, 0.906890596, 0.930737338,
      0.954196310, 0.977279923, 1.000000000};
  const std::uint32_t e = av2::floor_log2(v);
  // Normalized mantissa in [2^31, 2^32).
  const std::uint32_t m = (e >= 31) ? v : (v << (31u - e));
  const std::uint32_t idx = (m >> 26) & 31u;
  const double frac = static_cast<double>(m & 0x03FFFFFFu) /
                      static_cast<double>(1u << 26);
  const double mant = kFrac[idx] + (kFrac[idx + 1] - kFrac[idx]) * frac;
  return static_cast<double>(e) + mant;
}

// Interval-width accumulator: the §8.2.6 arithmetic without bit emission.
class IntervalSink final : public rdoq::av2::SymbolSink {
 public:
  explicit IntervalSink(bool adapt) : bank_(adapt) {}

  double symbol(const std::uint16_t* cdf, int n, int symbol) override {
    if (n <= 1) return 0.0;
    std::uint16_t* row = bank_.get(cdf, n);
    if (row == nullptr || symbol < 0 || symbol >= n) {
      failed_ = true;
      return detail::kImpossibleBits;
    }
    av2::Boundaries boundaries{};
    if (av2::compute_boundaries(range_, row, n, boundaries) !=
        av2::Status::kOk) {
      failed_ = true;
      return detail::kImpossibleBits;
    }
    const std::uint32_t cur = boundaries.cur[symbol];
    const std::uint32_t prev =
        (symbol == 0) ? range_ : boundaries.cur[symbol - 1];
    const std::uint32_t width = prev - cur;
    if (width == 0) {
      failed_ = true;
      return detail::kImpossibleBits;
    }
    const double bits = fast_log2(range_) - fast_log2(width);
    range_ = width << (15u - av2::floor_log2(width));
    bits_ += bits;
    bank_.adapt(row, n, symbol);
    return bits;
  }

  double literal(double bits) override {
    // §8.2.3: a raw bit halves the range and renormalises one bit, so the
    // interval is unchanged and the cost is exactly one bit per literal bit.
    bits_ += bits;
    return bits;
  }

  double bits() const { return bits_; }
  bool failed() const { return failed_ || bank_.failed(); }

 private:
  RowBank bank_;
  std::uint32_t range_ = av2::kSymbolRangeInit;
  double bits_ = 0.0;
  bool failed_ = false;
};

// R05 ablation: the same symbols through a real §8.2 encoder.  Reports the
// emitted prefix bits (what a trial encode costs before termination) and the
// finalized payload bits (marker + padding included).
class ExactSink final : public rdoq::av2::SymbolSink {
 public:
  explicit ExactSink(bool adapt) : bank_(adapt) { reset(); }

  double symbol(const std::uint16_t* cdf, int n, int symbol) override {
    if (n <= 1) return 0.0;
    std::uint16_t* row = bank_.get(cdf, n);
    if (row == nullptr || symbol < 0 || symbol >= n) {
      failed_ = true;
      return detail::kImpossibleBits;
    }
    const std::uint32_t before = enc_.prefix_bits;
    if (enc_.encode_symbol(row, n, symbol) != av2::Status::kOk) {
      failed_ = true;
      return detail::kImpossibleBits;
    }
    bank_.adapt(row, n, symbol);
    return static_cast<double>(enc_.prefix_bits - before);
  }

  double literal(double bits) override {
    const long count = std::lround(bits);
    if (count < 0 || count > 4096) {
      failed_ = true;
      return detail::kImpossibleBits;
    }
    for (long i = 0; i < count; ++i) {
      if (enc_.encode_bool(false) != av2::Status::kOk) {
        failed_ = true;
        return detail::kImpossibleBits;
      }
    }
    return bits;
  }

  // Conflict-free bits the symbol stream asked for (no §8.2.4 marker).
  double prefix_bits() const {
    return static_cast<double>(enc_.prefix_bits);
  }

  // Full trial-encode payload: marker, 14 zeros and byte padding included.
  double payload_bits() {
    if (!finished_) {
      finished_ = true;
      if (enc_.exit_symbol(&payload_bytes_) != av2::Status::kOk) {
        failed_ = true;
      }
    }
    return static_cast<double>(payload_bytes_) * 8.0;
  }

  bool failed() const { return failed_ || bank_.failed(); }

 private:
  void reset() {
    enc_.init_symbol(buffer_, sizeof(buffer_));
    enc_.disable_cdf_update = 1;  // the bank owns adaptation
  }
  RowBank bank_;
  av2::Encoder enc_{};
  std::uint8_t buffer_[16384] = {};
  std::size_t payload_bytes_ = 0;
  bool finished_ = false;
  bool failed_ = false;
};

// ---------------------------------------------------------------------------
// Leaf description
// ---------------------------------------------------------------------------

// One transform block of the leaf.  `levels` is the full adjusted-size
// §5.20.7.30 scan-order array (zeros after the EOB), the layout
// `rdoq::av2::Model::init()` expects.
struct LeafPart {
  int plane = 0;
  int tx_size = 0;
  int tx_type = 0;
  int eob = 0;
  std::span<const std::int32_t> levels;
};

struct LeafRateInput {
  bool is_inter = false;
  bool adapt = false;          // frame codes adaptive CDFs
  int base_q_idx = 90;
  bool tx_mode_select = false;
  int bsize = 0;               // partition::BlockSize
  int tx_partition = 0;        // txpartition::TxPartition
  bool has_chroma = true;
  bool skip_flag = false;
  // Inter mode info.  `inter_mode < 0` marks an intra leaf (no mode priced).
  int inter_mode = -1;
  int drl_index = 0;
  int mvd_y = 0;
  int mvd_x = 0;
  bool use_bawp = false;
  bool use_amvd = false;  // AMVD changes the MVD symbol set; caller rejects
  int part_count = 0;
  LeafPart parts[6];           // luma (TX decode order) then U, V
};

struct LeafRate {
  double mode_bits = 0.0;
  double residual_bits = 0.0;
  double exact_prefix_bits = 0.0;
  double exact_payload_bits = 0.0;
  bool failed = false;
  double code() const { return mode_bits + residual_bits; }
};

// Backing store for the scan-order arrays a LeafRateInput points at.
struct LeafLevelStorage {
  std::vector<std::int32_t> plane[6];
};

// Fill one part from a scan-order level list truncated at `eob`.
inline void set_part_scan(LeafRateInput& in, int index, int plane, int tx_size,
                          int tx_type, int eob,
                          std::span<const std::int32_t> scan_levels,
                          LeafLevelStorage& storage) {
  LeafPart& part = in.parts[index];
  part.plane = plane;
  part.tx_size = tx_size;
  part.tx_type = tx_type;
  part.eob = eob;
  const auto params = txv::make_tx_params(
      static_cast<txv::TxSize>(tx_size),
      plane == 0 ? static_cast<txv::TxType>(tx_type) : txv::TxType::kDctDct);
  const int area = params.has_value() ? params->w * params->h : 0;
  std::vector<std::int32_t>& levels = storage.plane[index];
  levels.assign(static_cast<std::size_t>(std::max(area, 0)), 0);
  const int n = std::min<int>(eob, static_cast<int>(scan_levels.size()));
  for (int c = 0; c < n; ++c) {
    levels[static_cast<std::size_t>(c)] =
        scan_levels[static_cast<std::size_t>(c)];
  }
  part.levels = levels;
}

// Fill one part from a raster (adjusted-size) level array.
inline void set_part_raster(LeafRateInput& in, int index, int plane,
                            int tx_size, int tx_type, int eob,
                            std::span<const std::int32_t> raster_levels, int w,
                            int h, LeafLevelStorage& storage) {
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(std::max(w * h, 0)));
  if (w > 0 && h > 0) syntax::build_scan(w, h, scan);
  std::vector<std::int32_t> ordered(static_cast<std::size_t>(std::max(eob, 0)));
  for (int c = 0; c < eob; ++c) {
    ordered[static_cast<std::size_t>(c)] =
        raster_levels[static_cast<std::size_t>(
            scan[static_cast<std::size_t>(c)])];
  }
  set_part_scan(in, index, plane, tx_size, tx_type, eob, ordered, storage);
}

// Build the input for one emitted leaf.  The default covers the
// single-reference no-compound inter profile the nightly aggregate codes;
// anything else is reported unsupported so the caller falls back.
//
// `mode_less_inter` is the joint-plan path: a compound or TIP leaf's mode
// syntax is priced exactly by the SyntaxProbe (`AV2ENC_COMPOUND_EXACT_RATE`),
// so this build contributes only the coefficient symbols (`inter_mode = -1`
// suppresses the probe's mode rows inside estimate()).  The writer's own
// diagnostics keep the default and never mis-price a compound leaf.
inline bool build_from_syntax(const syntax::BlockSyntax& node,
                              std::span<const std::int32_t> levels,
                              bool tx_mode_select, bool adapt, int base_q_idx,
                              LeafRateInput* in, LeafLevelStorage* storage,
                              bool mode_less_inter = false) {
  *in = LeafRateInput{};
  in->is_inter = node.is_inter != 0;
  in->adapt = adapt;
  in->base_q_idx = base_q_idx;
  in->tx_mode_select = tx_mode_select;
  in->bsize = static_cast<int>(node.bsize);
  in->tx_partition = static_cast<int>(node.tx_partition);
  in->skip_flag = node.is_inter != 0 && node.skip_flag != 0;
  in->has_chroma = true;
  if (node.is_inter) {
    if (mode_less_inter && (node.compound || node.tip_mode)) {
      in->inter_mode = -1;
    } else {
      if (node.compound || node.use_amvd || node.tip_mode) return false;
      in->inter_mode = static_cast<int>(node.inter_mode);
      in->drl_index = node.drl_index;
      in->mvd_y = node.mvd_y;
      in->mvd_x = node.mvd_x;
      in->use_bawp = node.use_bawp != 0;
    }
  }
  if (in->skip_flag) return true;

  const bool chroma_part = node.tree_type == 2;
  const bool has_chroma = (node.flags & syntax::kFlagHasChromaInfo) == 0 ||
                          node.has_chroma != 0;
  std::size_t cursor = node.levels_offset;
  int count = 0;
  if (!chroma_part) {
    int luma_parts = 1;
    if (tx_mode_select && node.tx_part_count != 0) {
      luma_parts = node.tx_part_count;
    }
    for (int i = 0; i < luma_parts; ++i) {
      const int tx_size = tx_mode_select && node.tx_part_count != 0
                              ? node.tx_part_tx_size[i]
                              : node.tx_size[0];
      const int tx_type = tx_mode_select && node.tx_part_count != 0
                              ? node.tx_part_tx_type[i]
                              : node.tx_type[0];
      const int eob = tx_mode_select && node.tx_part_count != 0
                          ? static_cast<int>(node.tx_part_eob[i])
                          : static_cast<int>(node.eob[0]);
      if (cursor + static_cast<std::size_t>(eob) > levels.size()) return false;
      set_part_scan(*in, count++, 0, tx_size, tx_type, eob,
                    levels.subspan(cursor, static_cast<std::size_t>(eob)),
                    *storage);
      cursor += static_cast<std::size_t>(eob);
    }
  }
  if (has_chroma && node.tree_type != 1) {
    for (int plane = 1; plane < 3; ++plane) {
      const int eob = static_cast<int>(node.eob[plane]);
      if (cursor + static_cast<std::size_t>(eob) > levels.size()) return false;
      set_part_scan(*in, count++, plane, node.tx_size[plane],
                    node.tx_type[plane], eob,
                    levels.subspan(cursor, static_cast<std::size_t>(eob)),
                    *storage);
      cursor += static_cast<std::size_t>(eob);
    }
  }
  in->part_count = count;
  return true;
}

namespace detail {

inline int floor_log2_int(int value) {
  int log = 0;
  while (value > 1) {
    value >>= 1;
    ++log;
  }
  return log;
}

// §8.3.2 inter_tx_type context, mirrored from the writer.
inline int inter_tx_type_ctx(int eob, int coded_w, int coded_h, bool use_min) {
  const int bwl = std::min(floor_log2_int(coded_w), 5);
  const int eoby = (eob - 1) >> bwl;
  const int eobx = (eob - 1) - (eoby << bwl);
  const int diag = eobx + eoby;
  const int w = use_min ? std::min(coded_w, 32) : coded_w;
  const int h = use_min ? std::min(coded_h, 32) : coded_h;
  if (diag < 2) return 1;
  if (diag > w + h - 4) return 2;
  return 0;
}

// §5.20.9.1 ns(n, value) literal bit count (writer write_ns_bits).
inline int ns_bit_count(std::uint32_t n, std::uint32_t value) {
  if (n <= 1) return 0;
  unsigned width = 0;
  while ((1u << width) < n) ++width;
  const std::uint32_t m = (1u << width) - n;
  return value < m ? static_cast<int>(width - 1u)
                   : static_cast<int>(width);
}

// Writes the inter TX type symbol group for one luma block.  Mirrors
// Writer::write_inter_tx_type() but reads `store`.
inline double inter_tx_type_bits(const CdfStore& store,
                                 rdoq::av2::SymbolSink& sink, int tx_size,
                                 int tx_type, int eob, int coded_w,
                                 int coded_h) {
  const auto ts = static_cast<txv::TxSize>(tx_size);
  const residual::InterTxSet set = residual::inter_tx_set(ts, /*reduced=*/0);
  if (set == residual::InterTxSet::kDctOnly) return 0.0;
  residual::InterTxCandidate cand;
  if (!residual::inter_tx_symbol(ts, /*reduced=*/0,
                                 /*enable_inter_ddt=*/false,
                                 static_cast<txv::TxType>(tx_type), &cand)) {
    return kImpossibleBits;
  }
  const int ctx = inter_tx_type_ctx(eob, coded_w, coded_h, true);
  const int sqr = residual::tx_size_sqr(ts);
  double bits = 0.0;
  auto sym = [&](const std::uint16_t* cdf, int n, int symbol) {
    bits += sink.symbol(cdf, n, symbol);
  };
  switch (set) {
    case residual::InterTxSet::kWide64:
    case residual::InterTxSet::kHigh64:
      sym(&store.inter_tx_type_long[ctx][sqr][0], 4, cand.symbol);
      break;
    case residual::InterTxSet::kWide32:
    case residual::InterTxSet::kHigh32:
      sym(&store.is_long_side_dct[1][0], 2, cand.is_long_side_dct ? 1 : 0);
      sym(&store.inter_tx_type_long[ctx][sqr][0], 4, cand.symbol);
      break;
    case residual::InterTxSet::kInter1: {
      sym(&store.inter_tx_type_set1[ctx][sqr][0], 2, cand.symbol);
      const int off_ctx = inter_tx_type_ctx(eob, coded_w, coded_h, false);
      if (cand.symbol == 0) {
        sym(&store.inter_tx_type_index_set1[off_ctx][0], 8,
            cand.offset_symbol);
      } else {
        sym(&store.inter_tx_type_offset_set1[off_ctx][0], 8,
            cand.offset_symbol);
      }
      break;
    }
    case residual::InterTxSet::kInter2: {
      sym(&store.inter_tx_type_set2[ctx][0], 2, cand.symbol);
      const int off_ctx = inter_tx_type_ctx(eob, coded_w, coded_h, false);
      if (cand.symbol == 0) {
        sym(&store.inter_tx_type_index_set2[off_ctx][0], 8,
            cand.offset_symbol);
      } else {
        sym(&store.inter_tx_type_offset_set2[off_ctx][0], 4,
            cand.offset_symbol);
      }
      break;
    }
    case residual::InterTxSet::kDctIdtx:
      sym(&store.inter_tx_type_set3[ctx][sqr][0], 2, cand.symbol);
      break;
    default:
      return kImpossibleBits;
  }
  return bits;
}

// §5.20.7.20 shell-coded MVD, mirrored from Writer::write_mv().
inline double mvd_bits(const CdfStore& store, rdoq::av2::SymbolSink& sink,
                       int dy, int dx) {
  const std::int32_t ay = std::abs(dy);
  const std::int32_t ax = std::abs(dx);
  const std::int32_t shell = ay + ax;
  if (shell >= (1 << 17)) return kImpossibleBits;
  const int cls = shell == 0 ? 0 : floor_log2_int(static_cast<int>(shell));
  const std::int32_t base = cls == 0 ? 0 : (1 << cls);
  const std::int32_t off = shell - base;
  double bits = 0.0;
  auto sym = [&](const std::uint16_t* cdf, int n, int symbol) {
    bits += sink.symbol(cdf, n, symbol);
  };
  const int setsym = cls >= 8 ? 1 : 0;
  const int class_sym = cls >= 8 ? cls - 8 : cls;
  sym(&store.shell_set[0], 2, setsym);
  sym(setsym ? &store.shell6_class1[0] : &store.shell6_class0[0], 8,
      class_sym);
  if (setsym && class_sym == 7) {
    sym(&store.shell_last_two[0], 2, cls - 15);
  }
  if (cls < 2) {
    sym(&store.offset_low[cls][0], 2, static_cast<int>(off));
  } else if (cls == 2) {
    if (off == 0) {
      sym(&store.offset_class2[0], 2, 0);
    } else {
      sym(&store.offset_class2[0], 2, 1);
      if (off == 1) {
        bits += sink.literal(1.0);
      } else {
        bits += sink.literal(2.0);
      }
    }
  } else {
    for (int i = 0; i < cls; ++i) {
      sym(&store.offset_other[i][0], 2, static_cast<int>((off >> i) & 1));
    }
  }
  if (shell > 0) {
    const std::int32_t max_pair = shell >> 1;
    bool skip_col_bit = false;
    std::int32_t col = 0;
    if (ax == max_pair && ((shell & 1) == 0)) {
      skip_col_bit = true;
      col = max_pair;
    } else if (ax <= max_pair) {
      col = ax;
    } else {
      col = ay;
    }
    if (max_pair > 0) {
      const int max_bits = std::min<std::int32_t>(max_pair, 2);
      const std::int32_t prefix_col = std::min<std::int32_t>(col, 2);
      for (int i = 0; i < max_bits; ++i) {
        const int greater = prefix_col > i ? 1 : 0;
        sym(&store.col_mv_greater[i][0], 2, greater);
        if (greater == 0) break;
      }
      if (max_pair > 2 && prefix_col == 2) {
        bits += sink.literal(static_cast<double>(
            ns_bit_count(static_cast<std::uint32_t>(max_pair - 1),
                         static_cast<std::uint32_t>(col - 2))));
      }
    }
    if (!skip_col_bit) {
      const int index = ax <= max_pair ? 0 : 1;
      sym(&store.col_mv_index[std::min(cls, 3)][0], 2, index);
    }
  }
  if (dy != 0) bits += sink.literal(1.0);
  if (dx != 0) bits += sink.literal(1.0);
  return bits;
}

// Inter mode group: is_inter, skip, single_mode, use_bawp, drl_idx, MVD.
// Contexts are the neutral 0 (the writer derives neighbour contexts that the
// decision layer does not have; measured as model error).
inline double inter_mode_bits(const CdfStore& store,
                              rdoq::av2::SymbolSink& sink,
                              const LeafRateInput& in, int max_drl_bits = 4) {
  double bits = 0.0;
  auto sym = [&](const std::uint16_t* cdf, int n, int symbol) {
    bits += sink.symbol(cdf, n, symbol);
  };
  sym(&store.is_inter[0][0], 2, 1);
  sym(&store.skip[0][0], 2, in.skip_flag ? 1 : 0);
  const int single_mode =
      in.inter_mode == syntax::kInterGlobalMv
          ? 1
          : (in.inter_mode == syntax::kInterNearMv ? 0 : 2);
  sym(&store.single_mode[0][0], 3, single_mode);
  if (in.use_bawp) {
    sym(&store.use_bawp[0], 2, 1);
    const int single_ctx = in.inter_mode == syntax::kInterNewMv ? 2 : 0;
    sym(&store.explicit_bawp[single_ctx][0], 2, 0);  // fitted form
    sym(&store.use_bawp_chroma[0], 2, 0);
  } else if (in.inter_mode != syntax::kInterGlobalMv) {
    sym(&store.use_bawp[0], 2, 0);
  }
  if (in.inter_mode == syntax::kInterNearMv ||
      in.inter_mode == syntax::kInterNewMv) {
    const int drl = in.inter_mode == syntax::kInterNewMv ? 0 : in.drl_index;
    for (int idx = 0; idx < max_drl_bits; ++idx) {
      const int mode = idx < drl ? 1 : 0;
      sym(&store.drl_mode[std::min(idx, 2)][0][0], 2, mode);
      if (mode == 0) break;
    }
  }
  if (in.inter_mode == syntax::kInterNewMv) {
    bits += mvd_bits(store, sink, in.mvd_y, in.mvd_x);
  }
  return bits;
}

// TX partition symbols (§5.20.6.3), via the reference module.
class PartitionSink final : public txpartition::SymbolSink {
 public:
  explicit PartitionSink(rdoq::av2::SymbolSink& sink) : sink_(sink) {}
  bool symbol(std::uint16_t* cdf, int n, int value,
              std::string& error) override {
    const double bits = sink_.symbol(cdf, n, value);
    if (bits >= kImpossibleBits) {
      error = "rate-interval: TX partition symbol failed";
      return false;
    }
    return true;
  }

 private:
  rdoq::av2::SymbolSink& sink_;
};

// One transform block through the §8.3 coefficient model.
inline double block_bits(const LeafRateInput& in, const CdfStore& store,
                         const still::qctx::CoeffCdfSlice& slice,
                         rdoq::av2::SymbolSink& sink, const LeafPart& part) {
  (void)store;  // rows are passed through `slice`; kept for call symmetry
  rdoq::av2::BlockInfo info;
  info.plane = part.plane;
  const auto params = txv::make_tx_params(
      static_cast<txv::TxSize>(part.tx_size),
      static_cast<txv::TxType>(part.tx_type));
  if (!params.has_value()) return kImpossibleBits;
  info.params = *params;
  info.txb_skip_context = 0;
  info.dc_sign_context = 0;
  info.qctx = still::qctx::slice_for_base_q_idx(in.base_q_idx);
  info.is_inter = in.is_inter;
  info.cdf_override = &slice;
  rdoq::av2::Model model;
  std::string error;
  if (!model.init(info, part.levels, error)) return kImpossibleBits;
  model.set_symbol_sink(&sink);
  return model.scalar_bits();
}

inline LeafRate estimate_impl(const LeafRateInput& in, const CdfStore& store,
                              Mode m, bool compare_exact) {
  LeafRate out;
  const still::qctx::CoeffCdfSlice slice = slice_of(store);
  const bool use_exact = (m == Mode::kExact);

  // The interval sink always runs: in exact mode it still prices the same
  // symbol sequence, which is what the error report compares against.
  IntervalSink interval(in.adapt);
  std::unique_ptr<ExactSink> exact;
  if (use_exact) exact = std::make_unique<ExactSink>(in.adapt);
  rdoq::av2::SymbolSink& coder =
      use_exact ? static_cast<rdoq::av2::SymbolSink&>(*exact) : interval;

  // Residual syntax: TX partition (once, for the luma plan), then luma
  // blocks in decode order, then U, then V.
  if (!in.skip_flag && in.part_count > 0) {
    if (in.tx_mode_select && in.bsize != txpartition::kBlock4x4) {
      PartitionSink partition_sink(coder);
      txpartition::BlockTxContext ctx;
      ctx.mi_size = in.bsize;
      ctx.is_inter = in.is_inter;
      ctx.fsc_mode = false;
      ctx.reduced_tx_part_set = false;
      txpartition::SplitGeometry geometry;
      std::string error;
      txpartition::TxPartitionCdfs cdfs = store.tx_partition;
      if (!txpartition::write_tx_partition(
              partition_sink, cdfs, ctx,
              static_cast<txpartition::TxPartition>(in.tx_partition),
              &geometry, error)) {
        out.failed = true;
        return out;
      }
    }
    for (int i = 0; i < in.part_count; ++i) {
      const LeafPart& part = in.parts[i];
      if (part.plane == 0 && in.is_inter && part.eob > 0) {
        const auto params = txv::make_tx_params(
            static_cast<txv::TxSize>(part.tx_size), txv::TxType::kDctDct);
        if (!params.has_value()) {
          out.failed = true;
          return out;
        }
        inter_tx_type_bits(store, coder, part.tx_size, part.tx_type, part.eob,
                           params->coded_w, params->coded_h);
      }
      block_bits(in, store, slice, coder, part);
    }
  }
  out.residual_bits = interval.failed() ? kImpossibleBits : interval.bits();
  if (use_exact) {
    out.exact_prefix_bits = exact->prefix_bits();
    out.exact_payload_bits = exact->payload_bits();
    if (exact->failed()) out.failed = true;
  }

  // Inter mode group.  Intra leaves price no mode: the neighbour-derived
  // y_mode set/index inversion lives in the writer and is out of scope.
  if (in.is_inter && in.inter_mode >= 0) {
    if (use_exact) {
      ExactSink mode_sink(in.adapt);
      inter_mode_bits(store, mode_sink, in);
      out.mode_bits = mode_sink.prefix_bits();
    } else {
      IntervalSink mode_sink(in.adapt);
      inter_mode_bits(store, mode_sink, in);
      out.mode_bits =
          mode_sink.failed() ? kImpossibleBits : mode_sink.bits();
    }
  }

  if (compare_exact && !use_exact) {
    LeafRate exact_rate = estimate_impl(in, store, Mode::kExact, false);
    out.exact_prefix_bits = exact_rate.residual_bits;
    out.exact_payload_bits = exact_rate.exact_payload_bits;
    if (exact_rate.failed) out.failed = true;
  }
  return out;
}

}  // namespace detail

// Main entry point.  `compare_exact` is the diagnostic ablation: it also runs
// the trial encoder on the same symbol sequence and reports its bits.
inline LeafRate estimate(const LeafRateInput& in, const CdfStore& store,
                         Mode m, bool compare_exact = false) {
  if (m == Mode::kOff) return LeafRate{};
  return detail::estimate_impl(in, store, m, compare_exact);
}

// ---------------------------------------------------------------------------
// Per-leaf predicted-vs-actual report (AV2ENC_RATE_INTERVAL_STATS=1).
//
// Predicted: this model run on the leaf the writer emitted (frame-start CDF
// copy, neutral neighbour contexts).  Actual: the writer's symbol
// information cost for the same groups, measured with -log2 p at the live
// CDF (`AV2ENC_SYMBOL_COST` meter), so the comparison is against what the
// coder really spent on the leaf.
// ---------------------------------------------------------------------------
struct LeafSample {
  int bsize = 0;
  int is_inter = 0;
  int inter_frame = 0;
  int skip = 0;
  int intra_mode = 0;
  double predicted_mode = 0.0;
  double predicted_residual = 0.0;
  double legacy_residual = 0.0;  // historical heuristic on the same leaf
  double exact_mode = 0.0;
  double exact_residual = 0.0;
  double actual_mode = 0.0;
  double actual_residual = 0.0;
  double actual_intra_mode = 0.0;
  double ns_interval = 0.0;
  double ns_exact = 0.0;
  bool exact_valid = false;
};

struct Stats {
  std::mutex mutex;
  std::vector<LeafSample> leaves;
  std::atomic<unsigned long long> estimates{0};
  std::atomic<unsigned long long> exact_runs{0};
  std::atomic<unsigned long long> skip_residual_flips{0};
  std::atomic<unsigned long long> intra_inter_flips{0};
  std::atomic<unsigned long long> skip_to_residual{0};
  std::atomic<unsigned long long> residual_to_skip{0};
  std::atomic<unsigned long long> intra_to_inter{0};
  std::atomic<unsigned long long> inter_to_intra{0};
};

inline Stats& stats() {
  // Heap-allocated: the atexit report runs after function-local statics are
  // destroyed, and this process-global registry must outlive them.
  static Stats* value = new Stats();
  return *value;
}

inline void record(const LeafSample& sample) {
  if (!detail::stats_enabled()) return;
  Stats& s = stats();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.leaves.push_back(sample);
}

namespace detail {

inline double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t index = static_cast<std::size_t>(
      std::min(1.0, std::max(0.0, p)) * static_cast<double>(v.size() - 1));
  return v[index];
}

inline void report_class(const char* name,
                         const std::vector<const LeafSample*>& rows,
                         bool residual_only) {
  if (rows.empty()) return;
  std::vector<double> err;
  std::vector<double> abs_err;
  std::vector<double> exact_err;
  std::vector<double> exact_abs;
  std::vector<double> legacy_err;
  std::vector<double> legacy_abs;
  double sum = 0.0;
  double exact_sum = 0.0;
  double legacy_sum = 0.0;
  double pred_sum = 0.0;
  std::size_t exact_count = 0;
  for (const LeafSample* row : rows) {
    const double predicted = residual_only
                                 ? row->predicted_residual
                                 : row->predicted_mode + row->predicted_residual;
    const double actual = residual_only
                              ? row->actual_residual
                              : row->actual_mode + row->actual_residual;
    const double e = predicted - actual;
    err.push_back(e);
    abs_err.push_back(std::abs(e));
    sum += e;
    pred_sum += predicted;
    const double legacy = row->legacy_residual;
    legacy_err.push_back(legacy - row->actual_residual);
    legacy_abs.push_back(std::abs(legacy - row->actual_residual));
    legacy_sum += legacy_err.back();
    if (row->exact_valid) {
      const double exact = residual_only
                               ? row->exact_residual
                               : row->exact_mode + row->exact_residual;
      exact_err.push_back(exact - actual);
      exact_abs.push_back(std::abs(exact - actual));
      exact_sum += exact - actual;
      ++exact_count;
    }
  }
  const double n = static_cast<double>(rows.size());
  const double actual_mean = (pred_sum - sum) / (n > 0.0 ? n : 1.0);
  std::fprintf(stderr,
               "rate-interval: class=%s leaves=%zu actual_mean=%.1f "
               "legacy_med=%.2f legacy_p90=%.2f legacy_bias=%.2f "
               "pred_med=%.2f pred_p90=%.2f pred_bias=%.2f",
               name, rows.size(), actual_mean, percentile(legacy_err, 0.5),
               percentile(legacy_abs, 0.9), legacy_sum / n,
               percentile(err, 0.5), percentile(abs_err, 0.9), sum / n);
  if (exact_count > 0) {
    std::fprintf(stderr, " exact_med=%.2f exact_p90=%.2f exact_bias=%.2f",
                 percentile(exact_err, 0.5), percentile(exact_abs, 0.9),
                 exact_sum / static_cast<double>(exact_count));
  }
  std::fprintf(stderr, "\n");
}

}  // namespace detail

inline void print_stats() {
  if (!detail::stats_enabled()) return;
  Stats& s = stats();
  std::vector<LeafSample> rows;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    rows = s.leaves;
  }
  if (rows.empty()) return;
  std::fprintf(stderr,
               "rate-interval: leaves=%zu estimates=%llu exact_runs=%llu "
               "skip_residual_flips=%llu intra_inter_flips=%llu "
               "skip->res=%llu res->skip=%llu intra->inter=%llu "
               "inter->intra=%llu\n",
               rows.size(),
               static_cast<unsigned long long>(s.estimates.load()),
               static_cast<unsigned long long>(s.exact_runs.load()),
               static_cast<unsigned long long>(
                   s.skip_residual_flips.load()),
               static_cast<unsigned long long>(s.intra_inter_flips.load()),
               static_cast<unsigned long long>(s.skip_to_residual.load()),
               static_cast<unsigned long long>(s.residual_to_skip.load()),
               static_cast<unsigned long long>(s.intra_to_inter.load()),
               static_cast<unsigned long long>(s.inter_to_intra.load()));
  double ns_interval = 0.0;
  double ns_exact = 0.0;
  double ns_interval_res = 0.0;
  double ns_exact_res = 0.0;
  std::size_t exact_count = 0;
  std::size_t residual_count = 0;
  std::size_t exact_res_count = 0;
  for (const LeafSample& row : rows) {
    ns_interval += row.ns_interval;
    if (row.exact_valid) {
      ns_exact += row.ns_exact;
      ++exact_count;
    }
    if (row.predicted_residual > 0.0) {
      ns_interval_res += row.ns_interval;
      ++residual_count;
      if (row.exact_valid) {
        ns_exact_res += row.ns_exact;
        ++exact_res_count;
      }
    }
  }
  std::fprintf(stderr,
               "rate-interval: cost all: interval=%.3f us/leaf "
               "exact=%.3f us/leaf\n",
               ns_interval / 1000.0 / static_cast<double>(rows.size()),
               exact_count > 0 ? ns_exact / 1000.0 /
                                     static_cast<double>(exact_count)
                               : 0.0);
  if (residual_count > 0) {
    std::fprintf(stderr,
                 "rate-interval: cost residual leaves (%zu): "
                 "interval=%.3f us/leaf exact=%.3f us/leaf\n",
                 residual_count,
                 ns_interval_res / 1000.0 /
                     static_cast<double>(residual_count),
                 exact_res_count > 0
                     ? ns_exact_res / 1000.0 /
                           static_cast<double>(exact_res_count)
                     : 0.0);
  }
  // Per-class residual error, split by leaf kind.
  {
    std::vector<const LeafSample*> inter_rows;
    std::vector<const LeafSample*> intra_rows;
    for (const LeafSample& row : rows) {
      (row.is_inter ? inter_rows : intra_rows).push_back(&row);
    }
    detail::report_class("inter(all)", inter_rows, /*residual_only=*/false);
    detail::report_class("intra(all)", intra_rows, /*residual_only=*/false);
    // Residual-only view: comparable across leaf kinds.
    detail::report_class("inter(res)", inter_rows, /*residual_only=*/true);
    detail::report_class("intra(res)", intra_rows, /*residual_only=*/true);
    // Leaves that actually code a residual: the decision-relevant subset.
    std::vector<const LeafSample*> inter_res;
    std::vector<const LeafSample*> intra_res;
    for (const LeafSample& row : rows) {
      if (row.predicted_residual <= 0.0 && row.actual_residual <= 0.0) {
        continue;
      }
      (row.is_inter ? inter_res : intra_res).push_back(&row);
    }
    detail::report_class("inter(coded-res)", inter_res,
                         /*residual_only=*/true);
    detail::report_class("intra(coded-res)", intra_res,
                         /*residual_only=*/true);
    // Intra leaves split by frame kind: the keyframe CDFs adapt across its
    // whole tile, the P-frame intra rows hardly move.
    std::vector<const LeafSample*> key_intra;
    std::vector<const LeafSample*> p_intra;
    for (const LeafSample* row : intra_res) {
      (row->inter_frame ? p_intra : key_intra).push_back(row);
    }
    detail::report_class("key-intra(res)", key_intra, /*residual_only=*/true);
    detail::report_class("p-intra(res)", p_intra, /*residual_only=*/true);
  }
  // Block-size split.
  for (int bsize = 0; bsize < 32; ++bsize) {
    std::vector<const LeafSample*> group;
    for (const LeafSample& row : rows) {
      if (row.bsize == bsize) group.push_back(&row);
    }
    if (group.size() < 16) continue;
    char name[32];
    std::snprintf(name, sizeof(name), "bsize%02d", bsize);
    detail::report_class(name, group, /*residual_only=*/true);
  }
}

}  // namespace av2enc::video::rate_interval
