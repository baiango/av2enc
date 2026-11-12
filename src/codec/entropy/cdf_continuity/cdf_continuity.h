// AV2 v1.0.0 tile/frame CDF continuity (§5.18.2, §5.18.7.2, §6.17.2,
// §7.4.4, §7.5, §8.2.2, §8.2.4).
//
// This is the missing layer between the syntax writers and the CDF bank in
// `src/encoder/entropy/avg_cdf/`.  The bank owns the arrays and the exact
// integer arithmetic; this module owns the *decisions* and the *order*:
//
//   * §5.18.7.2  tile_info(): is `context_update_tile_id` coded, and with how
//                many bits?
//   * §8.2.4     exit_symbol(): per tile, is it `copyCdf`, `avgCdf`, or
//                neither?
//   * §6.17.2    init_coeff_cdfs()/init_non_coeff_cdfs(): does the frame reset
//                to §9.3 defaults, load a reference context, or load + blend?
//   * §7.5       frame_end_update_cdf(): Saved* -> frame* with the 3/4 count.
//   * §7.23      save_cdfs(i) for every slot in refresh_frame_flags.
//
// The entropy coder itself is not touched: `av2::Encoder`/`av2::Decoder`
// already implement adaptation and the `disable_cdf_update` gate.  The frame
// header bit is carried in `FrameCdfPlan::disable_cdf_update` so a writer can
// assert that the bit it emits matches the flag it gives the coder.
//
// The one thing the writers cannot currently provide is a CDF *seed*: both
// `av2enc::still::syntax::Writer` and `av2enc::video::syntax::Writer` run `init_cdfs()` from the
// defaults inside their constructors and keep the resulting `CdfState`
// private (`still_syntax.cpp:1492`, `video_syntax.cpp:1992`).  This module therefore
// exposes a controller whose tile CDF arrays can be handed to any entropy
// coder; the lockstep test builds a standalone pair around it.  See README.md
// for why production streams can never enter the load/blend branches today.
//
// Ownership: this file may include `encoder/entropy/avg_cdf/` read-only.  No
// M1/M2/CMake/CLI file is modified.

#pragma once

#include "codec/entropy/avg_cdf/avg_cdf_bank.h"
#include "codec/entropy/qctx/qctx.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace av2enc::entropy::continuity {

// §3: NUM_REF_FRAMES.  One `save_cdfs` context per reference buffer slot.
inline constexpr std::uint32_t kNumRefFrames = 16;

// §6.17.2 init_coeff_cdfs(): four COEFF_CDF_Q_CTXS slices.  The registry and
// its selection rule live in `src/codec/entropy/qctx/`; alias them here so the
// 90/140/190 thresholds exist in exactly one place.
inline constexpr int kCoeffCdfSliceCount = still::qctx::kSliceCount;
inline constexpr int kSlice0MaxBaseQIdx = still::qctx::kSlice0MaxBaseQIdx;
inline constexpr int kSlice1MaxBaseQIdx = still::qctx::kSlice1MaxBaseQIdx;
inline constexpr int kSlice2MaxBaseQIdx = still::qctx::kSlice2MaxBaseQIdx;

// §6.17.2: the coefficient-CDF slice is selected from the *raw frame*
// `base_q_idx`, never from a tile/segment/delta-Q index.
constexpr int qctx_slice_for_base_q_idx(int raw_base_q_idx) noexcept {
  return still::qctx::slice_for_base_q_idx(raw_base_q_idx);
}

// §5.18.7.2 tile_info() tail.  `context_update_tile_id` is coded only when
//   (TileCols > 1 || TileRows > 1) && !IsBridge &&
//   TipFrameMode != TIP_FRAME_AS_OUTPUT &&
//   (!enable_avg_cdf || !avg_cdf_type)
// with `n = TileRowsLog2 + TileColsLog2` bits.  Otherwise the spec sets
// `context_update_tile_id = 0` and codes no bits.  `tile_size_bytes_minus_1`
// f(2) is always present in the multi-tile branch, independent of the CDF
// fields.
struct TileInfoCdfField {
  bool cdf_tail_coded = false;        // multi-tile, not bridge/TIP-output
  bool context_update_present = false;
  std::uint32_t tile_cols = 1;
  std::uint32_t tile_rows = 1;
  std::uint32_t tile_cols_log2 = 0;   // CeilLog2(TileCols)
  std::uint32_t tile_rows_log2 = 0;   // CeilLog2(TileRows)
  std::uint32_t context_update_bits = 0;  // 0 when absent
  std::uint32_t context_update_max = 0;   // (1 << bits) - 1
  // Bit cost of the whole tile_info() tail after the tile-log2 increments:
  // context_update_bits + (cdf_tail_coded ? 2 : 0).
  unsigned tail_bits() const noexcept {
    return static_cast<unsigned>(context_update_bits) +
           (cdf_tail_coded ? 2U : 0U);
  }
};

// CeilLog2 for the §5.18.7.3 `tile_log2(1, tileCols)` derivation.
std::uint32_t ceil_log2_u32(std::uint32_t value) noexcept;

TileInfoCdfField tile_info_cdf_field(bool enable_avg_cdf, bool avg_cdf_type,
                                     std::uint32_t tile_cols,
                                     std::uint32_t tile_rows,
                                     bool is_bridge = false,
                                     bool tip_frame_as_output = false) noexcept;

// §6.17.7.3: it is a requirement of bitstream conformance that
// context_update_tile_id < TileCols * TileRows when it is coded.  When it is
// absent the caller must pass 0 (the inference).
bool validate_context_update_tile_id(const TileInfoCdfField& field,
                                     std::uint32_t context_update_tile_id,
                                     std::string* error = nullptr);

// §8.2.4 exit process decision for one tile.
struct TileExitPlan {
  std::uint32_t tile_num = 0;
  std::uint32_t tile_count = 0;
  std::uint32_t num_log2 = 0;         // Min(3, FloorLog2(tile_count))
  std::uint32_t average_limit = 0;    // 1 << num_log2
  bool avg_cdf = false;
  bool copy_cdf = false;
};

TileExitPlan plan_tile_exit(bool enable_avg_cdf, bool avg_cdf_type,
                            std::uint32_t tile_num, std::uint32_t tile_cols,
                            std::uint32_t tile_rows,
                            std::uint32_t context_update_tile_id) noexcept;

// §5.18.2 / §6.17.2 frame CDF installation.
enum class FrameCdfSource {
  kDefaults,                // init_non_coeff_cdfs() + init_coeff_cdfs()
  kPrimaryReference,        // load_cdfs(ref_frame_idx[primary_ref_frame])
  kPrimaryReferenceBlended  // load_cdfs(...) then blend_cdfs(blendFrame)
};

const char* frame_cdf_source_name(FrameCdfSource source) noexcept;

// Everything the frame header has resolved by the time
// `set_primary_ref_frame_and_ctx(1)` runs in §5.18.2.  `tile_cols`/`tile_rows`
// and the average-CDF flags are here because tile_info() precedes the CDF
// installation in the same syntax function.
struct FrameCdfInput {
  // --- tile_info() / §8.2.4 ---
  std::uint32_t tile_cols = 1;
  std::uint32_t tile_rows = 1;
  std::uint32_t context_update_tile_id = 0;
  bool enable_avg_cdf = false;
  bool avg_cdf_type = false;
  bool is_bridge = false;
  bool tip_frame_as_output = false;

  // --- §5.18.2 / §6.17.2 ---
  bool keyframe = false;                    // CLK / OLK / FrameIsIntra
  bool primary_ref_none = false;            // signalled PRIMARY_REF_NONE
  bool derived_primary_ref_none = false;    // choose_primary_secondary_ref_frame()
  bool disable_cross_frame_cdf_init = false;
  // §7.4.4 blend condition also excludes BRU-inactive frames and TIP output.
  bool bru_inactive = false;
  // ref_frame_idx[primary_ref_frame], valid when the load branch is taken.
  std::uint32_t primary_ref_slot = 0;
  std::uint32_t blend_ref_slot = 0;
  bool blend_ref_present = false;           // blendFrame != PRIMARY_REF_NONE
  // Raw frame base_q_idx (not a tile qindex).
  int raw_base_q_idx = 0;
  // Frame-header bit handed to the entropy coder.
  bool disable_cdf_update = false;
};

struct FrameCdfPlan {
  FrameCdfSource source = FrameCdfSource::kDefaults;
  TileInfoCdfField tile_info{};
  std::uint32_t tile_count = 0;
  std::uint32_t context_update_tile_id = 0;
  bool enable_avg_cdf = false;
  bool avg_cdf_type = false;
  int qctx_slice = 0;
  bool reset = true;
  bool primary_ref_none = true;
  bool blend = false;
  std::uint32_t primary_ref_slot = 0;
  std::uint32_t blend_ref_slot = 0;
  bool disable_cdf_update = false;
  // False when the header claims adaptation is off; the coder must be given
  // the same value.  CopyCdf/avgCdf still run on exit in that case (the tile
  // arrays simply did not change).
  bool adaptation_enabled() const noexcept { return !disable_cdf_update; }
};

// Pure planner; never asserts on malformed input (use the validator below).
FrameCdfPlan plan_frame_cdfs(const FrameCdfInput& input) noexcept;

bool validate_frame_cdfs(const FrameCdfInput& input,
                         std::string* error = nullptr);

enum class ContinuityStatus {
  kOk = 0,
  kInvalidArgument,
  kNotInitialised,
  kFrameAlreadyOpen,
  kFrameNotOpen,
  kTileAlreadyOpen,
  kTileNotOpen,
  kTileOutOfRange,
  kFrameIncomplete,
  kContextOutOfRange,
  kCdfBankError,
};

const char* continuity_status_name(ContinuityStatus status) noexcept;

// Ordered CDF bookkeeping for one encoder/decoder context.  One instance per
// independent entropy context (a decoder has one; an encoder mirroring the
// decoder has one).  The exact call order is:
//
//   // frame header, at the §5.18.2 set_primary_ref_frame_and_ctx(1) seam
//   ctl.begin_frame(input);
//   for (tile = 0; tile < tile_count; ++tile) {
//     ctl.begin_tile();                 // immediately before init_symbol()
//     ... code symbols through ctl.bank().tile_cdf(i) ...
//     ctl.end_tile(tile);               // immediately after exit_symbol()
//   }
//   ctl.end_frame();                    // tg_end == NumTiles - 1 (§7.5)
//   ctl.save_reference(slot);           // §7.23 save_cdfs(slot)
//
// `begin_frame` is the only place the §6.17.2 reset/load/blend decision runs.
// `begin_tile` is the §8.2.2 frame -> Tile* copy.  `end_tile` applies the
// §8.2.4 copyCdf/avgCdf choice.  `end_frame` is §7.5.  `save_reference` is
// §7.4.4/§7.23.
class FrameCdfContinuity {
 public:
  FrameCdfContinuity() = default;
  FrameCdfContinuity(const FrameCdfContinuity&) = delete;
  FrameCdfContinuity& operator=(const FrameCdfContinuity&) = delete;
  FrameCdfContinuity(FrameCdfContinuity&&) = delete;
  FrameCdfContinuity& operator=(FrameCdfContinuity&&) = delete;

  // `reference_contexts` is normally kNumRefFrames.  Passing 0 disables the
  // load/save reference API and rejects any frame that needs it.
  bool init_owned(std::span<const avg_cdf::CdfSpec> specs,
                  std::uint32_t reference_contexts,
                  std::string* error = nullptr);
  bool init(std::span<const avg_cdf::CdfTable> tables,
            std::uint32_t reference_contexts, std::string* error = nullptr);

  ContinuityStatus begin_frame(const FrameCdfInput& input,
                               std::string* error = nullptr);
  ContinuityStatus begin_tile(std::string* error = nullptr);
  ContinuityStatus end_tile(std::uint32_t tile_num,
                            std::string* error = nullptr);
  ContinuityStatus end_frame(std::string* error = nullptr);
  ContinuityStatus save_reference(std::uint32_t slot,
                                  std::string* error = nullptr);

  bool frame_open() const noexcept { return frame_open_; }
  bool tile_open() const noexcept { return tile_open_; }
  std::uint32_t next_tile() const noexcept { return next_tile_; }
  std::uint32_t tile_count() const noexcept { return tile_count_; }
  const FrameCdfPlan& plan() const noexcept { return plan_; }
  std::uint32_t context_update_tile_id() const noexcept {
    return plan_.context_update_tile_id;
  }
  TileExitPlan tile_exit_plan(std::uint32_t tile_num) const noexcept;

  const avg_cdf::CdfBank& bank() const noexcept { return bank_; }
  avg_cdf::CdfBank& bank() noexcept { return bank_; }

 private:
  bool initialised_ = false;
  bool frame_open_ = false;
  bool tile_open_ = false;
  std::uint32_t next_tile_ = 0;
  std::uint32_t tile_count_ = 0;
  FrameCdfPlan plan_{};
  avg_cdf::CdfBank bank_{};
};

}  // namespace av2enc::entropy::continuity
