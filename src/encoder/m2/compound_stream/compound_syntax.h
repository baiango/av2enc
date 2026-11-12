// AV2 M2 compound / skip-mode inter mode-info syntax (§5.20.7.1-.16).
//
// This component owns the block-level inter mode information that the M2 core
// writer does not emit:
//
//   read_skip_mode()             §5.20.5.9   skip_mode S()
//   read_is_inter()              §5.20.7.3   (skip_mode implies is_inter = 1)
//   read_skip()                  §5.20.5.10  skip_flag S() with the +3 context
//   read_ref_frames()            §5.20.7.10  comp_mode / read_compound_ref /
//                                            read_single_ref loop
//   compound_mode_same_refs S()  §5.20.7.6
//   compound_mode_non_joint S()  §5.20.7.6
//   is_joint S()                 §5.20.7.6
//   use_optflow S()              §5.20.7.6
//   single_mode / read_drl_idx   §5.20.7.6 / .8
//   read_compound_type()         §5.20.7.16 (comp_group_idx / compound_type /
//                                            read_wedge_mode / mask_type)
//   cwp_idx S()                  §5.20.7.6
//
// Writer and reader are independent implementations of the same grammar: the
// writer is in compound_syntax.cpp, the reader is in compound_reader.cpp.  Both
// derive every context from their own copy of the per-4x4 neighbour state and
// their own copy of the §9.3 CDFs.
//
// Out of scope (already written by the M2 core, or a different component):
// NEWMV MVD shell coding, warp/BAWP/TIP/refine-MV/adaptive-MVD, residuals.
// A plan that needs any of those is rejected explicitly instead of emitting a
// non-conforming stream.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §5.20.5.9-.10, §5.20.7.1-.16, §9.3.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "encoder/inter/advanced/advanced_common.h"
#include "encoder/m2/multiref/compound_predict.h"
#include "encoder/m2/multiref/ref_map.h"
#include "entropy/av2/av2_entropy.h"

namespace av2enc::m2::compound {

namespace adv = av2enc::inter::adv;

inline constexpr int kNoneRef = -1;
inline constexpr int kIntraRef = 8;
inline constexpr int kTipRef = 7;

// §6.19.5 Table: YMode values used by this component.
enum YModeValue : int {
  kNearMv = 14,
  kGlobalMv = 15,
  kNewMv = 16,
  kWarpNewMv = 17,
  kNearNearMv = 19,
  kNearNewMv = 20,
  kNewNearMv = 21,
  kGlobalGlobalMv = 22,
  kNewNewMv = 23,
  kJointNewMv = 24,
};

// §5.20.7.6 / §6.4 opfl_refine_type values.
enum OpflRefineType : int {
  kRefineNone = 0,
  kRefineSwitchable = 1,
  kRefineAll = 2,
};

// §5.20.7.6 has_newmv / has_nearmv / has_second_drl.
bool has_newmv(int y_mode);
bool has_nearmv(int y_mode);
bool has_newmv_for_list(int y_mode, int list);

// §5.20.7.10 is_comp_ref_allowed() for a rectangular block.
bool comp_ref_allowed(int block_w, int block_h);

// §5.18.8.2 skip_mode_params(): SkipModeFrame[] for this frame.
void derive_skip_mode_frames(const multiref::RefFrameMap& map,
                             int skip_mode_frame[2]);

// Frame-level state the block grammar depends on.  The reference fields are
// the multiref model of §5.18.2 so the frame-header writer and the tile writer
// cannot disagree about NumTotalRefs / NumSameRefCompound / FrameDistance.
struct CompoundFrameConfig {
  multiref::RefFrameMap ref_map;
  bool skip_mode_present = false;
  int skip_mode_frame[2] = {0, 0};
  int opfl_refine_type = kRefineNone;
  bool enable_masked_compound = false;
  bool enable_cwp = false;
  bool enable_imp_msk_bld = false;
  int max_drl_bits_minus_1 = 4;
  bool disable_cdf_update = true;
  bool frame_is_switch = false;
  bool refs_scaled = false;

  // Tools that are not part of this component.  They must stay off; the
  // writer rejects a block that would need one of them.
  bool enable_tip = false;
  bool enable_bawp = false;
  bool allow_warpmv_mode = false;
  bool enable_adaptive_mvd = false;
  bool enable_flex_mvres = false;
  bool enable_refinemv = false;

  int num_total_refs() const { return ref_map.num_total_refs; }
  int num_same_ref_compound() const {
    return ref_map.num_same_ref_compound < ref_map.num_total_refs
               ? ref_map.num_same_ref_compound
               : ref_map.num_total_refs;
  }
  bool reference_select() const { return ref_map.reference_select; }
  int frame_distance(int ref) const { return ref_map.frame_distance[ref]; }
  int order_hint_of(int ref) const { return static_cast<int>(ref_map.order_hints[ref]); }
  bool same_side(int ref0, int ref1) const {
    return ref_map.is_same_side(ref0, ref1);
  }
  // §7.8 FurthestFuture: the future reference with the largest order hint.
  int furthest_future() const {
    int best = kNoneRef;
    for (int i = 0; i < ref_map.num_total_refs; ++i) {
      if (ref_map.frame_distance[i] >= 0) continue;
      if (best == kNoneRef ||
          ref_map.order_hints[i] > ref_map.order_hints[best]) {
        best = i;
      }
    }
    return best;
  }
  // §5.20.7.6 opfl_allowed_for_refs().
  bool opfl_allowed(int ref0, int ref1) const;
};

// One inter leaf.  `skip_flag`, `y_mode`, `ref_frame[]`, `drl_index[]`,
// `compound` and `use_optflow` are the signalled decisions.  The writer checks
// them against what the grammar can actually express and fails on anything
// that would desynchronise the decoder.
struct CompoundBlockPlan {
  int mi_row = 0;
  int mi_col = 0;
  int w4 = 1;
  int h4 = 1;
  bool skip_flag = true;
  bool skip_mode = false;
  int y_mode = kGlobalGlobalMv;
  int ref_frame[2] = {0, kNoneRef};
  int drl_index[2] = {0, 0};
  multiref::CompoundModeParams compound;
  bool use_optflow = false;
};

// The reader's view of one decoded leaf.  `is_compound`, `ref_mv_idx` and
// `cwp_weight` are derived rather than coded.
struct CompoundBlockResult {
  int mi_row = 0;
  int mi_col = 0;
  int w4 = 1;
  int h4 = 1;
  bool is_inter = true;
  bool skip_flag = true;
  bool skip_mode = false;
  int y_mode = 0;
  int ref_frame[2] = {kIntraRef, kNoneRef};
  int drl_index[2] = {0, 0};
  multiref::CompoundModeParams compound;
  bool use_optflow = false;
  bool is_compound = false;
  int ref_mv_idx = 0;
  int cwp_weight = adv::kCwpEqual;
};

// Per-4x4 neighbour state.  The writer and reader each own one; it is the
// model of the decoder's IsInters/Skips/SkipModes/YModes/RefFrames/
// CompGroupIdxs/CwpIdxs arrays.
struct NeighbourCell {
  std::uint8_t is_inter = 0;
  std::uint8_t skip = 0;
  std::uint8_t skip_mode = 0;
  std::uint8_t ymode = 0;
  std::uint8_t comp_group = 0;
  std::int8_t ref0 = kIntraRef;
  std::int8_t ref1 = kNoneRef;
};

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------
class ModeInfoWriter {
 public:
  ModeInfoWriter(const CompoundFrameConfig& config, int mi_cols, int mi_rows);
  ~ModeInfoWriter();
  ModeInfoWriter(const ModeInfoWriter&) = delete;
  ModeInfoWriter& operator=(const ModeInfoWriter&) = delete;

  // Codes read_skip_mode / read_is_inter / read_skip and, for an inter block,
  // the complete inter_block_mode_info() up to cwp_idx.  Residual syntax is the
  // caller's responsibility (this component codes skip blocks only).
  bool code_block(av2::Encoder& encoder, const CompoundBlockPlan& plan,
                  CompoundBlockResult* result, std::string& error);

 private:
  bool code_skip_mode(const CompoundBlockPlan& plan, std::string& error);
  bool code_is_inter(const CompoundBlockPlan& plan, std::string& error);
  bool code_skip_flag(const CompoundBlockPlan& plan, std::string& error);
  bool code_ref_frames(const CompoundBlockPlan& plan, CompoundBlockResult* out,
                       std::string& error);
  bool code_single_mode(const CompoundBlockPlan& plan, std::string& error);
  bool code_compound_mode(const CompoundBlockPlan& plan,
                          CompoundBlockResult* out, std::string& error);
  bool code_drl(const CompoundBlockPlan& plan, CompoundBlockResult* out,
                std::string& error);
  bool code_compound_type(const CompoundBlockPlan& plan,
                          CompoundBlockResult* out, std::string& error);
  bool code_cwp(const CompoundBlockPlan& plan, CompoundBlockResult* out,
                std::string& error);

  bool sym(std::uint16_t* cdf, int n, int symbol, std::string& error);
  bool raw_bit(bool bit, std::string& error);

  std::size_t cell_index(int r, int c) const;
  bool inside(int r, int c) const;
  int neighbour_positions(int pos[2][2], int r, int c, int w4, int h4) const;
  int is_inter_context(int r, int c, int w4, int h4) const;
  int skip_context(int r, int c, int w4, int h4, bool skip_mode) const;
  int skip_mode_context(int r, int c, int w4, int h4) const;
  int new_mv_context(int r, int c, int w4, int h4, bool is_compound,
                     int cur_ref0, int cur_ref1) const;
  int comp_mode_context(int r, int c, int w4, int h4) const;
  int ref_context(int r, int c, int w4, int h4, int ref) const;
  int is_joint_context(int ref0, int ref1) const;
  int comp_group_context(int r, int c, int w4, int h4, int ref0,
                         int ref1) const;
  int count_refs(int r, int c, int w4, int h4, int ref) const;
  bool check_backward(int ref) const;
  bool n_single(int r, int c) const;

  void update_grid(const CompoundBlockResult& result);

  const CompoundFrameConfig& config_;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  std::vector<NeighbourCell> cells_;
  av2::Encoder* encoder_ref_ = nullptr;

  struct Cdfs;
  std::unique_ptr<Cdfs> cdfs_;
};

// ---------------------------------------------------------------------------
// Independent reader
// ---------------------------------------------------------------------------
class ModeInfoReader {
 public:
  ModeInfoReader(const CompoundFrameConfig& config, int mi_cols, int mi_rows);
  ~ModeInfoReader();
  ModeInfoReader(const ModeInfoReader&) = delete;
  ModeInfoReader& operator=(const ModeInfoReader&) = delete;

  // Mirrors ModeInfoWriter::code_block().  Returns false only on a malformed
  // stream or an unsupported (never written) block.
  bool read_block(av2::Decoder& decoder, int mi_row, int mi_col, int w4,
                  int h4, CompoundBlockResult* result, std::string& error);

 private:
  bool read_block_internal(av2::Decoder& decoder, int mi_row, int mi_col,
                           int w4, int h4, CompoundBlockResult* result,
                           std::string& error);
  bool code_symbol(av2::Decoder& decoder, std::uint16_t* cdf, int n,
                   int* symbol, std::string& error);
  bool code_literal(av2::Decoder& decoder, int n, std::uint32_t* value,
                    std::string& error);

  std::size_t cell_index(int r, int c) const;
  bool inside(int r, int c) const;
  int neighbour_positions(int pos[2][2], int r, int c, int w4, int h4) const;
  int is_inter_context(int r, int c, int w4, int h4) const;
  int skip_context(int r, int c, int w4, int h4, bool skip_mode) const;
  int skip_mode_context(int r, int c, int w4, int h4) const;
  int new_mv_context(int r, int c, int w4, int h4, bool is_compound,
                     int cur_ref0, int cur_ref1) const;
  int comp_mode_context(int r, int c, int w4, int h4) const;
  int ref_context(int r, int c, int w4, int h4, int ref) const;
  int is_joint_context(int ref0, int ref1) const;
  int comp_group_context(int r, int c, int w4, int h4, int ref0,
                         int ref1) const;
  int count_refs(int r, int c, int w4, int h4, int ref) const;
  bool check_backward(int ref) const;
  bool n_single(int r, int c) const;
  void update_grid(const CompoundBlockResult& result);

  const CompoundFrameConfig& config_;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  std::vector<NeighbourCell> cells_;

  struct Cdfs;
  std::unique_ptr<Cdfs> cdfs_;
};

// ---------------------------------------------------------------------------
// Tile payload (one 64x64 PARTITION_NONE leaf per superblock)
// ---------------------------------------------------------------------------
bool write_compound_tile(const std::vector<CompoundBlockPlan>& blocks, int width,
                         int height, const CompoundFrameConfig& config,
                         std::vector<std::uint8_t>& payload, std::string& error);

bool read_compound_tile(std::span<const std::uint8_t> payload, int width,
                        int height, const CompoundFrameConfig& config,
                        std::vector<CompoundBlockResult>& blocks,
                        std::string& error);

}  // namespace av2enc::m2::compound
