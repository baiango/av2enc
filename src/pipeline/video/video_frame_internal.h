// Private split of video_frame.cpp: helpers and the InterFrameEncoder
// declaration shared by the per-role translation units.  No behaviour
// lives here that was not in video_frame.cpp before the split.
#pragma once

#include <atomic>

#include "pipeline/video/video_frame.h"

#include "codec/intra/modes/intra_modes.h"
#include "codec/intra/dip/dip.h"
#include "codec/intra/mrl/core/mrl.h"
#include "codec/intra/mrl/stream/mrl_interaction.h"
#include "codec/transform/secondary/tx_secondary.h"
#include "frame/metadata/content_interpretation/content_interpretation.h"
#include "frame/mfh/mfh.h"
#include "pipeline/video/video_exact_rd.h"
#include "pipeline/video/video_rate_interval.h"
#include "pipeline/video/video_frames.h"
#include "pipeline/video/video_headers.h"
#include "pipeline/video/video_lcr_ops.h"
#include "pipeline/video/video_chroma_search.h"
#include "pipeline/video/video_intra_search.h"
#include "pipeline/video/qp_lambda.h"
#include "pipeline/video/video_syntax.h"
#include "pipeline/video/video_rate_hybrid.h"
#include "pipeline/video/video_pipe_perc.h"
#include "pipeline/video/video_eff_perc.h"
#include "pipeline/video/video_pipe_ba.h"
#include "pipeline/video/video_temporal.h"
#include "pipeline/video/video_tx_search.h"
#include "pipeline/video/video_tail_tx.h"
#include "pipeline/video/video_part_diag.h"
#include "pipeline/video/video_tail_floor.h"
#include "pipeline/video/video_joint_plan.h"
#include "pipeline/video/video_pipe_budget.h"
#include "pipeline/video/video_decision_rate.h"
#include "codec/partition/tree/partition.h"
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
#include "codec/partition/ext/partition_ext.h"
#endif
#include "codec/partition/sdp/sdp.h"
#include "codec/rate/rc/rc_tables.h"
#if defined(AV2ENC_VIDEO_DELTAQ)
#include "codec/entropy/deltaq/still/deltaq.h"
#include "codec/rate/rc/rc.h"
#endif
#include "codec/transform/variants/tx_variants.h"
#include "codec/transform/delta_q/dcdeltaq.h"
#include "codec/rate/rdo/rdo.h"
#include "codec/rate/rdoq/rdoq.h"
#include "codec/rate/rdoq/rdoq_av2_model.h"
#include "codec/transform/ddt/ddt.h"
#include "codec/transform/qm/qm.h"
#include "codec/transform/tcq/cctx.h"
#include "codec/transform/tcq/tcq.h"
#include "codec/transform/tcq/tcq_av2_model.h"
#include "codec/transform/tx_partition/tx_partition.h"
#include "codec/entropy/qctx/qctx.h"
#include "codec/rate/rdo/rdo.h"
#include "pipeline/video/video_bawp.h"
#include "codec/inter/mc/mc.h"
#include "codec/inter/me/me.h"
#if defined(AV2ENC_PRIMITIVES)
#include "primitives/kernels.h"
#endif
#include "codec/inter/merge/merge.h"
#include "codec/inter/morph/morph_common.h"
#include "codec/inter/mv_stack/ref_map.h"
#include "codec/inter/temporal/tip.h"
#include "codec/inter/mv_precision/mv_precision.h"
#include "codec/intra/ibc/core/ibc.h"
#include "codec/intra/ibc/stream/ibc_stack.h"
#include "codec/intra/ibc/stream/still_ibc_stream.h"
#include "codec/intra/ibp/ibp.h"
#include "codec/intra/palette/core/palette.h"
#include "codec/intra/palette/stream/palette_stream.h"
#include "primitives/kernels.h"
#include "codec/inter/drl_reorder/video_stack_adapter.h"
#include "codec/inter/temporal/motion_field.h"
#include "codec/inter/temporal/ref_mv_bank.h"
#include "codec/inter/temporal/opfl.h"
#include "codec/inter/temporal/refinemv.h"
#include "codec/inter/temporal/temporal_predict.h"
#include "codec/inter/tip/tip_build.h"
#include "pipeline/grain/grain_stream.h"
#if defined(AV2ENC_VIDEO_FRAMES)
#include "codec/inter/multiref/compound_predict.h"
#endif
#include "codec/inter/advanced/compound.h"
#if defined(AV2ENC_VIDEO_FRAMES)
#include "codec/inter/bframes/bframes.h"
#include "codec/inter/tlayers/tlayers.h"
#endif
#include "codec/inter/gme/gme.h"
// §7.22 storage needs the model plan and the per-8x8 warp projection even when
// the GME block-decision path is compiled out (the standalone video_gate
// build), so this header is unconditional.
#include "pipeline/video/video_gme.h"
#if defined(AV2ENC_VIDEO_GME)
#include "pipeline/video/video_warp.h"
#endif
#include "pipeline/video/post/tiled_frame.h"
#include "pipeline/video/post/deblock_grid.h"
#include "core/parallel/parallel_for.h"
#if defined(AV2ENC_VIDEO_FORMATS)
#include "pipeline/video/hbd/video_hbd.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace av2enc::video {
namespace vf_detail {

using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionNode;
using partition::PartitionType;
namespace drl = inter::drl;
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
namespace pe = partition_ext;
#endif
namespace temporal = inter::temporal;
namespace tip = inter::tip;
namespace sdpp = av2enc::sdp;
namespace ibcs = still::ibc_stream;
namespace mrl_stream = still::mrl_stream;
namespace txsec = ::av2enc::txsec;
namespace morph = inter::morph;
namespace pals = still::palette_stream;
namespace txpart = txpartition;

// AV2 RefFrame labels used by the merge layer's MI grid.  mc::kRefLast is an
// internal MC label (1); §6.19.7.9 numbers LAST_FRAME = 0 and
// INTRA_FRAME = 8.
constexpr std::int8_t kRefLastAv2 = 0;
constexpr std::int8_t kRefIntraAv2 = 8;

// Rate-aware partition probe (--video-partition-rd).  The leaf cost is
//     distortion + lambda * estimated_bits,
// where distortion/bits come from one NEWMV prediction and one quantised luma
// residual.  lambda = scale * step^2 keeps the trade-off quantiser-scaled; the
// default scale (VideoOptions::partition_probe_lambda, 1.1e-5) was calibrated
// at the project's 240p30 / 100 kbps point (QP 190, step 9216 -> lambda ~ 934).
// The probe never changes tree legality, so every emitted partition is exactly
// what the shared solver and `partition::validate_frame` already accept.  It is
// bounded to the first split level: a superblock leaf against its
// HORZ/VERT/SPLIT children.
constexpr double kPartitionProbeSkipBits = 8.0;
constexpr double kPartitionProbeModeBits = 8.0;

// AV2ENC_RD_DRIVER (default off): the sequential key-frame driver.  It
// replaces the frame-wide variance partition tree on the key picture with a
// per-superblock decision that measures every candidate layout in the
// reconstruction the layout itself builds.  The gate is env-only (no CLI) and
// leaves every existing stream byte-identical when unset.
namespace rd_driver {

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

// Process-wide CLI flag set by the InterFrameEncoder constructor from
// VideoOptions::rd_driver (--video-rd-driver / --nightly-b).
inline std::atomic<bool>& cli_flag() {
  static std::atomic<bool> value{false};
  return value;
}

inline bool enabled() {
  static const bool env = [] {
    const char* env_var = std::getenv("AV2ENC_RD_DRIVER");
    return !parse_off(env_var);
  }();
  // --video-rd-driver / --nightly-b sets the process-wide CLI flag from the
  // encoder constructor; the env form stays for the raw gates.
  return env || cli_flag().load();
}

inline double env_positive(const char* name, double fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || !std::isfinite(parsed) || parsed <= 0.0) return fallback;
  return parsed;
}

// Candidate uniform leaf sizes (px).  The reference tree is always measured as
// well, so a target that loses cannot change the layout from the historical
// solver when the gate is on.  AV2ENC_RD_DRIVER_TARGETS is a comma list.
// Deepest uniform leaf size the prune may consider.  The expanded reference
// tree is split to this size and then collapsed node by node, so the search
// covers every uniform layout and every mixture in between.  The default 4
// keeps the full legal key tree (4x4/8x8/16x16/32x32/64x64); there is no
// 16x16 floor.  AV2ENC_RD_DRIVER_MIN=8/16 is the coarser ablation.
inline int prune_target() {
  static const int value = [] {
    const char* env = std::getenv("AV2ENC_RD_DRIVER_MIN");
    if (env == nullptr || *env == '\0') return 4;
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || parsed < 4 || parsed > 64) return 4;
    return static_cast<int>(parsed);
  }();
  return value;
}

// Set while a candidate layout is measured.  The diagnostic counters in the
// leaf coder (TX-plan histogram, UV-level counters) only count committed
// leaves, so a measured candidate cannot inflate the reported plan counts.
inline bool& measuring_flag() {
  // Per-thread: the joint-plan batch measures candidate layouts on parallel
  // trial workers, so a process-global flag races (TSan
  // video_frame_decision.cpp:8216).
  static thread_local bool value = false;
  return value;
}
inline bool measuring() { return measuring_flag(); }

// Driver counters, printed once at process exit when the gate is on.
struct Counters {
  long long superblocks = 0;
  long long layouts = 0;
  long long layout_failures = 0;
  long long measured_leaves = 0;
  long long fallback_reference = 0;
  double chosen_cost = 0.0;

  ~Counters() {
    if (!enabled()) return;
    std::fprintf(stderr,
                 "rd-driver: sbs=%lld layouts=%lld failed=%lld leaves=%lld "
                 "fallback=%lld chosen_cost=%.1f\n",
                 superblocks, layouts, layout_failures, measured_leaves,
                 fallback_reference, chosen_cost);
  }
};

inline Counters& counters() {
  static Counters value;
  return value;
}

}  // namespace rd_driver

// Diagnosis-only final-tree dump (AV2ENC_PART_TREE=<path>).  It walks the tree
// the encoder will actually code — not the solver trace — so every emitted node
// is guaranteed to exist in the stream.  Unset means no file is opened and the
// decisions are untouched.
inline FILE* part_tree_file() {
  static FILE* file = nullptr;
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    const char* path = std::getenv("AV2ENC_PART_TREE");
    if (path != nullptr && path[0] != '\0') file = std::fopen(path, "w");
  }
  return file;
}

inline void dump_partition_tree(const partition::PartitionNode& node, FILE* file) {
  std::fprintf(file,
               "tree r=%d c=%d w=%d h=%d depth=%d hc=%d co=%d chosen=%s\n",
               node.mi_row, node.mi_col, partition::block_width(node.size),
               partition::block_height(node.size), node.depth,
               node.has_chroma ? 1 : 0, node.chroma_offset ? 1 : 0,
               partition::partition_name(node.partition));
  for (const auto& child : node.children) {
    dump_partition_tree(child, file);
  }
}

#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
inline void dump_partition_tree_ext(const pe::PartitionNode& node, FILE* file) {
  std::fprintf(file,
               "tree r=%d c=%d w=%d h=%d depth=%d hc=%d co=%d chosen=%s\n",
               node.mi_row, node.mi_col, partition::block_width(node.size),
               partition::block_height(node.size), node.depth,
               node.has_chroma ? 1 : 0, node.chroma_offset ? 1 : 0,
               pe::partition_name(node.partition));
  for (const auto& child : node.children) {
    dump_partition_tree_ext(child, file);
  }
}
#endif

// Per-leaf decision dump (AV2ENC_BLOCKDUMP=<path>): diagnosis for the
// block-wavefront work (docs/wavefront-plan.md §6.2 stage 0).  It records one
// line per decided leaf, so a stream change can be localised to "leaf 431's
// drl_index moved" instead of "the md5 differs".  Unset means one predictable
// branch per leaf: no allocation, no formatting, no file.  Unlike the tree
// dump above, records are buffered during the walk and printed in raster leaf
// order after the frame's decisions are final, so the text does not depend on
// the order in which a future wavefront completes superblocks.
inline bool block_dump_enabled() {
  static const bool enabled = []() {
    const char* path = std::getenv("AV2ENC_BLOCKDUMP");
    return path != nullptr && path[0] != '\0';
  }();
  return enabled;
}

inline FILE* block_dump_file() {
  static FILE* file = nullptr;
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    if (block_dump_enabled()) {
      file = std::fopen(std::getenv("AV2ENC_BLOCKDUMP"), "w");
    }
  }
  return file;
}

// One decided leaf.  `eob`/`levels_digest` are the quantised coefficients as
// handed to the writer, so a coefficient change is visible even when the
// reconstructed pixels happen to match.
struct LeafDumpRecord {
  int mi_row = 0;
  int mi_col = 0;
  int bsize = 0;
  int partition = 0;
  int is_inter = 0;
  int mode = 0;  // inter: inter_mode; intra: y_mode
  int ref = 0;
  int drl_index = 0;
  int mv_x = 0;
  int mv_y = 0;
  int skip = 0;
  std::uint32_t eob[3] = {0, 0, 0};
  std::uint64_t levels_digest = 0;
  std::uint32_t palette_index = 0;
  int palette_size = 0;
  int palette_direction = 0;
  std::array<std::uint8_t, 8> palette_colors{};
};

// FNV-1a over the quantised levels in writer order.  Cheap, order-sensitive
// and locale-free; it only has to detect a difference, not describe it.
inline std::uint64_t hash_levels(const std::vector<std::int32_t>& levels,
                          std::size_t begin, std::size_t end) {
  std::uint64_t hash = 1469598103934665603ull;
  for (std::size_t i = begin; i < end; ++i) {
    const std::uint32_t value = static_cast<std::uint32_t>(levels[i]);
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= (value >> (8 * byte)) & 0xFFu;
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

namespace grain = ::av2enc::filters::grain;

// §5.14 film grain: estimate/plan one picture and insert the model OBU before
// the coded-frame OBU of its temporal unit (§7.3.3/.4, §7.3.8.8).
inline grain::GrainChromaIdc grain_chroma_idc_of(fmt::ChromaFormat chroma) {
  switch (chroma) {
    case fmt::ChromaFormat::k400:
      return grain::GrainChromaIdc::k400;
    case fmt::ChromaFormat::k422:
      return grain::GrainChromaIdc::k422;
    case fmt::ChromaFormat::k444:
      return grain::GrainChromaIdc::k444;
    case fmt::ChromaFormat::k420:
    default:
      return grain::GrainChromaIdc::k420;
  }
}

inline grain::GrainFrameView make_grain_view(const VideoInputFrame& frame) {
  grain::GrainFrameView view;
  // Prefer the packed uint16_t planes when they are present: the format cores
  // carry them even at 8-bit.
  const bool packed16 = !frame.y16.empty();
  const int sub_x = fmt::subsampling_x(frame.format.chroma);
  const bool has_chroma = fmt::has_chroma(frame.format.chroma);
  if (packed16) {
    view.y16 = frame.y16.data();
    if (has_chroma) {
      view.u16 = frame.u16.data();
      view.v16 = frame.v16.data();
    }
    view.bit_depth = frame.format.depth == fmt::BitDepth::k10 ? 10 : 8;
  } else {
    view.y = frame.y.data();
    if (has_chroma) {
      view.u = frame.u.data();
      view.v = frame.v.data();
    }
    view.bit_depth = 8;
  }
  view.width = static_cast<int>(frame.width);
  view.height = static_cast<int>(frame.height);
  view.y_stride = static_cast<int>(frame.width);
  view.c_stride = static_cast<int>(frame.width >> sub_x);
  return view;
}

inline bool plan_grain_frame(const VideoInputFrame& frame, std::uint32_t index,
                      grain::GrainStreamPlanner& planner,
                      grain::GrainFramePlan& plan, std::string& error,
                      int tlayer_id = 0, int mlayer_id = 0) {
  return planner.PlanFrame(make_grain_view(frame), index, tlayer_id, mlayer_id,
                           &plan, error);
}

inline bool insert_grain_obu(const grain::GrainFramePlan& plan,
                      std::vector<std::uint8_t>& unit, std::string& error) {
  if (!plan.emit_model) return true;
  std::string grain_error;
  const std::vector<std::uint8_t> obu =
      grain::MakeFilmGrainObu(plan.obu, grain_error);
  if (obu.empty()) {
    error = "film grain OBU: " + grain_error;
    return false;
  }
  std::vector<std::uint8_t> with_grain;
  if (!grain::InsertFilmGrainObu(unit, obu, &with_grain, grain_error)) {
    error = "film grain OBU: " + grain_error;
    return false;
  }
  unit.swap(with_grain);
  return true;
}

struct Plane {
  int width = 0;
  int height = 0;
  int stride = 0;
  std::vector<std::uint8_t> samples;

  void resize(int w, int h, std::uint8_t value) {
    width = w;
    height = h;
    stride = w;
    samples.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h),
                   value);
  }
  std::uint8_t& at(int x, int y) {
    return samples[static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(stride) +
                   static_cast<std::size_t>(x)];
  }
  std::uint8_t at(int x, int y) const {
    return samples[static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(stride) +
                   static_cast<std::size_t>(x)];
  }
};

struct TxPlan {
  int tx_size = 0;
  int adj_w = 0;
  int adj_h = 0;
  int eob = 0;
  std::vector<std::int32_t> levels;  // adjusted-size coefficient array
  // §5.20.7.27 TCQ: the adjusted-size Quant[] values (base levels mapped by
  // Quant = 2*base - q0) that the §7.14.4 dequant consumes.  Empty when TCQ
  // did not run on this plan.
  std::vector<std::int32_t> dequant_levels;
  std::vector<std::uint8_t> recon;   // final coded-size samples
  // §5.20.8.2 intra IST decision (--video-ist).  0 = no secondary transform.
  int tx_type = 0;       // txv::TxType of the single (non-searched) block
  int sec_tx_type = 0;
  int stx_set = 0;

  // Opt-in luma TX search (--video-tx-search).  When `searched` is true the
  // per-part list below is authoritative: `parts[i].levels` are in §5.20.7.30
  // scan order and `rate_bits` is the rate the device module also reports.
  bool searched = false;
  int partition = 0;
  double rate_bits = 0.0;
  struct Part {
    int x = 0;
    int y = 0;
    int coded_w = 0;
    int coded_h = 0;
    int tx_size = 0;
    int tx_type = 0;
    int mi_row = 0;
    int mi_col = 0;
    int eob = 0;
    std::vector<std::int32_t> levels;  // scan order
    int sec_tx_type = 0;
    int stx_set = 0;
  };
  std::vector<Part> parts;
};

struct InterCandidate {
  bool valid = false;
  bool compound = false;
  // GLOBALMV with an accepted non-identity model and the warped predictor.
  bool use_warp = false;
  int mode = syntax::kInterGlobalMv;
  int drl_index = 0;
  int ref_index = 0;
  inter::mc::Mv mv{};
  inter::mc::Mv mvd{};
  std::int64_t sse = 0;
  // §5.20.7.16 masked compound parameters (ignored for AVERAGE).
  int compound_type = 2;  // 0 = WEDGE, 1 = DIFFWTD, 2 = AVERAGE
  int wedge_index = 0;
  int wedge_sign = 0;
  int mask_type = 0;
  // §5.20.7.6 / §7.13.3.14 CWP + implicit-mask decision (NEAR_NEARMV only).
  int cwp_index = 0;
  int cwp_weight = 8;  // CWP_EQUAL
  bool use_imp_mask = false;
  // This candidate's MV/DRL index came from a §7.12.2.7/.8 temporal entry
  // (instrumentation for the gate; never coded).
  bool from_temporal = false;
  // This candidate came from the §7.12.2.21 reference-MV bank.
  bool from_refmvbank = false;
  // NEAR_NEARMV / optflow / refine-MV compound carrier.
  int compound_mode = 3;  // 0 = NEAR_NEARMV, 3 = GLOBAL_GLOBALMV
  // §5.20.7.6 is_joint: JOINT_NEWMV (compound_mode_non_joint is not coded).
  bool compound_joint = false;
  inter::mc::Mv mv1{};
  int drl_index1 = 0;
  bool use_optflow = false;
  bool use_refinemv = false;
  // §5.20.7.10 compound reference pair.  The historical carrier is (0,1);
  // --video-same-ref-compound also allows (0,0), where `same_ref` is set so
  // the writer codes compound_mode_same_refs instead of is_joint.
  int ref_index1 = 1;
  bool same_ref = false;
  // §5.20.7.6 MV-resolution decisions for a NEWMV candidate.
  bool use_amvd = false;
  int block_mv_precision = -1;  // me2::kMvPrecision*, -1 = frame precision
  bool mvd_sign_derived = false;
};

// One decoded picture buffer slot.  The planes are MI-aligned (mi_cols*4 by
// mi_rows*4 luma) because MC clips taps to the padded reference extent.
struct RefSlot {
  bool valid = false;
  bool is_key = false;
  std::uint32_t order_hint = 0;
  Plane plane[3];
};

inline int relative_dist(std::uint32_t a, std::uint32_t b, int bits) {
  const int modulus = 1 << bits;
  int diff = static_cast<int>((a - b) & static_cast<std::uint32_t>(modulus - 1));
  if (diff >= (modulus >> 1)) diff -= modulus;
  return diff;
}

// Round a 1/8-pel MV to the signalled frame precision: 1/8, 1/4, 1/2 or one
// pel (3, only with seq_force_integer_mv).
inline inter::mc::Mv quantize_mv(inter::mc::Mv mv, int precision) {
  const int quantum =
      precision == 1 ? 2 : (precision == 2 ? 4 : (precision == 3 ? 8 : 1));
  if (quantum == 1) return mv;
  const auto round_component = [quantum](int value) {
    const int magnitude = value < 0 ? -value : value;
    const int rounded = ((magnitude + quantum / 2) / quantum) * quantum;
    return value < 0 ? -rounded : rounded;
  };
  return inter::mc::Mv{round_component(mv.y), round_component(mv.x)};
}

inline int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}
inline int clip1(int value) { return clip3(0, 255, value); }

// §7.20.1 LrTxSkip input for one coded leaf.  The decoder sets a 4x4 cell when
// its luma transform block has eob == 0 (or the leaf is skipped); a
// TX_MODE_SELECT leaf therefore needs its per-transform eob map, not the
// coding leaf's first transform.  `block->mi_row/mi_col` must already be set
// (including any tile offset).
inline void fill_filter_block_tx(const syntax::BlockSyntax& leaf,
                                 post::VideoLoopFilterBlock* block) {
  block->luma_eob_zero = leaf.eob[0] == 0;
  block->tx_count = 0;
  if (leaf.tx_part_count == 0) return;
  const int count = std::min<int>(leaf.tx_part_count, 4);
  block->tx_count = static_cast<std::uint8_t>(count);
  for (int i = 0; i < count; ++i) {
    post::VideoLoopFilterBlock::LumaTx& tx = block->tx[i];
    tx.mi_row = static_cast<std::int16_t>(block->mi_row +
                                          leaf.tx_part_mi_row[i]);
    tx.mi_col = static_cast<std::int16_t>(block->mi_col +
                                          leaf.tx_part_mi_col[i]);
    tx.w4 = static_cast<std::uint8_t>(std::max(
        1, txpartition::tx_width(leaf.tx_part_tx_size[i]) / 4));
    tx.h4 = static_cast<std::uint8_t>(std::max(
        1, txpartition::tx_height(leaf.tx_part_tx_size[i]) / 4));
    tx.eob_zero = leaf.tx_part_eob[i] == 0 ? 1 : 0;
  }
}

// §7.7 dependency maps for the implicit reference mapping.  The M2 sequence
// headers either omit TLayerDependencyMap (default `ref_tlayer <=
// obu_tlayer_id`) or code it explicitly; MLayerDependencyMap is the default
// `ref_mlayer <= obu_mlayer_id`, and the M2 writers are single-mlayer.
inline inter::mv_stack_full::DependencyMaps implicit_dependency_maps(
    const VideoConfig& config, int tlayer_id) {
  inter::mv_stack_full::DependencyMaps deps;
  deps.mlayer_mask = ~std::uint64_t{0};
  const int layer = clip3(0, 3, tlayer_id);
  if (!config.tlayer_dependency_present_flag) {
    deps.tlayer_mask = (std::uint64_t{1} << (layer + 1)) - 1;
  } else {
    std::uint64_t mask = 0;
    for (int ref = 0; ref <= 3; ++ref) {
      if (config.tlayer_dependency[layer][ref]) {
        mask |= std::uint64_t{1} << ref;
      }
    }
    deps.tlayer_mask = mask;
  }
  return deps;
}

// Runs the §7.7 derivation over the §7.23 bank the encoder mirrors.  `width`
// and `height` are the current frame's size, which §7.7 checks with checkRes=1
// (both passes are identical while frame_size_override_flag = 0).
inline bool derive_implicit_fields(const inter::mv_stack_full::RefSlotBank& bank,
                            int order_hint, int order_hint_bits, int width,
                            int height, int tlayer_id,
                            const inter::mv_stack_full::DependencyMaps& deps,
                            inter::mv_stack_full::ImplicitRefMap* out,
                            std::string& error) {
  inter::mv_stack_full::RefMapParams params;
  params.num_ref_frames = bank.num_ref_frames;
  params.active_num_ref_frames =
      std::min(inter::mv_stack_full::kRefsPerFrameConst, bank.num_ref_frames);
  params.order_hint = order_hint;
  params.order_hint_bits = order_hint_bits;
  params.obu_mlayer_id = 0;
  params.obu_tlayer_id = tlayer_id;
  params.deps = deps;
  params.allowed_frames = ~std::uint32_t{0};
  params.is_bridge = false;
  params.check_res = true;
  params.frame_width = width;
  params.frame_height = height;
  params.slots = bank.slots;
  return inter::mv_stack_full::derive_implicit_ref_map(params, out, error);
}

// --video-trace line for the derived map.  The gate greps it to prove the
// stream actually uses the §7.7 mapping (variable NumTotalRefs).
inline void trace_implicit_map(const char* path, std::size_t frame, int num_refs,
                        const std::vector<int>& slots) {
  std::cerr << "  implicit ref map " << path << " frame=" << frame
            << " num_total_refs=" << num_refs << " slots=";
  for (int r = 0; r < num_refs; ++r) {
    std::cerr << (r == 0 ? "" : ",") << slots[static_cast<std::size_t>(r)];
  }
  std::cerr << "\n";
}

inline int source_sample(const VideoInputFrame& frame, int plane, int x, int y) {
  const int w = plane == 0 ? static_cast<int>(frame.width)
                           : static_cast<int>((frame.width + 1U) / 2U);
  const int h = plane == 0 ? static_cast<int>(frame.height)
                           : static_cast<int>((frame.height + 1U) / 2U);
  const int cx = clip3(0, w - 1, x);
  const int cy = clip3(0, h - 1, y);
  const std::span<const std::uint8_t> data =
      plane == 0 ? frame.y : (plane == 1 ? frame.u : frame.v);
  const int stride = w;
  return data[static_cast<std::size_t>(cy) * static_cast<std::size_t>(stride) +
               static_cast<std::size_t>(cx)];
}

// Edge-replicated 8-bit 4:2:0 source picture at the MI-aligned coded size.
// The §7.18-§7.20 decision stages read the source over the padded plane, so a
// visible size that is not a multiple of 8 needs the clamp on every row/column.
// An aligned frame keeps pointing at the caller's planes (no copy).
struct PaddedSourcePlanes {
  std::vector<std::uint8_t> storage[3];
  const std::uint8_t* data[3] = {nullptr, nullptr, nullptr};
  int stride[3] = {0, 0, 0};
};

inline void pad_source_420(const VideoInputFrame& frame, int mi_cols, int mi_rows,
                    PaddedSourcePlanes* out) {
  const int vis_w = static_cast<int>(frame.width);
  const int vis_h = static_cast<int>(frame.height);
  const int vis_cw = vis_w / 2;
  const int vis_ch = vis_h / 2;
  const int pad_w = mi_cols * 4;
  const int pad_h = mi_rows * 4;
  const int pad_cw = mi_cols * 2;
  const int pad_ch = mi_rows * 2;
  const std::uint8_t* src[3] = {frame.y.data(), frame.u.data(),
                                frame.v.data()};
  const int vis_plane_w[3] = {vis_w, vis_cw, vis_cw};
  const int vis_plane_h[3] = {vis_h, vis_ch, vis_ch};
  const int pad_plane_w[3] = {pad_w, pad_cw, pad_cw};
  const int pad_plane_h[3] = {pad_h, pad_ch, pad_ch};
  for (int plane = 0; plane < 3; ++plane) {
    out->storage[plane].clear();
    if (pad_plane_w[plane] == vis_plane_w[plane] &&
        pad_plane_h[plane] == vis_plane_h[plane]) {
      out->data[plane] = src[plane];
      out->stride[plane] = vis_plane_w[plane];
      continue;
    }
    const int w = pad_plane_w[plane];
    const int h = pad_plane_h[plane];
    const int vw = vis_plane_w[plane];
    const int vh = vis_plane_h[plane];
    out->storage[plane].assign(static_cast<std::size_t>(w) *
                                   static_cast<std::size_t>(h), 0);
    std::uint8_t* dst = out->storage[plane].data();
    for (int y = 0; y < h; ++y) {
      const int sy = std::min(y, vh - 1);
      const std::uint8_t* row =
          src[plane] + static_cast<std::size_t>(sy) * static_cast<std::size_t>(vw);
      std::uint8_t* dst_row = dst + static_cast<std::size_t>(y) *
                                        static_cast<std::size_t>(w);
      std::memcpy(dst_row, row, static_cast<std::size_t>(vw));
      for (int x = vw; x < w; ++x) dst_row[x] = row[vw - 1];
    }
    out->data[plane] = dst;
    out->stride[plane] = w;
  }
}

// Luma PSNR (dB) of a stored reference plane against the source picture it was
// coded from, over the visible width x height.  --video-drift-refresh uses it
// as its accumulated-error trigger: the flat P chain refreshes its only
// reference slot, so the stored plane *is* the reference of the next frame.
inline double reference_luma_psnr(const Plane& reference, const VideoInputFrame& source,
                           int width, int height) {
  if (reference.width < width || reference.height < height ||
      source.y.size() < static_cast<std::size_t>(width) *
                            static_cast<std::size_t>(height)) {
    return 99.0;
  }
  std::int64_t sse = 0;
  for (int y = 0; y < height; ++y) {
    const std::uint8_t* ref_row =
        reference.samples.data() + static_cast<std::size_t>(y) *
                                       static_cast<std::size_t>(reference.stride);
    const std::uint8_t* src_row =
        source.y.data() + static_cast<std::size_t>(y) *
                              static_cast<std::size_t>(width);
    for (int x = 0; x < width; ++x) {
      const int d = static_cast<int>(src_row[x]) - static_cast<int>(ref_row[x]);
      sse += static_cast<std::int64_t>(d) * d;
    }
  }
  if (sse <= 0) return 99.0;
  const double mse =
      static_cast<double>(sse) / (static_cast<double>(width) * height);
  return 10.0 * std::log10(255.0 * 255.0 / mse);
}

inline std::int64_t sse_against_source(const VideoInputFrame& frame, int plane, int x0,
                                int y0, int w, int h,
                                const std::vector<std::uint8_t>& samples,
                                int rows = -1) {
  const int count = rows >= 0 && rows < h ? rows : h;
  std::int64_t sse = 0;
  for (int y = 0; y < count; ++y) {
    for (int x = 0; x < w; ++x) {
      const int src = source_sample(frame, plane, x0 + x, y0 + y);
      const int p = samples[static_cast<std::size_t>(y) *
                               static_cast<std::size_t>(w) +
                           static_cast<std::size_t>(x)];
      const int d = src - p;
      sse += static_cast<std::int64_t>(d) * d;
    }
  }
  return sse;
}

inline bool intra_dc_predict(const Plane& recon, const VideoInputFrame& frame, int plane,
                      int x0, int y0, int w, int h, int max_x, int max_y,
                      bool have_left, bool have_above, bool enable_ibp,
                      std::vector<std::uint8_t>& out) {
  intra::NeighborRequest request;
  request.recon = intra::PlaneView{recon.samples.data(), recon.stride};
  request.x = x0;
  request.y = y0;
  request.w = w;
  request.h = h;
  request.max_x = max_x;
  request.max_y = max_y;
  request.have_left = have_left;
  request.have_above = have_above;
  request.num4_above_right = 0;
  request.num4_below_left = 0;
  request.color_plane = plane;
  out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  intra::IntraStatus status = intra::IntraStatus::kInvalidArgument;
  if (enable_ibp) {
    // §7.13.2.12: enable_ibp changes DC via the Ibp_Weights post-filter for
    // every non-4x4 block.  The decoder applies it whenever YMode is DC_PRED,
    // so the encoder's DC predictor must use the same path.
    ibp::PredictParams params;
    params.mode = intra::Mode::kDc;
    params.angle_delta = 0;
    params.enable_ibp = true;
    params.enable_edge_filter = false;
    status = ibp::predict_block(request, 8, params, out.data(),
                                static_cast<std::size_t>(w));
  } else {
    intra::PredictParams params;
    params.mode = intra::Mode::kDc;
    params.angle_delta = 0;
    params.enable_edge_filter = false;
    status = intra::predict_block(request, 8, params, out.data(),
                                  static_cast<std::size_t>(w));
  }
  (void)frame;
  return status == intra::IntraStatus::kOk;
}

// Single-mode luma prediction at an explicit origin with a caller-supplied
// §5.20.9.1 availability.  This is the per-TX-part predictor: the intra
// TX-partition search predicts every part of the coding block with the block's
// signalled (mode, angle_delta, MRL), but reads the neighbours and the
// BlockDecoded availability at the *transform* block origin.  Both predictors
// come from the shared intra component, so the mode/IBP/edge-filter semantics
// are the same ones `intra_search::search_luma` uses.
inline bool intra_luma_part_predict(const Plane& recon,
                             const intra_search::Availability& avail, int x0,
                             int y0, int w, int h, int y_mode,
                             int angle_delta, bool enable_ibp,
                             bool above_smooth, bool left_smooth,
                             std::vector<std::uint8_t>& out) {
  intra::NeighborRequest request;
  request.recon = intra::PlaneView{recon.samples.data(), recon.stride};
  request.x = x0;
  request.y = y0;
  request.w = w;
  request.h = h;
  request.max_x = avail.max_x;
  request.max_y = avail.max_y;
  request.have_left = avail.have_left != 0;
  request.have_above = avail.have_above != 0;
  request.num4_above_right = avail.num4_above_right;
  request.num4_below_left = avail.num4_below_left;
  request.color_plane = 0;
  intra::IntraEdges edges;
  if (intra::prepare_edges(request, 8, edges) != intra::IntraStatus::kOk) {
    return false;
  }
  out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  if (enable_ibp) {
    ibp::PredictParams params;
    params.mode = static_cast<intra::Mode>(y_mode);
    params.angle_delta = angle_delta;
    params.enable_ibp = true;
    // §5.4.8: the sequence header always signals enable_intra_edge_filter = 1
    // in this tree, and the full intra search predicts with it on.
    params.enable_edge_filter = true;
    params.above_neighbor_is_smooth = above_smooth;
    params.left_neighbor_is_smooth = left_smooth;
    return ibp::predict(edges, params, out.data(),
                        static_cast<std::size_t>(w)) ==
           intra::IntraStatus::kOk;
  }
  intra::PredictParams params;
  params.mode = static_cast<intra::Mode>(y_mode);
  params.angle_delta = angle_delta;
  params.enable_edge_filter = true;
  params.above_neighbor_is_smooth = above_smooth;
  params.left_neighbor_is_smooth = left_smooth;
  return intra::predict_intra(edges, params, out.data(),
                              static_cast<std::size_t>(w)) ==
         intra::IntraStatus::kOk;
}

// §7.14.2 qlookup(base_q_idx) for 8-bit streams.  528 at QP 90, the value the
// transform component bakes into TxParams.
inline int frame_quant_step(const VideoOptions& options) {
  const int q = options.base_q_idx;
  if (q <= 0 || q > 255) return txv::kQuantStep;
  return rc::kQlookupQ0To255[q];
}

#if defined(AV2ENC_VIDEO_DELTAQ)
// Number of active segment ids for the opt-in planner (default 2, which is the
// historical variance split).
inline int video_num_segments(const VideoOptions& options) {
  return std::clamp(options.num_segments, 1, deltaq::kMaxSegmentsSpec);
}
inline int video_skip_segment(const VideoOptions& options) {
  if (!options.enable_seg_skip) return -1;
  return video_num_segments(options) - 1;
}
inline int video_globalmv_segment(const VideoOptions& options) {
  if (!options.enable_seg_globalmv) return -1;
  return options.enable_seg_skip ? video_num_segments(options) - 2
                                 : video_num_segments(options) - 1;
}

// Builds the §5.18.7.1 feature table for one frame.  `primary_ref_is_none`
// selects the derived temporal_update value (key frames and a P frame whose
// only reference is a key frame infer 0 and code no bit).
inline bool make_video_segmentation(const VideoOptions& options,
                             bool primary_ref_is_none, bool frame_is_intra,
                             rc::SegmentationParams& seg, std::string& error) {
  seg = rc::SegmentationParams{};
  if (!options.enable_segmentation && !options.per_segment_q) return true;
  seg.enabled = true;
  seg.update_map = true;
  seg.temporal_update =
      options.enable_seg_temporal && !primary_ref_is_none;
  const int num_segments = video_num_segments(options);
  const int skip_seg = video_skip_segment(options);
  const int gm_seg = video_globalmv_segment(options);
  if (skip_seg == gm_seg && skip_seg >= 0) {
    error = "segmentation: SEG_LVL_SKIP and SEG_LVL_GLOBALMV need distinct "
            "segment ids (use --video-segments >= 3)";
    return false;
  }
  // Segment 0 is the high-variance baseline with no ALT_Q.  Segments 1..N-1
  // carry the requested ALT_Q (0 when only the map is exercised, which is the
  // historical --video-segmentation behaviour).
  const int delta = options.per_segment_q ? options.seg_q_delta : 0;
  for (int id = 1; id < num_segments; ++id) {
    // On an inter frame a feature segment keeps ALT_Q = 0 so an edge leaf that
    // cannot be forced inter can fall back to segment 0 with the same §7.14.2
    // qindex.  The intra key frame has no inter blocks, so it keeps ALT_Q and
    // the P-frame table enables the feature instead (the table is per frame).
    if (!frame_is_intra && (id == skip_seg || id == gm_seg)) continue;
    if (!rc::set_segment_alt_q(seg, id, delta, error)) {
      error = "segmentation: " + error;
      return false;
    }
  }
  // §6.19.4.1 conformance: if is_inter == 0 then seg_feature_active(
  // SEG_LVL_SKIP) must be 0.  An intra frame has no inter blocks, so the SKIP
  // feature cannot be active on a key frame.  The table is per frame, so the
  // P frame headers enable it while the key frame keeps only ALT_Q/GLOBALMV.
  if (skip_seg >= 0 && !frame_is_intra) {
    seg.segments[static_cast<std::size_t>(skip_seg)]
        .enabled[static_cast<std::size_t>(rc::kSegLvlSkip)] = true;
    seg.segments[static_cast<std::size_t>(skip_seg)]
        .data[static_cast<std::size_t>(rc::kSegLvlSkip)] = 0;
  }
  if (gm_seg >= 0) {
    seg.segments[static_cast<std::size_t>(gm_seg)]
        .enabled[static_cast<std::size_t>(rc::kSegLvlGlobalMv)] = true;
    seg.segments[static_cast<std::size_t>(gm_seg)]
        .data[static_cast<std::size_t>(rc::kSegLvlGlobalMv)] = 0;
  }
  return true;
}

// The §5.4.4 sequence table is the key-frame form of the planner: identical
// features to every other frame unless a P-only SKIP/GLOBALMV feature is
// enabled.  `num_segments` is the same MaxSegments the header writer uses.
inline bool same_segmentation_table(const rc::SegmentationParams& a,
                             const rc::SegmentationParams& b,
                             int num_segments) noexcept {
  for (int i = 0; i < num_segments; ++i) {
    const rc::SegmentFeatures& fa = a.segments[static_cast<std::size_t>(i)];
    const rc::SegmentFeatures& fb = b.segments[static_cast<std::size_t>(i)];
    for (int j = 0; j < 3; ++j) {
      if (fa.enabled[static_cast<std::size_t>(j)] !=
              fb.enabled[static_cast<std::size_t>(j)] ||
          fa.data[static_cast<std::size_t>(j)] !=
              fb.data[static_cast<std::size_t>(j)]) {
        return false;
      }
    }
  }
  return true;
}

// §5.4.8 sequence delta-Q fields.  Every writer that builds its own
// VideoConfig (the B / temporal-layer / TIP / refinement sequence writers, the
// InterFrameEncoder writer) must copy these from the VideoOptions the frame
// codec itself resolves its quantizer from.  Omitting them makes the encoder
// quantise chroma at the offset step while the stream carries the default
// group, which decodes at a different step and is not a valid stream.
inline void apply_video_sequence_delta_q(VideoConfig& config,
                                         const VideoOptions& options) {
  config.seq_delta_q = options.seq_delta_q;
}

// §5.18.7.1 segmentation_params() / §5.18.7.8 delta_q_params() for one frame.
// The feature table is coded per frame; the per-frame map lives in the tile
// payload.  Validates the plan before the header writer runs (the writer has no
// error channel).
inline bool build_video_frame_qp(const VideoOptions& options, int base_q_idx,
                          bool primary_ref_is_none, bool frame_is_intra,
                          VideoFrameHeader& header, std::string& error) {
  header.qp = VideoFrameQp{};
  header.qp.primary_ref_is_none = primary_ref_is_none;
  header.qp.frame_delta_q = options.frame_delta_q;
  header.qp.enable_ext_seg = options.enable_ext_seg;
  header.qp.seq_seg_info_present = options.seq_seg_info_present;
  header.qp.seq_allow_seg_info_change = options.seq_allow_seg_info_change;
  if (!options.enable_segmentation && !options.per_segment_q &&
      !options.enable_delta_q && !options.enable_uv_delta_q &&
      !options.enable_aq) {
    return true;
  }
  if (options.enable_segmentation || options.per_segment_q) {
    if (!make_video_segmentation(options, primary_ref_is_none, frame_is_intra,
                                 header.qp.segmentation, error)) {
      return false;
    }
  }
  header.qp.delta_q.present = options.enable_delta_q || options.enable_aq;
  header.qp.delta_q.res =
      (options.enable_delta_q || options.enable_aq) ? options.delta_q_res : 0;
  if (options.seq_seg_info_present && header.qp.segmentation.enabled) {
    // §5.18.7.1: reuse_seg_info = 1 matches the decoded SeqFeature* table, so
    // the frame table must equal the sequence table feature for feature.
    rc::SegmentationParams seq_table;
    if (!make_video_segmentation(options, /*primary_ref_is_none=*/true,
                                 /*frame_is_intra=*/true, seq_table, error)) {
      return false;
    }
    header.qp.reuse_seg_info = same_segmentation_table(
        header.qp.segmentation, seq_table, video_num_segments(options));
  }
  deltaq::FrameQpHeader qp_header;
  qp_header.primary_ref_is_none = primary_ref_is_none;
  qp_header.enable_ext_seg = options.enable_ext_seg;
  qp_header.using_qmatrix = options.enable_qm;
  deltaq::FrameSegmentation derived;
  if (!deltaq::derive_frame_segmentation(header.qp.segmentation,
                                         header.qp.delta_q, base_q_idx,
                                         qp_header, derived, error)) {
    error = "segmentation: " + error;
    return false;
  }
  if (derived.has_lossless_segment) {
    error = "segmentation: a segment qindex reaches 0 (lossless); the video "
            "transform path does not implement lossless";
    return false;
  }
  return true;
}
#endif  // AV2ENC_VIDEO_DELTAQ

// §7.14.2 get_dc_quant()/get_ac_quant() for one plane.  The default path has
// dc == ac == qlookup(base_q_idx), which keeps the historical scalar
// quantiser; --video-uv-delta-q gives each chroma plane its own pair.
struct PlaneQuantStep {
  int dc = txv::kQuantStep;
  int ac = txv::kQuantStep;
  bool operator==(const PlaneQuantStep& other) const noexcept {
    return dc == other.dc && ac == other.ac;
  }
};

inline bool compute_tx(const VideoInputFrame& frame, int plane, int x0, int y0, int cw,
                int ch, const std::vector<std::uint8_t>& pred,
                PlaneQuantStep quant, TxPlan& plan, std::string& error,
                const qm::QmFrameState* qm_state = nullptr,
                bool allow_tcq = false, int qctx_slice = 0,
                bool is_inter = false, bool allow_rdoq = false,
                int rdoq_qctx = 0,
                double rdoq_sse_scale = kVideoRdoqSseScale,
                int mask_row = -1) {
  joint_plan::ScopedTimer tx_timer(&joint_plan::profile().tx_plain);
  auto params_opt = txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
  if (!params_opt.has_value()) {
    error = "unsupported transform size";
    return false;
  }
  const bool split_dc_ac = quant.dc != quant.ac;
  if ((quant.ac != txv::kQuantStep || split_dc_ac) && qm_state == nullptr) {
    // With QM the per-coefficient q2 replaces the scalar step; the forward
    // expression below uses the unscaled quant_factor.
    params_opt->quant_factor *=
        static_cast<double>(txv::kQuantStep) / static_cast<double>(quant.ac);
  }
  const txv::TxParams& params = *params_opt;
  plan.tx_size = params.tx_size;
  plan.adj_w = params.w;
  plan.adj_h = params.h;

  const int coded_pels = cw * ch;
  std::vector<std::int16_t> residual_coded(
      static_cast<std::size_t>(coded_pels));
  for (int y = 0; y < ch; ++y) {
    // --video-edge-mask: rows at/after mask_row lie below the visible picture
    // bottom.  They are reconstructed by the decoder but never output and
    // never predicted from (MiRows clamp + reference edge replication), so the
    // source view is the prediction there: zero residual, no levels spent on
    // samples nobody sees.  Encoder-side only; no syntax change.
    const bool hidden = mask_row >= 0 && y >= mask_row;
    for (int x = 0; x < cw; ++x) {
      const int src = source_sample(frame, plane, x0 + x, y0 + y);
      const int p = pred[static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(cw) +
                         static_cast<std::size_t>(x)];
      residual_coded[static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(cw) +
                     static_cast<std::size_t>(x)] =
          static_cast<std::int16_t>(hidden ? 0 : src - p);
    }
  }
  std::vector<std::int16_t> residual_adj(
      static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
  txv::downsample_residual(residual_coded.data(), params,
                           residual_adj.data());
  std::vector<double> coeff(
      static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
  plan.levels.assign(coeff.size(), 0);
  qm::QmDequantParams qp;
  std::vector<std::int32_t> q2;
  if (qm_state != nullptr) {
    // §7.14.4 QM: q2 = Round2(q * m, 5) per coefficient.  The forward target
    // scales the scalar qf by q / q2 so the returned levels reconstruct to
    // the same residual under the QM dequantizer (still_encoder.cpp).
    qp.state = qm_state;
    qp.tx_size = params.tx_size;
    qp.plane = plane;
    qp.tx_type = params.tx_type;
    qp.segment_id = 0;
    qp.dc_quant = quant.dc;
    qp.ac_quant = quant.ac;
    qp.bit_depth = 8;
    q2.assign(coeff.size(), 0);
    if (!qm::build_q2_matrix(qp, q2.data())) {
      error = "qm q2 matrix build failed";
      return false;
    }
    txv::forward_block(residual_adj.data(), params, coeff.data());
    for (std::size_t i = 0; i < coeff.size(); ++i) {
      const double factor = static_cast<double>(params.quant_factor) *
                            static_cast<double>(quant.ac) /
                            static_cast<double>(q2[i]);
      plan.levels[i] = static_cast<std::int32_t>(std::lround(coeff[i] * factor));
    }
  } else if (!split_dc_ac) {
    txv::forward_transform(residual_adj.data(), params, coeff.data(),
                           plan.levels.data());
  } else {
    // §7.14.2 get_dc_quant()/get_ac_quant() with distinct DC and AC steps.
    // params.quant_factor was scaled for quant.ac; adjusted raster index 0 is
    // the DC coefficient.
    txv::forward_block(residual_adj.data(), params, coeff.data());
    const double ac_to_dc =
        static_cast<double>(quant.ac) / static_cast<double>(quant.dc);
    for (std::size_t i = 0; i < coeff.size(); ++i) {
      const double factor =
          static_cast<double>(params.quant_factor) * (i == 0 ? ac_to_dc : 1.0);
      plan.levels[i] = static_cast<std::int32_t>(std::lround(coeff[i] * factor));
    }
  }

  const int area = params.w * params.h;
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  syntax::build_scan(params.w, params.h, scan);

  // Opt-in coefficient-level RDOQ (--video-rdoq).  Luma inter residual only,
  // and only on the scalar quantiser: QM and split DC/AC change the per
  // coefficient step, which the trellis Config cannot express.  The trellis
  // rate is the static AV2 §8.3 model under the inter plane contexts; it
  // returns syntax levels, so the writer, the dequantiser and the decoder see
  // exactly the same coefficient set.  Scalar quantisation stays the fallback.
  const bool use_rdoq = allow_rdoq && plane == 0 && is_inter &&
                        qm_state == nullptr && !split_dc_ac &&
                        params.tx_type <= 9;
  if (use_rdoq) {
    std::vector<std::int32_t> scalar_scan(static_cast<std::size_t>(area));
    std::vector<av2enc::rdoq::Coefficient> input(
        static_cast<std::size_t>(area));
    for (int c = 0; c < area; ++c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      scalar_scan[static_cast<std::size_t>(c)] =
          plan.levels[static_cast<std::size_t>(pos)];
      input[static_cast<std::size_t>(c)].original =
          coeff[static_cast<std::size_t>(pos)];
      input[static_cast<std::size_t>(c)].scalar_level =
          plan.levels[static_cast<std::size_t>(pos)];
      input[static_cast<std::size_t>(c)].level_context = 0;
    }
    av2enc::rdoq::av2::BlockInfo info;
    info.plane = 0;
    info.params = params;
    info.txb_skip_context = 0;
    info.dc_sign_context = 0;
    // §8.3.1: the coefficient CDF slice follows the raw frame base_q_idx, not
    // a segmented SB qindex.
    info.qctx = rdoq_qctx;
    info.is_inter = true;
    av2enc::rdoq::av2::Model model;
    std::string rdoq_error;
    if (!model.init(info, scalar_scan, rdoq_error)) {
      error = "rdoq model init failed: " + rdoq_error;
      return false;
    }
    av2enc::rdoq::Config rdoq_config;
    rdoq_config.quant_step = 1.0 / params.quant_factor;
    rdoq_config.lambda = av2enc::rdo::lambda_sse_from_quant_step(
        rdoq_config.quant_step, rdoq_sse_scale);
    rdoq_config.max_coefficients = area;
    av2enc::rdoq::RdoqResult rdoq_result;
    if (!av2enc::rdoq::rdoq_quantize(input, rdoq_config, model.rate_model(),
                                     rdoq_result, rdoq_error)) {
      error = "rdoq quantize failed: " + rdoq_error;
      return false;
    }
    std::fill(plan.levels.begin(), plan.levels.end(), 0);
    for (int c = 0; c < rdoq_result.eob; ++c) {
      plan.levels[scan[static_cast<std::size_t>(c)]] =
          rdoq_result.levels[static_cast<std::size_t>(c)];
    }
  }

  // §5.20.7.27 trellis-coded quantisation.  Luma-only and TX_CLASS_2D-only;
  // QM changes the per-coefficient quantiser and is rejected with TCQ before
  // this point.  The DP returns the syntax base levels (what coeff_base/br and
  // the read_quant escape code) and the mapped Quant values §7.14.4
  // dequantises after the extra dqDenom shift.
  const bool use_tcq =
      allow_tcq && plane == 0 && params.tx_type <= 9 && qm_state == nullptr;
  if (use_tcq) {
    std::vector<av2enc::tcq::Coefficient> input(
        static_cast<std::size_t>(area));
    std::vector<std::int32_t> reference(static_cast<std::size_t>(area));
    for (int c = 0; c < area; ++c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      const int row = pos / params.w;
      const int col = pos % params.w;
      input[static_cast<std::size_t>(c)].original =
          coeff[static_cast<std::size_t>(pos)];
      input[static_cast<std::size_t>(c)].level_context = 0;
      input[static_cast<std::size_t>(c)].pre_max = (row + col) < 4 ? 8 : 6;
      // Frozen neighbour/escape reference: the scalar level at half the
      // syntax step (the same approximation the still TCQ path uses).
      const std::int32_t scalar = plan.levels[static_cast<std::size_t>(pos)];
      const std::int32_t magnitude = std::abs(scalar);
      const std::int32_t base = (magnitude + 1) / 2;
      reference[static_cast<std::size_t>(c)] = scalar < 0 ? -base : base;
    }
    av2enc::tcq::av2::BlockInfo info;
    info.plane = 0;
    info.params = params;
    info.txb_skip_context = 0;
    info.dc_sign_context = 0;
    info.qctx = qctx_slice;
    info.is_inter = is_inter;
    info.tx_class_2d = true;
    av2enc::tcq::av2::Model model;
    std::string tcq_error;
    if (!model.init(info, reference, tcq_error)) {
      error = "tcq model init failed: " + tcq_error;
      return false;
    }
    av2enc::tcq::Config tcq_config;
    tcq_config.quant_step = 1.0 / params.quant_factor;
    tcq_config.lambda = av2enc::rdo::lambda_sse_from_quant_step(
        tcq_config.quant_step, /*sse_scale=*/0.125);
    tcq_config.max_coefficients = area;
    av2enc::tcq::Result tcq_result;
    if (!av2enc::tcq::quantize(input, tcq_config, model.rate_model(),
                               tcq_result, tcq_error)) {
      error = "tcq quantize failed: " + tcq_error;
      return false;
    }
    std::fill(plan.levels.begin(), plan.levels.end(), 0);
    plan.dequant_levels.assign(static_cast<std::size_t>(area), 0);
    for (int c = 0; c < tcq_result.eob; ++c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      plan.levels[static_cast<std::size_t>(pos)] =
          tcq_result.base_levels[static_cast<std::size_t>(c)];
      plan.dequant_levels[static_cast<std::size_t>(pos)] =
          tcq_result.quant_levels[static_cast<std::size_t>(c)];
    }
  } else {
    plan.dequant_levels.clear();
  }

  plan.eob = 0;
  for (int c = area - 1; c >= 0; --c) {
    if (plan.levels[scan[static_cast<std::size_t>(c)]] != 0) {
      plan.eob = c + 1;
      break;
    }
  }



  std::vector<std::int32_t> dequant(
      static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
  std::vector<std::int16_t> recon_residual(
      static_cast<std::size_t>(coded_pels));
  if (qm_state != nullptr) {
    const int n = params.w * params.h;
    for (int i = 0; i < n; ++i) {
      const std::int32_t qc = plan.levels[static_cast<std::size_t>(i)];
      const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
      const std::int64_t high = (aq * q2[static_cast<std::size_t>(i)]) & 0xFFFFFF;
      const std::int32_t dq =
          txv::round2(static_cast<std::int32_t>(high), txv::kQuantTableBits) /
          params.dq_denom;
      const std::int32_t value = qc < 0 ? -dq : dq;
      dequant[static_cast<std::size_t>(i)] =
          std::max(-(1 << 15), std::min((1 << 15) - 1, value));
    }
    txv::inverse_block(dequant.data(), params, recon_residual.data());
  } else if (!plan.dequant_levels.empty()) {
    // §7.14.4 TCQ: dequantise the mapped Quant[] values with the extra
    // dqDenom shift (`shift += 1` for luma TX_CLASS_2D), then inverse.
    const int n = params.w * params.h;
    for (int i = 0; i < n; ++i) {
      const std::int32_t qc =
          plan.dequant_levels[static_cast<std::size_t>(i)];
      const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
      const std::int64_t high = (aq * quant.ac) & 0xFFFFFF;
      const std::int32_t dq =
          txv::round2(static_cast<std::int32_t>(high), txv::kQuantTableBits) /
          (params.dq_denom << 1);
      const std::int32_t value = qc < 0 ? -dq : dq;
      dequant[static_cast<std::size_t>(i)] =
          std::max(-(1 << 15), std::min((1 << 15) - 1, value));
    }
    txv::inverse_block(dequant.data(), params, recon_residual.data());
  } else if (quant.dc == txv::kQuantStep && quant.ac == txv::kQuantStep) {
    txv::inverse_transform(plan.levels.data(), params, dequant.data(),
                           recon_residual.data());
  } else {
    // §7.14.4 dequant with the plane's §7.14.2 DC/AC steps, then the shared
    // inverse.  Adjusted raster index 0 is DC.
    const int n = params.w * params.h;
    for (int i = 0; i < n; ++i) {
      const std::int32_t qc = plan.levels[static_cast<std::size_t>(i)];
      const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
      const int step = i == 0 ? quant.dc : quant.ac;
      const std::int64_t high = (aq * step) & 0xFFFFFF;
      const std::int32_t dq =
          txv::round2(static_cast<std::int32_t>(high), txv::kQuantTableBits) /
          params.dq_denom;
      const std::int32_t value = qc < 0 ? -dq : dq;
      dequant[static_cast<std::size_t>(i)] =
          std::max(-(1 << 15), std::min((1 << 15) - 1, value));
    }
    txv::inverse_block(dequant.data(), params, recon_residual.data());
  }
  plan.recon.resize(static_cast<std::size_t>(coded_pels));
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      const int p = pred[static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(cw) +
                         static_cast<std::size_t>(x)];
      const int r = recon_residual[static_cast<std::size_t>(y) *
                                       static_cast<std::size_t>(cw) +
                                   static_cast<std::size_t>(x)];
      plan.recon[static_cast<std::size_t>(y) *
                     static_cast<std::size_t>(cw) +
                 static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>(clip1(p + r));
    }
  }
  return true;
}

// Rough coefficient rate estimate used only for the skip/residual decision.
// The actual cost is measured by the entropy coder; this heuristic is not
// normative and never affects bitstream correctness.
inline double estimate_tx_bits(const TxPlan& plan, bool is_inter_luma) {
  if (plan.eob == 0) return 1.0;
  double bits = 8.0;  // EOB symbol + extra bits
  if (is_inter_luma) bits += 4.0;  // inter TX_TYPE syntax
  int nonzero = 0;
  for (int c = 0; c < plan.eob; ++c) {
    const int level = plan.levels[static_cast<std::size_t>(c)];
    if (level != 0) ++nonzero;
  }
  bits += static_cast<double>(nonzero) * 2.5 + plan.eob * 0.5;
  return bits;
}

// estimate_tx_bits() for a searched plan: the same heuristic applied per
// transform leaf (the plan's scan-order levels are already the exact nonzero
// set, so counting the first `eob` entries is the nonzero count).
inline double search_plan_estimate_bits(const TxPlan& plan, bool is_inter_luma) {
  double bits = 0.0;
  for (const TxPlan::Part& part : plan.parts) {
    if (part.eob == 0) {
      bits += 1.0;
      continue;
    }
    int nonzero = 0;
    for (int c = 0; c < part.eob; ++c) {
      if (part.levels[static_cast<std::size_t>(c)] != 0) ++nonzero;
    }
    bits += 8.0 + (is_inter_luma ? 4.0 : 0.0) +
            static_cast<double>(nonzero) * 2.5 + part.eob * 0.5;
  }
  return bits;
}

// Rate used by the skip/residual and intra/inter decisions.  The TX search's
// own cost model only selects the plan; the decision keeps the historical
// estimate_tx_bits heuristic (per leaf) so enabling the search does not
// re-price every residual against skip.
inline double luma_plan_bits(const TxPlan& plan, bool is_inter_luma) {
  return plan.searched ? search_plan_estimate_bits(plan, is_inter_luma)
                       : estimate_tx_bits(plan, is_inter_luma);
}

inline bool luma_plan_all_zero(const TxPlan& plan) {
  if (!plan.searched) return plan.eob == 0;
  for (const TxPlan::Part& part : plan.parts) {
    if (part.eob != 0) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// R09 (AV2ENC_RATE_HYBRID_S) coefficient model
//
// `decision_coeff_bits()` is the per-leaf coefficient price the skip/residual
// and intra/inter decisions charge.  Gate off it is byte-for-byte the
// historical `luma_plan_bits()` / `estimate_tx_bits()` pair.  Gate on, the
// EOB/TX-type overhead stays and the per-nonzero + per-EOB constants are
// replaced by the analytic two-sided-Laplacian rate of the coded scan prefix.
// ---------------------------------------------------------------------------
inline double laplacian_raster_prefix(std::span<const std::int32_t> levels,
                                      int w, int h, int eob) {
  if (eob <= 0 || w <= 0 || h <= 0) return 0.0;
  int nonzero = 0;
  std::int64_t sum_abs = 0;
  int x = 0;
  int y = 0;
  for (int c = 0; c < eob; ++c) {
    const std::size_t index =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
        static_cast<std::size_t>(x);
    if (index < levels.size()) {
      const std::int64_t v = levels[index];
      if (v != 0) {
        ++nonzero;
        sum_abs += v < 0 ? -v : v;
      }
    }
    ++x;
    --y;
    if (y < 0 || x >= w) {
      ++x;
      const int s = std::min(x, h - 1 - y);
      x -= s;
      y += s;
    }
  }
  if (nonzero == 0) return 0.0;
  const double b = static_cast<double>(sum_abs) / nonzero;
  const double p = std::exp(-1.0 / b);
  const double mag =
      static_cast<double>(nonzero) *
      std::log2(std::exp(1.0) * (1.0 + p) / (1.0 - p));
  return mag + rate_hybrid::kCoeffPositionBits * static_cast<double>(eob);
}

inline double laplacian_plan_bits(const TxPlan& plan, bool is_inter_luma) {
  if (!plan.searched) {
    if (plan.eob == 0) return 1.0;
    return rate_hybrid::kCoeffPartOverhead + (is_inter_luma ? 4.0 : 0.0) +
           laplacian_raster_prefix(plan.levels, plan.adj_w, plan.adj_h,
                                   plan.eob);
  }
  double bits = 0.0;
  for (const TxPlan::Part& part : plan.parts) {
    if (part.eob == 0) {
      bits += 1.0;
      continue;
    }
    bits += rate_hybrid::kCoeffPartOverhead + (is_inter_luma ? 4.0 : 0.0) +
            rate_hybrid::laplacian_bits(part.levels, part.eob);
  }
  return bits;
}

inline double decision_coeff_bits(const TxPlan& plan, bool is_inter_luma) {
  if (rate_hybrid::enabled() && rate_hybrid::coeff_enabled()) {
    return laplacian_plan_bits(plan, is_inter_luma);
  }
  // AV2ENC_DECISION_EXACT_RATE: the searched plan already carries the §8.3
  // coefficient model's own bit count (the TX search priced it), so the leaf
  // decision can charge that instead of the historical heuristic.
  if (decision_rate::exact() && plan.searched) return plan.rate_bits;
  return plan.searched ? search_plan_estimate_bits(plan, is_inter_luma)
                       : estimate_tx_bits(plan, is_inter_luma);
}

// ---------------------------------------------------------------------------
// Leaf RD cost (--video-leaf-rd)
//
// The historical intra/inter decision compared coded SSE with a fixed
// `inter_rd_lambda` (4.0) that was meaningless against a QP 142 block SSE, and
// it charged neither the intra mode/MRL syntax nor the inter mode/MV syntax.
// One calibrated cost replaces it:
//
//   J = SSE + lambda(qindex) * bits,   lambda = c * sqrt((qstep/8) / ref)
//
// qstep is the §7.14.2 qlookup() value and qstep/8 the per-sample luma
// reconstruction step the M1 transform path produces, so the lambda follows
// the frame quantiser instead of being a per-content constant.  The bit terms
// mirror the real codeword shape (symbol arities from the syntax writer) and
// stay in true bits, so the same lambda is valid for every comparison.
// ---------------------------------------------------------------------------

// c = 125 anchored at QP 104.  The measured optimum is nearly flat over QP
// 104..145, i.e. it grows far slower than the high-rate Q^2 relation; a sqrt()
// fit stays within 0.01 JOD of the best constant on both the 100 kbps point
// and the 50..80 kbps range, while still following the frame quantiser.  The
// reference lambda was re-calibrated to 125 when the nightly aggregate gained
// the partition probe and luma RDOQ (those tools already price coefficients,
// so the leaf needs less rate pressure); see docs/video-nightly.md.
constexpr double kLeafRdLambdaRefStep = 98.0;  // qlookup(104) / 8

inline double qp_luma_step(int qindex) {
  const int q = std::clamp(qindex, 0, 255);
  return static_cast<double>(rc::kQlookupQ0To255[q]) / 8.0;
}

// AV2ENC_RD_DRIVER_RATE_FIT (default off): rate-dependent driver tree price.
// docs/rd-key-partition-floor.md 0.4 identifies the flat kKeyTxLambdaFit
// (exponent 0 => one SSE-per-bit price at every qindex) as the root cause of
// the key driver's 4x4 over-expansion at low rate and prescribes the fix it
// never measured: "the price must be rate-dependent (~2.0 at qindex 82, ~2048
// at qindex 184)".  This gate implements exactly those two anchors:
//   price(q) = 2.0 * (step(q) / step(82))^e,  e = ln(1024) / ln(step(184)/step(82))
// so price(82) = 2.0 and price(184) = 2048.  When on, the fit IS the whole
// tree price: AV2ENC_RD_KEY_TX_SCALE and AV2ENC_RD_DRIVER_LAMBDA (the flat
// price's own knobs) are not applied.  Unset or 0 reproduces the flat product
// bit-for-bit, so every pin stays byte-identical.
inline bool driver_rate_fit_enabled() {
  static const bool value = [] {
    const char* v = std::getenv("AV2ENC_RD_DRIVER_RATE_FIT");
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
  }();
  return value;
}

inline double driver_rate_fit_price(int qindex) {
  const double step82 = qp_luma_step(82);
  const double step184 = qp_luma_step(184);
  const double e = std::log(2048.0 / 2.0) / std::log(step184 / step82);
  return 2.0 * std::pow(qp_luma_step(qindex) / step82, e);
}

inline double leaf_rd_lambda(int qindex, const VideoOptions& options) {
  const double step = qp_luma_step(qindex);
  if (!options.qp_lambda_scale || options.leaf_rd_lambda_explicit) {
    return options.leaf_rd_lambda_ref *
           std::sqrt(step / kLeafRdLambdaRefStep);
  }
  return lambda_from_fit(step, kLeafLambdaFit);
}

// --video-tx-lambda: constant in the legacy mode, power law in the QP-scaled
// mode.  An explicit flag keeps the exact value at every QP.
inline double tx_rd_lambda(const VideoOptions& options, int qindex) {
  if (!options.qp_lambda_scale || options.tx_rd_lambda_explicit) {
    return options.tx_rd_lambda;
  }
  return lambda_from_fit(qp_luma_step(qindex), kTxLambdaFit);
}

// --video-key-tx-lambda: 0 still inherits the (possibly scaled) inter weight
// in the legacy mode.  With QP scaling on, the key site uses its own measured
// fit instead: inheriting the inter constant is the defect being fixed.
inline double key_tx_rd_lambda(const VideoOptions& options, int qindex) {
  if (options.qp_lambda_scale && !options.key_tx_lambda_explicit) {
    return lambda_from_fit(qp_luma_step(qindex), kKeyTxLambdaFit);
  }
  if (options.key_tx_lambda > 0.0) return options.key_tx_lambda;
  return tx_rd_lambda(options, qindex);
}

// The coefficient trellis stores a scale in lambda = scale * quant_step^2.
inline double rdoq_sse_scale_at(const VideoOptions& options, int qindex) {
  if (!options.qp_lambda_scale || options.rdoq_sse_scale_explicit) {
    return options.rdoq_sse_scale;
  }
  return lambda_from_fit(qp_luma_step(qindex), kRdoqSseScaleFit);
}

// The partition probe stores a scale in lambda = scale * step^2 with the full
// qlookup() step.
inline double partition_probe_scale_at(const VideoOptions& options, int qindex) {
  if (!options.qp_lambda_scale || options.partition_lambda_explicit) {
    return options.partition_probe_lambda;
  }
  const double step = rc::quant_step_from_qindex(qindex);
  return lambda_from_fit(step, kPartitionProbeFit);
}

// Variance + syntax-bit fallback price (SSE per estimated bit).
inline double partition_model_price_at(const VideoOptions& options, int qindex) {
  if (!options.qp_lambda_scale || options.partition_lambda_explicit) {
    return 700.0;
  }
  const double step = rc::quant_step_from_qindex(qindex);
  return lambda_from_fit(step, kPartitionModelFit);
}

inline bool y_mode_directional(std::uint8_t mode) { return mode >= 1 && mode <= 8; }

// §5.20.5.5 YMode: 4-ary set symbol + 8-ary index + optional 6-ary offset
// escape.  A directional mode sits deep in the neighbour-derived list (or
// escapes), so it costs more than DC/SMOOTH/PAETH.  §5.20.5.5 MRL adds a
// 4-ary index and a 2-ary secondary flag; DIP adds its 2-ary flag.
//
// AV2ENC_RD_MODE_BITS_DIR / _NONDIR / _ANGLE (defaults 5.0 / 3.0 / 0.5): the
// three constants above, made reachable so the estimate can be calibrated
// against the tile's measured cost.  `AV2ENC_SYMBOL_COST` reports the intra
// mode family at ~1.97 bits/symbol and effectively rate-independent
// (27,894 bits at QP186 against 27,431 at QP196), so the default 5.0/3.0
// over-charges a mode by ~2x and, more importantly, prices the directional
// choice 2 bits dearer than the measured spread supports.  Defaults reproduce
// the historical estimate exactly, so every pin is byte-identical.
inline double mode_bits_dir() {
  static const double v = [] {
    const char* s = std::getenv("AV2ENC_RD_MODE_BITS_DIR");
    if (s == nullptr || *s == '\0') return 5.0;
    const double p = std::atof(s);
    return p > 0.0 ? p : 5.0;
  }();
  return v;
}

inline double mode_bits_nondir() {
  static const double v = [] {
    const char* s = std::getenv("AV2ENC_RD_MODE_BITS_NONDIR");
    if (s == nullptr || *s == '\0') return 3.0;
    const double p = std::atof(s);
    return p > 0.0 ? p : 3.0;
  }();
  return v;
}

inline double mode_bits_angle() {
  static const double v = [] {
    const char* s = std::getenv("AV2ENC_RD_MODE_BITS_ANGLE");
    if (s == nullptr || *s == '\0') return 0.5;
    const double p = std::atof(s);
    return p >= 0.0 ? p : 0.5;
  }();
  return v;
}

inline double intra_luma_syntax_bits(int mode, int angle_delta, int mrl_index,
                              bool mrl_sec, bool dip) {
  const bool dir = y_mode_directional(static_cast<std::uint8_t>(mode));
  double bits = dir ? mode_bits_dir() : mode_bits_nondir();
  if (dir && angle_delta != 0) {
    bits += mode_bits_angle();
  }
  if (mrl_index > 0) bits += 2.5 + (mrl_sec ? 1.0 : 0.0);
  if (dip) bits += 1.0;
  return bits;
}

// Chroma: 8-ary UV mode index (3-bit escape), the CfL flag and, when CfL is
// selected, cfl_index plus the sign/alpha symbols.  CCTX adds its joint type.
inline double intra_chroma_syntax_bits(bool cfl, bool cctx) {
  double bits = 2.0;  // UV mode index ~2 bits for DC
  if (cfl) bits += 1.0 + 1.0 + 5.0;
  if (cctx) bits += 1.5;
  return bits;
}

// §5.20.7.6 YMode = NEARMV + single_mode (3-ary), then the DRL index
// (2-ary chain) or the shell-coded MVD.  A shell code pays one 2-ary set
// symbol, one 8-ary class symbol and `floor_log2(shell)` offset bits, shared
// by both components.
inline double inter_syntax_bits(int mode, int drl_index, int mvd_y, int mvd_x,
                         bool amvd) {
  double bits = 1.7;
  if (mode == syntax::kInterNearMv) {
    bits += 1.0 + 0.7 * static_cast<double>(std::max(0, drl_index));
  } else if (mode == syntax::kInterNewMv) {
    if (amvd) {
      bits += 2.0;
    } else {
      const int shell = std::abs(mvd_y) + std::abs(mvd_x);
      if (shell > 0) {
        bits += 4.0 + std::floor(std::log2(static_cast<double>(shell)));
      }
    }
  }
  return bits;
}

// Copies one leaf's luma coefficient layout into the syntax node and appends
// the scan-order levels.  The no-search path is the historical single DCT; the
// search path fills the TX_MODE_SELECT per-part list.
inline void write_luma_plan(syntax::BlockSyntax& entry, const TxPlan& plan,
                     std::vector<std::int32_t>& levels) {
  entry.tx_type[0] = 0;
  entry.sec_tx_type[0] = static_cast<std::uint8_t>(plan.sec_tx_type);
  entry.most_probable_stx_set[0] =
      static_cast<std::uint8_t>(plan.stx_set);
  if (!plan.searched) {
    entry.tx_size[0] = static_cast<std::uint8_t>(plan.tx_size);
    entry.eob[0] = static_cast<std::uint32_t>(plan.eob);
    entry.tx_partition = 0;
    entry.tx_part_count = 0;
    const int area = plan.adj_w * plan.adj_h;
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    syntax::build_scan(plan.adj_w, plan.adj_h, scan);
    for (int c = 0; c < plan.eob; ++c) {
      levels.push_back(plan.levels[scan[static_cast<std::size_t>(c)]]);
    }
    return;
  }
  entry.tx_partition = static_cast<std::uint8_t>(plan.partition);
  entry.tx_part_count = static_cast<std::uint8_t>(plan.parts.size());
  const int count = static_cast<int>(plan.parts.size());
  for (int i = 0; i < count; ++i) {
    const TxPlan::Part& part = plan.parts[static_cast<std::size_t>(i)];
    entry.tx_part_tx_size[i] = static_cast<std::uint8_t>(part.tx_size);
    entry.tx_part_tx_type[i] = static_cast<std::uint8_t>(part.tx_type);
    entry.tx_part_sec_tx_type[i] =
        static_cast<std::uint8_t>(part.sec_tx_type);
    entry.tx_part_stx_set[i] = static_cast<std::uint8_t>(part.stx_set);
    entry.tx_part_mi_row[i] = static_cast<std::int16_t>(part.mi_row);
    entry.tx_part_mi_col[i] = static_cast<std::int16_t>(part.mi_col);
    entry.tx_part_eob[i] = static_cast<std::uint32_t>(part.eob);
    entry.tx_part_levels_offset[i] =
        static_cast<std::uint32_t>(levels.size());
    for (int c = 0; c < part.eob; ++c) {
      levels.push_back(part.levels[static_cast<std::size_t>(c)]);
    }
  }
  entry.tx_size[0] = entry.tx_part_tx_size[0];
  entry.tx_type[0] = entry.tx_part_tx_type[0];
  entry.eob[0] = entry.tx_part_eob[0];
}

// AV2ENC_TX_DETAIL_SEARCH_TAIL (default off): frequency-aware TX on the
// P-frame tail.  `docs/luma-audit.md` §4.3 puts 69% of the eff-perc f44-59
// luma error in coded residual leaves, and §5 puts the useful target at the
// 2-4 px band rather than the high-frequency rings.  This gate therefore
// (a) lets the partition tree reach a 16x16 coding block where the 2-4 px
// source energy is high (the only surface whose legal §5.20.6.3 SPLIT is four
// 8x8 transforms), and (b) prices the 2-4 px error band in the luma TX search
// there, so an 8x8 plan can win on RD instead of being forced.  Nothing runs
// unless the env var is set; tiled encodes, the key picture and the TIP second
// pass are inert, and the ABI/CUDA reference path never sees the state.
namespace tx_detail {

struct Config {
  bool enabled = false;
  double min_share = 0.5;
};

inline const Config& config() {
  static const Config cfg = [] {
    Config out;
    const char* on = std::getenv("AV2ENC_TX_DETAIL_SEARCH_TAIL");
    out.enabled = on != nullptr && on[0] != '\0' && std::strcmp(on, "0") != 0;
    if (const char* share = std::getenv("AV2ENC_TX_DETAIL_MIN")) {
      const double value = std::atof(share);
      if (value > 0.0) out.min_share = value;
    }
    return out;
  }();
  return cfg;
}

// One input picture ordinal, assigned once per `decide()` call.  The 60-frame
// clip's tail is frames 33-59 inclusive.
inline int& picture_ordinal() {
  static int ordinal = -1;
  return ordinal;
}

inline void begin_picture() { ++picture_ordinal(); }

// Display-order ordinal for the chained B carrier (set by the encoder from
// VideoOptions::display_ordinal; -1 keeps the flat chain's coding order).
inline int& display_ordinal() {
  static int value = -1;
  return value;
}

inline int arming_ordinal() {
  return display_ordinal() >= 0 ? display_ordinal() : picture_ordinal();
}

inline bool tail_armed() {
  const Config& cfg = config();
  const int ordinal = arming_ordinal();
  return cfg.enabled && ordinal >= 33 && ordinal <= 59;
}

// AV2ENC_RD_HEAD_PART (default off): the band-gated partition descent
// (AV2ENC_TX_DETAIL_SEARCH_TAIL) on the head pictures instead of the tail.
// The head's tree is otherwise clamped to the measured probe's first level
// (`partition_probe_depth` = 1), which is why the head codes nearly every
// leaf at 64x64 / 64x32 (`tools/lane-pdriver/head_stats.py`).
inline bool head_part_enabled() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_RD_HEAD_PART");
    if (env == nullptr || env[0] == '\0') return false;
    return std::strcmp(env, "0") != 0;
  }();
  return value;
}

inline int head_part_last() { return 32; }

// Armed pictures for the *band-gated descent*: the tail window under the
// historical gate, or pictures 1-32 under AV2ENC_RD_HEAD_PART.
inline bool band_descent_armed() {
  const int ordinal = arming_ordinal();
  if (config().enabled && ordinal >= 33 && ordinal <= 59) return true;
  if (head_part_enabled()) return ordinal >= 1 && ordinal <= head_part_last();
  return false;
}

// 2-4 px source band share of one luma block:
//   sum((mean_2x2 - mean_5x5)^2) / sum((src - mean_5x5)^2)
// over the block's visible rect.  The 2x2 mean is the aligned 2x2 cell that
// contains the sample and the 5x5 mean is centered on it; both are clipped to
// the rect.  A flat block has no band energy and returns 0.
inline double source_band_share(const VideoInputFrame& frame, int x0, int y0,
                                int w, int h) {
  if (w <= 0 || h <= 0) return 0.0;
  double band = 0.0;
  double all = 0.0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int cell_x = x & ~1;
      const int cell_y = y & ~1;
      int sum2 = 0;
      int used2 = 0;
      for (int dy = 0; dy < 2; ++dy) {
        const int py = cell_y + dy;
        if (py >= h) break;
        for (int dx = 0; dx < 2; ++dx) {
          const int px = cell_x + dx;
          if (px >= w) break;
          sum2 += source_sample(frame, 0, x0 + px, y0 + py);
          ++used2;
        }
      }
      int sum5 = 0;
      int used5 = 0;
      for (int dy = -2; dy <= 2; ++dy) {
        const int py = y + dy;
        if (py < 0 || py >= h) continue;
        for (int dx = -2; dx <= 2; ++dx) {
          const int px = x + dx;
          if (px < 0 || px >= w) continue;
          sum5 += source_sample(frame, 0, x0 + px, y0 + py);
          ++used5;
        }
      }
      const double mean2 = static_cast<double>(sum2) / used2;
      const double mean5 = static_cast<double>(sum5) / used5;
      const double delta = mean2 - mean5;
      const double residual =
          static_cast<double>(source_sample(frame, 0, x0 + x, y0 + y)) -
          mean5;
      band += delta * delta;
      all += residual * residual;
    }
  }
  if (all <= 1e-9) return 0.0;
  return band / all;
}

inline bool band_high(const VideoInputFrame& frame, int x0, int y0, int w,
                      int h) {
  const Config& cfg = config();
  return cfg.enabled &&
         source_band_share(frame, x0, y0, w, h) >= cfg.min_share;
}

}  // namespace tx_detail

// Opt-in luma TX size/type/partition search.  Builds the coded-size residual
// from the source and prediction and runs the shared CPU reference
// (`pipeline/video/video_tx_search.h`), which mirrors the device module under
// `src/cuda/pipeline/video/tx/`.
inline bool search_luma_tx(const VideoInputFrame& frame, int x0, int y0, int cw,
                    int ch, const std::vector<std::uint8_t>& pred, int qindex,
                    txtsearch::Mode mode, double lambda, bool allow_idtx,
                    int intra_mapped_mode, bool allow_tcq, int qctx_slice,
                    bool enable_inter_ddt, bool search_partition,
                    bool context_rate, bool trace, bool allow_rdoq,
                    int rdoq_qctx, double rdoq_sse_scale, TxPlan& plan,
                    std::string& error, int mask_row = -1,
                    const std::uint16_t* distortion_weight = nullptr,
                    const still::qctx::CoeffCdfSlice* cdf_slice = nullptr,
                    double lowpass_gain = 0.0,
                    double lowpass2_gain = 0.0,
                    bool intra_rdoq = false,
                    double model_rate_scale = 0.4,
                    int txb_skip_ctx_live = 0,
                    int dc_sign_ctx_live = 0,
                    bool use_live_ctx = false,
                    // §7.14.2 DC quantiser delta in qindex units.  Non-zero
                    // only when the frame resolved a DC step that differs from
                    // AC (§5.4.8 base_y_dc_delta_q); 0 keeps the historical
                    // behaviour and every pinned stream.
                    int delta_dc = 0) {
  joint_plan::ScopedTimer tx_timer(&joint_plan::profile().tx_search);
  std::vector<std::int16_t> residual(
      static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
  for (int y = 0; y < ch; ++y) {
    // Same --video-edge-mask contract as compute_tx(): hidden bottom rows get
    // a zero residual so the TX search does not spend bits on them.
    const bool hidden = mask_row >= 0 && y >= mask_row;
    for (int x = 0; x < cw; ++x) {
      const int src = source_sample(frame, 0, x0 + x, y0 + y);
      const int p =
          pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
               static_cast<std::size_t>(x)];
      residual[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
               static_cast<std::size_t>(x)] =
          static_cast<std::int16_t>(hidden ? 0 : src - p);
    }
  }
  txtsearch::Options options;
  options.lambda = lambda;
  options.allow_idtx = allow_idtx;
  options.intra_mapped_mode = intra_mapped_mode;
  options.tcq = allow_tcq;
  options.tcq_qctx = qctx_slice;
  options.rdoq = allow_rdoq;
  options.rdoq_qctx = rdoq_qctx;
  options.rdoq_sse_scale = rdoq_sse_scale;
  options.enable_inter_ddt = enable_inter_ddt;
  // The exact §8.3.2 model is the measured preset behavior
  // (--video-tx-context-rate).  Standalone --video-tx-search and the DDT-only
  // search keep the historical heuristic rate, byte-identical to the CUDA
  // device TX module.
  options.use_context_rate = context_rate;
  // AV2ENC_PIPE_PERC: the same source-driven per-sample weights the leaf
  // objective applies.  Null on every historical call, so the CUDA reference
  // contract and all pinned streams keep the SSE arithmetic.
  options.distortion_weight = distortion_weight;
  options.cdf_slice = cdf_slice;
  options.lowpass_gain = lowpass_gain;
  options.lowpass2_gain = lowpass2_gain;
  options.intra_rdoq = intra_rdoq;
  options.model_rate_scale = model_rate_scale;
  // AV2ENC_TX_LIVE_CTX: price with this block's real §8.3.2 contexts.
  options.txb_skip_ctx_live = txb_skip_ctx_live;
  options.dc_sign_ctx_live = dc_sign_ctx_live;
  options.use_live_ctx = use_live_ctx;
  // AV2ENC_TX_DETAIL_SEARCH_TAIL: inter luma only, and only where an 8x8 TX
  // leaf is legal (a coding block at most 16x16 on both axes).  The 2-4 px
  // price then lets the legal 16x16 -> four 8x8 SPLIT win on RD where the
  // source carries that band; the plan is never forced.
  if (mode == txtsearch::Mode::kInter && cw <= 16 && ch <= 16 &&
      tx_detail::band_descent_armed() &&
      tx_detail::band_high(frame, x0, y0, cw, ch)) {
    options.detail_band_gain = 4.0;
  }
  // §5.20.7.24 predicts per *transform block*, in decode order, and each part's
  // reconstruction feeds the next part's prediction.  A fixed block prediction
  // is only equivalent when the transform block covers the coding block, so the
  // partition search is inter-only: MC with one MV is position-independent, and
  // the decoder's per-part prediction equals the block prediction at the same
  // coordinates.  Intra keeps TX_PARTITION_NONE and searches the TX type only;
  // the full intra partition search needs the per-part prediction/recon
  // feedback the device module's fixed-predictor contract does not carry.
  // Under TX_MODE_LARGEST the bitstream codes no partition, so the DDT type
  // search must also restrict itself to TX_PARTITION_NONE.
  options.search_partition = search_partition;
  txtsearch::Result result;
  if (!txtsearch::search(residual.data(), cw, ch, mode, qindex, delta_dc, 0,
                         pred.data(), options, &result, error)) {
    return false;
  }
  plan.searched = true;
  plan.partition = result.partition;
  plan.rate_bits = result.rate_bits;
  plan.recon = std::move(result.recon);
  plan.parts.clear();
  plan.parts.reserve(static_cast<std::size_t>(result.count));
  for (int i = 0; i < result.count; ++i) {
    const txtsearch::Leaf& leaf = result.leaves[i];
    TxPlan::Part part;
    part.x = leaf.x;
    part.y = leaf.y;
    part.coded_w = leaf.coded_w;
    part.coded_h = leaf.coded_h;
    part.tx_size = leaf.tx_size;
    part.tx_type = leaf.tx_type;
    part.mi_row = leaf.mi_row;
    part.mi_col = leaf.mi_col;
    part.eob = leaf.eob;
    part.levels = leaf.levels;
    part.sec_tx_type = 0;
    part.stx_set = 0;
    if (trace) {
      std::cerr << "  video tx: leaf " << leaf.coded_w << "x" << leaf.coded_h
                << " type=" << leaf.tx_type << " at " << x0 + leaf.x << ","
                << y0 + leaf.y << "\n";
    }
    if (trace && leaf.tx_type == 9) {
      std::cerr << "  video tx: idtx leaf " << leaf.coded_w << "x"
                << leaf.coded_h << " at " << x0 + leaf.x << ","
                << y0 + leaf.y << "\n";
    }
    if (trace && enable_inter_ddt) {
      const auto tx_params = txv::make_tx_params(
          static_cast<txv::TxSize>(leaf.tx_size),
          static_cast<txv::TxType>(leaf.tx_type));
      if (tx_params.has_value()) {
        av2enc::video::ddt::Context ddt_ctx;
        ddt_ctx.enable_inter_ddt = true;
        ddt_ctx.is_inter = true;
        if (av2enc::video::ddt::plan_has_ddt(*tx_params, ddt_ctx)) {
          std::cerr << "  video tx: ddt leaf " << leaf.coded_w << "x"
                    << leaf.coded_h << " type=" << leaf.tx_type << " at "
                    << x0 + leaf.x << "," << y0 + leaf.y << "\n";
        }
      }
    }
    plan.parts.push_back(std::move(part));
  }
  if (plan.parts.empty()) {
    error = "TX search returned no transform leaves";
    return false;
  }
  plan.tx_size = plan.parts[0].tx_size;
  plan.eob = plan.parts[0].eob;
  return true;
}

// §7.13.2.1/.7/.8 MRL prediction for a directional luma candidate.  The M1
// still encoder's predict_luma_mrl() is the reference; the availability inputs
// come from the shared intra search.
inline bool predict_luma_mrl(const Plane& recon,
                      const intra_search::Availability& avail, int x0, int y0,
                      int w, int h, bool above_smooth, bool left_smooth,
                      int y_mode, int angle_delta, int mrl_index,
                      bool mrl_sec_index, int sb_height,
                      std::vector<std::uint8_t>& out, std::string& error) {
  mrl::Params params;
  if (!mrl_stream::make_predict_params(
          y_mode, angle_delta, mrl_index, mrl_sec_index,
          /*sequence_enable_edge_filter=*/true, above_smooth, left_smooth,
          &params, w, h)) {
    error = "MRL prediction parameters rejected";
    return false;
  }
  mrl::Block block;
  block.recon = intra::PlaneView{recon.samples.data(), recon.stride};
  block.x = x0;
  block.y = y0;
  block.w = w;
  block.h = h;
  block.max_x = avail.max_x;
  block.max_y = avail.max_y;
  block.have_left = avail.have_left != 0;
  block.have_above = avail.have_above != 0;
  block.num4_above_right = avail.num4_above_right;
  block.num4_below_left = avail.num4_below_left;
  block.plane = 0;
  block.sb_height = sb_height;
  out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  if (mrl::predict_block(block, 8, params, out.data(),
                         static_cast<std::size_t>(w)) !=
      mrl::MrlStatus::kOk) {
    error = "mrl::predict_block failed";
    return false;
  }
  return true;
}

// §7.15.3 luma IST decision for one intra leaf.  Mirrors the M1 still
// encoder's compute_tx() block: the projection is encoder-side analysis, and
// the cost is reconstructed SSE + lambda * §9.3 symbol bits.  Only a single
// transform block is considered; with a TX partition the search plan wins.
inline bool refine_luma_ist(const VideoInputFrame& frame, int ax0, int ay0, int cw,
                     int ch, const std::vector<std::uint8_t>& pred,
                     int quant_step, int y_mode, int angle_delta,
                     int mrl_index, TxPlan& plan, std::string& error) {
  int tx_size = plan.tx_size;
  int tx_type = plan.tx_type;
  int coded_w = cw;
  int coded_h = ch;
  int eob = plan.eob;
  std::vector<std::int32_t> full;
  TxPlan::Part* part = nullptr;
  if (plan.searched) {
    if (plan.parts.size() != 1) return true;
    part = &plan.parts[0];
    if (part->coded_w != cw || part->coded_h != ch) return true;
    tx_size = part->tx_size;
    tx_type = part->tx_type;
    eob = part->eob;
    const auto params_opt =
        txv::make_tx_params(static_cast<txv::TxSize>(tx_size),
                            static_cast<txv::TxType>(tx_type));
    if (!params_opt.has_value()) return true;
    const int area = params_opt->w * params_opt->h;
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    syntax::build_scan(params_opt->w, params_opt->h, scan);
    full.assign(static_cast<std::size_t>(area), 0);
    for (int c = 0; c < eob; ++c) {
      full[scan[static_cast<std::size_t>(c)]] =
          part->levels[static_cast<std::size_t>(c)];
    }
    plan.tx_size = tx_size;
  } else {
    full = plan.levels;
  }

  const auto params_opt =
      txv::make_tx_params(static_cast<txv::TxSize>(tx_size),
                          static_cast<txv::TxType>(tx_type));
  if (!params_opt.has_value()) return true;
  const txv::TxParams& params = *params_opt;
  if (params.coded_w != coded_w || params.coded_h != coded_h) return true;
  const int area = params.w * params.h;
  if (static_cast<int>(full.size()) < area) {
    error = "IST: coefficient array is smaller than the transform area";
    return false;
  }
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  syntax::build_scan(params.w, params.h, scan);
  const auto eob_of = [&](const std::vector<std::int32_t>& lv) {
    for (int c = area - 1; c >= 0; --c) {
      if (lv[scan[static_cast<std::size_t>(c)]] != 0) return c + 1;
    }
    return 0;
  };

  txsec::SecondaryTransformConfig ist_cfg;
  ist_cfg.enable_intra_ist = true;
  txsec::SecondarySyntax stx;
  stx.plane_tx_type = static_cast<txv::TxType>(tx_type);
  stx.coded_w = coded_w;
  stx.coded_h = coded_h;
  stx.y_mode = y_mode;
  stx.angle_delta = angle_delta;
  stx.mrl_index = mrl_index;
  stx.is_inter = false;
  stx.lossless = false;
  stx.bit_depth = 8;
  if (eob < 2 || !txsec::secondary_syntax_applicable(0, eob, stx, ist_cfg)) {
    return true;
  }
  txsec::SecondaryCandidate candidates[32];
  const int count =
      txsec::enumerate_secondary_candidates(stx, ist_cfg, eob, candidates, 32);
  if (count <= 0) return true;

  const auto reconstruct = [&](const std::vector<std::int32_t>& lv, int ev,
                               int sec, int set, std::vector<std::int16_t>* rr,
                               bool* ok) -> bool {
    txsec::SecondarySyntax cs = stx;
    cs.sec_tx_type = sec;
    cs.most_probable_stx_set = set;
    if (!txsec::secondary_syntax_valid(0, ev, cs, ist_cfg)) return false;
    std::vector<std::int32_t> dequant(static_cast<std::size_t>(area));
    for (int i = 0; i < area; ++i) {
      const std::int32_t qc = lv[static_cast<std::size_t>(i)];
      const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
      const std::int64_t high = (aq * quant_step) & 0xFFFFFF;
      const std::int32_t dq =
          txv::round2(static_cast<std::int32_t>(high), txv::kQuantTableBits) /
          params.dq_denom;
      dequant[static_cast<std::size_t>(i)] =
          std::max(-(1 << 15), std::min((1 << 15) - 1, qc < 0 ? -dq : dq));
    }
    const txsec::SecondaryStatus status = txsec::apply_inverse_secondary(
        params, 0, ev, cs, ist_cfg, dequant.data(), dequant.size());
    if (status != txsec::SecondaryStatus::kApplied) return false;
    rr->assign(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch), 0);
    txv::inverse_block(dequant.data(), params, rr->data());
    *ok = true;
    return true;
  };
  const auto recon_of = [&](const std::vector<std::int16_t>& rr) {
    std::vector<std::uint8_t> recon(
        static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const int p =
            pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                 static_cast<std::size_t>(x)];
        const int r =
            rr[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
               static_cast<std::size_t>(x)];
        recon[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
              static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>(clip1(p + r));
      }
    }
    return recon;
  };

  double best_cost = static_cast<double>(
      sse_against_source(frame, 0, ax0, ay0, cw, ch, plan.recon));
  std::vector<std::int32_t> best_levels;
  std::vector<std::uint8_t> best_recon;
  int best_eob = eob;
  int best_sec = 0;
  int best_set = 0;
  const double lambda =
      av2enc::rdo::lambda_sse_from_quant_step(1.0 / params.quant_factor, 0.125);
  for (int i = 0; i < count; ++i) {
    const txsec::SecondaryCandidate& candidate = candidates[i];
    std::vector<std::int32_t> lv = full;
    if (!txsec::project_ist_levels(params, stx.plane_tx_type, candidate.kernel,
                                   candidate.sec_tx_type, candidate.transpose,
                                   lv.data(), lv.size())) {
      continue;
    }
    const int ev = eob_of(lv);
    if (ev < 2) continue;
    bool ok = false;
    std::vector<std::int16_t> rr;
    if (!reconstruct(lv, ev, candidate.sec_tx_type,
                     candidate.most_probable_stx_set, &rr, &ok) ||
        !ok) {
      continue;
    }
    std::vector<std::uint8_t> recon = recon_of(rr);
    const double sse = static_cast<double>(
        sse_against_source(frame, 0, ax0, ay0, cw, ch, recon));
    const double bits =
        txsec::sec_tx_type_bits(candidate.sec_tx_type, params.tx_size, false) +
        txsec::most_probable_stx_set_bits(candidate.most_probable_stx_set,
                                          stx.plane_tx_type, coded_w, coded_h);
    const double cost = sse + lambda * bits;
    if (cost < best_cost) {
      best_cost = cost;
      best_levels = std::move(lv);
      best_recon = std::move(recon);
      best_eob = ev;
      best_sec = candidate.sec_tx_type;
      best_set = candidate.most_probable_stx_set;
    }
  }
  if (best_sec == 0) return true;

  if (part != nullptr) {
    part->eob = best_eob;
    part->sec_tx_type = best_sec;
    part->stx_set = best_set;
    part->levels.resize(static_cast<std::size_t>(best_eob));
    for (int c = 0; c < best_eob; ++c) {
      part->levels[static_cast<std::size_t>(c)] =
          best_levels[scan[static_cast<std::size_t>(c)]];
    }
    plan.eob = best_eob;
    plan.tx_type = tx_type;
  } else {
    plan.levels = std::move(best_levels);
    plan.eob = best_eob;
    plan.sec_tx_type = best_sec;
    plan.stx_set = best_set;
  }
  plan.recon = std::move(best_recon);
  return true;
}

inline void write_plane_block(Plane& plane, int x0, int y0, int w, int h,
                       const std::vector<std::uint8_t>& samples) {
  for (int y = 0; y < h; ++y) {
    const int py = y0 + y;
    if (py < 0 || py >= plane.height) continue;
    for (int x = 0; x < w; ++x) {
      const int px = x0 + x;
      if (px < 0 || px >= plane.width) continue;
      plane.at(px, py) =
          samples[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                  static_cast<std::size_t>(x)];
    }
  }
}

// §5.4.8 enable_cctx / §7.14.6.  Chroma U/V are quantised jointly: each of the
// seven rotation types is enumerated, and the exact decoder reconstruction
// (integer §7.14.4 dequant, §7.14.6 rotation, inverse transform) picks the
// winner by SSE + lambda * cctx_type bits.  Mirrors the still path
// (still_encoder.cpp compute_chroma_cctx) without RDOQ.
inline bool compute_chroma_cctx(const VideoInputFrame& frame, int x0, int y0, int cw,
                         int ch, const std::vector<std::uint8_t>& u_pred,
                         const std::vector<std::uint8_t>& v_pred,
                         int quant_step, bool is_inter, TxPlan& u_plan,
                         TxPlan& v_plan, int* cctx_type_out,
                         std::string& error, int mask_row = -1) {
  auto params_opt = txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
  if (!params_opt.has_value()) {
    error = "unsupported chroma transform size for CCTX";
    return false;
  }
  if (quant_step != txv::kQuantStep) {
    params_opt->quant_factor *=
        static_cast<double>(txv::kQuantStep) / static_cast<double>(quant_step);
  }
  const txv::TxParams params = *params_opt;
  const int area = params.w * params.h;
  const int coded_pels = cw * ch;

  const auto residual_adj = [&](int plane,
                                const std::vector<std::uint8_t>& pred) {
    std::vector<std::int16_t> coded(static_cast<std::size_t>(coded_pels));
    for (int y = 0; y < ch; ++y) {
      const bool hidden = mask_row >= 0 && y >= mask_row;
      for (int x = 0; x < cw; ++x) {
        const int src = source_sample(frame, plane, x0 + x, y0 + y);
        const int p =
            pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                 static_cast<std::size_t>(x)];
        coded[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
              static_cast<std::size_t>(x)] =
            static_cast<std::int16_t>(hidden ? 0 : src - p);
      }
    }
    std::vector<std::int16_t> adj(static_cast<std::size_t>(area));
    txv::downsample_residual(coded.data(), params, adj.data());
    return adj;
  };
  const std::vector<std::int16_t> u_adj = residual_adj(1, u_pred);
  const std::vector<std::int16_t> v_adj = residual_adj(2, v_pred);
  std::vector<double> u_coeff(static_cast<std::size_t>(area));
  std::vector<double> v_coeff(static_cast<std::size_t>(area));
  txv::forward_block(u_adj.data(), params, u_coeff.data());
  txv::forward_block(v_adj.data(), params, v_coeff.data());

  av2enc::tcq::CctxConfig config;
  config.enabled = true;
  config.quant_step_u = 1.0 / params.quant_factor;
  config.quant_step_v = config.quant_step_u;
  config.bitdepth = 8;
  std::vector<av2enc::tcq::CctxResult> candidates;
  std::string cctx_error;
  if (!av2enc::tcq::cctx_enumerate_scalar(u_coeff, v_coeff, params.w,
                                          params.h, config, &candidates,
                                          cctx_error)) {
    error = "cctx enumerate failed: " + cctx_error;
    return false;
  }

  // §7.14.4 dequant for one chroma plane (no QM/delta-Q/TCQ shift).
  const auto dequant_plane = [&](const std::vector<std::int32_t>& levels,
                                 std::vector<std::int32_t>& dequant) {
    dequant.assign(static_cast<std::size_t>(area), 0);
    for (int i = 0; i < area; ++i) {
      const std::int32_t qc = levels[static_cast<std::size_t>(i)];
      const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
      const std::int64_t high = (aq * quant_step) & 0xFFFFFF;
      const std::int32_t dq = txv::round2(static_cast<std::int32_t>(high),
                                          txv::kQuantTableBits) /
                              params.dq_denom;
      dequant[static_cast<std::size_t>(i)] = std::max(
          -(1 << 15), std::min((1 << 15) - 1, qc < 0 ? -dq : dq));
    }
  };
  const auto reconstruct = [&](const av2enc::tcq::CctxResult& candidate,
                               std::vector<std::int16_t>& u_res,
                               std::vector<std::int16_t>& v_res) {
    std::vector<std::int32_t> du;
    std::vector<std::int32_t> dv;
    dequant_plane(candidate.u_levels, du);
    dequant_plane(candidate.v_levels, dv);
    std::vector<std::int32_t> pu(static_cast<std::size_t>(area));
    std::vector<std::int32_t> pv(static_cast<std::size_t>(area));
    for (int i = 0; i < area; ++i) {
      const av2enc::tcq::CctxPair pair = av2enc::tcq::cctx_dequant_pair(
          du[static_cast<std::size_t>(i)], dv[static_cast<std::size_t>(i)],
          candidate.type, 8);
      pu[static_cast<std::size_t>(i)] = pair.u;
      pv[static_cast<std::size_t>(i)] = pair.v;
    }
    u_res.assign(static_cast<std::size_t>(coded_pels), 0);
    v_res.assign(static_cast<std::size_t>(coded_pels), 0);
    txv::inverse_block(pu.data(), params, u_res.data());
    txv::inverse_block(pv.data(), params, v_res.data());
  };
  const auto distortion = [&](const std::vector<std::int16_t>& u_res,
                              const std::vector<std::int16_t>& v_res) {
    double sse = 0.0;
    for (int plane = 1; plane <= 2; ++plane) {
      const std::vector<std::uint8_t>& pred = plane == 1 ? u_pred : v_pred;
      const std::vector<std::int16_t>& res = plane == 1 ? u_res : v_res;
      for (int y = 0; y < ch; ++y) {
        for (int x = 0; x < cw; ++x) {
          const std::size_t index =
              static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
              static_cast<std::size_t>(x);
          const int value = clip1(static_cast<int>(pred[index]) +
                                  static_cast<int>(res[index]));
          const double d =
              static_cast<double>(source_sample(frame, plane, x0 + x, y0 + y)) -
              static_cast<double>(value);
          sse += d * d;
        }
      }
    }
    return sse;
  };

  const double lambda = av2enc::rdo::lambda_sse_from_quant_step(
      1.0 / params.quant_factor, /*sse_scale=*/0.125);
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  syntax::build_scan(params.w, params.h, scan);
  const auto scan_eob = [&](const std::vector<std::int32_t>& levels) {
    for (int c = area - 1; c >= 0; --c) {
      if (levels[static_cast<std::size_t>(
              scan[static_cast<std::size_t>(c)])] != 0) {
        return c + 1;
      }
    }
    return 0;
  };

  int best = -1;
  double best_cost = std::numeric_limits<double>::infinity();
  int best_eob_u = 0;
  int best_eob_v = 0;
  std::vector<std::int16_t> best_u_res;
  std::vector<std::int16_t> best_v_res;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const av2enc::tcq::CctxResult& candidate = candidates[i];
    const int eob_u = scan_eob(candidate.u_levels);
    // The decoder takes the all_zero branch (cctx_type inferred 0, no symbol)
    // when U eob is 0; an intra block with U eob == 1 also infers 0.
    if (candidate.type != av2enc::tcq::CctxType::kNone) {
      if (eob_u == 0) continue;
      if (!is_inter && eob_u == 1) continue;
    }
    std::vector<std::int16_t> u_res;
    std::vector<std::int16_t> v_res;
    reconstruct(candidate, u_res, v_res);
    const double sse = distortion(u_res, v_res);
    // The symbol is present for every non-zero U block except an intra block
    // with eob == 1; CCTX_NONE pays it too (the decoder always reads the
    // symbol for an eligible block).
    const double bits =
        (eob_u > 0 && (is_inter || eob_u != 1))
            ? av2enc::tcq::cctx_type_bits(candidate.type)
            : 0.0;
    const double cost = sse + lambda * bits;
    if (cost < best_cost) {
      best_cost = cost;
      best = static_cast<int>(i);
      best_eob_u = eob_u;
      best_eob_v = scan_eob(candidate.v_levels);
      best_u_res = std::move(u_res);
      best_v_res = std::move(v_res);
    }
  }
  if (best < 0) {
    error = "no signallable CCTX candidate";
    return false;
  }
  const av2enc::tcq::CctxResult& chosen =
      candidates[static_cast<std::size_t>(best)];

  const auto fill_plan = [&](TxPlan& plan, const std::vector<std::int32_t>& levels,
                             int eob, const std::vector<std::uint8_t>& pred,
                             const std::vector<std::int16_t>& res) {
    plan.tx_size = params.tx_size;
    plan.adj_w = params.w;
    plan.adj_h = params.h;
    plan.eob = eob;
    plan.levels = levels;
    plan.dequant_levels.clear();
    plan.recon.resize(static_cast<std::size_t>(coded_pels));
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const std::size_t index =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
            static_cast<std::size_t>(x);
        plan.recon[index] = static_cast<std::uint8_t>(
            clip1(static_cast<int>(pred[index]) +
                  static_cast<int>(res[index])));
      }
    }
  };
  fill_plan(u_plan, chosen.u_levels, best_eob_u, u_pred, best_u_res);
  fill_plan(v_plan, chosen.v_levels, best_eob_v, v_pred, best_v_res);
  if (cctx_type_out != nullptr) {
    *cctx_type_out = static_cast<int>(chosen.type);
  }
  return true;
}

// §5.20.3.1 ChromaMi bookkeeping for a sub-8 luma leaf.  `size` is the
// ChromaMiSize in luma 4x4 units; `mi_row`/`mi_col` its luma MI origin.  The
// first leaf (in decode order) whose incoming HasChroma is still 1 codes the
// whole inherited chroma block; its siblings code luma only.
struct ChromaInfo {
  bool valid = false;
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
};

struct TemporalFrameContext {
  bool use_ref_frame_mvs = false;
  int order_hint = 0;
  int num_past_refs = 0;
  int num_future_refs = 0;
  int closest_past = -1;
  int closest_future = -1;
  const temporal::MotionField* field = nullptr;
  const temporal::TrajectoryField* mv_traj = nullptr;
  // §5.18.2 TipFrameMode == TIP_FRAME_AS_REF plus the frame's TIP parameters.
  // `field` above is already §7.10.2-.4 scaled (the decoder mutates tpl_mvs the
  // same way before any tile is decoded).  The order hints are
  // OrderHints[ClosestPast/ClosestFuture] and drive §7.10.5 get_tip_offsets().
  bool tip_as_ref = false;
  int tip_weight_index = 0;
  int tip_closest_past_order_hint = temporal::kRestrictedOrderHint;
  int tip_closest_future_order_hint = temporal::kRestrictedOrderHint;
  bool tip_allow_hole_fill = false;
};

}  // namespace vf_detail

using namespace vf_detail;

class InterFrameEncoder {
 public:
  // One superblock's decided output (docs/wavefront-plan.md §6.1).  Indices
  // and level offsets inside a sink are superblock-local; merge_sinks()
  // rebases them onto the frame vectors in raster superblock order and leaves
  // the writer's layout exactly as the serial walk produced it.  Serial today:
  // one sink is filled at a time, in raster order.  This is the unit a
  // wavefront will schedule, so anything a concurrent superblock must own
  // stays here rather than on the encoder.
  struct SbSink {
    std::vector<syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    std::vector<palette::PaletteResult> palettes;
    std::vector<post::VideoSubPuRect> sub_pus;
    // Stage-0 diagnosis records for this superblock.
    std::vector<LeafDumpRecord> leaf_dump;
    // AV2ENC_EXACT_RD diagnosis: one measured (D, R) per emitted leaf, indexed
    // like blocks.  Internal nodes keep an invalid record.  Only written when
    // the gate is on, so gate-off streams never touch it.
    std::vector<exact_rd::LeafCost> leaf_costs;
    // Prediction scratch that must be private to a scheduling worker
    // (docs/wavefront-plan.md §3, §5 D5: the sharpest naive-parallel bug in
    // the file).  Every use assigns or resizes before it reads, so owning it
    // per sink does not change a single result.
    std::vector<std::uint8_t> compound_y;
    std::vector<std::uint8_t> compound_u;
    std::vector<std::uint8_t> compound_v;

    // Superblock-scoped decision state (docs/wavefront-plan.md §3).  Every
    // field here used to be a member of InterFrameEncoder, which was safe only
    // while exactly one superblock was in flight.  Each is written and read by
    // one superblock's walk and by nothing else, so owning it per unit makes
    // two concurrent units independent without changing a value: plan_pass()
    // clears the frame-global state, the depth-0 branch clears these, and the
    // merge folds the counters back into the encoder in raster order.
    struct State {
      // §6.19.2.3 clear_block_decoded_flags() replay; superblock-local by
      // construction (above-right/below-left availability inside the unit).
      std::array<intra_search::BlockDecoded, 3> decoded;
      // §5.20.1 CurrentQIndex / §5.20.5.7 SegmentIds for this superblock and
      // the §7.14.2 quantiser derived from them.
      int current_qindex = 0;
      int current_segment_id = 0;
      int quant_step = txv::kQuantStep;
      PlaneQuantStep plane_quant[3] = {};
      // Accumulators that used to be encoder members.  They only feed the
      // frame summary and the gates; summing them in raster order is exact.
      std::uint32_t inter_blocks = 0;
      std::uint32_t intra_blocks = 0;
      int temporal_added = 0;
      int temporal_selected = 0;
      int same_ref_blocks = 0;
      int refmvbank_added = 0;
      int refmvbank_selected = 0;
      int mv_traj_derived = 0;
      int amvd_selected = 0;
      int flex_precision_selected = 0;
      int derived_sign_selected = 0;
      int tip_blocks = 0;

      // Scalar clear only: the §6.19.2.3 maps are reset by the depth-0 branch
      // of collect_node*(), exactly as the serial walk did, so this must not
      // free their storage (that would add an allocation per superblock).
      void clear() {
        current_qindex = 0;
        current_segment_id = 0;
        quant_step = txv::kQuantStep;
        for (int plane = 0; plane < 3; ++plane) {
          plane_quant[plane] = PlaneQuantStep{};
        }
        inter_blocks = 0;
        intra_blocks = 0;
        temporal_added = 0;
        temporal_selected = 0;
        same_ref_blocks = 0;
        refmvbank_added = 0;
        refmvbank_selected = 0;
        mv_traj_derived = 0;
        amvd_selected = 0;
        flex_precision_selected = 0;
        derived_sign_selected = 0;
        tip_blocks = 0;
      }
    } state;

    void reset() {
      blocks.clear();
      levels.clear();
      palettes.clear();
      sub_pus.clear();
      leaf_dump.clear();
      leaf_costs.clear();
      state.clear();
    }
  };

  // Sink-local marker for "this entry carries no levels".  merge_sinks() maps
  // it back to the writer's 0, so blocks that never coded a residual keep the
  // exact value they have today.
  static constexpr std::uint32_t kNoLevels = 0xFFFFFFFFu;

    InterFrameEncoder(const VideoInputFrame& source,
                    std::vector<inter::mc::RefFrame> references,
                    std::array<int, 7> frame_distance, int num_total_refs,
                    const VideoOptions& options, bool trace, bool keyframe,
                    int tile_x = 0, int tile_y = 0, int tile_w = 0,
                    int tile_h = 0, std::span<const std::uint16_t> cdf_seed = {},
                    std::vector<std::uint16_t>* cdf_final = nullptr,
                    const GmFramePlan* gm_plan = nullptr,
                    std::array<int, 7> ref_order_hints = {},
                    bool primary_ref_is_none = true,
                    const TemporalFrameContext* temporal = nullptr,
                    std::span<const std::uint8_t> prev_segment_ids = {},
                    double illum_delta = 0.0,
                    double ref_deficit_db = 0.0,
                    int census_frame = -1,
                    std::span<const VideoInputFrame> horizon_future = {});
    // Diagnostic frame ordinal for the census/decision dumps; never coded.
    void set_census_frame(int frame) { census_frame_ = frame; }
    // Display-order ordinal for the head/tail exact-rate TX arming windows.
    // The chained B coding order is not display order; -1 keeps the historical
    // coding-order `picture_ordinal()`.
    void set_display_ordinal(int display) { display_ordinal_ = display; }
    // The TIP-output quality probe must reproduce the base coded alternative,
    // not the joint tree: the probe is a PSNR guard, not a rate decision.
    void set_joint_plan_off() { joint_plan_off_ = true; }
    int arming_ordinal() const {
      return display_ordinal_ >= 0 ? display_ordinal_
                                   : tx_detail::picture_ordinal();
    }

    std::int64_t block_variance_cost_at(
      const partition::PartitionNodeInfo& info) const;
  // ---------------------------------------------------------------------
  // Rate-aware partition probe (--video-partition-rd)
  //
  // The historical tree prices a leaf by the variance of its source samples
  // about the block mean.  That ignores prediction: splitting a 64x64 block
  // into two 32-wide leaves can cut the inter residual a lot, but the variance
  // of a natural-image block is almost entirely within-child variance, so the
  // between-child term that a split can recover is tiny.  The probe measures
  // the quantity the encoder actually spends: one NEWMV prediction plus one
  // quantised luma residual, priced as distortion + lambda * estimated bits.
  //
  // Everything here is read-only with respect to encoder state: ME reads the
  // source/reference planes, compute_tx() writes only the local TxPlan, and
  // the result depends only on (frame, reference, options).  That keeps the
  // tree deterministic and lets the frame-level precompute run in parallel.
  // ---------------------------------------------------------------------
  struct PartitionProbeJob {
    int mi_row = 0;
    int mi_col = 0;
    partition::BlockSize size = partition::BlockSize::kInvalid;
  };

    static std::uint64_t partition_probe_key(int mi_row, int mi_col,
                                           partition::BlockSize size);

  // Mirrors encode_leaf()'s inter_fits test: the leaf coder handles up to
  // 64x64, the whole-frame path keeps reads inside the MI extent, and the
  // bottom row only has inter prediction when the edge-padded reference is
  // active.  Tiled coding keeps the historical tile-local probe-free path.
    bool partition_probe_fits(int mi_row, int mi_col,
                            partition::BlockSize size) const;

  // SSE price of one estimated syntax bit for the node at (mi_row, mi_col).
  // The historical formula reads the frame base_q_idx; with a per-superblock
  // delta-Q plan (--video-delta-q / --video-aq) the node must be priced with
  // its own superblock quantizer, exactly like the probe's residual (which
  // already quantises with the current superblock's plane_quant_).
    double probe_syntax_price(int mi_row, int mi_col) const {
      int qindex = static_cast<int>(options_.base_q_idx);
#if defined(AV2ENC_VIDEO_DELTAQ)
      qindex = base_q_index();
      if (qp_tools_active() && !sb_qindex_.empty()) {
        const int sb =
            (mi_row / qp_sb_mi()) * qp_sb_cols() + mi_col / qp_sb_mi();
        if (sb >= 0 && static_cast<std::size_t>(sb) < sb_qindex_.size()) {
          qindex = sb_qindex_[static_cast<std::size_t>(sb)];
        }
      }
#else
      (void)mi_row;
      (void)mi_col;
#endif
      const double step = rc::quant_step_from_qindex(qindex);
      return partition_probe_scale_at(options_, qindex) * step * step;
    }

  // Returns true and fills `cost` when the probe can model this leaf.  Nodes
  // that cannot be predicted (sub-8, right/bottom edges outside the coded
  // extent, ME failure) are left to the caller's historical estimate.
  // AV2ENC_PART_PROBE_CHROMA (default 0 = off): weight on the U/V cost the
  // partition probe adds to each candidate.  The probe historically prices
  // luma only; this lets it see a chroma-expensive split.
  static int partition_probe_chroma_weight() {
    static const int value = [] {
      const char* v = std::getenv("AV2ENC_PART_PROBE_CHROMA");
      if (v == nullptr || *v == '\0') return 0;
      char* end = nullptr;
      const long parsed = std::strtol(v, &end, 10);
      if (end == v || *end != '\0') return 0;
      return static_cast<int>(std::clamp<long>(parsed, 0, 64));
    }();
    return value;
  }

    bool partition_probe_leaf_cost(const partition::PartitionNodeInfo& info,
                                 double lambda, double& cost) const;

  // ---------------------------------------------------------------------
  // R19 interval-width rate model (AV2ENC_RATE_INTERVAL, default off).
  //
  // `plan_rate_bits` prices one plane of a decided TX plan with the §8.3
  // coefficient symbol sequence through the §8.2.6 coder interval arithmetic.
  // It replaces the historical `estimate_tx_bits` heuristic at the decision
  // sites; gate off keeps the historical call verbatim so every pinned
  // stream is byte-identical.
  // ---------------------------------------------------------------------
    void ensure_rate_interval_cdfs() const;
    double plan_rate_bits(const TxPlan& plan, int plane, bool is_inter,
                          int bsize) const;
    // AV2ENC_EFF_PERC: live frame-start coefficient CDF rows in the shape the
    // §8.3 model reads.  Used by the TX search when the gate selects the live
    // CDF rate variant.
    const still::qctx::CoeffCdfSlice* eff_cdf_slice() const {
      ensure_rate_interval_cdfs();
      if (!eff_cdf_slice_ready_) {
        eff_cdf_slice_ = rate_interval::slice_of(rate_interval_cdfs_);
        eff_cdf_slice_ready_ = true;
      }
      return &eff_cdf_slice_;
    }

  // Enumerates every node the bounded probe may price: the superblock root plus
  // its partition children down to `max_depth`.  Costs are looked up by
  // (mi_row, mi_col, size); nodes the solver reaches outside this set (forced
  // boundary partitions) fall back to a synchronous probe in the cost
  // callback, so the tree can never consult a stale or missing cost.
    void collect_partition_probe_jobs(
      int mi_row, int mi_col, partition::BlockSize size, int depth,
      int max_depth, int floor_px,
      std::vector<PartitionProbeJob>& jobs) const;

    std::unordered_map<std::uint64_t, double> build_partition_probe_costs(
      int max_depth, int floor_px) const;

  // --video-edge-mask: number of coded rows of a plane block that lie above
  // the visible picture bottom.  The tail rows are reconstructed by the
  // decoder but never output and never predicted from (MiRows clamp plus the
  // reference edge replication), so the encoder may treat them as hidden.
  // Returns h unchanged when the flag is off, the block is inside the
  // picture, or the frame is tiled (the edge path is whole-frame only).
    int visible_rows(int plane, int y0, int h) const;


#if defined(AV2ENC_VIDEO_DELTAQ)
  // ---------------------------------------------------------------------
  // §5.18.7.1/.8 per-superblock segmentation / delta-Q planning.
  // ---------------------------------------------------------------------
    int qp_sb_mi() const;
    int qp_sb_cols() const;
    int qp_sb_rows() const;
    int qp_sb_count() const;
    int base_q_index() const;
    bool qp_tools_active() const;

  // Deterministic per-superblock split, identical to the still
  // --still-per-segment-q planner: segment 1 is every superblock whose luma
  // variance is <= the frame mean variance.  --video-delta-q instead uses a
  // raster-parity staircase for its target qindex.
    bool build_sb_maps(std::string& error);

    bool build_segmentation(std::string& error);

    bool prepare_qp_plan(std::string& error);

  // Builds the per-superblock delta-Q steps from sb_target_q_.  A superblock
  // whose first (and only) leaf is a full-SB inter skip cannot code
  // read_delta_qindex() (§5.20.5.11: MiSize == SbSize && skip_flag), so the
  // plan must leave it at the carried qindex.  That target fix-up can shift
  // later steps, so re-run rc::plan_delta_q() until no target changes.
    bool plan_delta_q_steps(std::vector<int>& actual, std::string& error);
#endif  // AV2ENC_VIDEO_DELTAQ

  // §5.4.3 enable_sdp: decide the intra keyframe as per-64x64 luma and chroma
  // partition trees.  The luma tree uses the same --video-min-block /
  // --video-max-depth controls and rect-only family as the shared tree.  The
  // chroma tree is pinned to a single PARTITION_NONE block: the reference
  // model can search it and the writer can signal it, but the pinned dav2d
  // disagrees with the model's chroma-plane partition contexts inside a
  // decoded chroma tree, so no product stream produces one.
    bool decide_sdp(std::string& error);

  // --video-skip-collapse (§5.20.3.2 tree rewrite, opt-in).
  //
  // A superblock whose coded leaves are all plain zero-motion inter skips
  // predicts each 4x4 cell by copying the reference at the same position.
  // Replacing that subtree with one superblock-sized skip leaf therefore
  // reproduces exactly the same reconstruction while coding one partition
  // symbol per superblock instead of the tree.  Eligibility is read from the
  // leaves the decision pass just produced, so nothing is guessed: a leaf
  // that codes a residual, moves, or uses any mode that changes the predictor
  // (compound, TIP, BAWP, inter-intra, morph, global MV) disqualifies its
  // superblock.  A rewritten superblock that the leaf coder does not reproduce
  // as a plain zero-motion skip is restored and the frame re-collected.
  //
  // Returns true (and leaves the tree alone) when the option is off.
  template <typename CollectFn>
  bool collapse_static_superblocks(std::vector<PartitionNode>& superblocks,
                                   const partition::DecisionOptions& tree_options,
                                   CollectFn&& collect_frame,
                                   std::string& error) {
    if (!options_.skip_collapse || keyframe_) return true;
    // The rewrite assumes whole-frame MI coordinates, one tile and a 64x64
    // superblock (a 128/256 root always has to code a partition).
    if (sb_size_ > 64 || width_ != static_cast<int>(source_.width) ||
        height_ != static_cast<int>(source_.height) ||
        superblocks.empty()) {
      return true;
    }
    // §5.18.7.1/.8 codes a per-superblock delta at the first leaf unless the
    // block is a full-SB skip, so a quantiser plan is not modelled here.
    if (options_.enable_segmentation || options_.enable_delta_q ||
        options_.per_segment_q || options_.enable_sdp) {
      return true;
    }

    const int sb_cols = partition::sb_cols(geometry_);
    const int sb_rows = partition::sb_rows(geometry_);
    if (sb_cols * sb_rows != static_cast<int>(superblocks.size())) {
      error = "skip collapse: superblock grid does not match the tree";
      return false;
    }
    // Group the coded leaves by superblock root and mark the all-static ones.
    std::vector<char> candidate(superblocks.size(), 0);
    std::size_t index = 0;
    int sb = 0;
    while (index < blocks_.size()) {
      const syntax::BlockSyntax& root = blocks_[index];
      if ((root.flags & syntax::kFlagSbRoot) == 0) {
        error = "skip collapse: node is not rooted in a superblock";
        return false;
      }
      const std::size_t end =
          index + 1 + static_cast<std::size_t>(root.subtree);
      if (end > blocks_.size() || sb >= static_cast<int>(candidate.size())) {
        error = "skip collapse: inconsistent block tree";
        return false;
      }
      bool static_sb = true;
      // A GLOBALMV leaf predicts with the §7.12.2.2 global motion vector, so
      // it is only a plain copy when the frame codes no global motion at all.
      const bool global_mv_is_zero =
          !options_.enable_gme && !options_.enable_warp;
      for (std::size_t k = index; k < end && static_sb; ++k) {
        const syntax::BlockSyntax& leaf = blocks_[k];
        if (leaf.partition != static_cast<std::uint8_t>(PartitionType::kNone)) {
          continue;  // an internal node, not a coded leaf
        }
        const bool mode_is_zero_motion =
            leaf.inter_mode !=
                static_cast<std::uint8_t>(syntax::kInterGlobalMv) ||
            global_mv_is_zero;
        const bool plain_skip =
            leaf.is_inter != 0 && leaf.skip_flag != 0 && leaf.compound == 0 &&
            leaf.tip_mode == 0 && leaf.motion_mode == 0 &&
            leaf.use_bawp == 0 && leaf.use_bawp_chroma == 0 &&
            leaf.explicit_bawp == 0 && leaf.explicit_bawp_scale == 0 &&
            leaf.use_optflow == 0 && leaf.use_refinemv == 0 &&
            leaf.morph_pred == 0 && leaf.ref_index == 0 && leaf.mv_y == 0 &&
            leaf.mv_x == 0 && mode_is_zero_motion &&
            (leaf.flags & syntax::kFlagIsIbc) == 0 && leaf.tree_type == 0;
        if (!plain_skip) static_sb = false;
      }
      if (static_sb) candidate[static_cast<std::size_t>(sb)] = 1;
      index = end;
      ++sb;
    }
    if (sb != static_cast<int>(candidate.size())) {
      error = "skip collapse: block tree does not cover every superblock";
      return false;
    }

    // Cheapest legal tree for one superblock: PARTITION_NONE costs nothing and
    // every optional split is unaffordable.  Spec-forced splits (frame
    // boundary) carry no signal cost and are still taken.
    const partition::CostFunction collapse_cost =
        [](const partition::PartitionNodeInfo& info) {
          partition::NodeCosts costs;
          const int bw = partition::block_width(info.size);
          const int bh = partition::block_height(info.size);
          costs.leaf = (bw <= 64 && bh <= 64) ? 0.0 : partition::kInfiniteCost;
          constexpr double kUnreachable = 1.0e12;
          costs.split_signal = kUnreachable;
          costs.horz_signal = kUnreachable;
          costs.vert_signal = kUnreachable;
          return costs;
        };

    const int sb_mi = partition::sb_size_mi(geometry_);
    const std::vector<PartitionNode> solved = superblocks;
    std::vector<char> rewritten(superblocks.size(), 0);
    int rewrites = 0;
    for (std::size_t s = 0; s < superblocks.size(); ++s) {
      if (!candidate[s]) continue;
      if (superblocks[s].partition == PartitionType::kNone) continue;
      const int sb_row =
          static_cast<int>(s / static_cast<std::size_t>(sb_cols)) * sb_mi;
      const int sb_col =
          static_cast<int>(s % static_cast<std::size_t>(sb_cols)) * sb_mi;
      const partition::PartitionDecision decision =
          partition::decide_superblock(geometry_, sb_row, sb_col,
                                       collapse_cost, tree_options);
      if (decision.status != partition::DecisionStatus::kOk) continue;
      superblocks[s] = decision.root;
      rewritten[s] = 1;
      ++rewrites;
    }
    if (rewrites == 0) return true;

    // Verify: the rewritten superblock must come back as one plain
    // zero-motion skip leaf.  Anything else goes back to its solved tree.
    for (int pass = 0; pass < 3; ++pass) {
      if (!collect_frame()) return false;
      std::vector<char> revert(superblocks.size(), 0);
      std::size_t cursor = 0;
      int cell = 0;
      while (cursor < blocks_.size()) {
        const syntax::BlockSyntax& root = blocks_[cursor];
        const std::size_t end =
            cursor + 1 + static_cast<std::size_t>(root.subtree);
        if (end > blocks_.size() ||
            cell >= static_cast<int>(superblocks.size())) {
          error = "skip collapse: inconsistent block tree";
          return false;
        }
        if (rewritten[static_cast<std::size_t>(cell)]) {
          const bool collapsed =
              root.subtree == 0 &&
              root.partition ==
                  static_cast<std::uint8_t>(PartitionType::kNone) &&
              root.is_inter != 0 && root.skip_flag != 0 && root.compound == 0 &&
              root.mv_y == 0 && root.mv_x == 0 && root.ref_index == 0;
          if (!collapsed) revert[static_cast<std::size_t>(cell)] = 1;
        }
        cursor = end;
        ++cell;
      }
      if (cell != static_cast<int>(superblocks.size())) {
        error = "skip collapse: block tree does not cover every superblock";
        return false;
      }
      bool any_bad = false;
      for (std::size_t s = 0; s < revert.size(); ++s) {
        if (revert[s]) any_bad = true;
      }
      if (!any_bad) return true;
      for (std::size_t s = 0; s < superblocks.size(); ++s) {
        if (!revert[s]) continue;
        superblocks[s] = solved[s];
        rewritten[s] = 0;
      }
    }
    error = "skip collapse: partition collapse did not settle";
    return false;
  }

  // Decision + reconstruction phase (no syntax, no filters).  Kept
  // separate so a multi-tile frame can run one frame-level filter pass
  // after every tile has been reconstructed.
    bool decide(std::string& error);

  // Runs the §7.18-§7.20 decision + pixel stage on this encoder's
  // reconstruction, filling filter_plan_.  Single-tile path only.
    bool apply_loop_filters(std::string& error);

  // §7.14.2 plane quantizers for the current §5.20.1 qindex.  Luma keeps the
  // historical qlookup() table so the default stream is byte-identical; a UV
  // delta resolves each chroma plane/component through get_q().
    void frame_plane_quant(int qindex, PlaneQuantStep out[3],
                         int& quant_step) const;

    void update_plane_quant(SbSink::State& st) const;

  // §7.17 deblocking of this encoder's reconstruction.  Runs after decide()
  // and before the frame-level filter pass / reference update.
    bool deblock(std::string& error);

  // Syntax phase.  `filters` is the frame-level plan (a single-tile encoder
  // passes its own); a multi-tile caller passes the shared frame plan so the
  // per-superblock symbols match the filtered reference.
    bool write(std::vector<std::uint8_t>& tile_payload,
             std::vector<syntax::BlockSyntax>& blocks,
             std::vector<std::int32_t>& levels,
             const post::VideoFilterPlan* filters, std::string& error);

    bool encode(std::vector<std::uint8_t>& tile_payload,
              std::vector<syntax::BlockSyntax>& blocks,
              std::vector<std::int32_t>& levels, std::string& error);

    const std::vector<syntax::BlockSyntax>& blocks() const;
#if defined(AV2ENC_VIDEO_DELTAQ)
  // §5.20.5.7 SegmentIds after decide(); the next frame uses this as
  // PrevSegmentIds when segmentation_temporal_update is signalled.
    const std::vector<std::uint8_t>& segment_map() const;
#endif

    const Plane& recon(int plane) const;
  // --video-edge-inter: after the in-loop filters, extend the reconstruction
  // past the coded MI edge (MiRows*4 luma) by replicating the last MI row, so
  // this frame can enter a reference bank with the same padded extent the
  // decoder uses.  No-op when the edge path is off or the recon is not padded.
    void pad_edge_recon();
  // §7.22 motion-field source: the per-4x4 coded mode/MV grid and its extent.
  // Used by the opt-in TIP path to store a reference frame's SavedMvField.
    const std::vector<inter::mc::MiInfo>& mv_grid() const;
  // §5.18.9 model plan the block decisions used (null when global motion is
  // off or the frame is not on the GME carrier).  §7.22 storage projects the
  // per-8x8 warped MVs from it.
    const GmFramePlan* gm_plan() const;
    int mv_mi_cols() const;
    int mv_mi_rows() const;
    int recon_mi_cols() const;
    int recon_mi_rows() const;
    std::uint32_t inter_blocks() const;
    std::uint32_t intra_blocks() const;
    int temporal_added() const;
    int temporal_selected() const;
    int same_ref_blocks() const;
    int refmvbank_added() const;
    int refmvbank_selected() const;
    int mv_traj_derived() const;
    int amvd_selected() const;
    int flex_precision_selected() const;
    int derived_sign_selected() const;
    int tip_blocks() const;
    const post::VideoFilterPlan& filter_plan() const;

 private:
  // §5.20.3.2 symbol plan for one SDP node, derived by the reference planner
  // with the authoritative boundary/allowed set the solver stored.  The
  // writer emits exactly these symbols.
    bool plan_sdp_symbols(const sdpp::Node& node, syntax::BlockSyntax& entry,
                        std::string& error);

  // §5.20.3.1 SDP intra tree walk.  A split node carries no partition syntax
  // and its two children are the luma root and then the chroma root.  Leaves
  // are encoded with the tree-type/chroma-ownership arguments so a CHROMA_PART
  // leaf codes chroma only and a LUMA_PART leaf codes luma only.
    bool collect_sdp_node(SbSink& sink, const sdpp::Node& node,
                        std::string& error);

  // §5.20.3.1: a node with HasChroma and chromaOffset == 0 captures the
  // inherited chroma block before its children are decoded.  ChromaMi* is the
  // last such node on the path, so a sub-8 split hands the same ChromaMi block
  // to the child that keeps HasChroma.
    static ChromaInfo chroma_step(const PartitionNode& node,
                                const ChromaInfo& inherited);

  // Stage-0 AV2ENC_BLOCKDUMP capture.  `level_begin` is levels_.size() before
  // the leaf's encode_leaf() call, so [level_begin, levels_.size()) is exactly
  // the quantised run this leaf hands the writer.
    void record_leaf_dump(SbSink& sink, const syntax::BlockSyntax& entry,
                        std::size_t level_begin);

    void dump_leaf_decisions();

  // Assembles blocks_/levels_/palettes_/sub_pus_ in raster superblock order.
  // The only work is rebasing sink-local offsets; block order inside a sink is
  // already the writer's tree pre-order, so this reproduces the historical
  // vectors exactly.
    void merge_sinks();

  // Stage-1 scheduling unit: clear the unit's sink and run one superblock's
  // whole tree in pre-order.  The serial driver walks the units in raster
  // order, exactly the historical loop.
  //
  // The unit's superblock-scoped quantiser state is seeded from the frame-level
  // default.  The depth-0 branch overwrites it when a delta-Q plan assigns a
  // per-superblock qindex, exactly as the serial walk did, so a unit never
  // depends on which unit ran before it.
    bool run_one_sb(SbSink& sink, const PartitionNode& superblock,
                  std::string& error);
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
    bool run_one_sb(SbSink& sink, const pe::PartitionNode& superblock,
                  std::string& error);
#endif
    bool run_one_sb(SbSink& sink, const sdpp::Node& superblock,
                  std::string& error);

    void seed_unit_state(SbSink& sink);

  // ------------------------------------------------------------- wavefront ---
  //
  // docs/wavefront-plan.md §6.1/§6.2.  The superblock is the scheduling unit.
  // Units are dispatched on a level-synchronous diagonal with skew 2:
  //
  //     key(sbr, sbc) = 2*sbr + sbc
  //
  // Why skew 2 and not the naive `r + c` anti-diagonal: a superblock reads its
  // above-right neighbour one *superblock column* into the row above, both for
  // the §7.12.2 merge stack (merge.cpp:163 with the sb_border column rounding
  // at :586-589) and for the §7.13.2.1 above-right sample count
  // (video_intra_search.h:143-153, whose border rule reaches 2*sb_mi cells
  // from the SB origin).  With `r + c`, SB (0,2) and SB (1,1) share a diagonal
  // and (1,1)'s first 64x64 leaf reads cell (15,32) of (0,2) plus 64 samples of
  // row 15 at MI columns 32..47: a zero-filled recon when (0,2) has not run.
  // With skew 2, every unit a superblock reads has a strictly smaller key:
  // left key-1, above key-2, above-left key-3, above-right key-1.
  //
  // Within a unit nothing changes: collect_node*() keeps the exact serial
  // pre-order, and merge_sinks() reassembles the frame in raster order, so the
  // writer cannot tell which schedule produced the sinks.

  // Tuned in §9.4: below this average diagonal width the outer schedule loses
  // more candidate-level parallelism inside run_one_sb than it gains.
  static constexpr double kMinWavefrontWidth = 2.0;

  // Diagonal member lists in increasing superblock column, i.e. a stable order
  // for chunk assignment.  Returns false when the superblock vector is not the
  // whole-frame raster grid (then the caller keeps the serial loop).
    bool wavefront_diagonals(std::size_t count,
                           std::vector<std::vector<std::size_t>>& out) const;

  // Superblock-scoped state that a diagonal schedule cannot reproduce is kept
  // on the serial walk.
  //
  //   * an intra keyframe carries the §5.20.8 palette cache, the
  //     §5.20.5.3 IBC ring/neighbour map and the §7.13.3.26 morph state along
  //     the row, plus (opt-in) the SDP §8.3.2 plane grids in decode order;
  //   * --video-refmvbank seeds each superblock from the *whole* coded row
  //     above (reset_refmvbank_for_sb()), which is not available before the
  //     row above has finished;
  //   * segmentation / delta-Q keeps an above/left context map with a
  //     superblock-row clear (docs/wavefront-plan.md §7.4 notes no gate covers
  //     it).
    bool wavefront_allowed() const;

  // Average independent superblocks per diagonal over the skew-2 ~schedule.
  // This is the wavefront's width (§4.3: 2.3 at 480x240, 6.1 at 1440x720) and
  // therefore the ceiling on how much it can replace.
    double wavefront_width() const;

  // Diagnostic schedule override, mirroring AV2ENC_BLOCKDUMP: "0" keeps the
  // serial raster walk, "1" forces the wavefront, unset uses the width rule.
  // Both schedules code identical bytes, so this only exists to A/B them.
    static int wavefront_override();

  // The schedule selector.  `--cpu-threads 1` keeps the historical raster loop
  // byte-for-byte (no wavefront, no thread, no pool).
  //
  // Rule (measured in docs/wavefront-plan.md §9.4): go wide across superblocks
  // when the skew-2 diagonal holds at least kMinWavefrontWidth independent
  // units on average.  A wavefront region hides the candidate-level
  // parallelism inside run_one_sb (a nested parallel_for falls back to serial),
  // so below roughly two units per diagonal there is nothing to win.  Above it
  // the outer schedule measured faster on every point tested, at every thread
  // count from 2 to 8.
  template <typename SbVec>
  bool run_superblocks(const SbVec& superblocks, std::string& error) {
    const unsigned workers = parallel::resolve_workers(options_.cpu_threads);
    bool use_wavefront =
        workers > 1 && parallel::pool_active() && wavefront_allowed();
    if (use_wavefront) {
      const int forced = wavefront_override();
      use_wavefront =
          forced == 1 || (forced < 0 && wavefront_width() >= kMinWavefrontWidth);
    }
    if (!use_wavefront) {
      for (std::size_t sb = 0; sb < superblocks.size(); ++sb) {
        if (!run_one_sb(sb_sinks_[sb], superblocks[sb], error)) return false;
      }
      return true;
    }
    return run_wavefront(superblocks, workers, error);
  }

  // Stage 3: N-way, one fork-join per diagonal.  The persistent pool cannot
  // express a wait in the middle of a unit, so the schedule stays
  // level-synchronous: every member of diagonal d finishes before d+1 starts.
  // Chunk assignment is timing-dependent by design; results are per-unit sinks
  // merged in raster order, so only the unit results have to be reproducible.
  template <typename SbVec>
  bool run_wavefront(const SbVec& superblocks, unsigned workers,
                     std::string& error) {
    std::vector<std::vector<std::size_t>> diagonals;
    if (!wavefront_diagonals(superblocks.size(), diagonals)) {
      for (std::size_t sb = 0; sb < superblocks.size(); ++sb) {
        if (!run_one_sb(sb_sinks_[sb], superblocks[sb], error)) return false;
      }
      return true;
    }
    std::vector<std::string> unit_error(superblocks.size());
    std::vector<char> unit_ok(superblocks.size(), 1);
    for (const std::vector<std::size_t>& members : diagonals) {
      if (members.empty()) continue;
      const unsigned hint = std::min<unsigned>(
          workers, static_cast<unsigned>(members.size()));
      // A one-member diagonal runs inline with no parallel region, so its
      // candidate loops keep the pool (§9.4).
      parallel::parallel_for(members.size(), hint, [&](std::size_t i) {
        const std::size_t sb = members[i];
        unit_ok[sb] =
            run_one_sb(sb_sinks_[sb], superblocks[sb], unit_error[sb]) ? 1 : 0;
      });
    }
    for (std::size_t sb = 0; sb < superblocks.size(); ++sb) {
      if (!unit_ok[sb]) {
        error = unit_error[sb];
        return false;
      }
    }
    return true;
  }

  // Frame-level planning for one rect/extended decision pass: everything the
  // superblocks share.  A delta-Q fixpoint runs the walk more than once, so
  // every pass starts from exactly this state.  `sub_pus_` is deliberately not
  // cleared: the historical walk never cleared it, and changing that would
  // change what the deblock grid sees.
    void plan_pass();

  // The SDP keyframe walk's frame-level reset, unchanged from decide_sdp().
    void plan_sdp_pass();

    bool collect_node(SbSink& sink, const PartitionNode& node,
                    std::string& error,
                    const ChromaInfo& inherited_chroma = ChromaInfo{},
                    PartitionNode* emitted = nullptr);

  // AV2ENC_RD_TAIL_FLOOR: sequential, measured P-frame coding-block descent on
  // the tail pictures.  `emitted` is the node the coder actually wrote: the
  // reference node itself, or the measured HORZ/VERT/8x8 layout when the exact
  // trial won.  The per-superblock walk in collect_frame() feeds it back into
  // base_superblocks so the writer sees the committed tree.
    bool tail_floor_collect_node(SbSink& sink, const PartitionNode& node,
                               const ChromaInfo& inherited,
                               PartitionNode* emitted, std::string& error);
    bool tail_floor_leaf_cost(SbSink& sink, std::size_t index, double lambda,
                            double hf, double edge, double* d_out,
                            double* r_out, std::string& error);
    bool tail_floor_trial_cost(SbSink& sink, std::size_t begin, double lambda,
                             int splits, double* d_out, double* r_out,
                             std::string& error);
    bool tail_floor_candidate(const PartitionNode& node) const;

  // AV2ENC_PART_DIAG: measure both sides of an emitted P leaf and commit the
  // reference.  The stream never changes; only the counters move.
    bool part_diag_candidate(const PartitionNode& node) const;
    bool part_diag_collect_node(SbSink& sink, const PartitionNode& node,
                              const ChromaInfo& inherited,
                              PartitionNode* emitted, std::string& error);

#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
  // Extended-partition variant of collect_node().  The solver already resolved
  // §5.20.3.1 ChromaMi* for every node (including the HORZ_3/VERT_3 middle
  // pair and the 1/8 subsizes), so the leaf coder receives exactly the
  // ownership arguments the decoder will derive.
    bool collect_node_ext(SbSink& sink, const pe::PartitionNode& node,
                        std::string& error);
#endif  // AV2ENC_VIDEO_EXT_PARTITIONS

  // §7.12.2.2 GLOBALMV projection for this block.  A non-identity model may be
  // predicted with the warped predictor (`use_warp`); otherwise the projected
  // centre MV drives ordinary translational MC.  When global motion is off (or
  // the reference has no model) this is the historical zero MV.
    void global_mv_for_block(int ref_index, int mi_row, int mi_col, int bw4,
                           int bh4, bool& use_warp,
                           inter::mc::Mv& mv) const;

    merge::StackParams make_stack_params(int mi_row, int mi_col, int w4,
                                       int h4, int ref_index,
                                       SbSink::State& st) const;

    bool drl_constraint_sort_suppressed(int ref_index) const;

    bool load_stack(int mi_row, int mi_col, int w4, int h4, int ref_index,
                  SbSink::State& st, merge::StackParams& params,
                  merge::MvStack& stack) const;

  // §5.20.2.2 reset reference motion vector bank at the start of each
  // superblock.  Non-top rows seed from the coded row above (rowHits < 4).
    void reset_refmvbank_for_sb(int mi_row, int mi_col);

  // §5.20.5.3 decode_block(): after the block is decoded, update the bank.
    void update_refmvbank_leaf(const syntax::BlockSyntax& entry);

    bool predict_luma(int ref_index, inter::mc::Mv mv, int x0, int y0, int w,
                    int h, std::vector<std::uint8_t>& out) const;

    bool predict_all(int ref_index, inter::mc::Mv mv, int x0, int y0, int w,
                   int h, std::vector<std::uint8_t>& luma,
                   std::vector<std::uint8_t>& u,
                   std::vector<std::uint8_t>& v) const;

    bool run_me(int ref_index, int x0, int y0, int w, int h,
              inter::mc::Mv& mv, std::string& error);

  // Pure §7.13 NEWMV search for one block.  It reads only the source frame,
  // the reference frame and the frame-level ME options; neighbour
  // reconstruction and the MV stack are not inputs, so identical jobs produce
  // identical results regardless of the order (or thread) they run in.
    bool search_me(int ref_index, int x0, int y0, int w, int h,
                 inter::mc::Mv& mv, std::string& error) const;

  // ---------------------------------------------------------------------
  // Deterministic batched motion search.
  //
  // decide() knows the full partition tree before collect_node() walks it, and
  // the NEWMV search above is pure, so every leaf's search can run up front in
  // parallel.  Each job owns its result slot and me_cache_ is filled in job
  // order afterwards, so the per-leaf lookup returns exactly the MV the serial
  // search returns.  Only the whole-frame path opts in; a tiled frame already
  // parallelises whole tile encoders and must not nest thread pools.
  // ---------------------------------------------------------------------
    static std::uint64_t me_cache_key(int ref_index, int x0, int y0, int w,
                                    int h);

    void precompute_leaf_me(const std::vector<PartitionNode>& superblocks,
                          unsigned workers);

  // §7.13.2.15/.16 neighbour smooth flags at the coding-block origin, derived
  // from the IntraJointMode grid exactly like still_encoder's
  // neighborhood_smooth().  Only the opt-in full intra search needs them.
    void luma_neighbor_smooth(int r, int c, bool& above_smooth,
                            bool& left_smooth) const;

    void set_intra_joint(int r, int c, int w4, int h4, int joint);

  // §7.13.2.15/.16 chroma neighbour smooth flags at the chroma coding-block
  // origin (4x4 chroma units).  Only the AV2ENC_RD_UVFULL chroma mode search
  // needs them: with directional chroma prediction the edge filter reads
  // filterTypeAbove/Left from the neighbouring UVMode, exactly like the still
  // encoder's uv_smooth_ map.  Every non-intra leaf (and every CfL leaf) writes
  // 0, which is the §5.20.7.32 default.
    void chroma_neighbor_smooth(int r, int c, bool& above_smooth,
                              bool& left_smooth) const;
    void set_uv_smooth(int r, int c, int w4, int h4, int uv_mode, bool is_cfl);

  // ---------------------------------------------------------------------
  // §5.20.5.3/.4 intra block copy + §7.13.3.26 morphological prediction
  // ---------------------------------------------------------------------
    ibc::FrameView ibc_frame_view() const;

  struct IbcChoice {
    bool found = false;
    ibc::BlockVector bv{};
    int stack_index = -1;  // >= 0 => NEARMV with that RefMvIdx
    bool morph = false;    // §7.13.3.26 adaptation selected
    std::int64_t sse = 0;
    std::vector<std::uint8_t> luma;
    std::vector<std::uint8_t> u;
    std::vector<std::uint8_t> v;
    std::vector<std::uint8_t> morph_scratch;
  };

  // Chooses an IBC block vector for a keyframe leaf.  A plain exact copy and a
  // §7.13.3.26 morphological adaptation are both evaluated; the best luma SSE
  // below the caller's intra SSE wins.  Returns false only on a hard error.
    bool choose_ibc_leaf(int r, int c, int bsize, std::int64_t intra_sse,
                       IbcChoice& out, std::string& error);

    bool ibc_available_for_leaf(int bsize) const;

  // ---------------------------------------------------------------------
  // §5.20.8 luma palette (--video-palette)
  // ---------------------------------------------------------------------
  struct PaletteChoice {
    bool found = false;
    palette::PaletteResult result;
    std::vector<std::uint8_t> luma;
    std::int64_t sse = 0;
  };

    bool palette_available_for_leaf(int bsize, int w, int h) const;

  // Builds the §5.20.8 palette for one keyframe leaf from the same above/left
  // cache the decoder's get_palette_cache() sees.  The palette prediction is
  // scored against `intra_pred_sse`, the SSE of the ordinary intra prediction
  // (the palette codes eob=0 on all planes, so a residual-coded intra leaf is
  // not the right comparison).  The caller stores a winning palette in the
  // frame PaletteState only after the IBC decision, so the following blocks'
  // cache matches the writer.
    bool choose_palette_leaf(int r, int c, int bsize, int w, int h,
                           std::int64_t intra_pred_sse, PaletteChoice& out,
                           std::string& error);

  // §7.13.2 single-mode prediction for the inter-intra candidates.  The
  // neighbour counts come from the same §6.19.2.3 replay the intra search uses;
  // chroma smooth flags are always false in this encoder (UVMode is DC/CfL).
    bool predict_intra_plane_mode(int plane, int x0, int y0, int w, int h,
                                intra::Mode mode, bool above_smooth,
                                bool left_smooth, const SbSink::State& st,
                                std::vector<std::uint8_t>& out) const;

  // §7.13.2 single-mode prediction for the inter-intra candidates.  The
  // neighbour counts come from the same §6.19.2.3 replay the intra search uses;
  // chroma smooth flags are always false in this encoder (UVMode is DC/CfL).
  // §7.13.3.29/.30 inter-intra.  Evaluates the four intra modes plus the
  // signalable wedge masks against the inter prediction and returns the best
  // luma+chroma blend.  `inter_y/u/v` are the single-reference predictions.
    bool evaluate_inter_intra(int bsize, int x0, int y0, int w, int h,
                            const std::vector<std::uint8_t>& inter_y,
                            const std::vector<std::uint8_t>& inter_u,
                            const std::vector<std::uint8_t>& inter_v,
                            std::vector<std::uint8_t>& out_y,
                            std::vector<std::uint8_t>& out_u,
                            std::vector<std::uint8_t>& out_v, int& mode_out,
                            bool& wedge_out, int& index_out,
                            std::int64_t& sse_out, const SbSink::State& st,
                            std::string& error);

#if defined(AV2ENC_VIDEO_FRAMES)
  // --- §7.13.3.8-.12 / §7.13.3.3-.6 compound prediction -------------------
  //
  // The temporal tools predict from the two reference planes by absolute
  // frame coordinates and write a block-local result, so a candidate never
  // disturbs the reconstructed reference planes.
    static temporal::PlaneView temporal_plane(const inter::mc::RefFrame& ref,
                                            int plane);

    static temporal::RefArea full_area(const temporal::PlaneView& view);

  // §7.13.3.4 get ref area for one reference list (the input is identical for
  // both lists except the candidate MV).
    bool temporal_ref_area(int sub_x, int sub_y, int x, int y, int w, int h,
                         const inter::mc::Mv mvs[2],
                         temporal::RefArea out[2]) const;

  // Builds the two 1/16-sample predictions for one compound block.  Shared by
  // the plain path and the CWP / implicit-mask decision, which evaluates
  // several blend modes over the same pair of predictions.  `ref0`/`ref1`
  // select the pair; the historical carrier is (0,1) and
  // --video-same-ref-compound may pass (0,0).
    bool build_compound_block(const inter::mc::Mv mvs[2], int bsize, int x0,
                            int y0, int w, int h,
                            multiref::CompoundPrediction& prediction,
                            std::string& error, int ref0 = 0, int ref1 = 1);

  // Applies one blend mode and copies the block out of the frame-sized
  // scratch planes.  `params.use_imp_mask` is taken as given (the caller
  // derives it with multiref::implicit_mask_eligible()).
    bool store_compound_block(const multiref::CompoundPrediction& prediction,
                            const multiref::CompoundModeParams& params,
                            int x0, int y0, int w, int h,
                            std::vector<std::uint8_t>& y,
                            std::vector<std::uint8_t>& u,
                            std::vector<std::uint8_t>& v, std::string& error);

  // Plain COMPOUND_AVERAGE prediction at the coded MVs.
    bool compound_plain(const inter::mc::Mv mvs[2], int bsize, int x0, int y0,
                      int w, int h, std::vector<std::uint8_t>& y,
                      std::vector<std::uint8_t>& u,
                      std::vector<std::uint8_t>& v, std::string& error,
                      const multiref::CompoundModeParams* mode_params =
                          nullptr,
                      int ref0 = 0, int ref1 = 1);

  // §5.20.7.6 / §7.13.3.14: choose the AVERAGE blend for a NEAR_NEARMV
  // compound block.  With `enable_cwp_weighted` the non-equal
  // Cwp_Weighting_Factor row is searched; with `enable_imp_msk_bld` the
  // derived implicit mask is evaluated first.  Both are off by default and the
  // caller falls back to compound_plain(), keeping historical streams exact.
    bool compound_search_average(const inter::mc::Mv mvs[2], int bsize, int x0,
                               int y0, int w, int h,
                               std::vector<std::uint8_t>& y,
                               std::vector<std::uint8_t>& u,
                               std::vector<std::uint8_t>& v,
                               multiref::CompoundModeParams& chosen,
                               std::int64_t& chosen_sse, std::string& error);

  // §7.13.3.15 save_subpu for the §7.13.3.8 OPFL / §7.13.3.1 refine-MV
  // subblocks.  The two tools split luma and chroma independently, so each
  // plane contributes its own rect at the same plane coordinates the predictor
  // used.  Called only when --video-df-sub-pu is on.
    void collect_sub_pus(SbSink& sink, int x0, int y0, int w, int h,
                       bool optflow, bool refine);

    bool predict_compound_optflow(SbSink& sink, const inter::mc::Mv mvs[2],
                                int x0, int y0, int w, int h,
                                std::string& error);

  // §7.13.3.3-.6 refine-MV, writing sink.compound_y/u_/v_ with EIGHTTAP_SHARP.
  //
  // §7.13.3.1 predicts a refine-MV block in 16x16 (plane) sub-blocks:
  //   tw = Min(w, 16 >> subX), th = Min(h, 16 >> subY)
  // and invokes predict_refinemv() per sub-block.  The §7.13.3.6 search, the
  // §7.13.3.4 ref area and the prediction are therefore all sub-block sized;
  // only the luma pass searches, and chroma reuses the luma RefineMvs entry
  // for the colocated sub-block (predict_refinemv, plane > 0).
    bool predict_compound_refinemv(SbSink& sink, const inter::mc::Mv mvs[2],
                                 int x0, int y0, int w, int h,
                                 std::string& error);

#endif  // AV2ENC_VIDEO_FRAMES

  // --video-key-intra-txpart: keyframe intra luma TX-partition RD search.
  //
  // §5.20.7.24/§5.20.6.3 predict per *transform block*: with TX_MODE_SELECT and
  // an intra partition, every part is predicted from the neighbours that the
  // previously coded parts of the same coding block have just written, then
  // transformed, quantised and reconstructed before the next part predicts.
  // `search_luma_tx(..., search_partition=false)` is single-predictor by
  // contract, so it cannot be used for the whole block.  This routine runs the
  // equivalent chain per part and compares NONE against the legal SPLIT/HORZ/
  // VERT options by coded SSE + lambda * (per-part TX rate + partition bits).
  //
  // `block_pred` is the whole-block prediction the mode search already chose
  // (NONE reuses it verbatim, which keeps the unpartitioned winner identical to
  // the flag-off stream).  Returns false only on an internal failure.
    bool encode_intra_luma_tx_partition(
      int r, int c, int w, int h, BlockSize size,
      const std::vector<std::uint8_t>& block_pred, int y_mode,
      int angle_delta, int mrl_index, bool mrl_sec_index,
      bool above_smooth, bool left_smooth, int mask_row, SbSink::State& st,
      TxPlan* plan, std::string& error, int live_txb = 0, int live_dc = 0,
      bool live_ctx_on = false);

    bool encode_leaf(SbSink& sink, const PartitionNode& node, std::size_t index,
                   std::string& error, int tree_type = 0,
                   bool leaf_has_chroma = true, int chroma_mi_row = -1,
                   int chroma_mi_col = -1, int chroma_bsize = -1);

  // ---------------------------------------------------------------------
  // AV2ENC_RD_DRIVER: sequential key-frame layout decision.
  //
  // The historical key tree is solved for the whole picture before any leaf
  // is coded, so every node is priced by a variance proxy.  The driver keeps
  // the same legal tree (4x4..64x64) but decides one superblock at a time:
  // every candidate layout is walked in decode order in the encoder's current
  // reconstruction, its leaves are encoded for real, and the next candidate
  // starts from the same snapshot.  The winner is then committed through the
  // normal collect_node()/encode_leaf() path, which is what updates the
  // reconstruction and §6.19.2.3 BlockDecoded for the next superblock.
  //
  // A candidate that cannot be measured (leaf encode failure, invalid tree)
  // is not scored; if no candidate is finite the reference tree is returned.
  // ---------------------------------------------------------------------
    bool seq_driver_measure_leaf(const PartitionNode& node,
                                 const ChromaInfo& chroma, SbSink& sink,
                                 double* cost, std::string& error);
    bool seq_driver_decide_sb(std::size_t sb_index,
                            const PartitionNode& reference,
                            PartitionNode* out, std::string& error);
    // AV2ENC_PIPE_BUDGET (docs/k52-budget-pipe.md): the budget-closed
    // per-superblock allocation walk.  Its own frame loop (one superblock at a
    // time), its own leaf loop (every qindex on the grid over the
    // superblock's own expanded tree) and its own decision order (greedy
    // marginal gain per byte over one frame budget).  Commits through
    // collect_node() like every other sequential walk; default off.
    bool pipe_budget_decide_frame(
        std::vector<partition::PartitionNode>& base_superblocks,
        std::string& error);

  // AV2ENC_RD_TAIL_FLOOR state, live only during one armed tail picture's
  // sequential superblock walk.  `active` arms the collect_node() hook,
  // `trial` suppresses it for the candidate encodes, and the per-leaf
  // inter-syntax bits are stashed by encode_leaf() from the live probe state
  // it is about to commit (sink-local block index).
    bool tail_floor_active_ = false;
    bool tail_floor_trial_ = false;
    const PartitionNode* tail_floor_sb_root_ = nullptr;
    std::vector<double> tail_floor_leaf_bits_;
    std::vector<char> tail_floor_leaf_bits_ok_;

  // The J of the superblock `seq_driver_decide_sb` decided last.  The
  // process-wide counter is reset by nested probe encodes, so the budget walk
  // reads this instead (docs/k52-budget-pipe.md).
    double seq_last_sb_cost_ = 0.0;

  // AV2ENC_PART_DIAG state, live only during one armed picture's sequential
  // walk.  `active` arms both the collect_node() hook and the per-leaf
  // inter-syntax stash the exact trial cost needs; `trial` suppresses the
  // hook inside candidate encodes.
    bool part_diag_active_ = false;
    bool part_diag_trial_ = false;
    const PartitionNode* part_diag_sb_root_ = nullptr;

  // AV2ENC_RD_JOINT_P: measured joint partition + TX/levels search on the P
  // chain.  `active` arms the sequential walk for the picture and marks the
  // leaf the coder writes so the inter-syntax price is stashed for the trial
  // cost.  `off` is the TIP-output quality probe's opt-out.
    bool joint_p_active_ = false;
    bool joint_plan_off_ = false;
    bool joint_p_measure_leaf(const PartitionNode& node,
                              const ChromaInfo& inherited, SbSink& sink,
                              double lambda, double* cost,
                              std::string& error);
    bool joint_p_decide_sb(std::size_t sb_index,
                           const PartitionNode& reference, PartitionNode* out,
                           std::string& error);

  // AV2ENC_RD_JOINT_BATCH: the two-pass port of the same solve.
  //
  //   pass 1  every superblock's candidate layout is searched in parallel on
  //           a private trial clone seeded with the frame-start decision state
  //           (reconstruction, exact-syntax probe, §8.2 CDF image).  Nothing
  //           is committed: a clone's only output is a layout.
  //   pass 2  the candidate layouts are re-priced exactly in raster order
  //           against the live state, the reference layout is priced with the
  //           same objective, and the better one is committed through the
  //           normal collect_node() walk.
  //
  // The stream is therefore still produced entirely by the serial raster
  // commit, and the reference guard is now evaluated on live prices, so a
  // superblock can never commit a layout the exact objective prefers less
  // than its reference.
    bool joint_p_batch_frame(std::vector<PartitionNode>& superblocks,
                             std::string& error);
  // The serial objective (D + lambda * R) of one whole layout, measured in the
  // live state in pre-order.  This is `joint_p_decide_sb`'s measure_ref walk,
  // factored out so pass 2 prices the candidate and the reference with the
  // exact same code.
    bool joint_p_measure_tree(const PartitionNode& root, SbSink& sink,
                              double lambda, double* cost, std::string& error);
  // A trial clone: a second encoder over the same source/references with this
  // one's frame-start decision state copied in.  Used only by pass 1.
    std::unique_ptr<InterFrameEncoder> joint_p_trial_clone() const;
    void joint_p_copy_frame_state(const InterFrameEncoder& src);
  // The surface one candidate encode mutates, captured and restored around a
  // measurement.  Shared by the serial walk and pass 2 so both price a layout
  // from exactly the same starting state.
    struct JointTrialSnap {
      std::array<std::vector<std::uint8_t>, 3> samples;
      std::array<int, 3> stride{};
      std::array<int, 3> width{};
      std::array<int, 3> height{};
      std::vector<std::uint8_t> joint;
      std::vector<std::uint8_t> uv;
      std::vector<inter::mc::MiInfo> mv;
      SbSink::State state;
      std::size_t blocks = 0;
      std::size_t levels = 0;
      std::size_t palettes = 0;
      std::size_t sub_pus = 0;
      std::size_t leaf_dump = 0;
      std::size_t leaf_costs = 0;
      std::size_t leaf_bits = 0;
      std::size_t leaf_bits_ok = 0;
      rate_hybrid::SyntaxProbe probe;
      rate_hybrid::SyntaxProbe rate_probe;
    };
    JointTrialSnap joint_p_capture(const SbSink& sink) const;
    void joint_p_apply(SbSink& sink, const JointTrialSnap& snap);
  // Structural tree identity (mi_row/mi_col/size/partition/chroma/children).
    static bool joint_p_same_layout(const PartitionNode& a,
                                    const PartitionNode& b);

  // ---------------------------------------------------------------------
  // §5.20.7.10 / §5.20.7.6 block-level TIP reference
  // (--video-tip-block-ref).  The tail P of the TIP schedule is signalled
  // TipFrameMode = TIP_FRAME_AS_REF; its last coded inter leaf is coded as a
  // single-list TIP block predicted from the frame's own §7.10 field.
  // ---------------------------------------------------------------------
    bool tip_block_ref_active() const;

  // §5.20.7.10 get_tip_mv() per prediction unit: the §7.10 field MV at the
  // unit's top-left 8x8 cell, projected to each TIP reference and added to the
  // block MV.
    temporal::MvPair tip_unit_mvs(int unit_mi_row, int unit_mi_col, inter::mc::Mv block_mv,
                                const temporal::TipConfig& tip) const;

  // Predicts one TIP leaf into recon_[] and returns the per-unit list MVs (so
  // the caller can build the §7.22 field).  `unit_mvs` is indexed
  // [unit_row][unit_col][list] over the block's unit grid.
    bool predict_tip_leaf(int x0, int y0, int w, int h, int unit, inter::mc::Mv block_mv,
                        const temporal::TipConfig& tip, std::vector<temporal::MvPair>* unit_mvs,
                        std::string& error);

  // §5.18.2 TIP parameters of this frame for the block form.
    temporal::TipConfig tip_block_config() const;

  // §5.20.7.10 stack check for a TIP-ref leaf: the merge builder must
  // reproduce the decoder's stack exactly, otherwise the leaf cannot be coded
  // as TIP.
    bool tip_ref_stack_ok(int r, int c, int w4, int h4, std::string& error) const;

  // Locates the leaf the carrier will convert: the last coded inter leaf,
  // which every later block must be intra (an intra block never builds a
  // §7.12.2 stack, so no §7.12.2.15/.17 TIP candidate is needed).  Runs on the
  // probe pass; the leaf's block MV becomes the coded MVD.
    bool locate_tip_leaf(std::string& error);

  // Second-pass leaf coder for the target ring: predicts and reconstructs the
  // TIP leaf in decode order, so every later block (intra only) sees the same
  // reference samples as the decoder.
    bool encode_tip_leaf(SbSink& sink, const PartitionNode& node,
                       std::size_t index, std::string& error);

  // Runs the probe pass, locates the TIP leaf, then re-decides the frame with
  // the target leaf forced to TIP.  The second pass is required for recon
  // coherence: blocks decoded after the TIP leaf (intra only) predict from its
  // reconstructed samples, so converting a leaf after the fact would leave
  // them coded against different neighbours than the decoder sees.
    bool apply_tip_block_ref(std::string& error);

    void update_mv_grid(int r, int c, int w4, int h4, int is_inter,
                      int ref_index, inter::mc::Mv mv,
                      int ref_index1 = merge::kRefNone,
                      const inter::mc::Mv* mv1 = nullptr,
                      bool warp0 = false, bool warp1 = false,
                      bool is_wedge = false, int wedge_index = 0,
                      int wedge_sign = 0, int bsize = 0,
                      int cwp_index = 0);

  const VideoInputFrame& source_;
  // AV2ENC_PORT_HORIZON: the source pictures after this one, so the skipped/
  // residual leaf cost can include the error the choice installs in the next K
  // real source frames.  Empty by default and unused when the gate is off.
  std::span<const VideoInputFrame> horizon_future_;
  std::vector<inter::mc::RefFrame> references_;
  std::array<int, 7> frame_distance_{};
  int num_total_refs_ = 1;
  const VideoOptions& options_;
  bool trace_ = false;
  bool keyframe_ = false;
  // AV2ENC_SKIP_DRIFT_ILLUM: this frame's source illumination change against
  // the previous source frame (|d mean| + |d std| in LSB) and whether it is
  // large enough to charge every zero-MV skip.  Both are inert when the gate
  // is off.
  double illum_delta_ = 0.0;
  bool illum_active_ = false;
  // AV2ENC_SKIP_DRIFT_REFDEF: measured coding loss (dB) of the primary
  // reference against the key anchor, and whether it arms the zero-MV skip
  // surcharge.  Inert when the gate is off.
  double ref_deficit_db_ = 0.0;
  bool ref_deficit_active_ = false;
  // Deterministic batch results for the pure NEWMV search, keyed by
  // (reference index, absolute block origin, block size).  Empty unless
  // precompute_leaf_me() ran for a whole-frame P frame.
  std::unordered_map<std::uint64_t, inter::mc::Mv> me_cache_;
  int tile_x_ = 0;
  int tile_mi_row_ = 0;
  int tile_mi_col_ = 0;
  int tile_y_ = 0;
  int tile_w_ = 0;
  int tile_h_ = 0;
  std::span<const std::uint16_t> cdf_seed_;
  std::vector<std::uint16_t>* cdf_final_ = nullptr;
  const GmFramePlan* gm_plan_ = nullptr;
  // R19 (AV2ENC_RATE_INTERVAL): per-frame scratch copy of the §8.2 CDF image
  // the writer starts from, plus the freshly-walked rate estimate.  Built once
  // per tile (call_once) so the wavefront workers share a read-only state.
  mutable rate_interval::CdfStore rate_interval_cdfs_{};
  mutable std::once_flag rate_interval_once_;
  mutable still::qctx::CoeffCdfSlice eff_cdf_slice_{};
  mutable bool eff_cdf_slice_ready_ = false;
  int quant_step_ = txv::kQuantStep;  // §7.14.2 qlookup(base_q_idx)
  // §7.14.2 per-plane DC/AC steps.  The default keeps all three planes at
  // qlookup(qindex); --video-uv-delta-q resolves U/V from the frame deltas.
  // Frame-level copy for the partition probe; the walk reads the per-sink copy.
  PlaneQuantStep plane_quant_[3] = {};
  bool uv_delta_enabled_ = false;
  dcdeltaq::ResolvedDeltaQ resolved_delta_q_{};
  int current_qindex_ = 0;  // frame-level §5.20.1 CurrentQIndex (probe/costs)
#if defined(AV2ENC_VIDEO_DELTAQ)
  // §5.18.7.1/.8 segmentation + delta-Q planning for this tile.
  bool seg_enabled_ = false;
  bool delta_q_enabled_ = false;
  rc::SegmentationParams seg_params_{};
  rc::DeltaQParams dq_params_{};
  deltaq::FrameSegmentation enc_seg_{};
  deltaq::SegmentGrid enc_seg_grid_{};
  std::vector<std::uint8_t> sb_segment_;   // variance split: 1 = low variance
  std::vector<int> sb_target_q_;           // delta-Q target per superblock
  std::vector<int> sb_qindex_;             // effective qindex per superblock
  std::vector<rc::DeltaQPlanStep> sb_delta_steps_;  // raster order
  bool primary_ref_is_none_ = true;
  int skip_seg_id_ = -1;
  int globalmv_seg_id_ = -1;
  std::vector<std::uint8_t> prev_segment_ids_;
#endif
  qm::QmFrameState qm_state_;
  bool qm_enabled_ = false;
  int width_ = 0;
  int height_ = 0;
  int frame_mi_cols_ = 0;
  int frame_mi_rows_ = 0;
  int chroma_width_ = 0;
  int chroma_height_ = 0;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  // SB-aligned reconstruction height for --video-edge-inter (0 = disabled).
  // The visible picture is still mi_rows_*4 rows; the extra rows exist only so
  // a bottom leaf and its neighbours reconstruct the full coded block.
  int edge_pad_rows_ = 0;
  FrameGeometry geometry_;
  Plane recon_[3];
  Plane compound_pred_[3];
  post::VideoFilterPlan filter_plan_;
  // Per-superblock sinks for the current decision pass (raster order).  The
  // prediction scratch that used to live here is owned by SbSink so a worker
  // can own a unit (docs/wavefront-plan.md §6.1).
  std::vector<SbSink> sb_sinks_;
  // §7.13.3.15 sub-PU rectangles from OPFL/refine-MV prediction.  Only used
  // when --video-df-sub-pu is on; ordinary whole-PU leaves take the normative
  // early return and leave the deblock grid's Max_Tx_Size_Rect values.
  std::vector<post::VideoSubPuRect> sub_pus_;
  inter::mc::RefFrame ref_frame_;
  std::vector<inter::mc::MiInfo> mv_grid_;
  // R09 (AV2ENC_RATE_HYBRID_S): exact inter-syntax CDF probe.  It owns a copy
  // of the inter CDF rows seeded from `cdf_seed_` and the neighbour grids the
  // writer derives its contexts from, advanced in coded order.  Empty and
  // unused unless the gate (or the measure switch) is on.
  rate_hybrid::SyntaxProbe rate_probe_;
  bool rate_probe_ready_ = false;
  // AV2ENC_PIPE_PERC: source-driven 8x8 luma weights for the current frame,
  // and the exact-syntax probe advanced in coded order.  Both are inert
  // unless the gate is on.
  pipe_perc::FrameWeights pipe_weights_;
  rate_hybrid::SyntaxProbe pipe_probe_;
  bool pipe_probe_ready_ = false;
  // AV2ENC_EFF_PERC: per-pixel multi-scale weights.  Inert when the gate is
  // off; the probe above is shared with pipe-perc because both gates use the
  // same exact-syntax front end.
  eff_perc::FrameWeights eff_weights_;
  // AV2ENC_PIPE_BA: the Butteraugli-shaped frame mask (Q8, already blended
  // with the tool weight).  Built once per picture in plan_pass(), inert and
  // neutral when the gate is off.
  pipe_ba::FrameMask pipe_ba_mask_;
  std::vector<std::uint8_t> intra_joint_;
  // §7.13.2.15/.16 UV smooth flags per chroma 4x4 cell (UVMode is
  // SMOOTH/SMOOTH_V/SMOOTH_H and not CfL).  Inert unless the
  // AV2ENC_RD_UVFULL chroma search codes a non-DC chroma mode.
  std::vector<std::uint8_t> uv_smooth_;
  // §5.20.5.3/.4 IBC + §7.13.3.26 morph state (intra keyframe only).
  bool ibc_enabled_ = false;
  bool plan_morph_ = false;
  ibc::IbcBufferState ibc_buffer_;
  ibcs::BvNeighbourMap ibc_neighbours_;
  ibc::IbcConfig ibc_config_{};
  // §5.20.8 palette state (intra keyframe only).  The state is the same MI
  // cache source the writer's PaletteSyntaxWriter maintains, and the results
  // are handed to the writer through FramePlan::palettes.
  bool palette_enabled_ = false;
  palette::PaletteState palette_state_;
  std::vector<palette::PaletteResult> palettes_;
  std::vector<syntax::BlockSyntax> blocks_;
  // Stage-0 AV2ENC_BLOCKDUMP records, one per leaf of the last decision pass.
  // Empty (and never appended) when the variable is unset.
  std::vector<LeafDumpRecord> leaf_dump_;
  // §5.4.3 enable_sdp: the decided luma/chroma trees for the intra keyframe.
  sdpp::FrameDecision sdp_frame_;
  // §8.3.2 SDP plane grids, maintained in decode order while the trees are
  // collected.  Contexts for each node's partition symbols come from these.
  sdpp::MiSizeGrid sdp_luma_sizes_;
  sdpp::MiSizeGrid sdp_chroma_sizes_;
  std::array<int, 7> ref_order_hints_{};
  TemporalFrameContext temporal_ctx_;
  int census_frame_ = -1;
  int display_ordinal_ = -1;
  std::vector<std::int32_t> levels_;
  std::uint32_t inter_blocks_ = 0;
  std::uint32_t intra_blocks_ = 0;
  int sb_size_ = 64;
  // AV2ENC_RD_DRIVER_SBALLOC: running frame-wide source activity used to
  // normalise the per-superblock key rate allocation.  Reset each decision
  // pass in plan_pass(); inert when the gate is off.
  // AV2ENC_TX_LIVE_CTX: per-4x4 luma "any non-zero coefficient" grid for the
  // coded above/left neighbours, so the TX search can price each candidate in
  // the §8.3.2 txb_skip / DC-sign context the writer will actually use.
  // Derived only when the gate is on; the default path never allocates it.
  bool live_ctx_enabled_ = false;
  std::vector<std::uint8_t> live_ctx_above_;
  std::vector<std::uint8_t> live_ctx_left_;
  std::vector<std::uint8_t> live_ctx_dc_above_;
  std::vector<std::uint8_t> live_ctx_dc_left_;
  // §8.3.2 txb_skip / DC-sign context for the luma block at (mi_row, mi_col)
  // of size w4 x h4, from the coded neighbours.
  void live_ctx_for(int mi_row, int mi_col, int w4, int h4, int block_w4,
                    int block_h4, int* txb_skip, int* dc_sign) const;
  void live_ctx_commit(int mi_row, int mi_col, int w4, int h4, bool any_nz,
                       bool dc_nz);
  double sb_alloc_frame_activity_ = 0.0;
  long long sb_alloc_sbs_ = 0;
  // Per-superblock lambda scale, set by seq_driver_decide_sb and read by the
  // leaf cost and the tree prune it drives.  1.0 when the gate is off.
  double sb_alloc_scale_ = 1.0;
  mutable int temporal_added_ = 0;
  int temporal_selected_ = 0;
  // §5.4.6 same-reference compound leaves selected (gate evidence).
  int same_ref_blocks_ = 0;
  // §5.4.6 enable_refmvbank state and instrumentation.
  temporal::RefMvBank refmv_bank_;
  mutable int refmvbank_added_ = 0;
  int refmvbank_selected_ = 0;
  mutable int mv_traj_derived_ = 0;
  // §5.20.7.6 MV-resolution decisions on coded NEWMV leaves (gate evidence).
  int amvd_selected_ = 0;
  int flex_precision_selected_ = 0;
  int derived_sign_selected_ = 0;
  int tip_blocks_ = 0;
  // §5.20.7.10 block-form target (second decision pass).
  int tip_leaf_r_ = -1;
  int tip_leaf_c_ = -1;
  inter::mc::Mv tip_leaf_mv_{};
  bool tip_leaf_armed_ = false;
};

inline VideoReconFrame to_recon_frame(const Plane& y, const Plane& u, const Plane& v,
                            std::uint32_t width, std::uint32_t height) {
  VideoReconFrame out;
  out.width = width;
  out.height = height;
  const std::uint32_t cw = (width + 1U) / 2U;
  const std::uint32_t ch = (height + 1U) / 2U;
  out.y.resize(static_cast<std::size_t>(width) * height);
  out.u.resize(static_cast<std::size_t>(cw) * ch);
  out.v.resize(static_cast<std::size_t>(cw) * ch);
  for (std::uint32_t row = 0; row < height; ++row) {
    std::memcpy(out.y.data() + static_cast<std::size_t>(row) * width,
                y.samples.data() +
                    static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(y.stride),
                width);
  }
  for (std::uint32_t row = 0; row < ch; ++row) {
    std::memcpy(out.u.data() + static_cast<std::size_t>(row) * cw,
                u.samples.data() +
                    static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(u.stride),
                cw);
    std::memcpy(out.v.data() + static_cast<std::size_t>(row) * cw,
                v.samples.data() +
                    static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(v.stride),
                cw);
  }
  return out;
}

// ---------------------------------------------------------------------------
// §7.22 saved motion field for an ordinary reference slot.
//
// The decoder's `motion_field_motion_vector_storage()` stores the 8x8 motion
// field of every inter frame when `enable_ref_frame_mvs = 1`; a later frame
// with `use_ref_frame_mvs = 1` projects those fields in §7.9 and consumes the
// result in §7.12.2.7/.8.  This is the same field the TIP path already built
// for its slots, factored out for the P / B / temporal-layer loops.
// ---------------------------------------------------------------------------
struct SlotField {
  bool valid = false;
  int order_hint = 0;
  int w8 = 0;
  int h8 = 0;
  int num_saved_refs = 0;
  // True when this frame's grid contains a CWP-weighted compound block.  The
  // TIP field guard uses it to fall back to the coded B instead of relying on
  // a field the pinned decoder derives differently.
  bool has_cwp_compound = false;
  int saved_order_hint[7] = {temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint};
  std::vector<std::int8_t> field_ref;
  std::vector<std::int16_t> field_mv;

  temporal::SavedMvField saved() const {
    temporal::SavedMvField out;
    out.w8 = w8;
    out.h8 = h8;
    out.ref = field_ref.data();
    out.mv = field_mv.data();
    return out;
  }
};

// Allocates an empty §7.22 field (intra key / no coded inter blocks).
inline void init_slot_field(SlotField* slot, int mi_cols, int mi_rows) {
  slot->w8 = mi_cols / 2;
  slot->h8 = mi_rows / 2;
  slot->num_saved_refs = 0;
  slot->has_cwp_compound = false;
  for (int i = 0; i < 7; ++i) {
    slot->saved_order_hint[i] = temporal::kRestrictedOrderHint;
  }
  slot->field_ref.assign(static_cast<std::size_t>(slot->w8) *
                             static_cast<std::size_t>(slot->h8) * 2U,
                         static_cast<std::int8_t>(temporal::kRefNone));
  slot->field_mv.assign(static_cast<std::size_t>(slot->w8) *
                            static_cast<std::size_t>(slot->h8) * 4U,
                        0);
}

// §7.22 store from the encoder's per-4x4 mode/MV grid.
//
// `view` must already be sized to the grid (w8 = mi_cols>>1, h8 = mi_rows>>1)
// and zero/`kRefNone` initialised; the caller owns the arrays.  Two pieces of
// block state cannot be recovered from the per-4x4 ref/MV pair:
//
//   * a GLOBALMV block with a non-translation §5.18.9 model stores the model
//     projected at *each* 8x8 centre (§7.22 per-sub-block warped MV), not the
//     block-centre MV the grid carries;
//   * a §7.13.3.27 wedge compound block stores a masked allow-list, so one
//     list can be dropped for an 8x8 that lies almost entirely on the other
//     side of the wedge.
//
// Both are recorded per block in the grid cells (MiInfo::warp / wedge_*).
// Gate evidence for the §7.22 store: how many 8x8 cells needed the per-8x8
// warp projection, and how many wedge cells lost a list to the allow-list.
struct RefMvsStoreStats {
  int warp_cells = 0;
  int wedge_cells = 0;
  int wedge_dropped = 0;
};

inline bool store_field_grid(const InterFrameEncoder& encoder, int order_hint,
                      const int* ref_order_hint, int num_refs,
                      temporal::RefMvField* view, std::string& error,
                      RefMvsStoreStats* stats = nullptr) {
  const std::vector<inter::mc::MiInfo>& grid = encoder.mv_grid();
  const int mi_cols = encoder.mv_mi_cols();
  const int mi_rows = encoder.mv_mi_rows();
  if (view == nullptr || view->ref == nullptr || view->mv == nullptr ||
      view->w8 != mi_cols / 2 || view->h8 != mi_rows / 2) {
    error = "§7.22 field extent does not match the coded MI grid";
    return false;
  }
  const GmFramePlan* gm = encoder.gm_plan();
  std::vector<std::uint8_t> wedge_mask;
  int mask_row = -1;
  int mask_col = -1;
  for (int y8 = 0; y8 < view->h8; ++y8) {
    for (int x8 = 0; x8 < view->w8; ++x8) {
      const std::size_t index =
          static_cast<std::size_t>(y8 * 2) *
              static_cast<std::size_t>(mi_cols) +
          static_cast<std::size_t>(x8 * 2);
      const inter::mc::MiInfo& mi = grid[index];
      const int block_mi_row = mi.row_base;
      const int block_mi_col = mi.col_base;
      const int i8 = y8 - block_mi_row / 2;   // 8x8 row inside the block
      const int j8 = x8 - block_mi_col / 2;   // 8x8 column inside the block
      temporal::MotionFieldStoreEntry entry;
      for (int list = 0; list < 2; ++list) {
        const int ref = mi.ref[list];
        entry.refs[list] = (mi.is_inter != 0 && ref >= 0 && ref < num_refs)
                               ? ref
                               : temporal::kRefNone;
        entry.ref_order_hint[list] =
            entry.refs[list] >= 0
                ? ref_order_hint[entry.refs[list]]
                : temporal::kRestrictedOrderHint;
        inter::mc::Mv mv{mi.mv[list][0], mi.mv[list][1]};
        if (entry.refs[list] >= 0 && mi.warp[list] != 0 && gm != nullptr &&
            entry.refs[list] < gm->num_refs) {
          const inter::gme::GmParams& params =
              gm->params[static_cast<std::size_t>(entry.refs[list])];
          if (gm_model_uses_warp(params)) {
            gm_sub_block_warp_mv(params, block_mi_col * 4 + j8 * 8,
                                 block_mi_row * 4 + i8 * 8, &mv);
            if (stats != nullptr) ++stats->warp_cells;
          }
        }
        entry.mv[list] = temporal::Mv{mv.y, mv.x};
      }
      entry.order_hint = order_hint;
      entry.i8 = i8;
      entry.j8 = j8;
      if (mi.is_wedge != 0 && mi.bsize > 0) {
        if (mask_row != block_mi_row || mask_col != block_mi_col) {
          const int bw = mi.w4 * 4;
          const int bh = mi.h4 * 4;
          wedge_mask.resize(static_cast<std::size_t>(bw) *
                            static_cast<std::size_t>(bh));
          for (int i = 0; i < bh; ++i) {
            for (int j = 0; j < bw; ++j) {
              wedge_mask[static_cast<std::size_t>(i) *
                             static_cast<std::size_t>(bw) +
                         static_cast<std::size_t>(j)] =
                  static_cast<std::uint8_t>(inter::adv::wedge_mask_value(
                      mi.bsize, mi.wedge_sign, mi.wedge_index, i, j));
            }
          }
          mask_row = block_mi_row;
          mask_col = block_mi_col;
        }
        entry.is_wedge = true;
        entry.mask = wedge_mask.data();
        entry.mask_stride = mi.w4 * 4;
        bool dropped = false;
        entry.allow_dropped = &dropped;
        if (stats != nullptr) ++stats->wedge_cells;
        if (temporal::store_motion_field_entry(entry, view, y8, x8) !=
            temporal::TemporalStatus::kOk) {
          error = "§7.22 motion-field store failed";
          return false;
        }
        if (dropped && stats != nullptr) ++stats->wedge_dropped;
        continue;
      }
      if (temporal::store_motion_field_entry(entry, view, y8, x8) !=
          temporal::TemporalStatus::kOk) {
        error = "§7.22 motion-field store failed";
        return false;
      }
    }
  }
  return true;
}

// Stores one coded frame's §7.22 field from its per-4x4 mode/MV grid.
inline bool store_slot_field(const InterFrameEncoder& encoder, int order_hint,
                      const int* ref_order_hint, int num_refs,
                      SlotField* slot, std::string& error,
                      RefMvsStoreStats* stats = nullptr) {
  init_slot_field(slot, encoder.mv_mi_cols(), encoder.mv_mi_rows());
  for (const inter::mc::MiInfo& mi : encoder.mv_grid()) {
    if (mi.is_inter != 0 && mi.ref[0] != mi.ref[1] && mi.cwp_index != 0) {
      slot->has_cwp_compound = true;
      break;
    }
  }
  temporal::RefMvField view;
  view.w8 = slot->w8;
  view.h8 = slot->h8;
  view.ref = slot->field_ref.data();
  view.mv = slot->field_mv.data();
  return store_field_grid(encoder, order_hint, ref_order_hint, num_refs, &view,
                          error, stats);
}

// §7.9 estimation result plus the arrays it borrows.
struct MotionFieldFrame {
  int mi_cols = 0;
  int mi_rows = 0;
  std::vector<std::int32_t> mv;
  std::vector<std::int32_t> offset;
  std::vector<std::uint8_t> valid;
  // §7.9.2 trajectory field (only allocated when enable_mv_traj is set).
  std::vector<std::int32_t> traj_mv;
  std::vector<std::uint8_t> traj_valid;
  std::vector<std::int16_t> traj_pos;
  std::vector<std::uint8_t> traj_pos_valid;
  temporal::TrajectoryField traj;
  temporal::MotionField field;
  temporal::MotionFieldConfig config;
  temporal::MotionFieldResult result;
  bool ready = false;
};

// §7.9 input view over one saved slot field.  SlotField and TipSlotState both
// own the same §7.22 payload, so estimation takes this view instead of either
// storage type.
struct SavedFieldRef {
  int num_saved_refs = 0;
  const int* saved_order_hint = nullptr;
  temporal::SavedMvField field;
};

inline SavedFieldRef make_saved_ref(const SlotField& slot) {
  SavedFieldRef out;
  out.num_saved_refs = slot.num_saved_refs;
  out.saved_order_hint = slot.saved_order_hint;
  out.field = slot.saved();
  return out;
}

// Runs §7.9 for the current frame from its reference slots' saved fields and
// fills closest_past/closest_future from the §7.8 past/future split.
inline bool estimate_motion_field(const SavedFieldRef* ref_fields,
                           const int* ref_slots, const int* ref_order_hint,
                           const std::uint8_t* ref_is_key, bool ref_valid,
                           int num_total_refs, int order_hint, int frame_width,
                           int frame_height, int mi_cols, int mi_rows,
                           int proj_step, bool enable_mv_traj,
                           MotionFieldFrame* out, std::string& error,
                           bool enable_tip = false) {
  out->mi_cols = mi_cols;
  out->mi_rows = mi_rows;
  out->ready = false;
  const std::size_t cells =
      static_cast<std::size_t>(mi_cols / 2) *
      static_cast<std::size_t>(mi_rows / 2);
  out->mv.assign(cells * 2, 0);
  out->offset.assign(cells, 0);
  out->valid.assign(cells, 0);
  out->field.w8 = mi_cols / 2;
  out->field.h8 = mi_rows / 2;
  out->field.mv = out->mv.data();
  out->field.valid = out->valid.data();
  out->field.offset = out->offset.data();
  out->traj = temporal::TrajectoryField{};
  out->traj.w8 = out->field.w8;
  out->traj.h8 = out->field.h8;
  out->traj.num_refs = num_total_refs > 0 ? num_total_refs : 1;
  const std::size_t traj_cells =
      static_cast<std::size_t>(out->traj.num_refs) * cells;
  if (enable_mv_traj) {
    out->traj_mv.assign(traj_cells * 2, 0);
    out->traj_valid.assign(traj_cells, 0);
    out->traj_pos.assign(traj_cells * 3 * 2, 0);
    out->traj_pos_valid.assign(traj_cells * 3, 0);
    out->traj.mv = out->traj_mv.data();
    out->traj.valid = out->traj_valid.data();
    out->traj.pos = out->traj_pos.data();
    out->traj.pos_valid = out->traj_pos_valid.data();
  }

  temporal::MotionFieldConfig& cfg = out->config;
  cfg = temporal::MotionFieldConfig{};
  cfg.mi_cols = mi_cols;
  cfg.mi_rows = mi_rows;
  cfg.proj_step = proj_step;
  cfg.sb_64x64 = proj_step == 1;
  cfg.enable_tip = enable_tip;
  cfg.enable_mv_traj = enable_mv_traj;
  cfg.reduced_ref_frame_mvs_mode = false;
  cfg.use_bru = false;
  cfg.bru_ref = -1;
  cfg.order_hint = order_hint;
  cfg.num_total_refs = num_total_refs;
  cfg.num_past_refs = 0;
  cfg.num_future_refs = 0;
  cfg.frame_width = frame_width;
  cfg.frame_height = frame_height;

  int closest_past = -1;
  int closest_future = -1;
  int closest_past_hint = 0;
  int closest_future_hint = 0;
  for (int rf = 0; rf < num_total_refs; ++rf) {
    cfg.ref_order_hint[rf] = ref_order_hint[rf];
    cfg.refs[rf].slot = ref_slots[rf];
    cfg.refs[rf].type = ref_is_key[rf] ? temporal::FrameType::kKey
                                       : temporal::FrameType::kInter;
    cfg.refs[rf].width = frame_width;
    cfg.refs[rf].height = frame_height;
    cfg.refs[rf].order_hint = ref_order_hint[rf];
    cfg.refs[rf].num_saved_refs = ref_fields[rf].num_saved_refs;
    for (int k = 0; k < ref_fields[rf].num_saved_refs; ++k) {
      cfg.refs[rf].saved_order_hint[k] = ref_fields[rf].saved_order_hint[k];
    }
    cfg.saved[rf] = ref_fields[rf].field;
    const int dist = order_hint - ref_order_hint[rf];
    if (dist > 0) {
      ++cfg.num_past_refs;
      if (closest_past < 0 || ref_order_hint[rf] > closest_past_hint) {
        closest_past = rf;
        closest_past_hint = ref_order_hint[rf];
      }
    } else if (dist < 0) {
      ++cfg.num_future_refs;
      if (closest_future < 0 || ref_order_hint[rf] < closest_future_hint) {
        closest_future = rf;
        closest_future_hint = ref_order_hint[rf];
      }
    }
  }
  if (!ref_valid) {
    // No valid reference slot: the decoder leaves the field empty.
    out->result = temporal::MotionFieldResult{};
    out->ready = true;
    return true;
  }
  const temporal::TemporalStatus status = temporal::motion_field_estimation(
      cfg, &out->field, enable_mv_traj ? &out->traj : nullptr,
      &out->result);
  if (status != temporal::TemporalStatus::kOk) {
    error = "§7.9 motion-field estimation failed";
    return false;
  }
  // §5.18.2: fill_tpl_mvs_sample_gap() runs when TIP is disabled.
  if (temporal::fill_motion_field_sample_gap(cfg, &out->field) !=
      temporal::TemporalStatus::kOk) {
    error = "§7.10.5 motion-field gap fill failed";
    return false;
  }
  out->result.closest_past = closest_past;
  out->result.closest_future = closest_future;
  out->ready = true;
  return true;
}



namespace vf_detail {
#if defined(AV2ENC_VIDEO_FRAMES)
[[nodiscard]] bool encode_b_sequence(std::span<const VideoInputFrame> frames,
                       const VideoOptions& options, int gop_size,
                       VideoSequenceResult& result, std::string& error);
#endif
[[nodiscard]] bool encode_temporal_sequence(std::span<const VideoInputFrame> frames,
                              const VideoOptions& options,
                              VideoSequenceResult& result,
                              std::string& error);
[[nodiscard]] bool encode_tip_sequence(std::span<const VideoInputFrame> frames,
                         const VideoOptions& options,
                         VideoSequenceResult& result,
                         std::string& error);
}  // namespace vf_detail
}  // namespace av2enc::video
