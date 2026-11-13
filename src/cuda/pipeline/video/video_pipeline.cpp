// CUDA M2 P-frame orchestration.
//
// This is a decision-identical mirror of src/encoder/m2/video_frame.cpp.  The
// partition tree, MV-stack derivation, candidate ordering and tie-breaking are
// the same code paths (the CPU partition/MC decision helpers); the pixel and
// transform stages are delegated to the linked backend.
//
// The module never edits or wraps the CPU encoder: it is a separate consumer of
// the same public components, so the CPU gate keeps its own reference path.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <unordered_map>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "codec/inter/mc/mc.h"
#include "codec/inter/merge/merge.h"
#include "codec/inter/advanced/compound.h"
#include "cuda/pipeline/video/hbd/video_hbd_cuda.h"
#include "pipeline/video/video_gme.h"
#include "pipeline/video/video_gm_types.h"
#include "codec/inter/bframes/bframes.h"
#include "codec/inter/tlayers/tlayers.h"
#include "codec/entropy/avg_cdf/avg_cdf_bank.h"
#include "pipeline/video/video_bawp.h"
#include "pipeline/video/video_headers.h"
#include "pipeline/video/video_syntax.h"
#include "pipeline/video/post/tiled_frame.h"
#include "pipeline/tiles/tile_layout.h"
#include "codec/partition/tree/partition.h"
#include "codec/transform/variants/tx_variants.h"
#include "frame/mfh/mfh.h"
#include "frame/metadata/content_interpretation/content_interpretation.h"

#include "video_backend.h"
#include "video_gpu.h"
#include "video_sequence_internal.h"

namespace av2enc::cuda::video {

using av2enc::video::GmFramePlan;

namespace {
using av2enc::video::VideoConfig;
using av2enc::video::VideoFrameHeader;
using av2enc::video::VideoInputFrame;
using av2enc::video::VideoOptions;
using av2enc::video::VideoReconFrame;
using av2enc::video::VideoSequenceResult;
using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionNode;
using partition::PartitionType;

namespace detail = av2enc::cuda::video::detail;
namespace merge = ::av2enc::video::merge;
namespace syntax = ::av2enc::video::syntax;

// Device pixel hook for the opt-in M2 loop filters: the shared host stage runs
// the decisions and hands each stage's planes to the CUDA kernels.
class BackendFilterHooks final
    : public av2enc::video::post::VideoLoopFilterHooks {
 public:
  explicit BackendFilterHooks(detail::Backend& backend) : backend_(backend) {}

  bool apply_stage(av2enc::video::post::VideoFilterStage stage,
                   const av2enc::video::post::VideoFilterPlan& plan,
                   const av2enc::video::post::VideoLoopFilterDeviceFrame& frame,
                   std::string* error) override {
    return backend_.loop_filter_stage(stage, plan, frame, *error);
  }

 private:
  detail::Backend& backend_;
};

// The CPU M2 path grows optional tools over time (inter residual, DRL
// reordering).  Detect the option members instead of hard-depending on them,
// so this tree builds against both the baseline and the extended VideoOptions.
template <typename T, typename = void>
struct HasEnableInterResidual : std::false_type {};
template <typename T>
struct HasEnableInterResidual<
    T, std::void_t<decltype(std::declval<T&>().enable_inter_residual)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasForceInterResidual : std::false_type {};
template <typename T>
struct HasForceInterResidual<
    T, std::void_t<decltype(std::declval<T&>().force_inter_residual)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasDrlReorder : std::false_type {};
template <typename T>
struct HasDrlReorder<T,
                     std::void_t<decltype(std::declval<T&>().drl_reorder)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasTxSearch : std::false_type {};
template <typename T>
struct HasTxSearch<
    T, std::void_t<decltype(std::declval<const T&>().enable_tx_search)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasTxRdLambda : std::false_type {};
template <typename T>
struct HasTxRdLambda<
    T, std::void_t<decltype(std::declval<const T&>().tx_rd_lambda)>>
    : std::true_type {};

template <typename T>
double opt_tx_rd_lambda(const T& options) {
  if constexpr (HasTxRdLambda<T>::value) {
    return options.tx_rd_lambda;
  } else {
    return 256.0;
  }
}

template <typename T>
bool opt_enable_tx_search(const T& options) {
  if constexpr (HasTxSearch<T>::value) {
    return options.enable_tx_search;
  } else {
    return false;
  }
}

template <typename T, typename = void>
struct HasVideoDeblock : std::false_type {};
template <typename T>
struct HasVideoDeblock<
    T, std::void_t<decltype(std::declval<T&>().enable_deblock)>>
    : std::true_type {};

template <typename T>
bool opt_video_deblock(const T& options) {
  if constexpr (HasVideoDeblock<T>::value) {
    return options.enable_deblock;
  } else {
    return false;
  }
}

template <typename T>
bool opt_enable_inter_residual(const T& options) {
  if constexpr (HasEnableInterResidual<T>::value) {
    return options.enable_inter_residual;
  } else {
    return false;
  }
}

template <typename T, typename = void>
struct HasInterRdLambda : std::false_type {};
template <typename T>
struct HasInterRdLambda<
    T, std::void_t<decltype(std::declval<T&>().inter_rd_lambda)>>
    : std::true_type {};

template <typename T>
double opt_inter_rd_lambda(const T& options) {
  if constexpr (HasInterRdLambda<T>::value) {
    return options.inter_rd_lambda;
  } else {
    return 4.0;
  }
}

template <typename T>
bool opt_force_inter_residual(const T& options) {
  if constexpr (HasForceInterResidual<T>::value) {
    return options.force_inter_residual;
  } else {
    return false;
  }
}

template <typename T, typename = void>
struct HasEnableIntraFull : std::false_type {};
template <typename T>
struct HasEnableIntraFull<
    T, std::void_t<decltype(std::declval<T&>().enable_intra_full)>>
    : std::true_type {};

template <typename T>
bool opt_enable_intra_full(const T& options) {
  if constexpr (HasEnableIntraFull<T>::value) {
    return options.enable_intra_full;
  } else {
    return false;
  }
}

template <typename T, typename = void>
struct HasEnableCfl : std::false_type {};
template <typename T>
struct HasEnableCfl<T,
                    std::void_t<decltype(std::declval<T&>().enable_cfl)>>
    : std::true_type {};

template <typename T>
bool opt_enable_cfl(const T& options) {
  if constexpr (HasEnableCfl<T>::value) {
    return options.enable_cfl;
  } else {
    return false;
  }
}

template <typename T, typename = void>
struct HasEnableMhccp : std::false_type {};
template <typename T>
struct HasEnableMhccp<T,
                      std::void_t<decltype(std::declval<T&>().enable_mhccp)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasEnableQm : std::false_type {};
template <typename T>
struct HasEnableQm<T, std::void_t<decltype(std::declval<T&>().enable_qm)>>
    : std::true_type {};

template <typename T>
bool opt_enable_qm(const T& options) {
  if constexpr (HasEnableQm<T>::value) {
    return options.enable_qm;
  } else {
    return false;
  }
}

template <typename T>
bool opt_enable_mhccp(const T& options) {
  if constexpr (HasEnableMhccp<T>::value) {
    return options.enable_mhccp;
  } else {
    return false;
  }
}

template <typename T>
int opt_drl_reorder(const T& options) {
  if constexpr (HasDrlReorder<T>::value) {
    return options.drl_reorder;
  } else {
    return 0;
  }
}

template <typename C>
void set_config_drl_reorder(C& config, int value) {
  if constexpr (HasDrlReorder<C>::value) {
    config.drl_reorder = static_cast<std::uint8_t>(value);
  }
}

// estimate_tx_bits() from the CPU M2 inter-residual decision.  Deliberately a
// heuristic, but it must be bit-for-bit the same heuristic for parity.
double estimate_tx_bits(int eob, const std::vector<std::int32_t>& levels,
                        bool is_inter_luma) {
  if (eob == 0) return 1.0;
  double bits = 8.0;
  if (is_inter_luma) bits += 4.0;
  int nonzero = 0;
  for (int c = 0; c < eob; ++c) {
    if (levels[static_cast<std::size_t>(c)] != 0) ++nonzero;
  }
  bits += static_cast<double>(nonzero) * 2.5 +
          static_cast<double>(eob) * 0.5;
  return bits;
}

// Luma rate for the skip/residual and intra/inter decisions: the historical
// estimate_tx_bits heuristic applied per transform leaf.  The search's own
// cost model only selects the plan, so enabling the tool does not re-price
// every residual against skip.
double luma_search_bits(const detail::TxSearchPlan& plan, bool is_inter_luma) {
  double bits = 0.0;
  for (const detail::TxSearchPart& part : plan.parts) {
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

bool luma_search_all_zero(const detail::TxSearchPlan& plan) {
  for (const detail::TxSearchPart& part : plan.parts) {
    if (part.eob != 0) return false;
  }
  return true;
}

// Copies one searched luma plan into the syntax node and appends the
// per-part scan-order levels.
void write_luma_search_plan(syntax::BlockSyntax& entry,
                            const detail::TxSearchPlan& plan,
                            std::vector<std::int32_t>& levels) {
  entry.tx_partition = static_cast<std::uint8_t>(plan.partition);
  entry.tx_part_count = static_cast<std::uint8_t>(plan.parts.size());
  for (std::size_t i = 0; i < plan.parts.size(); ++i) {
    const detail::TxSearchPart& part = plan.parts[i];
    entry.tx_part_tx_size[i] = static_cast<std::uint8_t>(part.tx_size);
    entry.tx_part_tx_type[i] = static_cast<std::uint8_t>(part.tx_type);
    entry.tx_part_mi_row[i] = static_cast<std::int16_t>(part.mi_row);
    entry.tx_part_mi_col[i] = static_cast<std::int16_t>(part.mi_col);
    entry.tx_part_eob[i] = static_cast<std::uint32_t>(part.eob);
    entry.tx_part_levels_offset[i] =
        static_cast<std::uint32_t>(levels.size());
    for (int c = 0; c < part.eob; ++c) {
      levels.push_back(part.levels[static_cast<std::size_t>(c)]);
    }
  }
  if (!plan.parts.empty()) {
    entry.tx_size[0] = entry.tx_part_tx_size[0];
    entry.tx_type[0] = entry.tx_part_tx_type[0];
    entry.eob[0] = entry.tx_part_eob[0];
  }
}

// Round a 1/8-pel MV to the signalled frame precision: 1/8, 1/4 or 1/2 pel.
// Mirrors the CPU reference vp quantize_mv() exactly (half away from zero).
inter::mc::Mv quantize_mv(inter::mc::Mv mv, int precision) {
  const int quantum = precision == 1 ? 2 : (precision == 2 ? 4 : 1);
  if (quantum == 1) return mv;
  const auto round_component = [quantum](int value) {
    const int magnitude = value < 0 ? -value : value;
    const int rounded = ((magnitude + quantum / 2) / quantum) * quantum;
    return value < 0 ? -rounded : rounded;
  };
  return inter::mc::Mv{round_component(mv.y), round_component(mv.x)};
}

int eob_from_levels(const std::vector<std::int32_t>& levels, int w, int h,
                    std::vector<std::uint16_t>* scan_out) {
  const int area = w * h;
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  av2enc::video::syntax::build_scan(w, h, scan);
  int eob = 0;
  for (int c = area - 1; c >= 0; --c) {
    if (levels[scan[static_cast<std::size_t>(c)]] != 0) {
      eob = c + 1;
      break;
    }
  }
  if (scan_out != nullptr) *scan_out = std::move(scan);
  return eob;
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

int source_sample(const VideoInputFrame& frame, int plane, int x, int y) {
  const int w = plane == 0 ? static_cast<int>(frame.width)
                           : static_cast<int>((frame.width + 1U) / 2U);
  const int h = plane == 0 ? static_cast<int>(frame.height)
                           : static_cast<int>((frame.height + 1U) / 2U);
  const int cx = std::clamp(x, 0, w - 1);
  const int cy = std::clamp(y, 0, h - 1);
  const std::span<const std::uint8_t> data =
      plane == 0 ? frame.y : (plane == 1 ? frame.u : frame.v);
  return data[static_cast<std::size_t>(cy) * static_cast<std::size_t>(w) +
              static_cast<std::size_t>(cx)];
}

std::int64_t sse_against_source(const VideoInputFrame& frame, int plane, int x0,
                                int y0, int w, int h,
                                const std::vector<std::uint8_t>& samples) {
  std::int64_t sse = 0;
  for (int y = 0; y < h; ++y) {
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

void write_plane_block(Plane& plane, int x0, int y0, int w, int h,
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

std::uint64_t node_key(int mi_row, int mi_col, BlockSize size) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(mi_row)) << 32) |
         (static_cast<std::uint64_t>(static_cast<std::uint32_t>(mi_col)) << 8) |
         static_cast<std::uint64_t>(static_cast<std::uint8_t>(size));
}

// ME results are per (leaf, reference); pack the reference index in the low
// three bits so the map stays a flat uint64 key.
std::uint64_t me_key(int mi_row, int mi_col, BlockSize size, int ref_index) {
  return node_key(mi_row, mi_col, size) * 8u +
         static_cast<std::uint64_t>(ref_index);
}

struct InterCandidate {
  bool valid = false;
  bool compound = false;
  bool use_warp = false;
  int mode = av2enc::video::syntax::kInterGlobalMv;
  int drl_index = 0;
  int ref_index = 0;
  inter::mc::Mv mv{};
  inter::mc::Mv mvd{};
  std::int64_t sse = 0;
  // §5.20.7.16 masked compound parameters (ignored for AVERAGE).
  int compound_type = 0;  // 0 = WEDGE, 1 = DIFFWTD, 2 = AVERAGE
  int wedge_index = 0;
  int wedge_sign = 0;
  int mask_type = 0;
  // CPU-reference consideration order: (reference list index, mode kind,
  // DRL index).  The GPU batches warp candidates before translational ones,
  // so an equal-SSE tie must be resolved by this key instead of list position.
  int order = 0;};

// The partition node set is a pure function of the coded geometry and the
// partition options: only the per-node variance *costs* change per frame.  The
// CPU decision traversal that discovers it is a host tree walk (8.9 ms/frame at
// 720p), so cache it across the frames of a sequence.  The runtime is
// single-control-thread by contract (docs/cuda-interface.md), so a plain cache
// is safe; a different geometry simply misses and rebuilds.
struct PartitionNodeSet {
  int width = 0;
  int height = 0;
  int tile_x = 0;
  int tile_y = 0;
  int max_depth = 0;
  int min_block_px = 0;
  bool force_single = false;
  std::vector<detail::VarianceNode> nodes;
  std::vector<std::uint64_t> keys;
  std::unordered_map<std::uint64_t, std::size_t> index;

  bool matches(int w, int h, int tx, int ty, int depth, int min_px,
               bool single) const {
    return width == w && height == h && tile_x == tx && tile_y == ty &&
           max_depth == depth && min_block_px == min_px &&
           force_single == single;
  }
};

PartitionNodeSet& partition_node_cache() {
  static PartitionNodeSet cache;
  return cache;
}

class GpuFrameEncoder {
 public:
  GpuFrameEncoder(detail::Backend& backend, const VideoInputFrame& source,
                  std::span<const VideoReconFrame* const> references,
                  std::array<int, 7> frame_distance, int num_total_refs,
                  const VideoOptions& options, bool trace, bool keyframe,
                  std::string& error,
                  int tile_x = 0, int tile_y = 0, int tile_w = 0, int tile_h = 0,
                  std::span<const std::uint16_t> cdf_seed = {},
                  std::vector<std::uint16_t>* cdf_final = nullptr,
                  const GmFramePlan* gm_plan = nullptr,
                  std::array<int, 7> ref_order_hints = {})
      : backend_(backend),
        source_(source),
        options_(options),
        frame_distance_(frame_distance),
        num_total_refs_(num_total_refs),
        cdf_seed_(cdf_seed),
        cdf_final_(cdf_final),
        ref_order_hints_(ref_order_hints),
        trace_(trace),
        keyframe_(keyframe),
        tile_x_(tile_x),
        tile_y_(tile_y),
        gm_plan_(gm_plan),
        references_(references.begin(), references.end()) {
    width_ = tile_w > 0 ? tile_w : static_cast<int>(source.width);
    height_ = tile_h > 0 ? tile_h : static_cast<int>(source.height);
    frame_mi_cols_ = static_cast<int>(2U * ((source.width + 7U) >> 3U));
    frame_mi_rows_ = static_cast<int>(2U * ((source.height + 7U) >> 3U));

    geometry_.width = width_;
    geometry_.height = height_;
    geometry_.sb_size = 64;
    geometry_.has_chroma = true;
    geometry_.chroma_subsampling_x = true;
    geometry_.chroma_subsampling_y = true;
    mi_cols_ = partition::mi_cols(geometry_);
    mi_rows_ = partition::mi_rows(geometry_);

    recon_[0].resize(mi_cols_ * 4, mi_rows_ * 4, 0);
    recon_[1].resize(mi_cols_ * 2, mi_rows_ * 2, 0);
    recon_[2].resize(mi_cols_ * 2, mi_rows_ * 2, 0);

    mv_grid_.assign(static_cast<std::size_t>(mi_rows_) *
                        static_cast<std::size_t>(mi_cols_),
                    inter::mc::MiInfo{});
    // §5.20.5.5 IntraJointMode grid for the --video-intra-full filter flags.
    // Inter leaves write 0, matching inter_block_mode_info().
    intra_joint_.assign(static_cast<std::size_t>(mi_rows_) *
                            static_cast<std::size_t>(mi_cols_),
                        0);
    decoded_[0].reset(mi_cols_, mi_rows_);
    decoded_[1].reset(mi_cols_ / 2, mi_rows_ / 2);
    decoded_[2].reset(mi_cols_ / 2, mi_rows_ / 2);

    if (!backend_.begin_frame(source_, frame_mi_cols_, frame_mi_rows_, tile_x_,
                              tile_y_, mi_cols_, mi_rows_, references,
                              options_.base_q_idx, error)) {
      begin_ok_ = false;
    }
  }

  ~GpuFrameEncoder() = default;

  bool begin_ok() const { return begin_ok_; }

  // Decision + reconstruction phase (no syntax, no filters).
  bool decide(std::string& error) {
    if (!begin_ok_) {
      error = "backend begin_frame failed";
      return false;
    }

    if (!decide_partition(error)) return false;
    if (!precompute_me(error)) return false;

    blocks_.clear();
    levels_.clear();
    for (const PartitionNode& sb : frame_.superblocks) {
      if (!collect_node(sb, error)) return false;
    }
    return true;
  }

  // §7.18-§7.20 decision + device pixel pass on this encoder's
  // reconstruction.  Single-tile path only.
  bool apply_loop_filters(std::string& error) {
    // Opt-in M2 loop filters: decisions and tile syntax on the host, pixels on
    // the CUDA kernels through the backend hook.
    filter_plan_ = av2enc::video::post::VideoFilterPlan{};
    if (options_.loop_filters.any()) {
      av2enc::video::post::VideoLoopFilterFrame lf;
      lf.width = width_;
      lf.height = height_;
      lf.mi_cols = mi_cols_;
      lf.mi_rows = mi_rows_;
      lf.frame_is_intra = keyframe_;
      lf.base_q_idx = options_.base_q_idx;
      lf.num_total_refs = num_total_refs_;
      lf.restricted_oh = -1;
      for (int r = 0; r < 2; ++r) {
        lf.ref_order_hints[r] = ref_order_hints_[static_cast<std::size_t>(r)];
        lf.frame_distance[r] = frame_distance_[static_cast<std::size_t>(r)];
      }
      std::vector<av2enc::video::post::VideoLoopFilterBlock> filter_blocks;
      {
        filter_blocks.reserve(blocks_.size());
        for (const av2enc::video::syntax::BlockSyntax& leaf : blocks_) {
          if (leaf.partition !=
              static_cast<std::uint8_t>(PartitionType::kNone)) {
            continue;
          }
          av2enc::video::post::VideoLoopFilterBlock block;
          block.mi_row = leaf.mi_row;
          block.mi_col = leaf.mi_col;
          block.w4 =
              partition::block_width(
                  static_cast<partition::BlockSize>(leaf.bsize)) >>
              2;
          block.h4 =
              partition::block_height(
                  static_cast<partition::BlockSize>(leaf.bsize)) >>
              2;
          block.skip = leaf.is_inter && leaf.skip_flag != 0;
          block.luma_eob_zero = leaf.eob[0] == 0;
          filter_blocks.push_back(block);
        }
      }
      lf.blocks = filter_blocks;
      const std::span<const std::uint8_t> src[3] = {source_.y, source_.u,
                                                    source_.v};
      for (int plane = 0; plane < 3; ++plane) {
        lf.plane[plane] = recon_[plane].samples.data();
        lf.stride[plane] = recon_[plane].width;
        lf.source[plane] = src[plane].data();
        lf.source_stride[plane] =
            plane == 0 ? static_cast<int>(source_.width)
                       : static_cast<int>(source_.width / 2);
      }
      BackendFilterHooks hooks(backend_);
      if (!av2enc::video::post::run_video_loop_filters(
              lf, options_.loop_filters, &filter_plan_, &hooks, &error)) {
        return false;
      }
    }

    return true;
  }

  // §7.17 deblocking; runs after decide() and before the frame filter pass.
  bool deblock(std::string& error) {
    if (!opt_video_deblock(options_)) return true;
    // backend runs the CUDA deblock kernel and returns the filtered planes;
    // the host mirror is replaced so the emitted recon/reference matches the
    // decoder.
    if (opt_video_deblock(options_)) {
      av2enc::video::post::VideoDeblockGrid grid;
      if (!av2enc::video::post::build_video_deblock_grid(
              blocks_, mi_cols_, mi_rows_, options_.base_q_idx, keyframe_,
              &grid, &error)) {
        return false;
      }
      if (!backend_.deblock_frame(grid, recon_[0].samples, recon_[1].samples,
                                  recon_[2].samples, error)) {
        return false;
      }
    }
    return true;
  }

  // Syntax phase; `filters` is the frame-level plan for a multi-tile caller.
  bool write(std::vector<std::uint8_t>& tile_payload,
             std::vector<av2enc::video::syntax::BlockSyntax>& blocks,
             std::vector<std::int32_t>& levels,
             const av2enc::video::post::VideoFilterPlan* filters,
             std::string& error) {
    av2enc::video::syntax::FramePlan plan;
    plan.width = width_;
    plan.height = height_;
    plan.sb_size = 64;
    plan.frame_mi_cols = frame_mi_cols_;
    plan.frame_mi_rows = frame_mi_rows_;
    plan.tile_mi_row = tile_y_ / 4;
    plan.tile_mi_col = tile_x_ / 4;
    plan.inter_frame = !keyframe_;
    plan.tx_mode_select = opt_enable_tx_search(options_);
    plan.reduced_tx_part_set = false;
    plan.reduced_tx_set = 0;
    plan.num_total_refs = num_total_refs_;
    plan.reference_select = options_.compound && num_total_refs_ > 1;
    plan.allow_bawp = options_.enable_bawp && !keyframe_;
    plan.enable_masked_compound =
        plan.reference_select &&
        (options_.compound_wedge || options_.compound_diffwtd);    plan.mv_precision = options_.mv_precision;
    plan.frame_distance = frame_distance_;
    if (gm_plan_ != nullptr) {
      plan.gm_num_refs = gm_plan_->num_refs;
      plan.gm_params = gm_plan_->params;
    }
    VideoConfig config;
    config.max_frame_width = static_cast<std::uint16_t>(width_);
    config.max_frame_height = static_cast<std::uint16_t>(height_);
    config.base_q_idx = options_.base_q_idx;
    config.order_hint_bits = 4;
    config.num_ref_frames = 2;
    config.num_total_refs = 1;
    config.max_drl_bits_minus_1 = 4;
    config.disable_cdf_update = !options_.adaptive_cdf;
    config.enable_avg_cdf = options_.avg_cdf;
    config.avg_cdf_type = true;
    set_config_drl_reorder(config, opt_drl_reorder(options_));
    config.enable_cfl_intra = true;
    config.enable_mhccp = options_.enable_mhccp;
    config.using_qmatrix = options_.enable_qm;
    config.enable_deblock = opt_video_deblock(options_);
    config.tx_mode_select = opt_enable_tx_search(options_);
    config.enable_bawp = options_.enable_bawp;

    std::size_t tile_bytes = 0;
    if (!av2enc::video::syntax::assemble_video_frame(
            plan, blocks_, levels_, config, tile_payload, error, &tile_bytes,
            cdf_seed_, cdf_final_, filters)) {
      return false;
    }
    blocks = std::move(blocks_);
    levels = std::move(levels_);

    if (!backend_.end_frame(error)) return false;
    return true;
  }

  bool encode(std::vector<std::uint8_t>& tile_payload,
              std::vector<av2enc::video::syntax::BlockSyntax>& blocks,
              std::vector<std::int32_t>& levels, std::string& error) {
    if (!decide(error)) return false;
    if (!deblock(error)) return false;
    if (!apply_loop_filters(error)) return false;
    return write(tile_payload, blocks, levels,
                 filter_plan_.any() ? &filter_plan_ : nullptr, error);
  }

  const std::vector<av2enc::video::syntax::BlockSyntax>& blocks() const {
    return blocks_;
  }

  const Plane& recon(int plane) const { return recon_[plane]; }
  std::uint32_t inter_blocks() const { return inter_blocks_; }
  std::uint32_t intra_blocks() const { return intra_blocks_; }
  const av2enc::video::post::VideoFilterPlan& filter_plan() const {
    return filter_plan_;
  }

 private:
  bool decide_partition(std::string& error) {
    partition::DecisionOptions poptions;
    poptions.min_block_size_px = options_.min_block_size;
    poptions.max_depth = options_.max_partition_depth;
    poptions.allow_rect = true;
    poptions.allow_split = false;
    if (options_.force_single_block) {
      poptions.max_depth = 0;
      poptions.min_block_size_px = 64;
    }
    // The node set is geometry-driven: the callback returns finite zero costs
    // so both policy passes are explored, and extra nodes are harmless.
    PartitionNodeSet& cache = partition_node_cache();
    if (!cache.matches(width_, height_, tile_x_, tile_y_,
                       poptions.max_depth, poptions.min_block_size_px,
                       options_.force_single_block)) {
      cache.width = width_;
      cache.height = height_;
      cache.tile_x = tile_x_;
      cache.tile_y = tile_y_;
      cache.max_depth = poptions.max_depth;
      cache.min_block_px = poptions.min_block_size_px;
      cache.force_single = options_.force_single_block;
      cache.nodes.clear();
      cache.keys.clear();
      cache.index.clear();
      cache.nodes.reserve(4096);
      cache.keys.reserve(4096);
      cache.index.reserve(8192);
      const auto record = [&](const partition::PartitionNodeInfo& info) {
        const partition::BlockRect rect = partition::block_rect(
            geometry_, info.mi_row, info.mi_col, info.size);
        const std::uint64_t key = node_key(info.mi_row, info.mi_col, info.size);
        if (cache.index.find(key) == cache.index.end()) {
          cache.index.emplace(key, cache.nodes.size());
          cache.keys.push_back(key);
          cache.nodes.push_back(detail::VarianceNode{rect.x + tile_x_,
                                                     rect.y + tile_y_,
                                                     rect.visible_width,
                                                     rect.visible_height});
        }
        partition::NodeCosts costs;
        costs.leaf = 0.0;
        costs.none_signal = 0.0;
        costs.split_signal = 0.0;
        costs.horz_signal = 0.0;
        costs.vert_signal = 0.0;
        return costs;
      };
      const partition::FrameDecision discovery =
          partition::decide_frame(geometry_, record, poptions);
      if (discovery.status != partition::DecisionStatus::kOk) {
        error = "partition discovery failed: " + discovery.error;
        return false;
      }
    }
    const std::vector<detail::VarianceNode>& nodes = cache.nodes;
    const std::vector<std::uint64_t>& keys = cache.keys;

    std::vector<std::int64_t> costs;
    if (!backend_.variance_costs(nodes, costs, error)) return false;
    if (costs.size() != nodes.size()) {
      error = "backend returned the wrong variance-cost count";
      return false;
    }
    std::unordered_map<std::uint64_t, std::int64_t> cost_map;
    cost_map.reserve(keys.size() * 2);
    for (std::size_t i = 0; i < keys.size(); ++i) {
      cost_map.emplace(keys[i], costs[i]);
    }

    const double lambda =
        0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
    const double signal_cost = lambda * 2.0;
    const auto cost_fn = [this, &cost_map, signal_cost](
                             const partition::PartitionNodeInfo& info) {
      partition::NodeCosts node_costs;
      const auto it =
          cost_map.find(node_key(info.mi_row, info.mi_col, info.size));
      if (it == cost_map.end()) {
        missing_cost_ = true;
        node_costs.leaf = partition::kInfiniteCost;
        return node_costs;
      }
      node_costs.leaf = static_cast<double>(it->second);
      node_costs.none_signal = 0.0;
      node_costs.split_signal = signal_cost;
      node_costs.horz_signal = signal_cost;
      node_costs.vert_signal = signal_cost;
      return node_costs;
    };

    missing_cost_ = false;
    frame_ = partition::decide_frame(geometry_, cost_fn, poptions);
    if (missing_cost_) {
      error = "partition decision queried a node with no device variance cost";
      return false;
    }
    if (frame_.status != partition::DecisionStatus::kOk) {
      error = "partition decision failed: " + frame_.error;
      return false;
    }
    if (!partition::validate_frame(geometry_, frame_, error)) {
      error = "partition tree invalid: " + error;
      return false;
    }
    return true;
  }

  bool precompute_me(std::string& error) {
    me_results_.clear();
    me_index_.clear();
    if (keyframe_ || options_.force_global_mv || !options_.enable_newmv) {
      return true;
    }
    std::vector<partition::LeafBlock> leaves;
    for (const PartitionNode& sb : frame_.superblocks) {
      partition::collect_leaves(sb, leaves);
    }
    std::vector<detail::MeRequest> jobs;
    std::vector<std::uint64_t> job_keys;
    jobs.reserve(leaves.size() * static_cast<std::size_t>(num_total_refs_));
    job_keys.reserve(jobs.capacity());
    for (const partition::LeafBlock& leaf : leaves) {
      const int w = partition::block_width(leaf.size);
      const int h = partition::block_height(leaf.size);
      // The CPU reference only evaluates inter when the block fits inside the
      // reference planes (mi_cols*4 x mi_rows*4).  Edge blocks that straddle
      // the bottom/right boundary therefore never get an inter candidate.  ME
      // is skipped for them as well so the backend never reads out of plane.
      if (leaf.mi_col * 4 + w > mi_cols_ * 4 ||
          leaf.mi_row * 4 + h > mi_rows_ * 4) {
        continue;
      }
      for (int ref = 0; ref < num_total_refs_; ++ref) {
        job_keys.push_back(me_key(leaf.mi_row, leaf.mi_col, leaf.size, ref));
        detail::MeRequest request;
        request.x = leaf.mi_col * 4 + tile_x_;
        request.y = leaf.mi_row * 4 + tile_y_;
        request.w = w;
        request.h = h;
        request.ref_index = ref;
        jobs.push_back(request);
      }
    }
    if (jobs.empty()) return true;

    me::MeCoreOptions opts;
    opts.method = me::kMeDiamond;
    opts.metric = me::kMeSad;
    opts.range = options_.me_range;
    opts.subpel_stages = me::kMeSubpelHalf | me::kMeSubpelQuarter;
    if (options_.me_subpel_eighth) {
      opts.subpel_stages |= me::kMeSubpelEighth;
    }
    opts.subpel_interp = me::kMeInterpRegular;
    opts.use_predictors = 1;
    opts.use_merge_candidates = 0;

    if (!backend_.me_batch(jobs, opts, me_results_, error)) return false;
    if (me_results_.size() != jobs.size()) {
      error = "backend returned the wrong ME result count";
      return false;
    }
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      me_index_.emplace(job_keys[i], i);
    }
    return true;
  }

  bool collect_node(const PartitionNode& node, std::string& error) {
    if (node.depth == 0) {
      // §6.19.2.3 clear_block_decoded_flags(): the map is superblock-local.
      decoded_[0].reset(mi_cols_, mi_rows_);
      decoded_[1].reset(mi_cols_ / 2, mi_rows_ / 2);
      decoded_[2].reset(mi_cols_ / 2, mi_rows_ / 2);
    }
    const std::size_t index = blocks_.size();
    blocks_.emplace_back();
    av2enc::video::syntax::BlockSyntax& entry = blocks_.back();
    entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
    entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
    entry.bsize = static_cast<std::uint8_t>(node.size);
    entry.partition = static_cast<std::uint8_t>(node.partition);
    std::uint8_t mask = 0;
    for (int p = 0; p < partition::kPartitionTypeCount; ++p) {
      if (node.allowed.allowed[p]) {
        mask = static_cast<std::uint8_t>(mask | (1U << p));
      }
    }
    entry.allowed_mask = mask;
    if (node.forced) entry.flags |= av2enc::video::syntax::kFlagForced;
    if (node.depth == 0) entry.flags |= av2enc::video::syntax::kFlagSbRoot;

    if (node.partition == PartitionType::kNone) {
      if (!encode_leaf(node, index, error)) return false;
    } else {
      for (const PartitionNode& child : node.children) {
        if (!collect_node(child, error)) return false;
      }
    }
    const std::size_t subtree = blocks_.size() - index - 1;
    if (subtree > 0xFFFFU) {
      error = "partition subtree too large";
      return false;
    }
    blocks_[index].subtree = static_cast<std::uint16_t>(subtree);
    return true;
  }

  // §7.12.2.2 GLOBALMV projection.  Mirrors the CPU encoder: a non-identity
  // model selects the warped predictor for blocks >= 8x8 (§7.13.3.14).
  void global_mv_for_block(int ref_index, int mi_row, int mi_col, int bw4,
                           int bh4, bool& use_warp,
                           inter::mc::Mv& mv) const {
    use_warp = false;
    mv = inter::mc::Mv{0, 0};
    if (gm_plan_ == nullptr || ref_index < 0 ||
        ref_index >= gm_plan_->num_refs) {
      return;
    }
    const inter::gme::GmParams& params =
        gm_plan_->params[static_cast<std::size_t>(ref_index)];
    if (params.type == inter::gme::GmType::kIdentity) return;
    inter::gme::MvPrecision precision = inter::gme::MvPrecision::kEighthPel;
    if (options_.mv_precision == 1) {
      precision = inter::gme::MvPrecision::kQuarterPel;
    } else if (options_.mv_precision == 2) {
      precision = inter::gme::MvPrecision::kHalfPel;
    }
    int projected[2] = {0, 0};
    inter::gme::get_warp_motion_vector(params, mi_col, mi_row, bw4, bh4,
                                       mi_cols_, mi_rows_, precision,
                                       projected);
    mv = inter::mc::Mv{projected[0], projected[1]};
    use_warp = bw4 * 4 >= 8 && bh4 * 4 >= 8;
  }

  merge::StackParams make_stack_params(int mi_row, int mi_col, int w4,
                                       int h4, int ref_index) const {
    merge::StackParams params;
    params.mi_row = mi_row;
    params.mi_col = mi_col;
    params.bw4 = w4;
    params.bh4 = h4;
    params.sb_h4 = 16;
    params.mi_cols = mi_cols_;
    params.mi_rows = mi_rows_;
    params.mi = mv_grid_.data();
    // The merge layer uses AV2 RefFrame labels: 0..6 are reference indices.
    params.ref_frame[0] = static_cast<std::int8_t>(ref_index);
    params.ref_frame[1] = merge::kRefNone;
    params.is_compound = false;
    bool gm_use_warp = false;
    inter::mc::Mv gm_mv{};
    global_mv_for_block(ref_index, mi_row, mi_col, w4, h4, gm_use_warp, gm_mv);
    params.global_mv[0] = gm_mv;
    params.global_mv[1] = gm_mv;
    // DRL reordering is rejected by the CLI and by the entry-point validation;
    // keep the spec-exact stack with reordering disabled.
    params.reorder = merge::DrlReorder::kDisabled;
    params.max_drl_bits_minus_1 = 4;
    if (num_total_refs_ > 1) {
      params.num_refs = num_total_refs_;
      params.frame_distance = frame_distance_.data();
    }
    return params;
  }

  bool load_stack(int mi_row, int mi_col, int w4, int h4, int ref_index,
                  merge::StackParams& params, merge::MvStack& stack) const {
    params = make_stack_params(mi_row, mi_col, w4, h4, ref_index);
    return merge::build_stack_checked(params, &stack) == merge::Status::kOk;
  }

  bool encode_leaf(const PartitionNode& node, std::size_t index,
                   std::string& error) {
    const int r = node.mi_row;
    const int c = node.mi_col;
    const int w = partition::block_width(node.size);
    const int h = partition::block_height(node.size);
    const int w4 = w >> 2;
    const int h4 = h >> 2;
    const int x0 = c * 4;
    const int y0 = r * 4;
    const int cw = w / 2;
    const int ch = h / 2;
    const int cx0 = x0 / 2;
    const int cy0 = y0 / 2;
    // Absolute frame coordinates for source reads and motion compensation.
    // Coded geometry, block syntax and prediction availability stay tile-local.
    const int ax0 = x0 + tile_x_;
    const int ay0 = y0 + tile_y_;
    const int acx0 = cx0 + tile_x_ / 2;
    const int acy0 = cy0 + tile_y_ / 2;
    // Mirror the CPU reference: inter candidates are only evaluated when the
    // luma block fits inside the reference planes.  At the bottom/right frame
    // edge a 64x64 node can straddle the plane; the CPU's predict_plane()
    // rejects it, so the GPU path must not invent an inter candidate there.
    const bool inter_fits = x0 + w <= mi_cols_ * 4 && y0 + h <= mi_rows_ * 4;

    av2enc::video::syntax::BlockSyntax& entry = blocks_[index];

    InterCandidate best;
    const auto consider = [&](const InterCandidate& cand) {
      if (!cand.valid) return;
      // The CPU reference considers candidates in (reference, kind, DRL)
      // order and replaces only on a strictly smaller SSE.  The GPU builds
      // the warp list first, so compare the explicit order key on a tie to
      // keep the same first-wins decision.
      if (!best.valid || cand.sse < best.sse ||
          (cand.sse == best.sse && cand.order < best.order)) {
        best = cand;
      }
    };

    // Candidate order key shared by every list below: GLOBALMV = 0,
    // NEARMV = 1, NEWMV = 2 (the CPU's per-reference consideration order).
    const auto order_key = [](int ref_index, int kind, int drl) {
      return ref_index * 256 + kind * 32 + drl;
    };
    std::vector<detail::InterJob> score_jobs;
    std::vector<int> score_kind;  // 0 global, 1 near, 2 new
    std::vector<int> score_drl;
    std::vector<int> score_ref;
    std::vector<inter::mc::Mv> score_mv;
    std::vector<inter::mc::Mv> score_mvd;
    std::vector<int> score_order;

    // §7.13.3.19/.20 warped GLOBALMV candidates are scored separately because
    // their prediction is not a single translational MV.
    std::vector<detail::WarpJob> warp_jobs;
    std::vector<inter::mc::Mv> warp_mv;
    std::vector<int> warp_ref;
    std::vector<int> warp_order;

    const auto add_score = [&](int kind, int drl, inter::mc::Mv mv,
                               inter::mc::Mv mvd, int ref_index) {
      detail::InterJob job;
      job.x = x0 + tile_x_;
      job.y = y0 + tile_y_;
      job.w = w;
      job.h = h;
      job.mv_y = mv.y;
      job.mv_x = mv.x;
      job.ref_index = ref_index;
      score_jobs.push_back(job);
      score_kind.push_back(kind);
      score_drl.push_back(drl);
      score_ref.push_back(ref_index);
      score_mv.push_back(mv);
      score_mvd.push_back(mvd);
      score_order.push_back(order_key(ref_index, kind, drl));
    };

    if (!keyframe_) {
      for (int ref_index = 0; ref_index < num_total_refs_; ++ref_index) {
        merge::StackParams stack_params;
        merge::MvStack stack;
        const bool have_stack =
            load_stack(r, c, w4, h4, ref_index, stack_params, stack);

        if (inter_fits) {
          bool gm_use_warp = false;
          inter::mc::Mv gm_mv{};
          global_mv_for_block(ref_index, r, c, w4, h4, gm_use_warp, gm_mv);
          if (gm_use_warp && gm_plan_ != nullptr &&
              ref_index < gm_plan_->num_refs) {
            detail::WarpJob job;
            job.x = x0;
            job.y = y0;
            job.w = w;
            job.h = h;
            job.mi_row = r;
            job.mi_col = c;
            job.bw4 = w4;
            job.bh4 = h4;
            job.ref_index = ref_index;
            job.params =
                gm_plan_->params[static_cast<std::size_t>(ref_index)];
            warp_jobs.push_back(job);
            warp_mv.push_back(gm_mv);
            warp_ref.push_back(ref_index);
            warp_order.push_back(order_key(ref_index, 0, 0));
          } else {
            add_score(0, 0, gm_mv, inter::mc::Mv{0, 0}, ref_index);
          }
        }
        if (trace_ && inter_fits && !have_stack) {
          std::cerr << "  empty stack r=" << r << " c=" << c << " size="
                    << static_cast<int>(node.size) << " ref=" << ref_index
                    << "\n";
        }
        if (!inter_fits || !have_stack || options_.force_global_mv) continue;

        if (options_.enable_nearmv) {
          const int near_limit =
              (w > 32 && h > 32) ? stack.count : std::min(stack.count, 2);
          for (int i = 0; i < near_limit; ++i) {
            merge::ResolvedMode resolved;
            if (merge::resolve_single_mode(stack, stack_params,
                                           merge::SingleMode::kNearMv, i,
                                           inter::mc::Mv{0, 0}, &resolved) !=
                merge::Status::kOk) {
              continue;
            }
            add_score(1, i, resolved.mv[0], resolved.mvd[0], ref_index);
          }
        }

        if (options_.enable_newmv) {
          const auto it =
              me_index_.find(me_key(r, c, node.size, ref_index));
          if (it == me_index_.end() && trace_) {
            std::cerr << "  no me job r=" << r << " c=" << c << " size="
                      << static_cast<int>(node.size) << " stack=" << stack.count
                      << " ref=" << ref_index << "\n";
          }
          if (it != me_index_.end() && it->second < me_results_.size()) {
            const detail::MeOutcome& me_out = me_results_[it->second];
            const inter::mc::Mv me_mv = inter::mc::mv_from_me_1_16(
                me_out.mv_y, me_out.mv_x);
            if (options_.mv_precision != 0 && stack.count > 0) {
              // §5.20.7.13: quarter/half-pel frames keep the predictor as-is
              // and quantise the ME MV and the signalling MVD.  This mirrors
              // the CPU reference instead of routing through resolve_mode(),
              // which is only the 1/8-pel path.
              const inter::mc::Mv me_q =
                  quantize_mv(me_mv, options_.mv_precision);
              const inter::mc::Mv pred = stack.cand[0].mv[0];
              const inter::mc::Mv delta{me_q.y - pred.y, me_q.x - pred.x};
              const inter::mc::Mv mvd =
                  quantize_mv(delta, options_.mv_precision);
              add_score(2, 0, inter::mc::Mv{pred.y + mvd.y, pred.x + mvd.x},
                        mvd, ref_index);
            } else {
              const inter::mc::Mv pred = stack.cand[0].mv[0];
              const inter::mc::Mv mvd{me_mv.y - pred.y, me_mv.x - pred.x};
              merge::ResolvedMode resolved;
              if (merge::resolve_single_mode(stack, stack_params,
                                             merge::SingleMode::kNewMv, 0, mvd,
                                             &resolved) == merge::Status::kOk) {
                add_score(2, 0, resolved.mv[0], resolved.mvd[0], ref_index);
              }
            }
          }
        }
      }
    }

    if (!warp_jobs.empty()) {
      std::vector<std::int64_t> sse;
      if (!backend_.warp_sse_batch(warp_jobs, sse, error)) return false;
      if (sse.size() != warp_jobs.size()) {
        error = "backend returned the wrong warp-SSE count";
        return false;
      }
      for (std::size_t i = 0; i < warp_jobs.size(); ++i) {
        InterCandidate cand;
        cand.valid = true;
        cand.use_warp = true;
        cand.mode = av2enc::video::syntax::kInterGlobalMv;
        cand.drl_index = 0;
        cand.ref_index = warp_ref[i];
        cand.mv = warp_mv[i];
        cand.sse = sse[i];
        cand.order = warp_order[i];
        if (trace_) {
          std::cerr << "  cand warp ref=" << cand.ref_index << " mv=("
                    << cand.mv.y << "," << cand.mv.x << ") sse=" << cand.sse
                    << "\n";
        }
        consider(cand);
      }
    }

    if (!score_jobs.empty()) {
      std::vector<std::int64_t> sse;
      if (!backend_.inter_sse_batch(score_jobs, sse, error)) return false;
      if (sse.size() != score_jobs.size()) {
        error = "backend returned the wrong inter-SSE count";
        return false;
      }
      for (std::size_t i = 0; i < score_jobs.size(); ++i) {
        InterCandidate cand;
        cand.valid = true;
        cand.mode = score_kind[i] == 0
                        ? av2enc::video::syntax::kInterGlobalMv
                        : (score_kind[i] == 1
                               ? av2enc::video::syntax::kInterNearMv
                               : av2enc::video::syntax::kInterNewMv);
        cand.drl_index = score_drl[i];
        cand.ref_index = score_ref[i];
        cand.mv = score_mv[i];
        cand.mvd = score_mvd[i];
        cand.sse = sse[i];
        cand.order = score_order[i];
        if (trace_) {
          std::cerr << "  cand kind=" << score_kind[i]
                    << " drl=" << score_drl[i]
                    << " ref=" << score_ref[i]
                    << " mv=(" << cand.mv.y << ","
                    << cand.mv.x << ") sse=" << cand.sse << "\n";
        }
        consider(cand);
      }
    }

    // Compound candidates (§5.20.7.10 comp_mode + §7.13.3.14/.27-.30).
    // Gated subset: non-joint GLOBAL_GLOBALMV over two references, both MVs
    // zero, no residual.  The default is the CWP_EQUAL AVERAGE blend; the
    // opt-in masked modes enumerate the same candidate order as the CPU
    // `inter::adv::search_compound()` so the two paths tie-break identically.
    if (!keyframe_ && options_.compound && num_total_refs_ >= 2 && w >= 8 &&
        h >= 8 && !options_.force_intra && inter_fits) {
      detail::InterJob base;
      base.x = x0 + tile_x_;
      base.y = y0 + tile_y_;
      base.w = w;
      base.h = h;
      base.mv_y = 0;
      base.mv_x = 0;
      base.ref_index = 0;
      base.bsize = static_cast<int>(node.size);
      base.compound_type = 2;  // AVERAGE
      std::vector<detail::InterJob> jobs;
      jobs.push_back(base);
      const bool wedge_enabled = options_.compound_wedge &&
                                 inter::adv::bsize_has_wedge(base.bsize);
      if (wedge_enabled) {
        for (int sign = 0; sign <= 1; ++sign) {
          for (int index = 0; index < inter::adv::kWedgeTypes; ++index) {
            if (!inter::adv::wedge_index_signalable(index)) continue;
            detail::InterJob job = base;
            job.compound_type = 0;
            job.wedge_index = index;
            job.wedge_sign = sign;
            jobs.push_back(job);
          }
        }
      }
      if (options_.compound_diffwtd) {
        for (int mask_type = 0; mask_type <= 1; ++mask_type) {
          detail::InterJob job = base;
          job.compound_type = 1;
          job.mask_type = mask_type;
          jobs.push_back(job);
        }
      }
      std::vector<std::int64_t> sse;
      if (!backend_.compound_sse_batch(jobs, sse, error)) return false;
      if (sse.size() != jobs.size()) {
        error = "backend returned the wrong compound-SSE count";
        return false;
      }
      for (std::size_t i = 0; i < jobs.size(); ++i) {
        InterCandidate cand;
        cand.valid = true;
        cand.compound = true;
        cand.mode = av2enc::video::syntax::kInterGlobalMv;
        cand.ref_index = 0;
        cand.mv = inter::mc::Mv{0, 0};
        cand.sse = sse[i];
        cand.compound_type = jobs[i].compound_type;
        cand.wedge_index = jobs[i].wedge_index;
        cand.wedge_sign = jobs[i].wedge_sign;
        cand.mask_type = jobs[i].mask_type;
        if (trace_) {
          std::cerr << "  cand compound global-global type="
                    << cand.compound_type << " wedge=" << cand.wedge_index
                    << " sign=" << cand.wedge_sign
                    << " mask_type=" << cand.mask_type
                    << " sse=" << cand.sse << "\n";
        }
        consider(cand);
      }
    }

    // Intra fallback: DC prediction + DCT residual, the M1 transform path.
    // --video-intra-full runs the still product's 61-candidate luma search
    // through the device module; chroma stays DC_PRED.
    bool have_intra = false;
    std::int64_t intra_sse = 0;
    std::array<detail::PlaneTx, 3> intra_tx;
    int intra_y_mode = 0;        // AV2 YMode (0 = DC_PRED)
    int intra_y_angle_delta = 0;
    int intra_is_cfl = 0;        // §7.13.5 CfL winner (--video-cfl)
    int intra_cfl_index = 1;
    int intra_cfl_mh_dir = 0;    // §7.13.7 CFL_MULTI direction
    detail::TxSearchPlan intra_luma_plan;
    bool intra_luma_searched = false;
    const int tx_qindex = static_cast<int>(options_.base_q_idx);
    const double tx_lambda = opt_tx_rd_lambda(options_);
    if (keyframe_ || options_.enable_intra_fallback || options_.force_intra) {
      bool predicted = false;
      if (opt_enable_intra_full(options_)) {
        bool above_smooth = false;
        bool left_smooth = false;
        luma_neighbor_smooth(r, c, above_smooth, left_smooth);
        // §5.20.9.1 availability from the §6.19.2.3 BlockDecoded replay.
        const av2enc::video::intra_search::Availability luma_avail =
            av2enc::video::intra_search::availability(
                decoded_[0], c, r, w4, h4, mi_cols_, mi_rows_,
                /*sb_mi=*/16);
        const av2enc::video::intra_search::Availability chroma_avail =
            av2enc::video::intra_search::availability(
                decoded_[1], c / 2, r / 2, cw / 4, ch / 4, mi_cols_ / 2,
                mi_rows_ / 2, /*sb_mi=*/8);
        predicted = backend_.intra_full_candidate(
            x0, y0, w, h, luma_avail, chroma_avail, above_smooth, left_smooth,
            intra_tx, intra_y_mode, intra_y_angle_delta, error);
      } else {
        predicted = backend_.intra_candidate(x0, y0, w, h, intra_tx, error);
      }
      if (!predicted) return false;
      if (opt_enable_tx_search(options_)) {
        detail::InterJob unused_job;
        if (!backend_.tx_search_luma(false, unused_job, x0, y0, w, h,
                                     tx_qindex, tx_lambda,
                                     /*partition=*/false, intra_luma_plan,
                                     error)) {
          return false;
        }
        intra_luma_searched = true;
        intra_tx[0].recon = intra_luma_plan.recon;
      }
      if (opt_enable_cfl(options_)) {
        // §7.13.5 CfL: publish the luma reconstruction, then let the backend
        // run the {DC, CfL} chroma decision.  Both backends share the same
        // policy (device `still::chroma_decide`, host `chroma_search::choose`).
        if (!backend_.write_recon_luma(x0, y0, w, h, intra_tx[0].recon,
                                       error)) {
          return false;
        }
        // §7.13.6-.8 MHCCP availability from the §6.19.2.3 map.
        const av2enc::video::intra_search::Availability chroma_avail =
            av2enc::video::intra_search::availability(
                decoded_[1], c / 2, r / 2, cw / 4, ch / 4, mi_cols_ / 2,
                mi_rows_ / 2, /*sb_mi=*/8);
        std::array<detail::PlaneTx, 2> chroma;
        if (!backend_.chroma_cfl_candidate(
                x0, y0, w, h, chroma_avail.num4_above_right,
                chroma_avail.num4_below_left, opt_enable_mhccp(options_),
                chroma, intra_is_cfl, intra_cfl_index, intra_cfl_mh_dir,
                error)) {
          return false;
        }
        intra_tx[1] = std::move(chroma[0]);
        intra_tx[2] = std::move(chroma[1]);
      }
      intra_sse =
          sse_against_source(source_, 0, ax0, ay0, w, h, intra_tx[0].recon);
      if (trace_) {
        std::cerr << "  cand intra sse=" << intra_sse << "\n";
      }
      have_intra = true;
    }

    // Optional inter residual tool (present in the extended CPU M2 path).
    // Mirrors the CPU decision: predict with the winning MV, transform the
    // residual, and compare SSE + lambda * heuristic bits against skip.
    const bool inter_residual_enabled =
        !keyframe_ && !best.compound && opt_enable_inter_residual(options_);
    bool inter_residual = false;
    double res_bits = 0.0;
    std::vector<std::uint8_t> inter_y;
    std::vector<std::uint8_t> inter_u;
    std::vector<std::uint8_t> inter_v;
    std::array<detail::PlaneTx, 3> res_tx;
    detail::TxSearchPlan res_luma_plan;
    bool res_luma_searched = false;
    bool bawp_applied = false;
    bool prediction_ready = false;
    inter::adv::BawpFit bawp_fit{};
    std::int64_t inter_luma_sse = best.sse;
    detail::InterJob best_job;
    best_job.x = ax0;
    best_job.y = ay0;
    best_job.w = w;
    best_job.h = h;
    best_job.mv_y = best.mv.y;
    best_job.mv_x = best.mv.x;
    best_job.ref_index = best.ref_index;
    best_job.compound_type = best.compound_type;
    best_job.bsize = static_cast<int>(node.size);
    best_job.wedge_index = best.wedge_index;
    best_job.wedge_sign = best.wedge_sign;
    best_job.mask_type = best.mask_type;
    detail::WarpJob best_warp;
    best_warp.x = x0;
    best_warp.y = y0;
    best_warp.w = w;
    best_warp.h = h;
    best_warp.mi_row = r;
    best_warp.mi_col = c;
    best_warp.bw4 = w4;
    best_warp.bh4 = h4;
    best_warp.ref_index = best.ref_index;
    if (gm_plan_ != nullptr && best.ref_index < gm_plan_->num_refs) {
      best_warp.params =
          gm_plan_->params[static_cast<std::size_t>(best.ref_index)];
    }
    const auto predict_best = [&](std::vector<std::uint8_t>& py,
                                  std::vector<std::uint8_t>& pu,
                                  std::vector<std::uint8_t>& pv) {
      if (best.use_warp) {
        return backend_.warp_420(best_warp, py, pu, pv, error);
      }
      return backend_.mc_420(best_job, py, pu, pv, error);
    };
    if (inter_residual_enabled && best.valid) {
      if (!predict_best(inter_y, inter_u, inter_v)) {
        return false;
      }
      inter_luma_sse =
          sse_against_source(source_, 0, ax0, ay0, w, h, inter_y);
      const std::int64_t pred_total =
          inter_luma_sse +
          sse_against_source(source_, 1, acx0, acy0, cw, ch, inter_u) +
          sse_against_source(source_, 2, acx0, acy0, cw, ch, inter_v);
      if (!backend_.inter_residual_420(best_job, res_tx, error)) return false;
      if (opt_enable_tx_search(options_)) {
        // Inter search indexes the frame-wide source/reference planes, so it
        // needs absolute coordinates (the reconstruction is tile-local).
        if (!backend_.tx_search_luma(true, best_job, ax0, ay0, w, h, tx_qindex,
                                     tx_lambda, /*partition=*/true,
                                     res_luma_plan, error)) {
          return false;
        }
        res_luma_searched = true;
        res_tx[0].recon = res_luma_plan.recon;
      }
      const std::int64_t res_total =
          sse_against_source(source_, 0, ax0, ay0, w, h,
                             res_tx[0].recon) +
          sse_against_source(source_, 1, acx0, acy0, cw, ch,
                             res_tx[1].recon) +
          sse_against_source(source_, 2, acx0, acy0, cw, ch,
                             res_tx[2].recon);
      const int rcoded[3] = {w, cw, cw};
      const int rcoded_h[3] = {h, ch, ch};
      int eobs[3] = {0, 0, 0};
      for (int plane = 0; plane < 3; ++plane) {
        const auto params_opt = txv::make_tx_params(
            rcoded[plane], rcoded_h[plane], txv::TxType::kDctDct);
        if (!params_opt.has_value()) {
          error = "unsupported inter residual transform size";
          return false;
        }
        eobs[plane] = eob_from_levels(
            res_tx[static_cast<std::size_t>(plane)].levels,
            params_opt->w, params_opt->h, nullptr);
      }
      const bool all_zero =
          (res_luma_searched ? luma_search_all_zero(res_luma_plan)
                             : eobs[0] == 0) &&
          eobs[1] == 0 && eobs[2] == 0;
      const double kLambda = opt_inter_rd_lambda(options_);
      res_bits =
          3.0 + (res_luma_searched
                     ? luma_search_bits(res_luma_plan, true)
                     : estimate_tx_bits(eobs[0], res_tx[0].levels, true)) +
          estimate_tx_bits(eobs[1], res_tx[1].levels, false) +
          estimate_tx_bits(eobs[2], res_tx[2].levels, false);
      const bool want =
          opt_force_inter_residual(options_) ||
          (!all_zero &&
           static_cast<double>(res_total) + kLambda * res_bits <
               static_cast<double>(pred_total) + kLambda * 2.0);
      if (want) {
        inter_residual = true;
        inter_y = res_tx[0].recon;
        inter_u = res_tx[1].recon;
        inter_v = res_tx[2].recon;
        inter_luma_sse =
            sse_against_source(source_, 0, ax0, ay0, w, h, inter_y);
      }
      if (trace_) {
        std::cerr << "  cand inter final sse=" << inter_luma_sse
                  << " residual=" << (inter_residual ? 1 : 0) << "\n";
      }
    }

    // §5.20.7.6 / §7.13.3.25 fitted BAWP.  The gated product surface is the
    // skip-only path (BAWP cannot compose with the residual backend, which
    // re-runs MC into its own scratch predictor).  The decision matches the
    // CPU reference: adapt the winning prediction and keep it only when the
    // luma SSE improves.
    if (!keyframe_ && best.valid && options_.enable_bawp && !best.compound &&
        !best.use_warp && best.mode != av2enc::video::syntax::kInterGlobalMv &&
        !inter_residual_enabled && w >= 8 && h >= 8 && options_.tile_cols == 0 &&
        mi_cols_ * 4 == width_ && mi_rows_ * 4 == height_ &&
        best.ref_index >= 0 &&
        static_cast<std::size_t>(best.ref_index) < references_.size() &&
        references_[best.ref_index] != nullptr) {
      if (!predict_best(inter_y, inter_u, inter_v)) return false;
      prediction_ready = true;
      const std::int64_t plain_sse =
          sse_against_source(source_, 0, ax0, ay0, w, h, inter_y);
      std::vector<std::uint8_t> adapted = inter_y;
      inter::adv::BawpFit fit{};
      const VideoReconFrame& ref = *references_[best.ref_index];
      if (av2enc::video::bawp_adapt_luma(recon_[0].samples.data(),
                          recon_[0].stride,
                          ref.y.data(), static_cast<int>(ref.width), mi_cols_,
                          mi_rows_, r, c, x0, y0, w, h, best.mv.y, best.mv.x,
                          adapted, &fit)) {
        const std::int64_t adapted_sse =
            sse_against_source(source_, 0, ax0, ay0, w, h, adapted);
        if (adapted_sse < plain_sse) {
          bawp_applied = true;
          bawp_fit = fit;
          inter_y = adapted;
          inter_luma_sse = adapted_sse;
        }
      }
      if (!bawp_applied) inter_luma_sse = plain_sse;
      if (trace_) {
        std::cerr << "  cand bawp alpha=" << bawp_fit.alpha
                  << " count=" << bawp_fit.count
                  << " applied=" << (bawp_applied ? 1 : 0) << "\n";
      }
    }

    bool use_intra = keyframe_ || options_.force_intra || !best.valid;
    if (!use_intra && have_intra) {
      if (opt_enable_inter_residual(options_)) {
        // Rate-aware decision, mirroring the CPU residual path.
        const double lambda = opt_inter_rd_lambda(options_);
        int intra_eobs[3] = {0, 0, 0};
        double intra_bits = 3.0;
        const int coded_w[3] = {w, cw, cw};
        const int coded_h[3] = {h, ch, ch};
        for (int plane = 0; plane < 3; ++plane) {
          const auto params_opt = txv::make_tx_params(
              coded_w[plane], coded_h[plane], txv::TxType::kDctDct);
          if (!params_opt.has_value()) {
            error = "unsupported intra transform size";
            return false;
          }
          intra_eobs[plane] = eob_from_levels(
              intra_tx[static_cast<std::size_t>(plane)].levels,
              params_opt->w, params_opt->h, nullptr);
          if (plane == 0 && intra_luma_searched) {
            intra_bits += luma_search_bits(intra_luma_plan, false);
          } else {
            intra_bits += estimate_tx_bits(
                intra_eobs[plane],
                intra_tx[static_cast<std::size_t>(plane)].levels,
                false);
          }
        }
        double inter_bits = 8.0;
        if (inter_residual) inter_bits += res_bits;
        use_intra = static_cast<double>(intra_sse) + lambda * intra_bits <
                    static_cast<double>(inter_luma_sse) + lambda * inter_bits;
      } else {
        use_intra = intra_sse < inter_luma_sse;
      }
    }

    if (use_intra) {
      if (!have_intra) {
        error = "intra fallback unavailable";
        return false;
      }
      entry.is_inter = 0;
      entry.skip_flag = 0;
      entry.y_mode = static_cast<std::uint8_t>(intra_y_mode);
      entry.y_angle_delta = static_cast<std::int8_t>(intra_y_angle_delta);
      entry.uv_mode = 0;  // DC_PRED (also the CfL base mode)
      if (intra_is_cfl != 0) {
        entry.flags |= av2enc::video::syntax::kFlagIsCfl;
        entry.cfl_index = static_cast<std::uint8_t>(intra_cfl_index);
        entry.cfl_mh_dir = static_cast<std::uint8_t>(intra_cfl_mh_dir);
      }
      write_plane_block(recon_[0], x0, y0, w, h, intra_tx[0].recon);
      write_plane_block(recon_[1], cx0, cy0, cw, ch, intra_tx[1].recon);
      write_plane_block(recon_[2], cx0, cy0, cw, ch, intra_tx[2].recon);
      if (!backend_.write_recon_420(x0, y0, w, h, intra_tx[0].recon,
                                    intra_tx[1].recon, intra_tx[2].recon,
                                    error)) {
        return false;
      }
      entry.levels_offset = static_cast<std::uint32_t>(levels_.size());
      if (intra_luma_searched) {
        write_luma_search_plan(entry, intra_luma_plan, levels_);
      }
      const int coded[3] = {w, cw, cw};
      const int coded_h[3] = {h, ch, ch};
      for (int plane = intra_luma_searched ? 1 : 0; plane < 3; ++plane) {
        const auto params_opt = txv::make_tx_params(
            coded[plane], coded_h[plane], txv::TxType::kDctDct);
        if (!params_opt.has_value()) {
          error = "unsupported intra transform size";
          return false;
        }
        const txv::TxParams& params = *params_opt;
        const std::vector<std::int32_t>& plane_levels =
            intra_tx[static_cast<std::size_t>(plane)].levels;
        if (plane_levels.size() !=
            static_cast<std::size_t>(params.w) *
                static_cast<std::size_t>(params.h)) {
          error = "backend returned the wrong level count";
          return false;
        }
        int eob = 0;
        std::vector<std::uint16_t> scan;
        eob = eob_from_levels(plane_levels, params.w, params.h, &scan);
        entry.tx_size[plane] = static_cast<std::uint8_t>(params.tx_size);
        entry.eob[plane] = static_cast<std::uint32_t>(eob);
        for (int c2 = 0; c2 < eob; ++c2) {
          levels_.push_back(
              plane_levels[scan[static_cast<std::size_t>(c2)]]);
        }
      }
      update_mv_grid(r, c, w4, h4, 0, 0, inter::mc::Mv{0, 0});
      set_intra_joint(r, c, w4, h4,
                      av2enc::video::syntax::intra_joint_mode(
                          intra_y_mode, intra_y_angle_delta));
      ++intra_blocks_;
      if (trace_) {
        std::cerr << "m2 intra r=" << r << " c=" << c << " " << w << "x" << h
                  << " sse=" << intra_sse << "\n";
      }
    } else {
      if (!inter_residual_enabled && !prediction_ready) {
        if (best.compound) {
          if (!backend_.compound_420(best_job, inter_y, inter_u, inter_v,
                                     error)) {
            return false;
          }
        } else if (!predict_best(inter_y, inter_u, inter_v)) {
          return false;
        }
      }
      // The backend wrote the unadapted MC prediction into the device recon
      // planes; replace it with the BAWP-adapted samples so later intra
      // prediction in this frame reads the same reconstruction as the
      // decoder.
      if (bawp_applied &&
          !backend_.write_recon_420(x0, y0, w, h, inter_y, inter_u, inter_v,
                                    error)) {
        return false;
      }
      entry.is_inter = 1;
      entry.skip_flag = inter_residual ? 0 : 1;
      entry.inter_mode = static_cast<std::uint8_t>(best.mode);
      entry.drl_index = static_cast<std::uint8_t>(best.drl_index);
      entry.ref_index = static_cast<std::uint8_t>(best.ref_index);
      entry.compound = best.compound ? 1 : 0;
      entry.ref_index1 = 1;
      entry.compound_joint = 0;
      entry.compound_mode = 3;  // GLOBAL_GLOBALMV
      if (best.compound) {
        entry.compound_masked = best.compound_type != 2 ? 1 : 0;
        entry.compound_type = static_cast<std::uint8_t>(best.compound_type);
        entry.wedge_index = static_cast<std::uint8_t>(best.wedge_index);
        entry.wedge_sign = static_cast<std::uint8_t>(best.wedge_sign);
        entry.mask_type = static_cast<std::uint8_t>(best.mask_type);
      }
      entry.mv_y =
          best.mode == av2enc::video::syntax::kInterGlobalMv ? 0 : best.mv.y;
      entry.mv_x =
          best.mode == av2enc::video::syntax::kInterGlobalMv ? 0 : best.mv.x;
      entry.mvd_y = best.mvd.y;
      entry.mvd_x = best.mvd.x;
      entry.use_bawp = bawp_applied ? 1 : 0;
      entry.bawp_alpha = static_cast<std::int16_t>(bawp_fit.alpha);
      write_plane_block(recon_[0], x0, y0, w, h, inter_y);
      write_plane_block(recon_[1], cx0, cy0, cw, ch, inter_u);
      write_plane_block(recon_[2], cx0, cy0, cw, ch, inter_v);
      if (inter_residual) {
        if (!backend_.write_recon_420(x0, y0, w, h, inter_y, inter_u, inter_v,
                                      error)) {
          return false;
        }
        entry.levels_offset = static_cast<std::uint32_t>(levels_.size());
        if (res_luma_searched) {
          write_luma_search_plan(entry, res_luma_plan, levels_);
        }
        const int coded[3] = {w, cw, cw};
        const int coded_h[3] = {h, ch, ch};
        for (int plane = res_luma_searched ? 1 : 0; plane < 3; ++plane) {
          const auto params_opt = txv::make_tx_params(
              coded[plane], coded_h[plane], txv::TxType::kDctDct);
          if (!params_opt.has_value()) {
            error = "unsupported inter residual transform size";
            return false;
          }
          const txv::TxParams& params = *params_opt;
          const std::vector<std::int32_t>& plane_levels =
              res_tx[static_cast<std::size_t>(plane)].levels;
          int eob = 0;
          std::vector<std::uint16_t> scan;
          eob = eob_from_levels(plane_levels, params.w, params.h, &scan);
          entry.tx_size[plane] = static_cast<std::uint8_t>(params.tx_size);
          entry.eob[plane] = static_cast<std::uint32_t>(eob);
          for (int c2 = 0; c2 < eob; ++c2) {
            levels_.push_back(
                plane_levels[scan[static_cast<std::size_t>(c2)]]);
          }
        }
      }
      update_mv_grid(r, c, w4, h4, 1, best.ref_index, best.mv,
                     best.compound ? 1 : merge::kRefNone);
      set_intra_joint(r, c, w4, h4, 0);  // inter_block_mode_info(): DC
      ++inter_blocks_;
      if (trace_) {
        std::cerr << "m2 inter r=" << r << " c=" << c << " " << w << "x" << h
                  << " mode=" << static_cast<int>(best.mode)
                  << " drl=" << best.drl_index
                  << " ref=" << best.ref_index << " mv=(" << best.mv.y << ","
                  << best.mv.x << ") sse=" << inter_luma_sse
                  << " skip=" << static_cast<int>(entry.skip_flag)
                  << " comp=" << static_cast<int>(entry.compound)
                  << " ct=" << static_cast<int>(entry.compound_type)
                  << " wedge=" << static_cast<int>(entry.wedge_index)
                  << " sign=" << static_cast<int>(entry.wedge_sign)
                  << " mask=" << static_cast<int>(entry.mask_type) << "\n";
      }
    }

    // §6.19.2.3: this leaf's cells are now decoded for the following leaves.
    decoded_[0].fill(c, r, w4, h4);
    decoded_[1].fill(c / 2, r / 2, cw / 4, ch / 4);
    decoded_[2].fill(c / 2, r / 2, cw / 4, ch / 4);
    (void)node;
    return true;
  }

  // §7.13.2.15/.16 neighbour smooth flags at the coding-block origin, derived
  // from the IntraJointMode grid exactly like the CPU reference's
  // luma_neighbor_smooth().  Only --video-intra-full needs them.
  void luma_neighbor_smooth(int r, int c, bool& above_smooth,
                            bool& left_smooth) const {
    above_smooth = false;
    left_smooth = false;
    auto joint_is_smooth = [](int joint) { return joint >= 1 && joint <= 3; };
    if (r > 0 && c < mi_cols_) {
      above_smooth = joint_is_smooth(
          intra_joint_[static_cast<std::size_t>(r - 1) *
                           static_cast<std::size_t>(mi_cols_) +
                       static_cast<std::size_t>(c)]);
    }
    if (c > 0 && r < mi_rows_) {
      left_smooth = joint_is_smooth(
          intra_joint_[static_cast<std::size_t>(r) *
                           static_cast<std::size_t>(mi_cols_) +
                       static_cast<std::size_t>(c - 1)]);
    }
  }

  void set_intra_joint(int r, int c, int w4, int h4, int joint) {
    for (int yy = 0; yy < h4; ++yy) {
      const int rr = r + yy;
      if (rr >= mi_rows_) break;
      for (int xx = 0; xx < w4; ++xx) {
        const int cc = c + xx;
        if (cc >= mi_cols_) break;
        intra_joint_[static_cast<std::size_t>(rr) *
                         static_cast<std::size_t>(mi_cols_) +
                     static_cast<std::size_t>(cc)] =
            static_cast<std::uint8_t>(joint);
      }
    }
  }

  void update_mv_grid(int r, int c, int w4, int h4, int is_inter,
                      int ref_index, inter::mc::Mv mv,
                      int ref_index1 = merge::kRefNone) {
    for (int yy = 0; yy < h4; ++yy) {
      const int rr = r + yy;
      if (rr >= mi_rows_) break;
      for (int xx = 0; xx < w4; ++xx) {
        const int cc = c + xx;
        if (cc >= mi_cols_) break;
        inter::mc::MiInfo& mi =
            mv_grid_[static_cast<std::size_t>(rr) *
                         static_cast<std::size_t>(mi_cols_) +
                     static_cast<std::size_t>(cc)];
        mi = inter::mc::MiInfo{};
        // AV2 RefFrame labels used by the merge layer: 0..6 are reference
        // indices, 8 = INTRA.
        mi.ref[0] = is_inter ? static_cast<std::int8_t>(ref_index)
                             : merge::kIntraFrame;
        mi.ref[1] = static_cast<std::int8_t>(ref_index1);
        mi.mv[0][0] = mv.y;
        mi.mv[0][1] = mv.x;
        mi.mv[1][0] = mv.y;
        mi.mv[1][1] = mv.x;
        mi.is_inter = static_cast<std::int8_t>(is_inter);
        mi.w4 = static_cast<std::int8_t>(w4);
        mi.h4 = static_cast<std::int8_t>(h4);
        mi.col_base = static_cast<std::int16_t>(c);
        mi.row_base = static_cast<std::int16_t>(r);
      }
    }
  }

  detail::Backend& backend_;
  const VideoInputFrame& source_;
  const VideoOptions& options_;
  std::array<int, 7> frame_distance_{};
  int num_total_refs_ = 1;
  std::span<const std::uint16_t> cdf_seed_;
  std::vector<std::uint16_t>* cdf_final_ = nullptr;
  bool trace_ = false;
  bool keyframe_ = false;
  int tile_x_ = 0;
  int tile_y_ = 0;
  int frame_mi_cols_ = 0;
  int frame_mi_rows_ = 0;
  const GmFramePlan* gm_plan_ = nullptr;
  // Host reference frames for the §7.13.3.25 BAWP template fit.  The backend
  // owns the device copies; the fit only needs the visible planes.
  std::vector<const VideoReconFrame*> references_;
  bool begin_ok_ = true;
  int width_ = 0;
  int height_ = 0;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  FrameGeometry geometry_;
  partition::FrameDecision frame_;
  bool missing_cost_ = false;
  Plane recon_[3];
  av2enc::video::post::VideoFilterPlan filter_plan_;
  std::array<int, 7> ref_order_hints_{};
  std::vector<inter::mc::MiInfo> mv_grid_;
  std::vector<std::uint8_t> intra_joint_;
  std::array<av2enc::video::intra_search::BlockDecoded, 3> decoded_;
  std::vector<av2enc::video::syntax::BlockSyntax> blocks_;
  std::vector<std::int32_t> levels_;
  std::vector<detail::MeOutcome> me_results_;
  std::map<std::uint64_t, std::size_t> me_index_;
  std::uint32_t inter_blocks_ = 0;
  std::uint32_t intra_blocks_ = 0;
};

VideoReconFrame to_recon_frame(const Plane& y, const Plane& u, const Plane& v,
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
    std::memcpy(out.y.data() + static_cast<std::size_t>(row) *
                                   static_cast<std::size_t>(width),
                y.samples.data() + static_cast<std::size_t>(row) *
                                       static_cast<std::size_t>(y.stride),
                width);
  }
  for (std::uint32_t row = 0; row < ch; ++row) {
    std::memcpy(out.u.data() + static_cast<std::size_t>(row) *
                                   static_cast<std::size_t>(cw),
                u.samples.data() + static_cast<std::size_t>(row) *
                                       static_cast<std::size_t>(u.stride),
                cw);
    std::memcpy(out.v.data() + static_cast<std::size_t>(row) *
                                   static_cast<std::size_t>(cw),
                v.samples.data() + static_cast<std::size_t>(row) *
                                       static_cast<std::size_t>(v.stride),
                cw);
  }
  return out;
}

int relative_dist(std::uint32_t a, std::uint32_t b, int bits) {
  const int modulus = 1 << bits;
  int diff = static_cast<int>((a - b) & static_cast<std::uint32_t>(modulus - 1));
  if (diff >= (modulus >> 1)) diff -= modulus;
  return diff;
}

VideoConfig make_config(std::uint32_t width, std::uint32_t height,
                     const VideoOptions& options) {
  VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = 4;
  config.num_ref_frames = static_cast<std::uint8_t>(
      av2enc::video::resolved_num_ref_frames(options));
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_avg_cdf = options.avg_cdf;
  config.avg_cdf_type = true;
  config.enable_masked_compound =
      options.compound && (options.compound_wedge || options.compound_diffwtd);
  set_config_drl_reorder(config, opt_drl_reorder(options));
  config.enable_cfl_intra = true;
  config.enable_mhccp = options.enable_mhccp;
  config.using_qmatrix = options.enable_qm;
  config.enable_deblock = opt_video_deblock(options);
  config.enable_global_motion = options.enable_gme;
  config.enable_bawp = options.enable_bawp;
  // §5.4.10 opt-in M2 loop filters.
  if (options.loop_filters.cdef) {
    config.enable_cdef = true;
    config.cdef_on_skip_txfm_mode =
        options.loop_filters.cdef_on_skip_txfm_mode;
    config.cdef_on_skip_txfm_always_on =
        options.loop_filters.cdef_on_skip_txfm_mode == 1;
    config.cdef_on_skip_txfm_disabled =
        options.loop_filters.cdef_on_skip_txfm_mode == 0;
  }
  if (options.loop_filters.ccso) {
    config.enable_ccso = true;
    config.ccso_unit_matches_sb_size = true;
  }
  if (options.loop_filters.gdf) {
    config.enable_gdf = true;
    config.gdf_unit_matches_sb_size = true;
  }
  if (options.loop_filters.wiener || options.loop_filters.chroma_lr ||
      options.loop_filters.pc_wiener) {
    config.enable_restoration = true;
    config.lr_disable_luma_pc_wiener = !options.loop_filters.pc_wiener;
    config.lr_disable_luma_wiener_ns = !options.loop_filters.wiener;
    config.lr_uv_present = options.loop_filters.chroma_lr;
    config.lr_disable_chroma_wiener_ns = !options.loop_filters.chroma_lr;
  }  config.tx_mode_select = opt_enable_tx_search(options);  if (options.mfh) config.cur_mfh_id = 1;
  if (options.video_timing) {
    // §5.4.1 decoder-compatible timing block.  The nested §5.4.13 block stays
    // off: pinned dav2d f9dafbc cannot parse it (docs/negative-results.md).
    config.initial_display_delay_present = true;
    config.initial_display_delay_minus_1 = 0;
    config.decoder_model_info_present = true;
    config.num_units_in_decoding_tick =
        options.timing_num_units_in_display_tick;
  }
  return config;
}

}  // namespace

namespace detail {
namespace {

using av2enc::video::VideoConfig;
using av2enc::video::VideoFrameHeader;
using av2enc::video::VideoInputFrame;
using av2enc::video::VideoOptions;
using av2enc::video::VideoReconFrame;
using av2enc::video::VideoSequenceResult;

// Fills the per-frame header fields from a B/temporal plan.  The syntax/header
// writers are the CPU reference's; only the tile payload is device-produced.
VideoFrameHeader make_plan_header(bool keyframe, std::uint32_t order_hint,
                                  int num_refs, const int* ref_slots,
                                  std::uint32_t refresh_flags,
                                  bool immediate_output,
                                  bool implicit_output, bool reference_select,
                                  const VideoOptions& options) {
  VideoFrameHeader header;
  header.keyframe = keyframe;
  header.order_hint = order_hint;
  header.num_total_refs = static_cast<std::uint8_t>(num_refs);
  for (int r = 0; r < num_refs; ++r) {
    header.ref_frame_idx[static_cast<std::size_t>(r)] =
        static_cast<std::uint8_t>(ref_slots[r]);
  }
  header.ref_slot = num_refs > 0 ? header.ref_frame_idx[0] : 0;
  header.refresh_frame_flags = refresh_flags;
  header.immediate_output_frame = immediate_output;
  header.implicit_output_frame = implicit_output;
  header.reference_select = reference_select;
  header.use_qtr_precision_mv = options.mv_precision == 1;
  header.allow_high_precision_mv = options.mv_precision != 2;
  return header;
}

bool common_gop_checks(std::span<const VideoInputFrame> frames,
                       const VideoOptions& options, std::string& error) {
  (void)options;  // per-mode option checks live in the plan-specific wrappers
  if (frames.empty()) {
    error = "no input frames";
    return false;
  }
  const std::uint32_t width = frames[0].width;
  const std::uint32_t height = frames[0].height;
  if ((width & 7U) != 0 || (height & 7U) != 0) {
    error = "M2 requires width/height divisible by 8";
    return false;
  }
  for (const VideoInputFrame& f : frames) {
    if (f.width != width || f.height != height) {
      error = "all frames must have the same dimensions";
      return false;
    }
  }
  return true;
}

// Multi-tile key + flat P chain.  This mirrors src/pipeline/video/video_frame.cpp:
// every tile is an independent tile-local picture (fresh entropy state,
// tile-local coded geometry and reconstruction, cleared prediction contexts)
// while ME/MC read the whole-frame reference.  The tile payloads are then
// framed by the CPU post writer, so header bits and entropy bytes have exactly
// one implementation.
bool encode_tiled_sequence_gpu(Backend& backend,
                               std::span<const VideoInputFrame> frames,
                               const VideoOptions& options,
                               VideoSequenceResult& result, std::string& error) {
  namespace post = ::av2enc::video::post;
  if (!common_gop_checks(frames, options, error)) return false;
  if (options.num_total_refs != 1) {
    error = "--tiles requires --video-refs 1: the tiled frame header signals "
            "a single explicit reference slot";
    return false;
  }
  if (options.mv_precision != 0) {
    error = "--tiles requires eighth-pel MV precision (the tiled frame header "
            "writer signals the 1/8-pel pair)";
    return false;
  }
  if (options.cdf_continuity) {
    error = "GPU M2 baseline does not implement cross-frame CDF carry";
    return false;
  }
  if (options.enable_tip) {
    // The TIP frame schedule needs the §7.9/§7.10 + §7.22 device path
    // (src/cuda/codec/inter/tip/) assembled in this orchestration.  Reject by
    // name so the API cannot silently produce a stream without TIP.
    error = "--video-tip is not implemented on the GPU M2 backend yet; use "
            "--video for the CPU reference path";
    return false;
  }
  const int width = static_cast<int>(frames[0].width);
  const int height = static_cast<int>(frames[0].height);
  const post::VideoTilePlan planned = post::plan_video_tiles(
      width, height, options.tile_cols, options.tile_rows,
      tiles::kSuperblockPixels64, 7, 0);
  if (!planned.ok) {
    error = "--tiles " + std::to_string(options.tile_cols) + "x" +
            std::to_string(options.tile_rows) + " rejected: " + planned.error;
    return false;
  }
  const tiles::TileLayout& layout = planned.layout;
  post::VideoPostSequenceOptions tile_sequence;
  // Whole-frame filter pass: the decoder must filter shared tile edges too.
  // No filters: keep the historical per-tile independence bit.  With filters
  // the picture-global pass filters the shared edges, so the decoder must too.
  tile_sequence.filters.disable_loopfilters_across_tiles =
      !options.loop_filters.any();
  tile_sequence.filters.enable_cdef = options.loop_filters.cdef;
  tile_sequence.filters.enable_gdf = options.loop_filters.gdf;
  tile_sequence.filters.enable_restoration =
      options.loop_filters.wiener || options.loop_filters.chroma_lr;
  tile_sequence.filters.lr_disable_luma_pc_wiener = true;
  tile_sequence.filters.lr_disable_luma_wiener_ns =
      !options.loop_filters.wiener;
  tile_sequence.filters.lr_uv_present = options.loop_filters.chroma_lr;
  tile_sequence.filters.lr_disable_chroma_wiener_ns =
      !options.loop_filters.chroma_lr;
  tile_sequence.filters.enable_ccso = options.loop_filters.ccso;
  tile_sequence.filters.ccso_unit_matches_sb_size = true;
  tile_sequence.filters.gdf_unit_matches_sb_size = true;
  tile_sequence.filters.cdef_on_skip_txfm_always_on =
      options.loop_filters.cdef_on_skip_txfm_mode == 1;
  tile_sequence.filters.cdef_on_skip_txfm_disabled =
      options.loop_filters.cdef_on_skip_txfm_mode == 0;
  tile_sequence.enable_avg_cdf = options.avg_cdf;
  tile_sequence.avg_cdf_type = true;

  auto config = make_config(static_cast<std::uint32_t>(width),
                            static_cast<std::uint32_t>(height), options);

  // §8.2.4 average-CDF / §7.5 frame-end bookkeeping, one fresh state per tile.
  const bool cdf_chain = options.avg_cdf || options.adaptive_cdf;
  std::vector<std::uint16_t> cdf_tile;
  std::vector<std::uint16_t> cdf_scratch;
  std::vector<std::uint16_t> cdf_frame;
  std::vector<std::uint16_t> cdf_saved;
  std::vector<std::uint16_t> cdf_defaults;
  std::vector<entropy::avg_cdf::CdfTable> cdf_bank_tables;
  entropy::avg_cdf::CdfBank cdf_bank;
  std::array<std::vector<std::uint16_t>, 8> cdf_slots;
  if (cdf_chain) {
    const std::size_t words = av2enc::video::syntax::cdf_image_words();
    cdf_frame.assign(words, 0);
    cdf_tile.assign(words, 0);
    cdf_saved.assign(words, 0);
    cdf_defaults.assign(words, 0);
    if (!av2enc::video::syntax::cdf_image_defaults_for_base_q_idx(
            static_cast<int>(options.base_q_idx), cdf_defaults, &error)) {
      return false;
    }
    if (!av2enc::video::syntax::cdf_image_tables(
            cdf_frame, cdf_tile, cdf_saved, cdf_defaults, cdf_bank_tables,
            &error)) {
      return false;
    }
    entropy::avg_cdf::CdfBankConfig bank_config;
    bank_config.enable_avg_cdf = options.avg_cdf;
    bank_config.avg_cdf_type = true;
    bank_config.context_update_tile_id = 0;
    if (!cdf_bank.init(cdf_bank_tables, bank_config, &error)) {
      error = "average-CDF bank: " + error;
      return false;
    }
    cdf_bank.begin_frame(/*reset_to_defaults=*/true);
  }

  const auto frame_qp = [&options](std::size_t index) -> std::uint8_t {
    return options.frame_base_q_idx.empty()
               ? options.base_q_idx
               : options.frame_base_q_idx[index];
  };
  const auto options_for_frame = [&options, &frame_qp](std::size_t index) {
    VideoOptions copy = options;
    copy.base_q_idx = frame_qp(index);
    copy.frame_base_q_idx.clear();
    return copy;
  };

  // Full-frame reconstruction planes.  Each tile's coded-size output is pasted
  // into them once per frame; they are the decoder-visible frame and the next
  // P frame's reference.
  const int frame_mi_cols = static_cast<int>(2U * ((width + 7U) >> 3U));
  const int frame_mi_rows = static_cast<int>(2U * ((height + 7U) >> 3U));
  Plane frame_recon[3];
  frame_recon[0].resize(frame_mi_cols * 4, frame_mi_rows * 4, 0);
  frame_recon[1].resize(frame_mi_cols * 2, frame_mi_rows * 2, 0);
  frame_recon[2].resize(frame_mi_cols * 2, frame_mi_rows * 2, 0);
  const auto paste_tile = [](const Plane& src, Plane& dst, int x0, int y0,
                             int w, int h) {
    for (int y = 0; y < h; ++y) {
      std::memcpy(dst.samples.data() +
                      static_cast<std::size_t>(y0 + y) *
                          static_cast<std::size_t>(dst.stride) +
                      static_cast<std::size_t>(x0),
                  src.samples.data() +
                      static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(src.stride),
                  static_cast<std::size_t>(w));
    }
  };

  const std::size_t tile_count = layout.tiles.size();
  std::vector<std::vector<std::uint8_t>> payloads;
  payloads.reserve(tile_count);

  // ---- keyframe -----------------------------------------------------------
  {
    const std::array<int, 7> no_distance{};
    const VideoOptions key_options = options_for_frame(0);
    std::vector<std::uint16_t> cdf_seed;
    if (cdf_chain) cdf_seed = cdf_tile;
    std::uint32_t key_payload_bytes = 0;
    std::uint32_t key_intra_blocks = 0;
    std::uint32_t cdf_tile_index = 0;
    payloads.clear();
    std::vector<std::unique_ptr<GpuFrameEncoder>> key_encoders;
    key_encoders.reserve(tile_count);
    for (const tiles::TileRect& tile : layout.tiles) {
      if (cdf_chain) cdf_tile = cdf_seed;
      auto encoder = std::make_unique<GpuFrameEncoder>(
          backend, frames[0], std::span<const VideoReconFrame* const>{},
          no_distance, 1, key_options, options.trace, /*keyframe=*/true, error,
          tile.x0, tile.y0, tile.pixel_width(), tile.pixel_height(),
          cdf_chain ? std::span<const std::uint16_t>(cdf_tile)
                    : std::span<const std::uint16_t>(),
          cdf_chain ? &cdf_scratch : nullptr);
      if (!encoder->begin_ok()) {
        error = "keyframe tile " + std::to_string(cdf_tile_index) +
                " backend setup failed: " + error;
        return false;
      }
      if (!encoder->decide(error) || !encoder->deblock(error)) {
        error = "keyframe encode failed: " + error;
        return false;
      }
      key_intra_blocks += encoder->intra_blocks();
      paste_tile(encoder->recon(0), frame_recon[0], tile.x0, tile.y0,
                 tile.pixel_width(), tile.pixel_height());
      paste_tile(encoder->recon(1), frame_recon[1], tile.x0 / 2, tile.y0 / 2,
                 tile.pixel_width() / 2, tile.pixel_height() / 2);
      paste_tile(encoder->recon(2), frame_recon[2], tile.x0 / 2, tile.y0 / 2,
                 tile.pixel_width() / 2, tile.pixel_height() / 2);
      key_encoders.push_back(std::move(encoder));
    }
    post::VideoFilterPlan key_filters;
    if (options.loop_filters.any()) {
      post::VideoLoopFilterFrame lf;
      lf.width = width;
      lf.height = height;
      lf.mi_cols = frame_mi_cols;
      lf.mi_rows = frame_mi_rows;
      lf.frame_is_intra = true;
      lf.base_q_idx = key_options.base_q_idx;
      lf.num_total_refs = 0;
      lf.restricted_oh = -1;
      std::vector<post::VideoLoopFilterBlock> filter_blocks;
      for (std::size_t t = 0; t < layout.tiles.size(); ++t) {
        const tiles::TileRect& tile = layout.tiles[t];
        const int tile_mi_row = tile.y0 / 4;
        const int tile_mi_col = tile.x0 / 4;
        for (const av2enc::video::syntax::BlockSyntax& leaf :
             key_encoders[t]->blocks()) {
          if (leaf.partition !=
              static_cast<std::uint8_t>(PartitionType::kNone)) {
            continue;
          }
          post::VideoLoopFilterBlock block;
          block.mi_row = tile_mi_row + static_cast<int>(leaf.mi_row);
          block.mi_col = tile_mi_col + static_cast<int>(leaf.mi_col);
          block.w4 = av2enc::partition::block_width(
                         static_cast<av2enc::partition::BlockSize>(
                             leaf.bsize)) >> 2;
          block.h4 = av2enc::partition::block_height(
                         static_cast<av2enc::partition::BlockSize>(
                             leaf.bsize)) >> 2;
          block.skip = false;
          filter_blocks.push_back(block);
        }
      }
      lf.blocks = filter_blocks;
      for (int plane = 0; plane < 3; ++plane) {
        lf.plane[plane] = frame_recon[plane].samples.data();
        lf.stride[plane] = frame_recon[plane].stride;
        lf.source[plane] =
            plane == 0 ? frames[0].y.data()
                       : (plane == 1 ? frames[0].u.data() : frames[0].v.data());
        lf.source_stride[plane] =
            plane == 0 ? static_cast<int>(width)
                       : static_cast<int>(width / 2);
      }
      BackendFilterHooks hooks(backend);
      if (!post::run_video_loop_filters(lf, options.loop_filters,
                                        &key_filters, &hooks, &error)) {
        error = "keyframe: " + error;
        return false;
      }
    }
    for (std::size_t t = 0; t < layout.tiles.size(); ++t) {
      std::vector<std::uint8_t> payload;
      std::vector<av2enc::video::syntax::BlockSyntax> blocks;
      std::vector<std::int32_t> levels;
      if (cdf_chain) cdf_tile = cdf_seed;
      if (!key_encoders[t]->write(payload, blocks, levels,
                                  key_filters.any() ? &key_filters : nullptr,
                                  error)) {
        error = "keyframe encode failed: " + error;
        return false;
      }
      if (payload.empty()) {
        error = "keyframe produced an empty tile payload";
        return false;
      }
      if (cdf_chain) {
        std::copy(cdf_scratch.begin(), cdf_scratch.end(), cdf_tile.begin());
        if (!cdf_bank.save_tile_cdfs(
                static_cast<std::uint32_t>(t),
                static_cast<std::uint32_t>(tile_count))) {
          error = "average-CDF save_tile_cdfs rejected the keyframe tile";
          return false;
        }
      }
      key_payload_bytes += static_cast<std::uint32_t>(payload.size());
      payloads.push_back(std::move(payload));
    }
    if (cdf_chain) {
      cdf_bank.average_at_frame_end();
      for (std::vector<std::uint16_t>& slot : cdf_slots) {
        slot.assign(cdf_frame.begin(), cdf_frame.end());
      }
    }

    config.base_q_idx = frame_qp(0);
    post::VideoTiledFrameHeader header;
    header.keyframe = true;
    header.order_hint = 0;
    header.ref_slot = 0;
    header.refresh_frame_flags = (1U << config.num_ref_frames) - 1U;
    header.base_q_idx = config.base_q_idx;
    header.tx_mode_select = options.enable_tx_search;
    header.cdef = post::make_tiled_cdef_params(key_filters.cdef);
    header.ccso = post::make_tiled_ccso_params(key_filters.ccso);
    header.lr = key_filters.lr;
    header.gdf = key_filters.gdf;
    header.cdef_frame_enable = key_filters.cdef.enabled;
    header.ccso_frame_flag = key_filters.ccso.enabled;
    post::VideoTiledFrame frame;
    if (!post::assemble_video_tile_group_from_payloads(
            config, tile_sequence, header, layout, payloads, &frame, &error)) {
      error = "keyframe: " + error;
      return false;
    }
    const std::vector<std::uint8_t> td =
        av2enc::video::make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> seq =
        post::make_video_sequence_header_obu(config, tile_sequence, &error);
    if (seq.empty()) {
      error = "keyframe: tiled sequence header: " + error;
      return false;
    }
    std::vector<std::uint8_t> unit;
    unit.reserve(td.size() + seq.size() + frame.obu.size());
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), seq.begin(), seq.end());
    unit.insert(unit.end(), frame.obu.begin(), frame.obu.end());
    result.units.push_back(std::move(unit));
    result.recon.push_back(
        to_recon_frame(frame_recon[0], frame_recon[1], frame_recon[2],
                       static_cast<std::uint32_t>(width),
                       static_cast<std::uint32_t>(height)));
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(key_payload_bytes);
    result.inter_blocks.push_back(0);
    result.intra_blocks.push_back(key_intra_blocks);
  }

  std::array<VideoReconFrame, 8> slot_recon;
  std::array<std::uint32_t, 8> slot_order{};
  std::vector<int> recent_slots;
  slot_recon[0] = result.recon.back();
  slot_order[0] = 0;
  recent_slots.push_back(0);

  for (std::size_t i = 1; i < frames.size(); ++i) {
    const std::uint32_t order_hint = static_cast<std::uint32_t>(i) & 0xFU;
    const int ref_count = std::min<int>(
        options.num_total_refs, static_cast<int>(recent_slots.size()));
    std::vector<const VideoReconFrame*> references;
    references.reserve(static_cast<std::size_t>(ref_count));
    std::array<int, 7> frame_distance{};
    for (int r = 0; r < ref_count; ++r) {
      const int slot = recent_slots[static_cast<std::size_t>(r)];
      references.push_back(&slot_recon[static_cast<std::size_t>(slot)]);
      frame_distance[static_cast<std::size_t>(r)] =
          relative_dist(order_hint, slot_order[static_cast<std::size_t>(slot)],
                        4);
    }
    const VideoOptions frame_options = options_for_frame(i);
    std::array<int, 7> ref_order_hints{};
    for (int r = 0; r < ref_count && r < 2; ++r) {
      ref_order_hints[static_cast<std::size_t>(r)] = static_cast<int>(
          slot_order[static_cast<std::size_t>(
              recent_slots[static_cast<std::size_t>(r)])]);
    }
    if (cdf_chain) cdf_bank.begin_frame(/*reset_to_defaults=*/true);
    std::vector<std::uint16_t> cdf_seed;
    if (cdf_chain) cdf_seed = cdf_tile;
    std::uint32_t payload_bytes = 0;
    std::uint32_t inter_blocks = 0;
    std::uint32_t intra_blocks = 0;
    std::uint32_t cdf_tile_index = 0;
    payloads.clear();
    std::vector<std::unique_ptr<GpuFrameEncoder>> p_encoders;
    p_encoders.reserve(tile_count);
    for (const tiles::TileRect& tile : layout.tiles) {
      if (cdf_chain) cdf_tile = cdf_seed;
      auto encoder = std::make_unique<GpuFrameEncoder>(
          backend, frames[i], references, frame_distance, ref_count,
          frame_options, options.trace, /*keyframe=*/false, error, tile.x0,
          tile.y0, tile.pixel_width(), tile.pixel_height(),
          cdf_chain ? std::span<const std::uint16_t>(cdf_tile)
                    : std::span<const std::uint16_t>(),
          cdf_chain ? &cdf_scratch : nullptr, nullptr, ref_order_hints);
      if (!encoder->begin_ok()) {
        error = "P-frame " + std::to_string(i) + " tile " +
                std::to_string(cdf_tile_index) + " setup failed: " + error;
        return false;
      }
      if (!encoder->decide(error) || !encoder->deblock(error)) {
        error = "P-frame " + std::to_string(i) + ": " + error;
        return false;
      }
      inter_blocks += encoder->inter_blocks();
      intra_blocks += encoder->intra_blocks();
      paste_tile(encoder->recon(0), frame_recon[0], tile.x0, tile.y0,
                 tile.pixel_width(), tile.pixel_height());
      paste_tile(encoder->recon(1), frame_recon[1], tile.x0 / 2, tile.y0 / 2,
                 tile.pixel_width() / 2, tile.pixel_height() / 2);
      paste_tile(encoder->recon(2), frame_recon[2], tile.x0 / 2, tile.y0 / 2,
                 tile.pixel_width() / 2, tile.pixel_height() / 2);
      p_encoders.push_back(std::move(encoder));
    }
    post::VideoFilterPlan frame_filters;
    if (options.loop_filters.any()) {
      post::VideoLoopFilterFrame lf;
      lf.width = width;
      lf.height = height;
      lf.mi_cols = frame_mi_cols;
      lf.mi_rows = frame_mi_rows;
      lf.frame_is_intra = false;
      lf.base_q_idx = frame_options.base_q_idx;
      lf.num_total_refs = ref_count;
      lf.restricted_oh = -1;
      for (int r = 0; r < 2; ++r) {
        lf.ref_order_hints[r] = ref_order_hints[static_cast<std::size_t>(r)];
        lf.frame_distance[r] = frame_distance[static_cast<std::size_t>(r)];
      }
      std::vector<post::VideoLoopFilterBlock> filter_blocks;
      for (std::size_t t = 0; t < layout.tiles.size(); ++t) {
        const tiles::TileRect& tile = layout.tiles[t];
        const int tile_mi_row = tile.y0 / 4;
        const int tile_mi_col = tile.x0 / 4;
        for (const av2enc::video::syntax::BlockSyntax& leaf :
             p_encoders[t]->blocks()) {
          if (leaf.partition !=
              static_cast<std::uint8_t>(PartitionType::kNone)) {
            continue;
          }
          post::VideoLoopFilterBlock block;
          block.mi_row = tile_mi_row + static_cast<int>(leaf.mi_row);
          block.mi_col = tile_mi_col + static_cast<int>(leaf.mi_col);
          block.w4 = av2enc::partition::block_width(
                         static_cast<av2enc::partition::BlockSize>(
                             leaf.bsize)) >> 2;
          block.h4 = av2enc::partition::block_height(
                         static_cast<av2enc::partition::BlockSize>(
                             leaf.bsize)) >> 2;
          block.skip = leaf.is_inter && leaf.skip_flag != 0;
          filter_blocks.push_back(block);
        }
      }
      lf.blocks = filter_blocks;
      for (int plane = 0; plane < 3; ++plane) {
        lf.plane[plane] = frame_recon[plane].samples.data();
        lf.stride[plane] = frame_recon[plane].stride;
        lf.source[plane] =
            plane == 0 ? frames[i].y.data()
                       : (plane == 1 ? frames[i].u.data() : frames[i].v.data());
        lf.source_stride[plane] =
            plane == 0 ? static_cast<int>(width)
                       : static_cast<int>(width / 2);
      }
      BackendFilterHooks hooks(backend);
      if (!post::run_video_loop_filters(lf, options.loop_filters,
                                        &frame_filters, &hooks, &error)) {
        error = "P-frame " + std::to_string(i) + ": " + error;
        return false;
      }
    }
    for (std::size_t t = 0; t < layout.tiles.size(); ++t) {
      std::vector<std::uint8_t> payload;
      std::vector<av2enc::video::syntax::BlockSyntax> blocks;
      std::vector<std::int32_t> levels;
      if (cdf_chain) cdf_tile = cdf_seed;
      if (!p_encoders[t]->write(
              payload, blocks, levels,
              frame_filters.any() ? &frame_filters : nullptr, error)) {
        error = "P-frame " + std::to_string(i) + ": " + error;
        return false;
      }
      if (cdf_chain) {
        std::copy(cdf_scratch.begin(), cdf_scratch.end(), cdf_tile.begin());
        if (!cdf_bank.save_tile_cdfs(
                static_cast<std::uint32_t>(t),
                static_cast<std::uint32_t>(tile_count))) {
          error = "average-CDF save_tile_cdfs rejected the P-frame tile";
          return false;
        }
      }
      payload_bytes += static_cast<std::uint32_t>(payload.size());
      payloads.push_back(std::move(payload));
    }
    const int write_slot = static_cast<int>(
        static_cast<unsigned>(i) % static_cast<unsigned>(config.num_ref_frames));
    if (cdf_chain) {
      cdf_bank.average_at_frame_end();
      std::vector<std::uint16_t>& saved_slot =
          cdf_slots[static_cast<std::size_t>(write_slot)];
      std::copy(cdf_frame.begin(), cdf_frame.end(), saved_slot.begin());
    }

    config.base_q_idx = frame_qp(i);
    post::VideoTiledFrameHeader header;
    header.keyframe = false;
    header.order_hint = order_hint;
    header.ref_slot =
        static_cast<std::uint8_t>(recent_slots[static_cast<std::size_t>(0)]);
    header.refresh_frame_flags =
        1U << (static_cast<unsigned>(i) %
               static_cast<unsigned>(config.num_ref_frames));
    header.base_q_idx = config.base_q_idx;
    header.tx_mode_select = options.enable_tx_search;
    header.cdef = post::make_tiled_cdef_params(frame_filters.cdef);
    header.ccso = post::make_tiled_ccso_params(frame_filters.ccso);
    header.lr = frame_filters.lr;
    header.gdf = frame_filters.gdf;
    header.cdef_frame_enable = frame_filters.cdef.enabled;
    header.ccso_frame_flag = frame_filters.ccso.enabled;
    post::VideoTiledFrame frame;
    if (!post::assemble_video_tile_group_from_payloads(
            config, tile_sequence, header, layout, payloads, &frame, &error)) {
      error = "P-frame " + std::to_string(i) + ": " + error;
      return false;
    }
    const std::vector<std::uint8_t> td =
        av2enc::video::make_temporal_delimiter_obu();
    std::vector<std::uint8_t> unit;
    unit.reserve(td.size() + frame.obu.size());
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), frame.obu.begin(), frame.obu.end());
    result.units.push_back(std::move(unit));

    result.recon.push_back(
        to_recon_frame(frame_recon[0], frame_recon[1], frame_recon[2],
                       static_cast<std::uint32_t>(width),
                       static_cast<std::uint32_t>(height)));
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(payload_bytes);
    result.inter_blocks.push_back(inter_blocks);
    result.intra_blocks.push_back(intra_blocks);

    slot_recon[static_cast<std::size_t>(write_slot)] = result.recon.back();
    slot_order[static_cast<std::size_t>(write_slot)] = order_hint;
    recent_slots.erase(
        std::remove(recent_slots.begin(), recent_slots.end(), write_slot),
        recent_slots.end());
    recent_slots.insert(recent_slots.begin(), write_slot);
    if (recent_slots.size() > 7) recent_slots.resize(7);
  }

  result.frame_output_index.resize(frames.size());
  for (std::size_t i = 0; i < frames.size(); ++i) {
    result.frame_output_index[i] = static_cast<std::int32_t>(i);
  }
  return true;
}

}  // namespace

bool encode_b_sequence_gpu(Backend& backend,
                           std::span<const VideoInputFrame> frames,
                           const VideoOptions& options,
                           VideoSequenceResult& result, std::string& error) {
  namespace bf = ::av2enc::inter::bframes;
  if (!common_gop_checks(frames, options, error)) return false;
  if (!options.frame_base_q_idx.empty()) {
    error = "the GPU B-frame plan requires a fixed base_q_idx";
    return false;
  }
  const int gop_size = options.b_frames + 2;
  if (gop_size > 16) {
    error = "--b-frames supports at most 6 B frames per GOP";
    return false;
  }
  if (frames.size() != static_cast<std::size_t>(gop_size)) {
    error = "--b-frames codes one closed GOP: pass exactly " +
            std::to_string(gop_size) + " frames (N B frames + key + P anchor)";
    return false;
  }
  bf::SequenceConfig sc;
  sc.num_gops = static_cast<int>(frames.size()) / gop_size;
  sc.gop_size = gop_size;
  sc.pattern = bf::GopPattern::kIpbb;
  sc.order_hint_bits = 4;
  sc.num_slots = 16;
  sc.base_q_idx = options.base_q_idx;
  sc.leading_pictures = 0;
  if (sc.num_gops != 1) {
    error = "--b-frames codes exactly one GOP";
    return false;
  }
  bf::Plan plan;
  bf::PlanError plan_error = bf::PlanError::kOk;
  if (!bf::build_sequence(sc, plan, &plan_error)) {
    error = std::string("B-frame schedule: ") + bf::plan_error_name(plan_error);
    return false;
  }
  if (plan.display_count() != static_cast<int>(frames.size())) {
    error = "B-frame schedule display count mismatch";
    return false;
  }
  bf::ValidationReport report;
  if (!bf::validate_plan(plan, &report)) {
    error = std::string("B-frame plan invalid: ") +
            bf::plan_error_name(report.error) + " (coding " +
            std::to_string(report.error_coding_index) + ", display " +
            std::to_string(report.error_display_index) + ")";
    return false;
  }

  VideoConfig config;
  const int width = static_cast<int>(frames[0].width);
  const int height = static_cast<int>(frames[0].height);
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = static_cast<std::uint8_t>(plan.order_hint_bits);
  config.num_ref_frames = static_cast<std::uint8_t>(plan.num_slots);
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_avg_cdf = options.avg_cdf;
  config.avg_cdf_type = true;
  config.disable_drl_reorder = true;
  config.constrain_drl_reorder = true;
  config.enable_cfl_intra = true;
  config.enable_mhccp = options.enable_mhccp;
  config.using_qmatrix = options.enable_qm;
  config.enable_deblock = opt_video_deblock(options);
  av2enc::video::apply_video_loop_filter_sequence_config(
      config, options.loop_filters);
  config.tx_mode_select = opt_enable_tx_search(options);
  config.enable_bawp = options.enable_bawp;
  config.max_tlayer_id = 0;
  config.monotonic_output_order_flag = false;

  const int mi_cols = 2 * static_cast<int>((frames[0].width + 7U) >> 3U);
  const int mi_rows = 2 * static_cast<int>((frames[0].height + 7U) >> 3U);
  std::vector<VideoReconFrame> recon_by_display(
      static_cast<std::size_t>(plan.display_count()));
  VideoOptions frame_options = options;
  frame_options.temporal_layers = 0;
  frame_options.tlayer_drop = -1;

  bool wrote_sequence_header = false;
  for (std::size_t ci = 0; ci < plan.coding_order.size(); ++ci) {
    const int display = plan.coding_order[ci];
    const bf::FramePlan* fp = plan.frame_at_display(display);
    if (fp == nullptr) {
      error = "B-frame plan is missing a display index";
      return false;
    }
    const int num_refs = fp->num_total_refs;
    std::vector<const VideoReconFrame*> references;
    std::array<int, 7> frame_distance{};
    for (int r = 0; r < num_refs; ++r) {
      const int ref_display = fp->ref_display[r];
      if (ref_display < 0 ||
          recon_by_display[static_cast<std::size_t>(ref_display)].y.empty()) {
        error = "B-frame reference is not available";
        return false;
      }
      references.push_back(
          &recon_by_display[static_cast<std::size_t>(ref_display)]);
      frame_distance[static_cast<std::size_t>(r)] =
          relative_dist(static_cast<std::uint32_t>(fp->display_order_hint),
                        static_cast<std::uint32_t>(fp->ref_order_hint[r]),
                        plan.order_hint_bits);
    }

    std::array<int, 7> ref_order_hints{};
    for (int r = 0; r < num_refs; ++r) {
      ref_order_hints[static_cast<std::size_t>(r)] = fp->ref_order_hint[r];
    }
    GpuFrameEncoder encoder(
        backend, frames[static_cast<std::size_t>(display)], references,
        frame_distance, std::max(1, num_refs), frame_options, options.trace,
        fp->is_key, error, 0, 0, 0, 0, {}, nullptr, nullptr, ref_order_hints);
    if (!encoder.begin_ok()) {
      error = "B-frame display " + std::to_string(display) +
              " backend setup failed: " + error;
      return false;
    }
    std::vector<std::uint8_t> tile;
    std::vector<av2enc::video::syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    if (!encoder.encode(tile, blocks, levels, error)) {
      error = "B-frame display " + std::to_string(display) + ": " + error;
      return false;
    }

    const bf::FrameHeaderFields hf = bf::header_fields(*fp);
    int ref_slots[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int r = 0; r < hf.num_total_refs; ++r) {
      ref_slots[r] = hf.ref_frame_idx[r];
    }
    VideoFrameHeader header = make_plan_header(
        hf.keyframe, hf.order_hint_lsbs, hf.num_total_refs, ref_slots,
        hf.refresh_frame_flags, hf.immediate_output_frame,
        hf.implicit_output_frame,
        hf.reference_select && options.compound && hf.num_total_refs > 1,
        options);
    av2enc::video::apply_video_loop_filter_frame_header(
        encoder.filter_plan(), header);

    std::vector<std::uint8_t> unit;
    const std::vector<std::uint8_t> td =
        av2enc::video::make_temporal_delimiter_obu();
    unit.insert(unit.end(), td.begin(), td.end());
    if (!wrote_sequence_header) {
      const std::vector<std::uint8_t> seq =
          av2enc::video::make_sequence_header_obu(config);
      unit.insert(unit.end(), seq.begin(), seq.end());
      wrote_sequence_header = true;
    }
    const std::vector<std::uint8_t> obu =
        av2enc::video::make_frame_obu(config, header, tile, 0);
    unit.insert(unit.end(), obu.begin(), obu.end());
    result.units.push_back(std::move(unit));

    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(static_cast<std::uint32_t>(tile.size()));
    result.inter_blocks.push_back(encoder.inter_blocks());
    result.intra_blocks.push_back(encoder.intra_blocks());
    recon_by_display[static_cast<std::size_t>(display)] = to_recon_frame(
        encoder.recon(0), encoder.recon(1), encoder.recon(2),
        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
  }

  for (int display = 0; display < plan.display_count(); ++display) {
    result.recon.push_back(recon_by_display[static_cast<std::size_t>(display)]);
  }
  result.frame_output_index.resize(plan.coding_order.size());
  for (std::size_t ci = 0; ci < plan.coding_order.size(); ++ci) {
    result.frame_output_index[ci] = plan.coding_order[ci];
  }
  (void)mi_cols;
  (void)mi_rows;
  return true;
}

bool encode_temporal_sequence_gpu(Backend& backend,
                                  std::span<const VideoInputFrame> frames,
                                  const VideoOptions& options,
                                  VideoSequenceResult& result,
                                  std::string& error) {
  namespace tl = ::av2enc::inter::tlayers;
  if (!common_gop_checks(frames, options, error)) return false;
  if (!options.frame_base_q_idx.empty()) {
    error = "the GPU temporal-layer plan requires a fixed base_q_idx";
    return false;
  }
  const int gop = options.temporal_layers;
  if (frames.size() != static_cast<std::size_t>(gop)) {
    error = "--video-tlayers codes one closed GOP: pass exactly " +
            std::to_string(gop) + " input frames";
    return false;
  }
  tl::PlanConfig pc;
  pc.gop_size = gop;
  pc.schedule = tl::Schedule::kHierarchicalP;
  pc.dependency = tl::DependencyMode::kAllLower;
  pc.explicit_dependency_map = false;
  pc.order_hint_bits = 4;
  pc.num_slots = gop;
  pc.base_q_idx = options.base_q_idx;
  tl::Plan plan;
  tl::PlanError plan_error = tl::PlanError::kOk;
  if (!tl::build_plan(pc, plan, &plan_error)) {
    error = std::string("temporal-layer plan: ") +
            tl::plan_error_name(plan_error);
    return false;
  }
  tl::PlanError validation_error = tl::PlanError::kOk;
  int bad_display = -1;
  int bad_coding = -1;
  if (!tl::validate_plan(plan, &validation_error, &bad_display, &bad_coding)) {
    error = std::string("temporal-layer plan invalid: ") +
            tl::plan_error_name(validation_error) + " (display " +
            std::to_string(bad_display) + ", coding " +
            std::to_string(bad_coding) + ")";
    return false;
  }
  if (plan.display_count() != gop) {
    error = "temporal-layer plan display count mismatch";
    return false;
  }

  std::vector<int> coding = plan.coding_order;
  if (options.tlayer_drop >= 0) {
    if (options.tlayer_drop > plan.max_tlayer_id) {
      error = "--video-tlayer-drop is above the plan's max_tlayer_id";
      return false;
    }
    if (!tl::validate_layer_drop(plan, options.tlayer_drop, &plan_error)) {
      error = std::string("temporal-layer drop is not decodable: ") +
              tl::plan_error_name(plan_error);
      return false;
    }
    coding = plan.coding_order_up_to_layer(options.tlayer_drop);
    if (coding.empty()) {
      error = "--video-tlayer-drop removes every coded frame";
      return false;
    }
  }

  VideoConfig config;
  const int width = static_cast<int>(frames[0].width);
  const int height = static_cast<int>(frames[0].height);
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = static_cast<std::uint8_t>(pc.order_hint_bits);
  config.num_ref_frames = static_cast<std::uint8_t>(plan.num_slots);
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_avg_cdf = options.avg_cdf;
  config.avg_cdf_type = true;
  config.disable_drl_reorder = true;
  config.constrain_drl_reorder = true;
  config.enable_cfl_intra = true;
  config.enable_mhccp = options.enable_mhccp;
  config.using_qmatrix = options.enable_qm;
  config.enable_deblock = opt_video_deblock(options);
  av2enc::video::apply_video_loop_filter_sequence_config(
      config, options.loop_filters);
  config.tx_mode_select = opt_enable_tx_search(options);
  config.enable_bawp = options.enable_bawp;
  config.max_tlayer_id = static_cast<std::uint8_t>(plan.max_tlayer_id);
  config.monotonic_output_order_flag = false;
  config.tlayer_dependency_present_flag = plan.tlayer_dependency_present_flag;
  for (int curr = 0; curr <= plan.max_tlayer_id; ++curr) {
    for (int ref = 0; ref <= plan.max_tlayer_id; ++ref) {
      config.tlayer_dependency[curr][ref] =
          plan.tlayer_dependency[static_cast<std::size_t>(curr)]
                                [static_cast<std::size_t>(ref)];
    }
  }

  std::vector<VideoReconFrame> recon_by_display(
      static_cast<std::size_t>(plan.display_count()));
  VideoOptions frame_options = options;
  frame_options.temporal_layers = 0;
  frame_options.tlayer_drop = -1;

  bool wrote_sequence_header = false;
  for (std::size_t ci = 0; ci < coding.size(); ++ci) {
    const int display = coding[ci];
    const tl::PlannedFrame& pf =
        plan.frames[static_cast<std::size_t>(display)];
    if (pf.is_key && ci != 0) {
      error = "temporal-layer plan codes a key outside coding position 0";
      return false;
    }

    std::vector<const VideoReconFrame*> references;
    std::array<int, 7> frame_distance{};
    if (!pf.is_key) {
      const int ref_display = pf.ref_display[0];
      if (ref_display < 0 ||
          recon_by_display[static_cast<std::size_t>(ref_display)].y.empty()) {
        error = "temporal-layer reference is not available";
        return false;
      }
      references.push_back(
          &recon_by_display[static_cast<std::size_t>(ref_display)]);
      frame_distance[0] = relative_dist(
          pf.display_order_hint, pf.ref_order_hint[0],
          static_cast<int>(pc.order_hint_bits));
    }

    std::array<int, 7> ref_order_hints{};
    ref_order_hints[0] = pf.ref_order_hint[0];
    GpuFrameEncoder encoder(backend, frames[static_cast<std::size_t>(display)],
                            references, frame_distance, 1, frame_options,
                            options.trace, pf.is_key, error, 0, 0, 0, 0, {},
                            nullptr, nullptr, ref_order_hints);
    if (!encoder.begin_ok()) {
      error = "temporal-layer frame " + std::to_string(display) +
              " backend setup failed: " + error;
      return false;
    }
    std::vector<std::uint8_t> tile;
    std::vector<av2enc::video::syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    if (!encoder.encode(tile, blocks, levels, error)) {
      error = "temporal-layer frame " + std::to_string(display) + ": " + error;
      return false;
    }

    int ref_slots[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int r = 0; r < pf.num_refs; ++r) {
      ref_slots[r] = pf.ref_slot[r];
    }
    VideoFrameHeader header = make_plan_header(
        pf.is_key, pf.order_hint_lsbs, pf.num_refs, ref_slots,
        pf.refresh_frame_flags, pf.immediate_output_frame,
        pf.implicit_output_frame, false, options);
    av2enc::video::apply_video_loop_filter_frame_header(
        encoder.filter_plan(), header);

    std::vector<std::uint8_t> unit;
    const std::vector<std::uint8_t> td =
        av2enc::video::make_temporal_delimiter_obu();
    unit.insert(unit.end(), td.begin(), td.end());
    if (!wrote_sequence_header) {
      const std::vector<std::uint8_t> seq =
          av2enc::video::make_sequence_header_obu(config);
      unit.insert(unit.end(), seq.begin(), seq.end());
      wrote_sequence_header = true;
    }
    const std::vector<std::uint8_t> obu = av2enc::video::make_frame_obu(
        config, header, tile, static_cast<std::uint8_t>(pf.obu_tlayer_id()));
    unit.insert(unit.end(), obu.begin(), obu.end());
    result.units.push_back(std::move(unit));

    result.tile_bytes.push_back(static_cast<std::uint32_t>(tile.size()));
    result.inter_blocks.push_back(encoder.inter_blocks());
    result.intra_blocks.push_back(encoder.intra_blocks());
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    recon_by_display[static_cast<std::size_t>(display)] = to_recon_frame(
        encoder.recon(0), encoder.recon(1), encoder.recon(2),
        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
  }

  std::vector<int> retained(coding);
  std::sort(retained.begin(), retained.end());
  std::vector<std::int32_t> rank(static_cast<std::size_t>(plan.display_count()),
                                 -1);
  for (std::size_t i = 0; i < retained.size(); ++i) {
    const int display = retained[i];
    rank[static_cast<std::size_t>(display)] = static_cast<std::int32_t>(i);
    result.recon.push_back(recon_by_display[static_cast<std::size_t>(display)]);
  }
  result.frame_output_index.resize(coding.size());
  for (std::size_t ci = 0; ci < coding.size(); ++ci) {
    result.frame_output_index[ci] =
        rank[static_cast<std::size_t>(coding[ci])];
  }
  return true;
}

}  // namespace detail

bool encode_video_sequence_gpu(std::span<const VideoInputFrame> frames,
                            const VideoOptions& options, VideoSequenceResult& result,
                            std::string& error, VideoGpuStats* stats,
                            av2enc::cuda::Arena* arena) {
  error.clear();
  result = VideoSequenceResult{};
  if (frames.empty()) {
    error = "no input frames";
    return false;
  }
  // Non-default formats use the depth/chroma-generic core with its own CUDA
  // backend.  The historical 8-bit 4:2:0 pipeline below is untouched.
  if (options.format.chroma != fmt::ChromaFormat::k420 ||
      options.format.depth != fmt::BitDepth::k8) {
#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
    return hbd_gpu::encode_video_sequence_format_gpu(frames, options, result,
                                                     error, arena);
#else
    error = "format video requires the CUDA backend";
    return false;
#endif
  }
  if (opt_video_deblock(options) && detail::is_frame_type_mode(options)) {
    error = "--video-deblock does not compose with the frame-type wrappers "
            "(long-term/OLK/leading/SEF/SWITCH/RAS)";
    return false;
  }
  if (detail::is_frame_type_mode(options)) {
    return detail::encode_special_sequence_gpu(frames, options, result, error,
                                               stats);
  }
  const std::uint32_t width = frames[0].width;
  const std::uint32_t height = frames[0].height;
  if (width == 0 || height == 0 || (width & 7U) != 0 || (height & 7U) != 0) {
    error = "M2 requires non-zero width/height divisible by 8";
    return false;
  }
  if (width > 0xFFFFU || height > 0xFFFFU) {
    error = "M2 dimensions exceed the sequence header field width";
    return false;
  }
  if (options.base_q_idx == 0) {
    error = "M2 does not support lossless base_q_idx = 0";
    return false;
  }
  if (!options.frame_base_q_idx.empty() &&
      options.frame_base_q_idx.size() != frames.size()) {
    error = "frame_base_q_idx must be empty or have one entry per frame";
    return false;
  }
  for (const std::uint8_t qp : options.frame_base_q_idx) {
    if (qp == 0) {
      error = "frame_base_q_idx entries must be in [1, 255]";
      return false;
    }
  }
  const auto frame_qp = [&options](std::size_t index) -> std::uint8_t {
    return options.frame_base_q_idx.empty()
               ? options.base_q_idx
               : options.frame_base_q_idx[index];
  };
  // Per-frame base_q_idx is supported: the frame header, the coefficient-CDF
  // slice (config.base_q_idx below) and the device transform quantizer
  // (begin_frame's base_q_idx -> qlookup) all follow the same value.
  // The GPU pipeline mirrors the CPU reference model.  Every option the CPU
  // uses to build a different sequence/ref model that this backend has not
  // implemented is rejected by name here as well as in the CLI, so a caller
  // can never get bytes from a mode this backend did not implement.
  if (options.num_total_refs < 1 ||
      options.num_total_refs > av2enc::video::kRefsPerFrame) {
    error = "num_total_refs must be in [1, " +
            std::to_string(av2enc::video::kRefsPerFrame) + "]";
    return false;
  }
  if (options.num_ref_frames != 0 &&
      (options.num_ref_frames < 4 ||
       options.num_ref_frames > av2enc::video::kMaxRefSlots)) {
    error = "num_ref_frames must be 0 (auto) or in [4, " +
            std::to_string(av2enc::video::kMaxRefSlots) + "]";
    return false;
  }
  if (options.num_ref_frames != 0 &&
      options.num_total_refs > options.num_ref_frames) {
    error = "num_total_refs " + std::to_string(options.num_total_refs) +
            " exceeds NumRefFrames " + std::to_string(options.num_ref_frames);
    return false;
  }
  if (options.long_term || options.olk || options.leading_frames != 0 ||
      options.sef || options.ra_switch || options.ras) {
    error = "GPU M2 baseline does not implement the frame-type wrappers here";
    return false;
  }
  if ((options.loop_filters.wiener || options.loop_filters.chroma_lr) &&
      (options.tile_cols > 1 || options.tile_rows > 1)) {
    error = "video loop restoration (Wiener/chroma LR) does not compose with "
            "--tiles > 1x1 yet";
    return false;
  }
  if (options.tile_cols > 1 || options.tile_rows > 1) {
    if (options.b_frames > 0 || options.temporal_layers > 0) {
      error = "--tiles does not compose with --b-frames/--video-tlayers";
      return false;
    }
  }
  if (options.cdf_continuity) {
    error = "GPU M2 baseline does not implement cross-frame CDF carry";
    return false;
  }
  if ((options.avg_cdf || options.adaptive_cdf) &&
      !options.frame_base_q_idx.empty()) {
    // §8.2.4 seeds the frame CDF image from one base_q_idx; the CPU reference
    // rejects the same pair.
    error = "average-CDF/continuity requires a fixed base_q_idx "
            "(no --rc crf/vbr per-frame QP)";
    return false;
  }
  if (opt_drl_reorder(options) != 0) {
    error = "GPU M2 baseline keeps DRL reordering off (not decoder-gated)";
    return false;
  }
  if (options.mv_precision < 0 || options.mv_precision > 2) {
    error = "mv_precision must be 0 (1/8), 1 (1/4) or 2 (1/2) pel";
    return false;
  }
  if (options.enable_gme &&
      (options.enable_inter_residual || options.compound ||
       options.b_frames != 0 || options.temporal_layers != 0)) {
    error = "GPU global motion is gated on the plain skip-only P chain";
    return false;
  }
  if (opt_drl_reorder(options) < 0 || opt_drl_reorder(options) > 2) {
    error = "M2 DrlReorder must be 0 (disabled), 1 (constraint) or 2 (always)";
    return false;
  }
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const VideoInputFrame& f = frames[i];
    if (f.width != width || f.height != height) {
      error = "all frames must have the same dimensions";
      return false;
    }
    const std::size_t luma = static_cast<std::size_t>(width) * height;
    const std::size_t chroma = static_cast<std::size_t>((width + 1U) / 2U) *
                               ((height + 1U) / 2U);
    if (f.y.size() < luma || f.u.size() < chroma || f.v.size() < chroma) {
      error = "input frame planes are too small";
      return false;
    }
  }

  if (options.b_frames > 0 || options.temporal_layers > 0) {
    std::unique_ptr<detail::Backend> gop_backend = detail::make_backend(arena);
    if (!gop_backend) {
      error = "no M2 backend linked";
      return false;
    }
    gop_backend->reset_stats();
    const bool ok = options.b_frames > 0
                        ? detail::encode_b_sequence_gpu(
                              *gop_backend, frames, options, result, error)
                        : detail::encode_temporal_sequence_gpu(
                              *gop_backend, frames, options, result, error);
    if (!ok) return false;
    if (stats != nullptr) {
      *stats = gop_backend->stats();
      std::uint64_t inter = 0;
      std::uint64_t intra = 0;
      for (std::size_t i = 0; i < result.inter_blocks.size(); ++i) {
        inter += result.inter_blocks[i];
        intra += result.intra_blocks[i];
      }
      stats->inter_blocks = inter;
      stats->intra_blocks = intra;
    }
    return true;
  }

  std::unique_ptr<detail::Backend> backend = detail::make_backend(arena);
  if (!backend) {
    error = "no M2 backend linked";
    return false;
  }
  backend->reset_stats();
  backend->set_qm(opt_enable_qm(options));

  if (options.tile_cols > 1 || options.tile_rows > 1) {
    if (!detail::encode_tiled_sequence_gpu(*backend, frames, options, result,
                                           error)) {
      return false;
    }
    if (stats != nullptr) {
      *stats = backend->stats();
      std::uint64_t inter = 0;
      std::uint64_t intra = 0;
      for (std::size_t i = 0; i < result.inter_blocks.size(); ++i) {
        inter += result.inter_blocks[i];
        intra += result.intra_blocks[i];
      }
      stats->inter_blocks = inter;
      stats->intra_blocks = intra;
    }
    return true;
  }

  VideoConfig config = make_config(width, height, options);

  // §8.2.4 average-CDF / §7.5 frame-end bookkeeping.  The decision path and
  // tile writer are the CPU reference's; only the seed/final CDF images are
  // threaded through the device backend so they stay host-side serial state.
  const bool cdf_chain = options.avg_cdf || options.adaptive_cdf;
  std::vector<std::uint16_t> cdf_tile;
  std::vector<std::uint16_t> cdf_scratch;
  std::vector<std::uint16_t> cdf_frame;
  std::vector<std::uint16_t> cdf_saved;
  std::vector<std::uint16_t> cdf_defaults;
  std::vector<entropy::avg_cdf::CdfTable> cdf_bank_tables;
  entropy::avg_cdf::CdfBank cdf_bank;
  std::array<std::vector<std::uint16_t>, 8> cdf_slots;
  if (cdf_chain) {
    const std::size_t words = av2enc::video::syntax::cdf_image_words();
    cdf_frame.assign(words, 0);
    cdf_tile.assign(words, 0);
    cdf_saved.assign(words, 0);
    cdf_defaults.assign(words, 0);
    if (!av2enc::video::syntax::cdf_image_defaults_for_base_q_idx(
            static_cast<int>(options.base_q_idx), cdf_defaults, &error)) {
      return false;
    }
    if (!av2enc::video::syntax::cdf_image_tables(
            cdf_frame, cdf_tile, cdf_saved, cdf_defaults, cdf_bank_tables,
            &error)) {
      return false;
    }
    entropy::avg_cdf::CdfBankConfig bank_config;
    bank_config.enable_avg_cdf = options.avg_cdf;
    bank_config.avg_cdf_type = true;
    bank_config.context_update_tile_id = 0;
    if (!cdf_bank.init(cdf_bank_tables, bank_config, &error)) {
      error = "average-CDF bank: " + error;
      return false;
    }
    cdf_bank.begin_frame(/*reset_to_defaults=*/true);
  }

  // Frame 0: intra keyframe (DC intra + DCT), mirroring the CPU reference.
  {
    const std::array<int, 7> no_distance{};
    VideoOptions key_options = options;
    key_options.base_q_idx = frame_qp(0);
    key_options.frame_base_q_idx.clear();
    GpuFrameEncoder key_encoder(
        *backend, frames[0],
        std::span<const VideoReconFrame* const>{}, no_distance, 1, key_options,
        options.trace, /*keyframe=*/true, error, 0, 0, 0, 0,
        cdf_chain ? std::span<const std::uint16_t>(cdf_tile)
                  : std::span<const std::uint16_t>(),
        cdf_chain ? &cdf_scratch : nullptr);
    if (!key_encoder.begin_ok()) {
      error = "keyframe backend setup failed: " + error;
      return false;
    }
    std::vector<std::uint8_t> key_tile;
    std::vector<av2enc::video::syntax::BlockSyntax> key_blocks;
    std::vector<std::int32_t> key_levels;
    if (!key_encoder.encode(key_tile, key_blocks, key_levels, error)) {
      error = "keyframe encode failed: " + error;
      return false;
    }
    if (key_tile.empty()) {
      error = "keyframe produced an empty tile payload";
      return false;
    }
    if (cdf_chain) {
      // exit_symbol() -> §8.2.4 Saved* average (§7.5 runs after the last tile).
      std::copy(cdf_scratch.begin(), cdf_scratch.end(), cdf_tile.begin());
      if (!cdf_bank.save_tile_cdfs(0, 1)) {
        error = "average-CDF save_tile_cdfs rejected the keyframe tile";
        return false;
      }
      cdf_bank.average_at_frame_end();
      for (std::vector<std::uint16_t>& slot : cdf_slots) {
        slot.assign(cdf_frame.begin(), cdf_frame.end());
      }
    }

    VideoFrameHeader key_header;
    key_header.keyframe = true;
    key_header.order_hint = 0;
    key_header.ref_slot = 0;
    key_header.refresh_frame_flags = (1U << config.num_ref_frames) - 1U;
    key_header.use_qtr_precision_mv = options.mv_precision == 1;
    key_header.allow_high_precision_mv = options.mv_precision != 2;
    key_header.cdef = key_encoder.filter_plan().cdef;
    key_header.ccso = key_encoder.filter_plan().ccso;
    key_header.lr = key_encoder.filter_plan().lr;
    key_header.gdf = key_encoder.filter_plan().gdf;

    config.base_q_idx = frame_qp(0);

    const std::vector<std::uint8_t> td =
        av2enc::video::make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> seq =
        av2enc::video::make_sequence_header_obu(config);
    std::vector<std::uint8_t> ci;
    if (options.video_timing) {
      // §5.15 content interpretation OBU carrying §5.4.12 timing_info().  It is
      // the first OBU of the coded output frame unit after the sequence header.
      av2enc::ci::ContentInterpretation params =
          av2enc::ci::default_content_interpretation();
      params.has_timing_info = true;
      params.timing.num_units_in_display_tick =
          options.timing_num_units_in_display_tick;
      params.timing.time_scale = options.timing_time_scale;
      params.timing.equal_picture_interval = true;
      params.timing.num_ticks_per_picture_minus_1 = 0;
      ci = av2enc::ci::make_content_interpretation_obu(params, error);
      if (ci.empty()) {
        error = "content interpretation (timing) OBU: " + error;
        return false;
      }
    }
    std::vector<std::uint8_t> mfh_obu;
    if (options.mfh) {
      // §5.7: one minimal MFH (sequence header 0, id 1, no frame size /
      // deblocking update / segmentation table).  Availability precedes the
      // first frame header that references it (§7.3.8.7).
      av2enc::frame::mfh::MultiFrameHeader header;
      header.seq_header_id = 0;
      header.mfh_id = 1;
      header.frame_size_present = false;
      header.deblocking_filter_update = false;
      header.seg_info_present = false;
      std::string mfh_error;
      mfh_obu =
          av2enc::frame::mfh::make_multi_frame_header_obu(header, &mfh_error);
      if (mfh_obu.empty()) {
        error = "MFH OBU: " + mfh_error;
        return false;
      }
    }
    const std::vector<std::uint8_t> obu =
        av2enc::video::make_frame_obu(config, key_header, key_tile);
    std::vector<std::uint8_t> unit;
    unit.reserve(td.size() + seq.size() + ci.size() + mfh_obu.size() +
                 obu.size());
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), seq.begin(), seq.end());
    unit.insert(unit.end(), ci.begin(), ci.end());
    unit.insert(unit.end(), mfh_obu.begin(), mfh_obu.end());
    unit.insert(unit.end(), obu.begin(), obu.end());

    result.units.push_back(std::move(unit));
    result.recon.push_back(to_recon_frame(
        key_encoder.recon(0), key_encoder.recon(1), key_encoder.recon(2),
        width, height));
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(static_cast<std::uint32_t>(key_tile.size()));
    result.inter_blocks.push_back(0);
    result.intra_blocks.push_back(key_encoder.intra_blocks());
  }

  // Decoded picture buffer.  This mirrors the CPU reference exactly: the key
  // frame refreshes every slot, `recent_slots` is kept most-recent-first, and
  // each P frame references the first `num_total_refs` entries and refreshes
  // slot `i % NumRefFrames`.
  std::array<VideoReconFrame, 8> slot_recon;
  std::array<std::uint32_t, 8> slot_order{};
  std::vector<int> recent_slots;
  slot_recon[0] = result.recon.back();
  slot_order[0] = 0;
  recent_slots.push_back(0);

  for (std::size_t i = 1; i < frames.size(); ++i) {
    const std::uint32_t order_hint =
        static_cast<std::uint32_t>(i) & 0xFU;
    const int ref_count = std::min<int>(
        options.num_total_refs, static_cast<int>(recent_slots.size()));
    std::vector<const VideoReconFrame*> references;
    references.reserve(static_cast<std::size_t>(ref_count));
    std::array<int, 7> frame_distance{};
    for (int r = 0; r < ref_count; ++r) {
      const int slot = recent_slots[static_cast<std::size_t>(r)];
      references.push_back(&slot_recon[static_cast<std::size_t>(slot)]);
      frame_distance[static_cast<std::size_t>(r)] =
          relative_dist(order_hint, slot_order[static_cast<std::size_t>(slot)],
                        4);
    }
    VideoOptions frame_options = options;
    frame_options.base_q_idx = frame_qp(i);
    frame_options.frame_base_q_idx.clear();

    // §5.18.9 global motion: collect the deterministic block-MV grid and run
    // the fit on the backend (device estimator on the CUDA path, the CPU
    // reference on the emulation path).  Both produce the same quantised model.
    GmFramePlan gm_plan;
    const GmFramePlan* gm_ptr = nullptr;
    if (options.enable_gme) {
      std::vector<inter::mc::RefFrame> gm_refs;
      gm_refs.reserve(static_cast<std::size_t>(ref_count));
      std::vector<int> ref_order_hint(static_cast<std::size_t>(ref_count));
      std::vector<std::uint8_t> ref_slots(static_cast<std::size_t>(ref_count));
      std::vector<int> ref_distance(static_cast<std::size_t>(ref_count));
      for (int r = 0; r < ref_count; ++r) {
        const int slot = recent_slots[static_cast<std::size_t>(r)];
        const VideoReconFrame& recon =
            slot_recon[static_cast<std::size_t>(slot)];
        inter::mc::RefFrame gm_ref;
        gm_ref.plane[0] =
            inter::mc::PlaneView{recon.y.data(), static_cast<int>(width)};
        gm_ref.plane[1] =
            inter::mc::PlaneView{recon.u.data(), static_cast<int>(width / 2)};
        gm_ref.plane[2] =
            inter::mc::PlaneView{recon.v.data(), static_cast<int>(width / 2)};
        gm_ref.mi_cols = 2 * static_cast<int>((width + 7U) >> 3U);
        gm_ref.mi_rows = 2 * static_cast<int>((height + 7U) >> 3U);
        gm_refs.push_back(gm_ref);
        ref_order_hint[static_cast<std::size_t>(r)] =
            static_cast<int>(slot_order[static_cast<std::size_t>(slot)]);
        ref_slots[static_cast<std::size_t>(r)] =
            static_cast<std::uint8_t>(slot);
        ref_distance[static_cast<std::size_t>(r)] =
            frame_distance[static_cast<std::size_t>(r)];
      }
      std::vector<std::vector<inter::gme::GmBlockMv>> points =
          av2enc::video::collect_gm_block_mvs(frames[i], gm_refs,
                                              frame_options, error);
      if (points.size() != gm_refs.size()) {
        error = "P-frame global motion: block-MV collection failed: " + error;
        return false;
      }
      const inter::gme::GmOptions gm_options =
          av2enc::video::make_gm_options(frames[i], frame_options);
      std::vector<inter::gme::GmEstimate> estimates;
      estimates.reserve(points.size());
      for (const std::vector<inter::gme::GmBlockMv>& ref_points : points) {
        inter::gme::GmEstimate estimate;
        if (!backend->gme_estimate(ref_points, gm_options, estimate, error)) {
          return false;
        }
        estimates.push_back(std::move(estimate));
      }
      gm_plan = av2enc::video::make_gm_frame_plan(estimates, ref_order_hint, ref_slots,
                                   ref_distance);
      gm_ptr = &gm_plan;
    }
    if (cdf_chain) cdf_bank.begin_frame(/*reset_to_defaults=*/true);
    std::vector<std::uint16_t> cdf_seed;
    if (cdf_chain) cdf_seed = cdf_tile;
    // §7.20.1 reference OrderHints for the GDF table derivation.
    std::array<int, 7> ref_order_hints{};
    for (int r = 0; r < ref_count && r < 2; ++r) {
      ref_order_hints[static_cast<std::size_t>(r)] = static_cast<int>(
          slot_order[static_cast<std::size_t>(
              recent_slots[static_cast<std::size_t>(r)])]);
    }
    GpuFrameEncoder encoder(*backend, frames[i], references, frame_distance,
                            ref_count, frame_options, options.trace,
                            /*keyframe=*/false, error, 0, 0, 0, 0,
                            cdf_chain ? std::span<const std::uint16_t>(cdf_seed)
                                      : std::span<const std::uint16_t>(),
                            cdf_chain ? &cdf_scratch : nullptr, gm_ptr,
                            ref_order_hints);
    if (!encoder.begin_ok()) {
      error = "P-frame " + std::to_string(i) + " setup failed: " + error;
      return false;
    }
    std::vector<std::uint8_t> tile;
    std::vector<av2enc::video::syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    if (!encoder.encode(tile, blocks, levels, error)) {
      error = "P-frame " + std::to_string(i) + ": " + error;
      return false;
    }
    const int write_slot = static_cast<int>(
        static_cast<unsigned>(i) % static_cast<unsigned>(config.num_ref_frames));
    if (cdf_chain) {
      std::copy(cdf_scratch.begin(), cdf_scratch.end(), cdf_tile.begin());
      if (!cdf_bank.save_tile_cdfs(0, 1)) {
        error = "average-CDF save_tile_cdfs rejected the P-frame tile";
        return false;
      }
      cdf_bank.average_at_frame_end();
      std::vector<std::uint16_t>& saved_slot =
          cdf_slots[static_cast<std::size_t>(write_slot)];
      std::copy(cdf_frame.begin(), cdf_frame.end(), saved_slot.begin());
    }

    VideoFrameHeader header;
    header.keyframe = false;
    header.order_hint = order_hint;
    if (cdf_chain) {
      header.signal_primary_ref_frame = false;
      header.disable_cross_frame_cdf_init = true;
      header.primary_ref_frame = 0;
    }
    header.num_total_refs = static_cast<std::uint8_t>(ref_count);
    for (int r = 0; r < ref_count; ++r) {
      header.ref_frame_idx[static_cast<std::size_t>(r)] =
          static_cast<std::uint8_t>(recent_slots[static_cast<std::size_t>(r)]);
    }
    header.ref_slot = header.ref_frame_idx[0];
    // §7.23 rotating refresh slot: the CPU reference writes slot i modulo
    // NumRefFrames, so the refresh mask must be the same bit.
    header.refresh_frame_flags =
        1U << (static_cast<unsigned>(i) %
               static_cast<unsigned>(config.num_ref_frames));
    header.use_qtr_precision_mv = options.mv_precision == 1;
    header.allow_high_precision_mv = options.mv_precision != 2;
    header.reference_select = options.compound && ref_count > 1;
    header.gm = gm_plan;
    header.cdef = encoder.filter_plan().cdef;
    header.ccso = encoder.filter_plan().ccso;
    header.lr = encoder.filter_plan().lr;
    header.gdf = encoder.filter_plan().gdf;
    // SEF wrapper: code the last picture hidden so the following SEF unit can
    // re-show it (the decoder output count is unchanged).
    header.immediate_output_frame =
        !(options.hide_last_picture && i + 1U == frames.size());

    config.base_q_idx = frame_qp(i);
    const std::vector<std::uint8_t> td =
        av2enc::video::make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> obu =
        av2enc::video::make_frame_obu(config, header, tile);
    std::vector<std::uint8_t> unit;
    unit.reserve(td.size() + obu.size());
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), obu.begin(), obu.end());
    result.units.push_back(std::move(unit));

    result.recon.push_back(to_recon_frame(
        encoder.recon(0), encoder.recon(1), encoder.recon(2), width, height));
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(static_cast<std::uint32_t>(tile.size()));
    result.inter_blocks.push_back(encoder.inter_blocks());
    result.intra_blocks.push_back(encoder.intra_blocks());

    slot_recon[static_cast<std::size_t>(write_slot)] = result.recon.back();
    slot_order[static_cast<std::size_t>(write_slot)] = order_hint;
    recent_slots.erase(
        std::remove(recent_slots.begin(), recent_slots.end(), write_slot),
        recent_slots.end());
    recent_slots.insert(recent_slots.begin(), write_slot);
    if (recent_slots.size() > 7) recent_slots.resize(7);
  }

  // Default path: every coded picture is output in decode order.  A hidden
  // last picture (SEF wrapper) maps to -1; the following SEF unit carries the
  // display entry.
  result.frame_output_index.resize(frames.size());
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const bool hidden =
        options.hide_last_picture && i + 1U == frames.size();
    result.frame_output_index[i] =
        hidden ? -1 : static_cast<std::int32_t>(i);
  }

  if (stats != nullptr) {
    *stats = backend->stats();
    std::uint64_t inter = 0;
    std::uint64_t intra = 0;
    for (std::size_t i = 0; i < result.inter_blocks.size(); ++i) {
      inter += result.inter_blocks[i];
      intra += result.intra_blocks[i];
    }
    stats->inter_blocks = inter;
    stats->intra_blocks = intra;
  }
  return true;
}

const char* video_backend_name() {
  static const char* name = detail::make_backend_name();
  return name;
}

}  // namespace av2enc::cuda::video
