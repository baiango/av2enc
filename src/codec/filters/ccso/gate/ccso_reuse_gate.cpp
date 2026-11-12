// Temporal CCSO reuse gate: 3-frame M2 stream (key + two P frames).
//
// Frame 0 (CLK) and frame 1 (P) signal fresh CCSO parameters and fresh
// CcsoBlks.  Frame 2 (P) signals reuse_ccso[plane] = 1 and
// sb_reuse_ccso[plane] = 1, selects ccso_ref_idx = 0, and codes no CCSO
// parameters or ccso_blk symbols at all.  Every frame's decoded output must
// match the encoder's own §7.19 reconstruction exactly.
//
// A second, no-reuse control stream re-signals frame 1's parameters in frame 2
// and codes the block flags again.  It must decode to the same pictures; the
// reuse stream must be smaller.
//
// The M2 core has no read_ccso() hook and is not edited.  vendor_video_syntax.py
// copies src/encoder/m2/video_syntax.cpp and injects exactly one §5.20.10.2 call
// per CCSO unit origin.  The frame header is written here because
// m2/post/tiled_frame.cpp delegates ccso_params() to the filters_stream writer,
// which rejects reuse by design.
//
// CCSO units are 256x256 luma samples: ccso_unit_matches_sb_size = 0 and a
// single tile whose start is 0 select CcsoLumaSizeLog2 = 8.  §6.17.7.8 only
// permits sb_reuse_ccso when CcsoLumaSizeLog2 == CCSO_LUMA_SIZE_LOG2 (8).
//
// Usage:
//   ccso_reuse_gate [--width 256] [--height 192] [--out /tmp/ccso_reuse.ivf]
//                   [--decoder tools/bin/dav2d-macos]
//
// Exit: 0 pass, 1 fail, 77 decoder unavailable.

#include "codec/filters/ccso/ccso_reuse.h"

#include "codec/filters/loop_restoration/restoration.h"
#include "pipeline/video/video_headers.h"
#include "pipeline/video/video_syntax.h"
#include "pipeline/video/post/loop_filters.h"
#include "pipeline/video/post/tiled_frame.h"
#include "pipeline/tiles/tile_group.h"
#include "pipeline/tiles/tile_layout.h"

#include "core/bitstream/bitwriter.h"
#include "core/bitstream/basic_headers.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "video_syntax_ccso.h"

namespace {

namespace cr = av2enc::ccso_reuse;
namespace fs = av2enc::still::filters_stream;
namespace post = av2enc::video::post;
namespace rest = av2enc::restoration;
using av2enc::BitWriter;
using av2enc::video::VideoConfig;
using av2enc::video::syntax::BlockSyntax;
using av2enc::video::syntax::FramePlan;

constexpr int kPlanes = 3;

struct Frame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y, u, v;
  int cw() const { return (width + 1) / 2; }
  int ch() const { return (height + 1) / 2; }
  std::vector<std::uint8_t>& plane(int p) {
    return p == 0 ? y : (p == 1 ? u : v);
  }
  const std::vector<std::uint8_t>& plane(int p) const {
    return p == 0 ? y : (p == 1 ? u : v);
  }
  int plane_width(int p) const { return p == 0 ? width : cw(); }
  int plane_height(int p) const { return p == 0 ? height : ch(); }
};

std::string find_decoder() {
  if (const char* env = std::getenv("DAV2D")) {
    if (std::filesystem::exists(env)) return env;
  }
  for (const char* candidate :
       {"tools/bin/dav2d", "tools/bin/dav2d-macos", "dav2d", "dav2d-macos"}) {
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return {};
}

bool write_ivf(const std::string& path,
               const std::vector<std::vector<std::uint8_t>>& units, int width,
               int height, std::string* error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = static_cast<std::uint32_t>(units.size());
  if (!writer.open(path, header, *error)) return false;
  for (std::size_t i = 0; i < units.size(); ++i) {
    if (!writer.write_frame(units[i], i, *error)) return false;
  }
  return writer.close(*error);
}

bool decode(const std::string& decoder, const std::string& ivf,
            const std::string& y4m, std::string* error) {
  const std::string command = "\"" + decoder + "\" -q -i \"" + ivf + "\" -o \"" +
                              y4m + "\" --threads 1 --strict 1";
  const int status = std::system(command.c_str());
  if (status != 0) {
    *error = "dav2d failed: " + command;
    return false;
  }
  return true;
}

bool read_all_frames(const std::string& path, std::vector<Frame>* frames,
                     std::string* error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame frame;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, *error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    Frame out;
    out.width = static_cast<int>(frame.width());
    out.height = static_cast<int>(frame.height());
    out.y.assign(frame.y().begin(), frame.y().end());
    out.u.assign(frame.u().begin(), frame.u().end());
    out.v.assign(frame.v().begin(), frame.v().end());
    frames->push_back(std::move(out));
  }
  return true;
}

int count_diffs(const Frame& a, const Frame& b) {
  int diffs = 0;
  for (std::size_t i = 0; i < a.y.size(); ++i) diffs += a.y[i] != b.y[i];
  for (std::size_t i = 0; i < a.u.size(); ++i) diffs += a.u[i] != b.u[i];
  for (std::size_t i = 0; i < a.v.size(); ++i) diffs += a.v[i] != b.v[i];
  return diffs;
}

// ---------------------------------------------------------------------------
// Tile fixture: one PARTITION_NONE 64x64 leaf per superblock.  The keyframe
// codes a DC-only residual; P frames are GLOBALMV skip copies of the
// post-CCSO reference.
// ---------------------------------------------------------------------------

struct TileData {
  FramePlan plan;
  std::vector<BlockSyntax> blocks;
  std::vector<std::int32_t> levels;
};

int dc_level(int sb, int plane) {
  static const int kY[6] = {28, -30, 18, -14, 34, -22};
  static const int kC[6] = {16, -12, 10, -8, 14, -10};
  const int* table = plane == 0 ? kY : kC;
  return table[sb % 6];
}

TileData make_tile(int width, int height, bool inter) {
  TileData t;
  t.plan.width = width;
  t.plan.height = height;
  t.plan.sb_size = 64;
  t.plan.inter_frame = inter;
  const int sb_cols = (width + 63) / 64;
  const int sb_rows = (height + 63) / 64;
  int sb = 0;
  for (int sb_row = 0; sb_row < sb_rows; ++sb_row) {
    for (int sb_col = 0; sb_col < sb_cols; ++sb_col, ++sb) {
      BlockSyntax node;
      node.mi_row = static_cast<std::uint16_t>(sb_row * 16);
      node.mi_col = static_cast<std::uint16_t>(sb_col * 16);
      node.bsize = 12;     // BLOCK_64X64
      node.partition = 0;  // PARTITION_NONE
      node.allowed_mask = 1U << 0;
      node.flags = av2enc::video::syntax::kFlagSbRoot;
      node.subtree = 0;
      node.y_mode = 0;
      node.uv_mode = 0;
      node.tx_size[0] = 4;
      node.tx_size[1] = 3;
      node.tx_size[2] = 3;
      node.levels_offset = static_cast<std::uint32_t>(t.levels.size());
      if (inter) {
        node.is_inter = 1;
        node.skip_flag = 1;
        node.inter_mode = av2enc::video::syntax::kInterGlobalMv;
        node.eob[0] = node.eob[1] = node.eob[2] = 0;
      } else {
        node.is_inter = 0;
        node.eob[0] = node.eob[1] = node.eob[2] = 1;
        for (int plane = 0; plane < 3; ++plane) {
          t.levels.push_back(dc_level(sb, plane));
        }
      }
      t.blocks.push_back(node);
    }
  }
  return t;
}

// ---------------------------------------------------------------------------
// CCSO parameters
// ---------------------------------------------------------------------------

cr::PlaneParams make_plane_params(int plane) {
  // Band-offset only: one offset per band, 8 bands.
  static const std::uint8_t kOffsets[kPlanes][8] = {
      {7, 6, 5, 4, 3, 2, 1, 0},
      {6, 4, 2, 0, 1, 3, 5, 7},
      {0, 2, 4, 6, 7, 5, 3, 1},
  };
  cr::PlaneParams p;
  p.enabled = true;
  p.bo_only = true;
  p.scale_idx = 1;
  p.max_band_log2 = 3;
  p.offset_idx.assign(kOffsets[plane], kOffsets[plane] + 8);
  return p;
}

struct FrameSpec {
  bool keyframe = false;
  std::uint32_t order_hint = 0;
  std::uint32_t refresh_frame_flags = 1;
  std::uint8_t ref_slot = 0;
  cr::FrameParams params;
  cr::TileState tile;
  bool use_tile = true;
};

struct StreamOptions {
  VideoConfig config;
  post::VideoPostSequenceOptions sequence;
};

int count_units_for_plane(int mi_cols, int mi_rows, int plane,
                          int luma_size_log2) {
  const int width = (mi_cols * 4) >> (plane == 0 ? 0 : 1);
  const int height = (mi_rows * 4) >> (plane == 0 ? 0 : 1);
  int cols = 0;
  int rows = 0;
  cr::ccso_blk_grid_size(width, height, luma_size_log2, plane == 0 ? 0 : 1,
                         plane == 0 ? 0 : 1, &cols, &rows);
  return cols * rows;
}

bool write_frame_header_ccso(BitWriter& writer, const StreamOptions& options,
                             const post::VideoTiledFrameHeader& header,
                             const av2enc::tiles::TileLayout& layout,
                             cr::FrameParams params,
                             const cr::SlotStore& store, int tile_size_bytes,
                             std::size_t* ccso_bits, std::string* error) {
  writer.put_bit(true);  // is_first_tile_group
  // frame_header_present_flag is inferred 1 for the first tile group.

  // frame_header_info() (§5.18.2) for the M2 subset.
  auto put_uvlc = [&writer](std::uint32_t value) {
    unsigned leading_zeros = 0;
    while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) ++leading_zeros;
    writer.put_bits(0U, leading_zeros);
    writer.put_bit(true);
    const std::uint64_t payload =
        static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
    writer.put_bits(payload, leading_zeros);
  };
  put_uvlc(0);  // cur_mfh_id
  put_uvlc(0);  // seq_header_id_in_frame_header
  const bool key = header.keyframe;
  if (!key) writer.put_bit(true);  // frame_is_inter
  writer.put_bit(true);            // immediate_output_frame
  writer.put_bit(false);           // frame_size_override_flag
  writer.put_bits(header.order_hint, options.config.order_hint_bits);
  if (key) {
    writer.put_bit(false);  // allow_intrabc
  } else {
    writer.put_bit(false);  // signal_primary_ref_frame
    writer.put_bit(true);   // disable_cross_frame_cdf_init
    writer.put_bits(header.refresh_frame_flags, options.config.num_ref_frames);
    writer.put_bit(true);  // frame_explicit_ref_frame_map
    writer.put_bits(options.config.num_total_refs, 3U);
    writer.put_bits(header.ref_slot,
                    static_cast<unsigned>(cr::ceil_log2_int(
                        static_cast<int>(options.config.num_ref_frames))));
    writer.put_bit(false);  // allow_intrabc
    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // EIGHTTAP
  }
  writer.put_bit(options.config.disable_cdf_update);
  if (!post::write_video_tile_info(writer, layout, tile_size_bytes, error)) {
    return false;
  }
  const int base_q_idx = header.base_q_idx >= 0 ? header.base_q_idx
                                                : options.config.base_q_idx;
  writer.put_bits(static_cast<std::uint64_t>(base_q_idx), 8U);
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present
  bool deblock[4] = {false, false, false, false};
  bool delta[4] = {false, false, false, false};
  int dq[4] = {0, 0, 0, 0};
  post::write_video_deblock_params(writer, deblock, delta, dq, 0, error);
  if (error != nullptr && !error->empty()) return false;
  // cdef_params() is absent (sequence enable_cdef = 0); lr_params() is absent
  // (enable_restoration = 0).  ccso_params() is next in §5.18.7.2 order.

  cr::HeaderContext ctx;
  ctx.coded_lossless = false;
  ctx.enable_ccso = options.sequence.filters.enable_ccso;
  ctx.single_picture_header_flag = false;
  ctx.frame_is_intra = key;
  ctx.num_planes = 3;
  ctx.sb_size = layout.sb_size;
  ctx.ccso_unit_matches_sb_size =
      options.sequence.filters.ccso_unit_matches_sb_size;
  int tile_start_or = 0;
  const int sb_mi = layout.sb_size / 4;
  for (int sb : layout.col_starts_sb) tile_start_or |= sb * sb_mi;
  for (int sb : layout.row_starts_sb) tile_start_or |= sb * sb_mi;
  ctx.tile_start_or = tile_start_or;
  ctx.mi_cols = layout.width / 4;
  ctx.mi_rows = layout.height / 4;
  ctx.sub_x = 1;
  ctx.sub_y = 1;
  ctx.num_total_refs = static_cast<int>(options.config.num_total_refs);
  const int ref_slot_value = header.ref_slot;
  ctx.ref_frame_idx = &ref_slot_value;
  ctx.slots = &store;
  const std::size_t start = writer.bit_size();
  cr::write_ccso_params(writer, params, ctx, error);
  if (error != nullptr && !error->empty()) return false;
  if (ccso_bits != nullptr) *ccso_bits = writer.bit_size() - start;

  writer.put_bit(false);  // tx_mode_select = 0
  if (!key) {
    writer.put_bit(false);  // reference_select
    writer.put_bit(false);  // skip_mode_present
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set
  return true;
}

fs::CcsoParams to_fs_ccso(const cr::FrameParams& params, int luma_size) {
  fs::CcsoParams out;
  out.frame_flag = params.frame_flag;
  out.ccso_luma_size_log2 = luma_size;
  for (int plane = 0; plane < 3; ++plane) {
    const cr::PlaneParams& src = params.plane[plane];
    fs::CcsoPlaneParams& dst = out.plane[plane];
    dst.enabled = src.enabled;
    dst.bo_only = src.bo_only;
    dst.scale_idx = src.scale_idx;
    dst.quant_idx = src.quant_idx;
    dst.ext_filter = src.ext_filter;
    dst.edge_clf = src.edge_clf;
    dst.max_band_log2 = src.max_band_log2;
    dst.offset_idx = src.offset_idx;
  }
  return out;
}

bool has_ccso_reuse(const cr::FrameParams& params) {
  for (int plane = 0; plane < 3; ++plane) {
    if (params.reuse_ccso[plane] || params.sb_reuse_ccso[plane]) return true;
  }
  return false;
}

bool write_reference_header(BitWriter& writer, const StreamOptions& options,
                            const FrameSpec& spec,
                            const av2enc::tiles::TileLayout& layout,
                            int tile_size_bytes, std::size_t* bits,
                            std::string* error) {
  post::VideoTiledFrameHeader header;
  header.keyframe = spec.keyframe;
  header.order_hint = spec.order_hint;
  header.ref_slot = spec.ref_slot;
  header.refresh_frame_flags = spec.refresh_frame_flags;
  header.ccso_frame_flag = spec.params.frame_flag;
  header.ccso = to_fs_ccso(spec.params, 8);
  writer.put_bit(true);  // is_first_tile_group
  return post::write_video_frame_header_syntax(writer, options.config,
                                            options.sequence, header, layout,
                                            tile_size_bytes, bits, error);
}

bool encode_stream(const StreamOptions& options, const TileData& key_tile,
                   const TileData& inter_tile,
                   std::array<FrameSpec, 3>& frames, cr::SlotStore* store,
                   const std::string& path, std::size_t* reuse_ccso_bits,
                   std::string* error) {
  const auto tile_plan = post::plan_video_tiles(key_tile.plan.width,
                                             key_tile.plan.height, 1, 1);
  if (!tile_plan.ok) {
    *error = tile_plan.error;
    return false;
  }
  const auto& layout = tile_plan.layout;
  auto seq_obu = post::make_video_sequence_header_obu(options.config,
                                                   options.sequence, error);
  if (seq_obu.empty()) return false;

  std::vector<std::vector<std::uint8_t>> units;
  units.push_back(av2enc::video::make_temporal_delimiter_obu());
  units.push_back(seq_obu);

  for (std::size_t index = 0; index < frames.size(); ++index) {
    FrameSpec& spec = frames[index];
    const TileData& tile = spec.keyframe ? key_tile : inter_tile;
    std::vector<std::uint8_t> payload;
    std::size_t tile_bytes = 0;
    cr::TileState* state = spec.use_tile ? &spec.tile : nullptr;
    if (!cr::gate::assemble_video_frame_ccso(
            tile.plan, tile.blocks, tile.levels, options.config, state, payload,
            *error, &tile_bytes)) {
      return false;
    }
    if (payload.size() < 2) {
      *error = "tile payload shorter than 2 bytes";
      return false;
    }
    const int tile_size_bytes = av2enc::tiles::tile_size_bytes_for(
        static_cast<std::uint32_t>(payload.size()));
    if (tile_size_bytes == 0) {
      *error = "tile size not representable";
      return false;
    }
    post::VideoTiledFrameHeader header;
    header.keyframe = spec.keyframe;
    header.order_hint = spec.order_hint;
    header.ref_slot = spec.ref_slot;
    header.refresh_frame_flags = spec.refresh_frame_flags;
    BitWriter writer;
    std::size_t ccso_bits = 0;
    cr::SlotStore empty_store;
    const cr::SlotStore& header_store = store != nullptr ? *store
                                                         : empty_store;
    if (!write_frame_header_ccso(writer, options, header, layout,
                                 spec.params, header_store, tile_size_bytes,
                                 &ccso_bits, error)) {
      return false;
    }
    // Fresh (non-reuse) rows must be byte-identical to the verified
    // m2/post/tiled_frame.cpp header writer.   Reuse rows are the new syntax
    // this component exists to produce.
    if (!has_ccso_reuse(spec.params)) {
      BitWriter reference;
      if (!write_reference_header(reference, options, spec, layout,
                                  tile_size_bytes, nullptr, error)) {
        return false;
      }
      if (reference.bit_size() != writer.bit_size()) {
        *error = "fresh CCSO header differs from m2/post reference writer";
        return false;
      }
      const std::span<const std::uint8_t> a = writer.bytes();
      const std::span<const std::uint8_t> b = reference.bytes();
      for (std::size_t i = 0; i < writer.bit_size(); ++i) {
        const bool ba = ((a[i / 8U] >> (7U - (i & 7U))) & 1U) != 0U;
        const bool bb = ((b[i / 8U] >> (7U - (i & 7U))) & 1U) != 0U;
        if (ba != bb) {
          *error = "fresh CCSO header differs from m2/post reference writer";
          return false;
        }
      }
    }
    if (index == 2 && reuse_ccso_bits != nullptr) *reuse_ccso_bits = ccso_bits;
    writer.align_to_byte();
    std::vector<std::uint8_t> group = writer.take_bytes();
    group.insert(group.end(), payload.begin(), payload.end());
    const av2enc::basic::ObuType type =
        spec.keyframe ? av2enc::basic::ObuType::kClosedLoopKey
                      : av2enc::basic::ObuType::kRegularTileGroup;
    units.push_back(av2enc::basic::make_obu(av2enc::basic::ObuHeader{type}, group));
  }
  return write_ivf(path, units, key_tile.plan.width, key_tile.plan.height,
                   error);
}

// Applies §7.19 using the effective parameters and CcsoBlks in `state`.
bool filter_frame(const Frame& raw, const cr::TileState& state, Frame* out,
                  std::string* error) {
  *out = raw;
  std::vector<std::uint8_t> curr[3];
  std::vector<std::uint8_t> cdef[3];
  for (int plane = 0; plane < 3; ++plane) {
    curr[plane] = raw.plane(plane);
    cdef[plane] = raw.plane(plane);
  }
  rest::RestorationFrame rf;
  for (int plane = 0; plane < 3; ++plane) {
    rf.curr[plane].data = curr[plane].data();
    rf.curr[plane].stride = raw.plane_width(plane);
    rf.cdef[plane].data = cdef[plane].data();
    rf.cdef[plane].stride = raw.plane_width(plane);
  }
  rf.mi_cols = (raw.width + 3) / 4;
  rf.mi_rows = (raw.height + 3) / 4;
  rf.subsampling_x = 1;
  rf.subsampling_y = 1;
  rf.bit_depth = 8;
  rf.sb_width = 64;
  rf.sb_height = 64;
  rf.sb_shift = 4;
  rest::CcsoFrameParams params;
  if (!cr::make_restoration_params(state, &params, error)) return false;
  const rest::RestorationStatus status = rest::ccso_frame(rf, params);
  if (status != rest::RestorationStatus::kOk) {
    *error = std::string("ccso_frame failed: ") +
             rest::restoration_status_string(status);
    return false;
  }
  for (int plane = 0; plane < 3; ++plane) out->plane(plane) = cdef[plane];
  return true;
}

bool setup_fresh_state(FrameSpec* spec, int mi_cols, int mi_rows,
                       std::string* error) {
  cr::EffectiveFrame effective;
  effective.frame_flag = true;
  effective.ccso_luma_size_log2 =
      cr::derive_ccso_luma_size_log2(64, 0, /*matches_sb=*/false);
  for (int plane = 0; plane < 3; ++plane) {
    effective.plane[plane].params = spec->params.plane[plane];
    effective.plane[plane].sb_reuse_ccso = false;
    const int width = (mi_cols * 4) >> (plane == 0 ? 0 : 1);
    const int height = (mi_rows * 4) >> (plane == 0 ? 0 : 1);
    cr::ccso_blk_grid_size(width, height, effective.ccso_luma_size_log2,
                           plane == 0 ? 0 : 1, plane == 0 ? 0 : 1,
                           &effective.plane[plane].blk_cols,
                           &effective.plane[plane].blk_rows);
    effective.plane[plane].blks.assign(
        static_cast<std::size_t>(effective.plane[plane].blk_cols) *
            static_cast<std::size_t>(effective.plane[plane].blk_rows),
        1);
  }
  if (!spec->tile.init(effective, mi_cols, mi_rows, 1, 1, error)) return false;
  for (int plane = 0; plane < 3; ++plane) {
    if (!spec->tile.set_block_flags(plane, spec->tile.blks(plane).data(),
                                    spec->tile.blk_cols(plane),
                                    spec->tile.blk_rows(plane), error)) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  int width = 256;
  int height = 192;
  std::string out_path = "/tmp/ccso_reuse_gate.ivf";
  std::string decoder;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--width" && i + 1 < argc) {
      width = std::atoi(argv[++i]);
    } else if (a == "--height" && i + 1 < argc) {
      height = std::atoi(argv[++i]);
    } else if (a == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (a == "--decoder" && i + 1 < argc) {
      decoder = argv[++i];
    } else {
      std::fprintf(stderr,
                   "usage: %s [--width W] [--height H] [--out F.ivf] "
                   "[--decoder D]\n",
                   argv[0]);
      return 2;
    }
  }
  if ((width & 7) != 0 || (height & 7) != 0) {
    std::fprintf(stderr, "width/height must be divisible by 8\n");
    return 2;
  }
  if (decoder.empty()) decoder = find_decoder();
  if (decoder.empty()) {
    std::fprintf(stderr, "SKIP: no dav2d decoder found\n");
    return 77;
  }

  const int mi_cols = width / 4;
  const int mi_rows = height / 4;
  std::string error;
  StreamOptions options;
  options.config.max_frame_width = static_cast<std::uint16_t>(width);
  options.config.max_frame_height = static_cast<std::uint16_t>(height);
  options.config.base_q_idx = 90;
  options.config.num_ref_frames = 2;
  options.config.num_total_refs = 1;
  options.config.order_hint_bits = 4;
  options.config.disable_cdf_update = true;
  options.sequence.filters.enable_cdef = false;
  options.sequence.filters.enable_ccso = true;
  options.sequence.filters.ccso_unit_matches_sb_size = false;  // 256x256

  const TileData key_tile = make_tile(width, height, false);
  const TileData inter_tile = make_tile(width, height, true);

  // -----------------------------------------------------------------------
  // Baseline (CCSO off) reconstruction used as the keyframe source.
  // -----------------------------------------------------------------------
  StreamOptions base_options = options;
  base_options.sequence.filters.enable_ccso = false;
  std::array<FrameSpec, 3> base_frames;
  for (std::size_t i = 0; i < base_frames.size(); ++i) {
    base_frames[i].keyframe = i == 0;
    base_frames[i].order_hint = static_cast<std::uint32_t>(i);
    base_frames[i].refresh_frame_flags = i == 0 ? 0U : (i == 1 ? 1U : 2U);
    base_frames[i].ref_slot = 0;
    base_frames[i].use_tile = false;
  }
  const std::string base_ivf = "/tmp/ccso_reuse_base.ivf";
  const std::string base_y4m = "/tmp/ccso_reuse_base.y4m";
  if (!encode_stream(base_options, key_tile, inter_tile, base_frames, nullptr,
                     base_ivf, nullptr, &error)) {
    std::fprintf(stderr, "baseline encode failed: %s\n", error.c_str());
    return 1;
  }
  if (!decode(decoder, base_ivf, base_y4m, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::vector<Frame> base;
  if (!read_all_frames(base_y4m, &base, &error) || base.size() != 3) {
    std::fprintf(stderr, "baseline decode read failed: %s\n", error.c_str());
    return 1;
  }

  // -----------------------------------------------------------------------
  // Frame 0 and frame 1: fresh CCSO parameters and fresh block flags.
  // -----------------------------------------------------------------------
  std::array<FrameSpec, 3> frames;
  for (std::size_t i = 0; i < frames.size(); ++i) {
    frames[i].keyframe = i == 0;
    frames[i].order_hint = static_cast<std::uint32_t>(i);
    frames[i].refresh_frame_flags = i == 0 ? 0U : (i == 1 ? 1U : 2U);
    frames[i].ref_slot = 0;
    frames[i].params.frame_flag = true;
    for (int plane = 0; plane < 3; ++plane) {
      frames[i].params.plane[plane] = make_plane_params(plane);
    }
  }
  for (std::size_t i = 0; i < 2; ++i) {
    if (!setup_fresh_state(&frames[i], mi_cols, mi_rows, &error)) {
      std::fprintf(stderr, "state init failed: %s\n", error.c_str());
      return 1;
    }
  }

  // Encoder-side reconstruction: frame 0 filtered, then frame 1 copies the
  // filtered reference and filters again.
  Frame out0;
  Frame out1;
  if (!filter_frame(base[0], frames[0].tile, &out0, &error) ||
      !filter_frame(out0, frames[1].tile, &out1, &error)) {
    std::fprintf(stderr, "encoder CCSO failed: %s\n", error.c_str());
    return 1;
  }

  // §7.23 reference frame update: save frame 1's CCSO state in slot 0 before
  // frame 2 is signalled.
  cr::SlotStore store;
  {
    cr::FrameParams saved_params = frames[1].params;
    saved_params.ccso_luma_size_log2 = 8;
    std::vector<std::uint8_t> blks[3];
    int cols[3] = {0, 0, 0};
    int rows[3] = {0, 0, 0};
    for (int plane = 0; plane < 3; ++plane) {
      blks[plane] = frames[1].tile.blks(plane);
      cols[plane] = frames[1].tile.blk_cols(plane);
      rows[plane] = frames[1].tile.blk_rows(plane);
    }
    cr::SavedFrame frame;
    frame.ccso_luma_size_log2 = 8;
    frame.order_hint = 1;
    frame.mi_cols = mi_cols;
    frame.mi_rows = mi_rows;
    frame.params = &saved_params;
    frame.blks = blks;
    frame.blk_cols = cols;
    frame.blk_rows = rows;
    if (!cr::save_ccso_params(store, 0, frame, &error)) {
      std::fprintf(stderr, "save_ccso_params failed: %s\n", error.c_str());
      return 1;
    }
  }

  // Frame 2: select reuse from reference index 0 (the only total ref).
  cr::HeaderContext reuse_context;
  reuse_context.enable_ccso = true;
  reuse_context.single_picture_header_flag = false;
  reuse_context.frame_is_intra = false;
  reuse_context.num_planes = 3;
  reuse_context.sb_size = 64;
  reuse_context.ccso_unit_matches_sb_size = false;
  reuse_context.tile_start_or = 0;
  reuse_context.mi_cols = mi_cols;
  reuse_context.mi_rows = mi_rows;
  reuse_context.sub_x = 1;
  reuse_context.sub_y = 1;
  reuse_context.num_total_refs = 1;
  const int ref_slot_value = 0;
  reuse_context.ref_frame_idx = &ref_slot_value;
  reuse_context.slots = &store;

  cr::ReuseChoice choice;
  if (!cr::use_ccso(frames[1].params.plane[0],
                    frames[1].tile.blks(0), frames[1].tile.blk_cols(0),
                    frames[1].tile.blk_rows(0), 0, reuse_context, store,
                    &choice, &error)) {
    std::fprintf(stderr, "use_ccso failed: %s\n", error.c_str());
    return 1;
  }
  if (!choice.reuse_ccso || !choice.sb_reuse_ccso) {
    std::fprintf(stderr, "FAIL: frame 2 did not select full CCSO reuse\n");
    return 1;
  }
  for (int plane = 0; plane < 3; ++plane) {
    frames[2].params.plane[plane] = frames[1].params.plane[plane];
    frames[2].params.reuse_ccso[plane] = true;
    frames[2].params.sb_reuse_ccso[plane] = true;
    frames[2].params.ccso_ref_idx[plane] = choice.ccso_ref_idx;
  }
  cr::EffectiveFrame effective2;
  if (!cr::resolve_ccso_params(reuse_context, frames[2].params, store,
                               &effective2, &error)) {
    std::fprintf(stderr, "resolve frame 2 failed: %s\n", error.c_str());
    return 1;
  }
  if (!frames[2].tile.init(effective2, mi_cols, mi_rows, 1, 1, &error)) {
    std::fprintf(stderr, "frame 2 tile init failed: %s\n", error.c_str());
    return 1;
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (frames[2].tile.codes_block_flags(plane)) {
      std::fprintf(stderr, "FAIL: sb_reuse still codes block flags\n");
      return 1;
    }
  }
  Frame out2;
  if (!filter_frame(out1, frames[2].tile, &out2, &error)) {
    std::fprintf(stderr, "frame 2 CCSO failed: %s\n", error.c_str());
    return 1;
  }
  const std::array<Frame, 3> expected = {out0, out1, out2};

  // -----------------------------------------------------------------------
  // Reuse stream.
  // -----------------------------------------------------------------------
  const std::string reuse_ivf = out_path;
  std::size_t reuse_header_bits = 0;
  if (!encode_stream(options, key_tile, inter_tile, frames, &store, reuse_ivf,
                     &reuse_header_bits, &error)) {
    std::fprintf(stderr, "reuse encode failed: %s\n", error.c_str());
    return 1;
  }
  const std::string reuse_y4m = "/tmp/ccso_reuse_decoded.y4m";
  if (!decode(decoder, reuse_ivf, reuse_y4m, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::vector<Frame> decoded;
  if (!read_all_frames(reuse_y4m, &decoded, &error) || decoded.size() != 3) {
    std::fprintf(stderr, "decoded read failed: %s\n", error.c_str());
    return 1;
  }

  // -----------------------------------------------------------------------
  // No-reuse control: frame 2 re-signals the same parameters and block flags.
  // -----------------------------------------------------------------------
  std::array<FrameSpec, 3> control = frames;
  control[2].params.reuse_ccso[0] = false;
  control[2].params.reuse_ccso[1] = false;
  control[2].params.reuse_ccso[2] = false;
  control[2].params.sb_reuse_ccso[0] = false;
  control[2].params.sb_reuse_ccso[1] = false;
  control[2].params.sb_reuse_ccso[2] = false;
  control[2].params.ccso_ref_idx[0] = 0;
  control[2].params.ccso_ref_idx[1] = 0;
  control[2].params.ccso_ref_idx[2] = 0;
  if (!setup_fresh_state(&control[2], mi_cols, mi_rows, &error)) {
    std::fprintf(stderr, "control state init failed: %s\n", error.c_str());
    return 1;
  }
  const std::string control_ivf = "/tmp/ccso_reuse_control.ivf";
  std::size_t control_header_bits = 0;
  if (!encode_stream(options, key_tile, inter_tile, control, nullptr,
                     control_ivf, &control_header_bits, &error)) {
    std::fprintf(stderr, "control encode failed: %s\n", error.c_str());
    return 1;
  }
  const std::string control_y4m = "/tmp/ccso_reuse_control.y4m";
  if (!decode(decoder, control_ivf, control_y4m, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::vector<Frame> control_decoded;
  if (!read_all_frames(control_y4m, &control_decoded, &error) ||
      control_decoded.size() != 3) {
    std::fprintf(stderr, "control decode read failed: %s\n", error.c_str());
    return 1;
  }

  int diffs[3] = {0, 0, 0};
  int control_diffs = 0;
  for (int i = 0; i < 3; ++i) {
    diffs[i] = count_diffs(decoded[static_cast<std::size_t>(i)],
                           expected[static_cast<std::size_t>(i)]);
  }
  control_diffs = count_diffs(control_decoded[2], out2);

  std::error_code size_error;
  const std::uintmax_t reuse_bytes = std::filesystem::file_size(
      reuse_ivf, size_error);
  size_error.clear();
  const std::uintmax_t control_bytes = std::filesystem::file_size(
      control_ivf, size_error);
  const std::uintmax_t base_bytes = std::filesystem::file_size(base_ivf,
                                                              size_error);

  std::printf("fixture: %dx%d, 3 frames, CcsoLumaSizeLog2=%d, units/plane=%d\n",
              width, height, 8, count_units_for_plane(mi_cols, mi_rows, 0, 8));
  for (int i = 0; i < 3; ++i) {
    std::printf("decoded frame%d vs encoder §7.19: %d diffs\n", i, diffs[i]);
  }
  std::printf("control frame2 (no reuse) vs encoder §7.19: %d diffs\n",
              control_diffs);
  std::printf(
      "frame2 ccso_params row: reuse %zu bits, fresh %zu bits (%+jd)\n",
      reuse_header_bits, control_header_bits,
      static_cast<std::intmax_t>(reuse_header_bits) -
          static_cast<std::intmax_t>(control_header_bits));
  std::printf(
      "bit cost: baseline(no CCSO) %ju B, no-reuse %ju B, reuse %ju B "
      "(%+jd B vs no-reuse, %+.2f%%)\n",
      base_bytes, control_bytes, reuse_bytes,
      static_cast<std::intmax_t>(reuse_bytes) -
          static_cast<std::intmax_t>(control_bytes),
      100.0 * (static_cast<double>(reuse_bytes) -
               static_cast<double>(control_bytes)) /
          static_cast<double>(control_bytes));

  if (diffs[0] != 0 || diffs[1] != 0 || diffs[2] != 0 ||
      control_diffs != 0) {
    std::fprintf(stderr, "FAIL: temporal CCSO reuse gate\n");
    return 1;
  }
  if (reuse_bytes >= control_bytes) {
    std::fprintf(stderr,
                 "FAIL: reuse stream is not smaller than the no-reuse control\n");
    return 1;
  }
  std::printf(
      "PASS: frame 2 reused frame 1's CCSO parameters and CcsoBlks; decoded "
      "reconstruction bit-exact (reuse saves %jd header bits and %jd bytes)\n",
      static_cast<std::intmax_t>(control_header_bits) -
          static_cast<std::intmax_t>(reuse_header_bits),
      static_cast<std::intmax_t>(control_bytes) -
          static_cast<std::intmax_t>(reuse_bytes));
  return 0;
}
