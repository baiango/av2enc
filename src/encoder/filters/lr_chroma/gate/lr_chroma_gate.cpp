// Chroma loop-restoration decoder gate (2-frame M2 key + P).
//
// The M2 core has no restoration hook and must not be edited.  This gate
// vendors the M2 tile writer with exactly one §5.20.10.4 read_lr() call per
// superblock (see vendor_m2_syntax.py), enables chroma
// RESTORE_WIENER_NONSEP in the sequence/frame headers, and drives the
// encoder-side chroma decision on real reconstruction.
//
// Closed loop:
//   1. encode the fixture with every restoration type NONE and decode it with
//      the pinned dav2d -> base reconstruction,
//   2. choose per-unit chroma filters on real reconstruction vs a target
//      picture (chroma_decide),
//   3. rebuild the same frames with chroma restoration on (RESTORE_WIENER_NONSEP
//      units, uv_sym where selected, non-zero coefficients),
//   4. decode again with `dav2d --strict 1 --threads 1` and compare every Y/U/V
//      sample with the encoder's own §7.20 output (chroma_apply).  The pass
//      condition is zero differing samples.
//
// Usage:
//   lr_chroma_gate [--width 256] [--height 192] [--out /tmp/chroma_gate.ivf]
//                  [--decoder tools/bin/dav2d-macos] [--lambda L]
//                  [--shift 0..3] [--keep-y4m]
//
// Exit: 0 pass, 1 fail, 77 decoder unavailable.

#include "encoder/filters/lr_chroma/chroma_lr.h"
#include "encoder/m1/filters_stream/filters_stream.h"
#include "encoder/m2/m2_headers.h"
#include "encoder/m2/m2_syntax.h"
#include "encoder/m2/post/loop_filters.h"
#include "encoder/m2/post/tiled_frame.h"
#include "encoder/tiles/tile_layout.h"

#include "bitstream/bitwriter.h"
#include "bitstream/m0_headers.h"
#include "io/ivf.h"
#include "io/y4m.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "m2_syntax_lr.h"

namespace {

namespace fs = av2enc::m1::filters_stream;
namespace rest = av2enc::restoration;
namespace lr = av2enc::lr_chroma;
using av2enc::BitWriter;
using av2enc::m2::M2Config;
using av2enc::m2::syntax::BlockSyntax;
using av2enc::m2::syntax::FramePlan;

std::uint32_t ceil_log2_u32(std::uint32_t value) {
  if (value <= 1U) return 0U;
  std::uint32_t bits = 0;
  --value;
  while (value != 0) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) ++leading_zeros;
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

struct Frame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y, u, v;
  int cw() const { return (width + 1) / 2; }
  int ch() const { return (height + 1) / 2; }
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

// ---------------------------------------------------------------------------
// Tile fixture: one PARTITION_NONE 64x64 leaf per superblock.  The keyframe
// codes a DC-only residual so the reconstruction is non-flat; the P frame is a
// GLOBALMV skip copy of the (filtered) reference.
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
      node.flags = av2enc::m2::syntax::kFlagSbRoot;
      node.subtree = 0;
      node.y_mode = 0;
      node.uv_mode = 0;
      node.tx_size[0] = 4;  // TX_64X64
      node.tx_size[1] = 3;  // TX_32X32
      node.tx_size[2] = 3;
      node.levels_offset = static_cast<std::uint32_t>(t.levels.size());
      if (inter) {
        node.is_inter = 1;
        node.skip_flag = 1;
        node.inter_mode = av2enc::m2::syntax::kInterGlobalMv;
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
// Frame header writer: m2::post::write_m2_frame_header_syntax() for one tile
// with cdef/ccso off and lr_params() in the §5.18.7.11 position.
// ---------------------------------------------------------------------------

bool write_frame_header_lr(BitWriter& writer, const M2Config& config,
                           const av2enc::m2::post::M2PostSequenceOptions& seq,
                           const av2enc::m2::post::M2TiledFrameHeader& header,
                           const av2enc::tiles::TileLayout& layout,
                           const fs::LrParams& lr_params,
                           const fs::LrHeaderContext& lr_context,
                           int tile_size_bytes, std::string* error) {
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  const bool key = header.keyframe;
  if (!key) writer.put_bit(true);  // frame_is_inter
  writer.put_bit(true);            // immediate_output_frame
  writer.put_bit(false);           // frame_size_override_flag
  writer.put_bits(header.order_hint, config.order_hint_bits);
  if (key) {
    writer.put_bit(false);  // allow_intrabc
  } else {
    writer.put_bit(false);  // signal_primary_ref_frame
    writer.put_bit(true);   // disable_cross_frame_cdf_init
    writer.put_bits(header.refresh_frame_flags, config.num_ref_frames);
    writer.put_bit(true);   // frame_explicit_ref_frame_map
    writer.put_bits(config.num_total_refs, 3U);
    writer.put_bits(header.ref_slot,
                    static_cast<unsigned>(ceil_log2_u32(config.num_ref_frames)));
    writer.put_bit(false);  // allow_intrabc
    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // EIGHTTAP
  }
  writer.put_bit(config.disable_cdf_update);
  if (!av2enc::m2::post::write_m2_tile_info(writer, layout, tile_size_bytes,
                                            error)) {
    return false;
  }
  const int base_q_idx =
      header.base_q_idx >= 0 ? header.base_q_idx : config.base_q_idx;
  writer.put_bits(static_cast<std::uint64_t>(base_q_idx), 8U);
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present

  bool deblock[4] = {false, false, false, false};
  bool delta[4] = {false, false, false, false};
  av2enc::m2::post::write_m2_deblock_params(writer, deblock, delta, error);
  (void)seq;

  if (!fs::write_lr_params(writer, const_cast<fs::LrParams&>(lr_params),
                           lr_context, error)) {
    return false;
  }

  writer.put_bit(false);    // tx_mode_select = 0
  if (!key) {
    writer.put_bit(false);  // reference_select
    writer.put_bit(false);  // skip_mode_present
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set
  return true;
}

struct StreamOptions {
  M2Config config;
  av2enc::m2::post::M2PostSequenceOptions sequence;
};

bool encode_stream(const StreamOptions& options, const TileData& key,
                   const TileData& inter, const fs::LrParams& key_lr,
                   const fs::LrHeaderContext& key_context,
                   const fs::LrParams& inter_lr,
                   const fs::LrHeaderContext& inter_context,
                   fs::LrState* key_state,
                   const lr::ChromaDecision* key_decision,
                   fs::LrState* inter_state,
                   const lr::ChromaDecision* inter_decision,
                   const std::string& path, std::string* error) {
  const auto tile_plan =
      av2enc::m2::post::plan_m2_tiles(key.plan.width, key.plan.height, 1, 1);
  if (!tile_plan.ok) {
    *error = tile_plan.error;
    return false;
  }
  const auto& layout = tile_plan.layout;

  auto seq_obu = av2enc::m2::post::make_m2_sequence_header_obu(
      options.config, options.sequence, error);
  if (seq_obu.empty()) return false;

  std::vector<std::vector<std::uint8_t>> units;
  units.push_back(av2enc::m2::make_temporal_delimiter_obu());
  units.push_back(seq_obu);

  auto build = [&](bool keyframe, const TileData& tile, const fs::LrParams& lr,
                   const fs::LrHeaderContext& context, fs::LrState* state,
                   const lr::ChromaDecision* decision,
                   std::vector<std::uint8_t>* obu, std::string* err) -> bool {
    std::vector<std::uint8_t> payload;
    std::size_t tile_bytes = 0;
    if (!av2enc::lr_chroma::gate::assemble_m2_frame_lr(
            tile.plan, tile.blocks, tile.levels, options.config, state,
            decision, payload, *err, &tile_bytes)) {
      return false;
    }
    if (payload.size() < 2) {
      *err = "tile payload shorter than 2 bytes";
      return false;
    }
    const int tile_size_bytes = av2enc::tiles::tile_size_bytes_for(
        static_cast<std::uint32_t>(payload.size()));
    if (tile_size_bytes == 0) {
      *err = "tile size not representable";
      return false;
    }
    av2enc::m2::post::M2TiledFrameHeader header;
    header.keyframe = keyframe;
    header.order_hint = keyframe ? 0 : 1;
    header.ref_slot = 0;
    header.refresh_frame_flags = 1;
    BitWriter writer;
    writer.put_bit(true);  // is_first_tile_group
    if (!write_frame_header_lr(writer, options.config, options.sequence, header,
                               layout, lr, context, tile_size_bytes, err)) {
      return false;
    }
    writer.align_to_byte();
    std::vector<std::uint8_t> group = writer.take_bytes();
    group.insert(group.end(), payload.begin(), payload.end());
    const av2enc::m0::ObuType type =
        keyframe ? av2enc::m0::ObuType::kClosedLoopKey
                 : av2enc::m0::ObuType::kRegularTileGroup;
    *obu = av2enc::m0::make_obu(av2enc::m0::ObuHeader{type}, group);
    return true;
  };

  std::vector<std::uint8_t> key_obu;
  if (!build(true, key, key_lr, key_context, key_state, key_decision, &key_obu,
             error)) {
    return false;
  }
  units.push_back(std::move(key_obu));
  std::vector<std::uint8_t> inter_obu;
  if (!build(false, inter, inter_lr, inter_context, inter_state, inter_decision,
             &inter_obu, error)) {
    return false;
  }
  units.push_back(std::move(inter_obu));
  return write_ivf(path, units, key.plan.width, key.plan.height, error);
}

// ---------------------------------------------------------------------------
// Decoding + comparison
// ---------------------------------------------------------------------------

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

bool plane_differs(const std::vector<std::uint8_t>& a,
                   const std::vector<std::uint8_t>& b) {
  return a != b;
}

void add_noise(Frame* source, int seed, int amplitude) {
  std::uint32_t state = static_cast<std::uint32_t>(seed) * 2654435761U + 1U;
  auto next = [&state]() {
    state = state * 1664525U + 1013904223U;
    return static_cast<int>((state >> 16) & 0x7FFF);
  };
  auto apply = [&](std::vector<std::uint8_t>* plane, int amp) {
    for (std::uint8_t& s : *plane) {
      const int delta = (next() % (2 * amp + 1)) - amp;
      s = static_cast<std::uint8_t>(
          std::max(0, std::min(255, static_cast<int>(s) + delta)));
    }
  };
  apply(&source->y, amplitude);
  apply(&source->u, amplitude / 2 + 1);
  apply(&source->v, amplitude / 2 + 1);
}

lr::ChromaLrInput make_input(const Frame& base, const Frame& target) {
  lr::ChromaLrInput in;
  in.plane_width[0] = in.plane_width[1] = base.cw();
  in.plane_height[0] = in.plane_height[1] = base.ch();
  in.curr[0] = base.u.data();
  in.curr_stride[0] = base.cw();
  in.curr[1] = base.v.data();
  in.curr_stride[1] = base.cw();
  in.cdef[0] = base.u.data();
  in.cdef_stride[0] = base.cw();
  in.cdef[1] = base.v.data();
  in.cdef_stride[1] = base.cw();
  in.luma_curr = base.y.data();
  in.luma_curr_stride = base.width;
  in.luma_cdef = base.y.data();
  in.luma_cdef_stride = base.width;
  in.luma_width = base.width;
  in.luma_height = base.height;
  in.source[0] = target.u.data();
  in.source_stride[0] = target.cw();
  in.source[1] = target.v.data();
  in.source_stride[1] = target.cw();
  in.mi_cols = (base.width + 3) / 4;
  in.mi_rows = (base.height + 3) / 4;
  in.sub_x = 1;
  in.sub_y = 1;
  in.sb_size = 64;
  return in;
}

}  // namespace

int main(int argc, char** argv) {
  int width = 256;
  int height = 192;
  std::string out_path = "/tmp/lr_chroma_gate.ivf";
  std::string decoder;
  bool keep_y4m = false;
  double lambda = 0.0;
  int forced_shift = -1;
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
    } else if (a == "--keep-y4m") {
      keep_y4m = true;
    } else if (a == "--lambda" && i + 1 < argc) {
      lambda = std::atof(argv[++i]);
    } else if (a == "--shift" && i + 1 < argc) {
      forced_shift = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr,
                   "usage: %s [--width W] [--height H] [--out F.ivf] "
                   "[--decoder D] [--lambda L] [--shift 0..3] [--keep-y4m]\n",
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

  std::string error;
  StreamOptions options;
  options.config.max_frame_width = static_cast<std::uint16_t>(width);
  options.config.max_frame_height = static_cast<std::uint16_t>(height);
  options.config.base_q_idx = 90;
  options.config.num_total_refs = 1;
  options.config.disable_cdf_update = true;
  options.sequence.filters.enable_restoration = true;
  options.sequence.filters.lr_uv_present = true;
  options.sequence.filters.lr_disable_luma_pc_wiener = true;
  options.sequence.filters.lr_disable_luma_wiener_ns = false;
  options.sequence.filters.lr_disable_chroma_wiener_ns = false;
  options.sequence.filters.enable_cdef = false;
  options.sequence.filters.enable_ccso = false;

  const TileData key = make_tile(width, height, false);
  const TileData inter = make_tile(width, height, true);

  fs::LrHeaderContext key_context;
  key_context.coded_lossless = false;
  key_context.enable_restoration = true;
  key_context.single_picture_header_flag = false;
  key_context.frame_is_intra = true;
  key_context.sb_size = 64;
  key_context.num_planes = 3;
  key_context.sub_x = 1;
  key_context.sub_y = 1;
  key_context.base_q_idx = 90;
  fs::LrHeaderContext inter_context = key_context;
  inter_context.frame_is_intra = false;

  fs::LrParams none;
  none.frame_type[0] = none.frame_type[1] = none.frame_type[2] =
      rest::kRestoreNone;
  none.luma_shift = 3;
  none.chroma_shift = 3;

  const std::string base_ivf = "/tmp/lr_chroma_gate_base.ivf";
  const std::string base_y4m = "/tmp/lr_chroma_gate_base.y4m";
  if (!encode_stream(options, key, inter, none, key_context, none, inter_context,
                     nullptr, nullptr, nullptr, nullptr, base_ivf, &error)) {
    std::fprintf(stderr, "baseline encode failed: %s\n", error.c_str());
    return 1;
  }
  if (!decode(decoder, base_ivf, base_y4m, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::vector<Frame> base_frames;
  if (!read_all_frames(base_y4m, &base_frames, &error) ||
      base_frames.size() != 2) {
    std::fprintf(stderr, "baseline decode read failed: %s\n", error.c_str());
    return 1;
  }
  std::printf("baseline: %dx%d, %zu frames decoded\n", width, height,
              base_frames.size());

  Frame target0 = base_frames[0];
  add_noise(&target0, 1, 8);
  const lr::ChromaLrInput in0 = make_input(base_frames[0], target0);
  lr::ChromaLrOptions lr_options;
  lr_options.lambda_bits = lambda;
  lr_options.forced_shift = forced_shift;
  lr::ChromaDecision decision0;
  if (!lr::chroma_decide(in0, lr_options, &decision0, &error)) {
    std::fprintf(stderr, "frame0 chroma decide failed: %s\n", error.c_str());
    return 1;
  }
  Frame out0 = base_frames[0];
  {
    std::uint8_t* dst[2] = {out0.u.data(), out0.v.data()};
    int stride[2] = {out0.cw(), out0.cw()};
    if (lr::chroma_apply(in0, decision0, dst, stride) !=
        rest::RestorationStatus::kOk) {
      std::fprintf(stderr, "frame0 chroma_apply failed\n");
      return 1;
    }
  }

  Frame target1 = out0;
  add_noise(&target1, 7, 10);
  const lr::ChromaLrInput in1 = make_input(out0, target1);
  lr::ChromaDecision decision1;
  if (!lr::chroma_decide(in1, lr_options, &decision1, &error)) {
    std::fprintf(stderr, "frame1 chroma decide failed: %s\n", error.c_str());
    return 1;
  }
  Frame out1 = out0;
  {
    std::uint8_t* dst[2] = {out1.u.data(), out1.v.data()};
    int stride[2] = {out1.cw(), out1.cw()};
    if (lr::chroma_apply(in1, decision1, dst, stride) !=
        rest::RestorationStatus::kOk) {
      std::fprintf(stderr, "frame1 chroma_apply failed\n");
      return 1;
    }
  }

  int active0 = 0;
  int sym0 = 0;
  int non_zero0 = 0;
  for (const lr::ChromaUnitPlan& p : decision0.units) {
    if (p.on) ++active0;
    if (p.on && p.filter.uv_sym) ++sym0;
    if (p.on) {
      for (int j = 0; j < rest::kWienerNsChromaCoeffs; ++j) {
        if (p.filter.coeffs[j] != 0) {
          ++non_zero0;
          break;
        }
      }
    }
  }
  int active1 = 0;
  int sym1 = 0;
  for (const lr::ChromaUnitPlan& p : decision1.units) {
    if (p.on) ++active1;
    if (p.on && p.filter.uv_sym) ++sym1;
  }
  std::printf(
      "frame0: shift=%d unit=%d units=%zu active=%d uv_sym=%d "
      "sse %.0f -> %.0f (%.2f dB -> %.2f dB)\n",
      decision0.shift, decision0.unit_size, decision0.units.size(), active0,
      sym0, decision0.identity_sse, decision0.sse,
      lr::chroma_psnr(target0.u.data(), target0.cw(), base_frames[0].u.data(),
                      base_frames[0].cw(), target0.cw(), target0.ch()),
      lr::chroma_psnr(target0.u.data(), target0.cw(), out0.u.data(),
                      out0.cw(), target0.cw(), target0.ch()));
  std::printf(
      "frame1: shift=%d unit=%d units=%zu active=%d uv_sym=%d "
      "sse %.0f -> %.0f (%.2f dB -> %.2f dB)\n",
      decision1.shift, decision1.unit_size, decision1.units.size(), active1,
      sym1, decision1.identity_sse, decision1.sse,
      lr::chroma_psnr(target1.u.data(), target1.cw(), out0.u.data(),
                      out0.cw(), target1.cw(), target1.ch()),
      lr::chroma_psnr(target1.u.data(), target1.cw(), out1.u.data(),
                      out1.cw(), target1.cw(), target1.ch()));
  if (active0 + active1 == 0 || non_zero0 == 0) {
    std::fprintf(stderr,
                 "no non-trivial chroma filter selected; gate is trivial\n");
    return 1;
  }

  fs::LrParams key_lr;
  fs::LrParams inter_lr;
  if (!lr::chroma_make_lr_params(decision0, &key_lr, &error) ||
      !lr::chroma_make_lr_params(decision1, &inter_lr, &error)) {
    std::fprintf(stderr, "lr params failed: %s\n", error.c_str());
    return 1;
  }
  fs::LrState key_state;
  fs::LrState inter_state;
  if (!lr::chroma_init_tile_state(decision0, in0, &key_state, &error) ||
      !lr::chroma_init_tile_state(decision1, in1, &inter_state, &error)) {
    std::fprintf(stderr, "state init failed: %s\n", error.c_str());
    return 1;
  }
  if (!encode_stream(options, key, inter, key_lr, key_context, inter_lr,
                     inter_context, &key_state, &decision0, &inter_state,
                     &decision1, out_path, &error)) {
    std::fprintf(stderr, "LR encode failed: %s\n", error.c_str());
    return 1;
  }
  const std::string out_y4m = "/tmp/lr_chroma_gate_decoded.y4m";
  if (!decode(decoder, out_path, out_y4m, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::vector<Frame> decoded;
  if (!read_all_frames(out_y4m, &decoded, &error) || decoded.size() != 2) {
    std::fprintf(stderr, "decoded read failed: %s\n", error.c_str());
    return 1;
  }

  const int diffs0 = count_diffs(decoded[0], out0);
  const int diffs1 = count_diffs(decoded[1], out1);
  const bool changed0 = plane_differs(decoded[0].u, base_frames[0].u) &&
                        plane_differs(decoded[0].v, base_frames[0].v);
  std::printf("decoded frame0 vs encoder §7.20: %d diffs\n", diffs0);
  std::printf("decoded frame1 vs encoder §7.20: %d diffs\n", diffs1);
  std::printf("filter active: frame0 U and V both changed=%d\n",
              changed0 ? 1 : 0);
  std::error_code size_error;
  const std::uintmax_t base_bytes =
      std::filesystem::file_size(base_ivf, size_error);
  size_error.clear();
  const std::uintmax_t lr_bytes =
      std::filesystem::file_size(out_path, size_error);
  if (!size_error) {
    std::printf(
        "bit cost: baseline %ju bytes, chroma LR %ju bytes (%+jd, %+.2f%%); "
        "measured syntax %d header + %d tile bits (frame0) / %d + %d "
        "(frame1)\n",
        base_bytes, lr_bytes,
        static_cast<std::intmax_t>(lr_bytes) -
            static_cast<std::intmax_t>(base_bytes),
        100.0 * (static_cast<double>(lr_bytes) -
                 static_cast<double>(base_bytes)) /
            static_cast<double>(base_bytes),
        decision0.header_bits, decision0.tile_bits, decision1.header_bits,
        decision1.tile_bits);
  }
  if (keep_y4m) std::printf("decoded y4m: %s\n", out_y4m.c_str());
  if (diffs0 != 0 || diffs1 != 0 || !changed0) {
    std::fprintf(stderr, "FAIL: chroma restoration gate\n");
    return 1;
  }
  std::printf(
      "PASS: chroma restoration decoded bit-exactly "
      "(frame0 %d active units, %d uv_sym, %d non-zero; frame1 %d active, "
      "%d uv_sym)\n",
      active0, sym0, non_zero0, active1, sym1);
  return 0;
}
