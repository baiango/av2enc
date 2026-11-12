// LOCALWARP decoder gate for the AV2 warp machinery.
//
// Stream shape (192x128, 64x64 superblocks, one tile):
//   frame 0  OBU_CLOSED_LOOP_KEY, intra, produced by the unmodified M2 core
//   frame 1  OBU_REGULAR_TILE_GROUP, single reference (slot 0), six leaves:
//
//     A (0,0)   NEWMV    MV (0,0)   left/above neighbours for the rest
//     B (0,1)   NEWMV    MV (0,8)
//     C (1,0)   NEWMV    MV (8,0)
//     D (1,1)   WARP_NEWMV + LOCALWARP   (three warp samples => affine fit)
//     E (0,2)   WARP_NEWMV + LOCALWARP   (one warp sample  => det == 0)
//     F (1,2)   WARP_NEWMV + LOCALWARP   (D + E, and an above corner that is
//                                         itself a LOCALWARP block)
//
// The P frame header enables LOCALWARP and DELTAWARP at sequence level and
// LOCALWARP at frame level, so `motion_mode_allowed()` and
// `frame_enabled_motion_modes[]` both admit `use_local_warp`.  `is_warp` uses
// `TileIsWarpCdf[WarpMvCount]` computed by warp_stack.h; `use_local_warp` uses
// the §8.3.2 NPos context.  Both are derived by the same code the encoder uses,
// so a context mismatch in either direction desynchronises the arithmetic
// decoder instead of silently changing a prediction.
//
// Reconstruction uses §7.12.3 + §7.13.3.23 (derive_local_warp) and the existing
// adv::warp_predict() pixel backend.  The gate compares every Y/U/V sample of
// the decoder output with that prediction.
//
// The gate needs tools/bin/dav2d[-macos] (or DAV2D=...).  Without a decoder it
// exits 77 (skip).

#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "core/bitstream/basic_headers.h"
#include "codec/inter/advanced/warp.h"
#include "codec/inter/mc/mc.h"
#include "codec/inter/warp_stack/gate/warp_gate_tile.h"
#include "codec/inter/warp_stack/warp_stack.h"
#include "pipeline/video/video_frame.h"
#include "pipeline/video/video_headers.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

namespace {

using av2enc::BitWriter;
using av2enc::inter::adv::MutablePlaneView;
using av2enc::inter::adv::PlaneView;
using av2enc::inter::adv::WarpBlockGeometry;
using av2enc::inter::adv::WarpParams;
using av2enc::inter::warp::gate::GateLeaf;
using av2enc::inter::warp::gate::GateTileConfig;
using av2enc::inter::warp::gate::GateTilePlan;
using av2enc::inter::warp::WarpMi;
using av2enc::basic::ObuHeader;
using av2enc::basic::ObuType;

struct OwnedFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

std::uint8_t texture_sample(int x, int y) {
  const double fx = static_cast<double>(x);
  const double fy = static_cast<double>(y);
  const double value = 128.0 + 52.0 * std::sin(fx * 0.061) +
                       44.0 * std::cos(fy * 0.077) +
                       30.0 * std::sin((fx + fy) * 0.033) +
                       22.0 * std::cos((fx - fy) * 0.047);
  return static_cast<std::uint8_t>(
      std::clamp(static_cast<int>(std::lround(value)), 0, 255));
}

OwnedFrame make_key(int width, int height) {
  OwnedFrame f;
  f.width = static_cast<std::uint32_t>(width);
  f.height = static_cast<std::uint32_t>(height);
  f.y.resize(static_cast<std::size_t>(width) * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      f.y[static_cast<std::size_t>(y) * width + x] = texture_sample(x, y);
    }
  }
  const int cw = width / 2;
  const int ch = height / 2;
  f.u.resize(static_cast<std::size_t>(cw) * ch);
  f.v.resize(static_cast<std::size_t>(cw) * ch);
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      f.u[static_cast<std::size_t>(y) * cw + x] =
          texture_sample(2 * x + 3, 2 * y + 5);
      f.v[static_cast<std::size_t>(y) * cw + x] =
          texture_sample(2 * x + 11, 2 * y + 7);
    }
  }
  return f;
}

// ---------------------------------------------------------------------------
// Header writers (copied from video_headers.cpp; the motion-mode fields differ)
// ---------------------------------------------------------------------------

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) ++leading_zeros;
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  writer.put_bits(static_cast<std::uint64_t>(value) -
                      ((1ULL << leading_zeros) - 1ULL),
                  leading_zeros);
}

void put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value) {
  if (n <= 1U) return;
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    writer.put_bits(value, w - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, w - 1U);
    writer.put_bit((x & 1U) != 0U);
  }
}

unsigned bit_width(std::uint32_t value) {
  unsigned bits = 0;
  while (value != 0) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

std::uint32_t ceil_log2(std::uint32_t value) {
  if (value <= 1U) return 0U;
  std::uint32_t bits = 0;
  std::uint32_t v = value - 1U;
  while (v != 0) {
    v >>= 1U;
    ++bits;
  }
  return bits;
}

void write_sequence_header_warp(BitWriter& writer,
                                const av2enc::video::VideoConfig& c) {
  put_uvlc(writer, 0);
  writer.put_bits(0U, 5U);          // seq_profile_idc
  writer.put_bit(false);            // single_picture_header_flag
  writer.put_bits(c.seq_level_idx, 5U);
  writer.put_bit(false);            // seq_tier
  put_uvlc(writer, 0);              // chroma_format_idc
  put_uvlc(writer, 1);              // bit_depth_idc = 8-bit
  writer.put_bits(0U, 3U);          // seq_lcr_id
  writer.put_bit(false);            // still_picture
  writer.put_bits(0U, 2U);          // max_tlayer_id
  writer.put_bits(0U, 3U);          // max_mlayer_id
  writer.put_bit(true);             // monotonic_output_order_flag
  unsigned width_bits = bit_width(static_cast<std::uint32_t>(c.max_frame_width) - 1U);
  unsigned height_bits = bit_width(static_cast<std::uint32_t>(c.max_frame_height) - 1U);
  if (width_bits == 0) width_bits = 1;
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_width) - 1U, width_bits);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_height) - 1U, height_bits);
  writer.put_bit(false);  // seq_cropping_window_present_flag
  writer.put_bit(false);  // seq_initial_display_delay_present_flag
  writer.put_bit(false);  // decoder_model_info_present_flag
  writer.put_bit(false);  // use_256x256_superblock
  writer.put_bit(false);  // use_128x128_superblock
  writer.put_bit(false);  // enable_sdp
  writer.put_bit(false);  // enable_ext_partitions
  writer.put_bit(false);  // reduce_pb_aspect_ratio
  writer.put_bit(false);  // enable_ext_seg
  writer.put_bit(false);  // seq_seg_info_present_flag
  writer.put_bit(false);  // enable_dip
  writer.put_bit(true);   // enable_intra_edge_filter
  writer.put_bit(false);  // enable_mrls
  writer.put_bit(true);   // enable_cfl_intra
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);  // enable_mhccp
  writer.put_bit(false);  // enable_ibp

  // §5.4.6 sequence_inter_config().
  writer.put_bit(false);  // seq_enabled_motion_modes[INTERINTRA]
  writer.put_bit(true);   // seq_enabled_motion_modes[LOCALWARP]
  writer.put_bit(true);   // seq_enabled_motion_modes[DELTAWARP]
  writer.put_bit(false);  // seq_enabled_motion_modes[EXTENDWARP]
  writer.put_bit(true);   // seq_frame_motion_modes_present_flag
  writer.put_bit(false);  // enable_six_param_warp_delta (DELTAWARP enabled)
  writer.put_bit(false);  // enable_masked_compound
  writer.put_bit(false);  // enable_ref_frame_mvs
  writer.put_bits(static_cast<std::uint64_t>(c.order_hint_bits) - 1U, 4U);
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(true);   // disable_drl_reorder
  writer.put_bit(true);   // explicit_ref_frame_map
  writer.put_bit(true);   // explicit_num_ref_frames
  writer.put_bits(static_cast<std::uint64_t>(c.num_ref_frames) - 1U, 4U);
  writer.put_bits(0U, 3U);  // long_term_frame_id_bits
  put_ns(writer, 5U, c.max_drl_bits_minus_1);
  writer.put_bit(false);  // allow_frame_max_drl_bits
  put_ns(writer, 3U, 0U);
  writer.put_bit(false);  // allow_frame_max_bvp_drl_bits
  writer.put_bits(0U, 2U);  // num_same_ref_compound
  writer.put_bit(false);  // enable_tip
  writer.put_bit(false);  // enable_mv_traj
  writer.put_bit(false);  // enable_bawp
  writer.put_bit(false);  // enable_cwp
  writer.put_bit(false);  // enable_imp_msk_bld
  writer.put_bit(false);  // enable_df_sub_pu
  writer.put_bits(0U, 2U);  // enable_opfl_refine = REFINE_NONE
  writer.put_bit(false);  // enable_refinemv
  writer.put_bit(false);  // enable_bru
  writer.put_bit(false);  // enable_adaptive_mvd
  writer.put_bit(false);  // enable_mvd_sign_derive
  writer.put_bit(false);  // enable_flex_mvres
  writer.put_bit(false);  // enable_global_motion
  writer.put_bit(false);  // enable_short_refresh_frame_flags
  writer.put_bit(false);  // seq_choose_screen_content_tools
  writer.put_bit(false);  // seq_force_screen_content_tools
  writer.put_bit(false);  // enable_fsc
  writer.put_bit(false);  // enable_idtx_intra
  writer.put_bit(false);  // enable_intra_ist
  writer.put_bit(false);  // enable_inter_ist
  writer.put_bit(true);   // enable_chroma_dctonly
  writer.put_bit(false);  // enable_inter_ddt
  writer.put_bit(false);  // reduced_tx_part_set
  writer.put_bit(false);  // enable_cctx
  writer.put_bit(false);  // enable_tcq
  writer.put_bit(false);  // enable_parity_hiding
  writer.put_bit(false);  // enable_avg_cdf
  writer.put_bit(false);  // separate_uv_delta_q
  writer.put_bit(true);   // equal_ac_dc_q
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q
  writer.put_bit(false);     // uv_ac_delta_q_enabled
  writer.put_bit(false);  // disable_loopfilters_across_tiles
  writer.put_bit(false);  // enable_cdef
  writer.put_bit(false);  // enable_gdf
  writer.put_bit(false);  // enable_restoration
  writer.put_bit(false);  // enable_ccso
  writer.put_bit(false);  // cdef_on_skip_txfm_always_on
  writer.put_bit(true);   // cdef_on_skip_txfm_disabled
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2
  writer.put_bit(false);  // seq_tile_info_present_flag
  writer.put_bit(false);  // film_grain_params_present
}

std::vector<std::uint8_t> make_sequence_obu(const av2enc::video::VideoConfig& config) {
  BitWriter writer;
  write_sequence_header_warp(writer, config);
  writer.put_bit(false);  // obu_extension_flag
  writer.put_bit(true);   // trailing_bits
  writer.align_to_byte();
  return av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

void write_tile_info(BitWriter& writer, const av2enc::video::VideoConfig& c) {
  const std::uint32_t mi_cols =
      2U * ((static_cast<std::uint32_t>(c.max_frame_width) + 7U) >> 3U);
  const std::uint32_t mi_rows =
      2U * ((static_cast<std::uint32_t>(c.max_frame_height) + 7U) >> 3U);
  const std::uint32_t sb_cols = (mi_cols + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows + 15U) >> 4U;
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1U) writer.put_bit(false);
  if (sb_rows > 1U) writer.put_bit(false);
}

void write_frame_header_warp(BitWriter& writer,
                             const av2enc::video::VideoConfig& c,
                             const av2enc::video::VideoFrameHeader& h) {
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  const bool key = h.keyframe;
  if (!key) writer.put_bit(true);  // frame_is_inter
  writer.put_bit(true);            // immediate_output_frame
  writer.put_bit(false);           // frame_size_override_flag
  writer.put_bits(h.order_hint, c.order_hint_bits);
  if (key) {
    writer.put_bit(false);  // allow_intrabc
  } else {
    writer.put_bit(false);  // signal_primary_ref_frame
    writer.put_bit(true);   // disable_cross_frame_cdf_init
    writer.put_bits(h.refresh_frame_flags, c.num_ref_frames);
    writer.put_bit(true);   // frame_explicit_ref_frame_map
    writer.put_bits(c.num_total_refs, 3U);
    writer.put_bits(h.ref_slot,
                    static_cast<unsigned>(ceil_log2(c.num_ref_frames)));
    writer.put_bit(false);  // allow_intrabc
    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // interpolation_filter = EIGHTAP
    // §5.18.2 frame_enabled_motion_modes[]: one bit per sequence-enabled mode.
    writer.put_bit(true);   // frame_enabled_motion_modes[LOCALWARP]
    writer.put_bit(true);   // frame_enabled_motion_modes[DELTAWARP]
  }
  writer.put_bit(c.disable_cdf_update);
  write_tile_info(writer, c);
  writer.put_bits(c.base_q_idx, 8U);
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present
  writer.put_bit(false);  // apply_deblocking_filter[0]
  writer.put_bit(false);  // apply_deblocking_filter[1]
  writer.put_bit(false);  // tx_mode_select
  if (!key) {
    writer.put_bit(false);  // reference_select
    writer.put_bit(false);  // skip_mode_present
    // enable_bawp == 0 -> allow_bawp absent.
    writer.put_bit(true);   // allow_warpmv_mode
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set
  // global_motion_params() absent (enable_global_motion == 0).
  // film_grain_config() absent.
}

std::vector<std::uint8_t> make_p_obu(const av2enc::video::VideoConfig& config,
                                     const av2enc::video::VideoFrameHeader& header,
                                     std::span<const std::uint8_t> tile) {
  BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header_warp(writer, config, header);
  writer.align_to_byte();
  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile.begin(), tile.end());
  return av2enc::basic::make_obu(ObuHeader{ObuType::kRegularTileGroup}, payload);
}

struct Obu {
  std::uint8_t type = 0;
  std::vector<std::uint8_t> whole;
  std::vector<std::uint8_t> payload;
};

bool parse_obus(std::span<const std::uint8_t> bytes, std::vector<Obu>& out) {
  std::size_t pos = 0;
  while (pos < bytes.size()) {
    const std::size_t leb_start = pos;
    std::uint32_t length = 0;
    int shift = 0;
    bool ok = false;
    while (pos < bytes.size()) {
      const std::uint8_t byte = bytes[pos++];
      length |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
      shift += 7;
      if ((byte & 0x80U) == 0U) {
        ok = true;
        break;
      }
    }
    if (!ok || length == 0U || pos + length > bytes.size()) return false;
    Obu obu;
    obu.type = static_cast<std::uint8_t>((bytes[pos] >> 2U) & 0x1FU);
    const std::size_t header_bytes = ((bytes[pos] & 0x80U) != 0U) ? 2U : 1U;
    if (pos + header_bytes > pos + length) return false;
    obu.whole.assign(bytes.begin() + static_cast<std::ptrdiff_t>(leb_start),
                     bytes.begin() + static_cast<std::ptrdiff_t>(pos + length));
    obu.payload.assign(
        bytes.begin() + static_cast<std::ptrdiff_t>(pos + header_bytes),
        bytes.begin() + static_cast<std::ptrdiff_t>(pos + length));
    out.push_back(std::move(obu));
    pos += length;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------

bool warp_plane(const WarpParams& params, const std::uint8_t* ref, int ref_stride,
                int plane_w, int plane_h, int plane, int sub, int x, int y,
                int w, int h, int mi_cols, int mi_rows,
                std::vector<std::int16_t>& scratch, std::uint8_t* dst,
                int dst_stride) {
  WarpBlockGeometry geom;
  geom.plane = plane;
  geom.sub_x = sub;
  geom.sub_y = sub;
  geom.rect = av2enc::inter::adv::BlockRect{x, y, w, h};
  geom.dst_x = 0;
  geom.dst_y = 0;
  geom.mi_col = (x << sub) / 4;
  geom.mi_row = (y << sub) / 4;
  geom.bw4 = (w << sub) / 4;
  geom.bh4 = (h << sub) / 4;
  geom.mi_cols = mi_cols;
  geom.mi_rows = mi_rows;
  geom.ref_w = plane_w;
  geom.ref_h = plane_h;
  geom.frame_w = plane_w;
  geom.frame_h = plane_h;
  geom.bit_depth = 8;

  PlaneView ref_view;
  ref_view.data = ref;
  ref_view.stride = ref_stride;
  ref_view.max_x = plane_w - 1;
  ref_view.max_y = plane_h - 1;

  scratch.resize(static_cast<std::size_t>(w) * h);
  const bool extended = av2enc::inter::adv::warp_needs_extended(params, geom);
  const bool ok = extended
                      ? av2enc::inter::adv::warp_predict_ext(params, ref_view,
                                                             geom, false,
                                                             scratch.data(), w)
                      : av2enc::inter::adv::warp_predict(params, ref_view, geom,
                                                         false, scratch.data(), w);
  if (!ok) return false;
  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < w; ++col) {
      dst[static_cast<std::ptrdiff_t>(row) * dst_stride + col] =
          static_cast<std::uint8_t>(
              scratch[static_cast<std::size_t>(row) * w + col]);
    }
  }
  return true;
}

bool reconstruct_p(const OwnedFrame& key, const std::vector<WarpMi>& grid,
                   int mi_cols, int mi_rows, OwnedFrame* out, std::string& error) {
  const int w = static_cast<int>(key.width);
  const int h = static_cast<int>(key.height);
  const int cw = w / 2;
  const int ch = h / 2;
  out->width = key.width;
  out->height = key.height;
  out->y.assign(static_cast<std::size_t>(w) * h, 0);
  out->u.assign(static_cast<std::size_t>(cw) * ch, 0);
  out->v.assign(static_cast<std::size_t>(cw) * ch, 0);

  av2enc::inter::mc::RefFrame ref;
  ref.plane[0] = av2enc::inter::mc::PlaneView{key.y.data(), w};
  ref.plane[1] = av2enc::inter::mc::PlaneView{key.u.data(), cw};
  ref.plane[2] = av2enc::inter::mc::PlaneView{key.v.data(), cw};
  ref.mi_cols = mi_cols;
  ref.mi_rows = mi_rows;

  std::vector<std::int16_t> scratch;
  for (int r = 0; r < mi_rows; r += 16) {
    for (int c = 0; c < mi_cols; c += 16) {
      const WarpMi& m = grid[static_cast<std::size_t>(r) * mi_cols + c];
      if (m.motion_mode >= av2enc::inter::warp::kMotionLocalWarp) {
        av2enc::inter::warp::LocalWarpParams lp;
        lp.mi_row = r;
        lp.mi_col = c;
        lp.w4 = 16;
        lp.h4 = 16;
        lp.mi_cols = mi_cols;
        lp.mi_rows = mi_rows;
        lp.sb_h4 = 16;
        lp.grid = av2enc::inter::warp::WarpGrid{mi_cols, mi_rows, grid.data()};
        lp.ref_frame[0] = 0;
        lp.ref_frame[1] = av2enc::inter::warp::kRefNone;
        lp.avail_u = r > 0;
        lp.avail_l = c > 0;
        lp.block_mv = av2enc::inter::mc::Mv{m.mv[0][0], m.mv[0][1]};
        const auto derived = av2enc::inter::warp::derive_local_warp(lp, 0);
        if (!derived.has_value()) {
          error = "derive_local_warp failed";
          return false;
        }
        if (!(derived->params.p[0] == m.warp[0].p[0] &&
              derived->params.p[1] == m.warp[0].p[1] &&
              derived->params.p[2] == m.warp[0].p[2] &&
              derived->params.p[3] == m.warp[0].p[3] &&
              derived->params.p[4] == m.warp[0].p[4] &&
              derived->params.p[5] == m.warp[0].p[5])) {
          error = "encoder/decoder local warp model mismatch";
          return false;
        }
        const int bx = c * 4;
        const int by = r * 4;
        if (!warp_plane(derived->params, key.y.data(), w, w, h, 0, 0, bx, by,
                        64, 64, mi_cols, mi_rows, scratch,
                        out->y.data() + static_cast<std::ptrdiff_t>(by) * w + bx,
                        w)) {
          error = "luma warp prediction failed";
          return false;
        }
        if (!warp_plane(derived->params, key.u.data(), cw, cw, ch, 1, 1, bx / 2,
                        by / 2, 32, 32, mi_cols, mi_rows, scratch,
                        out->u.data() + static_cast<std::ptrdiff_t>(by / 2) * cw +
                            bx / 2,
                        cw)) {
          error = "chroma U warp prediction failed";
          return false;
        }
        if (!warp_plane(derived->params, key.v.data(), cw, cw, ch, 2, 1, bx / 2,
                        by / 2, 32, 32, mi_cols, mi_rows, scratch,
                        out->v.data() + static_cast<std::ptrdiff_t>(by / 2) * cw +
                            bx / 2,
                        cw)) {
          error = "chroma V warp prediction failed";
          return false;
        }
        continue;
      }

      const av2enc::inter::mc::BlockRect rect{c * 4, r * 4, 64, 64};
      av2enc::inter::mc::MutablePlane dst[3];
      dst[0] = av2enc::inter::mc::MutablePlane{
          out->y.data() + static_cast<std::ptrdiff_t>(r * 4) * w + c * 4, w};
      dst[1] = av2enc::inter::mc::MutablePlane{
          out->u.data() + static_cast<std::ptrdiff_t>(r * 2) * cw + c * 2, cw};
      dst[2] = av2enc::inter::mc::MutablePlane{
          out->v.data() + static_cast<std::ptrdiff_t>(r * 2) * cw + c * 2, cw};
      const auto status = av2enc::inter::mc::predict_block_420(
          ref, rect, av2enc::inter::mc::Mv{m.mv[0][0], m.mv[0][1]},
          av2enc::inter::mc::InterpFilter::kEighttap, dst);
      if (status != av2enc::inter::mc::McStatus::kOk) {
        error = std::string("MC failed: ") +
                av2enc::inter::mc::mc_status_string(status);
        return false;
      }
    }
  }
  return true;
}

std::string find_dav2d() {
  if (const char* env = std::getenv("DAV2D")) {
    if (std::filesystem::exists(env)) return env;
  }
  for (const char* candidate :
       {"tools/bin/dav2d", "tools/bin/dav2d-macos", "dav2d", "dav2d-macos"}) {
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return {};
}

bool run_decoder(const std::string& decoder, const std::string& ivf,
                 const std::string& y4m, std::string& error) {
  const std::string command = "\"" + decoder + "\" --strict 1 --threads 1 -q -i \"" +
                              ivf + "\" -o \"" + y4m + "\" >/dev/null 2>&1";
  const int status = std::system(command.c_str());
  if (status != 0) {
    error = "dav2d exited with status " + std::to_string(status);
    return false;
  }
  return true;
}

std::uint64_t compare_plane(std::span<const std::uint8_t> decoded,
                            const std::vector<std::uint8_t>& expected,
                            int* mismatches, int* first_x, int* first_y,
                            int stride) {
  std::uint64_t total = 0;
  const std::size_t n = std::min(decoded.size(), expected.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (decoded[i] != expected[i]) {
      ++total;
      if (*mismatches == 0) {
        *first_x = static_cast<int>(i % static_cast<std::size_t>(stride));
        *first_y = static_cast<int>(i / static_cast<std::size_t>(stride));
        std::fprintf(stderr, "first diff at (%d,%d): decoded=%d expected=%d\n",
                     *first_x, *first_y, static_cast<int>(decoded[i]),
                     static_cast<int>(expected[i]));
      }
      ++*mismatches;
    }
  }
  return total;
}

bool write_ivf(const std::string& path,
               const std::vector<std::vector<std::uint8_t>>& units, int width,
               int height, std::string& error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = static_cast<std::uint32_t>(units.size());
  if (!writer.open(path, header, error)) return false;
  for (std::size_t i = 0; i < units.size(); ++i) {
    if (!writer.write_frame(units[i], i, error)) return false;
  }
  return writer.close(error);
}

bool load_key_recon(const av2enc::video::VideoReconFrame& src, OwnedFrame* out) {
  out->width = src.width;
  out->height = src.height;
  out->y = src.y;
  out->u = src.u;
  out->v = src.v;
  return !out->y.empty();
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_path = "/tmp/warp_gate.ivf";
  std::string decoded_path = "/tmp/warp_gate_decoded.y4m";
  bool decode = true;
  bool negative_context = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--decoded" && i + 1 < argc) {
      decoded_path = argv[++i];
    } else if (arg == "--no-decode") {
      decode = false;
    } else if (arg == "--negative-context") {
      negative_context = true;
    } else if (arg == "-h" || arg == "--help") {
      std::printf(
          "usage: %s [--out out.ivf] [--decoded out.y4m] [--no-decode]\n"
          "          [--negative-context]\n",
          argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  const int width = 192;
  const int height = 128;
  const int mi_cols = width / 4;
  const int mi_rows = height / 4;
  const OwnedFrame key_source = make_key(width, height);

  // Keyframe tile from the unmodified M2 core.
  std::vector<av2enc::video::VideoInputFrame> inputs;
  {
    av2enc::video::VideoInputFrame in;
    in.width = key_source.width;
    in.height = key_source.height;
    in.y = std::span<const std::uint8_t>(key_source.y.data(), key_source.y.size());
    in.u = std::span<const std::uint8_t>(key_source.u.data(), key_source.u.size());
    in.v = std::span<const std::uint8_t>(key_source.v.data(), key_source.v.size());
    inputs.push_back(in);
    inputs.push_back(in);
  }
  av2enc::video::VideoOptions options;
  options.base_q_idx = 90;
  options.force_global_mv = true;
  options.force_single_block = true;
  options.enable_intra_fallback = false;
  options.enable_newmv = false;
  options.enable_nearmv = false;
  options.force_intra = false;
  options.min_block_size = 64;
  options.max_partition_depth = 0;
  av2enc::video::VideoSequenceResult sequence;
  std::string error;
  if (!av2enc::video::encode_video_sequence(inputs, options, sequence, error)) {
    std::fprintf(stderr, "M2 keyframe encode failed: %s\n", error.c_str());
    return 1;
  }
  OwnedFrame key;
  if (!load_key_recon(sequence.recon[0], &key)) {
    std::fprintf(stderr, "missing M2 key reconstruction\n");
    return 1;
  }
  std::vector<Obu> key_obus;
  if (!parse_obus(sequence.units[0], key_obus)) {
    std::fprintf(stderr, "failed to parse the M2 key temporal unit\n");
    return 1;
  }
  std::vector<std::uint8_t> key_frame_obu;
  for (const Obu& obu : key_obus) {
    if (obu.type == static_cast<std::uint8_t>(ObuType::kClosedLoopKey)) {
      key_frame_obu = obu.whole;
      break;
    }
  }
  if (key_frame_obu.empty()) {
    std::fprintf(stderr, "missing CLK OBU\n");
    return 1;
  }

  // Leaf plan: A/B/C introduce distinct MVs; D/E/F are LOCALWARP.
  GateTilePlan plan;
  plan.width = width;
  plan.height = height;
  plan.sb_size = 64;
  // Leaves are in superblock raster order: (0,0) A, (0,1) B, (0,2) E,
  // (1,0) C, (1,1) D, (1,2) F.
  struct LeafSpec {
    int mv_y;
    int mv_x;
    bool warp;
  };
  const LeafSpec leaf_specs[6] = {
      {0, 0, false},  // A (0,0) NEWMV
      {0, 8, false},  // B (0,1) NEWMV
      {0, 8, true},   // E (0,2) LOCALWARP, one sample
      {8, 0, false},  // C (1,0) NEWMV
      {8, 0, true},   // D (1,1) LOCALWARP, three samples
      {8, 0, true},   // F (1,2) LOCALWARP, D + E
  };
  for (int sb = 0; sb < 6; ++sb) {
    GateLeaf leaf;
    leaf.use_target_mv = true;
    leaf.target_mv_y = leaf_specs[sb].mv_y;
    leaf.target_mv_x = leaf_specs[sb].mv_x;
    leaf.drl_index = 0;
    if (!leaf_specs[sb].warp) {
      leaf.is_warp = false;
      leaf.single_mode = 2;  // NEWMV
      leaf.y_mode = av2enc::inter::warp::kYModeNewMv;
      leaf.motion_mode = av2enc::inter::warp::kMotionSimple;
    } else {
      leaf.is_warp = true;
      leaf.warp_mv = false;  // WARP_NEWMV
      leaf.use_local_warp = true;
      leaf.y_mode = av2enc::inter::warp::kYModeWarpNewMv;
      leaf.motion_mode = av2enc::inter::warp::kMotionLocalWarp;
    }
    plan.leaves.push_back(leaf);
  }

  GateTileConfig tile_config;
  tile_config.allow_warpmv_mode = true;
  tile_config.frame_enable_local_warp = true;
  tile_config.frame_enable_extend_warp = false;
  tile_config.force_integer_mv = false;
  tile_config.disable_cdf_update = true;
  tile_config.max_drl_bits_minus_1 = 4;
  if (negative_context) tile_config.force_warp_mv_ctx = 0;

  std::vector<std::uint8_t> tile;
  std::vector<WarpMi> grid;
  int tile_mi_cols = 0;
  int tile_mi_rows = 0;
  if (!av2enc::inter::warp::gate::write_gate_warp_tile(
          plan, tile_config, tile, &grid, &tile_mi_cols, &tile_mi_rows, error)) {
    std::fprintf(stderr, "tile write failed: %s\n", error.c_str());
    return 1;
  }
  if (tile_mi_cols != mi_cols || tile_mi_rows != mi_rows) {
    std::fprintf(stderr, "tile geometry mismatch\n");
    return 1;
  }

  av2enc::video::VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = 90;
  config.order_hint_bits = 4;
  config.num_ref_frames = 2;
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = true;

  av2enc::video::VideoFrameHeader p_header;
  p_header.keyframe = false;
  p_header.order_hint = 1;
  p_header.ref_slot = 0;
  p_header.refresh_frame_flags = 1;

  std::vector<std::uint8_t> unit0;
  {
    const std::vector<std::uint8_t> td =
        av2enc::video::make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> seq = make_sequence_obu(config);
    unit0.insert(unit0.end(), td.begin(), td.end());
    unit0.insert(unit0.end(), seq.begin(), seq.end());
    unit0.insert(unit0.end(), key_frame_obu.begin(), key_frame_obu.end());
  }
  std::vector<std::uint8_t> unit1;
  {
    const std::vector<std::uint8_t> td =
        av2enc::video::make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> p_obu = make_p_obu(config, p_header, tile);
    unit1.insert(unit1.end(), td.begin(), td.end());
    unit1.insert(unit1.end(), p_obu.begin(), p_obu.end());
  }
  const std::vector<std::vector<std::uint8_t>> units = {unit0, unit1};
  if (!write_ivf(out_path, units, width, height, error)) {
    std::fprintf(stderr, "IVF write failed: %s\n", error.c_str());
    return 1;
  }

  OwnedFrame expected;
  if (!reconstruct_p(key, grid, mi_cols, mi_rows, &expected, error)) {
    std::fprintf(stderr, "reconstruction failed: %s\n", error.c_str());
    return 1;
  }

  std::printf("warp_gate: %dx%d, tile %zu bytes, %d LOCALWARP blocks\n", width,
              height, tile.size(), 3);

  if (!decode) return 0;
  const std::string decoder = find_dav2d();
  if (decoder.empty()) {
    std::fprintf(stderr, "warp_gate: no dav2d found (skip)\n");
    return 77;
  }
  if (!run_decoder(decoder, out_path, decoded_path, error)) {
    if (negative_context) {
      std::printf("warp_gate: negative control rejected by the decoder (%s)\n",
                  error.c_str());
      return 0;
    }
    std::fprintf(stderr, "warp_gate: %s\n", error.c_str());
    return 1;
  }
  av2enc::Y4mReader reader(decoded_path);
  if (!reader.open(error)) {
    std::fprintf(stderr, "decoded y4m open failed: %s\n", error.c_str());
    return 1;
  }
  av2enc::Y4mFrame frame;
  if (reader.read_frame(frame, error) != av2enc::Y4mReadResult::kFrame) {
    std::fprintf(stderr, "decoded key frame missing\n");
    return 1;
  }
  int key_mismatch = 0;
  int kx = 0;
  int ky = 0;
  compare_plane(frame.y(), key.y, &key_mismatch, &kx, &ky, width);
  if (key_mismatch != 0) {
    std::fprintf(stderr, "key recon mismatch at (%d,%d)\n", kx, ky);
    return 1;
  }
  if (reader.read_frame(frame, error) != av2enc::Y4mReadResult::kFrame) {
    if (negative_context) {
      std::printf("warp_gate: negative control dropped the P frame\n");
      return 0;
    }
    std::fprintf(stderr, "decoded P frame missing\n");
    return 1;
  }
  int mismatch = 0;
  int fx = 0;
  int fy = 0;
  compare_plane(frame.y(), expected.y, &mismatch, &fx, &fy, width);
  compare_plane(frame.u(), expected.u, &mismatch, &fx, &fy, width / 2);
  compare_plane(frame.v(), expected.v, &mismatch, &fx, &fy, width / 2);
  if (negative_context) {
    if (mismatch == 0) {
      std::fprintf(stderr,
                   "warp_gate: negative control was expected to diverge but "
                   "reconstructed exactly\n");
      return 1;
    }
    std::printf("warp_gate: negative control diverged (%d mismatches, first "
                "at (%d,%d))\n",
                mismatch, fx, fy);
    return 0;
  }
  if (mismatch != 0) {
    std::fprintf(stderr,
                 "warp_gate: %d sample mismatches, first at (%d,%d)\n",
                 mismatch, fx, fy);
    return 1;
  }
  std::printf("warp_gate: LOCALWARP reconstruction exact (%d luma + chroma samples)\n",
              width * height + 2 * (width / 2) * (height / 2));
  return 0;
}
