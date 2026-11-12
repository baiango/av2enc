// Minimal two-frame decoder gate for AV2 global motion.
//
// The M2 core hard-codes enable_global_motion = 0 in its sequence header and
// has no global_motion_params() writer. This gate therefore:
//   1. uses the M2 keyframe/P-tile writers unchanged to obtain a valid intra
//      keyframe and an all-GLOBALMV skip tile,
//   2. re-wraps them with a sequence header whose enable_global_motion bit is
//      set and a P frame header containing global_motion_params(),
//   3. reconstructs the P frame with the normative §7.13.3.19/§7.13.3.20 warp
//      predictor using the *decoded* parameters,
//   4. decodes the IVF with the pinned dav2d and compares every sample.
//
// Nothing outside src/encoder/inter/gme/ is edited.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "core/bitstream/basic_headers.h"
#include "codec/inter/advanced/warp.h"
#include "codec/inter/gme/gme.h"
#include "codec/inter/gme/gme_syntax.h"
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
using av2enc::inter::gme::GmBitReader;
using av2enc::inter::gme::GmEstimate;
using av2enc::inter::gme::GmOptions;
using av2enc::inter::gme::GmParams;
using av2enc::inter::gme::GmSyntaxContext;
using av2enc::inter::gme::GmType;
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
  const double value = 128.0 + 58.0 * std::sin(fx * 0.057) +
                       48.0 * std::cos(fy * 0.071) +
                       34.0 * std::sin((fx + fy) * 0.031) +
                       24.0 * std::cos((fx - fy) * 0.043);
  return static_cast<std::uint8_t>(std::clamp(static_cast<int>(std::lround(value)), 0, 255));
}

std::vector<OwnedFrame> make_synthetic(int width, int height, int frames,
                                       const GmParams& motion) {
  std::vector<OwnedFrame> out;
  const int cw = width / 2;
  const int ch = height / 2;
  OwnedFrame base;
  base.width = static_cast<std::uint32_t>(width);
  base.height = static_cast<std::uint32_t>(height);
  base.y.resize(static_cast<std::size_t>(width) * height);
  base.u.assign(static_cast<std::size_t>(cw) * ch, 128);
  base.v.assign(static_cast<std::size_t>(cw) * ch, 128);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      base.y[static_cast<std::size_t>(y) * width + x] = texture_sample(x, y);
    }
  }
  out.push_back(base);
  for (int i = 1; i < frames; ++i) {
    OwnedFrame warped = out.back();
    const bool ok = av2enc::inter::gme::warp_plane_bilinear(
        motion, out.back().y.data(), width, height, width, warped.y.data(), width, height,
        width, 8);
    if (!ok) return {};
    out.push_back(std::move(warped));
  }
  return out;
}

bool load_y4m(const std::string& path, int max_frames, std::vector<OwnedFrame>& out,
              std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  while (static_cast<int>(out.size()) < max_frames) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    OwnedFrame owned;
    owned.width = frame.width();
    owned.height = frame.height();
    owned.y.assign(frame.y().begin(), frame.y().end());
    owned.u.assign(frame.u().begin(), frame.u().end());
    owned.v.assign(frame.v().begin(), frame.v().end());
    out.push_back(std::move(owned));
  }
  return !out.empty();
}

// Crop to a multiple of 64 so the M2 force_single_block keyframe path has no
// partial bottom/right superblock.
OwnedFrame crop64(const OwnedFrame& input) {
  OwnedFrame out;
  out.width = (input.width / 64U) * 64U;
  out.height = (input.height / 64U) * 64U;
  if (out.width == 0 || out.height == 0) return out;
  const int cw = static_cast<int>(out.width / 2U);
  const int chh = static_cast<int>(out.height / 2U);
  out.y.resize(static_cast<std::size_t>(out.width) * out.height);
  out.u.resize(static_cast<std::size_t>(cw) * chh);
  out.v.resize(static_cast<std::size_t>(cw) * chh);
  for (std::uint32_t y = 0; y < out.height; ++y) {
    std::memcpy(out.y.data() + static_cast<std::size_t>(y) * out.width,
                input.y.data() + static_cast<std::size_t>(y) * input.width,
                out.width);
  }
  for (std::uint32_t y = 0; y < static_cast<std::uint32_t>(chh); ++y) {
    std::memcpy(out.u.data() + static_cast<std::size_t>(y) * cw,
                input.u.data() + static_cast<std::size_t>(y) * (input.width / 2U), cw);
    std::memcpy(out.v.data() + static_cast<std::size_t>(y) * cw,
                input.v.data() + static_cast<std::size_t>(y) * (input.width / 2U), cw);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Sequence header writer with the global-motion bit.
// Copied from video_headers.cpp; verified byte-identical when enable_gm == false.
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

void write_sequence_header_with_gm(BitWriter& writer, const av2enc::video::VideoConfig& c,
                                   bool enable_gm) {
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
  for (int mode = 1; mode < 5; ++mode) writer.put_bit(false);  // motion modes
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
  writer.put_bits(0U, 2U);  // enable_opfl_refine
  writer.put_bit(false);  // enable_refinemv
  writer.put_bit(false);  // enable_bru
  writer.put_bit(false);  // enable_adaptive_mvd
  writer.put_bit(false);  // enable_mvd_sign_derive
  writer.put_bit(false);  // enable_flex_mvres
  writer.put_bit(enable_gm);  // enable_global_motion
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

std::vector<std::uint8_t> make_sequence_obu(const av2enc::video::VideoConfig& config, bool enable_gm) {
  BitWriter writer;
  write_sequence_header_with_gm(writer, config, enable_gm);
  writer.put_bit(false);  // obu_extension_flag
  writer.put_bit(true);   // trailing_bits
  writer.align_to_byte();
  return av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

// ---------------------------------------------------------------------------
// OBU parsing (Annex B length-delimited).
// ---------------------------------------------------------------------------

struct Obu {
  std::uint8_t type = 0;
  std::vector<std::uint8_t> whole;    // leb128 length + header + payload
  std::vector<std::uint8_t> payload;  // header excluded
};

bool parse_obus(std::span<const std::uint8_t> bytes, std::vector<Obu>& out,
                std::string& error) {
  std::size_t pos = 0;
  while (pos < bytes.size()) {
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
    if (!ok || length == 0U || pos + length > bytes.size()) {
      error = "malformed leb128/OBU";
      return false;
    }
    Obu obu;
    obu.type = static_cast<std::uint8_t>((bytes[pos] >> 2U) & 0x1FU);
    obu.whole.assign(bytes.begin() + static_cast<std::ptrdiff_t>(pos) - (shift / 7),
                     bytes.begin() + static_cast<std::ptrdiff_t>(pos + length));
    const std::size_t header_bytes = ((bytes[pos] & 0x80U) != 0U) ? 2U : 1U;
    if (pos + header_bytes > pos + length) {
      error = "truncated OBU header";
      return false;
    }
    obu.payload.assign(bytes.begin() + static_cast<std::ptrdiff_t>(pos + header_bytes),
                       bytes.begin() + static_cast<std::ptrdiff_t>(pos + length));
    out.push_back(std::move(obu));
    pos += length;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Reconstruction using the normative warp predictor.
// ---------------------------------------------------------------------------

bool warp_block(const GmParams& params, const std::uint8_t* ref, int ref_stride, int plane_w,
                int plane_h, int plane, int sub, int x, int y, int block_w, int block_h,
                int mi_cols, int mi_rows, std::vector<std::int16_t>& scratch,
                std::uint8_t* dst, int dst_stride, std::string& error) {
  if (params.type == GmType::kIdentity) {
    for (int row = 0; row < block_h; ++row) {
      std::memcpy(dst + static_cast<std::ptrdiff_t>(row) * dst_stride, 
                  ref + static_cast<std::ptrdiff_t>(y + row) * ref_stride + x,
                  static_cast<std::size_t>(block_w));
    }
    return true;
  }
  WarpParams warp;
  for (int i = 0; i < 6; ++i) warp.p[i] = params.p[static_cast<std::size_t>(i)];
  WarpBlockGeometry geom;
  geom.plane = plane;
  geom.sub_x = sub;
  geom.sub_y = sub;
  geom.rect = av2enc::inter::adv::BlockRect{x, y, block_w, block_h};
  geom.dst_x = 0;
  geom.dst_y = 0;
  geom.mi_col = (x << sub) / 4;
  geom.mi_row = (y << sub) / 4;
  geom.bw4 = (block_w << sub) / 4;
  geom.bh4 = (block_h << sub) / 4;
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

  scratch.resize(static_cast<std::size_t>(block_w) * block_h);
  const bool extended = av2enc::inter::adv::warp_needs_extended(warp, geom);
  const bool ok = extended
                      ? av2enc::inter::adv::warp_predict_ext(warp, ref_view, geom, false,
                                                             scratch.data(), block_w)
                      : av2enc::inter::adv::warp_predict(warp, ref_view, geom, false,
                                                         scratch.data(), block_w);
  if (!ok) {
    error = "warp_predict failed";
    return false;
  }
  for (int row = 0; row < block_h; ++row) {
    for (int col = 0; col < block_w; ++col) {
      dst[static_cast<std::ptrdiff_t>(row) * dst_stride + col] =
          static_cast<std::uint8_t>(scratch[static_cast<std::size_t>(row) * block_w + col]);
    }
  }
  return true;
}

OwnedFrame reconstruct_p(const OwnedFrame& key, const GmParams& params) {
  OwnedFrame out;
  out.width = key.width;
  out.height = key.height;
  const int w = static_cast<int>(out.width);
  const int h = static_cast<int>(out.height);
  const int cw = w / 2;
  const int ch = h / 2;
  out.y.resize(static_cast<std::size_t>(w) * h);
  out.u.resize(static_cast<std::size_t>(cw) * ch);
  out.v.resize(static_cast<std::size_t>(cw) * ch);
  std::vector<std::int16_t> scratch;
  std::string error;
  const int mi_cols = w / 4;
  const int mi_rows = h / 4;
  for (int y = 0; y < h; y += 64) {
    for (int x = 0; x < w; x += 64) {
      const int bw = std::min(64, w - x);
      const int bh = std::min(64, h - y);
      if (!warp_block(params, key.y.data(), w, w, h, 0, 0, x, y, bw, bh, mi_cols, mi_rows,
                      scratch, out.y.data() + static_cast<std::ptrdiff_t>(y) * w + x, w,
                      error)) {
        std::fprintf(stderr, "recon luma failed: %s\n", error.c_str());
        return {};
      }
      const int cbw = std::max(1, bw / 2);
      const int cbh = std::max(1, bh / 2);
      if (!warp_block(params, key.u.data(), cw, cw, ch, 1, 1, x / 2, y / 2, cbw, cbh, mi_cols,
                      mi_rows, scratch,
                      out.u.data() + static_cast<std::ptrdiff_t>(y / 2) * cw + x / 2, cw,
                      error)) {
        std::fprintf(stderr, "recon u failed: %s\n", error.c_str());
        return {};
      }
      if (!warp_block(params, key.v.data(), cw, cw, ch, 2, 1, x / 2, y / 2, cbw, cbh, mi_cols,
                      mi_rows, scratch,
                      out.v.data() + static_cast<std::ptrdiff_t>(y / 2) * cw + x / 2, cw,
                      error)) {
        std::fprintf(stderr, "recon v failed: %s\n", error.c_str());
        return {};
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Decoder.
// ---------------------------------------------------------------------------

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

bool run_decoder(const std::string& decoder, const std::string& ivf, const std::string& y4m,
                 std::string& error) {
  const std::string command = "\"" + decoder + "\" -q -i \"" + ivf + "\" -o \"" + y4m +
                              "\" --threads 1 >/dev/null 2>&1";
  const int status = std::system(command.c_str());
  if (status != 0) {
    error = "dav2d exited with status " + std::to_string(status);
    return false;
  }
  return true;
}

std::uint64_t compare_plane(std::span<const std::uint8_t> decoded,
                            const std::vector<std::uint8_t>& expected, int* max_diff) {
  std::uint64_t mismatches = 0;
  const std::size_t n = std::min(decoded.size(), expected.size());
  for (std::size_t i = 0; i < n; ++i) {
    const int diff = std::abs(static_cast<int>(decoded[i]) - static_cast<int>(expected[i]));
    if (diff != 0) {
      ++mismatches;
      *max_diff = std::max(*max_diff, diff);
    }
  }
  return mismatches;
}

bool write_ivf(const std::string& path, const std::vector<std::vector<std::uint8_t>>& units,
               int width, int height, std::string& error) {
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

}  // namespace

int main(int argc, char** argv) {
  std::string input_path;
  std::string out_path = "/tmp/gme_gate.ivf";
  std::string decoded_path = "/tmp/gme_gate_decoded.y4m";
  bool decode = true;
  bool affine = false;
  bool translation = false;
  bool shear = false;
  bool base_diff = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input_path = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--decoded" && i + 1 < argc) {
      decoded_path = argv[++i];
    } else if (arg == "--affine") {
      affine = true;
    } else if (arg == "--translation") {
      translation = true;
    } else if (arg == "--shear") {
      shear = true;
    } else if (arg == "--base-diff") {
      base_diff = true;
    } else if (arg == "--no-decode") {
      decode = false;
    } else if (arg == "-h" || arg == "--help") {
      std::printf(
          "usage: %s [--input clip.y4m] [--out out.ivf] [--decoded out.y4m]\n"
          "          [--affine|--translation] [--no-decode]\n",
          argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  // Global motion model chosen for the gate: a mild pan + zoom (+ rotation or
  // shear depending on the selected syntax type).
  GmOptions gm_options;
  gm_options.frame_width = 0;
  gm_options.frame_height = 0;
  GmParams desired;
  if (affine) {
    desired.type = GmType::kAffine;
    desired.p = {4 * 65536, -3 * 65536, 66000, 8000, -6000, 72000};
  } else if (shear) {
    // Fails §7.13.3.21 shear validity, so the decoder must take the
    // §7.13.3.20 extended-warp fallback.
    desired.type = GmType::kAffine;
    desired.p = {4 * 65536, -3 * 65536, 65536, 30000, -30000, 70000};
  } else if (translation) {
    desired.type = GmType::kRotZoom;
    desired.p = {5 * 65536, -4 * 65536, 65536, 0, 0, 65536};
  } else {
    desired.type = GmType::kRotZoom;
    desired.p = {4 * 65536, -3 * 65536, 67000, -1800, 1800, 67000};
  }
  GmEstimate estimate = av2enc::inter::gme::finalize_gm_params(desired, gm_options);
  if (!estimate.diag.accepted || estimate.diag.identity) {
    std::fprintf(stderr, "gate model rejected: %s\n", estimate.diag.reason.c_str());
    return 1;
  }
  const GmParams model = estimate.params;

  const int needed_frames = base_diff ? 3 : 2;
  std::vector<OwnedFrame> frames;
  if (!input_path.empty()) {
    std::string error;
    if (!load_y4m(input_path, needed_frames, frames, error)) {
      std::fprintf(stderr, "input load failed: %s\n", error.c_str());
      return 1;
    }
    for (OwnedFrame& frame : frames) frame = crop64(frame);
    if (static_cast<int>(frames.size()) < needed_frames || frames[0].width < 64 ||
        frames[0].height < 64) {
      std::fprintf(stderr, "input must provide at least %d frames >= 64x64\n",
                   needed_frames);
      return 1;
    }
  } else {
    frames = make_synthetic(256, 256, needed_frames, model);
    if (frames.size() < 2) {
      std::fprintf(stderr, "synthetic frame generation failed\n");
      return 1;
    }
  }
  const int width = static_cast<int>(frames[0].width);
  const int height = static_cast<int>(frames[0].height);
  gm_options.frame_width = width;
  gm_options.frame_height = height;
  estimate = av2enc::inter::gme::finalize_gm_params(desired, gm_options);
  if (!estimate.diag.accepted || estimate.diag.identity) {
    std::fprintf(stderr, "gate model rejected with frame geometry: %s\n",
                 estimate.diag.reason.c_str());
    return 1;
  }
  const GmParams final_model = estimate.params;

  // Keyframe and all-GLOBALMV skip tile from the unmodified M2 writer.
  std::vector<av2enc::video::VideoInputFrame> inputs;
  for (const OwnedFrame& frame : frames) {
    av2enc::video::VideoInputFrame input;
    input.width = frame.width;
    input.height = frame.height;
    input.y = std::span<const std::uint8_t>(frame.y.data(), frame.y.size());
    input.u = std::span<const std::uint8_t>(frame.u.data(), frame.u.size());
    input.v = std::span<const std::uint8_t>(frame.v.data(), frame.v.size());
    inputs.push_back(input);
  }
  av2enc::video::VideoOptions video_options;
  video_options.base_q_idx = 90;
  video_options.force_global_mv = true;
  video_options.force_single_block = true;
  video_options.enable_intra_fallback = false;
  video_options.enable_newmv = false;
  video_options.enable_nearmv = false;
  video_options.force_intra = false;
  video_options.min_block_size = 64;
  video_options.max_partition_depth = 0;
  av2enc::video::VideoSequenceResult sequence;
  std::string error;
  if (!av2enc::video::encode_video_sequence(inputs, video_options, sequence, error)) {
    std::fprintf(stderr, "M2 encode failed: %s\n", error.c_str());
    return 1;
  }

  av2enc::video::VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = video_options.base_q_idx;
  config.order_hint_bits = 4;
  config.num_ref_frames = 2;
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = true;
  config.enable_cfl_intra = true;

  // Sanity: our sequence header must match the M2 header when GM is disabled.
  {
    BitWriter baseline_writer;
    write_sequence_header_with_gm(baseline_writer, config, false);
    baseline_writer.put_bit(false);
    baseline_writer.put_bit(true);
    baseline_writer.align_to_byte();
    const std::vector<std::uint8_t> baseline =
        av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader},
                             baseline_writer.take_bytes());
    const std::vector<std::uint8_t> reference =
        av2enc::video::make_sequence_header_obu(config);
    if (baseline != reference) {
      std::fprintf(stderr, "sequence header copy differs from m2 baseline\n");
      return 1;
    }
  }

  std::vector<Obu> key_obus;
  std::vector<Obu> p_obus;
  if (!parse_obus(sequence.units[0], key_obus, error) ||
      !parse_obus(sequence.units[1], p_obus, error)) {
    std::fprintf(stderr, "OBU parse failed: %s\n", error.c_str());
    return 1;
  }
  const Obu* key_obu = nullptr;
  const Obu* p_obu = nullptr;
  for (const Obu& obu : key_obus) {
    if (obu.type == static_cast<std::uint8_t>(ObuType::kClosedLoopKey)) key_obu = &obu;
  }
  for (const Obu& obu : p_obus) {
    if (obu.type == static_cast<std::uint8_t>(ObuType::kRegularTileGroup)) p_obu = &obu;
  }
  if (key_obu == nullptr || p_obu == nullptr) {
    std::fprintf(stderr, "missing key/P OBU in M2 units\n");
    return 1;
  }

  av2enc::video::VideoFrameHeader p_header;
  p_header.keyframe = false;
  p_header.order_hint = 1;
  p_header.ref_slot = 0;
  p_header.refresh_frame_flags = 1;
  // Compute the tile offset by writing the same M2 header into a scratch
  // writer and rounding up to a byte boundary.
  BitWriter header_scratch;
  av2enc::video::write_tile_group_header(header_scratch, config, p_header);
  const std::size_t video_header_bytes = (header_scratch.bit_size() + 7U) / 8U;
  if (p_obu->payload.size() < video_header_bytes) {
    std::fprintf(stderr, "P OBU payload is shorter than the M2 tile header\n");
    return 1;
  }
  const std::vector<std::uint8_t> p_tile(p_obu->payload.begin() +
                                             static_cast<std::ptrdiff_t>(video_header_bytes),
                                         p_obu->payload.end());

  // Build the global-motion P frame header.
  GmSyntaxContext gm_context;
  gm_context.frame_is_intra = false;
  gm_context.enable_global_motion = true;
  gm_context.use_global_motion = true;
  gm_context.num_total_refs = 1;
  gm_context.order_hint = 1;
  gm_context.ref_order_hint[0] = 0;
  gm_context.ref_frame_idx[0] = 0;
  gm_context.our_ref = -1;  // NumTotalRefs: no base model
  gm_context.params[0] = final_model;

  BitWriter p_writer;
  p_writer.put_bit(true);  // is_first_tile_group
  av2enc::video::write_frame_header(p_writer, config, p_header);
  const std::size_t gm_bits =
      av2enc::inter::gme::write_global_motion_params(p_writer, gm_context, &error);
  if (gm_bits == 0) {
    std::fprintf(stderr, "global_motion_params failed: %s\n", error.c_str());
    return 1;
  }
  p_writer.align_to_byte();
  std::vector<std::uint8_t> p_payload = p_writer.take_bytes();
  p_payload.insert(p_payload.end(), p_tile.begin(), p_tile.end());
  const std::vector<std::uint8_t> p_frame_obu =
      av2enc::basic::make_obu(ObuHeader{ObuType::kRegularTileGroup}, p_payload);

  // Read the GM syntax back; reconstruction must use exactly the decoded
  // parameters.
  GmSyntaxContext read_context = gm_context;
  BitWriter check_writer;
  check_writer.put_bit(true);
  av2enc::video::write_frame_header(check_writer, config, p_header);
  const std::size_t header_bits = check_writer.bit_size();
  GmBitReader full_reader(std::span<const std::uint8_t>(
      p_payload.data(), p_payload.size()));
  for (std::size_t i = 0; i < header_bits; ++i) {
    bool skip = false;
    if (!full_reader.read_bit(&skip)) {
      std::fprintf(stderr, "truncated P header while seeking GM syntax\n");
      return 1;
    }
  }
  std::array<GmParams, av2enc::inter::gme::kMaxRefsPerFrame> decoded_params{};
  bool decoded_use = false;
  if (!av2enc::inter::gme::read_global_motion_params(full_reader, read_context,
                                                     &decoded_params, &error, &decoded_use) ||
      !decoded_use) {
    std::fprintf(stderr, "global_motion_params read-back failed: %s\n", error.c_str());
    return 1;
  }
  if (!(decoded_params[0] == final_model)) {
    std::fprintf(stderr, "decoded global motion differs from the signalled model\n");
    return 1;
  }

  const OwnedFrame key_recon{sequence.recon[0].width, sequence.recon[0].height,
                            sequence.recon[0].y, sequence.recon[0].u,
                            sequence.recon[0].v};
  const OwnedFrame recon_p = reconstruct_p(key_recon, decoded_params[0]);
  if (recon_p.y.empty()) return 1;

  const std::vector<std::uint8_t> seq = make_sequence_obu(config, true);
  const std::vector<std::uint8_t> td = av2enc::video::make_temporal_delimiter_obu();
  auto make_unit = [&](const std::vector<std::uint8_t>& obu) {
    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), obu.begin(), obu.end());
    return unit;
  };

  std::vector<std::vector<std::uint8_t>> units;
  std::vector<OwnedFrame> recon_frames;
  {
    std::vector<std::uint8_t> unit0;
    unit0.insert(unit0.end(), td.begin(), td.end());
    unit0.insert(unit0.end(), seq.begin(), seq.end());
    unit0.insert(unit0.end(), key_obu->whole.begin(), key_obu->whole.end());
    units.push_back(std::move(unit0));
  }
  units.push_back(make_unit(p_frame_obu));
  recon_frames.push_back(recon_p);

  if (base_diff) {
    if (sequence.units.size() < 3) {
      std::fprintf(stderr, "base-diff gate needs a third M2 frame\n");
      return 1;
    }
    std::vector<Obu> p2_obus;
    if (!parse_obus(sequence.units[2], p2_obus, error)) {
      std::fprintf(stderr, "P2 OBU parse failed: %s\n", error.c_str());
      return 1;
    }
    const Obu* p2_obu = nullptr;
    for (const Obu& obu : p2_obus) {
      if (obu.type == static_cast<std::uint8_t>(ObuType::kRegularTileGroup)) {
        p2_obu = &obu;
      }
    }
    if (p2_obu == nullptr) {
      std::fprintf(stderr, "missing P2 OBU\n");
      return 1;
    }
    av2enc::video::VideoFrameHeader p2_header = p_header;
    p2_header.order_hint = 2;
    BitWriter p2_header_scratch;
    av2enc::video::write_tile_group_header(p2_header_scratch, config, p2_header);
    const std::size_t p2_header_bytes = (p2_header_scratch.bit_size() + 7U) / 8U;
    if (p2_obu->payload.size() < p2_header_bytes) {
      std::fprintf(stderr, "P2 OBU payload is shorter than the M2 header\n");
      return 1;
    }
    const std::vector<std::uint8_t> p2_tile(
        p2_obu->payload.begin() + static_cast<std::ptrdiff_t>(p2_header_bytes),
        p2_obu->payload.end());

    GmParams second_raw;
    second_raw.type = GmType::kRotZoom;
    second_raw.p = {2 * 65536, -1 * 65536, 70000, -900, 900, 70000};
    GmEstimate second_estimate =
        av2enc::inter::gme::finalize_gm_params(second_raw, gm_options);
    if (!second_estimate.diag.accepted || second_estimate.diag.identity) {
      std::fprintf(stderr, "base-diff second model rejected: %s\n",
                   second_estimate.diag.reason.c_str());
      return 1;
    }

    GmSyntaxContext p2_context;
    p2_context.frame_is_intra = false;
    p2_context.enable_global_motion = true;
    p2_context.use_global_motion = true;
    p2_context.num_total_refs = 1;
    p2_context.order_hint = 2;
    p2_context.ref_order_hint[0] = 1;  // P1 (slot 0)
    p2_context.ref_frame_idx[0] = 0;
    p2_context.our_ref = 0;           // base = P1
    p2_context.their_ref = 0;         // P1's reference = key
    p2_context.history.ref_num_total_refs[0] = 1;
    p2_context.history.saved_order_hints[0][0] = 0;
    for (int i = 0; i < 6; ++i) {
      p2_context.history.saved_gm_params[0][0][static_cast<std::size_t>(i)] =
          decoded_params[0].p[static_cast<std::size_t>(i)];
    }
    p2_context.params[0] = second_estimate.params;

    BitWriter p2_writer;
    p2_writer.put_bit(true);
    av2enc::video::write_frame_header(p2_writer, config, p2_header);
    const std::size_t p2_gm_bits =
        av2enc::inter::gme::write_global_motion_params(p2_writer, p2_context, &error);
    if (p2_gm_bits == 0) {
      std::fprintf(stderr, "P2 global_motion_params failed: %s\n", error.c_str());
      return 1;
    }
    p2_writer.align_to_byte();
    std::vector<std::uint8_t> p2_payload = p2_writer.take_bytes();
    p2_payload.insert(p2_payload.end(), p2_tile.begin(), p2_tile.end());
    const std::vector<std::uint8_t> p2_frame_obu =
        av2enc::basic::make_obu(ObuHeader{ObuType::kRegularTileGroup}, p2_payload);

    BitWriter p2_check_writer;
    p2_check_writer.put_bit(true);
    av2enc::video::write_frame_header(p2_check_writer, config, p2_header);
    GmBitReader p2_reader(std::span<const std::uint8_t>(
        p2_payload.data(), p2_payload.size()));
    for (std::size_t i = 0; i < p2_check_writer.bit_size(); ++i) {
      bool skip = false;
      if (!p2_reader.read_bit(&skip)) {
        std::fprintf(stderr, "truncated P2 header while seeking GM syntax\n");
        return 1;
      }
    }
    std::array<GmParams, av2enc::inter::gme::kMaxRefsPerFrame> decoded2{};
    bool decoded2_use = false;
    if (!av2enc::inter::gme::read_global_motion_params(p2_reader, p2_context, &decoded2,
                                                       &error, &decoded2_use) ||
        !decoded2_use) {
      std::fprintf(stderr, "P2 global_motion_params read-back failed: %s\n",
                   error.c_str());
      return 1;
    }
    if (!(decoded2[0] == second_estimate.params)) {
      std::fprintf(stderr, "decoded P2 global motion differs from the signalled model\n");
      return 1;
    }
    const OwnedFrame recon_p2 = reconstruct_p(recon_p, decoded2[0]);
    if (recon_p2.y.empty()) return 1;
    units.push_back(make_unit(p2_frame_obu));
    recon_frames.push_back(recon_p2);
    std::printf("gme_gate: base-diff P2 model=%s p=[%d %d %d %d %d %d] gm_bits=%zu\n",
                av2enc::inter::gme::gm_type_name(decoded2[0].type), decoded2[0].p[0],
                decoded2[0].p[1], decoded2[0].p[2], decoded2[0].p[3], decoded2[0].p[4],
                decoded2[0].p[5], p2_gm_bits);
  }

  if (!write_ivf(out_path, units, width, height, error)) {
    std::fprintf(stderr, "IVF write failed: %s\n", error.c_str());
    return 1;
  }

  std::printf("gme_gate: %dx%d key=%zuB p=%zuB gm_bits=%zu model=%s "
              "p=[%d %d %d %d %d %d] ivf=%s\n",
              width, height, key_obu->whole.size(), p_frame_obu.size(), gm_bits,
              av2enc::inter::gme::gm_type_name(decoded_params[0].type),
              decoded_params[0].p[0], decoded_params[0].p[1], decoded_params[0].p[2],
              decoded_params[0].p[3], decoded_params[0].p[4], decoded_params[0].p[5],
              out_path.c_str());

  if (!decode) return 0;
  const std::string decoder = find_dav2d();
  if (decoder.empty()) {
    std::fprintf(stderr, "gme_gate: SKIP (no dav2d; set DAV2D)\n");
    return 77;
  }
  if (!run_decoder(decoder, out_path, decoded_path, error)) {
    std::fprintf(stderr, "gme_gate: decoder failed: %s\n", error.c_str());
    return 1;
  }
  av2enc::Y4mReader reader(decoded_path);
  if (!reader.open(error)) {
    std::fprintf(stderr, "gme_gate: %s\n", error.c_str());
    return 1;
  }
  av2enc::Y4mFrame decoded;
  std::vector<av2enc::Y4mFrame> decoded_frames;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(decoded, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) {
      std::fprintf(stderr, "gme_gate: %s\n", error.c_str());
      return 1;
    }
    decoded_frames.push_back(decoded);
  }
  if (decoded_frames.size() != units.size()) {
    std::fprintf(stderr, "gme_gate: decoder produced %zu of %zu frames\n",
                 decoded_frames.size(), units.size());
    return 1;
  }
  std::uint64_t total_mismatches = 0;
  int max_diff = 0;
  for (std::size_t i = 0; i < recon_frames.size(); ++i) {
    std::uint64_t mismatches = 0;
    mismatches += compare_plane(decoded_frames[i + 1].y(), recon_frames[i].y, &max_diff);
    mismatches += compare_plane(decoded_frames[i + 1].u(), recon_frames[i].u, &max_diff);
    mismatches += compare_plane(decoded_frames[i + 1].v(), recon_frames[i].v, &max_diff);
    total_mismatches += mismatches;
    std::printf("gme_gate: decoded P%zu mismatches=%llu\n", i + 1,
                static_cast<unsigned long long>(mismatches));
  }
  std::printf("gme_gate: decoded frames mismatches=%llu max_diff=%d\n",
              static_cast<unsigned long long>(total_mismatches), max_diff);
  return total_mismatches == 0 ? 0 : 1;
}
