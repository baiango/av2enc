// Decoder gate for TIP frame-level signalling and build TIP (§5.18.2, §7.10).
//
// The stream is a 4-frame IPTT sequence:
//
//   frame 0  OBU_CLOSED_LOOP_KEY, order hint 0, refresh all slots  (M2 tile)
//   frame 1  OBU_REGULAR_TILE_GROUP, order hint 4, refresh slot 1 (M2 tile)
//            every block GLOBALMV + skip, so its decoded motion field is the
//            constant {ref 0, MV 0} field this gate synthesises
//   frame 2  OBU_REGULAR_TIP, order hint 1, refresh slot 2
//            past = frame 0, future = frame 1, tip_global_wtd_index = 4
//            (weight 20), TipGlobalMv = (8,8) -> a non-zero TIP frame whose
//            own §7.22 motion field becomes the past ref of frame 3
//   frame 3  OBU_REGULAR_TIP, order hint 2, refresh slot 3
//            past = frame 2, future = frame 1, weight index 3 (18),
//            TipGlobalMv = 0 -> the motion field is projected from frame 2's
//            stored field, so §7.10.2/.3/.4 all run with real MVs
//
// Both TIP frames are TIP_FRAME_AS_OUTPUT: no tile payload is coded, the
// decoder builds them with §7.10.6.  The gate reconstructs them with
// inter/tip_stream (motion field -> hole fill -> block average -> sample
// prediction) and requires zero differences from the pinned dav2d decode
// (`--strict 1 --threads 1`).
//
// The `--control` stream re-emits frames 0 and 1 with enable_tip = 0 and
// enable_ref_frame_mvs = 0.  It must be byte-identical to the M2 baseline
// temporal units, proving the TIP implementation adds no bits when disabled.
//
// Usage:
//   tip_gate [--width 640 --height 360] [--input clip.y4m]
//            [--out out.ivf] [--decoded out.y4m] [--control control.ivf]
//            [--width 64 --height 64]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "core/bitstream/basic_headers.h"
#include "codec/inter/temporal/motion_field.h"
#include "codec/inter/temporal/temporal_math.h"
#include "codec/inter/tip/tip_build.h"
#include "codec/inter/tip/tip_output.h"
#include "codec/inter/tip/tip_signalling.h"
#include "pipeline/video/video_frame.h"
#include "pipeline/video/video_headers.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

namespace tip = av2enc::inter::tip;
namespace temporal = av2enc::inter::temporal;
using av2enc::BitWriter;

namespace {

constexpr int kNumRefFrames = 8;
// The TIP stream uses a 5-bit order hint so a distant future reference (12)
// and a TIP frame at hint 1 are unambiguous.  The enable_tip=0 control uses
// the M2 baseline's 4 bits.
constexpr int kTipOrderHintBits = 5;
constexpr int kBaseOrderHintBits = 4;
constexpr int kBaseQIdx = 90;

// ---------------------------------------------------------------------------
// Syntax helpers (mirroring src/encoder/m2/video_headers.cpp).
// ---------------------------------------------------------------------------

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) ++leading_zeros;
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

void put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value) {
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

unsigned ceil_log2(std::uint32_t value) {
  if (value <= 1U) return 0U;
  unsigned bits = 0;
  std::uint32_t v = value - 1U;
  while (v != 0) {
    v >>= 1U;
    ++bits;
  }
  return bits;
}

std::uint32_t mi_cols_of(int width) {
  return 2U * ((static_cast<std::uint32_t>(width) + 7U) >> 3U);
}
std::uint32_t mi_rows_of(int height) {
  return 2U * ((static_cast<std::uint32_t>(height) + 7U) >> 3U);
}

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

void write_tile_info(BitWriter& writer, int width, int height) {
  const std::uint32_t sb_cols = (mi_cols_of(width) + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows_of(height) + 15U) >> 4U;
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1U) writer.put_bit(false);
  if (sb_rows > 1U) writer.put_bit(false);
}

struct SequenceSpec {
  int width = 64;
  int height = 64;
  int num_ref_frames = kNumRefFrames;
  int order_hint_bits = kTipOrderHintBits;
  bool enable_ref_frame_mvs = true;
  bool enable_tip = true;
  bool enable_tip_hole_fill = true;
};

// §5.4 sequence header, field order identical to video_headers.cpp so the M2 key
// and P tile payloads stay decodable.  With enable_tip = 0 and
// num_ref_frames = 2 this is byte-identical to make_sequence_header_obu().
std::vector<std::uint8_t> make_sequence_obu(const SequenceSpec& spec) {
  BitWriter writer;
  put_uvlc(writer, 0);              // seq_header_id
  writer.put_bits(0U, 5U);          // seq_profile_idc
  writer.put_bit(false);            // single_picture_header_flag
  writer.put_bits(7U, 5U);          // seq_level_idx = 5.1
  writer.put_bit(false);            // seq_tier
  put_uvlc(writer, 0);              // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);              // bit_depth_idc = 8-bit
  writer.put_bits(0U, 3U);          // seq_lcr_id
  writer.put_bit(false);            // still_picture
  writer.put_bits(0U, 2U);          // max_tlayer_id
  writer.put_bits(0U, 3U);          // max_mlayer_id
  writer.put_bit(true);             // monotonic_output_order_flag
  unsigned width_bits = bit_width(static_cast<std::uint32_t>(spec.width) - 1U);
  unsigned height_bits = bit_width(static_cast<std::uint32_t>(spec.height) - 1U);
  if (width_bits == 0) width_bits = 1;
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(spec.width) - 1U, width_bits);
  writer.put_bits(static_cast<std::uint64_t>(spec.height) - 1U, height_bits);
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
  for (int mode = 1; mode < 5; ++mode) {
    writer.put_bit(false);  // seq_enabled_motion_modes[mode]
  }
  writer.put_bit(false);  // enable_masked_compound
  writer.put_bit(spec.enable_ref_frame_mvs);
  if (spec.enable_ref_frame_mvs) writer.put_bit(false);  // reduced mode
  writer.put_bits(static_cast<std::uint64_t>(spec.order_hint_bits) - 1U, 4U);
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(true);   // disable_drl_reorder
  writer.put_bit(true);   // explicit_ref_frame_map
  writer.put_bit(true);   // explicit_num_ref_frames
  writer.put_bits(static_cast<std::uint64_t>(spec.num_ref_frames) - 1U, 4U);
  writer.put_bits(0U, 3U);  // long_term_frame_id_bits
  put_ns(writer, 5U, 4U);   // seq_max_drl_bits_minus_1
  writer.put_bit(false);    // allow_frame_max_drl_bits
  put_ns(writer, 3U, 0U);   // seq_max_bvp_drl_bits_minus_1
  writer.put_bit(false);    // allow_frame_max_bvp_drl_bits
  writer.put_bits(0U, 2U);  // num_same_ref_compound
  writer.put_bit(spec.enable_tip);
  if (spec.enable_tip) {
    writer.put_bit(false);  // disable_tip_output -> EnableTipOutput = 1
    writer.put_bit(spec.enable_tip_hole_fill);
  }
  writer.put_bit(false);    // enable_mv_traj
  writer.put_bit(false);    // enable_bawp
  writer.put_bit(false);    // enable_cwp
  writer.put_bit(false);    // enable_imp_msk_bld
  writer.put_bit(false);    // enable_df_sub_pu
  // enable_tip_explicit_qp is absent (enable_df_sub_pu == 0).
  writer.put_bits(0U, 2U);  // enable_opfl_refine = REFINE_NONE
  writer.put_bit(false);    // enable_refinemv
  // enable_tip_refinemv is absent (opfl NONE, refinemv 0).
  writer.put_bit(false);    // enable_bru
  writer.put_bit(false);    // enable_adaptive_mvd
  writer.put_bit(false);    // enable_mvd_sign_derive
  writer.put_bit(false);    // enable_flex_mvres
  writer.put_bit(false);    // enable_global_motion
  writer.put_bit(false);    // enable_short_refresh_frame_flags
  writer.put_bit(false);    // seq_choose_screen_content_tools
  writer.put_bit(false);    // seq_force_screen_content_tools
  writer.put_bit(false);    // enable_fsc
  writer.put_bit(false);    // enable_idtx_intra
  writer.put_bit(false);    // enable_intra_ist
  writer.put_bit(false);    // enable_inter_ist
  writer.put_bit(true);     // enable_chroma_dctonly
  writer.put_bit(false);    // enable_inter_ddt
  writer.put_bit(false);    // reduced_tx_part_set
  writer.put_bit(false);    // enable_cctx
  writer.put_bit(false);    // enable_tcq
  writer.put_bit(false);    // enable_parity_hiding
  writer.put_bit(false);    // enable_avg_cdf
  writer.put_bit(false);    // separate_uv_delta_q
  writer.put_bit(true);     // equal_ac_dc_q
  writer.put_bits(23U, 5U); // base_uv_ac_delta_q
  writer.put_bit(false);    // uv_ac_delta_q_enabled
  writer.put_bit(false);    // disable_loopfilters_across_tiles
  writer.put_bit(false);    // enable_cdef
  writer.put_bit(false);    // enable_gdf
  writer.put_bit(false);    // enable_restoration
  writer.put_bit(false);    // enable_ccso
  writer.put_bit(false);    // cdef_on_skip_txfm_always_on
  writer.put_bit(true);     // cdef_on_skip_txfm_disabled
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2
  writer.put_bit(false);    // seq_tile_info_present_flag
  writer.put_bit(false);    // film_grain_params_present
  writer.put_bit(false);    // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::basic::make_obu(
      av2enc::basic::ObuHeader{av2enc::basic::ObuType::kSequenceHeader},
      writer.take_bytes());
}

struct FrameSpec {
  bool key = false;
  bool tip_frame = false;
  int order_hint = 0;
  int num_ref_frames = kNumRefFrames;
  int order_hint_bits = kTipOrderHintBits;
  std::uint32_t refresh_frame_flags = 1;
  int num_total_refs = 0;
  std::uint8_t ref_frame_idx[2] = {0, 0};
  bool use_ref_frame_mvs = false;
  // §5.18.2: the bit exists only when the sequence enables
  // enable_ref_frame_mvs (otherwise use_ref_frame_mvs is inferred 0).
  bool write_use_ref_frame_mvs = true;
  tip::TipFrameSyntax tip_syntax;
};

// Frame header up to the end of frame_header_info() (no is_first_tile_group,
// no byte_alignment).  For a TIP frame the function stops after tip_sharp /
// tip_regular, which is where §5.18.2 reaches decode_frame_wrapup().
// Everything up to and including use_ref_frame_mvs.  The §5.18.2 TIP slice
// starts immediately after this point.
bool write_frame_header_prefix(BitWriter& writer, const FrameSpec& spec) {
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  if (!spec.key && !spec.tip_frame) writer.put_bit(true);  // frame_is_inter
  writer.put_bit(true);   // immediate_output_frame
  writer.put_bit(false);  // frame_size_override_flag
  writer.put_bits(static_cast<std::uint64_t>(spec.order_hint),
                  static_cast<unsigned>(spec.order_hint_bits));
  if (spec.key) {
    writer.put_bit(false);  // allow_intrabc
  } else {
    writer.put_bit(false);  // signal_primary_ref_frame
    if (!spec.tip_frame) {
      writer.put_bit(true);  // disable_cross_frame_cdf_init
    }
    writer.put_bits(spec.refresh_frame_flags,
                    static_cast<unsigned>(spec.num_ref_frames));
    writer.put_bit(true);  // frame_explicit_ref_frame_map
    writer.put_bits(static_cast<std::uint64_t>(spec.num_total_refs), 3U);
    for (int i = 0; i < spec.num_total_refs; ++i) {
      writer.put_bits(spec.ref_frame_idx[i],
                      ceil_log2(static_cast<std::uint32_t>(
                          spec.num_ref_frames)));
    }
    if (spec.write_use_ref_frame_mvs) {
      writer.put_bit(spec.use_ref_frame_mvs);
    }
  }
  return true;
}

bool write_frame_header_bits(BitWriter& writer, const FrameSpec& spec,
                             int width, int height) {
  if (!write_frame_header_prefix(writer, spec)) return false;
  if (!spec.key) {
    // SbSize == BLOCK_64X64, so tmvp_sample_step_minus_1 is absent.
    tip::TipFrameSyntax ts = spec.tip_syntax;
    ts.use_ref_frame_mvs = spec.use_ref_frame_mvs;
    ts.num_total_refs = spec.num_total_refs;
    ts.sb_64x64 = true;
    const temporal::TemporalStatus status =
        tip::write_tip_frame_syntax(writer, ts);
    if (status != temporal::TemporalStatus::kOk) return false;
    if (spec.tip_frame) return true;
    // Rest of §5.18.2 for a tile-coded inter frame, in M2 order.
    writer.put_bit(false);  // allow_intrabc
    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv -> 1/8 pel
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // interpolation_filter = EIGHTAP
    // frame_enabled_motion_modes[] all zero -> no bits.
  }
  writer.put_bit(true);   // disable_cdf_update
  write_tile_info(writer, width, height);
  writer.put_bits(static_cast<std::uint64_t>(kBaseQIdx), 8U);
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present
  writer.put_bit(false);  // apply_deblocking_filter[0]
  writer.put_bit(false);  // apply_deblocking_filter[1]
  writer.put_bit(false);  // tx_mode_select = 0
  if (!spec.key) {
    writer.put_bit(false);  // reference_select
    writer.put_bit(false);  // skip_mode_present
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set
  return true;
}

std::vector<std::uint8_t> make_frame_obu(const FrameSpec& spec, int width,
                                         int height,
                                         std::span<const std::uint8_t> tile) {
  BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  if (!write_frame_header_bits(writer, spec, width, height)) return {};
  writer.align_to_byte();
  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile.begin(), tile.end());
  const av2enc::basic::ObuType type =
      spec.key ? av2enc::basic::ObuType::kClosedLoopKey
               : (spec.tip_frame ? tip::tip_obu_type(tip::TipObuType::kRegularTip)
                                 : av2enc::basic::ObuType::kRegularTileGroup);
  return av2enc::basic::make_obu(av2enc::basic::ObuHeader{type}, payload);
}

// TIP frames have no tile payload: frame_header(1) runs decode_frame_wrapup()
// and the OBU ends with §5.2.3 trailing bits after the §5.2.1 extension flag.
std::vector<std::uint8_t> make_tip_obu(const FrameSpec& spec, int width,
                                       int height) {
  BitWriter writer;
  if (!write_frame_header_bits(writer, spec, width, height)) return {};
  return tip::make_tip_frame_obu(tip::TipObuType::kRegularTip, &writer);
}

// Bit offset of the TIP slice inside a TIP-frame OBU payload.
std::size_t tip_slice_bit_offset(const FrameSpec& spec) {
  BitWriter scratch;
  (void)write_frame_header_prefix(scratch, spec);
  return scratch.bit_size();
}

// Annex B.2: leb128 size, obu_header (1 byte, no extension), payload.
std::span<const std::uint8_t> obu_payload(const std::vector<std::uint8_t>& obu) {
  std::size_t offset = 0;
  std::uint32_t size = 0;
  unsigned shift = 0;
  while (offset < obu.size() && offset < 4) {
    const std::uint8_t byte = obu[offset++];
    size |= static_cast<std::uint32_t>(byte & 0x7f) << shift;
    shift += 7;
    if ((byte & 0x80) == 0) break;
  }
  (void)size;
  ++offset;  // obu_header
  if (offset > obu.size()) return {};
  return std::span<const std::uint8_t>(obu.data() + offset,
                                       obu.size() - offset);
}

std::vector<std::uint8_t> make_td() {
  return av2enc::basic::make_obu(av2enc::basic::ObuHeader{
                                  av2enc::basic::ObuType::kTemporalDelimiter},
                              std::span<const std::uint8_t>{});
}

// ---------------------------------------------------------------------------
// Frames and planes.
// ---------------------------------------------------------------------------

std::uint8_t synthetic_sample(int x, int y, int frame, int plane) {
  if (plane == 0) {
    const int sx = 9 + 5 * frame;
    const int sy = 7 + 3 * frame;
    const bool square = x >= sx && x < sx + 23 && y >= sy && y < sy + 19;
    if (square) {
      return static_cast<std::uint8_t>(30 + ((x * 5 + y * 3) & 0x7f));
    }
    return static_cast<std::uint8_t>(92 + ((x + 2 * y) >> 4));
  }
  const bool bright = (x + y + frame) & 1;
  if (plane == 1) return static_cast<std::uint8_t>(bright ? 120 : 136);
  return static_cast<std::uint8_t>(bright ? 140 : 124);
}

struct Picture {
  int width = 0;
  int height = 0;
  int chroma_width = 0;
  int chroma_height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;

  void init(int w, int h) {
    width = w;
    height = h;
    chroma_width = (w + 1) / 2;
    chroma_height = (h + 1) / 2;
    y.assign(static_cast<std::size_t>(w) * h, 0);
    u.assign(static_cast<std::size_t>(chroma_width) * chroma_height, 0);
    v.assign(static_cast<std::size_t>(chroma_width) * chroma_height, 0);
  }
  std::uint8_t* data(int plane) {
    return plane == 0 ? y.data() : (plane == 1 ? u.data() : v.data());
  }
  const std::uint8_t* data(int plane) const {
    return plane == 0 ? y.data() : (plane == 1 ? u.data() : v.data());
  }
  int plane_width(int plane) const {
    return plane == 0 ? width : chroma_width;
  }
  int plane_height(int plane) const {
    return plane == 0 ? height : chroma_height;
  }
  temporal::PlaneView view(int plane) const {
    temporal::PlaneView out;
    out.data = data(plane);
    out.stride = plane_width(plane);
    out.width = plane_width(plane);
    out.height = plane_height(plane);
    return out;
  }
  void write(int plane, int x0, int y0, int wide, int high, int seed) {
    for (int r = 0; r < high; ++r) {
      for (int c = 0; c < wide; ++c) {
        data(plane)[static_cast<std::size_t>(y0 + r) * plane_width(plane) +
                    x0 + c] =
            synthetic_sample(x0 + c, y0 + r, seed, plane);
      }
    }
  }
  void fill_from_y4m(const av2enc::Y4mFrame& frame) {
    width = static_cast<int>(frame.width());
    height = static_cast<int>(frame.height());
    chroma_width = static_cast<int>(frame.chroma_width());
    chroma_height = static_cast<int>(frame.chroma_height());
    y.assign(frame.y().begin(), frame.y().end());
    u.assign(frame.u().begin(), frame.u().end());
    v.assign(frame.v().begin(), frame.v().end());
  }
};

bool load_y4m(const std::string& path, int max_frames,
              std::vector<Picture>* out, std::string* error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame frame;
  while (static_cast<int>(out->size()) < max_frames) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, *error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    Picture picture;
    picture.fill_from_y4m(frame);
    out->push_back(std::move(picture));
  }
  return true;
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
                 const std::string& y4m, std::string* error) {
  const std::string command = "\"" + decoder + "\" -q -i \"" + ivf +
                              "\" -o \"" + y4m +
                              "\" --strict 1 --threads 1 >/dev/null 2>&1";
  const int status = std::system(command.c_str());
  if (status != 0) {
    *error = "dav2d exited with status " + std::to_string(status);
    return false;
  }
  return true;
}

std::uint64_t compare_picture(const Picture& decoded, const Picture& expected,
                              int* max_diff) {
  std::uint64_t mismatches = 0;
  for (int plane = 0; plane < 3; ++plane) {
    const int w = expected.plane_width(plane);
    const int h = expected.plane_height(plane);
    if (decoded.plane_width(plane) != w || decoded.plane_height(plane) != h) {
      *max_diff = 1000;
      return static_cast<std::uint64_t>(w) * h;
    }
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int a = decoded.data(plane)[static_cast<std::size_t>(y) * w + x];
        const int b = expected.data(plane)[static_cast<std::size_t>(y) * w + x];
        const int diff = std::abs(a - b);
        if (diff != 0) {
          if (mismatches < 8) {
            std::fprintf(stderr, "    diff plane=%d x=%d y=%d dec=%d enc=%d\n",
                         plane, x, y, a, b);
          }
          ++mismatches;
          *max_diff = std::max(*max_diff, diff);
        }
      }
    }
  }
  return mismatches;
}

// ---------------------------------------------------------------------------
// Motion-field fixtures.
// ---------------------------------------------------------------------------

struct MvFieldOwner {
  int w8 = 0;
  int h8 = 0;
  std::vector<std::int8_t> ref;
  std::vector<std::int16_t> mv;

  void init(int w, int h) {
    w8 = w;
    h8 = h;
    ref.assign(static_cast<std::size_t>(w) * h * 2, temporal::kRefNone);
    mv.assign(static_cast<std::size_t>(w) * h * 4, 0);
  }
  temporal::RefMvField view() {
    temporal::RefMvField out;
    out.w8 = w8;
    out.h8 = h8;
    out.ref = ref.data();
    out.mv = mv.data();
    return out;
  }
  temporal::SavedMvField saved() {
    temporal::SavedMvField out;
    out.w8 = w8;
    out.h8 = h8;
    out.ref = ref.data();
    out.mv = mv.data();
    return out;
  }
  // All-GLOBALMV skip P frame: every 8x8 stored as list 0/1 ref 0 with MV 0.
  void fill_zero() {
    for (int y = 0; y < h8; ++y) {
      for (int x = 0; x < w8; ++x) {
        const std::size_t pos =
            static_cast<std::size_t>(y) * w8 + static_cast<std::size_t>(x);
        ref[pos * 2 + 0] = 0;
        ref[pos * 2 + 1] = 0;
      }
    }
  }
};

struct FieldOwner {
  temporal::MotionField field{};
  temporal::MotionFieldScratch scratch{};
  std::vector<std::int32_t> mv;
  std::vector<std::uint8_t> valid;
  std::vector<std::int32_t> offset;
  std::vector<std::int32_t> avg_mv;
  std::vector<std::uint8_t> avg_valid;

  void init(int mi_cols, int mi_rows) {
    const std::size_t cells = static_cast<std::size_t>(mi_cols / 2) *
                              static_cast<std::size_t>(mi_rows / 2);
    mv.assign(cells * 2, 0);
    valid.assign(cells, 0);
    offset.assign(cells, 0);
    avg_mv.assign(cells * 2, 0);
    avg_valid.assign(cells, 0);
    field.w8 = mi_cols / 2;
    field.h8 = mi_rows / 2;
    field.mv = mv.data();
    field.valid = valid.data();
    field.offset = offset.data();
    scratch.avg_mv = avg_mv.data();
    scratch.avg_valid = avg_valid.data();
  }
};

struct PlanOwner {
  int mi_cols = 0;
  int mi_rows = 0;
  std::vector<temporal::TipBlock> blocks;
  std::vector<temporal::Mv> grid0;
  std::vector<temporal::Mv> grid1;
  int capacity = 0;

  void init(int cols, int rows) {
    mi_cols = cols;
    mi_rows = rows;
    capacity = ((cols + 3) / 4) * ((rows + 3) / 4);
    blocks.assign(static_cast<std::size_t>(capacity), temporal::TipBlock{});
    grid0.assign(static_cast<std::size_t>(cols) * rows, temporal::Mv{});
    grid1.assign(static_cast<std::size_t>(cols) * rows, temporal::Mv{});
  }
  temporal::TipPlan view() {
    temporal::TipPlan out;
    out.num_blocks = capacity;
    out.blocks = blocks.data();
    out.mv_grid = grid0.data();
    out.mv_grid1 = grid1.data();
    return out;
  }
};

struct TipFrameResult {
  Picture recon;
  MvFieldOwner stored;
  temporal::MotionFieldResult field_result{};
  int num_blocks = 0;
  int field_valid_estimation = 0;
  int field_valid_setup = 0;
  int hole_filled = 0;
  int field_invalid = 0;
  temporal::TipConfig used_tip{};
};

bool build_tip_frame(const Picture& past, const Picture& future,
                     const temporal::MotionFieldConfig& config,
                     const temporal::TipConfig& tip, TipFrameResult* result,
                     std::string* error) {
  const int width = past.width;
  const int height = past.height;
  result->recon.init(width, height);
  FieldOwner field;
  field.init(config.mi_cols, config.mi_rows);
  PlanOwner plan;
  plan.init(config.mi_cols, config.mi_rows);
  std::vector<std::int16_t> pred_scratch(2 * 16 * 16, 0);
  std::vector<std::uint8_t> block_scratch(16 * 16, 0);

  tip::TipBuildInput in;
  in.config = config;
  in.tip = tip;
  in.field = &field.field;
  in.scratch = &field.scratch;
  temporal::TipPlan plan_view = plan.view();
  in.plan = &plan_view;
  for (int plane = 0; plane < 3; ++plane) {
    in.ref[plane][0] = past.view(plane);
    in.ref[plane][1] = future.view(plane);
    in.dst[plane] = result->recon.data(plane);
    in.dst_stride[plane] = result->recon.plane_width(plane);
    in.dst_width[plane] = result->recon.plane_width(plane);
    in.dst_height[plane] = result->recon.plane_height(plane);
    in.sub_x[plane] = plane == 0 ? 0 : 1;
    in.sub_y[plane] = plane == 0 ? 0 : 1;
  }
  in.pred_scratch = pred_scratch.data();
  in.block_scratch = block_scratch.data();

  tip::TipBuildOutput out;
  const temporal::TemporalStatus status = tip::build_tip_frame(in, &out);
  if (status != temporal::TemporalStatus::kOk) {
    *error = "build_tip_frame failed";
    return false;
  }
  result->field_result = out.field_result;
  result->num_blocks = out.num_blocks;
  result->field_valid_estimation = out.field_valid_after_estimation;
  result->field_valid_setup = out.field_valid_after_setup;
  result->hole_filled = out.hole_filled;
  result->field_invalid = out.field_invalid;
  result->used_tip = tip;
  tip::fill_tip_closest_refs(config, out.field_result, &result->used_tip);

  result->stored.init(field.field.w8, field.field.h8);
  temporal::RefMvField stored = result->stored.view();
  tip::TipStoreInput store;
  store.mi_cols = config.mi_cols;
  store.mi_rows = config.mi_rows;
  store.order_hint = config.order_hint;
  store.closest_past = out.field_result.closest_past;
  store.closest_future = out.field_result.closest_future;
  store.closest_past_order_hint =
      result->used_tip.closest_past_order_hint;
  store.closest_future_order_hint =
      result->used_tip.closest_future_order_hint;
  store.weight16 =
      temporal::kTipWeightingFactor[tip.tip_global_wtd_index] == 16;
  if (tip::store_tip_frame_motion_field(plan_view, store, &stored) !=
      temporal::TemporalStatus::kOk) {
    *error = "store_tip_frame_motion_field failed";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Stream assembly.
// ---------------------------------------------------------------------------

struct StreamInput {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> key_tile;
  std::vector<std::uint8_t> p_tile;
};

bool build_stream(const StreamInput& input, bool enable_tip,
                  std::vector<std::vector<std::uint8_t>>* units,
                  std::string* error) {
  (void)error;
  SequenceSpec seq;
  seq.width = input.width;
  seq.height = input.height;
  seq.enable_tip = enable_tip;
  seq.enable_ref_frame_mvs = enable_tip;
  seq.num_ref_frames = enable_tip ? kNumRefFrames : 2;
  seq.order_hint_bits = enable_tip ? kTipOrderHintBits : kBaseOrderHintBits;
  const std::vector<std::uint8_t> seq_obu = make_sequence_obu(seq);
  const std::vector<std::uint8_t> td = make_td();

  FrameSpec key;
  key.key = true;
  key.order_hint = 0;
  key.num_ref_frames = seq.num_ref_frames;
  key.order_hint_bits = seq.order_hint_bits;
  const std::vector<std::uint8_t> key_obu =
      make_frame_obu(key, input.width, input.height, input.key_tile);

  // The control (enable_tip = 0) must reproduce the M2 baseline bitstream
  // exactly, including M2's own P order hint (1) and refresh of slot 0.  The
  // TIP stream places P at order hint 4 in slot 1 so it is a future reference.
  FrameSpec p;
  p.key = false;
  p.num_ref_frames = seq.num_ref_frames;
  p.order_hint_bits = seq.order_hint_bits;
  p.order_hint = enable_tip ? 12 : 1;
  p.refresh_frame_flags = enable_tip ? (1U << 1) : 1U;
  p.num_total_refs = 1;
  p.ref_frame_idx[0] = 0;
  p.use_ref_frame_mvs = enable_tip;
  p.write_use_ref_frame_mvs = enable_tip;
  const std::vector<std::uint8_t> p_obu =
      make_frame_obu(p, input.width, input.height, input.p_tile);

  units->clear();
  {
    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), seq_obu.begin(), seq_obu.end());
    unit.insert(unit.end(), key_obu.begin(), key_obu.end());
    units->push_back(std::move(unit));
  }
  {
    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), p_obu.begin(), p_obu.end());
    units->push_back(std::move(unit));
  }
  if (!enable_tip) return true;

  // TIP1: past = key (ref 0), future = P (ref 1), order hint 1.
  FrameSpec tip1;
  tip1.tip_frame = true;
  tip1.order_hint = 1;
  tip1.refresh_frame_flags = 1U << 2;
  tip1.num_total_refs = 2;
  tip1.ref_frame_idx[0] = 0;
  tip1.ref_frame_idx[1] = 1;
  tip1.use_ref_frame_mvs = true;
  tip1.write_use_ref_frame_mvs = true;
  tip1.order_hint_bits = kTipOrderHintBits;
  tip1.tip_syntax.use_ref_frame_mvs = true;
  tip1.tip_syntax.num_total_refs = 2;
  tip1.tip_syntax.enable_tip = true;
  tip1.tip_syntax.enable_tip_hole_fill = true;
  tip1.tip_syntax.enable_tip_output = true;
  tip1.tip_syntax.enable_opfl_refine = temporal::RefineType::kNone;
  tip1.tip_syntax.num_past_refs = 1;
  tip1.tip_syntax.num_future_refs = 1;
  tip1.tip_syntax.is_tip_frame = true;
  tip1.tip_syntax.tip_global_wtd_index = 2;  // weight 16 -> single list
  tip1.tip_syntax.tip_global_mv = temporal::Mv{15, 15};
  tip1.tip_syntax.allow_tip_hole_fill = true;
  tip1.tip_syntax.tip_sharp = true;
  const std::vector<std::uint8_t> tip1_obu =
      make_tip_obu(tip1, input.width, input.height);
  // Read the emitted slice back through the reader; the gate must not ship a
  // writer the reader cannot reproduce.
  {
    const std::span<const std::uint8_t> payload = obu_payload(tip1_obu);
    temporal::BitReader reader(payload.data(), payload.size());
    for (std::size_t i = 0; i < tip_slice_bit_offset(tip1); ++i) {
      bool bit = false;
      if (!reader.get_bit(&bit)) {
        *error = "truncated TIP1 header";
        return false;
      }
    }
    tip::TipFrameSyntax read;
    read.use_ref_frame_mvs = true;
    read.num_total_refs = 2;
    read.sb_64x64 = true;
    read.enable_tip = true;
    read.enable_tip_output = true;
    read.enable_tip_hole_fill = true;
    read.num_past_refs = 1;
    read.num_future_refs = 1;
    read.is_tip_frame = true;
    if (tip::read_tip_frame_syntax(reader, &read) !=
            temporal::TemporalStatus::kOk ||
        read.mode != temporal::TipFrameMode::kAsOutput ||
        read.tip_global_wtd_index != 2 ||
        !(read.tip_global_mv == temporal::Mv(15, 15)) ||
        !read.allow_tip_hole_fill ||
        read.interp_filter != temporal::InterpFilter::kEighttapSharp) {
      *error = "TIP1 syntax read-back mismatch";
      return false;
    }
  }
  units->push_back([&] {
    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), tip1_obu.begin(), tip1_obu.end());
    return unit;
  }());

  // TIP2: past = TIP1 (ref 0 = slot 2), future = P (ref 1), hint 2.
  FrameSpec tip2 = tip1;
  tip2.order_hint_bits = kTipOrderHintBits;
  tip2.order_hint = 9;
  tip2.refresh_frame_flags = 1U << 3;
  tip2.ref_frame_idx[0] = 2;
  tip2.ref_frame_idx[1] = 1;
  tip2.tip_syntax.tip_global_wtd_index = 3;  // weight 18
  tip2.tip_syntax.tip_global_mv = temporal::Mv{0, 0};
  const std::vector<std::uint8_t> tip2_obu =
      make_tip_obu(tip2, input.width, input.height);
  {
    const std::span<const std::uint8_t> payload = obu_payload(tip2_obu);
    temporal::BitReader reader(payload.data(), payload.size());
    for (std::size_t i = 0; i < tip_slice_bit_offset(tip2); ++i) {
      bool bit = false;
      if (!reader.get_bit(&bit)) {
        *error = "truncated TIP2 header";
        return false;
      }
    }
    tip::TipFrameSyntax read;
    read.use_ref_frame_mvs = true;
    read.num_total_refs = 2;
    read.sb_64x64 = true;
    read.enable_tip = true;
    read.enable_tip_output = true;
    read.enable_tip_hole_fill = true;
    read.num_past_refs = 1;
    read.num_future_refs = 1;
    read.is_tip_frame = true;
    if (tip::read_tip_frame_syntax(reader, &read) !=
            temporal::TemporalStatus::kOk ||
        read.mode != temporal::TipFrameMode::kAsOutput ||
        read.tip_global_wtd_index != 3 ||
        !(read.tip_global_mv == temporal::Mv(0, 0))) {
      *error = "TIP2 syntax read-back mismatch";
      return false;
    }
  }
  units->push_back([&] {
    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), tip2_obu.begin(), tip2_obu.end());
    return unit;
  }());
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  int width = 64;
  int height = 64;
  std::string input_path;
  std::string out_path = "/tmp/tip_gate.ivf";
  std::string decoded_path = "/tmp/tip_gate_decoded.y4m";
  std::string control_path;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--width" && i + 1 < argc) {
      width = std::atoi(argv[++i]);
    } else if (arg == "--height" && i + 1 < argc) {
      height = std::atoi(argv[++i]);
    } else if (arg == "--input" && i + 1 < argc) {
      input_path = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--decoded" && i + 1 < argc) {
      decoded_path = argv[++i];
    } else if (arg == "--control" && i + 1 < argc) {
      control_path = argv[++i];
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }
  if (width < 64 || height < 64 || width % 8 != 0 || height % 8 != 0) {
    std::fprintf(stderr, "width/height must be 8-multiples >= 64\n");
    return 2;
  }

  std::string error;
  std::vector<Picture> source;
  if (!input_path.empty()) {
    if (!load_y4m(input_path, 2, &source, &error)) {
      std::fprintf(stderr, "input load failed: %s\n", error.c_str());
      return 1;
    }
    for (Picture& picture : source) {
      if (picture.width < width || picture.height < height) {
        std::fprintf(stderr, "input frame %dx%d is smaller than %dx%d\n",
                     picture.width, picture.height, width, height);
        return 1;
      }
      Picture cropped;
      cropped.init(width, height);
      for (int plane = 0; plane < 3; ++plane) {
        const int pw = std::min(picture.plane_width(plane),
                                cropped.plane_width(plane));
        const int ph = std::min(picture.plane_height(plane),
                                cropped.plane_height(plane));
        for (int y = 0; y < ph; ++y) {
          std::memcpy(cropped.data(plane) +
                          static_cast<std::size_t>(y) *
                              cropped.plane_width(plane),
                      picture.data(plane) +
                          static_cast<std::size_t>(y) *
                              picture.plane_width(plane),
                      static_cast<std::size_t>(pw));
        }
      }
      picture = std::move(cropped);
    }
    if (source.size() < 2) {
      std::fprintf(stderr, "input must provide at least 2 frames\n");
      return 1;
    }
  } else {
    for (int frame = 0; frame < 2; ++frame) {
      Picture picture;
      picture.init(width, height);
      for (int plane = 0; plane < 3; ++plane) {
        picture.write(plane, 0, 0, picture.plane_width(plane),
                      picture.plane_height(plane), frame);
      }
      source.push_back(std::move(picture));
    }
  }

  // M2 baseline: key + all-GLOBALMV skip P.  Its tile payloads are reused
  // verbatim; the gate only replaces the wrappers (sequence + frame headers).
  std::vector<av2enc::video::VideoInputFrame> video_inputs;
  for (const Picture& picture : source) {
    av2enc::video::VideoInputFrame in;
    in.width = static_cast<std::uint32_t>(picture.width);
    in.height = static_cast<std::uint32_t>(picture.height);
    in.y = std::span<const std::uint8_t>(picture.y.data(), picture.y.size());
    in.u = std::span<const std::uint8_t>(picture.u.data(), picture.u.size());
    in.v = std::span<const std::uint8_t>(picture.v.data(), picture.v.size());
    video_inputs.push_back(in);
  }
  av2enc::video::VideoOptions options;
  options.base_q_idx = kBaseQIdx;
  // The future reference is a frame with FrameType INTER_FRAME but only intra
  // blocks.  Its §7.22 saved field is therefore empty (MfRefFrames = NONE
  // everywhere), which the gate can synthesise exactly, and it decodes at
  // 640x360 without the 64x64-block edge problem that force_global_mv hits on
  // a 40-line bottom superblock row.
  options.force_global_mv = true;
  options.force_single_block = false;
  options.enable_intra_fallback = true;
  options.enable_newmv = false;
  options.enable_nearmv = false;
  options.force_intra = true;
  options.min_block_size = 16;
  options.max_partition_depth = 4;
  av2enc::video::VideoSequenceResult sequence;
  if (!av2enc::video::encode_video_sequence(video_inputs, options, sequence, error)) {
    std::fprintf(stderr, "M2 encode failed: %s\n", error.c_str());
    return 1;
  }
  if (sequence.units.size() < 2) {
    std::fprintf(stderr, "M2 produced fewer than two temporal units\n");
    return 1;
  }
  // The gate assumes the future reference is an all-intra INTER_FRAME: its
  // stored motion field is then empty, i.e. every 8x8 keeps MfRefFrames=NONE.
  if (sequence.inter_blocks[1] != 0 || sequence.intra_blocks[1] == 0) {
    std::fprintf(stderr, "P frame is not all-intra (intra=%u inter=%u)\n",
                 sequence.intra_blocks[1], sequence.inter_blocks[1]);
    return 1;
  }

  const std::vector<std::uint8_t>& key_unit = sequence.units[0];
  const std::vector<std::uint8_t>& p_unit = sequence.units[1];
  std::vector<std::uint8_t> key_tile(
      key_unit.end() - sequence.tile_bytes[0], key_unit.end());
  std::vector<std::uint8_t> p_tile(p_unit.end() - sequence.tile_bytes[1],
                                   p_unit.end());
  StreamInput input;
  input.width = width;
  input.height = height;
  input.key_tile = key_tile;
  input.p_tile = p_tile;

  // Encoder-side pictures.  Key and P share the decoded samples.
  Picture key_picture;
  key_picture.init(width, height);
  key_picture.y = sequence.recon[0].y;
  key_picture.u = sequence.recon[0].u;
  key_picture.v = sequence.recon[0].v;
  Picture p_picture;
  p_picture.init(width, height);
  p_picture.y = sequence.recon[1].y;
  p_picture.u = sequence.recon[1].u;
  p_picture.v = sequence.recon[1].v;

  const int mi_cols = static_cast<int>(mi_cols_of(width));
  const int mi_rows = static_cast<int>(mi_rows_of(height));
  const int w8 = mi_cols / 2;
  const int h8 = mi_rows / 2;

  // Empty §7.22 field for the all-intra future reference.
  MvFieldOwner zero_field;
  zero_field.init(w8, h8);

  temporal::MotionFieldConfig tip1_config;
  tip1_config.mi_cols = mi_cols;
  tip1_config.mi_rows = mi_rows;
  tip1_config.proj_step = 1;
  tip1_config.sb_64x64 = true;
  tip1_config.enable_tip = true;
  tip1_config.order_hint = 1;
  tip1_config.num_total_refs = 2;
  tip1_config.num_past_refs = 1;
  tip1_config.num_future_refs = 1;
  tip1_config.frame_width = width;
  tip1_config.frame_height = height;
  tip1_config.ref_order_hint[0] = 0;
  tip1_config.ref_order_hint[1] = 12;
  tip1_config.refs[0].slot = 0;
  tip1_config.refs[0].type = temporal::FrameType::kKey;
  tip1_config.refs[0].width = width;
  tip1_config.refs[0].height = height;
  tip1_config.refs[0].order_hint = 0;
  tip1_config.refs[0].num_saved_refs = 0;
  tip1_config.refs[1].slot = 1;
  tip1_config.refs[1].type = temporal::FrameType::kInter;
  tip1_config.refs[1].width = width;
  tip1_config.refs[1].height = height;
  tip1_config.refs[1].order_hint = 12;
  tip1_config.refs[1].num_saved_refs = 1;
  tip1_config.refs[1].saved_order_hint[0] = 0;
  tip1_config.saved[0] = temporal::SavedMvField{};
  tip1_config.saved[1] = zero_field.saved();

  temporal::TipConfig tip1_tip;
  tip1_tip.mode = temporal::TipFrameMode::kAsOutput;
  tip1_tip.tip_global_wtd_index = 2;
  tip1_tip.tip_global_mv = temporal::Mv{15, 15};
  tip1_tip.interp_filter = temporal::InterpFilter::kEighttapSharp;
  tip1_tip.enable_tip_refinemv = false;
  tip1_tip.opfl_refine_type = temporal::RefineType::kNone;
  tip1_tip.allow_tip_hole_fill = true;
  tip1_tip.num_past_refs = 1;
  tip1_tip.num_future_refs = 1;
  tip1_tip.order_hint = 1;

  TipFrameResult tip1;
  if (!build_tip_frame(key_picture, p_picture, tip1_config, tip1_tip, &tip1,
                       &error)) {
    std::fprintf(stderr, "TIP1 build failed: %s\n", error.c_str());
    return 1;
  }

  temporal::MotionFieldConfig tip2_config = tip1_config;
  tip2_config.order_hint = 9;
  tip2_config.ref_order_hint[0] = 1;
  tip2_config.ref_order_hint[1] = 12;
  tip2_config.refs[0].slot = 2;
  tip2_config.refs[0].type = temporal::FrameType::kInter;
  tip2_config.refs[0].order_hint = 1;
  tip2_config.refs[0].num_saved_refs = 2;
  tip2_config.refs[0].saved_order_hint[0] = 0;
  tip2_config.refs[0].saved_order_hint[1] = 12;
  tip2_config.refs[1].slot = 1;
  tip2_config.refs[1].type = temporal::FrameType::kInter;
  tip2_config.refs[1].order_hint = 12;
  tip2_config.refs[1].num_saved_refs = 1;
  tip2_config.refs[1].saved_order_hint[0] = 0;
  tip2_config.saved[0] = tip1.stored.saved();
  tip2_config.saved[1] = zero_field.saved();

  temporal::TipConfig tip2_tip = tip1_tip;
  tip2_tip.tip_global_wtd_index = 3;
  tip2_tip.tip_global_mv = temporal::Mv{0, 0};
  tip2_tip.order_hint = 9;

  TipFrameResult tip2;
  if (!build_tip_frame(tip1.recon, p_picture, tip2_config, tip2_tip, &tip2,
                       &error)) {
    std::fprintf(stderr, "TIP2 build failed: %s\n", error.c_str());
    return 1;
  }
  if (tip2.field_result.closest_past < 0 || tip2.field_result.closest_future < 0) {
    std::fprintf(stderr, "TIP2 has no closest past/future\n");
    return 1;
  }
  if (tip2.field_valid_estimation == 0) {
    std::fprintf(stderr, "TIP2 motion field is empty; TIP field not exercised\n");
    return 1;
  }
  if (tip2.hole_filled == 0) {
    std::fprintf(stderr, "TIP2 has no holes; §7.10.3/.4 not exercised\n");
    return 1;
  }
  if (tip1.recon.y == key_picture.y && tip1.recon.u == key_picture.u &&
      tip1.recon.v == key_picture.v) {
    std::fprintf(stderr, "TIP1 is a copy of the key; build TIP did not run\n");
    return 1;
  }
  // §7.10.1 setup-TIP conformance for both TIP frames.
  {
    tip::TipFrameSyntax frame;
    frame.mode = temporal::TipFrameMode::kAsOutput;
    frame.use_ref_frame_mvs = true;
    frame.num_past_refs = 1;
    const tip::TipSetupConditions c1 = tip::make_tip_setup_conditions(
        false, frame, tip1.field_result.closest_past,
        tip1.field_result.closest_future, /*past_type=*/false,
        /*future_type=*/true, true, true);
    const tip::TipSetupConditions c2 = tip::make_tip_setup_conditions(
        false, frame, tip2.field_result.closest_past,
        tip2.field_result.closest_future, /*past_type=*/true,
        /*future_type=*/true, true, true);
    const char* reason = nullptr;
    if (tip::check_tip_setup_conditions(c1, &reason) !=
            temporal::TemporalStatus::kOk ||
        tip::check_tip_setup_conditions(c2, &reason) !=
            temporal::TemporalStatus::kOk) {
      std::fprintf(stderr, "setup-TIP conditions rejected: %s\n",
                   reason != nullptr ? reason : "?");
      return 1;
    }
  }
  // Output rules (§5.4.6 EnableTipOutput, §7.21.1) for every emitted frame.
  {
    tip::TipOutputConfig cfg;
    cfg.enable_tip = true;
    cfg.disable_tip_output = false;
    cfg.is_tip_frame = true;
    cfg.mode = temporal::TipFrameMode::kAsOutput;
    cfg.immediate_output = true;
    cfg.refreshes_slot = true;
    cfg.has_closest_past = true;
    cfg.has_closest_future = true;
    tip::TipOutputPlan plan;
    if (tip::plan_tip_output(cfg, &plan) != temporal::TemporalStatus::kOk ||
        !plan.tip_obu_allowed || !plan.output_now) {
      std::fprintf(stderr, "TIP output plan rejected: %s\n", plan.reason);
      return 1;
    }
    cfg.immediate_output = false;  // a reference-only TIP frame
    if (tip::plan_tip_output(cfg, &plan) != temporal::TemporalStatus::kOk ||
        !plan.non_output_tip) {
      std::fprintf(stderr, "non-output TIP plan rejected: %s\n", plan.reason);
      return 1;
    }
  }

  std::vector<std::vector<std::uint8_t>> units;
  if (!build_stream(input, true, &units, &error)) {
    std::fprintf(stderr, "stream assembly failed: %s\n", error.c_str());
    return 1;
  }
  if (units.size() != 4) {
    std::fprintf(stderr, "expected 4 temporal units, got %zu\n", units.size());
    return 1;
  }
  if (!write_ivf(out_path, units, width, height, &error)) {
    std::fprintf(stderr, "IVF write failed: %s\n", error.c_str());
    return 1;
  }

  // enable_tip = 0 control: the same two M2 tiles wrapped by our writers.
  std::vector<std::vector<std::uint8_t>> control;
  if (!build_stream(input, false, &control, &error)) {
    std::fprintf(stderr, "control assembly failed: %s\n", error.c_str());
    return 1;
  }
  const bool control_identical =
      control.size() == 2 && control[0] == sequence.units[0] &&
      control[1] == sequence.units[1];
  if (!control_identical) {
    std::fprintf(stderr, "control is NOT byte-identical to the M2 baseline "
                         "(sizes %zu/%zu vs %zu/%zu)\n",
                 control.size() > 0 ? control[0].size() : 0,
                 control.size() > 1 ? control[1].size() : 0,
                 sequence.units[0].size(), sequence.units[1].size());
    for (std::size_t u = 0; u < control.size() && u < 2; ++u) {
      const std::vector<std::uint8_t>& a = control[u];
      const std::vector<std::uint8_t>& b = sequence.units[u];
      for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        if (a[i] != b[i]) {
          std::fprintf(stderr, "  unit %zu first diff at byte %zu: %02x vs "
                               "%02x\n", u, i, a[i], b[i]);
          std::fprintf(stderr, "  ours:");
          for (std::size_t k = 0; k < std::min<std::size_t>(12, a.size()); ++k)
            std::fprintf(stderr, " %02x", a[k]);
          std::fprintf(stderr, "\n  base:");
          for (std::size_t k = 0; k < std::min<std::size_t>(12, b.size()); ++k)
            std::fprintf(stderr, " %02x", b[k]);
          std::fprintf(stderr, "\n");
          break;
        }
      }
    }
    return 1;
  }
  {
    // The control has no TIP OBU at all: enable_tip = 0.
    tip::TipOutputConfig cfg;
    cfg.enable_tip = false;
    cfg.is_tip_frame = false;
    cfg.mode = temporal::TipFrameMode::kDisabled;
    cfg.immediate_output = true;
    cfg.refreshes_slot = true;
    tip::TipOutputPlan plan;
    if (tip::plan_tip_output(cfg, &plan) != temporal::TemporalStatus::kOk ||
        plan.tip_obu_allowed) {
      std::fprintf(stderr, "control output plan rejected: %s\n", plan.reason);
      return 1;
    }
  }
  if (!control_path.empty()) {
    if (!write_ivf(control_path, control, width, height, &error)) {
      std::fprintf(stderr, "control IVF write failed: %s\n", error.c_str());
      return 1;
    }
  }

  const std::string decoder = find_dav2d();
  if (decoder.empty()) {
    std::fprintf(stderr, "tip_gate: SKIP (no dav2d; set DAV2D)\n");
    return 77;
  }
  if (!run_decoder(decoder, out_path, decoded_path, &error)) {
    std::fprintf(stderr, "decode failed: %s\n", error.c_str());
    return 1;
  }
  std::vector<Picture> decoded;
  if (!load_y4m(decoded_path, 8, &decoded, &error)) {
    std::fprintf(stderr, "decoded Y4M load failed: %s\n", error.c_str());
    return 1;
  }
  if (decoded.size() < 4) {
    std::fprintf(stderr, "decoder produced %zu frames, expected 4\n",
                 decoded.size());
    return 1;
  }

  int max_key = 0;
  int max_p = 0;
  int max_tip1 = 0;
  int max_tip2 = 0;
  Picture key_recon;
  key_recon.init(width, height);
  key_recon.y = sequence.recon[0].y;
  key_recon.u = sequence.recon[0].u;
  key_recon.v = sequence.recon[0].v;
  Picture p_recon;
  p_recon.init(width, height);
  p_recon.y = sequence.recon[1].y;
  p_recon.u = sequence.recon[1].u;
  p_recon.v = sequence.recon[1].v;
  const std::uint64_t diff_key =
      compare_picture(decoded[0], key_recon, &max_key);
  const std::uint64_t diff_p = compare_picture(decoded[1], p_recon, &max_p);
  const std::uint64_t diff_tip1 =
      compare_picture(decoded[2], tip1.recon, &max_tip1);
  const std::uint64_t diff_tip2 =
      compare_picture(decoded[3], tip2.recon, &max_tip2);

  std::printf(
      "tip_gate %dx%d: units=%zu bytes=%zu control_bytes=%zu "
      "control_identical=1\n",
      width, height, units.size(),
      [&] {
        std::size_t total = 0;
        for (const auto& unit : units) total += unit.size();
        return total;
      }(),
      [&] {
        std::size_t total = 0;
        for (const auto& unit : control) total += unit.size();
        return total;
      }());
  std::printf(
      "  tip1: blocks=%d field_valid=%d/%d holes_filled=%d stored=%dx%d\n",
      tip1.num_blocks, tip1.field_valid_setup,
      w8 * h8, tip1.hole_filled, tip1.stored.w8, tip1.stored.h8);
  std::printf(
      "  tip2: blocks=%d field_valid(est)=%d field_valid(setup)=%d holes=%d "
      "past=%d future=%d\n",
      tip2.num_blocks, tip2.field_valid_estimation,
      tip2.field_valid_setup, tip2.hole_filled,
      tip2.field_result.closest_past, tip2.field_result.closest_future);
  std::printf(
      "  decode: key_diff=%llu(max %d) p_diff=%llu(max %d) "
      "tip1_diff=%llu(max %d) tip2_diff=%llu(max %d)\n",
      static_cast<unsigned long long>(diff_key), max_key,
      static_cast<unsigned long long>(diff_p), max_p,
      static_cast<unsigned long long>(diff_tip1), max_tip1,
      static_cast<unsigned long long>(diff_tip2), max_tip2);
  if (diff_key != 0 || diff_p != 0 || diff_tip1 != 0 || diff_tip2 != 0) {
    std::fprintf(stderr, "tip_gate: reconstruction mismatch\n");
    return 1;
  }
  std::printf("tip_gate: PASS (recon exact, control byte-identical)\n");
  return 0;
}
