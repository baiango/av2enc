// GDF decoder gate.
//
// Builds a real M1 still-picture stream with the shipped CPU M1 encoder,
// rewrites the sequence and frame headers to signal §5.4.10 enable_gdf /
// gdf_unit_matches_sb_size and §5.18.7.9 gdf_params, applies §7.20.5 to the
// encoder reconstruction, and decodes with the pinned dav2d
// (`--strict 1 --threads 1`).  Every Y/U/V sample must match.
//
// Three phases:
//   A. Control: enable_gdf = 0.  The rewritten stream must be byte-identical to
//      the M1 encoder's own output (proves the header writers add nothing and
//      that the GDF code is off the byte path when disabled).
//   B. GDF with the encoder decision (gdf_pic_qc_idx/gdf_pic_scale_idx
//      searched, per-block flags evaluated).  The gate forces gdf_per_block = 0
//      for the coded stream because inserting use_gdf symbols into an existing
//      arithmetic-coded tile payload would require the tile writer; per-block
//      syntax is covered by the unit goldens in gdf_stream_test.cpp.
//   C. Forced non-trivial parameters (gdf_pic_qc_idx = 2,
//      gdf_pic_scale_idx = 1).  This checks the parameter path independently of
//      the search.
//
// Usage:
//   gdf_stream_gate [--input clip.y4m] [--qp Q] [--work DIR]
//                   [--dav2d PATH] [--matches 0|1] [--skip-forced]
//
// Exit codes: 0 pass, 1 fail/mismatch, 77 dav2d unavailable.
#include "core/bitstream/bitwriter.h"
#include "core/bitstream/basic_headers.h"
#include "codec/filters/gdf/gdf_decision.h"
#include "codec/filters/gdf/gdf_syntax.h"
#include "pipeline/still/still_headers.h"
#include "pipeline/still/still_encoder.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;
namespace gdf = av2enc::filters::gdf;

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
  if (condition) {
    std::printf("  ok   %s\n", what.c_str());
  } else {
    std::printf("  FAIL %s\n", what.c_str());
    ++g_failures;
  }
}

void PutUvlc(av2enc::BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) ++leading_zeros;
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  writer.put_bits(
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL),
      leading_zeros);
}

void PutNs(av2enc::BitWriter& writer, unsigned n, std::uint32_t value) {
  if (n <= 1U) return;
  unsigned width = 0;
  while ((1U << width) < n) ++width;
  const std::uint32_t m = (1U << width) - n;
  if (value < m) {
    writer.put_bits(value, width - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, width - 1U);
    writer.put_bit((x & 1U) != 0U);
  }
}

unsigned BitWidth(std::uint32_t value) {
  unsigned bits = 0;
  while (value != 0U) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

// §5.4.1 sequence_header_obu() as written by M1, with the §5.4.10 GDF rows.
std::vector<std::uint8_t> MakeSequenceHeader(
    const av2enc::still::StillConfig& config, const gdf::SequenceGdfFlags& flags,
    std::string* error) {
  av2enc::BitWriter writer;
  PutUvlc(writer, 0);                          // seq_header_id
  writer.put_bits(0U, 5U);                     // seq_profile_idc
  writer.put_bit(true);                        // single_picture_header_flag
  writer.put_bits(config.seq_level_idx, 5U);
  PutUvlc(writer, 0);                          // chroma_format_idc 4:2:0
  PutUvlc(writer, 1);                          // bit_depth_idc 8-bit
  unsigned width_bits =
      BitWidth(static_cast<std::uint32_t>(config.max_frame_width) - 1U);
  if (width_bits == 0) width_bits = 1;
  unsigned height_bits =
      BitWidth(static_cast<std::uint32_t>(config.max_frame_height) - 1U);
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(config.max_frame_width) - 1U,
                  width_bits);
  writer.put_bits(static_cast<std::uint64_t>(config.max_frame_height) - 1U,
                  height_bits);
  writer.put_bit(false);  // seq_cropping_window_present_flag
  writer.put_bit(false);  // use_256x256_superblock
  writer.put_bit(false);  // use_128x128_superblock
  writer.put_bit(false);  // enable_sdp
  writer.put_bit(false);  // enable_ext_partitions
  writer.put_bit(false);  // reduce_pb_aspect_ratio
  writer.put_bit(false);  // enable_ext_seg
  writer.put_bit(false);  // seq_seg_info_present_flag
  writer.put_bit(false);  // enable_dip
  writer.put_bit(config.enable_intra_edge_filter);
  writer.put_bit(false);  // enable_mrls
  writer.put_bit(config.enable_cfl_intra);
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);    // enable_mhccp
  writer.put_bit(false);    // enable_ibp
  writer.put_bit(false);    // enable_refmvbank
  writer.put_bit(true);     // disable_drl_reorder
  PutNs(writer, 3U, 0U);    // seq_max_bvp_drl_bits_minus_1
  writer.put_bit(false);    // allow_frame_max_bvp_drl_bits
  writer.put_bit(false);    // enable_bawp
  writer.put_bit(false);    // enable_fsc
  writer.put_bit(config.enable_idtx_intra);
  writer.put_bit(false);    // enable_intra_ist
  writer.put_bit(false);    // enable_inter_ist
  writer.put_bit(config.enable_chroma_dctonly);
  writer.put_bit(false);    // reduced_tx_part_set
  writer.put_bit(false);    // enable_cctx
  writer.put_bit(false);    // enable_tcq
  writer.put_bit(false);    // enable_parity_hiding
  writer.put_bit(false);    // separate_uv_delta_q
  writer.put_bit(true);     // equal_ac_dc_q
  writer.put_bits(23U, 5U); // base_uv_ac_delta_q
  writer.put_bit(false);    // uv_ac_delta_q_enabled
  // §5.4.10 sequence_filter_config().
  writer.put_bit(false);    // disable_loopfilters_across_tiles
  writer.put_bit(config.cdef.enable);
  if (gdf::WriteSequenceGdfFlags(writer, flags, 64, error) == 0 &&
      flags.enable_gdf) {
    return {};
  }
  writer.put_bit(false);    // enable_restoration
  writer.put_bit(config.ccso.enable);
  if (config.ccso.enable) writer.put_bit(true);  // ccso_unit_matches_sb_size
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2
  writer.put_bit(false);    // seq_tile_info_present_flag
  writer.put_bit(false);    // film_grain_params_present
  writer.put_bit(false);    // obu_extension_flag
  writer.put_bit(true);     // trailing_bits
  writer.align_to_byte();
  return av2enc::basic::make_obu({av2enc::basic::ObuType::kSequenceHeader},
                              writer.bytes());
}

// frame_header_info() for one M1 intra picture, with §5.18.7.9 gdf_params()
// inserted after deblocking_filter_params() and before cdef_params().
std::vector<std::uint8_t> MakeFrameObu(
    const av2enc::still::StillConfig& config, const gdf::FrameGdfContext& context,
    const gdf::FrameGdfParams& params, bool gdf_on,
    std::span<const std::uint8_t> tile_payload, std::string* error) {
  av2enc::BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  PutUvlc(writer, 0);    // cur_mfh_id
  PutUvlc(writer, 0);    // seq_header_id_in_frame_header
  writer.put_bit(false); // allow_screen_content_tools
  writer.put_bit(false); // allow_intrabc
  writer.put_bit(config.disable_cdf_update);
  const std::uint32_t mi_cols =
      2U * ((static_cast<std::uint32_t>(config.max_frame_width) + 7U) >> 3U);
  const std::uint32_t mi_rows =
      2U * ((static_cast<std::uint32_t>(config.max_frame_height) + 7U) >> 3U);
  const std::uint32_t sb_cols = (mi_cols + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows + 15U) >> 4U;
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1U) writer.put_bit(false);  // increment_tile_cols_log2 = 0
  if (sb_rows > 1U) writer.put_bit(false);  // increment_tile_rows_log2 = 0
  writer.put_bits(config.base_q_idx, 8U);
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present
  writer.put_bit(config.apply_deblock_y_vert);
  writer.put_bit(config.apply_deblock_y_horz);
  const bool have_chroma =
      config.apply_deblock_y_vert || config.apply_deblock_y_horz;
  if (have_chroma) {
    writer.put_bit(config.apply_deblock_u);
    writer.put_bit(config.apply_deblock_v);
  }
  if (config.apply_deblock_y_vert) writer.put_bit(false);
  if (config.apply_deblock_y_horz) writer.put_bit(false);
  if (have_chroma && config.apply_deblock_u) writer.put_bit(false);
  if (have_chroma && config.apply_deblock_v) writer.put_bit(false);
  if (gdf_on) {
    gdf::FrameGdfParams copy = params;
    gdf::WriteFrameGdfParams(writer, context, &copy, error);
  }
  writer.put_bit(config.tx_mode_select);
  writer.put_bits(static_cast<std::uint32_t>(config.reduced_tx_set), 2U);
  writer.align_to_byte();
  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile_payload.begin(), tile_payload.end());
  return av2enc::basic::make_obu({av2enc::basic::ObuType::kClosedLoopKey}, payload);
}

bool SplitObus(std::span<const std::uint8_t> bytes,
               std::vector<std::vector<std::uint8_t>>* obus) {
  std::size_t position = 0;
  while (position < bytes.size()) {
    std::uint64_t length = 0;
    int shift = 0;
    while (position < bytes.size()) {
      const std::uint8_t byte = bytes[position++];
      length |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
      if ((byte & 0x80U) == 0U) break;
      shift += 7;
    }
    if (position + length > bytes.size()) return false;
    obus->emplace_back(
        bytes.begin() + static_cast<std::ptrdiff_t>(position),
        bytes.begin() + static_cast<std::ptrdiff_t>(position + length));
    position += static_cast<std::size_t>(length);
  }
  return true;
}

std::vector<std::uint8_t> AssembleStream(
    const av2enc::still::StillConfig& config, const gdf::SequenceGdfFlags& seq_flags,
    bool gdf_on, const gdf::FrameGdfContext& context,
    const gdf::FrameGdfParams& params,
    std::span<const std::uint8_t> tile_payload, std::string* error) {
  const std::vector<std::uint8_t> temporal =
      av2enc::basic::make_obu({av2enc::basic::ObuType::kTemporalDelimiter}, {});
  const std::vector<std::uint8_t> sequence =
      MakeSequenceHeader(config, seq_flags, error);
  if (sequence.empty()) return {};
  const std::vector<std::uint8_t> frame =
      MakeFrameObu(config, context, params, gdf_on, tile_payload, error);
  std::vector<std::uint8_t> out;
  out.insert(out.end(), temporal.begin(), temporal.end());
  out.insert(out.end(), sequence.begin(), sequence.end());
  out.insert(out.end(), frame.begin(), frame.end());
  return out;
}

bool WriteIvf(const fs::path& path, std::span<const std::uint8_t> frame,
              int width, int height, std::string* error) {
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = 1;
  av2enc::IvfWriter writer;
  if (!writer.open(path, header, *error)) return false;
  if (!writer.write_frame(frame, 0, *error)) return false;
  return writer.close(*error);
}

struct Planes {
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

bool DecodeAndCompare(const std::string& decoder, const fs::path& ivf,
                      const fs::path& y4m, int width, int height,
                      const Planes& expected, std::string* error) {
  const std::string command = "\"" + decoder + "\" -q -i \"" + ivf.string() +
                              "\" -o \"" + y4m.string() +
                              "\" --strict 1 --threads 1";
  const int status = std::system(command.c_str());
  if (status != 0) {
    *error = "dav2d failed";
    return false;
  }
  av2enc::Y4mReader reader(y4m);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame frame;
  if (reader.read_frame(frame, *error) != av2enc::Y4mReadResult::kFrame) {
    return false;
  }
  if (static_cast<int>(frame.width()) != width ||
      static_cast<int>(frame.height()) != height) {
    *error = "decoded size mismatch";
    return false;
  }
  const std::size_t luma_size =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const std::size_t chroma_size =
      static_cast<std::size_t>((width + 1) / 2) *
      static_cast<std::size_t>((height + 1) / 2);
  std::size_t diffs = 0;
  std::size_t first = 0;
  bool have_first = false;
  const auto compare = [&](std::span<const std::uint8_t> got,
                           const std::vector<std::uint8_t>& want) {
    for (std::size_t i = 0; i < want.size(); ++i) {
      if (got[i] != want[i]) {
        if (!have_first) {
          first = i;
          have_first = true;
        }
        ++diffs;
      }
    }
  };
  compare(frame.y(), expected.y);
  compare(frame.u(), expected.u);
  compare(frame.v(), expected.v);
  if (diffs != 0) {
    *error = "decoded samples differ";
    std::printf("       first diff at %zu (%zu diffs, luma %zu, chroma %zu)\n",
                first, diffs, luma_size, chroma_size);
    return false;
  }
  return true;
}

bool MakeContext(const av2enc::still::StillConfig& config, bool matches,
                 gdf::FrameGdfContext* context) {
  *context = gdf::FrameGdfContext{};
  context->enable_gdf = true;
  context->gdf_unit_matches_sb_size = matches;
  context->single_picture_header_flag = true;
  context->sb_size = 64;
  context->width = config.max_frame_width;
  context->height = config.max_frame_height;
  context->mi_cols = 2 * ((context->width + 7) / 8);
  context->mi_rows = 2 * ((context->height + 7) / 8);
  context->mi_col_starts = {0};
  context->mi_row_starts = {0};
  return true;
}

bool RunGdfPhase(const std::string& name, const std::string& decoder,
                 const av2enc::still::StillConfig& config,
                 std::span<const std::uint8_t> tile_payload,
                 const Planes& recon, const Planes& original, int width,
                 int height, bool matches, bool force_params,
                 int force_qc, int force_scale, const fs::path& work,
                 gdf::GdfDecision* out_decision) {
  std::string error;
  gdf::FrameGdfContext context;
  MakeContext(config, matches, &context);
  const gdf::GdfUnitLayout layout = gdf::DeriveGdfUnitLayout(context);

  gdf::GdfDecisionOptions options;
  options.filter.base_q_idx = static_cast<int>(config.base_q_idx);
  options.filter.frame_is_intra = true;
  options.filter.bit_depth = 8;
  options.disable_cdf_update = true;
  options.search_pic_indices = !force_params;
  options.pic_qc_idx = force_qc;
  options.pic_scale_idx = force_scale;
  options.lambda_bits = 0.0;

  std::vector<std::uint16_t> recon16(static_cast<std::size_t>(width) *
                                     static_cast<std::size_t>(height));
  std::vector<std::uint16_t> orig16(static_cast<std::size_t>(width) *
                                    static_cast<std::size_t>(height));
  for (int i = 0; i < width * height; ++i) {
    recon16[static_cast<std::size_t>(i)] = recon.y[static_cast<std::size_t>(i)];
    orig16[static_cast<std::size_t>(i)] =
        original.y[static_cast<std::size_t>(i)];
  }
  gdf::GdfSourceFrame source;
  source.curr = recon16.data();
  source.cdef = recon16.data();
  source.stride = width;
  gdf::GdfPlane original_plane;
  original_plane.data = orig16.data();
  original_plane.width = width;
  original_plane.height = height;
  original_plane.stride = width;

  gdf::GdfDecision decision;
  if (!gdf::DecideGdf(source, original_plane, context, options, &decision,
                      &error)) {
    std::printf("  FAIL %s decision: %s\n", name.c_str(), error.c_str());
    ++g_failures;
    return false;
  }
  gdf::FrameGdfParams params = decision.params;
  if (!params.frame_enable) {
    params.frame_enable = true;
    params.pic_qc_idx = force_params ? force_qc : decision.chosen_qc_idx;
    params.pic_scale_idx = force_params ? force_scale : decision.chosen_scale_idx;
    params.pix_scale = 1 + params.pic_scale_idx;
    params.blk_size = layout.blk_size;
    params.cols = layout.cols;
    params.rows = layout.rows;
  }
  // The gate stream codes gdf_per_block = 0: inserting use_gdf symbols into the
  // existing range-coded tile would need the tile writer, and the unit tests
  // pin that syntax independently.
  params.per_block = false;

  std::vector<std::uint16_t> out16(recon16);
  if (!gdf::ApplyGdfToFrame(source, options.filter, context, params, {},
                            out16.data(), width, &error)) {
    std::printf("  FAIL %s apply: %s\n", name.c_str(), error.c_str());
    ++g_failures;
    return false;
  }
  std::vector<std::uint8_t> filtered(
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
  for (int i = 0; i < width * height; ++i) {
    filtered[static_cast<std::size_t>(i)] =
        static_cast<std::uint8_t>(out16[static_cast<std::size_t>(i)]);
  }

  std::int64_t applied_sse = 0;
  for (std::size_t i = 0; i < orig16.size(); ++i) {
    const std::int64_t d =
        static_cast<std::int64_t>(orig16[i]) - out16[i];
    applied_sse += d * d;
  }
  const double applied_psnr =
      applied_sse <= 0
          ? 99.0
          : 10.0 * std::log10(
                       255.0 * 255.0 /
                       (static_cast<double>(applied_sse) /
                        static_cast<double>(width) / static_cast<double>(height)));
  const int seq_bits = 1 + (config.max_frame_width >= 64 ? 1 : 0);
  const int frame_bits = (context.single_picture_header_flag ? 0 : 1) +
                         (layout.per_block_present ? 1 : 0) + 4;
  const int stream_header_bits = seq_bits + frame_bits;

  Planes expected;
  expected.y = std::move(filtered);
  expected.u = recon.u;
  expected.v = recon.v;

  gdf::SequenceGdfFlags seq_flags;
  seq_flags.enable_gdf = true;
  seq_flags.gdf_unit_matches_sb_size = matches;
  const std::vector<std::uint8_t> stream =
      AssembleStream(config, seq_flags, true, context, params, tile_payload,
                     &error);
  if (stream.empty()) {
    std::printf("  FAIL %s assemble: %s\n", name.c_str(), error.c_str());
    ++g_failures;
    return false;
  }
  const fs::path ivf = work / (name + ".ivf");
  const fs::path y4m = work / (name + ".decoded.y4m");
  if (!WriteIvf(ivf, stream, width, height, &error)) {
    std::printf("  FAIL %s ivf: %s\n", name.c_str(), error.c_str());
    ++g_failures;
    return false;
  }
  std::printf(
      "  %s: qc=%d scale=%d blk=%d gdf_per_block=0 frame=%zu B "
      "(GDF header=%d bits, symbols=0) psnr %.3f -> %.3f\n",
      name.c_str(), params.pic_qc_idx, params.pic_scale_idx, params.blk_size,
      stream.size(), stream_header_bits, decision.psnr_before, applied_psnr);
  if (decision.params.per_block) {
    std::printf(
        "       (decision preferred per-block: %d/%d units, %lld symbol bits; "
        "gate stream codes gdf_per_block=0)\n",
        decision.enabled_units, decision.total_units,
        static_cast<long long>(decision.symbol_bits));
  }
  Check(DecodeAndCompare(decoder, ivf, y4m, width, height, expected, &error),
        name + " recon 0 diffs");
  if (out_decision != nullptr) *out_decision = decision;
  return true;
}

struct Options {
  fs::path input;
  fs::path work;
  std::string dav2d;
  int qp = 90;
  int matches = 1;
  bool skip_forced = false;
};

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", what);
        return nullptr;
      }
      return argv[++i];
    };
    if (arg == "--input") {
      const char* value = next("--input");
      if (value == nullptr) return false;
      options->input = value;
    } else if (arg == "--work") {
      const char* value = next("--work");
      if (value == nullptr) return false;
      options->work = value;
    } else if (arg == "--dav2d") {
      const char* value = next("--dav2d");
      if (value == nullptr) return false;
      options->dav2d = value;
    } else if (arg == "--qp") {
      const char* value = next("--qp");
      if (value == nullptr) return false;
      options->qp = std::atoi(value);
    } else if (arg == "--matches") {
      const char* value = next("--matches");
      if (value == nullptr) return false;
      options->matches = std::atoi(value) != 0 ? 1 : 0;
    } else if (arg == "--skip-forced") {
      options->skip_forced = true;
    } else if (arg == "-h" || arg == "--help") {
      return false;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return false;
    }
  }
  return !options->input.empty();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    std::printf(
        "usage: gdf_stream_gate --input clip.y4m [--qp Q] [--work DIR]\n"
        "                       [--dav2d PATH] [--matches 0|1] "
        "[--skip-forced]\n");
    return 1;
  }
  if (options.dav2d.empty()) {
    std::printf("gdf_stream_gate: no dav2d (set --dav2d or DAV2D)\n");
    return 77;
  }
  std::error_code ec;
  fs::create_directories(options.work, ec);

  std::printf("GDF stream gate: %s (qp=%d, matches=%d)\n",
              options.input.string().c_str(), options.qp, options.matches);
  std::string error;
  av2enc::Y4mReader reader(options.input);
  if (!reader.open(error)) {
    std::printf("  FAIL input: %s\n", error.c_str());
    return 1;
  }
  av2enc::Y4mFrame frame;
  if (reader.read_frame(frame, error) != av2enc::Y4mReadResult::kFrame) {
    std::printf("  FAIL input frame: %s\n", error.c_str());
    return 1;
  }
  const int width = static_cast<int>(frame.width());
  const int height = static_cast<int>(frame.height());

  av2enc::still::StillPicture picture;
  picture.width = frame.width();
  picture.height = frame.height();
  picture.y = frame.y();
  picture.u = frame.u();
  picture.v = frame.v();
  av2enc::still::StillOptions still_options;
  still_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
  still_options.enable_deblock = true;
  av2enc::still::StillResult result;
  if (!av2enc::still::encode_still_picture(picture, still_options, result, error)) {
    std::printf("  FAIL M1 encode: %s\n", error.c_str());
    return 1;
  }

  std::vector<std::vector<std::uint8_t>> obus;
  if (!SplitObus(result.frame, &obus) || obus.size() < 3) {
    std::printf("  FAIL M1 OBU split\n");
    return 1;
  }
  av2enc::still::StillConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = static_cast<std::uint8_t>(options.qp);
  av2enc::BitWriter header_writer;
  const std::size_t header_bits = av2enc::still::write_frame_header(header_writer, config);
  const std::size_t header_bytes = (1U + header_bits + 7U) / 8U;
  const std::vector<std::uint8_t>& clk = obus[2];
  if (clk.size() <= 1U + header_bytes) {
    std::printf("  FAIL M1 frame too short\n");
    return 1;
  }
  const std::span<const std::uint8_t> tile(
      clk.data() + 1U + header_bytes, clk.size() - 1U - header_bytes);

  Planes recon;
  recon.y = result.recon_y;
  recon.u = result.recon_u;
  recon.v = result.recon_v;
  Planes original;
  original.y.assign(frame.y().begin(), frame.y().end());
  original.u.assign(frame.u().begin(), frame.u().end());
  original.v.assign(frame.v().begin(), frame.v().end());

  // Phase A: enable_gdf = 0 must be byte-identical to the M1 stream.
  {
    gdf::SequenceGdfFlags seq_flags;
    gdf::FrameGdfContext context;
    MakeContext(config, options.matches != 0, &context);
    context.enable_gdf = false;
    gdf::FrameGdfParams params;
    const std::vector<std::uint8_t> stream =
        AssembleStream(config, seq_flags, false, context, params, tile, &error);
    std::vector<std::uint8_t> baseline(result.frame.begin(),
                                       result.frame.end());
    Check(!stream.empty() && stream == baseline,
          "enable_gdf=0 stream is byte-identical to the M1 baseline");
    const fs::path ivf = options.work / "gdf_disabled.ivf";
    const fs::path y4m = options.work / "gdf_disabled.decoded.y4m";
    Check(WriteIvf(ivf, stream, width, height, &error),
          "disabled stream muxed");
    Check(DecodeAndCompare(options.dav2d, ivf, y4m, width, height, recon,
                           &error),
          "disabled stream decodes to the M1 reconstruction");
  }

  // Phase B: decision-selected parameters.
  gdf::GdfDecision decision;
  RunGdfPhase("gdf_decision", options.dav2d, config, tile, recon,
              original, width, height, options.matches != 0, false, 0, 0,
              options.work, &decision);

  // Phase C: forced non-trivial parameters.
  if (!options.skip_forced) {
    RunGdfPhase("gdf_qc2_scale1", options.dav2d, config, tile, recon,
                original, width, height, options.matches != 0, true, 2, 1,
                options.work, nullptr);
  }

  // Cross-check the opposite GdfBlkSize when the frame is large enough for the
  // grid to differ.
  if (width >= 128 && height >= 128) {
    const bool other_matches = options.matches == 0;
    RunGdfPhase(other_matches ? "gdf_matches" : "gdf_nomatch", options.dav2d,
                config, tile, recon, original, width, height,
                other_matches, true, 2, 1, options.work, nullptr);
  }

  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
