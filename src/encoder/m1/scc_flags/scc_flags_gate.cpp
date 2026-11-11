// dav2d gate for the §5.4.7/§5.18.3.3 SCC signalling module.
//
// Four stills are produced.  Each is coded by an existing tile coder
// (ibc_stream or palette_stream), then its header is re-authored by scc_flags:
//
//   off      screen tools off, IBC off             (byte-identity control)
//   integer  screen tools on, force_integer_mv=1
//   intrabc  screen tools on, force_integer_mv=1, allow_intrabc=1 (global)
//   palette  screen tools on, force_integer_mv=1, real palette tile
//
// For every mode the gate asserts:
//   * scc_flags::assemble_still_picture() reproduces the existing module's
//     frame byte-for-byte (so the header bits did not drift);
//   * the independent §5.4.7/§5.18.3.3/.4 reader recovers the expected state;
//   * the "off" stream equals m1::assemble_m1_still_picture() byte-for-byte.
// build_and_test.sh then decodes each IVF with dav2d and compares the
// reconstruction to the writer's Y4M.
#include "encoder/m1/scc_flags/scc_flags.h"
#include "encoder/m1/scc_flags/scc_flags_bridge.h"

#include "encoder/m1/ibc_stream/m1_ibc_stream.h"
#include "encoder/m1/m1_headers.h"
#include "encoder/m1/palette_stream/palette_stream.h"
#include "io/ivf.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace scc = av2enc::m1::scc_flags;
namespace ibc_stream = av2enc::m1::ibc_stream;
namespace palette_stream = av2enc::m1::palette_stream;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

bool find_clk_payload(std::span<const std::uint8_t> frame,
                      std::span<const std::uint8_t>& payload,
                      std::string& error) {
  std::size_t offset = 0;
  while (offset < frame.size()) {
    std::uint32_t size = 0;
    unsigned shift = 0;
    for (;;) {
      if (offset >= frame.size() || shift >= 32U) {
        error = "malformed leb128";
        return false;
      }
      const std::uint8_t byte = frame[offset++];
      size |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
      if ((byte & 0x80U) == 0U) break;
      shift += 7U;
    }
    if (size == 0U || offset + size > frame.size()) {
      error = "OBU size out of range";
      return false;
    }
    const std::uint8_t header = frame[offset];
    const unsigned type = (header >> 2U) & 0x1FU;
    if (type == 4U) {  // OBU_CLOSED_LOOP_KEY
      payload = frame.subspan(offset + 1U, size - 1U);
      return true;
    }
    offset += size;
  }
  error = "CLK OBU not found";
  return false;
}

av2enc::m1::M1Config m1_config_for_ibc(const ibc_stream::StreamConfig& c) {
  av2enc::m1::M1Config config;
  config.max_frame_width = static_cast<std::uint16_t>(c.width);
  config.max_frame_height = static_cast<std::uint16_t>(c.height);
  config.seq_level_idx = c.seq_level_idx;
  config.base_q_idx = c.base_q_idx;
  config.disable_cdf_update = c.disable_cdf_update;
  config.enable_intra_edge_filter = c.enable_intra_edge_filter;
  config.enable_cfl_intra = c.enable_cfl_intra;
  config.enable_chroma_dctonly = c.enable_chroma_dctonly;
  config.tx_mode_select = false;
  config.reduced_tx_set = c.reduced_tx_set;
  config.apply_deblock_y_vert = c.apply_deblock_y_vert;
  config.apply_deblock_y_horz = c.apply_deblock_y_horz;
  config.apply_deblock_u = c.apply_deblock_u;
  config.apply_deblock_v = c.apply_deblock_v;
  return config;
}

av2enc::m1::M1Config m1_config_for_palette(
    const palette_stream::MinimalPaletteFramePlan& plan) {
  av2enc::m1::M1Config config;
  config.max_frame_width = static_cast<std::uint16_t>(plan.width);
  config.max_frame_height = static_cast<std::uint16_t>(plan.height);
  config.base_q_idx = plan.base_q_idx;
  config.enable_intra_edge_filter = plan.enable_intra_edge_filter;
  config.enable_cfl_intra = plan.enable_cfl_intra;
  config.enable_chroma_dctonly = true;
  config.tx_mode_select = false;
  config.apply_deblock_y_vert = plan.apply_deblock_y_vert;
  config.apply_deblock_y_horz = plan.apply_deblock_y_horz;
  config.apply_deblock_u = plan.apply_deblock_u;
  config.apply_deblock_v = plan.apply_deblock_v;
  return config;
}

bool write_ivf(const std::vector<std::uint8_t>& frame, int width, int height,
               const std::string& path, std::string& error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = 1;
  if (!writer.open(std::filesystem::path(path), header, error)) return false;
  if (!writer.write_frame(frame, 0, error)) return false;
  return writer.close(error);
}

bool write_y4m(const std::vector<std::uint8_t>& y,
               const std::vector<std::uint8_t>& u,
               const std::vector<std::uint8_t>& v, int width, int height,
               const std::string& path, std::string& error) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    error = "cannot open " + path;
    return false;
  }
  std::fprintf(f, "YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n", width, height);
  std::fputs("FRAME\n", f);
  std::fwrite(y.data(), 1, y.size(), f);
  std::fwrite(u.data(), 1, u.size(), f);
  std::fwrite(v.data(), 1, v.size(), f);
  const bool ok = std::ferror(f) == 0;
  std::fclose(f);
  if (!ok) error = "Y4M write failed";
  return ok;
}

// Shared body: extract the tile from an existing still, re-assemble it with
// scc_flags, and validate byte identity + the independent reader.
bool reassemble_and_check(const std::string& mode,
                          const av2enc::m1::M1Config& m1cfg,
                          const scc::BridgedSccConfig& bridged,
                          const std::vector<std::uint8_t>& existing_frame,
                          const std::vector<std::uint8_t>& recon_y,
                          const std::vector<std::uint8_t>& recon_u,
                          const std::vector<std::uint8_t>& recon_v, int width,
                          int height, const std::string& out_prefix) {
  std::string error;

  std::size_t header_bits = 0;
  std::size_t header_bytes = 0;
  scc::FrameSccState expected;
  if (!scc::still_frame_header_size(
          m1cfg, scc::StillSccConfig{bridged.sequence, bridged.frame},
          bridged.seq_max_bvp_drl_bits_minus_1,
          bridged.allow_frame_max_bvp_drl_bits, &header_bits, &header_bytes,
          &expected, error)) {
    check(false, mode + ": header size (" + error + ")");
    return false;
  }

  std::span<const std::uint8_t> clk;
  if (!find_clk_payload(existing_frame, clk, error)) {
    check(false, mode + ": locate CLK (" + error + ")");
    return false;
  }
  if (header_bytes > clk.size()) {
    check(false, mode + ": header longer than CLK payload");
    return false;
  }
  const std::span<const std::uint8_t> tile = clk.subspan(header_bytes);

  scc::StillSccConfig scc_cfg;
  scc_cfg.sequence = bridged.sequence;
  scc_cfg.frame = bridged.frame;
  scc::StillSccResult ours;
  if (!scc::assemble_still_picture(
          m1cfg, scc_cfg, bridged.seq_max_bvp_drl_bits_minus_1,
          bridged.allow_frame_max_bvp_drl_bits, tile, ours, error)) {
    check(false, mode + ": re-assemble (" + error + ")");
    return false;
  }

  check(ours.frame == existing_frame,
        mode + ": scc_flags re-assembly is byte-identical to the tile coder");
  check(ours.frame_header_bytes == header_bytes,
        mode + ": header byte count matches the tile offset");
  check(ours.frame_state.allow_screen_content_tools ==
            expected.allow_screen_content_tools &&
            ours.frame_state.force_integer_mv == expected.force_integer_mv &&
            ours.frame_state.allow_intrabc == expected.allow_intrabc &&
            ours.frame_state.allow_global_intrabc ==
                expected.allow_global_intrabc &&
            ours.frame_state.allow_local_intrabc ==
                expected.allow_local_intrabc,
        mode + ": assembler state");

  // Independent read-back from the bytes actually gated.
  const scc::SequenceSccState sequence =
      scc::derive_sequence_scc_state(bridged.sequence);
  scc::FrameSccContext context;
  context.frame_is_intra = true;
  scc::FrameSccState read;
  if (!scc::read_still_frame_scc(
          existing_frame, sequence, context,
          bridged.seq_max_bvp_drl_bits_minus_1,
          bridged.allow_frame_max_bvp_drl_bits, read, error)) {
    check(false, mode + ": independent reader (" + error + ")");
    return false;
  }
  check(read.allow_screen_content_tools ==
            ours.frame_state.allow_screen_content_tools,
        mode + ": reader allow_screen_content_tools");
  check(read.force_integer_mv == ours.frame_state.force_integer_mv,
        mode + ": reader force_integer_mv");
  check(read.allow_intrabc == ours.frame_state.allow_intrabc,
        mode + ": reader allow_intrabc");
  check(read.allow_global_intrabc == ours.frame_state.allow_global_intrabc,
        mode + ": reader allow_global_intrabc");
  check(read.allow_local_intrabc == ours.frame_state.allow_local_intrabc,
        mode + ": reader allow_local_intrabc");
  check(read.IsIntraBC == read.allow_intrabc,
        mode + ": derived IsIntraBC");
  check(read.coded_allow_screen_content_tools,
        mode + ": allow_screen_content_tools is present (single picture)");

  if (mode == "off") {
    const std::vector<std::uint8_t> reference =
        av2enc::m1::assemble_m1_still_picture(m1cfg, tile);
    check(existing_frame == reference,
          "off: byte-identical to m1::assemble_m1_still_picture()");
    check(!read.allow_screen_content_tools && !read.force_integer_mv &&
              !read.allow_intrabc,
          "off: all screen-content flags are zero");
  }
  if (mode == "integer" || mode == "palette") {
    check(read.allow_screen_content_tools && read.force_integer_mv &&
              !read.allow_intrabc,
          mode + ": screen tools + integer MVs, IBC off");
    check(read.coded_force_integer_mv,
          mode + ": force_integer_mv is coded");
  }
  if (mode == "intrabc") {
    check(read.allow_screen_content_tools && read.force_integer_mv &&
              read.allow_intrabc && read.allow_global_intrabc &&
              !read.allow_local_intrabc,
          "intrabc: global-only intra block copy");
  }

  const std::string ivf = out_prefix + ".ivf";
  const std::string y4m = out_prefix + ".y4m";
  if (!write_ivf(ours.frame, width, height, ivf, error)) {
    check(false, mode + ": write IVF (" + error + ")");
    return false;
  }
  if (!write_y4m(recon_y, recon_u, recon_v, width, height, y4m, error)) {
    check(false, mode + ": write Y4M (" + error + ")");
    return false;
  }
  std::printf("%s: %dx%d header=%zu bits/%zu bytes tile=%zu frame=%zu\n",
              mode.c_str(), width, height, header_bits, header_bytes,
              tile.size(), ours.frame.size());
  return true;
}

// The proven 768x128 IBC plan shape from ibc_stream_test.cpp.
ibc_stream::StreamPlan make_ibc_plan(bool with_ibc, bool screen_tools,
                                     bool intrabc) {
  ibc_stream::StreamPlan plan;
  plan.config.width = 768;
  plan.config.height = 128;
  plan.config.allow_screen_content_tools = screen_tools;
  plan.config.force_integer_mv = true;
  plan.config.allow_intrabc = intrabc;
  plan.config.allow_global_intrabc = intrabc;
  plan.config.allow_local_intrabc = false;

  const int sb_cols = 12;
  const int sb_rows = 2;
  for (int sb = 0; sb < sb_cols * sb_rows; ++sb) {
    const int sb_row = sb / sb_cols;
    const int sb_col = sb % sb_cols;
    ibc_stream::StreamSb block;
    block.partition = ibc_stream::StreamSb::Partition::kHorz;
    for (int half = 0; half < 2; ++half) {
      ibc_stream::StreamBlock leaf;
      leaf.mi_row = sb_row * 16 + half * 8;
      leaf.mi_col = sb_col * 16;
      leaf.bsize = 11;  // BLOCK_64X32
      if (with_ibc && sb_row == 1 && half == 0 && sb_col == 8) {
        leaf.ibc = true;
        leaf.skip = true;
        leaf.syntax.mode = ibc_stream::IntrabcMode::kNewMv;
        leaf.syntax.ref_mv_idx = 0;
        leaf.bv = ibc_stream::BlockVector{0, -384 * 8};
      }
      block.leaves.push_back(leaf);
    }
    plan.sbs.push_back(block);
  }
  return plan;
}

bool gate_ibc_mode(const std::string& mode, bool with_ibc, bool screen_tools,
                   bool intrabc, const std::string& out_prefix) {
  ibc_stream::StreamPlan plan =
      make_ibc_plan(with_ibc, screen_tools, intrabc);
  ibc_stream::StreamResult result;
  std::string error;
  if (!ibc_stream::encode_ibc_stream(plan, result, error)) {
    check(false, mode + ": ibc_stream encode (" + error + ")");
    return false;
  }
  if (with_ibc) {
    check(result.ibc_blocks > 0,
          mode + ": the gate stream really contains intra block copy");
  } else {
    check(result.ibc_blocks == 0,
          mode + ": the gate stream contains no intra block copy");
  }
  scc::BridgedSccConfig bridged;
  if (!scc::from_ibc_config(plan.config, bridged, error)) {
    check(false, mode + ": bridge (" + error + ")");
    return false;
  }
  return reassemble_and_check(mode, m1_config_for_ibc(plan.config), bridged,
                              result.frame, result.recon_y, result.recon_u,
                              result.recon_v, plan.config.width,
                              plan.config.height, out_prefix);
}

bool gate_palette_mode(const std::string& out_prefix) {
  palette_stream::MinimalPaletteFramePlan plan;
  plan.flags = palette_stream::palette_screen_content_flags();
  plan.width = 64;
  plan.height = 64;
  palette_stream::MinimalPaletteSuperblock sb;
  sb.partition = palette_stream::MinimalPaletteSuperblock::Partition::kNone;
  palette_stream::MinimalPaletteLeaf leaf;
  leaf.mi_row = 0;
  leaf.mi_col = 0;
  leaf.bsize = 12;
  std::vector<std::uint8_t> pattern(64U * 64U, 0);
  static constexpr std::uint8_t kValues[4] = {16, 80, 160, 240};
  for (int r = 0; r < 64; ++r) {
    for (int c = 0; c < 64; ++c) {
      const int idx = ((r / 8) + (c / 8)) & 3;
      pattern[static_cast<std::size_t>(r) * 64U +
              static_cast<std::size_t>(c)] = kValues[idx];
    }
  }
  std::string error;
  if (!av2enc::palette::analyze_palette(pattern.data(), 64, 64, 64, 64, 64,
                                        nullptr, 0,
                                        av2enc::palette::PaletteParams{},
                                        leaf.palette, error)) {
    check(false, "palette: analyze (" + error + ")");
    return false;
  }
  sb.leaves.push_back(leaf);
  plan.sbs.push_back(sb);

  palette_stream::MinimalPaletteFrameResult result;
  if (!palette_stream::encode_minimal_palette_frame(plan, result, error)) {
    check(false, "palette: encode (" + error + ")");
    return false;
  }
  scc::BridgedSccConfig bridged;
  if (!scc::from_palette_flags(plan.flags, bridged, error)) {
    check(false, "palette: bridge (" + error + ")");
    return false;
  }
  return reassemble_and_check("palette", m1_config_for_palette(plan), bridged,
                              result.frame, result.recon_y, result.recon_u,
                              result.recon_v, plan.width, plan.height,
                              out_prefix);
}

}  // namespace

int main(int argc, char** argv) {
  std::string prefix;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out-prefix" && i + 1 < argc) {
      prefix = argv[++i];
    } else {
      std::fprintf(stderr, "usage: %s --out-prefix PREFIX\n", argv[0]);
      return 2;
    }
  }
  if (prefix.empty()) {
    std::fprintf(stderr, "--out-prefix is required\n");
    return 2;
  }

  gate_ibc_mode("off", false, false, false, prefix + ".off");
  gate_ibc_mode("integer", false, true, false, prefix + ".integer");
  gate_ibc_mode("intrabc", true, true, true, prefix + ".intrabc");
  gate_palette_mode(prefix + ".palette");

  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
