// dav2d gate for the frame-header reference machinery.
//
// Three 3-frame streams are built from the same two-tile tile payloads
// (128x64, 2x1 uniform tiles, one 64x64 intra/inter leaf per tile):
//
//   baseline : the M2 header form.  Every reference element is either coded
//              with the generic form or inferred:
//                - refresh_frame_flags f(2) (no short-refresh syntax)
//                - max_drl_bits_minus_1 = sequence value (no change_drl)
//                - no primary_ref_frame (disable_cross_frame_cdf_init = 1)
//                - frame_size_override_flag = 0
//   found_ref: frame 1 uses frame_size_override_flag = 1 and inherits
//              128x64 through frame_size_with_refs() found_ref = 1
//              (§5.18.4.3).  Both P frames signal primary_ref_frame = 0 with
//              disable_cross_frame_cdf_init = 0 (so the CDF state comes from
//              the reference slot), use has_refresh_frame_flags =
//              frame_to_refresh = 0, and change max_drl_bits_minus_1 to 3.
//              Frame 2 also sets allow_intrabc and changes
//              max_bvp_drl_bits_minus_1 to 1.
//   explicit : the control.  Identical to `found_ref` except frame 1 codes
//              found_ref = 0 followed by the explicit frame_size() fields.
//
// Frame 2 is split into two tile groups in both non-baseline streams; the
// second group sets frame_header_present_flag = 1 and repeats the first header
// bit for bit through frame_header_copy() (§5.18.1).
//
// All three streams must decode with `dav2d --strict 1 --threads 1` to the
// same samples as the encoder model: every tile is a single 64x64 leaf, so the
// keyframe DC predictor is 128 and every P frame is GLOBALMV (copy).  The
// decoded Y4M must therefore be a constant 128 in all planes, and the three
// streams must be byte-identical to each other.
//
// The gate also parses the produced headers with the independent reader in
// ref_fields.cpp and asserts the fields each stream is supposed to exercise.
//
// Usage:
//   ref_fields_gate [--out prefix] [--dav2d path]
//
// Exit: 0 pass, 1 failure, 2 usage, 77 skip (no dav2d).
#include "encoder/frame/ref_fields/ref_fields.h"
#include "encoder/frame/ref_fields/ref_fields_m2_adapter.h"
#include "encoder/m2/m2_headers.h"
#include "encoder/m2/m2_syntax.h"
#include "encoder/m2/post/tiled_frame.h"
#include "encoder/tiles/tile_group.h"
#include "encoder/tiles/tile_layout.h"
#include "io/ivf.h"
#include "io/y4m.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace av2enc::frame::ref_fields;

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
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

bool write_ivf(const std::string& path,
               const std::vector<std::vector<std::uint8_t>>& units,
               std::uint16_t width, std::uint16_t height, std::string& error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = width;
  header.height = height;
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = static_cast<std::uint32_t>(units.size());
  if (!writer.open(path, header, error)) return false;
  for (std::size_t i = 0; i < units.size(); ++i) {
    if (!writer.write_frame(units[i], i, error)) return false;
  }
  return writer.close(error);
}

bool run_decoder(const std::string& dav2d, const std::string& ivf,
                 const std::string& y4m, std::string& error) {
  const std::string command = "\"" + dav2d + "\" -q -i \"" + ivf + "\" -o \"" +
                              y4m + "\" --threads 1 --strict 1 2>&1";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    error = "dav2d failed (" + std::to_string(rc) + "): " + command;
    return false;
  }
  if (!std::filesystem::exists(y4m)) {
    error = "dav2d produced no output";
    return false;
  }
  return true;
}

bool check_constant_128(const std::string& path, int frames, int width,
                        int height, std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  int count = 0;
  std::uint64_t mismatches = 0;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    if (frame.width() != static_cast<std::uint32_t>(width) ||
        frame.height() != static_cast<std::uint32_t>(height)) {
      error = "decoded frame has the wrong dimensions";
      return false;
    }
    ++count;
    for (std::span<const std::uint8_t> plane :
         {frame.y(), frame.u(), frame.v()}) {
      for (std::uint8_t sample : plane) {
        if (sample != 128) ++mismatches;
      }
    }
  }
  if (count != frames) {
    error = "decoder produced " + std::to_string(count) + " of " +
            std::to_string(frames) + " frames";
    return false;
  }
  if (mismatches != 0) {
    error = "decoded samples differ from the DC/GLOBALMV constant: " +
            std::to_string(mismatches);
    return false;
  }
  return true;
}

// --- synthetic tile payloads -------------------------------------------------
av2enc::m2::syntax::BlockSyntax make_root(int mi_row, int mi_col, bool inter) {
  av2enc::m2::syntax::BlockSyntax node;
  node.mi_row = static_cast<std::uint16_t>(mi_row);
  node.mi_col = static_cast<std::uint16_t>(mi_col);
  node.bsize = 12;     // BLOCK_64X64
  node.partition = 0;  // PARTITION_NONE
  node.allowed_mask = 1U << 0;
  node.flags = av2enc::m2::syntax::kFlagSbRoot;
  node.subtree = 0;
  node.is_inter = inter ? 1 : 0;
  node.skip_flag = inter ? 1 : 0;
  node.inter_mode = av2enc::m2::syntax::kInterGlobalMv;
  node.mv_y = 0;
  node.mv_x = 0;
  node.y_mode = 0;  // DC_PRED
  node.uv_mode = 0;
  node.tx_size[0] = 4;
  node.tx_size[1] = 3;
  node.tx_size[2] = 3;
  node.eob[0] = node.eob[1] = node.eob[2] = 0;
  node.levels_offset = 0;
  return node;
}

bool make_tile_payloads(const av2enc::tiles::TileLayout& layout, bool inter,
                        const M2RefConfig& config,
                        std::vector<std::vector<std::uint8_t>>* payloads,
                        std::string& error) {
  std::vector<av2enc::m2::syntax::BlockSyntax> blocks;
  for (int sb_row = 0; sb_row < layout.sb_rows; ++sb_row) {
    for (int sb_col = 0; sb_col < layout.sb_cols; ++sb_col) {
      blocks.push_back(make_root(sb_row * 16, sb_col * 16, inter));
    }
  }
  const std::vector<std::int32_t> levels;
  payloads->clear();
  for (int tile = 0; tile < layout.tile_count(); ++tile) {
    av2enc::m2::post::M2TileBlocks tile_blocks;
    if (!av2enc::m2::post::split_m2_blocks_for_tile(layout, tile, blocks,
                                                    &tile_blocks, &error)) {
      return false;
    }
    av2enc::m2::syntax::FramePlan plan;
    plan.width = tile_blocks.width;
    plan.height = tile_blocks.height;
    plan.sb_size = layout.sb_size;
    plan.inter_frame = inter;
    av2enc::m2::M2Config local;
    local.max_frame_width = static_cast<std::uint16_t>(tile_blocks.width);
    local.max_frame_height = static_cast<std::uint16_t>(tile_blocks.height);
    local.base_q_idx = config.base_q_idx;
    local.order_hint_bits = config.order_hint_bits;
    local.num_ref_frames = config.num_ref_frames;
    local.num_total_refs = config.num_total_refs;
    local.max_drl_bits_minus_1 = config.seq_max_drl_bits_minus_1;
    local.disable_cdf_update = config.disable_cdf_update;
    local.enable_cfl_intra = config.enable_cfl_intra;
    std::vector<std::uint8_t> payload;
    if (!av2enc::m2::syntax::assemble_m2_frame(
            plan, tile_blocks.blocks, levels, local, payload, error)) {
      return false;
    }
    payloads->push_back(std::move(payload));
  }
  return true;
}

struct StreamSpec {
  bool short_refresh = false;
  bool allow_drl_change = false;
  bool allow_bvp_change = false;
  bool signal_primary = false;
  bool override_size = false;
  bool prefer_found_ref = true;
  bool split_tile_groups = false;
};

bool build_stream(const M2RefConfig& config, const StreamSpec& spec,
                  const M2TileInfo& tile_info,
                  const std::vector<std::vector<std::vector<std::uint8_t>>>&
                      payloads,
                  std::vector<std::vector<std::uint8_t>>* units,
                  std::vector<M2RefParsedHeader>* parsed, std::string& error) {
  const std::uint16_t width = config.max_frame_width;
  const std::uint16_t height = config.max_frame_height;
  const std::vector<std::uint8_t> seq =
      make_m2_ref_sequence_header_obu(config, &error);
  if (seq.empty()) return false;

  FrameRefState state;
  state.num_ref_frames = config.num_ref_frames;
  state.max_frame_width = width;
  state.max_frame_height = height;
  state.frame_width_bits = bit_width(static_cast<std::uint32_t>(width) - 1U);
  state.frame_height_bits = bit_width(static_cast<std::uint32_t>(height) - 1U);
  state.order_hint_bits = config.order_hint_bits;
  state.enable_order_hint = true;
  state.single_picture_header_flag = false;

  RefList refs;
  refs.num_total_refs = config.num_total_refs;
  refs.index[0] = 0;

  units->clear();
  parsed->clear();
  for (int i = 0; i < 3; ++i) {
    const bool key = i == 0;
    state.slot[0].valid = i > 0;
    state.slot[0].width = width;
    state.slot[0].height = height;
    state.slot[0].base_q_idx = config.base_q_idx;
    state.slot[0].order_hint = i > 0 ? i - 1 : 0;
    state.slot[0].frame_type = i > 1 ? FrameType::kInter : FrameType::kKey;
    state.order_hint = i;

    M2RefFrameHeader header;
    header.keyframe = key;
    header.frame_is_inter = !key;
    header.order_hint = static_cast<std::uint32_t>(i);
    header.refresh_frame_flags = key ? 3U : 1U;
    header.signal_primary_ref_frame = spec.signal_primary && !key;
    header.primary_ref_frame = 0;
    header.disable_cross_frame_cdf_init =
        header.signal_primary_ref_frame ? false : true;
    header.frame_size_override = spec.override_size && !key;
    header.prefer_found_ref = spec.prefer_found_ref;
    header.frame_width = width;
    header.frame_height = height;
    header.allow_intrabc = spec.allow_bvp_change && i == 2;
    header.change_drl = spec.allow_drl_change && !key;
    header.max_drl_bits_minus_1 =
        header.change_drl ? config.seq_max_drl_bits_minus_1 - 1U
                          : config.seq_max_drl_bits_minus_1;
    header.change_bvp_drl = header.allow_intrabc;
    header.max_bvp_drl_bits_minus_1 =
        header.change_bvp_drl ? config.seq_max_bvp_drl_bits_minus_1 + 1U
                              : config.seq_max_bvp_drl_bits_minus_1;

    const std::vector<std::vector<std::uint8_t>> obus =
        make_m2_ref_frame_obus(config, header, refs, state, tile_info,
                               payloads[i], spec.split_tile_groups && !key,
                               &error);
    if (obus.empty()) {
      error = "frame " + std::to_string(i) + ": " + error;
      return false;
    }
    std::vector<std::uint8_t> unit;
    const std::vector<std::uint8_t> td =
        av2enc::m2::make_temporal_delimiter_obu();
    unit.insert(unit.end(), td.begin(), td.end());
    if (key) unit.insert(unit.end(), seq.begin(), seq.end());
    for (const auto& obu : obus) {
      unit.insert(unit.end(), obu.begin(), obu.end());
    }
    units->push_back(std::move(unit));

    // Parse every tile-group OBU back with the independent reader.
    FrameHeaderStore store;
    M2RefParsedHeader info;
    const unsigned tile_bits =
        static_cast<unsigned>(tile_info.cols_log2 + tile_info.rows_log2);
    for (std::size_t g = 0; g < obus.size(); ++g) {
      const auto& obu = obus[g];
      const std::size_t leb = (obu[0] & 0x80U) != 0U ? 2U : 1U;
      const std::size_t payload_offset = 1U + leb;  // leb128 + OBU header byte
      const auto payload =
          std::span<const std::uint8_t>(obu)
              .subspan(payload_offset, obu.size() - payload_offset);
      RefBitReader reader(payload);
      bool is_first = false;
      if (!reader.bit(&is_first)) {
        error = "frame " + std::to_string(i) + ": truncated tile group";
        return false;
      }
      if (is_first != (g == 0)) {
        error = "frame " + std::to_string(i) + ": is_first_tile_group";
        return false;
      }
      if (g == 0) {
        const std::size_t header_start = reader.position();
        if (!read_m2_ref_frame_header(reader, key, config, refs, state,
                                      tile_info, &info, &error)) {
          error = "frame " + std::to_string(i) + ": " + error;
          return false;
        }
        if (obus.size() > 1) {
          store.record(payload, header_start, reader.position());
        }
      } else {
        bool present = false;
        if (!read_m2_ref_nonfirst_tile_group_header(reader, store, &present,
                                                    &error)) {
          error = "frame " + std::to_string(i) + ": " + error;
          return false;
        }
        if (!present) {
          error = "frame " + std::to_string(i) +
                  ": expected a shared-header copy";
          return false;
        }
      }
      if (tile_info.tile_count > 1) {
        bool flag = false;
        std::uint32_t tg_start = 0;
        std::uint32_t tg_end =
            static_cast<std::uint32_t>(tile_info.tile_count) - 1U;
        if (!reader.bit(&flag)) {
          error = "frame " + std::to_string(i) + ": truncated tile range";
          return false;
        }
        if (flag && (!reader.bits(tile_bits, &tg_start) ||
                     !reader.bits(tile_bits, &tg_end))) {
          error = "frame " + std::to_string(i) + ": truncated tile range";
          return false;
        }
        const std::uint32_t expected =
            obus.size() > 1 ? static_cast<std::uint32_t>(g) : 0U;
        const std::uint32_t expected_end =
            obus.size() > 1 ? static_cast<std::uint32_t>(g)
                            : static_cast<std::uint32_t>(tile_info.tile_count) -
                                  1U;
        if (tg_start != expected || tg_end != expected_end) {
          error = "frame " + std::to_string(i) + ": unexpected tile range";
          return false;
        }
      }
    }
    parsed->push_back(info);
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_prefix = "ref_fields_gate.ivf";
  std::string dav2d;
  long feature_mask = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_prefix = argv[++i];
    } else if (arg == "--dav2d" && i + 1 < argc) {
      dav2d = argv[++i];
    } else if (arg == "--features" && i + 1 < argc) {
      // Diagnostic: bit 0 short refresh, 1 change_drl, 2 change_bvp,
      // 3 primary_ref_frame, 4 frame_size_with_refs, 5 tile-group copy,
      // 6 explicit size instead of found_ref (requires bit 4).
      feature_mask = std::strtol(argv[++i], nullptr, 0);
    } else if (arg == "--help") {
      std::printf(
      "usage: ref_fields_gate [--out prefix] [--dav2d path] "
      "[--features mask]\n");
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }
  if (dav2d.empty()) dav2d = find_dav2d();
  if (dav2d.empty() || !std::filesystem::exists(dav2d)) {
    std::fprintf(stderr, "SKIP: no dav2d binary (set DAV2D=...)\n");
    return 77;
  }

  const int width = 128;
  const int height = 64;
  std::string error;
  const auto plan = av2enc::m2::post::plan_m2_tiles(width, height, 2, 1);
  if (!plan.ok) {
    std::fprintf(stderr, "TILE PLAN FAIL: %s\n", plan.error.c_str());
    return 1;
  }
  const av2enc::tiles::TileLayout& layout = plan.layout;

  M2RefConfig base;
  base.max_frame_width = static_cast<std::uint16_t>(width);
  base.max_frame_height = static_cast<std::uint16_t>(height);
  base.seq_level_idx = 7;
  base.base_q_idx = 90;
  base.order_hint_bits = 4;
  base.num_ref_frames = 2;
  base.num_total_refs = 1;
  base.seq_max_drl_bits_minus_1 = 4;
  base.seq_max_bvp_drl_bits_minus_1 = 0;
  base.disable_cdf_update = true;
  base.enable_cfl_intra = true;

  std::vector<std::vector<std::uint8_t>> key_payloads;
  std::vector<std::vector<std::uint8_t>> inter_payloads;
  if (!make_tile_payloads(layout, false, base, &key_payloads, error) ||
      !make_tile_payloads(layout, true, base, &inter_payloads, error)) {
    std::fprintf(stderr, "TILE PAYLOAD FAIL: %s\n", error.c_str());
    return 1;
  }
  std::uint32_t largest = 0;
  for (const auto& payload : key_payloads) {
    largest = std::max(largest, static_cast<std::uint32_t>(payload.size()));
  }
  for (const auto& payload : inter_payloads) {
    largest = std::max(largest, static_cast<std::uint32_t>(payload.size()));
  }
  const int tile_size_bytes = av2enc::tiles::tile_size_bytes_for(largest);
  if (tile_size_bytes == 0) {
    std::fprintf(stderr, "TILE SIZE FAIL\n");
    return 1;
  }
  const M2TileInfo tile_info = make_m2_tile_info(layout, tile_size_bytes);
  const std::vector<std::vector<std::vector<std::uint8_t>>> payloads = {
      key_payloads, inter_payloads, inter_payloads};

  M2RefConfig off = base;
  M2RefConfig on = base;
  on.allow_frame_max_drl_bits = true;
  on.allow_frame_max_bvp_drl_bits = true;
  on.enable_short_refresh_frame_flags = true;

  const StreamSpec baseline_spec{};
  StreamSpec found_spec;
  found_spec.short_refresh = true;
  found_spec.allow_drl_change = true;
  found_spec.allow_bvp_change = true;
  found_spec.signal_primary = true;
  found_spec.override_size = true;
  found_spec.prefer_found_ref = true;
  found_spec.split_tile_groups = true;
  if (feature_mask >= 0) {
    const long mask = feature_mask;
    found_spec.short_refresh = (mask & 0x01) != 0;
    found_spec.allow_drl_change = (mask & 0x02) != 0;
    found_spec.allow_bvp_change = (mask & 0x04) != 0;
    found_spec.signal_primary = (mask & 0x08) != 0;
    found_spec.override_size = (mask & 0x10) != 0;
    found_spec.prefer_found_ref = (mask & 0x40) == 0;
    found_spec.split_tile_groups = (mask & 0x20) != 0;
  }
  StreamSpec explicit_spec = found_spec;
  explicit_spec.prefer_found_ref = false;

  std::vector<std::vector<std::uint8_t>> baseline_units;
  std::vector<std::vector<std::uint8_t>> found_units;
  std::vector<std::vector<std::uint8_t>> explicit_units;
  std::vector<M2RefParsedHeader> baseline_parsed;
  std::vector<M2RefParsedHeader> found_parsed;
  std::vector<M2RefParsedHeader> explicit_parsed;

  if (!build_stream(off, baseline_spec, tile_info, payloads, &baseline_units,
                    &baseline_parsed, error)) {
    std::fprintf(stderr, "BASELINE BUILD FAIL: %s\n", error.c_str());
    return 1;
  }
  if (!build_stream(on, found_spec, tile_info, payloads, &found_units,
                    &found_parsed, error)) {
    std::fprintf(stderr, "FOUND_REF BUILD FAIL: %s\n", error.c_str());
    return 1;
  }
  if (!build_stream(on, explicit_spec, tile_info, payloads, &explicit_units,
                    &explicit_parsed, error)) {
    std::fprintf(stderr, "EXPLICIT BUILD FAIL: %s\n", error.c_str());
    return 1;
  }

  const std::uint16_t w16 = static_cast<std::uint16_t>(width);
  const std::uint16_t h16 = static_cast<std::uint16_t>(height);
  const std::string base_path = out_prefix + ".baseline.ivf";
  const std::string found_path = out_prefix + ".found_ref.ivf";
  const std::string explicit_path = out_prefix + ".explicit.ivf";
  if (!write_ivf(base_path, baseline_units, w16, h16, error) ||
      !write_ivf(found_path, found_units, w16, h16, error) ||
      !write_ivf(explicit_path, explicit_units, w16, h16, error)) {
    std::fprintf(stderr, "IVF WRITE FAIL: %s\n", error.c_str());
    return 1;
  }

  std::printf("ref_fields_gate: decoder %s\n", dav2d.c_str());
  std::printf("ref_fields_gate: %dx%d, 2x1 tiles, 3 frames, "
              "TileSizeBytes %d\n",
              width, height, tile_size_bytes);

  std::uint64_t sizes[3] = {0, 0, 0};
  const std::vector<std::vector<std::uint8_t>>* all_units[3] = {
      &baseline_units, &found_units, &explicit_units};
  for (int s = 0; s < 3; ++s) {
    for (const auto& unit : *all_units[s]) sizes[s] += unit.size();
  }
  std::printf("  stream bytes: baseline %llu, found_ref %llu, explicit %llu\n",
              static_cast<unsigned long long>(sizes[0]),
              static_cast<unsigned long long>(sizes[1]),
              static_cast<unsigned long long>(sizes[2]));
  if (feature_mask < 0) {
    check(sizes[1] < sizes[2],
          "found_ref inherits the size and is shorter than the explicit "
          "control");
  }

  // --- field assertions ------------------------------------------------------
  if (feature_mask < 0) {
  check(baseline_parsed.size() == 3, "baseline: parsed three headers");
  check(found_parsed.size() == 3, "found_ref: parsed three headers");
  check(explicit_parsed.size() == 3, "explicit: parsed three headers");
  if (baseline_parsed.size() == 3) {
    const M2RefParsedHeader& f1 = baseline_parsed[1];
    check(!f1.header.frame_size_override,
          "baseline frame 1: frame_size_override = 0");
    check(f1.refresh.refresh_frame_flags == 1,
          "baseline frame 1: refresh mask == 1");
    check(!f1.refresh.has_refresh_frame_flags,
          "baseline frame 1: no short-refresh syntax");
    check(!f1.drl.change_drl, "baseline frame 1: no change_drl");
    check(f1.primary.primary_ref_frame == kPrimaryRefNone,
          "baseline frame 1: no primary_ref_frame (defaults)");
    check(f1.primary.disable_cross_frame_cdf_init,
          "baseline frame 1: CDFs are reset");
  }
  if (found_parsed.size() == 3) {
    const M2RefParsedHeader& f1 = found_parsed[1];
    check(f1.header.frame_size_override, "frame 1: frame_size_override = 1");
    check(f1.size.inherited_from == 0, "frame 1: found_ref selected ref 0");
    check(!f1.size.explicit_size, "frame 1: no explicit frame_size() bits");
    check(f1.size.width == w16 && f1.size.height == h16,
          "frame 1: inherited 128x64");
    check(f1.header.signal_primary_ref_frame,
          "frame 1: signal_primary_ref_frame = 1");
    check(f1.header.primary_ref_frame == 0, "frame 1: primary_ref_frame = 0");
    check(f1.primary.primary_ref_frame == 0,
          "frame 1: primary_ref_frame resolves to 0");
    check(!f1.primary.disable_cross_frame_cdf_init,
          "frame 1: disable_cross_frame_cdf_init = 0");
    check(f1.primary.load_cdfs_from_reference,
          "frame 1: CDFs load from the reference slot");
    check(f1.primary.cdf_reference_slot == 0,
          "frame 1: CDF source is slot 0");
    check(f1.refresh.has_refresh_frame_flags,
          "frame 1: has_refresh_frame_flags = 1");
    check(f1.refresh.frame_to_refresh == 0,
          "frame 1: frame_to_refresh = 0");
    check(f1.refresh.refresh_frame_flags == 1,
          "frame 1: refresh_frame_flags resolves to 1");
    check(f1.drl.change_drl, "frame 1: change_drl = 1");
    check(f1.drl.max_drl_bits_minus_1 == on.seq_max_drl_bits_minus_1 - 1U,
          "frame 1: max_drl_bits_minus_1 changed to 3");

    const M2RefParsedHeader& f2 = found_parsed[2];
    check(f2.header.allow_intrabc, "frame 2: allow_intrabc = 1");
    check(f2.bvp_drl.change_bvp_drl, "frame 2: change_bvp_drl = 1");
    check(f2.bvp_drl.max_bvp_drl_bits_minus_1 == 1,
          "frame 2: max_bvp_drl_bits_minus_1 = 1");
  }
  if (explicit_parsed.size() == 3) {
    const M2RefParsedHeader& f1 = explicit_parsed[1];
    check(f1.header.frame_size_override,
          "control frame 1: frame_size_override = 1");
    check(f1.size.explicit_size, "control frame 1: explicit frame_size()");
    check(f1.size.inherited_from == kNoRef,
          "control frame 1: no found_ref");
    check(!f1.size.found_ref[0], "control frame 1: found_ref[0] = 0");
    check(f1.size.width == w16 && f1.size.height == h16,
          "control frame 1: explicit 128x64");
  }
  }  // feature_mask < 0

  // --- decoder gate ----------------------------------------------------------
  struct DecodeCase {
    const char* label;
    std::string path;
  } decode_cases[3] = {{"baseline", base_path},
                       {"found_ref", found_path},
                       {"explicit", explicit_path}};
  bool ok = true;
  for (const DecodeCase& item : decode_cases) {
    const std::string decoded = item.path + ".decoded.y4m";
    if (!run_decoder(dav2d, item.path, decoded, error)) {
      check(false, std::string(item.label) + ": " + error);
      ok = false;
      continue;
    }
    if (!check_constant_128(decoded, 3, width, height, error)) {
      check(false, std::string(item.label) + ": " + error);
      ok = false;
      continue;
    }
    std::printf("  %-10s decoded 3 frames, all samples 128\n", item.label);
  }

  if (ok) {
    const std::string a = base_path + ".decoded.y4m";
    const std::string b = found_path + ".decoded.y4m";
    const std::string c = explicit_path + ".decoded.y4m";
    check(std::filesystem::file_size(a) == std::filesystem::file_size(b) &&
              std::filesystem::file_size(a) == std::filesystem::file_size(c),
          "decoded sizes agree");
    const auto read_all = [](const std::string& path) {
      std::ifstream file(path, std::ios::binary);
      return std::vector<std::uint8_t>(
          (std::istreambuf_iterator<char>(file)),
          std::istreambuf_iterator<char>());
    };
    const std::vector<std::uint8_t> ba = read_all(a);
    const std::vector<std::uint8_t> bb = read_all(b);
    const std::vector<std::uint8_t> bc = read_all(c);
    check(ba == bb, "found_ref decode is byte-identical to the baseline");
    check(ba == bc, "explicit control decode is byte-identical to the baseline");
  }

  std::printf("ref_fields_gate: %d checks, %d failure(s)\n", g_checks,
              g_failures);
  std::printf("%s\n", (ok && g_failures == 0) ? "REF FIELDS GATE PASS"
                                              : "REF FIELDS GATE FAIL");
  return (ok && g_failures == 0) ? 0 : 1;
}
