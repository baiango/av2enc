// dav2d gate for frames split across multiple M2 tile-group OBUs.
//
// Four streams are produced and decoded with
//     dav2d --strict 1 --threads 1
// at 640x360 and 1920x1080:
//   * 4x1 tiles split 2+2 (second group has frame_header_present_flag = 0,
//     i.e. the frame header is shared);
//   * 4x1 tiles split 1+2+1 (the two later groups carry a bit-identical
//     frame_header_copy(), §5.18.1).
//
// Each stream is a keyframe followed by a P frame. Every superblock is one
// 64x64 leaf with no residual; the keyframe uses DC_PRED with unavailable
// neighbours (DC = 1 << 7 = 128) and the P frame is GLOBALMV + skip (zero MV
// copy). The encoder reconstruction is therefore provably constant 128 in Y, U
// and V for both frames, and the gate checks every decoded sample against it.
// All loop filters are off, so no tile-edge filtering can perturb the result
// (see docs/negative-results.md).
//
// The tile-group framing is validated by the independent reader before the
// stream is handed to the decoder; the reader re-derives tg_start/tg_end,
// bru_tile_active (not coded here: use_bru = 0), size fields and header copies
// from the produced bytes.
//
// Returns 77 when no decoder is available (CTest-style skip).
#include "encoder/m2/m2_headers.h"
#include "encoder/m2/m2_syntax.h"
#include "encoder/m2/post/tiled_frame.h"
#include "encoder/m2/tile_groups/tile_groups.h"

#include "bitstream/m0_headers.h"
#include "io/ivf.h"
#include "io/y4m.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using av2enc::m2::syntax::BlockSyntax;
using av2enc::m2::tile_groups::FrameHeaderBits;
using av2enc::m2::tile_groups::FrameTileGroupPlan;
using av2enc::m2::tile_groups::ParsedTileGroup;
using av2enc::m2::tile_groups::TileGroupGeometry;
using av2enc::m2::tile_groups::TileGroupSequence;
using av2enc::m2::tile_groups::TileGroupSpan;

BlockSyntax make_root(int mi_row, int mi_col, bool inter) {
  BlockSyntax node;
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
  node.tx_size[0] = 4;  // 64x64
  node.tx_size[1] = 3;  // 32x32 chroma
  node.tx_size[2] = 3;
  node.eob[0] = node.eob[1] = node.eob[2] = 0;
  node.levels_offset = 0;
  return node;
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

bool check_decoded_constant(const std::string& path, int width, int height,
                            int expected_frames, std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  int frames = 0;
  long mismatches = 0;
  int max_diff = 0;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    if (frame.width() != static_cast<std::uint32_t>(width) ||
        frame.height() != static_cast<std::uint32_t>(height)) {
      error = "decoded frame has the wrong dimensions";
      return false;
    }
    ++frames;
    const auto check_plane = [&](std::span<const std::uint8_t> plane) {
      for (std::uint8_t sample : plane) {
        const int diff = std::abs(static_cast<int>(sample) - 128);
        if (diff != 0) {
          ++mismatches;
          max_diff = std::max(max_diff, diff);
        }
      }
    };
    check_plane(frame.y());
    check_plane(frame.u());
    check_plane(frame.v());
  }
  if (frames != expected_frames) {
    error = "decoder produced " + std::to_string(frames) + " frames, expected " +
            std::to_string(expected_frames);
    return false;
  }
  if (mismatches != 0) {
    error = "recon mismatch against the constant DC/GLOBALMV reconstruction: " +
            std::to_string(mismatches) + " samples, max_diff " +
            std::to_string(max_diff);
    return false;
  }
  return true;
}

bool make_header_bits(const av2enc::m2::M2Config& config,
                      const av2enc::m2::post::M2PostSequenceOptions& sequence,
                      const av2enc::m2::post::M2TiledFrameHeader& header,
                      const av2enc::tiles::TileLayout& layout,
                      int tile_size_bytes, FrameHeaderBits* out,
                      std::string& error) {
  av2enc::BitWriter writer;
  std::size_t bits = 0;
  if (!av2enc::m2::post::write_m2_frame_header_syntax(
          writer, config, sequence, header, layout, tile_size_bytes, &bits,
          &error)) {
    return false;
  }
  const auto bytes = writer.bytes();
  std::string capture_error;
  if (!av2enc::m2::tile_groups::capture_frame_header_bits(
          bytes, bits, out, &capture_error)) {
    error = "capture frame header: " + capture_error;
    return false;
  }
  return true;
}

// Re-reads the produced group payloads and checks that the framing the writer
// intended is what the bytes say.
bool verify_framing(const av2enc::m2::post::M2TiledFrame& frame,
                    const TileGroupGeometry& geometry,
                    std::span<const std::vector<std::uint8_t>> group_payloads,
                    const std::vector<TileGroupSpan>& planned,
                    std::string& error) {
  TileGroupSequence parsed;
  if (!av2enc::m2::tile_groups::read_tile_group_sequence(
          group_payloads, geometry, frame.frame_header_bits, &parsed, &error)) {
    return false;
  }
  if (parsed.groups.size() != planned.size()) {
    error = "group count differs from the plan";
    return false;
  }
  for (std::size_t i = 0; i < parsed.groups.size(); ++i) {
    const ParsedTileGroup& group = parsed.groups[i];
    if (group.tg_start != planned[i].tg_start ||
        group.tg_end != planned[i].tg_end) {
      error = "tg_start/tg_end differ from the plan";
      return false;
    }
    if (group.tile_bytes.size() !=
        static_cast<std::size_t>(planned[i].tile_count())) {
      error = "tile count differs from the plan";
      return false;
    }
    for (int tile = group.tg_start; tile <= group.tg_end; ++tile) {
      const std::size_t local = static_cast<std::size_t>(tile - group.tg_start);
      if (group.tile_bytes[local] !=
          frame.tile_bytes[static_cast<std::size_t>(tile)]) {
        error = "tile size differs from the encoded payload";
        return false;
      }
      const std::size_t offset = group.tile_offsets[local];
      const std::size_t size = group.tile_bytes[local];
      const auto& expected =
          frame.tile_payloads[static_cast<std::size_t>(tile)];
      if (!std::equal(expected.begin(), expected.end(),
                      group_payloads[i].begin() +
                          static_cast<std::ptrdiff_t>(offset))) {
        error = "tile payload bytes differ from the encoded payload";
        return false;
      }
      (void)size;
    }
  }
  return true;
}

bool run_case(int width, int height, int cols, int rows,
              std::span<const int> tiles_per_group, bool copy_header,
              const std::string& dav2d, const std::string& out_path) {
  std::string error;
  av2enc::m2::M2Config config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = 90;

  const auto tile_plan =
      av2enc::m2::post::plan_m2_tiles(width, height, cols, rows);
  if (!tile_plan.ok) {
    std::cerr << "TILE PLAN FAIL: " << tile_plan.error << "\n";
    return false;
  }
  const auto& layout = tile_plan.layout;

  av2enc::m2::post::M2PostSequenceOptions sequence;
  // All filters stay off so the reconstruction is exactly the prediction
  // (constant 128) and no tile-edge filtering can perturb the recon compare.
  // The sequence bit is still signalled the multi-tile-safe way.
  sequence.filters.disable_loopfilters_across_tiles = true;

  const auto seq_obu =
      av2enc::m2::post::make_m2_sequence_header_obu(config, sequence, &error);
  if (seq_obu.empty()) {
    std::cerr << "SEQ FAIL: " << error << "\n";
    return false;
  }

  std::vector<std::int32_t> levels;
  std::vector<BlockSyntax> key_blocks;
  std::vector<BlockSyntax> inter_blocks;
  for (int sb_row = 0; sb_row < layout.sb_rows; ++sb_row) {
    for (int sb_col = 0; sb_col < layout.sb_cols; ++sb_col) {
      key_blocks.push_back(make_root(sb_row * 16, sb_col * 16, false));
      inter_blocks.push_back(make_root(sb_row * 16, sb_col * 16, true));
    }
  }

  av2enc::m2::post::M2TiledFrameHeader key_header;
  key_header.keyframe = true;
  key_header.order_hint = 0;
  av2enc::m2::post::M2TiledFrameHeader inter_header;
  inter_header.keyframe = false;
  inter_header.order_hint = 1;
  inter_header.ref_slot = 0;
  inter_header.refresh_frame_flags = 1;

  av2enc::m2::post::M2TiledFrame key_frame;
  av2enc::m2::post::M2TiledFrame inter_frame;
  if (!av2enc::m2::post::assemble_m2_tiled_frame(
          config, sequence, key_header, layout, key_blocks, levels, &key_frame,
          &error)) {
    std::cerr << "KEY FRAME FAIL: " << error << "\n";
    return false;
  }
  if (!av2enc::m2::post::assemble_m2_tiled_frame(
          config, sequence, inter_header, layout, inter_blocks, levels,
          &inter_frame, &error)) {
    std::cerr << "INTER FRAME FAIL: " << error << "\n";
    return false;
  }

  const auto emit_frame = [&](const av2enc::m2::post::M2TiledFrameHeader& fh,
                              const av2enc::m2::post::M2TiledFrame& frame,
                              std::vector<std::vector<std::uint8_t>>* obus,
                              std::string* err) -> bool {
    const TileGroupGeometry geometry =
        av2enc::m2::tile_groups::geometry_from_layout(
            layout, frame.tile_size_bytes, /*use_bru=*/false,
            /*bru_inactive=*/false);

    FrameHeaderBits header_bits;
    if (!make_header_bits(config, sequence, fh, layout, frame.tile_size_bytes,
                          &header_bits, *err)) {
      return false;
    }
    if (header_bits.bit_count != frame.frame_header_bits) {
      *err = "frame header bit count differs from the tile-group writer";
      return false;
    }

    std::vector<TileGroupSpan> groups;
    if (!av2enc::m2::tile_groups::plan_contiguous_tile_groups(
            geometry, tiles_per_group, copy_header, /*bru_active_default=*/false,
            &groups, err)) {
      return false;
    }
    FrameTileGroupPlan plan;
    plan.geometry = geometry;
    plan.frame_header = header_bits;
    plan.copy_frame_header = copy_header;
    plan.groups = groups;

    // Compatibility: a one-group plan must reproduce the existing single-group
    // writer byte for byte (`post::assemble_m2_tile_group_from_payloads`).
    {
      std::vector<TileGroupSpan> single;
      if (!av2enc::m2::tile_groups::plan_contiguous_tile_groups(
              geometry, std::vector<int>{geometry.num_tiles}, false, false,
              &single, err)) {
        return false;
      }
      FrameTileGroupPlan single_plan;
      single_plan.geometry = geometry;
      single_plan.frame_header = header_bits;
      single_plan.groups = single;
      std::vector<std::vector<std::uint8_t>> single_payloads;
      if (!av2enc::m2::tile_groups::write_frame_tile_groups(
              single_plan, frame.tile_payloads, &single_payloads, err)) {
        return false;
      }
      const av2enc::m0::ObuType parity_type =
          fh.keyframe ? av2enc::m0::ObuType::kClosedLoopKey
                      : av2enc::m0::ObuType::kRegularTileGroup;
      const auto parity_obu = av2enc::m0::make_obu(
          av2enc::m0::ObuHeader{parity_type}, single_payloads[0]);
      if (parity_obu != frame.obu) {
        *err = "one-group framing differs from the existing single-group writer";
        return false;
      }
    }

    std::vector<std::vector<std::uint8_t>> payloads;
    if (!av2enc::m2::tile_groups::write_frame_tile_groups(
            plan, frame.tile_payloads, &payloads, err)) {
      return false;
    }
    if (!verify_framing(frame, geometry, payloads, groups, *err)) return false;

    const av2enc::m0::ObuType type =
        fh.keyframe ? av2enc::m0::ObuType::kClosedLoopKey
                    : av2enc::m0::ObuType::kRegularTileGroup;
    for (const auto& payload : payloads) {
      obus->push_back(
          av2enc::m0::make_obu(av2enc::m0::ObuHeader{type}, payload));
    }
    return true;
  };

  std::vector<std::vector<std::uint8_t>> key_obus;
  std::vector<std::vector<std::uint8_t>> inter_obus;
  if (!emit_frame(key_header, key_frame, &key_obus, &error)) {
    std::cerr << "KEY FRAMING FAIL: " << error << "\n";
    return false;
  }
  if (!emit_frame(inter_header, inter_frame, &inter_obus, &error)) {
    std::cerr << "INTER FRAMING FAIL: " << error << "\n";
    return false;
  }

  std::vector<std::vector<std::uint8_t>> units;
  {
    const auto td = av2enc::m2::make_temporal_delimiter_obu();
    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), seq_obu.begin(), seq_obu.end());
    for (const auto& obu : key_obus) unit.insert(unit.end(), obu.begin(), obu.end());
    units.push_back(std::move(unit));
  }
  {
    const auto td = av2enc::m2::make_temporal_delimiter_obu();
    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), td.begin(), td.end());
    for (const auto& obu : inter_obus) unit.insert(unit.end(), obu.begin(), obu.end());
    units.push_back(std::move(unit));
  }

  if (!write_ivf(out_path, units, static_cast<std::uint16_t>(width),
                 static_cast<std::uint16_t>(height), error)) {
    std::cerr << "IVF FAIL: " << error << "\n";
    return false;
  }

  std::cout << "  " << width << "x" << height << " " << cols << "x" << rows
            << " tiles, groups=";
  for (std::size_t i = 0; i < tiles_per_group.size(); ++i) {
    std::cout << (i != 0 ? "+" : "") << tiles_per_group[i];
  }
  std::cout << (copy_header ? " header_copy" : " shared_header")
            << " key_tile_bytes=";
  for (std::size_t i = 0; i < key_frame.tile_bytes.size(); ++i) {
    std::cout << (i != 0 ? "/" : "") << key_frame.tile_bytes[i];
  }
  std::cout << " tile_size_bytes=" << key_frame.tile_size_bytes << "\n";

  if (dav2d.empty()) return true;
  const std::string decoded = out_path + ".decoded.y4m";
  const std::string command = dav2d + " -q -i " + out_path + " -o " + decoded +
                              " --strict 1 --threads 1";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    std::cerr << "DECODE FAIL: dav2d returned " << rc << "\n";
    return false;
  }
  if (!check_decoded_constant(decoded, width, height, 2, error)) {
    std::cerr << "DECODE CHECK FAIL: " << error << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_path = "m2_tile_groups_gate.ivf";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--help") {
      std::cout << "usage: tile_groups_gate [--out out.ivf]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }

  const std::string dav2d = find_dav2d();
  if (dav2d.empty()) {
    std::cout << "SKIP: no dav2d binary found (set DAV2D)\n";
    return 77;
  }

  const std::vector<int> two_groups = {2, 2};
  const std::vector<int> three_groups = {1, 2, 1};

  const bool pass_640_2 =
      run_case(640, 360, 4, 1, two_groups, /*copy_header=*/false, dav2d,
               out_path + ".640x360.2group.ivf");
  const bool pass_640_3 =
      run_case(640, 360, 4, 1, three_groups, /*copy_header=*/true, dav2d,
               out_path + ".640x360.3group.ivf");
  const bool pass_hd_2 =
      run_case(1920, 1080, 4, 1, two_groups, /*copy_header=*/false, dav2d,
               out_path + ".1080p.2group.ivf");
  const bool pass_hd_3 =
      run_case(1920, 1080, 4, 1, three_groups, /*copy_header=*/true, dav2d,
               out_path + ".1080p.3group.ivf");

  if (!pass_640_2 || !pass_640_3 || !pass_hd_2 || !pass_hd_3) {
    std::cout << "M2 MULTI-TILE-GROUP GATE FAIL\n";
    return 1;
  }
  std::cout << "PASS: 4 multi-tile-group streams decoded (dav2d --strict 1 "
               "--threads 1), decoded samples match the constant "
               "DC/GLOBALMV reconstruction exactly\n";
  return 0;
}
