// dav2d gate for tiled two-frame M2 streams.
//
// The M2 core gate is single-tile and compares reconstruction sample by sample.
// This gate exercises the layer in tiled_frame.h instead:
//   * a keyframe followed by a P frame;
//   * uniform tiles, including a case with multiple superblocks per tile;
//   * every superblock is one intra/inter leaf with zero residual;
//   * the stream is assembled from per-tile payloads and a multi-tile frame
//     header (context_update_tile_id, TileSizeBytes, tile_size_minus_1);
//   * dav2d must decode exactly two frames at the requested size.
//
// The source is implicit: a keyframe DC predictor with no available neighbours
// is 1 << (BitDepth - 1) = 128, and the P frame is GLOBALMV (zero MV) copy.
// With all loop filters off the decoded pictures must therefore be a constant
// 128 in Y, U and V.  That turns "it decodes" into a sample-level check without
// needing an encoder reconstruction.
//
// Returns 77 when no decoder is available (CTest-style skip).
#include "pipeline/video/video_headers.h"
#include "pipeline/video/video_syntax.h"
#include "pipeline/video/post/tiled_frame.h"

#include "core/io/ivf.h"
#include "core/io/y4m.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using av2enc::video::syntax::BlockSyntax;

BlockSyntax make_root(int mi_row, int mi_col, bool inter) {
  BlockSyntax node;
  node.mi_row = static_cast<std::uint16_t>(mi_row);
  node.mi_col = static_cast<std::uint16_t>(mi_col);
  node.bsize = 12;     // BLOCK_64X64
  node.partition = 0;  // PARTITION_NONE
  node.allowed_mask = 1U << 0;
  node.flags = av2enc::video::syntax::kFlagSbRoot;
  node.subtree = 0;
  node.is_inter = inter ? 1 : 0;
  node.skip_flag = inter ? 1 : 0;
  node.inter_mode = av2enc::video::syntax::kInterGlobalMv;
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

bool check_decoded(const std::string& path, int width, int height,
                   std::string& error) {
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
  if (frames != 2) {
    error = "decoder produced " + std::to_string(frames) +
            " frames, expected 2";
    return false;
  }
  if (mismatches != 0) {
    error = "decoded samples differ from the DC/GLOBALMV constant: " +
            std::to_string(mismatches) + " mismatches, max_diff " +
            std::to_string(max_diff);
    return false;
  }
  return true;
}

// One 2-frame stream with the requested geometry and uniform tile grid.
bool run_case(int width, int height, int cols, int rows,
              const std::string& dav2d, const std::string& out_path) {
  std::string error;
  av2enc::video::VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = 90;

  const auto tile_plan =
      av2enc::video::post::plan_video_tiles(width, height, cols, rows);
  if (!tile_plan.ok) {
    std::cerr << "TILE PLAN FAIL: " << tile_plan.error << "\n";
    return false;
  }
  const auto& layout = tile_plan.layout;

  av2enc::video::post::VideoPostSequenceOptions sequence;
  // Enable the tools at sequence level and disable them per frame.  That
  // exercises the full §5.4.10 filter row plus the cdef_params()/ccso_params()
  // frame rows without needing their per-block tile symbols (which the current
  // M2 syntax writer does not emit).
  sequence.filters.enable_cdef = true;
  sequence.filters.enable_ccso = true;
  sequence.filters.ccso_unit_matches_sb_size = true;
  const auto seq_obu =
      av2enc::video::post::make_video_sequence_header_obu(config, sequence, &error);
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

  av2enc::video::post::VideoTiledFrameHeader key_header;
  key_header.keyframe = true;
  key_header.order_hint = 0;
  av2enc::video::post::VideoTiledFrameHeader inter_header;
  inter_header.keyframe = false;
  inter_header.order_hint = 1;
  inter_header.ref_slot = 0;
  inter_header.refresh_frame_flags = 1;
  // Exercise deblocking_filter_params() in front of the real decoder.  The
  // pictures are constant, so deblocking changes no samples; a wrong bit order
  // still desynchronises the header.
  for (int plane = 0; plane < 4; ++plane) {
    key_header.apply_deblocking_filter[plane] = true;
    inter_header.apply_deblocking_filter[plane] = true;
  }

  av2enc::video::post::VideoTiledFrame key_frame;
  av2enc::video::post::VideoTiledFrame inter_frame;
  if (!av2enc::video::post::assemble_video_tiled_frame(
          config, sequence, key_header, layout, key_blocks, levels, &key_frame,
          &error)) {
    std::cerr << "KEY FRAME FAIL: " << error << "\n";
    return false;
  }
  if (!av2enc::video::post::assemble_video_tiled_frame(
          config, sequence, inter_header, layout, inter_blocks, levels,
          &inter_frame, &error)) {
    std::cerr << "INTER FRAME FAIL: " << error << "\n";
    return false;
  }

  std::vector<std::vector<std::uint8_t>> units;
  {
    const auto td = av2enc::video::make_temporal_delimiter_obu();
    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), seq_obu.begin(), seq_obu.end());
    unit.insert(unit.end(), key_frame.obu.begin(), key_frame.obu.end());
    units.push_back(std::move(unit));
  }
  {
    const auto td = av2enc::video::make_temporal_delimiter_obu();
    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), inter_frame.obu.begin(), inter_frame.obu.end());
    units.push_back(std::move(unit));
  }

  if (!write_ivf(out_path, units, static_cast<std::uint16_t>(width),
                 static_cast<std::uint16_t>(height), error)) {
    std::cerr << "IVF FAIL: " << error << "\n";
    return false;
  }
  std::cout << "  " << width << "x" << height << " " << cols << "x" << rows
            << " tiles: key_tile_bytes=";
  for (std::size_t i = 0; i < key_frame.tile_bytes.size(); ++i) {
    std::cout << (i != 0 ? "/" : "") << key_frame.tile_bytes[i];
  }
  std::cout << " tile_size_bytes=" << key_frame.tile_size_bytes
            << " inter_tile_bytes=";
  for (std::size_t i = 0; i < inter_frame.tile_bytes.size(); ++i) {
    std::cout << (i != 0 ? "/" : "") << inter_frame.tile_bytes[i];
  }
  std::cout << "\n";

  if (dav2d.empty()) return true;
  const std::string decoded = out_path + ".decoded.y4m";
  const std::string command =
      dav2d + " -q -i " + out_path + " -o " + decoded + " --threads 0";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    std::cerr << "DECODE FAIL: dav2d returned " << rc << "\n";
    return false;
  }
  if (!check_decoded(decoded, width, height, error)) {
    std::cerr << "DECODE CHECK FAIL: " << error << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_path = "video_tiled_post_gate.ivf";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--help") {
      std::cout << "usage: tiled_gate [--out out.ivf]\n";
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

  const bool one_sb_per_tile =
      run_case(128, 64, 2, 1, dav2d, out_path + ".2x1.ivf");
  const bool two_sb_per_tile =
      run_case(256, 64, 2, 1, dav2d, out_path + ".4sb.ivf");
  const bool four_tiles =
      run_case(256, 128, 2, 2, dav2d, out_path + ".2x2.ivf");

  if (!one_sb_per_tile || !two_sb_per_tile || !four_tiles) {
    std::cout << "M2 TILED POST GATE FAIL\n";
    return 1;
  }
  std::cout << "PASS: 3 tiled 2-frame streams decoded, all samples are the "
               "expected DC/GLOBALMV constant\n";
  return 0;
}
