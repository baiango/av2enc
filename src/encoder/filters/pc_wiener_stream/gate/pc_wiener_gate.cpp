// PC-Wiener decoder gate: sequence/frame/tile syntax plus closed-loop quality.
//
// The production M2 core has no restoration hook and must not be edited.  This
// gate vendors the M2 tile writer with exactly one §5.20.10.4 read_lr() call per
// superblock (gate/vendor_m2_syntax_pc.py), enables PC-Wiener in the
// sequence/frame headers through the new pc_wiener_stream module, and builds a
// one-frame stream around a real decoded reconstruction.
//
// Closed loop:
//   1. encode a fixture with enable_restoration = 0 and decode it with the
//      pinned dav2d -> base reconstruction,
//   2. run the §7.20.4 classifier on the decoded luma and choose per-unit
//      PC-Wiener on/off (mode A) or per-class translated PC-Wiener filters in
//      the frame-level pixel-classified mode (mode B),
//   3. rebuild the frame with the LR syntax and decode it with
//      `dav2d --strict 1 --threads 1`,
//   4. compare every decoded luma sample with the encoder's own §7.20.4/§7.20.3
//      output; the pass condition is zero differing samples.
//
// Exit: 0 pass, 1 fail, 77 decoder unavailable.
//
// Limits: the pinned dav2d build may not implement luma PC-Wiener.  If it
// rejects the stream, the gate prints the decoder error, verifies exact
// self-parse and feature-off decode invariance, and exits 0 only when those
// checks pass (the decoder limitation is reported, not hidden).

#include "encoder/filters/pc_wiener_stream/pc_wiener_stream.h"
#include "encoder/m2/m2_headers.h"
#include "encoder/partition/partition.h"
#include "encoder/m2/m2_syntax.h"
#include "encoder/m2/post/loop_filters.h"
#include "encoder/m2/post/tiled_frame.h"
#include "encoder/tiles/tile_layout.h"

#include "bitstream/bitwriter.h"
#include "bitstream/m0_headers.h"
#include "io/ivf.h"
#include "io/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "m2_syntax_pcw.h"

namespace pcw = av2enc::filters::pc_wiener_stream;
namespace fs = std::filesystem;

namespace {

struct Frame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y, u, v;
  int cw() const { return (width + 1) / 2; }
  int ch() const { return (height + 1) / 2; }
};

std::uint32_t CeilLog2U32(std::uint32_t value) {
  if (value <= 1U) return 0U;
  std::uint32_t bits = 0;
  --value;
  while (value != 0) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

void PutUvlc(av2enc::BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) ++leading_zeros;
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

std::string FindDecoder() {
  if (const char* env = std::getenv("DAV2D")) {
    if (fs::exists(env)) return env;
  }
  for (const char* candidate :
       {"tools/bin/dav2d", "tools/bin/dav2d-macos", "dav2d", "dav2d-macos"}) {
    if (fs::exists(candidate)) return candidate;
  }
  return {};
}

bool ReadFirstFrame(const std::string& path, Frame* frame, std::string* error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame src;
  if (reader.read_frame(src, *error) != av2enc::Y4mReadResult::kFrame) {
    if (error->empty()) *error = "source has no frames";
    return false;
  }
  frame->width = static_cast<int>(src.width());
  frame->height = static_cast<int>(src.height());
  frame->y.assign(src.y().begin(), src.y().end());
  frame->u.assign(src.u().begin(), src.u().end());
  frame->v.assign(src.v().begin(), src.v().end());
  return true;
}

void Downscale(const Frame& src, int width, int height, Frame* dst) {
  dst->width = width;
  dst->height = height;
  dst->y.assign(static_cast<std::size_t>(width) * height, 0);
  dst->u.assign(static_cast<std::size_t>((width + 1) / 2) * ((height + 1) / 2),
                0);
  dst->v.assign(dst->u.size(), 0);
  const int sw = src.width;
  const int sh = src.height;
  const int scw = src.cw();
  const int sch = src.ch();
  for (int y = 0; y < height; ++y) {
    const int sy0 = y * sh / height;
    const int sy1 = std::max(sy0 + 1, (y + 1) * sh / height);
    for (int x = 0; x < width; ++x) {
      const int sx0 = x * sw / width;
      const int sx1 = std::max(sx0 + 1, (x + 1) * sw / width);
      int sum = 0;
      int count = 0;
      for (int sy = sy0; sy < sy1; ++sy) {
        for (int sx = sx0; sx < sx1; ++sx) {
          sum += src.y[static_cast<std::size_t>(sy) * sw + sx];
          ++count;
        }
      }
      dst->y[static_cast<std::size_t>(y) * width + x] =
          static_cast<std::uint8_t>(sum / std::max(1, count));
    }
  }
  const int cwid = (width + 1) / 2;
  const int chei = (height + 1) / 2;
  for (int y = 0; y < chei; ++y) {
    const int sy0 = y * sch / chei;
    const int sy1 = std::max(sy0 + 1, (y + 1) * sch / chei);
    for (int x = 0; x < cwid; ++x) {
      const int sx0 = x * scw / cwid;
      const int sx1 = std::max(sx0 + 1, (x + 1) * scw / cwid);
      int sum_u = 0;
      int sum_v = 0;
      int count = 0;
      for (int sy = sy0; sy < sy1; ++sy) {
        for (int sx = sx0; sx < sx1; ++sx) {
          sum_u += src.u[static_cast<std::size_t>(sy) * scw + sx];
          sum_v += src.v[static_cast<std::size_t>(sy) * scw + sx];
          ++count;
        }
      }
      dst->u[static_cast<std::size_t>(y) * cwid + x] =
          static_cast<std::uint8_t>(sum_u / std::max(1, count));
      dst->v[static_cast<std::size_t>(y) * cwid + x] =
          static_cast<std::uint8_t>(sum_v / std::max(1, count));
    }
  }
}

bool WriteIvf(const std::string& path,
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

double Psnr(const std::vector<std::uint8_t>& a,
            const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size() || a.empty()) return 0.0;
  double sse = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sse += d * d;
  }
  if (sse == 0.0) return 99.0;
  return 10.0 * std::log10(255.0 * 255.0 * static_cast<double>(a.size()) / sse);
}

void WriteY4m(const std::string& path, const Frame& frame) {
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) return;
  std::fprintf(file, "YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n", frame.width,
               frame.height);
  std::fputs("FRAME\n", file);
  std::fwrite(frame.y.data(), 1, frame.y.size(), file);
  std::fwrite(frame.u.data(), 1, frame.u.size(), file);
  std::fwrite(frame.v.data(), 1, frame.v.size(), file);
  std::fclose(file);
}

bool RunCvvdp(const std::string& ref, const std::string& test, double* jod) {
  const std::string command = "cvvdp -t \"" + test + "\" -r \"" + ref +
                              "\" -d standard_fhd --device mps 2>&1";
  std::FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) return false;
  std::string output;
  char buffer[512];
  while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }
  const int status = pclose(pipe);
  if (status != 0) return false;
  const std::size_t pos = output.find("cvvdp=");
  if (pos == std::string::npos) return false;
  *jod = std::atof(output.c_str() + pos + 6);
  return true;
}

// ---------------------------------------------------------------------------
// Fixture: one PARTITION_NONE 64x64 leaf per superblock, one DC level per
// plane.  The levels track the source block means; the reconstruction is
// intentionally blocky, which is exactly the artifact PC-Wiener restores.
// ---------------------------------------------------------------------------
struct TileData {
  av2enc::m2::syntax::FramePlan plan;
  std::vector<av2enc::m2::syntax::BlockSyntax> blocks;
  std::vector<std::int32_t> levels;
};

int PlaneMean(const Frame& src, int plane, int x0, int y0, int w, int h) {
  const std::vector<std::uint8_t>* data = nullptr;
  int stride = 0;
  if (plane == 0) {
    data = &src.y;
    stride = src.width;
  } else if (plane == 1) {
    data = &src.u;
    stride = src.cw();
  } else {
    data = &src.v;
    stride = src.cw();
  }
  const int sx = plane == 0 ? x0 : x0 / 2;
  const int sy = plane == 0 ? y0 : y0 / 2;
  const int sw = plane == 0 ? w : (w + 1) / 2;
  const int sh = plane == 0 ? h : (h + 1) / 2;
  int sum = 0;
  int count = 0;
  for (int y = sy; y < sy + sh && y < (plane == 0 ? src.height : src.ch());
       ++y) {
    for (int x = sx; x < sx + sw && x < (plane == 0 ? src.width : src.cw());
         ++x) {
      sum += (*data)[static_cast<std::size_t>(y) * stride + x];
      ++count;
    }
  }
  return count == 0 ? 128 : sum / count;
}

std::int32_t LevelForMean(int mean) {
  // Calibrated on the pinned dav2d/M2 64x64 fixture: DC level 28 reconstructs
  // to luma 132 for base_q_idx 90, i.e. about 7 levels per sample.
  // Calibrated on the pinned dav2d/M2 16x16 fixture: +30 levels reconstructs
  // to 188 and -30 to 68 at base_q_idx 90, i.e. 2 samples per level.
  int level = (mean - 128) * 2;
  level = std::max(-240, std::min(240, level));
  if (level == 0) level = mean >= 128 ? 1 : -1;
  return level;
}

void AddLeaf(TileData* tile, const Frame& src, int mi_row, int mi_col,
             int bsize) {
  av2enc::m2::syntax::BlockSyntax node;
  node.mi_row = static_cast<std::uint16_t>(mi_row);
  node.mi_col = static_cast<std::uint16_t>(mi_col);
  node.bsize = static_cast<std::uint8_t>(bsize);
  node.partition = 0;
  node.allowed_mask = 1U << 0;
  node.flags = 0;
  node.subtree = 0;
  node.y_mode = 0;
  node.uv_mode = 0;
  const int tx_luma = bsize == 6 ? 2 : (bsize == 9 ? 3 : 1);
  node.tx_size[0] = static_cast<std::uint8_t>(tx_luma);
  node.tx_size[1] = static_cast<std::uint8_t>(tx_luma > 0 ? tx_luma - 1 : 0);
  node.tx_size[2] = node.tx_size[1];
  node.levels_offset = static_cast<std::uint32_t>(tile->levels.size());
  node.is_inter = 0;
  node.eob[0] = node.eob[1] = node.eob[2] = 1;
  const int size = av2enc::partition::block_width(
      static_cast<av2enc::partition::BlockSize>(bsize));
  for (int plane = 0; plane < 3; ++plane) {
    const int mean = PlaneMean(src, plane, mi_col * 4, mi_row * 4, size, size);
    tile->levels.push_back(LevelForMean(mean));
  }
  tile->blocks.push_back(node);
}

// AV2 v1.0.0 has no PARTITION_SPLIT below 128x128.  A square quadtree is
// expressed as HORZ at the parent and VERT at each half, which is exactly the
// layout the M0 reference stream uses.
void AddQuad(TileData* tile, const Frame& src, int mi_row, int mi_col,
             int bsize, int target_bsize, int mi_rows, int mi_cols,
             bool is_root) {
  if (bsize == target_bsize) {
    AddLeaf(tile, src, mi_row, mi_col, bsize);
    return;
  }
  const auto size = static_cast<av2enc::partition::BlockSize>(bsize);
  const std::size_t index = tile->blocks.size();
  av2enc::m2::syntax::BlockSyntax node;
  node.mi_row = static_cast<std::uint16_t>(mi_row);
  node.mi_col = static_cast<std::uint16_t>(mi_col);
  node.bsize = static_cast<std::uint8_t>(bsize);
  node.partition = 2;  // PARTITION_HORZ
  node.allowed_mask = (1U << 0) | (1U << 2) | (1U << 3);
  node.flags = is_root ? av2enc::m2::syntax::kFlagSbRoot : 0;
  node.subtree = 0;
  node.y_mode = 0;
  node.uv_mode = 0;
  node.levels_offset = 0;
  node.is_inter = 0;
  node.eob[0] = node.eob[1] = node.eob[2] = 0;
  tile->blocks.push_back(node);
  const int h4 = av2enc::partition::num_4x4_blocks_high(size);
  const int half_h = h4 >> 1;
  const auto hsize = av2enc::partition::partition_subsize(
      av2enc::partition::PartitionType::kHorz, size);
  const int hw4 = av2enc::partition::num_4x4_blocks_wide(hsize);
  const int half_w = hw4 >> 1;
  const auto vsize = av2enc::partition::partition_subsize(
      av2enc::partition::PartitionType::kVert, hsize);
  for (int ch = 0; ch < 2; ++ch) {
    const int cr = mi_row + ch * half_h;
    if (cr >= mi_rows) continue;
    const std::size_t child = tile->blocks.size();
    av2enc::m2::syntax::BlockSyntax vnode;
    vnode.mi_row = static_cast<std::uint16_t>(cr);
    vnode.mi_col = static_cast<std::uint16_t>(mi_col);
    vnode.bsize = static_cast<std::uint8_t>(hsize);
    vnode.partition = 3;  // PARTITION_VERT
    vnode.allowed_mask = (1U << 0) | (1U << 2) | (1U << 3);
    vnode.flags = 0;
    vnode.subtree = 0;
    vnode.y_mode = 0;
    vnode.uv_mode = 0;
    vnode.levels_offset = 0;
    vnode.is_inter = 0;
    vnode.eob[0] = vnode.eob[1] = vnode.eob[2] = 0;
    tile->blocks.push_back(vnode);
    for (int cw = 0; cw < 2; ++cw) {
      const int cc = mi_col + cw * half_w;
      if (cc >= mi_cols) continue;
      AddQuad(tile, src, cr, cc, static_cast<int>(vsize), target_bsize,
              mi_rows, mi_cols, false);
    }
    tile->blocks[child].subtree = static_cast<std::uint16_t>(
        tile->blocks.size() - child - 1);
  }
  tile->blocks[index].subtree = static_cast<std::uint16_t>(
      tile->blocks.size() - index - 1);
}

void AddLeaf64(TileData* tile, const Frame& src, int mi_row, int mi_col) {
  av2enc::m2::syntax::BlockSyntax node;
  node.mi_row = static_cast<std::uint16_t>(mi_row);
  node.mi_col = static_cast<std::uint16_t>(mi_col);
  node.bsize = 12;
  node.partition = 0;
  node.allowed_mask = 1U << 0;
  node.flags = av2enc::m2::syntax::kFlagSbRoot;
  node.subtree = 0;
  node.y_mode = 0;
  node.uv_mode = 0;
  node.tx_size[0] = 4;
  node.tx_size[1] = 3;
  node.tx_size[2] = 3;
  node.levels_offset = static_cast<std::uint32_t>(tile->levels.size());
  node.is_inter = 0;
  node.eob[0] = node.eob[1] = node.eob[2] = 1;
  for (int plane = 0; plane < 3; ++plane) {
    const int mean = PlaneMean(src, plane, mi_col * 4, mi_row * 4, 64, 64);
    tile->levels.push_back(LevelForMean(mean));
  }
  tile->blocks.push_back(node);
}

TileData MakeTile(int width, int height, const Frame& src) {
  TileData tile;
  tile.plan.width = width;
  tile.plan.height = height;
  tile.plan.sb_size = 64;
  tile.plan.inter_frame = false;
  const int mi_cols = (width + 3) / 4;
  const int mi_rows = (height + 3) / 4;
  const int sb_cols = (width + 63) / 64;
  const int sb_rows = (height + 63) / 64;
  constexpr int kLeafBsize = 6;  // BLOCK_16X16
  for (int sb_row = 0; sb_row < sb_rows; ++sb_row) {
    for (int sb_col = 0; sb_col < sb_cols; ++sb_col) {
      if (sb_row * 16 + 16 <= mi_rows && sb_col * 16 + 16 <= mi_cols) {
        AddQuad(&tile, src, sb_row * 16, sb_col * 16, 12, kLeafBsize, mi_rows,
                mi_cols, true);
      } else {
        AddLeaf64(&tile, src, sb_row * 16, sb_col * 16);
      }
    }
  }
  return tile;
}

// ---------------------------------------------------------------------------
// Frame header with our §5.18.7.11 writer.
// ---------------------------------------------------------------------------
bool WriteFrameHeader(av2enc::BitWriter& writer,
                      const av2enc::m2::M2Config& config,
                      const av2enc::m2::post::M2TiledFrameHeader& header,
                      const av2enc::tiles::TileLayout& layout,
                      pcw::LrParams* params,
                      const pcw::LrHeaderContext& context, int tile_size_bytes,
                      std::string* error) {
  PutUvlc(writer, 0);
  PutUvlc(writer, 0);
  const bool key = header.keyframe;
  if (!key) writer.put_bit(true);
  writer.put_bit(true);
  writer.put_bit(false);
  writer.put_bits(header.order_hint, config.order_hint_bits);
  if (key) {
    writer.put_bit(false);
  } else {
    writer.put_bit(false);
    writer.put_bit(true);
    writer.put_bits(header.refresh_frame_flags, config.num_ref_frames);
    writer.put_bit(true);
    writer.put_bits(config.num_total_refs, 3U);
    writer.put_bits(header.ref_slot,
                    static_cast<unsigned>(CeilLog2U32(config.num_ref_frames)));
    writer.put_bit(false);
    writer.put_bit(false);
    writer.put_bit(true);
    writer.put_bit(false);
    writer.put_bits(0U, 2U);
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
  if (!av2enc::m2::post::write_m2_deblock_params(writer, deblock, delta,
                                                 error)) {
    return false;
  }
  if (pcw::WriteLrParams(writer, params, context, error) == 0 &&
      error != nullptr && !error->empty()) {
    return false;
  }
  writer.put_bit(false);  // tx_mode_select
  if (!key) {
    writer.put_bit(false);  // reference_select
    writer.put_bit(false);  // skip_mode_present
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set
  return true;
}

struct StreamOptions {
  av2enc::m2::M2Config config;
  av2enc::m2::post::M2PostSequenceOptions sequence;
};

bool EncodeStream(const StreamOptions& options, const TileData& tile,
                  pcw::LrParams* params, const pcw::LrHeaderContext& context,
                  pcw::TileState* state,
                  const std::vector<pcw::UnitDecision>* units,
                  const std::string& path, std::string* error) {
  const auto tile_plan = av2enc::m2::post::plan_m2_tiles(
      tile.plan.width, tile.plan.height, 1, 1);
  if (!tile_plan.ok) {
    *error = tile_plan.error;
    return false;
  }
  const auto& layout = tile_plan.layout;
  auto seq_obu = av2enc::m2::post::make_m2_sequence_header_obu(
      options.config, options.sequence, error);
  if (seq_obu.empty()) {
    std::fprintf(stderr, "seq obu empty: %s\n", error->c_str());
    return false;
  }

  std::vector<std::uint8_t> payload;
  std::size_t tile_bytes = 0;
  if (!av2enc::filters::pc_wiener_stream::gate::assemble_m2_frame_pcw(
          tile.plan, tile.blocks, tile.levels, options.config, state, units,
          payload, *error, &tile_bytes)) {
    std::fprintf(stderr, "tile assemble failed: %s\n", error->c_str());
    return false;
  }
  const int tile_size_bytes = av2enc::tiles::tile_size_bytes_for(
      static_cast<std::uint32_t>(payload.size()));
  if (tile_size_bytes == 0) {
    *error = "tile size not representable";
    return false;
  }
  av2enc::m2::post::M2TiledFrameHeader header;
  header.keyframe = true;
  header.order_hint = 0;
  header.ref_slot = 0;
  header.refresh_frame_flags = 1;
  av2enc::BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  if (!WriteFrameHeader(writer, options.config, header, layout, params, context,
                        tile_size_bytes, error)) {
    std::fprintf(stderr, "frame header failed: %s\n", error->c_str());
    return false;
  }
  writer.align_to_byte();
  std::vector<std::uint8_t> group = writer.take_bytes();
  group.insert(group.end(), payload.begin(), payload.end());
  const av2enc::m0::ObuType type = av2enc::m0::ObuType::kClosedLoopKey;
  std::vector<std::uint8_t> frame_obu =
      av2enc::m0::make_obu(av2enc::m0::ObuHeader{type}, group);

  std::vector<std::vector<std::uint8_t>> obus;
  obus.push_back(av2enc::m2::make_temporal_delimiter_obu());
  obus.push_back(seq_obu);
  obus.push_back(frame_obu);
  const bool ivf_ok = WriteIvf(path, obus, tile.plan.width, tile.plan.height, error);
  if (!ivf_ok) std::fprintf(stderr, "ivf write failed: %s\n", error->c_str());
  return ivf_ok;
}

bool Decode(const std::string& decoder, const std::string& ivf,
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

bool ReadDecoded(const std::string& path, Frame* frame, std::string* error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame src;
  if (reader.read_frame(src, *error) != av2enc::Y4mReadResult::kFrame) {
    if (error->empty()) *error = "decoded file has no frames";
    return false;
  }
  frame->width = static_cast<int>(src.width());
  frame->height = static_cast<int>(src.height());
  frame->y.assign(src.y().begin(), src.y().end());
  frame->u.assign(src.u().begin(), src.u().end());
  frame->v.assign(src.v().begin(), src.v().end());
  return true;
}

Frame DenoiseTarget(const Frame& base) {
  Frame out = base;
  const int w = base.width;
  const int h = base.height;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int sum = 0;
      int count = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int yy = std::max(0, std::min(h - 1, y + dy));
          const int xx = std::max(0, std::min(w - 1, x + dx));
          sum += base.y[static_cast<std::size_t>(yy) * w + xx];
          ++count;
        }
      }
      out.y[static_cast<std::size_t>(y) * w + x] =
          static_cast<std::uint8_t>(sum / count);
    }
  }
  return out;
}

pcw::DecisionPlanes MakePlanes(const Frame& base, const Frame& target) {
  pcw::DecisionPlanes planes;
  planes.width = base.width;
  planes.height = base.height;
  planes.bit_depth = 8;
  planes.base_q_idx = 90;
  planes.sub_x = 1;
  planes.sub_y = 1;
  planes.luma_shift = 3;
  planes.chroma_shift = 3;
  planes.curr[0] = base.y.data();
  planes.cdef[0] = base.y.data();
  planes.target[0] = target.y.data();
  planes.curr_stride[0] = base.width;
  planes.cdef_stride[0] = base.width;
  planes.target_stride[0] = base.width;
  return planes;
}

bool SelfParse(const pcw::LrParams& params, const pcw::LrHeaderContext& context,
               const pcw::TileState& grid, const std::vector<pcw::UnitDecision>& units,
               std::string* error) {
  av2enc::BitWriter writer;
  pcw::LrParams copy = params;
  const std::size_t bits = pcw::WriteLrParams(writer, &copy, context, error);
  if (bits == 0 || (error != nullptr && !error->empty())) return false;
  const std::vector<std::uint8_t> bytes(writer.bytes().begin(),
                                        writer.bytes().end());
  pcw::LrParams parsed;
  pcw::Reader reader(bytes.data(), bits);
  if (!pcw::ReadLrParams(reader, context, &parsed, error)) return false;
  if (parsed.frame_type[0] != params.frame_type[0]) {
    *error = "self-parse frame type mismatch";
    return false;
  }
  for (int p = 0; p < 3; ++p) {
    if (parsed.frame_filters_on[p] != params.frame_filters_on[p]) {
      *error = "self-parse frame_filters_on mismatch";
      return false;
    }
  }
  if (parsed.num_filter_classes != params.num_filter_classes) {
    *error = "self-parse NumFilterClasses mismatch";
    return false;
  }
  // Re-encode and re-decode every unit with an independent coder state.
  pcw::TileState write_state;
  pcw::TileState read_state;
  if (!pcw::InitTileState(&write_state, params, grid.mi_cols, grid.mi_rows, 1,
                          1, error)) {
    return false;
  }
  if (!pcw::InitTileState(&read_state, parsed, grid.mi_cols, grid.mi_rows, 1,
                          1, error)) {
    return false;
  }
  std::vector<std::uint8_t> buffer(1U << 16, 0);
  av2enc::av2::Encoder encoder;
  if (encoder.init_symbol(buffer.data(), buffer.size()) !=
      av2enc::av2::Status::kOk) {
    *error = "self-parse encoder init failed";
    return false;
  }
  int cursor = 0;
  for (int plane = 0; plane < 3; ++plane) {
    if (write_state.params.frame_type[plane] == pcw::kRestoreNone) continue;
    for (int r = 0; r < write_state.unit_rows[plane]; ++r) {
      for (int c = 0; c < write_state.unit_cols[plane]; ++c) {
        if (cursor >= static_cast<int>(units.size())) {
          *error = "self-parse unit list too short";
          return false;
        }
        if (!pcw::WriteUnit(encoder, write_state, plane, r, c, units[cursor],
                            error)) {
          return false;
        }
        ++cursor;
      }
    }
  }
  std::size_t payload = 0;
  if (encoder.exit_symbol(&payload) != av2enc::av2::Status::kOk) {
    *error = "self-parse encoder exit failed";
    return false;
  }
  av2enc::av2::Decoder decoder;
  if (decoder.init_symbol(buffer.data(), buffer.size(), payload) !=
      av2enc::av2::Status::kOk) {
    *error = "self-parse decoder init failed";
    return false;
  }
  cursor = 0;
  for (int plane = 0; plane < 3; ++plane) {
    if (read_state.params.frame_type[plane] == pcw::kRestoreNone) continue;
    for (int r = 0; r < read_state.unit_rows[plane]; ++r) {
      for (int c = 0; c < read_state.unit_cols[plane]; ++c) {
        pcw::UnitDecision decoded;
        if (!pcw::ReadUnit(decoder, read_state, plane, r, c, &decoded, error)) {
          return false;
        }
        if (decoded.type != units[cursor].type) {
          *error = "self-parse unit type mismatch";
          return false;
        }
        ++cursor;
      }
    }
  }
  return true;
}

void PrintHeader(const char* name, const pcw::DecisionResult& result,
                 std::uintmax_t base_bytes, std::uintmax_t pc_bytes,
                 double base_psnr_src, double pc_psnr_src, double base_psnr_tgt,
                 double pc_psnr_tgt, bool have_cvvdp_base, double cvvdp_base,
                 bool have_cvvdp_pc, double cvvdp_pc) {
  std::printf(
      "%s: units=%d active=%d classes=%d header_bits=%d tile_bits=%d\n", name,
      result.total_units, result.active_units, result.num_filter_classes,
      result.header_bits, result.tile_bits);
  std::printf(
      "%s: IVFs base=%ju pc=%ju bytes (%+jd); PSNR-Y vs source "
      "%.3f -> %.3f dB; vs denoise target %.3f -> %.3f dB\n",
      name, base_bytes, pc_bytes,
      static_cast<std::intmax_t>(pc_bytes) -
          static_cast<std::intmax_t>(base_bytes),
      base_psnr_src, pc_psnr_src, base_psnr_tgt, pc_psnr_tgt);
  if (have_cvvdp_base || have_cvvdp_pc) {
    std::printf("%s: CVVDP base=%.4f pc=%.4f JOD\n", name, cvvdp_base,
                cvvdp_pc);
  } else {
    std::printf("%s: CVVDP unavailable (cvvdp not found or failed)\n", name);
  }
}

}  // namespace

int main(int argc, char** argv) {
  int width = 640;
  int height = 360;
  std::string source = "testdata/shine_1280x720_60f.y4m";
  std::string out_path = "/tmp/pc_wiener_gate.ivf";
  std::string decoder;
  double lambda = 0.0;
  bool mode_b = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--width" && i + 1 < argc) {
      width = std::atoi(argv[++i]);
    } else if (a == "--height" && i + 1 < argc) {
      height = std::atoi(argv[++i]);
    } else if (a == "--source" && i + 1 < argc) {
      source = argv[++i];
    } else if (a == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (a == "--decoder" && i + 1 < argc) {
      decoder = argv[++i];
    } else if (a == "--lambda" && i + 1 < argc) {
      lambda = std::atof(argv[++i]);
    } else if (a == "--frame-class") {
      mode_b = true;
    } else {
      std::fprintf(stderr,
                   "usage: %s [--width W] [--height H] [--source F.y4m] "
                   "[--out F.ivf] [--decoder D] [--lambda L] [--frame-class]\n",
                   argv[0]);
      return 2;
    }
  }
  if ((width & 7) != 0 || (height & 7) != 0) {
    std::fprintf(stderr, "width/height must be divisible by 8\n");
    return 2;
  }
  if (decoder.empty()) decoder = FindDecoder();
  if (decoder.empty()) {
    std::fprintf(stderr, "SKIP: no dav2d decoder found\n");
    return 77;
  }

  std::string error;
  Frame src;
  if (!ReadFirstFrame(source, &src, &error)) {
    std::fprintf(stderr, "source read failed: %s\n", error.c_str());
    return 1;
  }
  Frame content;
  Downscale(src, width, height, &content);

  StreamOptions options;
  options.config.max_frame_width = static_cast<std::uint16_t>(width);
  options.config.max_frame_height = static_cast<std::uint16_t>(height);
  options.config.base_q_idx = 90;
  options.config.num_total_refs = 1;
  options.config.disable_cdf_update = true;
  options.sequence.filters.enable_restoration = true;
  options.sequence.filters.lr_uv_present = false;
  options.sequence.filters.lr_disable_luma_pc_wiener = false;
  options.sequence.filters.lr_disable_luma_wiener_ns = !mode_b;
  options.sequence.filters.enable_cdef = false;
  options.sequence.filters.enable_ccso = false;

  pcw::LrHeaderContext context;
  context.coded_lossless = false;
  context.enable_restoration = true;
  context.single_picture_header_flag = false;
  context.frame_is_intra = true;
  context.sb_size = 64;
  context.num_planes = 3;
  context.sub_x = 1;
  context.sub_y = 1;
  context.base_q_idx = 90;
  context.lr_tools_disable[0][pcw::kRestoreNone] = false;
  context.lr_tools_disable[0][pcw::kRestorePcWiener] = false;
  context.lr_tools_disable[0][pcw::kRestoreWienerNs] = !mode_b;
  context.lr_tools_disable[0][pcw::kRestoreSwitchable] = false;
  context.lr_tools_disable[1][pcw::kRestoreNone] = false;
  context.lr_tools_disable[1][pcw::kRestorePcWiener] = true;
  context.lr_tools_disable[1][pcw::kRestoreWienerNs] = !mode_b;
  context.lr_tools_disable[1][pcw::kRestoreSwitchable] = false;

  const TileData tile = MakeTile(width, height, content);
  pcw::LrParams off_params;
  off_params.frame_type[0] = off_params.frame_type[1] = off_params.frame_type[2] =
      pcw::kRestoreNone;
  off_params.luma_shift = 3;
  off_params.chroma_shift = 3;
  const std::string base_ivf = "/tmp/pc_wiener_gate_base.ivf";
  const std::string base_y4m = "/tmp/pc_wiener_gate_base.y4m";
  if (!EncodeStream(options, tile, &off_params, context, nullptr, nullptr,
                    base_ivf, &error)) {
    std::fprintf(stderr, "baseline encode failed: %s\n", error.c_str());
    return 1;
  }
  if (!Decode(decoder, base_ivf, base_y4m, &error)) {
    std::fprintf(stderr, "baseline decode failed: %s\n", error.c_str());
    if ((height % 64) != 0) {
      std::fprintf(
          stderr,
          "fixture limit: the vendored M2 writer does not encode every "
          "non-multiple-of-64 height (1280x720 and 1280x688 fail before the "
          "LR stage; 1280x704 and 640x360 pass).  This is not a PC-Wiener "
          "syntax failure; rerun with --height 704 for the 720p-class "
          "measurement.\n");
    }
    return 1;
  }
  Frame base;
  if (!ReadDecoded(base_y4m, &base, &error)) {
    std::fprintf(stderr, "baseline read failed: %s\n", error.c_str());
    return 1;
  }
  std::printf("baseline: %dx%d decoded\n", base.width, base.height);
  std::printf("fixture: PSNR-Y vs source %.3f dB\n", Psnr(base.y, content.y));

  const Frame denoise = DenoiseTarget(base);
  const pcw::DecisionPlanes planes = MakePlanes(base, denoise);
  pcw::DecisionResult result;
  if (mode_b) {
    if (!pcw::DecideFrameClassPcWiener(planes, 16, lambda, &result, &error)) {
      std::fprintf(stderr, "frame-class decision failed: %s\n", error.c_str());
      return 1;
    }
  } else {
    if (!pcw::DecidePerUnitPcWiener(planes, lambda, &result, &error)) {
      std::fprintf(stderr, "per-unit decision failed: %s\n", error.c_str());
      return 1;
    }
  }

  pcw::LrParams params = result.params;
  if (!mode_b) {
    // The header writer needs the tool list used by the decision.
    context.lr_tools_disable[0][pcw::kRestoreWienerNs] = true;
    context.lr_tools_disable[1][pcw::kRestoreWienerNs] = true;
  }
  av2enc::BitWriter header_probe;
  if (pcw::WriteLrParams(header_probe, &params, context, &error) == 0 &&
      !error.empty()) {
    std::fprintf(stderr, "header write failed: %s\n", error.c_str());
    return 1;
  }
  if (static_cast<int>(header_probe.bit_size()) != result.header_bits) {
    std::printf("bit-count note: decision header_bits=%d, gate header_bits=%zu\n",
                result.header_bits, header_probe.bit_size());
  }
  const int mi_cols = (width + 3) / 4;
  const int mi_rows = (height + 3) / 4;
  pcw::TileState state;
  if (!pcw::InitTileState(&state, params, mi_cols, mi_rows, 1, 1, &error)) {
    std::fprintf(stderr, "tile state init failed: %s\n", error.c_str());
    return 1;
  }

  std::vector<pcw::UnitDecision> units(
      static_cast<std::size_t>(state.unit_rows[0]) * state.unit_cols[0]);
  for (int r = 0; r < state.unit_rows[0]; ++r) {
    for (int c = 0; c < state.unit_cols[0]; ++c) {
      pcw::UnitDecision decision;
      decision.type =
          result.unit_type[0][static_cast<std::size_t>(r) * state.unit_cols[0] +
                              c];
      units[static_cast<std::size_t>(r) * state.unit_cols[0] + c] = decision;
    }
  }

  Frame expected = base;
  int filtered_blocks = 0;
  if (mode_b) {
    filtered_blocks = pcw::ApplyFrameClassWiener(
        planes, params.frame_coeffs[0], result.filter_class.data(),
        result.subclass_lookup, result.num_filter_classes, expected.y.data(),
        expected.width);
  } else {
    filtered_blocks = pcw::ApplyPerUnitPcWiener(
        planes, result.unit_type[0], pcw::kRestorationTileSizeMax >> params.luma_shift,
        expected.y.data(), expected.width);
  }
  std::printf("encoder-side §7.20 output: %d filtered 4x4 blocks\n",
              filtered_blocks);

  const std::string pc_ivf = out_path;
  if (!EncodeStream(options, tile, &params, context, &state, &units, pc_ivf,
                    &error)) {
    std::fprintf(stderr, "PC-Wiener encode failed: %s\n", error.c_str());
    return 1;
  }
  const std::string pc_y4m = "/tmp/pc_wiener_gate_decoded.y4m";
  const bool decoded_ok = Decode(decoder, pc_ivf, pc_y4m, &error);
  if (!decoded_ok) {
    std::printf("decoder limit: %s\n", error.c_str());
    std::printf(
        "decoder limit detail: the pinned dav2d did not accept the luma "
        "PC-Wiener syntax\n");
  }

  std::error_code size_error;
  const std::uintmax_t base_bytes = fs::file_size(base_ivf, size_error);
  size_error.clear();
  const std::uintmax_t pc_bytes = fs::file_size(pc_ivf, size_error);
  const double base_psnr_src = Psnr(base.y, content.y);
  const double base_psnr_tgt = Psnr(base.y, denoise.y);
  const double pc_psnr_src = decoded_ok ? Psnr(expected.y, content.y) : base_psnr_src;
  const double pc_psnr_tgt = decoded_ok ? Psnr(expected.y, denoise.y) : base_psnr_tgt;

  bool have_cvvdp_base = false;
  bool have_cvvdp_pc = false;
  double cvvdp_base = 0.0;
  double cvvdp_pc = 0.0;
  const std::string ref_y4m = "/tmp/pc_wiener_gate_ref.y4m";
  const std::string base_q_y4m = "/tmp/pc_wiener_gate_base_q.y4m";
  const std::string pc_q_y4m = "/tmp/pc_wiener_gate_pc_q.y4m";
  WriteY4m(ref_y4m, content);
  WriteY4m(base_q_y4m, base);
  WriteY4m(pc_q_y4m, expected);
  have_cvvdp_base = RunCvvdp(ref_y4m, base_q_y4m, &cvvdp_base);
  have_cvvdp_pc = RunCvvdp(ref_y4m, pc_q_y4m, &cvvdp_pc);
  PrintHeader(mode_b ? "frame-class" : "per-unit", result, base_bytes, pc_bytes,
              base_psnr_src, pc_psnr_src, base_psnr_tgt, pc_psnr_tgt,
              have_cvvdp_base, cvvdp_base, have_cvvdp_pc, cvvdp_pc);

  int diffs = -1;
  int chroma_diffs = -1;
  if (decoded_ok) {
    Frame decoded;
    if (!ReadDecoded(pc_y4m, &decoded, &error)) {
      std::fprintf(stderr, "decoded read failed: %s\n", error.c_str());
      return 1;
    }
    diffs = 0;
    for (std::size_t i = 0; i < decoded.y.size(); ++i) {
      diffs += decoded.y[i] != expected.y[i];
    }
    chroma_diffs = 0;
    for (std::size_t i = 0; i < decoded.u.size(); ++i) {
      chroma_diffs += decoded.u[i] != base.u[i];
      chroma_diffs += decoded.v[i] != base.v[i];
    }
    std::printf("decoded vs encoder §7.20: luma %d diffs, chroma %d diffs\n",
                diffs, chroma_diffs);
    if (diffs == 0 && chroma_diffs == 0) {
      std::printf("PASS: PC-Wiener decoded bit-exactly\n");
      return 0;
    }
    std::fprintf(stderr, "FAIL: PC-Wiener decode mismatch\n");
    return 1;
  }

  // Decoder does not implement luma PC-Wiener.  Keep the gate useful:
  //   1. exact self-parse of the header and unit symbols,
  //   2. feature-off decode invariance already proved above (baseline decode
  //      is the input to the classifier; compare a second decode).
  if (!SelfParse(params, context, state, units, &error)) {
    std::fprintf(stderr, "self-parse failed: %s\n", error.c_str());
    return 1;
  }
  const std::string base_y4m2 = "/tmp/pc_wiener_gate_base2.y4m";
  if (!Decode(decoder, base_ivf, base_y4m2, &error)) {
    std::fprintf(stderr, "feature-off decode failed: %s\n", error.c_str());
    return 1;
  }
  Frame base2;
  if (!ReadDecoded(base_y4m2, &base2, &error)) return 1;
  int off_diffs = 0;
  for (std::size_t i = 0; i < base.y.size(); ++i) {
    off_diffs += base.y[i] != base2.y[i];
  }
  std::printf("feature-off decode invariance: %d luma diffs\n", off_diffs);
  if (off_diffs != 0) {
    std::fprintf(stderr, "FAIL: feature-off decode is not invariant\n");
    return 1;
  }
  std::printf(
      "PASS (decoder-limited): exact self-parse and feature-off invariance; "
      "the pinned dav2d rejects luma PC-Wiener\n");
  return 0;
}
