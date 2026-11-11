// Minimal extended-partition stream generator for the dav2d block-level gate.
//
// Produces four IVF files, all 64x64, all-DC, all-skip:
//
//   baseline_frozen.ivf    one PARTITION_NONE superblock, disable_cdf_update=1
//   ext_frozen.ivf         one HORZ_3 superblock,          disable_cdf_update=1
//   baseline_adaptive.ivf  one PARTITION_NONE superblock, disable_cdf_update=0
//   ext_adaptive.ivf       one HORZ_3 superblock,          disable_cdf_update=0
//
// The sequence header is written with enable_ext_partitions = 0 (the M1
// header writer owns that bit); gate_dav2d.py inserts the two §5.4.3 flag bits
// into the ext_* files before decoding.  Baseline and ext must decode to the
// same all-128 pixels, which proves the pinned decoder parses do_ext_partition
// and follows the HORZ_3 child geometry.
//
// This tool is deliberately narrow: it writes partition symbols through the
// adapter (plan_node/emit_symbols) and the smallest legal intra leaf
// (DC_PRED, no residual).  It is not a replacement for the M1 writer; it exists
// so block-level gating does not have to wait for the m1_syntax edits listed
// in README.md.

#include "partition_stream.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "encoder/m0_cdf_tables.h"
#include "encoder/m1/m1_cdf_tables.h"
#include "encoder/m1/m1_headers.h"
#include "encoder/m1/m1_tables.h"
#include "encoder/tx_variants/tx_variants.h"
#include "io/ivf.h"

namespace ps = av2enc::m1::partition_stream;
namespace av2 = av2enc::av2;
namespace p = av2enc::partition;
namespace pe = av2enc::partition_ext;

namespace {

constexpr int kWidth = 64;
constexpr int kHeight = 64;

// Costs that pick `want` at every node where the spec allows it, else NONE.
pe::CostFunction force_cost(
    const std::function<pe::PartitionType(const pe::NodeInfo&)>& chooser) {
  return [chooser](const pe::NodeInfo& info) {
    const double w = static_cast<double>(p::block_width(info.size));
    const double h = static_cast<double>(p::block_height(info.size));
    pe::NodeCosts costs;
    costs.leaf = w * h * w * h;
    costs.none_signal = pe::kInfiniteCost;
    costs.split_signal = pe::kInfiniteCost;
    costs.horz_signal = pe::kInfiniteCost;
    costs.vert_signal = pe::kInfiniteCost;
    costs.horz3_signal = pe::kInfiniteCost;
    costs.vert3_signal = pe::kInfiniteCost;
    costs.horz4a_signal = pe::kInfiniteCost;
    costs.horz4b_signal = pe::kInfiniteCost;
    costs.vert4a_signal = pe::kInfiniteCost;
    costs.vert4b_signal = pe::kInfiniteCost;
    pe::PartitionType want = chooser(info);
    if (!info.allowed.is_allowed(want)) {
      if (info.boundary.implied &&
          info.allowed.is_allowed(info.boundary.implied_partition)) {
        want = info.boundary.implied_partition;
      } else {
        for (int i = 0; i < pe::kPartitionCount; ++i) {
          const auto candidate = static_cast<pe::PartitionType>(i);
          if (info.allowed.is_allowed(candidate)) {
            want = candidate;
            break;
          }
        }
      }
    }
    switch (want) {
      case pe::PartitionType::kNone: costs.none_signal = 0.0; break;
      case pe::PartitionType::kSplit: costs.split_signal = 0.0; break;
      case pe::PartitionType::kHorz: costs.horz_signal = 0.0; break;
      case pe::PartitionType::kVert: costs.vert_signal = 0.0; break;
      case pe::PartitionType::kHorz3: costs.horz3_signal = 0.0; break;
      case pe::PartitionType::kVert3: costs.vert3_signal = 0.0; break;
      case pe::PartitionType::kHorz4A: costs.horz4a_signal = 0.0; break;
      case pe::PartitionType::kHorz4B: costs.horz4b_signal = 0.0; break;
      case pe::PartitionType::kVert4A: costs.vert4a_signal = 0.0; break;
      case pe::PartitionType::kVert4B: costs.vert4b_signal = 0.0; break;
    }
    return costs;
  };
}

// The mode CDFs a minimal DC / all-skip leaf touches.
struct ModeCdfs {
  std::uint16_t y_mode_set[5];
  std::uint16_t y_mode_index[3][9];
  std::uint16_t uv_mode[2][9];
  std::uint16_t is_cfl[3][3];
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
};

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

void init_mode_cdfs(ModeCdfs& c) {
  copy_array(c.y_mode_set, av2enc::m1::kYModeSetCdf);
  copy_array(c.y_mode_index, av2enc::m1::kYModeIndexCdf);
  copy_array(c.uv_mode, av2enc::m1::kUvModeCflNotAllowedCdf);
  copy_array(c.is_cfl, av2enc::m1::kIsCflCdf);
  copy_array(c.txb_skip, av2enc::m0::kTxbSkipCdf);
  copy_array(c.v_txb_skip, av2enc::m0::kVTxbSkipCdf);
}

int tx_sz_ctx(int tx_size) {
  return (av2enc::m1::kTxSizeSqr[tx_size] +
          av2enc::m1::kTxSizeSqrUp[tx_size] + 1) >> 1;
}

struct LeafState {
  std::vector<int> left_mi;
  std::vector<int> above_mi;
};

bool encode_leaf(const p::FrameGeometry& geometry, const ps::M1Node& node,
                 ModeCdfs& mode, av2::Encoder& encoder, LeafState& state,
                 std::string& error) {
  const p::BlockSize size = static_cast<p::BlockSize>(node.bsize);
  const int r = node.mi_row;
  const int c = node.mi_col;
  const int w4 = p::num_4x4_blocks_wide(size);
  const int h4 = p::num_4x4_blocks_high(size);

  // DC_PRED is non-directional: YModeSet symbol 0, YModeIndex 0 (ctx 0).
  if (encoder.encode_symbol(mode.y_mode_set, 4, 0) != av2::Status::kOk ||
      encoder.encode_symbol(mode.y_mode_index[0], 8, 0) != av2::Status::kOk) {
    error = "failed to code the DC luma mode";
    return false;
  }
  if (node.has_chroma) {
    // is_cfl = 0 (CfL enabled) and UV mode DC (index 0, ctx 0).
    if (encoder.encode_symbol(mode.is_cfl[0], 2, 0) != av2::Status::kOk ||
        encoder.encode_symbol(mode.uv_mode[0], 8, 0) != av2::Status::kOk) {
      error = "failed to code the DC chroma mode";
      return false;
    }
  }

  // A leaf does not always code chroma with its own luma size: HORZ_4A/4B
  // children with chromaOffset == 1 leave the first three children chroma-less
  // and the last one codes the whole parent chroma block (32x32 here).  The
  // adapter carries that block in node.chroma.
  const p::BlockSize chroma_size =
      p::chroma_plane_residual_size(geometry, node.chroma.size);
  if (node.has_chroma && !p::is_valid_block_size(chroma_size)) {
    error = "invalid chroma residual size";
    return false;
  }
  const int coded_w[3] = {p::block_width(size), p::block_width(chroma_size),
                          p::block_width(chroma_size)};
  const int coded_h[3] = {p::block_height(size), p::block_height(chroma_size),
                          p::block_height(chroma_size)};
  const int planes = node.has_chroma ? 3 : 1;
  for (int plane = 0; plane < planes; ++plane) {
    const auto params = av2enc::txv::make_tx_params(
        coded_w[plane], coded_h[plane], av2enc::txv::TxType::kDctDct);
    if (!params.has_value()) {
      error = "unsupported transform size";
      return false;
    }
    const int tsc = tx_sz_ctx(params->tx_size);
    // All neighbour levels are zero and coded == block size, so the luma
    // context is 0, U is 6 and V is 0 (eob_u_ == 0).
    const int ctx = plane == 0 ? 0 : (plane == 1 ? 6 : 0);
    std::uint16_t* row = plane == 2 ? &mode.v_txb_skip[ctx][0]
                                    : &mode.txb_skip[0][tsc][ctx][0];
    if (encoder.encode_symbol(row, 2, 1) != av2::Status::kOk) {
      error = "failed to code txb_skip";
      return false;
    }
  }

  // Update the partition context arrays exactly like m1_syntax::write_block().
  for (int yy = 0; yy < h4; ++yy) {
    const int row = r + yy;
    if (row < static_cast<int>(state.left_mi.size())) {
      state.left_mi[static_cast<std::size_t>(row)] =
          p::mi_height_log2(size);
    }
  }
  for (int xx = 0; xx < w4; ++xx) {
    const int col = c + xx;
    if (col < static_cast<int>(state.above_mi.size())) {
      state.above_mi[static_cast<std::size_t>(col)] =
          p::mi_width_log2(size);
    }
  }
  return true;
}

bool build_tile(const p::FrameGeometry& geometry, const pe::Tools& tools,
                const pe::FrameDecision& frame, bool disable_cdf_update,
                std::vector<std::uint8_t>& tile, std::size_t* symbols,
                std::size_t* literals, std::string& error) {
  ps::TilePartitionCdfs partition_cdfs;
  ps::init_tile_partition_cdfs(partition_cdfs);
  ModeCdfs mode;
  init_mode_cdfs(mode);

  std::vector<std::uint8_t> buffer(1u << 20);
  av2::Encoder encoder;
  if (encoder.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    error = "entropy init failed";
    return false;
  }
  encoder.disable_cdf_update = disable_cdf_update ? 1U : 0U;

  ps::FrameStream stream;
  if (!ps::build_frame_stream(geometry, tools, frame, stream, error)) {
    return false;
  }

  p::MiSizeGrid grid(geometry);
  LeafState state;
  state.left_mi.assign(static_cast<std::size_t>(p::mi_rows(geometry)) + 16, 6);
  state.above_mi.assign(static_cast<std::size_t>(p::mi_cols(geometry)) + 16, 6);
  pe::SyntaxStats stats{};

  for (std::size_t i = 0; i < stream.nodes.size(); ++i) {
    const ps::M1Node& node = stream.nodes[i];
    const ps::NodePlan plan = ps::plan_node(geometry, tools, node, grid);
    if (!plan.ok) {
      error = "plan failed at node " + std::to_string(i) + ": " + plan.error;
      return false;
    }
    if (!ps::emit_symbols(plan.symbols, partition_cdfs, encoder, &stats,
                          error)) {
      return false;
    }
    const bool is_leaf =
        node.partition == static_cast<std::uint8_t>(pe::PartitionType::kNone);
    if (is_leaf) {
      if (!encode_leaf(geometry, node, mode, encoder, state, error)) {
        return false;
      }
      grid.fill_block(node.mi_row, node.mi_col,
                      static_cast<p::BlockSize>(node.bsize));
    }
  }

  std::size_t payload_bytes = 0;
  if (encoder.exit_symbol(&payload_bytes) != av2::Status::kOk) {
    error = "entropy exit failed";
    return false;
  }
  tile.assign(buffer.begin(),
              buffer.begin() + static_cast<std::ptrdiff_t>(payload_bytes));
  if (symbols != nullptr) *symbols = static_cast<std::size_t>(stats.symbols);
  if (literals != nullptr) *literals = static_cast<std::size_t>(stats.literal_bits);
  return true;
}

bool write_ivf(const std::filesystem::path& path,
               const std::vector<std::uint8_t>& payload, std::string& error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = kWidth;
  header.height = kHeight;
  header.fps_num = 30;
  header.fps_den = 1;
  if (!writer.open(path, header, error)) return false;
  if (!writer.write_frame(payload, 0, error)) return false;
  return writer.close(error);
}

pe::FrameDecision make_frame(
    const p::FrameGeometry& geometry, const pe::Tools& tools,
    const std::function<pe::PartitionType(const pe::NodeInfo&)>& chooser,
    int max_depth) {
  pe::DecisionOptions options;
  options.min_block_size_px = 4;
  options.max_depth = max_depth;
  return pe::decide_frame(geometry, force_cost(chooser), tools, options);
}

struct Case {
  const char* name;
  std::function<pe::PartitionType(const pe::NodeInfo&)> chooser;
  int max_depth;
};

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path out_dir = "gate_out";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_dir = argv[++i];
    } else {
      std::cerr << "usage: " << argv[0] << " --out DIR\n";
      return 2;
    }
  }
  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  if (ec) {
    std::cerr << "cannot create " << out_dir << ": " << ec.message() << '\n';
    return 2;
  }

  p::FrameGeometry geometry;
  geometry.width = kWidth;
  geometry.height = kHeight;
  geometry.sb_size = 64;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;

  pe::Tools tools;
  tools.enable_ext_partitions = true;
  tools.enable_uneven_4way_partitions = true;

  av2enc::m1::M1Config config;
  config.max_frame_width = kWidth;
  config.max_frame_height = kHeight;
  config.base_q_idx = 90;
  config.enable_cfl_intra = true;
  config.enable_chroma_dctonly = true;

  const Case cases[] = {
      {"baseline",
       [](const pe::NodeInfo&) { return pe::PartitionType::kNone; }, 0},
      {"ext",
       [](const pe::NodeInfo& info) {
         return info.depth == 0 ? pe::PartitionType::kHorz3
                                : pe::PartitionType::kNone;
       },
       1},
      {"ext4",
       [](const pe::NodeInfo& info) {
         if (info.depth == 0) return pe::PartitionType::kHorz;
         if (info.size == p::BlockSize::k64x32 && info.mi_row == 0) {
           return pe::PartitionType::kVert;
         }
         if (info.size == p::BlockSize::k32x32 && info.mi_row == 0 &&
             info.mi_col == 0) {
           return pe::PartitionType::kHorz4A;
         }
         return pe::PartitionType::kNone;
       },
       3},
      {"ext4b",
       [](const pe::NodeInfo& info) {
         if (info.depth == 0) return pe::PartitionType::kHorz;
         if (info.size == p::BlockSize::k64x32 && info.mi_row == 0) {
           return pe::PartitionType::kVert;
         }
         if (info.size == p::BlockSize::k32x32 && info.mi_row == 0) {
           return info.mi_col == 0 ? pe::PartitionType::kHorz4B
                                   : pe::PartitionType::kVert4B;
         }
         return pe::PartitionType::kNone;
       },
       3},
  };

  for (const Case& test : cases) {
    const pe::FrameDecision frame =
        make_frame(geometry, tools, test.chooser, test.max_depth);
    if (frame.status != pe::DecisionStatus::kOk) {
      std::cerr << test.name << ": decision failed: " << frame.error << '\n';
      return 1;
    }
    if (frame.superblocks.empty()) {
      std::cerr << test.name << ": empty frame\n";
      return 1;
    }
    std::string error;
    if (!pe::validate_frame(geometry, tools, frame, error)) {
      std::cerr << test.name << ": frame invalid: " << error << '\n';
      return 1;
    }

    for (bool adaptive : {false, true}) {
      std::vector<std::uint8_t> tile;
      std::size_t symbols = 0;
      std::size_t literals = 0;
      config.disable_cdf_update = !adaptive;
      if (!build_tile(geometry, tools, frame, config.disable_cdf_update, tile,
                      &symbols, &literals, error)) {
        std::cerr << test.name << ": tile failed: " << error << '\n';
        return 1;
      }
      const std::vector<std::uint8_t> payload =
          av2enc::m1::assemble_m1_still_picture(config, tile);
      const std::string file = std::string(test.name) +
                               (adaptive ? "_adaptive.ivf" : "_frozen.ivf");
      if (!write_ivf(out_dir / file, payload, error)) {
        std::cerr << test.name << ": ivf failed: " << error << '\n';
        return 1;
      }
      std::cout << file << ": leaves=" << frame.leaves
                << " symbols=" << symbols << " literal_bits=" << literals
                << " tile=" << tile.size() << " bytes\n";
    }
  }
  std::cout << "wrote gate streams to " << out_dir << '\n';
  return 0;
}
