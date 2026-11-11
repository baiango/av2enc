// Minimal AV2 v1.0.0 stream that signals `dc_sign_horz_vert S()`.
//
// Why this exists
// ---------------
// `dc_sign_horz_vert` (§5.20.7.27) is reached only when the luma transform
// class is TX_CLASS_HORIZ or TX_CLASS_VERT.  The shipped M1 writer filters
// every 1D identity candidate before TX-type decision
// (`src/encoder/m1/m1_still.cpp:848-855`), so no production stream can code
// the element.  This tool writes the smallest legal frame that can:
//
//   32x16 visible frame, one forced partition leaf (64x64 SB -> HORZ ->
//   64x32 -> VERT -> 32x32 -> HORZ -> 32x16), DC_PRED, one 32x16 luma
//   transform, WIDE_32 transform set, is_long_side_dct = 1, intra_tx_type =
//   H_DCT (symbol 3), eob = 2 with non-zero levels at
//     * scan position 0                     -> dc_sign
//     * scan position 1 (row 1, col 0)      -> dc_sign_horz_vert (ctx 0)
//
// Streams written:
//   hv_pos.ivf   horz-vert coefficient sign = 0, correct ctx 0
//   hv_zero.ivf  no horz-vert coefficient (eob = 1, DCT_DCT control)
//   hv_ctx1.ivf  horz-vert sign 0 but deliberately coded with ctx 1
//
// `gate_dav2d.py` decodes hv_pos and hv_zero with
// `dav2d --strict 1 --threads 1` and checks pinned IVF/decoded MD5s.  It
// requires hv_ctx1 to be rejected: the decoder derives ctx 0
// (`TileDcSignCdf[ptype][isHidden][0]`), so a writer that uses any other
// context desynchronises.  That is the proof that the element's context is
// part of the syntax.
//
// A negative horz-vert coefficient (sign = 1) is unit-tested in
// missing_bits_test.cpp and its entropy bytes are pinned there.  The pinned
// dav2d build rejects *any* negative 1D-transform coefficient under
// `--strict 1` (both H_DCT and V_DCT, sign-only change); with `--strict 0`
// it decodes to different pixels.  See README.md "dav2d strict limitation".
// The gate therefore uses sign = 0 plus the wrong-context control.
//
// Adds files only; no core/CMake/main edit.

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
#include "encoder/m1/missing_bits/dc_sign_horz_vert.h"
#include "encoder/m1/partition_stream/partition_stream.h"
#include "encoder/partition/partition.h"
#include "encoder/partition_ext/ext_syntax.h"
#include "io/ivf.h"

namespace mb = av2enc::m1::missing_bits;
namespace ps = av2enc::m1::partition_stream;
namespace av2 = av2enc::av2;
namespace p = av2enc::partition;
namespace pe = av2enc::partition_ext;
namespace m0 = av2enc::m0;
namespace m1 = av2enc::m1;

namespace {

constexpr int kWidth = 32;
constexpr int kHeight = 16;

// TX_32X16 (TX_SIZES_ALL index 10).
constexpr int kTx32x16 = 10;
// TX_16X8 (index 8), chroma residual for 4:2:0.
constexpr int kTx16x8 = 8;

enum class Variant {
  kHvPositive = 0,
  kHvZero = 1,
  kHvWrongContext = 2,
  kHvNegative = 3,
};

pe::CostFunction force_cost() {
  return [](const pe::NodeInfo& info) {
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
    if (info.boundary.implied &&
        info.allowed.is_allowed(info.boundary.implied_partition)) {
      switch (info.boundary.implied_partition) {
        case pe::PartitionType::kNone:
          costs.none_signal = 0.0;
          break;
        case pe::PartitionType::kSplit:
          costs.split_signal = 0.0;
          break;
        case pe::PartitionType::kHorz:
          costs.horz_signal = 0.0;
          break;
        case pe::PartitionType::kVert:
          costs.vert_signal = 0.0;
          break;
        default:
          break;
      }
    } else if (info.allowed.is_allowed(pe::PartitionType::kNone)) {
      costs.none_signal = 0.0;
    }
    return costs;
  };
}

// CDFs touched by this one leaf.
struct Cdfs {
  std::uint16_t y_mode_set[5];
  std::uint16_t y_mode_index[3][9];
  std::uint16_t is_cfl[3][3];
  std::uint16_t uv_mode[2][9];
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
  std::uint16_t eob_pt_512[3][9];
  std::uint16_t is_long_side_dct[2][3];
  std::uint16_t intra_tx_type_long[4][5];
  std::uint16_t coeff_base_lf[5][33][2][7];
  std::uint16_t coeff_base_lf_eob[5][4][6];
  std::uint16_t dc_sign[2][2][3][3];
};

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

void init_cdfs(Cdfs& c) {
  copy_array(c.y_mode_set, m1::kYModeSetCdf);
  copy_array(c.y_mode_index, m1::kYModeIndexCdf);
  copy_array(c.is_cfl, m1::kIsCflCdf);
  copy_array(c.uv_mode, m1::kUvModeCflNotAllowedCdf);
  copy_array(c.txb_skip, m0::kTxbSkipCdf);
  copy_array(c.v_txb_skip, m0::kVTxbSkipCdf);
  copy_array(c.eob_pt_512, m0::kEobPt512Cdf);
  copy_array(c.is_long_side_dct, m1::kIsLongSideDctCdf);
  copy_array(c.intra_tx_type_long, m1::kIntraTxTypeLongCdf);
  copy_array(c.coeff_base_lf, m0::kCoeffBaseLfCdf);
  copy_array(c.coeff_base_lf_eob, m0::kCoeffBaseLfEobCdf);
  copy_array(c.dc_sign, m0::kDcSignCdf);
}

bool sym(av2::Encoder& encoder, std::uint16_t* cdf, int n, int symbol,
         std::string& error) {
  if (encoder.encode_symbol(cdf, n, symbol) != av2::Status::kOk) {
    error = "entropy encode_symbol failed";
    return false;
  }
  return true;
}

bool code_luma(av2::Encoder& encoder, Cdfs& c, Variant variant,
               std::string& error) {
  const int tsc =
      (m1::kTxSizeSqr[kTx32x16] + m1::kTxSizeSqrUp[kTx32x16] + 1) >> 1;
  // all_zero = 0
  if (!sym(encoder, &c.txb_skip[0][tsc][0][0], 2, 0, error)) return false;

  if (variant == Variant::kHvZero) {
    // eob = 1 -> eob_pt_512 = 0, DCT_DCT is forced, no 1D class.
    if (!sym(encoder, &c.eob_pt_512[0][0], 8, 0, error)) return false;
    // c == 0 is both the EOB and DC position: coeff_base_eob, level 1.
    if (!sym(encoder, &c.coeff_base_lf_eob[tsc][0][0], 5, 0, error)) {
      return false;
    }
    // dc_sign, ctx 0 (first block).
    if (!sym(encoder, mb::dc_sign_cdf_slot(c.dc_sign, 0, 0, 0), 2, 0,
             error)) {
      return false;
    }
    return true;
  }

  // eob = 2 -> eob_pt_512 = 1.
  if (!sym(encoder, &c.eob_pt_512[0][0], 8, 1, error)) return false;

  // §5.20.8.2 WIDE_32: is_long_side_dct = 1, intra_tx_type = H_DCT (3).
  if (!sym(encoder, &c.is_long_side_dct[0][0], 2, 1, error)) return false;
  if (!sym(encoder, &c.intra_tx_type_long[m1::kTxSizeSqr[kTx32x16]][0], 4, 3,
           error)) {
    return false;
  }

  // Forward coeff_base loop, c = 0..eob-1.
  // c = 0: DC, low-frequency luma, ctx 0, level 1 -> coeff_base symbol 0.
  if (!sym(encoder, &c.coeff_base_lf[tsc][0][0][0], 6, 0, error)) {
    return false;
  }
  // c = 1: last coefficient, row 1 col 0 (HORIZ scan), level 1.
  // coeff_base_eob ctx = 1 (1 <= area/8).
  if (!sym(encoder, &c.coeff_base_lf_eob[tsc][1][0], 5, 0, error)) {
    return false;
  }

  // Reverse sign loop, c = 1 then c = 0.
  // c = 1: row 1, col 0, TX_CLASS_HORIZ -> dc_sign_horz_vert, ctx 0.
  const bool wrong_context = variant == Variant::kHvWrongContext;
  const int hv_sign =
      (variant == Variant::kHvNegative || wrong_context) ? 1 : 0;
  if (wrong_context) {
    // Non-normative control: the element's context is 0, so coding with ctx 1
    // must make the decoder desynchronise.
    if (!mb::write_dc_sign(encoder,
                           mb::dc_sign_cdf_slot(c.dc_sign, 0, 0, 1), 1,
                           hv_sign)) {
      error = "wrong-context control write failed";
      return false;
    }
  } else if (!mb::write_dc_sign_horz_vert(
                 encoder, mb::dc_sign_horz_vert_cdf_slot(c.dc_sign, 0, 0),
                 hv_sign)) {
    error = "dc_sign_horz_vert write failed";
    return false;
  }
  // c = 0: dc_sign, ctx 0.
  if (!mb::write_dc_sign(encoder, mb::dc_sign_cdf_slot(c.dc_sign, 0, 0, 0), 0,
                         0)) {
    error = "dc_sign write failed";
    return false;
  }
  return true;
}

bool code_chroma(av2::Encoder& encoder, Cdfs& c, std::string& error) {
  const int tsc =
      (m1::kTxSizeSqr[kTx16x8] + m1::kTxSizeSqrUp[kTx16x8] + 1) >> 1;
  // U: all_zero = 1, txb_skip ctx = 6.
  if (!sym(encoder, &c.txb_skip[0][tsc][6][0], 2, 1, error)) return false;
  // V: all_zero = 1, ctx = 0.
  if (!sym(encoder, &c.v_txb_skip[0][0], 2, 1, error)) return false;
  return true;
}

bool encode_leaf(av2::Encoder& encoder, Cdfs& c, Variant variant,
                 std::string& error) {
  // DC_PRED: YModeSet symbol 0, YModeIndex ctx 0 symbol 0.
  if (!sym(encoder, &c.y_mode_set[0], 4, 0, error)) return false;
  if (!sym(encoder, &c.y_mode_index[0][0], 8, 0, error)) return false;
  // is_cfl = 0, UV mode DC.
  if (!sym(encoder, &c.is_cfl[0][0], 2, 0, error)) return false;
  if (!sym(encoder, &c.uv_mode[0][0], 8, 0, error)) return false;
  if (!code_luma(encoder, c, variant, error)) return false;
  if (!code_chroma(encoder, c, error)) return false;
  return true;
}

bool build_tile(const p::FrameGeometry& geometry, const pe::Tools& tools,
                const pe::FrameDecision& frame, Variant variant,
                std::vector<std::uint8_t>& tile, std::string& error) {
  ps::TilePartitionCdfs partition_cdfs;
  ps::init_tile_partition_cdfs(partition_cdfs);
  Cdfs cdfs;
  init_cdfs(cdfs);

  std::vector<std::uint8_t> buffer(1u << 20);
  av2::Encoder encoder;
  if (encoder.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    error = "entropy init failed";
    return false;
  }

  ps::FrameStream stream;
  if (!ps::build_frame_stream(geometry, tools, frame, stream, error)) {
    return false;
  }

  p::MiSizeGrid grid(geometry);
  pe::SyntaxStats stats{};
  int leaves = 0;
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
      ++leaves;
      if (leaves > 1) {
        error = "expected exactly one leaf";
        return false;
      }
      if (!encode_leaf(encoder, cdfs, variant, error)) return false;
      grid.fill_block(node.mi_row, node.mi_col,
                      static_cast<p::BlockSize>(node.bsize));
    }
  }
  if (leaves != 1) {
    error = "expected exactly one leaf, got " + std::to_string(leaves);
    return false;
  }

  std::size_t payload_bytes = 0;
  if (encoder.exit_symbol(&payload_bytes) != av2::Status::kOk) {
    error = "entropy exit failed";
    return false;
  }
  tile.assign(buffer.begin(),
              buffer.begin() + static_cast<std::ptrdiff_t>(payload_bytes));
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
  tools.enable_ext_partitions = false;
  tools.enable_uneven_4way_partitions = false;

  pe::DecisionOptions options;
  options.min_block_size_px = 4;
  options.max_depth = 4;
  const pe::FrameDecision frame =
      pe::decide_frame(geometry, force_cost(), tools, options);
  if (frame.status != pe::DecisionStatus::kOk) {
    std::cerr << "partition decision failed: " << frame.error << '\n';
    return 1;
  }
  std::string error;
  if (!pe::validate_frame(geometry, tools, frame, error)) {
    std::cerr << "partition frame invalid: " << error << '\n';
    return 1;
  }

  av2enc::m1::M1Config config;
  config.max_frame_width = kWidth;
  config.max_frame_height = kHeight;
  config.base_q_idx = 90;
  config.enable_intra_edge_filter = true;
  config.enable_cfl_intra = true;
  config.enable_chroma_dctonly = true;
  // H_DCT is not IDTX; §5.20.8.3 get_tx_set() does not consult this bit.
  // Keeping it 0 also prevents the decoder from expecting fsc_mode in the
  // intra mode info (§5.20.5.5 allow_fsc_intra()).
  config.enable_idtx_intra = false;
  config.disable_cdf_update = true;

  struct Case {
    Variant variant;
    const char* name;
  };
  const Case cases[] = {
      {Variant::kHvPositive, "hv_pos"},
      {Variant::kHvZero, "hv_zero"},
      {Variant::kHvWrongContext, "hv_ctx1"},
      {Variant::kHvNegative, "hv_neg"},
  };
  for (const Case& test : cases) {
    std::vector<std::uint8_t> tile;
    if (!build_tile(geometry, tools, frame, test.variant, tile, error)) {
      std::cerr << test.name << ": tile failed: " << error << '\n';
      return 1;
    }
    const std::vector<std::uint8_t> payload =
        av2enc::m1::assemble_m1_still_picture(config, tile);
    if (!write_ivf(out_dir / (std::string(test.name) + ".ivf"), payload,
                   error)) {
      std::cerr << test.name << ": ivf failed: " << error << '\n';
      return 1;
    }
    std::cout << test.name << ".ivf: tile=" << tile.size() << " bytes\n";
  }
  std::cout << "wrote " << out_dir << '\n';
  return 0;
}
