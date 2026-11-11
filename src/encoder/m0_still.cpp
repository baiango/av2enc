#include "encoder/m0_still.h"

#include "encoder/m0_cdf_tables.h"
#include "encoder/m0_tables.h"
#include "entropy/av2/av2_entropy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::m0 {
namespace {

// Block sizes from Table 6.22 (the enum order is normative; see
// spec/tables/block_tables.json `enum_block_size`).
constexpr int kBlock8x16 = 4;
constexpr int kBlock16x8 = 5;
constexpr int kBlock16x32 = 7;
constexpr int kBlock16x4 = 20;
constexpr int kBlock32x8 = 22;
constexpr int kBlock32x16 = 8;
constexpr int kBlock32x32 = 9;
constexpr int kBlock32x64 = 10;
constexpr int kBlock64x32 = 11;
constexpr int kBlock64x64 = 12;

// RECT_HORZ / RECT_VERT (§6.19.3.1).
constexpr int kRectHorz = 0;
constexpr int kRectVert = 1;

// Partition_Size_Adjust and Partition_Size_Adjust_Rect_Type (§9.2) for the
// block sizes the M0 tree can reach, including the rectangular sizes produced
// by implied partitions at the frame edge.
constexpr int kSizeAdjust[11] = {2, 3, 3, 3, 2, 1, 1, 1, 1, 13, 11};
constexpr int kRectSizeAdjust[11] = {3, 4, 6, 5, 2, 1, 2, 1, 0, 14, 14};

// Mi_Width_Log2 / Mi_Height_Log2 for the same sizes.
constexpr int kMiLog2W[11] = {3, 3, 4, 4, 3, 2, 2, 1, 2, 3, 2};
constexpr int kMiLog2H[11] = {3, 4, 4, 3, 2, 3, 1, 2, 2, 1, 0};

constexpr int kSizeIndex(int block_size) {
  switch (block_size) {
    case kBlock32x32:
      return 0;
    case kBlock32x64:
      return 1;
    case kBlock64x32:
      return 3;
    case kBlock32x16:
      return 4;
    case kBlock16x32:
      return 5;
    case kBlock16x8:
      return 6;
    case kBlock8x16:
      return 7;
    case 6:  // BLOCK_16X16
      return 8;
    case kBlock32x8:
      return 9;
    case kBlock16x4:
      return 10;
    default:
      return 2;
  }
}

// ---------------------------------------------------------------------------
// Edge-superblock recursion (§5.20.3.2)
// ---------------------------------------------------------------------------
//
// The default M0 tree is 64x64 -> 32x64 -> 32x32 for full superblocks. A
// partial superblock must follow the spec's implied-partition rules instead,
// which can force a different tree (and smaller blocks) at the frame edge.
// These are the block sizes M0 can actually code. Anything below them is a
// hard limitation of this path; `geometry_supported()` reports it so the CLI
// can use M1 instead of emitting a desynchronised stream.
enum M0EdgeBlock : int {
  kM0Block64x64 = 0,
  kM0Block64x32 = 1,
  kM0Block32x64 = 2,
  kM0Block32x32 = 3,
  kM0Block32x16 = 4,
  kM0Block16x32 = 5,
  kM0Block32x8 = 6,
  kM0BlockCount = 7,
};

constexpr int kM0BlockSize[kM0BlockCount] = {
    kBlock64x64, kBlock64x32, kBlock32x64, kBlock32x32,
    kBlock32x16, kBlock16x32, kBlock32x8};
// Width/height in 4x4 units and Mi_*_Log2 for each supported block.
constexpr int kM0W4[kM0BlockCount] = {16, 16, 8, 8, 8, 4, 8};
constexpr int kM0H4[kM0BlockCount] = {16, 8, 16, 8, 4, 8, 2};
constexpr int kM0WLog2[kM0BlockCount] = {4, 4, 3, 3, 3, 2, 3};
constexpr int kM0HLog2[kM0BlockCount] = {4, 3, 4, 3, 2, 3, 1};
// Partition_Subsize for HORZ/VERT. -1 means M0 has no coder for that subsize.
constexpr int kM0HorzSub[kM0BlockCount] = {
    kM0Block64x32, -1, kM0Block32x32, kM0Block32x16,
    kM0Block32x8,  -1, -1};
constexpr int kM0VertSub[kM0BlockCount] = {
    kM0Block32x64, kM0Block32x32, -1, kM0Block16x32,
    -1,            -1,            -1};

struct M0Boundary {
  bool has_rows = false;
  bool has_cols = false;
  bool implied = false;
  int partition = -1;  // kRectHorz / kRectVert when implied.
};

// partition_implied_at_boundary() (§5.20.3.2) for the M0 block set.
M0Boundary m0_boundary(int mi_row, int mi_col, int mi_rows, int mi_cols,
                       int block) {
  M0Boundary out;
  const int w4 = kM0W4[block];
  const int h4 = kM0H4[block];
  out.has_rows = (mi_row + (h4 >> 1)) < mi_rows;
  out.has_cols = (mi_col + (w4 >> 1)) < mi_cols;
  if (out.has_rows && out.has_cols) {
    return out;
  }
  int partition = -1;
  if (w4 == h4) {
    partition = out.has_rows ? kRectVert : kRectHorz;
  } else if (h4 > w4) {
    if (!out.has_rows) {
      partition = kRectHorz;
    } else {
      const bool sub_has_cols = (mi_col + (w4 >> 2)) < mi_cols;
      if (w4 >= 4 && !sub_has_cols) {
        partition = kRectHorz;
      }
    }
  } else {
    if (!out.has_cols) {
      partition = kRectVert;
    } else {
      const bool sub_has_rows = (mi_row + (h4 >> 2)) < mi_rows;
      if (h4 >= 4 && !sub_has_rows) {
        partition = kRectVert;
      }
    }
  }
  out.implied = partition != -1;
  out.partition = partition;
  return out;
}

// One node of the edge tree: either PARTITION_NONE (partition == -1) or a
// HORZ/VERT split with in-frame children in decode order.
struct EdgePlanNode {
  int mi_row = 0;
  int mi_col = 0;
  int block = kM0Block64x64;
  int partition = -1;
  bool forced = false;        // no partition syntax is coded
  bool none_allowed = false;  // PARTITION_NONE is in the spec's allowed set
  bool rect_coded = false;    // rect_type is coded (both orientations legal)
  std::vector<EdgePlanNode> children;
};

// Plans the edge recursion using only blocks M0 can code.  If the spec forces
// (or the best legal choice needs) a smaller block, the plan fails and records
// the missing size so the caller can explain the limitation.
class EdgePlanner {
 public:
  EdgePlanner(int mi_rows, int mi_cols)
      : mi_rows_(mi_rows), mi_cols_(mi_cols) {}

  bool build_superblock(int mi_row, int mi_col, EdgePlanNode* out) {
    missing_.clear();
    missing_list_.clear();
    return build(mi_row, mi_col, kM0Block64x64, out);
  }

  std::string missing_sizes() const {
    std::string result;
    for (const std::string& size : missing_list_) {
      if (!result.empty()) {
        result += ", ";
      }
      result += size;
    }
    return result;
  }

 private:
  // 64x32 and 32x64 are partition nodes only: M0 has no correct transform for
  // them (Adjusted_Tx_Size maps both to TX_32X32, which the M0 residual writer
  // does not implement; its diagnostic block writers use zero residual).
  // Blocks below 32 pixels in either dimension are also excluded: their
  // coefficient/tx-type paths are not part of the validated M0 set, so the CLI
  // falls back to M1 instead of risking a desynchronised stream.
  static bool leaf_codable(int block) {
    switch (block) {
      case kM0Block64x64:
      case kM0Block32x32:
      case kM0Block32x16:
      case kM0Block32x8:
        return true;
      default:
        return false;
    }
  }

  // Nodes that may appear in the planned tree. 64x32/32x64 are split further
  // (they have no valid M0 leaf coder); everything else must be a leaf.
  static bool usable_node(int block) {
    return leaf_codable(block) || block == kM0Block64x32 ||
           block == kM0Block32x64;
  }

  int subsize(int block, int partition) const {
    return partition == kRectHorz ? kM0HorzSub[block] : kM0VertSub[block];
  }

  void note_missing(int block, int partition) {
    const int w4 = partition == kRectVert ? (kM0W4[block] >> 1) : kM0W4[block];
    const int h4 = partition == kRectHorz ? (kM0H4[block] >> 1) : kM0H4[block];
    const std::string name =
        std::to_string(w4 * 4) + "x" + std::to_string(h4 * 4);
    if (missing_.insert(name).second) {
      missing_list_.push_back(name);
    }
  }

  bool build_children(int mi_row, int mi_col, int block, int sub,
                      int partition, std::vector<EdgePlanNode>* out) {
    const int second_row =
        mi_row + (partition == kRectHorz ? (kM0H4[block] >> 1) : 0);
    const int second_col =
        mi_col + (partition == kRectVert ? (kM0W4[block] >> 1) : 0);
    EdgePlanNode child;
    if (mi_row < mi_rows_ && mi_col < mi_cols_) {
      if (!build(mi_row, mi_col, sub, &child)) {
        return false;
      }
      out->push_back(std::move(child));
    }
    EdgePlanNode second;
    if (second_row < mi_rows_ && second_col < mi_cols_) {
      if (!build(second_row, second_col, sub, &second)) {
        return false;
      }
      out->push_back(std::move(second));
    }
    return true;
  }

  bool build(int mi_row, int mi_col, int block, EdgePlanNode* out) {
    const M0Boundary boundary =
        m0_boundary(mi_row, mi_col, mi_rows_, mi_cols_, block);
    out->mi_row = mi_row;
    out->mi_col = mi_col;
    out->block = block;
    out->partition = -1;
    out->forced = false;
    out->none_allowed = boundary.has_rows && boundary.has_cols;
    out->rect_coded = false;
    out->children.clear();

    if (boundary.implied) {
      const int sub = subsize(block, boundary.partition);
      if (sub < 0 || !usable_node(sub)) {
        note_missing(block, boundary.partition);
        return false;
      }
      out->partition = boundary.partition;
      out->forced = true;
      return build_children(mi_row, mi_col, block, sub, boundary.partition,
                            &out->children);
    }
    if (out->none_allowed && leaf_codable(block)) {
      // NONE is legal and M0 can code this block size directly.
      return true;
    }

    // PARTITION_NONE is either not allowed, or this is a 64x32/32x64 node that
    // exists only to be split. Both HORZ and VERT are spec-valid for every
    // block in the M0 set, so rect_type is coded; try HORZ first, then VERT,
    // and keep the first tree M0 can represent.
    for (const int candidate : {kRectHorz, kRectVert}) {
      const int sub = subsize(block, candidate);
      if (sub < 0 || !usable_node(sub)) {
        note_missing(block, candidate);
        continue;
      }
      std::vector<EdgePlanNode> children;
      if (!build_children(mi_row, mi_col, block, sub, candidate, &children)) {
        continue;
      }
      out->partition = candidate;
      out->rect_coded = true;
      out->children = std::move(children);
      return true;
    }
    return false;
  }

  int mi_rows_ = 0;
  int mi_cols_ = 0;
  std::set<std::string> missing_;
  std::vector<std::string> missing_list_;
};

constexpr std::size_t kTileCapacityBytes = 4U * 1024U * 1024U;
// The generated header carries qctx slice 0, i.e. base_q_idx <= 90.
constexpr std::uint8_t kPinnedBaseQ = 90;

struct CdfState {
  std::uint16_t do_split[2][64][3];
  std::uint16_t rect_type[2][64][3];
  std::uint16_t y_mode_set[5];
  std::uint16_t y_mode_index[3][9];
  std::uint16_t uv_mode[2][9];
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
  std::uint16_t eob_pt_1024[3][9];
  std::uint16_t eob_pt_512[3][9];
  std::uint16_t eob_pt_256[3][9];
  std::uint16_t eob_pt_128[3][9];
  std::uint16_t eob_pt_64[3][8];
  std::uint16_t eob_extra[3];
  std::uint16_t coeff_base[5][20][2][5];
  std::uint16_t coeff_base_lf[5][33][2][7];
  std::uint16_t coeff_base_eob[5][4][4];
  std::uint16_t coeff_base_lf_eob[5][4][6];
  std::uint16_t coeff_br[7][5];
  std::uint16_t coeff_br_lf[14][5];
  std::uint16_t coeff_base_uv[12][5];
  std::uint16_t coeff_base_lf_uv[12][7];
  std::uint16_t coeff_base_eob_uv[4][4];
  std::uint16_t coeff_base_lf_eob_uv[4][6];
  std::uint16_t coeff_br_uv[4][5];
  std::uint16_t dc_sign[2][2][3][3];
  std::uint16_t intra_tx_type_set1[3][8];
  std::uint16_t intra_tx_type_long[4][5];
  std::uint16_t is_long_side_dct[2][3];
};

// §9.3 default CDFs for the luma transform-type syntax. These are needed only
// for the rectangular coding blocks the edge recursion can reach (32x16,
// 32x8, 16x32); square 32x32 and 64x64 are TX_SET_DCTONLY and code nothing.
constexpr std::uint16_t kDefaultIntraTxTypeLongCdf[4][5] = {
    {10692, 26586, 29231, 31, 0},
    {26700, 32160, 32748, 6, 0},
    {26915, 32411, 32748, 6, 0},
    {8192, 16384, 24576, 0, 0}};
constexpr std::uint16_t kDefaultIsLongSideDctCdf[2][3] = {
    {36, 0, 0},
    {128, 31, 0}};

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

void init_cdfs(CdfState& cdfs) {
  copy_array(cdfs.do_split, kDoSplitCdf);
  copy_array(cdfs.rect_type, kRectTypeCdf);
  copy_array(cdfs.y_mode_set, kYModeSetCdf);
  copy_array(cdfs.y_mode_index, kYModeIndexCdf);
  copy_array(cdfs.uv_mode, kUvModeCflNotAllowedCdf);
  copy_array(cdfs.txb_skip, kTxbSkipCdf);
  copy_array(cdfs.v_txb_skip, kVTxbSkipCdf);
  copy_array(cdfs.eob_pt_1024, kEobPt1024Cdf);
  copy_array(cdfs.eob_pt_512, kEobPt512Cdf);
  copy_array(cdfs.eob_pt_256, kEobPt256Cdf);
  copy_array(cdfs.eob_pt_128, kEobPt128Cdf);
  copy_array(cdfs.eob_pt_64, kEobPt64Cdf);
  copy_array(cdfs.eob_extra, kEobExtraCdf);
  copy_array(cdfs.coeff_base, kCoeffBaseCdf);
  copy_array(cdfs.coeff_base_lf, kCoeffBaseLfCdf);
  copy_array(cdfs.coeff_base_eob, kCoeffBaseEobCdf);
  copy_array(cdfs.coeff_base_lf_eob, kCoeffBaseLfEobCdf);
  copy_array(cdfs.coeff_br, kCoeffBrCdf);
  copy_array(cdfs.coeff_br_lf, kCoeffBrLfCdf);
  copy_array(cdfs.coeff_base_uv, kCoeffBaseUvCdf);
  copy_array(cdfs.coeff_base_lf_uv, kCoeffBaseLfUvCdf);
  copy_array(cdfs.coeff_base_eob_uv, kCoeffBaseEobUvCdf);
  copy_array(cdfs.coeff_base_lf_eob_uv, kCoeffBaseLfEobUvCdf);
  copy_array(cdfs.coeff_br_uv, kCoeffBrUvCdf);
  copy_array(cdfs.dc_sign, kDcSignCdf);
  copy_array(cdfs.intra_tx_type_set1, kIntraTxTypeSet1Cdf);
  copy_array(cdfs.intra_tx_type_long, kDefaultIntraTxTypeLongCdf);
  copy_array(cdfs.is_long_side_dct, kDefaultIsLongSideDctCdf);
}

struct Plane {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> samples;

  void resize(int w, int h, std::uint8_t value) {
    width = w;
    height = h;
    samples.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), value);
  }

  std::uint8_t& at(int x, int y) {
    return samples[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)];
  }
  std::uint8_t at(int x, int y) const {
    return samples[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)];
  }
};

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

int clip1(int value) { return clip3(0, 255, value); }

// §4.8: Round2(x, n) = (x + (1 << (n - 1))) >> n (arithmetic shift).
int round2(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// §7.13.2.10 DC prediction uses approx_divide(sum, w + h). §7.13.3.22
// resolve_divisor() supplies a 9-bit reciprocal from Div_Lut; only den = 0
// (no division) and powers of two (f = 0) collapsed to a constant in the
// original M0 square-block paths. Rectangular coding blocks have
// w + h = 24, 40, 48, ... and need the real table.
constexpr int kDivLutBits = 7;
constexpr int kDivLutPrecBits = 9;
constexpr int kDivLutNum = 129;
constexpr int kDivLut[kDivLutNum] = {
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465, 462,
    458, 455, 452, 449, 446, 443, 440, 437, 434, 431, 428, 426, 423, 420, 417,
    415, 412, 410, 407, 405, 402, 400, 397, 395, 392, 390, 388, 386, 383, 381,
    379, 377, 374, 372, 370, 368, 366, 364, 362, 360, 358, 356, 354, 352, 350,
    349, 347, 345, 343, 341, 340, 338, 336, 334, 333, 331, 329, 328, 326, 324,
    323, 321, 320, 318, 317, 315, 314, 312, 311, 309, 308, 306, 305, 303, 302,
    301, 299, 298, 297, 295, 294, 293, 291, 290, 289, 287, 286, 285, 284, 282,
    281, 280, 279, 278, 277, 275, 274, 273, 272, 271, 270, 269, 267, 266, 265,
    264, 263, 262, 261, 260, 259, 258, 257, 256};

// §7.13.2.10 / §7.13.3.22.
int approx_divide(int numerator, int denominator) {
  const int abs_denominator = std::abs(denominator);
  const int n = floor_log2(abs_denominator);
  const int e = abs_denominator - (1 << n);
  const int f =
      n > kDivLutBits ? round2(e, n - kDivLutBits)
                      : e << (kDivLutBits - n);
  int scale = kDivLut[f];
  if (denominator < 0) {
    scale = -scale;
  }
  return round2(numerator * scale, n + kDivLutPrecBits);
}

class TileEncoder {
 public:
  TileEncoder(const StillPicture& picture, const StillPictureConfig& config,
              M0PartitionMode partition_mode)
      : picture_(picture), config_(config), partition_mode_(partition_mode) {
    mi_cols_ = 2 * ((picture.width + 7U) >> 3U);
    mi_rows_ = 2 * ((picture.height + 7U) >> 3U);
    chroma_width_ = (picture.width + 1U) / 2U;
    chroma_height_ = (picture.height + 1U) / 2U;

    const int padded_width = static_cast<int>((picture.width + 63U) & ~63U);
    const int padded_height = static_cast<int>((picture.height + 63U) & ~63U);
    const int padded_mi_cols = ((mi_cols_ + 15) / 16) * 16;
    const int padded_mi_rows = ((mi_rows_ + 15) / 16) * 16;
    intra_joint_cols_ = padded_mi_cols;

    init_cdfs(cdfs_);
    real_coeffs_ = partition_mode == M0PartitionMode::k32x32 ||
                   partition_mode == M0PartitionMode::k32x32HorzFirst;
    tile_buffer_.resize(kTileCapacityBytes);
    enc_.init_symbol(tile_buffer_.data(), tile_buffer_.size());
    enc_.disable_cdf_update = config.cdf_update_disabled() ? 1 : 0;

    recon_[0].resize(padded_width, padded_height, 128);
    recon_[1].resize((padded_width + 1) / 2, (padded_height + 1) / 2, 128);
    recon_[2].resize((padded_width + 1) / 2, (padded_height + 1) / 2, 128);

    for (int plane = 0; plane < 3; ++plane) {
      above_level_[plane].assign(static_cast<std::size_t>(mi_cols_) + 8U, 0);
      left_level_[plane].assign(static_cast<std::size_t>(mi_rows_) + 8U, 0);
      above_dc_[plane].assign(static_cast<std::size_t>(mi_cols_) + 8U, 0);
      left_dc_[plane].assign(static_cast<std::size_t>(mi_rows_) + 8U, 0);
    }
    for (int plane = 0; plane < 2; ++plane) {
      above_mi_[plane].assign(static_cast<std::size_t>(mi_cols_) + 8U, 6);
      left_mi_[plane].assign(static_cast<std::size_t>(mi_rows_) + 8U, 6);
    }
    intra_joint_.assign(static_cast<std::size_t>(padded_mi_rows) *
                            static_cast<std::size_t>(padded_mi_cols),
                        0);
    clear_above_context();
  }

  bool encode(std::vector<std::uint8_t>* tile_payload, std::string& error) {
    clear_above_context();
    for (int r = 0; r < mi_rows_; r += 16) {
      clear_left_context();
      for (int c = 0; c < mi_cols_; c += 16) {
        bool ok = false;
        const bool edge_sb = (r + 16 > mi_rows_) || (c + 16 > mi_cols_);
        if (partition_mode_ == M0PartitionMode::k32x32 && edge_sb) {
          // A partial superblock must follow the spec's implied-partition
          // rules; PARTITION_NONE at 64x64 is only legal when both half-block
          // bounds stay inside the frame. encode_edge_superblock() plans the
          // exact §5.20.3.2 tree for this SB.
          ok = encode_edge_superblock(r, c);
        } else {
        switch (partition_mode_) {
          case M0PartitionMode::k64x64Single:
            ok = encode_partition_none_64(r, c);
            break;
          case M0PartitionMode::k64x64VertNone:
            ok = encode_partition_64_rect(r, c, kRectVert, false);
            break;
          case M0PartitionMode::k64x64HorzNone:
            ok = encode_partition_64_rect(r, c, kRectHorz, false);
            break;
          case M0PartitionMode::k32x32HorzFirst:
            ok = encode_partition_64_rect(r, c, kRectHorz, true);
            break;
          case M0PartitionMode::k32x32:
          default:
            ok = encode_partition_64_rect(r, c, kRectVert, true);
            break;
        }
        }
        if (!ok) {
          error = error_;
          return false;
        }
      }
    }
    std::size_t payload_bytes = 0;
    const av2::Status status = enc_.exit_symbol(&payload_bytes);
    if (status != av2::Status::kOk) {
      error = std::string("entropy exit_symbol failed: ") + av2::status_name(status);
      return false;
    }
    tile_payload->assign(tile_buffer_.begin(),
                         tile_buffer_.begin() + static_cast<std::ptrdiff_t>(payload_bytes));
    return true;
  }

  const Plane& recon(int plane) const { return recon_[plane]; }

 private:
  bool sym(std::uint16_t* cdf, int n, int symbol) {
    const av2::Status status = enc_.encode_symbol(cdf, n, symbol);
    if (status != av2::Status::kOk) {
      error_ = std::string("entropy encode_symbol failed: ") + av2::status_name(status);
      return false;
    }
    return true;
  }

  bool raw_bit(bool bit) {
    const av2::Status status = enc_.encode_bool(bit);
    if (status != av2::Status::kOk) {
      error_ = std::string("entropy encode_bool failed: ") + av2::status_name(status);
      return false;
    }
    return true;
  }

  void clear_above_context() {
    for (int plane = 0; plane < 3; ++plane) {
      std::fill(above_level_[plane].begin(), above_level_[plane].end(), 0);
      std::fill(above_dc_[plane].begin(), above_dc_[plane].end(), 0);
    }
    for (int plane = 0; plane < 2; ++plane) {
      std::fill(above_mi_[plane].begin(), above_mi_[plane].end(), 6);
    }
  }

  void clear_left_context() {
    for (int plane = 0; plane < 3; ++plane) {
      std::fill(left_level_[plane].begin(), left_level_[plane].end(), 0);
      std::fill(left_dc_[plane].begin(), left_dc_[plane].end(), 0);
    }
    for (int plane = 0; plane < 2; ++plane) {
      std::fill(left_mi_[plane].begin(), left_mi_[plane].end(), 6);
    }
  }

  std::uint8_t& above_level(int plane, int x) {
    return above_level_[plane][static_cast<std::size_t>(x)];
  }
  std::uint8_t& left_level(int plane, int y) {
    return left_level_[plane][static_cast<std::size_t>(y)];
  }
  std::uint8_t& above_dc(int plane, int x) {
    return above_dc_[plane][static_cast<std::size_t>(x)];
  }
  std::uint8_t& left_dc(int plane, int y) {
    return left_dc_[plane][static_cast<std::size_t>(y)];
  }
  std::uint8_t& above_mi(int plane, int x) {
    return above_mi_[plane][static_cast<std::size_t>(x)];
  }
  std::uint8_t& left_mi(int plane, int y) {
    return left_mi_[plane][static_cast<std::size_t>(y)];
  }
  std::uint8_t& intra_joint(int r, int c) {
    return intra_joint_[static_cast<std::size_t>(r) * static_cast<std::size_t>(intra_joint_cols_) +
                        static_cast<std::size_t>(c)];
  }

  bool is_inside(int r, int c) const {
    return r >= 0 && r < mi_rows_ && c >= 0 && c < mi_cols_;
  }

  // get_joint_mode(dir) (§5.20.5.5).
  int get_joint_mode(int r, int c, int w4, int h4, int dir) const {
    int mv_row = 0;
    int mv_col = 0;
    if (dir != 0) {
      mv_row = r - 1;
      mv_col = c + w4 - 1;
    } else {
      mv_col = c - 1;
      mv_row = r + h4 - 1;
    }
    if (is_inside(mv_row, mv_col)) {
      return intra_joint_[static_cast<std::size_t>(mv_row) *
                              static_cast<std::size_t>(intra_joint_cols_) +
                          static_cast<std::size_t>(mv_col)];
    }
    return 0;  // DC_PRED
  }

  bool encode_partition_64_rect(int r, int c, int rect_type,
                                bool split_children) {
    const int idx = kSizeIndex(kBlock64x64);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int do_split_ctx =
        kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
        (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][do_split_ctx][0], 2, 1)) {
      return false;
    }
    const int rect_ctx =
        kRectSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
        (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.rect_type[0][rect_ctx][0], 2, rect_type)) {
      return false;
    }
    if (rect_type == kRectVert) {
      // PARTITION_VERT -> two 32x64 blocks at (r, c) and (r, c + 8).
      if (split_children) {
        return encode_partition_32x64(r, c) && encode_partition_32x64(r, c + 8);
      }
      return encode_partition_none_32x64(r, c) &&
             encode_partition_none_32x64(r, c + 8);
    }
    // PARTITION_HORZ -> two 64x32 blocks at (r, c) and (r + 8, c).
    if (split_children) {
      return encode_partition_64x32(r, c) && encode_partition_64x32(r + 8, c);
    }
    return encode_partition_none_64x32(r, c) &&
           encode_partition_none_64x32(r + 8, c);
  }

  bool encode_partition_none_32x64(int r, int c) {
    const int idx = kSizeIndex(kBlock32x64);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int ctx = kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
                    (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][ctx][0], 2, 0)) {
      return false;
    }
    return encode_block_32x64(r, c);
  }

  bool encode_partition_none_64x32(int r, int c) {
    const int idx = kSizeIndex(kBlock64x32);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int ctx = kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
                    (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][ctx][0], 2, 0)) {
      return false;
    }
    return encode_block_64x32(r, c);
  }

  bool encode_partition_64x32(int r, int c) {
    // 64x32 -> two 32x32 blocks by VERT.
    const int idx = kSizeIndex(kBlock64x32);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int ctx = kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
                    (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][ctx][0], 2, 1)) {
      return false;
    }
    const int rect_ctx = kRectSizeAdjust[idx] * 4 +
                         (left_mi(0, r) < bsh ? 2 : 0) +
                         (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.rect_type[0][rect_ctx][0], 2, kRectVert)) {
      return false;
    }
    return encode_partition_32x32(r, c) && encode_partition_32x32(r, c + 8);
  }

  bool encode_partition_32x64(int r, int c) {
    const int idx = kSizeIndex(kBlock32x64);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int do_split_ctx =
        kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
        (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][do_split_ctx][0], 2, 1)) {
      return false;
    }
    const int rect_ctx =
        kRectSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
        (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.rect_type[0][rect_ctx][0], 2, kRectHorz)) {
      return false;
    }
    // PARTITION_HORZ -> two 32x32 blocks at (r, c) and (r + 8, c).
    return encode_partition_32x32(r, c) && encode_partition_32x32(r + 8, c);
  }

  // §5.20.3.2 partition_implied() for the square/rectangular sizes M0 uses.
  bool partition_implied(int r, int c, int block_size, int* implied) const {
    const int idx = kSizeIndex(block_size);
    const int w4 = 1 << kMiLog2W[idx];
    const int h4 = 1 << kMiLog2H[idx];
    const bool has_rows = (r + (h4 >> 1)) < mi_rows_;
    const bool has_cols = (c + (w4 >> 1)) < mi_cols_;
    if (has_rows && has_cols) {
      *implied = 0;
      return false;
    }
    int part = 0;
    if (w4 == h4) {
      part = has_rows ? kRectVert : kRectHorz;
    } else if (h4 > w4) {
      if (!has_rows) {
        part = kRectHorz;
      } else {
        const bool sub_has_cols = (c + (w4 >> 2)) < mi_cols_;
        if (w4 >= 4 && !sub_has_cols) {
          part = kRectHorz;
        }
      }
    } else {
      if (!has_cols) {
        part = kRectVert;
      } else {
        const bool sub_has_rows = (r + (h4 >> 2)) < mi_rows_;
        if (h4 >= 4 && !sub_has_rows) {
          part = kRectVert;
        }
      }
    }
    *implied = part;
    return part != 0;
  }

  bool encode_partition_32x32(int r, int c) {
    int implied = 0;
    if (partition_implied(r, c, kBlock32x32, &implied)) {
      // The decoder infers this partition without reading do_split/rect_type.
      if (implied == kRectHorz) {
        if (r < mi_rows_ && !encode_partition_32x16(r, c)) {
          return false;
        }
        if (r + 4 < mi_rows_ && !encode_partition_32x16(r + 4, c)) {
          return false;
        }
        return true;
      }
      if (c < mi_cols_ && !encode_partition_16x32(r, c)) {
        return false;
      }
      if (c + 4 < mi_cols_ && !encode_partition_16x32(r, c + 4)) {
        return false;
      }
      return true;
    }
    const int idx = kSizeIndex(kBlock32x32);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int do_split_ctx =
        kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
        (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][do_split_ctx][0], 2, 0)) {
      return false;
    }
    return encode_block_32x32(r, c);
  }

  bool encode_partition_32x16(int r, int c) {
    const int idx = kSizeIndex(kBlock32x16);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int w4 = 1 << bsw;
    const int h4 = 1 << bsh;
    const bool has_rows = (r + (h4 >> 1)) < mi_rows_;
    const bool has_cols = (c + (w4 >> 1)) < mi_cols_;
    if (!has_rows && has_cols) {
      // PARTITION_NONE is not allowed at the bottom edge. Both HORZ and VERT
      // are allowed, so rect_type is coded; choose HORZ and keep shrinking the
      // block height (32x16 -> 32x8).
      const int rect_ctx =
          kRectSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
          (above_mi(0, c) < bsw ? 1 : 0);
      if (!sym(&cdfs_.rect_type[0][rect_ctx][0], 2, kRectHorz)) {
        return false;
      }
      if (r < mi_rows_ && !encode_partition_32x8(r, c)) {
        return false;
      }
      if (r + 2 < mi_rows_ && !encode_partition_32x8(r + 2, c)) {
        return false;
      }
      return true;
    }
    const int do_split_ctx =
        kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
        (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][do_split_ctx][0], 2, 0)) {
      return false;
    }
    return encode_block_32x16(r, c);
  }

  bool encode_partition_32x8(int r, int c) {
    const int idx = kSizeIndex(kBlock32x8);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int do_split_ctx =
        kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
        (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][do_split_ctx][0], 2, 0)) {
      return false;
    }
    return encode_block_32x8(r, c);
  }

  bool encode_partition_16x32(int r, int c) {
    const int idx = kSizeIndex(kBlock16x32);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int do_split_ctx =
        kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
        (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][do_split_ctx][0], 2, 0)) {
      return false;
    }
    return encode_block_16x32(r, c);
  }

  bool encode_partition_none_64(int r, int c) {
    const int idx = kSizeIndex(kBlock64x64);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int do_split_ctx =
        kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
        (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][do_split_ctx][0], 2, 0)) {
      return false;
    }
    return encode_block_64x64(r, c);
  }

  int partition_do_split_context(int r, int c, int block) {
    const int size_index = kSizeIndex(kM0BlockSize[block]);
    const int bsh = kM0HLog2[block];
    const int bsw = kM0WLog2[block];
    return kSizeAdjust[size_index] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
           (above_mi(0, c) < bsw ? 1 : 0);
  }

  int partition_rect_type_context(int r, int c, int block) {
    const int size_index = kSizeIndex(kM0BlockSize[block]);
    const int bsh = kM0HLog2[block];
    const int bsw = kM0WLog2[block];
    return kRectSizeAdjust[size_index] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
           (above_mi(0, c) < bsw ? 1 : 0);
  }

  bool encode_block_by_edge_index(int block, int r, int c) {
    switch (block) {
      case kM0Block64x64:
        return encode_block_64x64(r, c);
      case kM0Block64x32:
        return encode_block_64x32(r, c);
      case kM0Block32x64:
        return encode_block_32x64(r, c);
      case kM0Block32x32:
        return encode_block_32x32(r, c);
      case kM0Block32x16:
        return encode_block_32x16(r, c);
      case kM0Block16x32:
        return encode_block_16x32(r, c);
      case kM0Block32x8:
        return encode_block_32x8(r, c);
      default:
        error_ = "M0 internal error: unhandled edge block size";
        return false;
    }
  }

  // Writes the partition syntax (if any) for one planned edge node and recurses
  // in decode order. This mirrors read_partition() §5.20.3.2:
  //   * an implied partition, or a single allowed one, codes no bits;
  //   * otherwise do_split is coded only when PARTITION_NONE is allowed;
  //   * SPLIT is never legal below a 128x128 root, so do_square_split is
  //     implied 0 and not coded;
  //   * rect_type is coded when both HORZ and VERT are legal.
  bool encode_edge_plan(const EdgePlanNode& node) {
    if (!node.forced) {
      if (node.none_allowed) {
        const int do_split = node.partition < 0 ? 0 : 1;
        const int ctx = partition_do_split_context(node.mi_row, node.mi_col,
                                                   node.block);
        if (!sym(&cdfs_.do_split[0][ctx][0], 2, do_split)) {
          return false;
        }
      }
      if (node.partition >= 0 && node.rect_coded) {
        const int ctx = partition_rect_type_context(
            node.mi_row, node.mi_col, node.block);
        if (!sym(&cdfs_.rect_type[0][ctx][0], 2, node.partition)) {
          return false;
        }
      }
    }
    if (node.partition < 0) {
      return encode_block_by_edge_index(node.block, node.mi_row, node.mi_col);
    }
    for (const EdgePlanNode& child : node.children) {
      if (!encode_edge_plan(child)) {
        return false;
      }
    }
    return true;
  }

  bool encode_edge_superblock(int r, int c) {
    EdgePlanner planner(mi_rows_, mi_cols_);
    EdgePlanNode plan;
    if (!planner.build_superblock(r, c, &plan)) {
      error_ = "M0 cannot represent this edge superblock without uncoded "
               "block size(s): " +
               planner.missing_sizes();
      return false;
    }
    return encode_edge_plan(plan);
  }

  bool encode_mode_info_dc(int r, int c, int w4, int h4) {
    // mode_info() -> intra_frame_mode_info() (§5.20.5.3) for DC_PRED.
    if (!sym(&cdfs_.y_mode_set[0], 4, 0)) {
      return false;
    }
    const int joint_ctx = (get_joint_mode(r, c, w4, h4, 0) >= 5 ? 1 : 0) +
                          (get_joint_mode(r, c, w4, h4, 1) >= 5 ? 1 : 0);
    if (!sym(&cdfs_.y_mode_index[joint_ctx][0], 8, 0)) {
      return false;
    }
    return sym(&cdfs_.uv_mode[0][0], 8, 0);
  }

  bool encode_block_32x32(int r, int c) {
    if (!encode_mode_info_dc(r, c, 8, 8)) {
      return false;
    }

    // read_block_tx_size(): TxMode == TX_MODE_LARGEST, no syntax.
    // residual(): plane 0 is one TX_32X32; chroma planes are one TX_16X16.
    if (!encode_residual_plane(0, r, c, 32, 32, 3, 32, 32)) {
      return false;
    }
    if (!encode_residual_plane(1, r, c, 16, 16, 2, 16, 16)) {
      return false;
    }
    if (!encode_residual_plane(2, r, c, 16, 16, 2, 16, 16)) {
      return false;
    }

    // decode_block() stores MiSize for the mode/partition contexts.
    for (int y = 0; y < 8; ++y) {
      left_mi(0, r + y) = 3;  // Mi_Height_Log2[BLOCK_32X32]
      above_mi(0, c + y) = 3;  // Mi_Width_Log2[BLOCK_32X32]
    }
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        intra_joint(r + y, c + x) = 0;  // DC_PRED
      }
    }
    return true;
  }

  bool encode_block_32x16(int r, int c) {
    if (!encode_mode_info_dc(r, c, 8, 4)) {
      return false;
    }
    // TX_MODE_LARGEST -> TX_32X16 luma, TX_16X8 chroma.
    if (!encode_residual_plane(0, r, c, 32, 16, 3, 32, 16)) {
      return false;
    }
    if (!encode_residual_plane(1, r, c, 16, 8, 2, 16, 8)) {
      return false;
    }
    if (!encode_residual_plane(2, r, c, 16, 8, 2, 16, 8)) {
      return false;
    }
    for (int y = 0; y < 4; ++y) {
      left_mi(0, r + y) = 2;  // Mi_Height_Log2[BLOCK_32X16]
    }
    for (int x = 0; x < 8; ++x) {
      above_mi(0, c + x) = 3;  // Mi_Width_Log2[BLOCK_32X16]
    }
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 8; ++x) {
        intra_joint(r + y, c + x) = 0;
      }
    }
    return true;
  }

  bool encode_block_32x8(int r, int c) {
    if (!encode_mode_info_dc(r, c, 8, 2)) {
      return false;
    }
    // TX_MODE_LARGEST -> TX_32X8 luma, TX_16X4 chroma.
    if (!encode_residual_plane(0, r, c, 32, 8, 2, 32, 8)) {
      return false;
    }
    if (!encode_residual_plane(1, r, c, 16, 4, 1, 16, 4)) {
      return false;
    }
    if (!encode_residual_plane(2, r, c, 16, 4, 1, 16, 4)) {
      return false;
    }
    for (int y = 0; y < 2; ++y) {
      left_mi(0, r + y) = 1;  // Mi_Height_Log2[BLOCK_32X8]
    }
    for (int x = 0; x < 8; ++x) {
      above_mi(0, c + x) = 3;  // Mi_Width_Log2[BLOCK_32X8]
    }
    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 8; ++x) {
        intra_joint(r + y, c + x) = 0;
      }
    }
    return true;
  }

  bool encode_block_16x32(int r, int c) {
    if (!encode_mode_info_dc(r, c, 4, 8)) {
      return false;
    }
    // TX_MODE_LARGEST -> TX_16X32 luma, TX_8X16 chroma.
    if (!encode_residual_plane(0, r, c, 16, 32, 3, 16, 32)) {
      return false;
    }
    if (!encode_residual_plane(1, r, c, 8, 16, 2, 8, 16)) {
      return false;
    }
    if (!encode_residual_plane(2, r, c, 8, 16, 2, 8, 16)) {
      return false;
    }
    for (int y = 0; y < 8; ++y) {
      left_mi(0, r + y) = 3;  // Mi_Height_Log2[BLOCK_16X32]
    }
    for (int x = 0; x < 4; ++x) {
      above_mi(0, c + x) = 2;  // Mi_Width_Log2[BLOCK_16X32]
    }
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 4; ++x) {
        intra_joint(r + y, c + x) = 0;
      }
    }
    return true;
  }

  bool encode_block_32x64(int r, int c) {
    if (!encode_mode_info_dc(r, c, 8, 16)) {
      return false;
    }
    // TX_MODE_LARGEST -> TX_32X64 luma, TX_16X32 chroma.
    if (!encode_residual_plane(0, r, c, 32, 64, 4, 32, 64)) {
      return false;
    }
    if (!encode_residual_plane(1, r, c, 16, 32, 3, 16, 32)) {
      return false;
    }
    if (!encode_residual_plane(2, r, c, 16, 32, 3, 16, 32)) {
      return false;
    }
    for (int y = 0; y < 16; ++y) {
      left_mi(0, r + y) = 4;  // Mi_Height_Log2[BLOCK_32X64]
    }
    for (int x = 0; x < 8; ++x) {
      above_mi(0, c + x) = 3;  // Mi_Width_Log2[BLOCK_32X64]
    }
    for (int y = 0; y < 16; ++y) {
      for (int x = 0; x < 8; ++x) {
        intra_joint(r + y, c + x) = 0;
      }
    }
    return true;
  }

  bool encode_block_64x32(int r, int c) {
    if (!encode_mode_info_dc(r, c, 16, 8)) {
      return false;
    }
    // TX_MODE_LARGEST -> TX_64X32 luma, TX_32X16 chroma.
    if (!encode_residual_plane(0, r, c, 64, 32, 4, 64, 32)) {
      return false;
    }
    if (!encode_residual_plane(1, r, c, 32, 16, 3, 32, 16)) {
      return false;
    }
    if (!encode_residual_plane(2, r, c, 32, 16, 3, 32, 16)) {
      return false;
    }
    for (int y = 0; y < 8; ++y) {
      left_mi(0, r + y) = 3;   // Mi_Height_Log2[BLOCK_64X32]
    }
    for (int x = 0; x < 16; ++x) {
      above_mi(0, c + x) = 4;  // Mi_Width_Log2[BLOCK_64X32]
    }
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 16; ++x) {
        intra_joint(r + y, c + x) = 0;
      }
    }
    return true;
  }

  bool encode_block_64x64(int r, int c) {
    if (!encode_mode_info_dc(r, c, 16, 16)) {
      return false;
    }
    // TX_MODE_LARGEST -> TX_64X64 luma, TX_32X32 chroma.
    if (!encode_residual_plane(0, r, c, 64, 64, 4, 64, 64)) {
      return false;
    }
    if (!encode_residual_plane(1, r, c, 32, 32, 3, 32, 32)) {
      return false;
    }
    if (!encode_residual_plane(2, r, c, 32, 32, 3, 32, 32)) {
      return false;
    }
    for (int y = 0; y < 16; ++y) {
      left_mi(0, r + y) = 4;   // Mi_Height_Log2[BLOCK_64X64]
      above_mi(0, c + y) = 4;  // Mi_Width_Log2[BLOCK_64X64]
    }
    for (int y = 0; y < 16; ++y) {
      for (int x = 0; x < 16; ++x) {
        intra_joint(r + y, c + x) = 0;
      }
    }
    return true;
  }

  std::uint8_t source_sample(int plane, int x, int y) const {
    const std::span<const std::uint8_t> data =
        plane == 0 ? picture_.y : (plane == 1 ? picture_.u : picture_.v);
    const int stride = plane == 0 ? static_cast<int>(picture_.width)
                                  : static_cast<int>(chroma_width_);
    const int max_x = (plane == 0 ? static_cast<int>(picture_.width)
                                  : static_cast<int>(chroma_width_)) - 1;
    const int max_y = (plane == 0 ? static_cast<int>(picture_.height)
                                  : static_cast<int>(chroma_height_)) - 1;
    const int cx = clip3(0, max_x, x);
    const int cy = clip3(0, max_y, y);
    return data[static_cast<std::size_t>(cy) * static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(cx)];
  }

  bool write_literal(int bits, std::uint32_t value) {
    for (int i = bits - 1; i >= 0; --i) {
      if (!raw_bit(((value >> static_cast<unsigned>(i)) & 1U) != 0U)) {
        return false;
      }
    }
    return true;
  }

  // Encoder inverse of read_quant() (§5.20.7.28) for allowTcq = 0 and
  // isHidden = 0. `x` is the value added to the pre-escape level; for a
  // DC-only block there is at most one escape, so hr_level_avg_ starts at 0.
  bool write_quant_escape(int x) {
    const int pred_level = hr_level_avg_;
    const int m = clip3(1, 6, pred_level <= 0 ? 0 : floor_log2(pred_level));
    const int k = m + 1;
    const int c_max = std::min(m + 4, 6);
    if (x < (c_max << m)) {
      const int q = x >> m;
      for (int i = 0; i < q; ++i) {
        if (!raw_bit(false)) {
          return false;
        }
      }
      if (!raw_bit(true)) {
        return false;
      }
      if (!write_literal(m, static_cast<std::uint32_t>(x & ((1 << m) - 1)))) {
        return false;
      }
    } else {
      for (int i = 0; i < c_max; ++i) {
        if (!raw_bit(false)) {
          return false;
        }
      }
      int g = 0;
      while (true) {
        const int length = k + g;
        const int x_base = (c_max << m) + (1 << length) - (1 << k);
        if (x < x_base + (1 << length)) {
          for (int i = 0; i < g; ++i) {
            if (!raw_bit(false)) {
              return false;
            }
          }
          if (!raw_bit(true)) {
            return false;
          }
          if (!write_literal(length, static_cast<std::uint32_t>(x - x_base))) {
            return false;
          }
          break;
        }
        ++g;
      }
    }
    hr_level_avg_ = (x + hr_level_avg_) >> 1;
    return true;
  }

  static const std::int16_t* dct_kernel(int n) {
    switch (n) {
      case 4:
        return &kDct4[0][0];
      case 8:
        return &kDct8[0][0];
      case 16:
        return &kDct16[0][0];
      default:
        return &kDct32[0][0];
    }
  }

  static void transform_shifts(int w, int h, int* row_shift,
                               int* col_shift) {
    if (w == 32 && h == 32) {
      *row_shift = 6;
      *col_shift = 13;
    } else if (w == 16 && h == 16) {
      *row_shift = 6;
      *col_shift = 13;
    } else if ((w == 32 && h == 16) || (w == 16 && h == 32)) {
      *row_shift = 6;
      *col_shift = 12;
    } else if ((w == 32 && h == 8) || (w == 8 && h == 32)) {
      *row_shift = 6;
      *col_shift = 13;
    } else if ((w == 16 && h == 4) || (w == 4 && h == 16)) {
      *row_shift = 6;
      *col_shift = 12;
    } else if (w <= 16 && h <= 16) {
      *row_shift = 7;
      *col_shift = 11;
    } else {
      *row_shift = 6;
      *col_shift = 13;
    }
  }

  // §7.15.4.1 rectangular pre-scale. When the log2 width/height parity
  // differs, the normative transform applies a 4096/2896 factor (inverse:
  // Round2(x * 2896, 12)). M0's original 64x64/32x32 paths are both even and
  // never exercised this; the edge recursion does for 32x16 / 16x32 and their
  // chroma sizes.
  static bool odd_rect_transform(int w, int h) {
    return ((floor_log2(w) ^ floor_log2(h)) & 1) != 0;
  }

  // Orthonormal 2D DCT-II, computed through the normative integer kernel:
  // Dct_Kernel{n}[j][i] = 64 * sqrt(n) * B[j][i].
  void forward_dct(const std::vector<int>& block, int w, int h,
                   std::vector<double>* out) const {
    const std::int16_t* kw = dct_kernel(w);
    const std::int16_t* kh = dct_kernel(h);
    const double scale_w = 1.0 / (64.0 * std::sqrt(static_cast<double>(w)));
    const double scale_h = 1.0 / (64.0 * std::sqrt(static_cast<double>(h)));
    std::vector<double> tmp(static_cast<std::size_t>(w) * h, 0.0);
    out->assign(static_cast<std::size_t>(w) * h, 0.0);
    for (int i = 0; i < h; ++i) {
      for (int v = 0; v < w; ++v) {
        double sum = 0.0;
        for (int j = 0; j < w; ++j) {
          sum += static_cast<double>(kw[v * w + j]) *
                 static_cast<double>(block[static_cast<std::size_t>(i) * w + j]);
        }
        tmp[static_cast<std::size_t>(i) * w + v] = sum * scale_w;
      }
    }
    for (int u = 0; u < h; ++u) {
      for (int v = 0; v < w; ++v) {
        double sum = 0.0;
        for (int i = 0; i < h; ++i) {
          sum += static_cast<double>(kh[u * h + i]) *
                 tmp[static_cast<std::size_t>(i) * w + v];
        }
        (*out)[static_cast<std::size_t>(u) * w + v] = sum * scale_h;
      }
    }
    if (odd_rect_transform(w, h)) {
      constexpr double kRectPrescale = 4096.0 / 2896.0;
      for (double& value : *out) {
        value *= kRectPrescale;
      }
    }
  }

  // Normative 2D inverse matrix transform (§7.15.4.1) for rectangular and
  // square TX sizes.
  void inverse_dct(const std::vector<int>& dequant, int w, int h,
                   std::vector<int>* residual) const {
    int row_shift = 0;
    int col_shift = 0;
    transform_shifts(w, h, &row_shift, &col_shift);
    const std::int16_t* kw = dct_kernel(w);
    const std::int16_t* kh = dct_kernel(h);
    const bool odd_rect = odd_rect_transform(w, h);
    const auto prescale = [odd_rect](int value) {
      return odd_rect
                 ? round2(static_cast<int>(
                              static_cast<std::int64_t>(value) * 2896),
                          12)
                 : value;
    };
    std::vector<int> intermediate(static_cast<std::size_t>(w) * h, 0);
    for (int i = 0; i < h; ++i) {
      for (int oj = 0; oj < w; ++oj) {
        std::int64_t sum = 0;
        for (int j = 0; j < w; ++j) {
          sum += static_cast<std::int64_t>(kw[j * w + oj]) *
                 prescale(dequant[static_cast<std::size_t>(i) * w + j]);
        }
        intermediate[static_cast<std::size_t>(i) * w + oj] = clip3(
            -(1 << 15), (1 << 15) - 1,
            round2(static_cast<int>(sum), row_shift));
      }
    }
    residual->assign(static_cast<std::size_t>(w) * h, 0);
    for (int oi = 0; oi < h; ++oi) {
      for (int j = 0; j < w; ++j) {
        std::int64_t sum = 0;
        for (int i = 0; i < h; ++i) {
          sum += static_cast<std::int64_t>(kh[i * h + oi]) *
                 intermediate[static_cast<std::size_t>(i) * w + j];
        }
        (*residual)[static_cast<std::size_t>(oi) * w + j] =
            clip3(-256, 255, round2(static_cast<int>(sum), col_shift));
      }
    }
  }

  static void build_diagonal_scan(int w, int h, std::vector<std::uint16_t>* scan) {
    scan->resize(static_cast<std::size_t>(w) * h);
    int x = 0;
    int y = 0;
    for (int c = 0; c < w * h; ++c) {
      (*scan)[static_cast<std::size_t>(c)] =
          static_cast<std::uint16_t>(y * w + x);
      ++x;
      --y;
      if (y < 0 || x >= w) {
        ++x;
        const int s = std::min(x, h - 1 - y);
        x -= s;
        y += s;
      }
    }
  }

  static int tx_sqr_index(int w, int h) {
    const int m = std::min(w, h);
    if (m >= 32) {
      return 3;
    }
    if (m >= 16) {
      return 2;
    }
    if (m >= 8) {
      return 1;
    }
    return 0;
  }

  static int tx_sqr_up_index(int w, int h) {
    const int m = std::max(w, h);
    if (m >= 64) {
      return 4;
    }
    if (m >= 32) {
      return 3;
    }
    if (m >= 16) {
      return 2;
    }
    if (m >= 8) {
      return 1;
    }
    return 0;
  }

  // §5.20.8.2 transform_type() for intra luma, DC_PRED, DCT_DCT.
  // TX_SET_DCTONLY (32x32 / 64x64) codes nothing, so the original M0 streams
  // are unchanged; rectangular 32-wide/tall sizes select TX_SET_WIDE_32 or
  // TX_SET_HIGH_32 and must code is_long_side_dct = 1 + intra_tx_type = 0
  // (DCT_DCT). TX_SET_INTRA_1 codes intra_tx_type = 0 (DCT_DCT for DC_PRED).
  bool write_luma_tx_type(int w, int h, int eob) {
    if (eob <= 1) {
      return true;  // transform_type() forces DCT_DCT without syntax
    }
    const int sqr = tx_sqr_index(w, h);
    const int sqr_up = tx_sqr_up_index(w, h);
    int set = 5;  // TX_SET_INTRA_1
    if (sqr_up > 3) {
      set = sqr >= 3 ? 0 : (w > h ? 1 : 2);  // DCTONLY / WIDE_64 / HIGH_64
    } else if (sqr_up == 3) {
      set = sqr == 3 ? 0 : (w > h ? 3 : 4);  // DCTONLY / WIDE_32 / HIGH_32
    }
    if (set == 0) {
      return true;
    }
    if (set == 3 || set == 4) {
      if (!sym(&cdfs_.is_long_side_dct[0][0], 2, 1)) {
        return false;
      }
      return sym(&cdfs_.intra_tx_type_long[sqr][0], 4, 0);
    }
    if (set == 1 || set == 2) {
      return sym(&cdfs_.intra_tx_type_long[sqr][0], 4, 0);
    }
    return sym(&cdfs_.intra_tx_type_set1[sqr][0], 7, 0);
  }

  bool write_eob(int eob, int plane, int w, int h) {
    int eob_pt = 1;
    if (eob == 2) {
      eob_pt = 2;
    } else if (eob >= 3) {
      eob_pt = floor_log2(eob - 1) + 2;
    }
    const int multisize = std::min(5, floor_log2(w)) +
                          std::min(5, floor_log2(h)) - 4;
    std::uint16_t* cdf = nullptr;
    int escape_bits = 0;
    int cdf_n = 8;
    const int cdf_ctx = plane > 0 ? 2 : 0;
    switch (multisize) {
      case 2:
        cdf = &cdfs_.eob_pt_64[cdf_ctx][0];
        escape_bits = 0;
        cdf_n = 7;
        break;
      case 3:
        cdf = &cdfs_.eob_pt_128[cdf_ctx][0];
        escape_bits = 0;
        break;
      case 4:
        cdf = &cdfs_.eob_pt_256[cdf_ctx][0];
        escape_bits = 1;
        break;
      case 5:
        cdf = &cdfs_.eob_pt_512[cdf_ctx][0];
        escape_bits = 2;
        break;
      default:
        cdf = &cdfs_.eob_pt_1024[cdf_ctx][0];
        escape_bits = 2;
        break;
    }
    int symbol = 0;
    int escape = 0;
    if (eob_pt <= 8) {
      symbol = eob_pt - 1;
    } else {
      symbol = 7;
      escape = eob_pt - 8;
    }
    if (!sym(cdf, cdf_n, symbol)) {
      return false;
    }
    if (symbol == 7 && escape_bits > 0) {
      if (!write_literal(escape_bits, static_cast<std::uint32_t>(escape))) {
        return false;
      }
    }
    if (eob_pt >= 3) {
      const int base = (1 << (eob_pt - 2)) + 1;
      const int extra = eob - base;
      const int eob_extra = (extra >> (eob_pt - 3)) & 1;
      if (!sym(&cdfs_.eob_extra[0], 2, eob_extra)) {
        return false;
      }
      for (int i = eob_pt - 4; i >= 0; --i) {
        if (!raw_bit(((extra >> i) & 1) != 0)) {
          return false;
        }
      }
    }
    return true;
  }

  int dc_sign_context(int x4, int y4, int w4, int h4) const {
    int dc_sign = 0;
    for (int k = 0; k < w4; ++k) {
      if (x4 + k < mi_cols_) {
        const int sign = above_dc_[0][static_cast<std::size_t>(x4 + k)];
        if (sign == 1) {
          --dc_sign;
        } else if (sign == 2) {
          ++dc_sign;
        }
      }
    }
    for (int k = 0; k < h4; ++k) {
      if (y4 + k < mi_rows_) {
        const int sign = left_dc_[0][static_cast<std::size_t>(y4 + k)];
        if (sign == 1) {
          --dc_sign;
        } else if (sign == 2) {
          ++dc_sign;
        }
      }
    }
    if (dc_sign < 0) {
      return 1;
    }
    if (dc_sign > 0) {
      return 2;
    }
    return 0;
  }

  bool encode_residual_plane(int plane, int mi_row, int mi_col, int width,
                             int height, int tx_sz_ctx, int bsize_w,
                             int bsize_h) {
    const int w4 = width >> 2;
    const int h4 = height >> 2;
    const int sub_x = plane > 0 ? 1 : 0;
    const int sub_y = plane > 0 ? 1 : 0;
    const int x4 = (mi_col >> sub_x);
    const int y4 = (mi_row >> sub_y);

    int ctx = 0;
    if (plane == 0) {
      int top = 0;
      int left = 0;
      for (int k = 0; k < w4; ++k) {
        if (x4 + k < mi_cols_) {
          top |= above_level(0, x4 + k);
        }
      }
      for (int k = 0; k < h4; ++k) {
        if (y4 + k < mi_rows_) {
          left |= left_level(0, y4 + k);
        }
      }
      top = std::min(top, 4);
      left = std::min(left, 4);
      // With TX_MODE_LARGEST the residual block always equals the transform
      // block for M0 luma, so the `bw == w && bh == h` branch wins.
      if (bsize_w == width && bsize_h == height) {
        ctx = 0;
      } else {
        ctx = (top + left + 3) >> 1;
      }
    } else {
      int above = 0;
      int left = 0;
      for (int i = 0; i < w4; ++i) {
        if (x4 + i < mi_cols_ / 2) {
          above |= above_level(plane, x4 + i);
          above |= above_dc(plane, x4 + i);
        }
      }
      for (int i = 0; i < h4; ++i) {
        if (y4 + i < mi_rows_ / 2) {
          left |= left_level(plane, y4 + i);
          left |= left_dc(plane, y4 + i);
        }
      }
      ctx = (above != 0 ? 1 : 0) + (left != 0 ? 1 : 0);
      if (plane == 2) {
        if (bsize_w * bsize_h > width * height) {
          ctx += 3;
        }
        if (eob_u_ != 0) {
          ctx += 6;
        }
      } else {
        ctx += 6;
      }
    }

    hr_level_avg_ = 0;
    std::uint16_t* skip_cdf = nullptr;
    if (plane == 2) {
      skip_cdf = &cdfs_.v_txb_skip[ctx][0];
    } else {
      skip_cdf = &cdfs_.txb_skip[0][tx_sz_ctx][ctx][0];
    }

    if (!real_coeffs_) {
      // Diagnostic partition modes: zero residual, output is DC prediction.
      if (!sym(skip_cdf, 2, 1)) {
        return false;
      }
      for (int i = 0; i < w4; ++i) {
        above_level(plane, x4 + i) = 0;
        above_dc(plane, x4 + i) = 0;
      }
      for (int i = 0; i < h4; ++i) {
        left_level(plane, y4 + i) = 0;
        left_dc(plane, y4 + i) = 0;
      }
      if (plane == 1) {
        eob_u_ = 0;
      }
      return true;
    }

    // Prediction (DC_PRED, §7.13.2.10) from already reconstructed samples.
    int x0 = 0;
    int y0 = 0;
    if (plane == 0) {
      x0 = mi_col * 4;
      y0 = mi_row * 4;
    } else {
      x0 = (mi_col >> 1) * 4;
      y0 = (mi_row >> 1) * 4;
    }
    const bool have_left = is_inside(mi_row, mi_col - 1);
    const bool have_above = is_inside(mi_row - 1, mi_col);
    const int plane_max_x =
        (plane == 0 ? static_cast<int>(picture_.width)
                    : static_cast<int>(chroma_width_)) - 1;
    const int plane_max_y =
        (plane == 0 ? static_cast<int>(picture_.height)
                    : static_cast<int>(chroma_height_)) - 1;
    int sum_left = 0;
    int sum_above = 0;
    if (have_left) {
      for (int i = 0; i < height; ++i) {
        sum_left += recon_[plane].at(x0 - 1, clip3(0, plane_max_y, y0 + i));
      }
    }
    if (have_above) {
      for (int j = 0; j < width; ++j) {
        sum_above += recon_[plane].at(clip3(0, plane_max_x, x0 + j), y0 - 1);
      }
    }
    int pred_value = 128;
    if (have_left && have_above) {
      pred_value = clip1(approx_divide(sum_left + sum_above, width + height));
    } else if (have_left) {
      pred_value = round2(sum_left, floor_log2(height));
    } else if (have_above) {
      pred_value = round2(sum_above, floor_log2(width));
    }

    constexpr int kQuantStep = 528;
    constexpr int kQuantTableBits = 3;  // QUANT_TABLE_BITS
    int tw = width;
    int th = height;
    int upscale_x = 1;
    int upscale_y = 1;
    if (width == 64 && height == 64 && plane == 0) {
      // TX_64X64 is coded through Adjusted_Tx_Size = TX_32X32; the 32x32
      // residual is duplicated to 64x64 by the normative transform process.
      tw = 32;
      th = 32;
      upscale_x = 2;
      upscale_y = 2;
    }
    const int tx_pels = (upscale_x * tw) * (upscale_y * th);
    const int dq_denom = 1 << ((tx_pels > 256) + (tx_pels > 1024));
    int row_shift = 0;
    int col_shift = 0;
    transform_shifts(tw, th, &row_shift, &col_shift);
    const double inv_scale =
        (64.0 * std::sqrt(static_cast<double>(tw)) /
         static_cast<double>(1 << row_shift)) *
        (64.0 * std::sqrt(static_cast<double>(th)) /
         static_cast<double>(1 << col_shift));
    const double quant_factor =
        8.0 * static_cast<double>(dq_denom) /
        (inv_scale * static_cast<double>(kQuantStep));

    std::vector<int> residual_block(static_cast<std::size_t>(tw) * th, 0);
    if (upscale_x == 2 && upscale_y == 2) {
      for (int j = 0; j < th; ++j) {
        for (int i = 0; i < tw; ++i) {
          int sum = 0;
          for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
              sum += static_cast<int>(source_sample(
                         plane, x0 + 2 * i + dx, y0 + 2 * j + dy)) -
                     pred_value;
            }
          }
          residual_block[static_cast<std::size_t>(j) * tw + i] =
              round2(sum, 2);
        }
      }
    } else {
      for (int j = 0; j < th; ++j) {
        for (int i = 0; i < tw; ++i) {
          residual_block[static_cast<std::size_t>(j) * tw + i] =
              static_cast<int>(source_sample(plane, x0 + i, y0 + j)) -
              pred_value;
        }
      }
    }
    std::vector<double> coeff;
    forward_dct(residual_block, tw, th, &coeff);
    std::vector<int> quantized(static_cast<std::size_t>(tw) * th, 0);
    for (int pos = 0; pos < tw * th; ++pos) {
      quantized[static_cast<std::size_t>(pos)] = static_cast<int>(
          std::lround(coeff[static_cast<std::size_t>(pos)] * quant_factor));
    }
    std::vector<std::uint16_t> scan;
    build_diagonal_scan(tw, th, &scan);
    int eob = 0;
    for (int c = tw * th - 1; c >= 0; --c) {
      if (quantized[scan[c]] != 0) {
        eob = c + 1;
        break;
      }
    }

    if (!sym(skip_cdf, 2, eob == 0 ? 1 : 0)) {
      return false;
    }
    if (eob == 0) {
      for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
          recon_[plane].at(x0 + i, y0 + j) = static_cast<std::uint8_t>(pred_value);
        }
      }
      for (int i = 0; i < w4; ++i) {
        above_level(plane, x4 + i) = 0;
        above_dc(plane, x4 + i) = 0;
      }
      for (int i = 0; i < h4; ++i) {
        left_level(plane, y4 + i) = 0;
        left_dc(plane, y4 + i) = 0;
      }
      if (plane == 1) {
        eob_u_ = 0;
      }
      return true;
    }

    if (!write_eob(eob, plane, tw, th)) {
      return false;
    }
    if (plane == 0 && !write_luma_tx_type(tw, th, eob)) {
      return false;
    }
    // transform_type() is luma-only. For chroma, compute_tx_type() derives
    // DCT_DCT from UVMode = DC_PRED and codes no intra_tx_type symbol.

    // Sig_Ref_Diff_Offset[TX_CLASS_2D] and
    // Mag_Ref_Offset_With_Tx_Class[TX_CLASS_2D] (§8.3.2).
    static constexpr int kSigRef[5][2] = {
        {0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
    static constexpr int kMagRef[3][2] = {{0, 1}, {1, 0}, {1, 1}};

    std::vector<int> level_pre(static_cast<std::size_t>(tw) * th, 0);
    for (int c = eob - 1; c >= 0; --c) {
      const int pos = scan[c];
      const int row = pos / tw;
      const int col = pos % tw;
      const int abs_value = std::abs(quantized[pos]);
      const bool is_lf = plane == 0 ? (row + col) < 4 : (row + col) < 1;
      const int base_levels = is_lf ? 4 : 2;
      const int base_max = is_lf ? 5 : 3;
      const int br_max = (is_lf && plane > 0) ? 0 : 3;
      const int pre_max = base_max + br_max;
      int base_level = 0;
      int br_level = 0;
      int escape_x = 0;
      if (abs_value <= base_levels) {
        base_level = abs_value;
      } else {
        base_level = std::min(base_levels + 1, base_max);
        const int pre = std::min(abs_value, pre_max);
        br_level = pre - base_level;
        escape_x = abs_value - pre;
      }

      int ctx_eob = 0;
      if (c != 0) {
        const int area = tw * th;
        if (c <= area / 8) {
          ctx_eob = 1;
        } else if (c <= area / 4) {
          ctx_eob = 2;
        } else {
          ctx_eob = 3;
        }
      }

      if (c == eob - 1) {
        const int symbol = base_level - 1;
        if (plane == 0) {
          if (is_lf) {
            if (!sym(&cdfs_.coeff_base_lf_eob[tx_sz_ctx][ctx_eob][0], 5,
                     symbol)) {
              return false;
            }
          } else {
            if (!sym(&cdfs_.coeff_base_eob[tx_sz_ctx][ctx_eob][0], 3, symbol)) {
              return false;
            }
          }
        } else {
          if (is_lf) {
            if (!sym(&cdfs_.coeff_base_lf_eob_uv[ctx_eob][0], 5, symbol)) {
              return false;
            }
          } else {
            if (!sym(&cdfs_.coeff_base_eob_uv[ctx_eob][0], 3, symbol)) {
              return false;
            }
          }
        }
      } else {
        int mag = 0;
        const int num_ref = plane == 0 ? 5 : 3;
        for (int idx = 0; idx < num_ref; ++idx) {
          const int rr = row + kSigRef[idx][0];
          const int cc = col + kSigRef[idx][1];
          if (rr < th && cc < tw) {
            const int limit = is_lf ? 5 : 3;
            mag += std::min(level_pre[static_cast<std::size_t>(rr) * tw + cc],
                            limit);
          }
        }
        const int ctx = (mag + 1) >> 1;
        if (plane == 0) {
          if (is_lf) {
            int lf_ctx = 0;
            if (c == 0) {
              lf_ctx = std::min(ctx, 8);
            } else if (row + col < 2) {
              lf_ctx = std::min(ctx, 6) + 9;
            } else {
              lf_ctx = std::min(ctx, 4) + 16;
            }
            if (!sym(&cdfs_.coeff_base_lf[tx_sz_ctx][lf_ctx][0][0], 6,
                     base_level)) {
              return false;
            }
          } else {
            const int ctx2 = std::min(ctx, 4);
            int hf_ctx = ctx2;
            if (row + col >= 6 && row + col < 8) {
              hf_ctx = ctx2 + 5;
            } else if (row + col >= 8) {
              hf_ctx = ctx2 + 10;
            }
            if (!sym(&cdfs_.coeff_base[tx_sz_ctx][hf_ctx][0][0], 4, base_level)) {
              return false;
            }
          }
        } else {
          const int ctx2 = std::min(ctx, 3);
          const int uv_ctx = plane == 1 ? ctx2 : ctx2 + 4;
          if (is_lf) {
            if (!sym(&cdfs_.coeff_base_lf_uv[uv_ctx][0], 6, base_level)) {
              return false;
            }
          } else {
            if (!sym(&cdfs_.coeff_base_uv[uv_ctx][0], 4, base_level)) {
              return false;
            }
          }
        }
      }

      const int pre_level = base_level + br_level;
      if (pre_level > base_levels && !(is_lf && plane > 0)) {
        int mag = 0;
        for (int idx = 0; idx < 3; ++idx) {
          const int rr = row + kMagRef[idx][0];
          const int cc = col + kMagRef[idx][1];
          if (rr < th && cc < tw) {
            mag += std::min(level_pre[static_cast<std::size_t>(rr) * tw + cc],
                            5);
          }
        }
        mag = std::min((mag + 1) >> 1, 6);
        if (plane == 0) {
          int br_ctx = mag;
          if (is_lf) {
            if (pos != 0) {
              br_ctx = mag + 7;
            }
            if (!sym(&cdfs_.coeff_br_lf[br_ctx][0], 4, br_level)) {
              return false;
            }
          } else {
            if (!sym(&cdfs_.coeff_br[mag][0], 4, br_level)) {
              return false;
            }
          }
        } else {
          if (!sym(&cdfs_.coeff_br_uv[std::min(mag, 3)][0], 4, br_level)) {
            return false;
          }
        }
      }
      level_pre[static_cast<std::size_t>(pos)] = pre_level;
      (void)escape_x;
    }

    // Sign and read_quant() pass, in reverse scan order (§5.20.7.27).
    int cul_level = 0;
    int dc_category = 0;
    for (int c = eob - 1; c >= 0; --c) {
      const int pos = scan[c];
      const int row = pos / tw;
      const int col = pos % tw;
      const int pre_level = level_pre[static_cast<std::size_t>(pos)];
      const int abs_value = std::abs(quantized[pos]);
      const bool negative = quantized[pos] < 0;
      if (pre_level != 0) {
        if (row == 0 && col == 0 && plane == 0) {
          const int sign_ctx = dc_sign_context(x4, y4, w4, h4);
          if (!sym(&cdfs_.dc_sign[0][0][sign_ctx][0], 2, negative ? 1 : 0)) {
            return false;
          }
        } else {
          if (!raw_bit(negative)) {
            return false;
          }
        }
        const bool is_lf = plane == 0 ? (row + col) < 4 : (row + col) < 1;
        const int max_level = is_lf ? (plane == 0 ? 8 : 5) : 6;
        if (abs_value >= max_level) {
          if (!write_quant_escape(abs_value - pre_level)) {
            return false;
          }
        }
      }
      if (pos == 0 && abs_value > 0) {
        dc_category = negative ? 1 : 2;
      }
      cul_level = std::min(4, cul_level + abs_value);
    }

    // Dequant + normative 2D inverse transform, then add the prediction.
    std::vector<int> dequant(static_cast<std::size_t>(tw) * th, 0);
    for (int pos = 0; pos < tw * th; ++pos) {
      const int qc = quantized[pos];
      if (qc == 0) {
        continue;
      }
      const int dq_high = (std::abs(qc) * kQuantStep) & 0xFFFFFF;
      const int dq = round2(dq_high, kQuantTableBits);
      const int value = dq / dq_denom;
      dequant[static_cast<std::size_t>(pos)] = qc < 0 ? -value : value;
    }
    std::vector<int> residual;
    inverse_dct(dequant, tw, th, &residual);
    for (int j = 0; j < th; ++j) {
      for (int i = 0; i < tw; ++i) {
        const int value = clip1(
            pred_value + residual[static_cast<std::size_t>(j) * tw + i]);
        for (int dy = 0; dy < upscale_y; ++dy) {
          for (int dx = 0; dx < upscale_x; ++dx) {
            recon_[plane].at(x0 + i * upscale_x + dx,
                             y0 + j * upscale_y + dy) =
                static_cast<std::uint8_t>(value);
          }
        }
      }
    }

    for (int i = 0; i < w4; ++i) {
      above_level(plane, x4 + i) = static_cast<std::uint8_t>(cul_level);
      above_dc(plane, x4 + i) = static_cast<std::uint8_t>(dc_category);
    }
    for (int i = 0; i < h4; ++i) {
      left_level(plane, y4 + i) = static_cast<std::uint8_t>(cul_level);
      left_dc(plane, y4 + i) = static_cast<std::uint8_t>(dc_category);
    }
    if (plane == 1) {
      eob_u_ = 1;
    }
    return true;
  }

  const StillPicture& picture_;
  const StillPictureConfig& config_;
  M0PartitionMode partition_mode_ = M0PartitionMode::k32x32;
  bool real_coeffs_ = false;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  int intra_joint_cols_ = 0;
  std::uint32_t chroma_width_ = 0;
  std::uint32_t chroma_height_ = 0;

  CdfState cdfs_{};
  av2::Encoder enc_{};
  std::vector<std::uint8_t> tile_buffer_;
  std::string error_;

  Plane recon_[3];

  std::array<std::vector<std::uint8_t>, 3> above_level_;
  std::array<std::vector<std::uint8_t>, 3> left_level_;
  std::array<std::vector<std::uint8_t>, 3> above_dc_;
  std::array<std::vector<std::uint8_t>, 3> left_dc_;
  std::array<std::vector<std::uint8_t>, 2> above_mi_;
  std::array<std::vector<std::uint8_t>, 2> left_mi_;
  std::vector<std::uint8_t> intra_joint_;
  int eob_u_ = 0;
  int hr_level_avg_ = 0;
};

}  // namespace

bool encode_still_picture(const StillPicture& picture,
                          const StillPictureConfig& config,
                          StillEncodeResult& result,
                          std::string& error,
                          M0PartitionMode partition_mode) {
  error.clear();
  if (picture.width == 0 || picture.height == 0) {
    error = "empty picture";
    return false;
  }
  if ((picture.width & 1U) != 0U || (picture.height & 1U) != 0U) {
    error = "M0 4:2:0 requires even dimensions";
    return false;
  }
  if (picture.width != config.max_frame_width ||
      picture.height != config.max_frame_height) {
    error = "StillPictureConfig frame size does not match the picture";
    return false;
  }
  if (config.base_q_idx != kPinnedBaseQ) {
    error = "M0 pins base_q_idx = 90 (coefficient CDF tables are qctx slice 0)";
    return false;
  }
  const std::size_t luma_size =
      static_cast<std::size_t>(picture.width) * static_cast<std::size_t>(picture.height);
  const std::size_t chroma_size =
      static_cast<std::size_t>((picture.width + 1U) / 2U) *
      static_cast<std::size_t>((picture.height + 1U) / 2U);
  if (picture.y.size() < luma_size || picture.u.size() < chroma_size ||
      picture.v.size() < chroma_size) {
    error = "picture planes are too small";
    return false;
  }

  TileEncoder encoder(picture, config, partition_mode);
  std::vector<std::uint8_t> tile_payload;
  if (!encoder.encode(&tile_payload, error)) {
    return false;
  }

  result.frame = assemble_m0_still_picture(config, tile_payload);
  result.tile_bytes = static_cast<std::uint32_t>(tile_payload.size());
  result.frame_bytes = static_cast<std::uint32_t>(result.frame.size());
  const auto crop = [](const Plane& plane, int width, int height) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(width) *
                                  static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
      std::copy_n(plane.samples.begin() + static_cast<std::ptrdiff_t>(y) * plane.width,
                  width,
                  out.begin() + static_cast<std::ptrdiff_t>(y) * width);
    }
    return out;
  };
  result.recon_y = crop(encoder.recon(0), static_cast<int>(picture.width),
                        static_cast<int>(picture.height));
  result.recon_u = crop(encoder.recon(1), static_cast<int>((picture.width + 1U) / 2U),
                        static_cast<int>((picture.height + 1U) / 2U));
  result.recon_v = crop(encoder.recon(2), static_cast<int>((picture.width + 1U) / 2U),
                        static_cast<int>((picture.height + 1U) / 2U));
  return true;
}

bool geometry_supported(std::uint32_t width, std::uint32_t height,
                        std::string& reason) {
  reason.clear();
  if (width == 0 || height == 0) {
    reason = "empty picture";
    return false;
  }
  if ((width & 1U) != 0U || (height & 1U) != 0U) {
    reason = "M0 4:2:0 requires even dimensions";
    return false;
  }
  const int mi_cols = static_cast<int>(2U * ((width + 7U) >> 3U));
  const int mi_rows = static_cast<int>(2U * ((height + 7U) >> 3U));
  for (int r = 0; r < mi_rows; r += 16) {
    for (int c = 0; c < mi_cols; c += 16) {
      if (r + 16 <= mi_rows && c + 16 <= mi_cols) {
        continue;  // full superblock: fixed 32x32 tree
      }
      EdgePlanner planner(mi_rows, mi_cols);
      EdgePlanNode plan;
      if (!planner.build_superblock(r, c, &plan)) {
        reason = "edge superblock needs uncoded block size(s): " +
                 planner.missing_sizes();
        return false;
      }
    }
  }
  return true;
}

}  // namespace av2enc::m0
