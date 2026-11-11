// AV2 M2 P-frame tile syntax writer.  See m2_syntax.h for the contract.
//
// The partition, intra-mode and residual code is the M1 still-picture writer
// (byte-exact against dav2d for the M1 gate).  The inter block layer is new:
// is_inter/skip flags, single_mode, DRL index and shell-coded MVD, with all
// §8.3.2 contexts re-derived from the coded stream.
#include "encoder/m2/m2_syntax.h"

#include "encoder/m0_cdf_tables.h"
#include "encoder/m0_tables.h"
#include "encoder/m1/m1_cdf_tables.h"
#include "encoder/m1/m1_tables.h"
#include "encoder/m2/m2_cdf_tables.h"
#include "encoder/partition/partition.h"
#include "encoder/tx_variants/tx_variants.h"
#include "entropy/av2/av2_entropy.h"
#include "encoder/inter/mc/mc.h"
#include "encoder/m2/merge/merge.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace av2enc::m2::syntax {
namespace {

using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionType;
// The M1 table headers are generated and live in av2enc::m1. They are the
// same §9.2/§9.3 defaults the M1 syntax writer uses, so share them rather than
// duplicating 180 CDF arrays.
using namespace av2enc::m1;

// AV2 RefFrame labels (§6.19.7.9): LAST_FRAME = 0, INTRA_FRAME = 8.  The
// merge layer's MI grid uses these, not the MC-internal labels.
constexpr std::int8_t kRefLastAv2 = 0;
constexpr std::int8_t kRefIntraAv2 = 8;

// YMode numeric values (§6.19.5.5).
constexpr int kDcPred = 0;
constexpr int kV = 1;
constexpr int kH = 2;
constexpr int kD45 = 3;
constexpr int kD135 = 4;
constexpr int kD113 = 5;
constexpr int kD157 = 6;
constexpr int kD203 = 7;
constexpr int kD67 = 8;
constexpr int kSmooth = 9;
constexpr int kSmoothV = 10;
constexpr int kSmoothH = 11;
constexpr int kPaeth = 12;
constexpr int kIntraModesCount = 13;

// Reordered_Y_Mode[INTRA_MODES] (§5.20.5.5).
constexpr int kReorderedY[kIntraModesCount] = {
    kDcPred, kSmooth, kSmoothV, kSmoothH, kPaeth, kD45, kD67,
    kV,      kD113,   kD135,    kD157,    kH,     kD203};

// Default_Mode_List_Y[DIRECTIONAL_MODES_COUNT] (§5.20.5.5).
constexpr int kDefaultModeListY[56] = {
    17, 45, 3,  10, 24, 31, 38, 52, 15, 19, 43, 47, 1,  5,  8,  12,
    22, 26, 29, 33, 36, 40, 50, 54, 16, 18, 44, 46, 2,  4,  9,  11,
    23, 25, 30, 32, 37, 39, 51, 53, 14, 20, 42, 48, 0,  6,  7,  13,
    21, 27, 28, 34, 35, 41, 49, 55};

// Default_Mode_List_Uv[UV_INTRA_MODES_CFL_NOT_ALLOWED] (§5.20.5.6).
constexpr int kDefaultModeListUv[kIntraModesCount] = {
    kDcPred, kSmooth, kSmoothV, kSmoothH, kPaeth,  kV,   kH,
    kD45,    kD135,   kD67,     kD113,    kD157,   kD203};

constexpr int kRectHorz = 0;
constexpr int kRectVert = 1;
constexpr std::size_t kTileCapacityBytes = 16U * 1024U * 1024U;

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}
int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

bool y_mode_is_directional(int mode) { return mode >= kV && mode <= kD67; }

// Reordered index of a YMode (0..12).
int reordered_index(int mode) {
  for (int i = 0; i < kIntraModesCount; ++i) {
    if (kReorderedY[i] == mode) return i;
  }
  return 0;
}

// IntraJointMode and y_modeIndex for a (YMode, angleDelta) pair.
struct YModeCode {
  int joint = 0;
  int mode_index = 0;
};

YModeCode y_mode_code(int mode, int angle_delta) {
  YModeCode out;
  const int r = reordered_index(mode);
  if (r < 5) {
    out.joint = r;
    out.mode_index = r;
    return out;
  }
  const int base = r - 5;                               // 0..7
  const int dir_index = base * 7 + (angle_delta + 3);   // 0..55
  out.joint = 5 + dir_index;
  int pos = 0;
  for (int i = 0; i < 56; ++i) {
    if (kDefaultModeListY[i] == dir_index) {
      pos = i;
      break;
    }
  }
  out.mode_index = 5 + pos;
  return out;
}

int uv_mode_index(int uv_mode, int y_mode) {
  int index = 0;
  if (y_mode_is_directional(y_mode)) {
    if (uv_mode == y_mode) return 0;
    index = 1;
  }
  for (int i = 0; i < kIntraModesCount; ++i) {
    const int mode = kDefaultModeListUv[i];
    if (mode == y_mode && y_mode_is_directional(y_mode)) continue;
    if (mode == uv_mode) return index;
    ++index;
  }
  return 0;
}

int tx_sz_ctx(int tx_size) {
  return (kTxSizeSqr[tx_size] + kTxSizeSqrUp[tx_size] + 1) >> 1;
}

// Transform-type syntax class for an intra luma TX block.  Mirrors
// get_tx_set() for reduced_tx_set = 0 and enable_idtx_intra = 0.
enum class TxTypeSyntax { kNone, kIntraSet1, kLong32, kLong64 };
TxTypeSyntax luma_tx_type_syntax(int tx_size, int eob) {
  if (eob == 0 || eob == 1) return TxTypeSyntax::kNone;
  const int sqr = kTxSizeSqr[tx_size];
  const int sqr_up = kTxSizeSqrUp[tx_size];
  if (sqr_up > 3) {
    return sqr >= 3 ? TxTypeSyntax::kNone : TxTypeSyntax::kLong64;
  }
  if (sqr_up == 3) {
    return sqr == 3 ? TxTypeSyntax::kNone : TxTypeSyntax::kLong32;
  }
  return TxTypeSyntax::kIntraSet1;
}

// Inter luma transform set from get_tx_set(txSz, 0) for reduced_tx_set = 0
// and enable_chroma_dctonly = 1.  §5.20.8.2 codes a TX_TYPE for every inter
// luma TX block whose set is not TX_SET_DCTONLY, including eob == 1 (the
// eob == 1 shortcut is intra-only).
enum class InterTxSet {
  kDctOnly,
  kDctIdtx,   // 32x32
  kInter1,    // square <= 8x8 / rectangles with sqr <= 8x8
  kInter2,    // 16x16
  kWide32,
  kHigh32,
  kWide64,
  kHigh64,
};
InterTxSet inter_tx_set(int tx_size) {
  const int sqr = kTxSizeSqr[tx_size];
  const int sqr_up = kTxSizeSqrUp[tx_size];
  const auto params =
      txv::make_tx_params(static_cast<txv::TxSize>(tx_size),
                          txv::TxType::kDctDct);
  const bool wider = params.has_value() && params->coded_w > params->coded_h;
  if (sqr_up > 3) {
    if (sqr >= 3) return InterTxSet::kDctOnly;
    return wider ? InterTxSet::kWide64 : InterTxSet::kHigh64;
  }
  if (sqr_up == 3 && sqr != 3) {
    return wider ? InterTxSet::kWide32 : InterTxSet::kHigh32;
  }
  if (sqr_up == 3) return InterTxSet::kDctIdtx;
  return sqr == 2 ? InterTxSet::kInter2 : InterTxSet::kInter1;
}

// §8.3.2 inter_tx_type context from the final EOB.
int inter_tx_type_context(int eob, int coded_w, int coded_h, bool use_min) {
  const int bwl = std::min(floor_log2(coded_w), 5);
  const int eoby = (eob - 1) >> bwl;
  const int eobx = (eob - 1) - (eoby << bwl);
  const int diag = eobx + eoby;
  const int w = use_min ? std::min(coded_w, 32) : coded_w;
  const int h = use_min ? std::min(coded_h, 32) : coded_h;
  if (diag < 2) return 1;
  if (diag > w + h - 4) return 2;
  return 0;
}

// Default_Eob_Pt_16_Cdf [3][6] and Default_Eob_Pt_32_Cdf [3][7] (§9.3),
// COEFF_CDF_Q_CTXS slice 0.  The M1 syntax path previously had no 4x4/4x8
// transform and only imported the 64+ tables, so these two live here.
constexpr std::uint16_t kEobPt16Cdf[3][6] = {
    {1946, 3059, 6834, 15123, 6, 0},
    {558, 835, 2739, 6534, 25, 0},
    {8000, 10366, 14466, 19569, 31, 0},
};
constexpr std::uint16_t kEobPt32Cdf[3][7] = {
    {1203, 1920, 4502, 10281, 17078, 6, 0},
    {1078, 1340, 2464, 5152, 11416, 5, 0},
    {11670, 14271, 19287, 23919, 27084, 31, 0},
};

struct CdfState {
  std::uint16_t do_split[2][64][3];
  std::uint16_t rect_type[2][64][3];
  std::uint16_t y_mode_set[5];
  std::uint16_t y_mode_index[3][9];
  std::uint16_t y_mode_offset[3][7];
  std::uint16_t uv_mode[2][9];
  std::uint16_t is_cfl[3][3];
  std::uint16_t cfl_index[3];
  std::uint16_t cfl_sign[9];
  std::uint16_t cfl_alpha[6][9];
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
  std::uint16_t eob_pt_1024[3][9];
  std::uint16_t eob_pt_512[3][9];
  std::uint16_t eob_pt_256[3][9];
  std::uint16_t eob_pt_128[3][9];
  std::uint16_t eob_pt_64[3][8];
  std::uint16_t eob_pt_32[3][7];
  std::uint16_t eob_pt_16[3][6];
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

  // Inter block layer (§8.3.2 CDF selection).
  std::uint16_t single_mode[5][4];
  std::uint16_t skip[6][3];
  std::uint16_t skip_mode[3][3];
  std::uint16_t is_inter[4][3];
  std::uint16_t single_ref[3][6][3];
  std::uint16_t drl_mode[3][5][3];
  std::uint16_t shell_set[3];
  std::uint16_t shell6_class0[9];
  std::uint16_t shell6_class1[9];
  std::uint16_t shell_last_two[3];
  std::uint16_t offset_low[2][3];
  std::uint16_t offset_class2[3];
  std::uint16_t offset_other[16][3];
  std::uint16_t col_mv_greater[2][3];
  std::uint16_t col_mv_index[4][3];

  // Inter residual luma transform-type symbols (§5.20.8.2).
  std::uint16_t inter_tx_type_set1[3][2][3];
  std::uint16_t inter_tx_type_set2[3][3];
  std::uint16_t inter_tx_type_set3[3][4][3];
  std::uint16_t inter_tx_type_long[3][4][5];
  std::uint16_t inter_tx_type_index_set1[3][9];
  std::uint16_t inter_tx_type_offset_set1[3][9];
  std::uint16_t inter_tx_type_index_set2[3][9];
  std::uint16_t inter_tx_type_offset_set2[3][5];
};

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

void init_cdfs(CdfState& c) {
  copy_array(c.do_split, av2enc::m0::kDoSplitCdf);
  copy_array(c.rect_type, av2enc::m0::kRectTypeCdf);
  copy_array(c.y_mode_set, kYModeSetCdf);
  copy_array(c.y_mode_index, kYModeIndexCdf);
  copy_array(c.y_mode_offset, kYModeOffsetCdf);
  copy_array(c.uv_mode, kUvModeCflNotAllowedCdf);
  copy_array(c.is_cfl, kIsCflCdf);
  copy_array(c.cfl_index, kCflIndexCdf);
  copy_array(c.cfl_sign, kCflSignCdf);
  copy_array(c.cfl_alpha, kCflAlphaCdf);
  copy_array(c.txb_skip, av2enc::m0::kTxbSkipCdf);
  copy_array(c.v_txb_skip, av2enc::m0::kVTxbSkipCdf);
  copy_array(c.eob_pt_1024, av2enc::m0::kEobPt1024Cdf);
  copy_array(c.eob_pt_512, av2enc::m0::kEobPt512Cdf);
  copy_array(c.eob_pt_256, av2enc::m0::kEobPt256Cdf);
  copy_array(c.eob_pt_128, av2enc::m0::kEobPt128Cdf);
  copy_array(c.eob_pt_64, av2enc::m0::kEobPt64Cdf);
  copy_array(c.eob_pt_32, kEobPt32Cdf);
  copy_array(c.eob_pt_16, kEobPt16Cdf);
  copy_array(c.eob_extra, av2enc::m0::kEobExtraCdf);
  copy_array(c.coeff_base, av2enc::m0::kCoeffBaseCdf);
  copy_array(c.coeff_base_lf, av2enc::m0::kCoeffBaseLfCdf);
  copy_array(c.coeff_base_eob, av2enc::m0::kCoeffBaseEobCdf);
  copy_array(c.coeff_base_lf_eob, av2enc::m0::kCoeffBaseLfEobCdf);
  copy_array(c.coeff_br, av2enc::m0::kCoeffBrCdf);
  copy_array(c.coeff_br_lf, av2enc::m0::kCoeffBrLfCdf);
  copy_array(c.coeff_base_uv, av2enc::m0::kCoeffBaseUvCdf);
  copy_array(c.coeff_base_lf_uv, av2enc::m0::kCoeffBaseLfUvCdf);
  copy_array(c.coeff_base_eob_uv, av2enc::m0::kCoeffBaseEobUvCdf);
  copy_array(c.coeff_base_lf_eob_uv, av2enc::m0::kCoeffBaseLfEobUvCdf);
  copy_array(c.coeff_br_uv, av2enc::m0::kCoeffBrUvCdf);
  copy_array(c.dc_sign, av2enc::m0::kDcSignCdf);
  copy_array(c.intra_tx_type_set1, av2enc::m0::kIntraTxTypeSet1Cdf);
  copy_array(c.intra_tx_type_long, kIntraTxTypeLongCdf);
  copy_array(c.is_long_side_dct, kIsLongSideDctCdf);
  copy_array(c.single_mode, kSingleModeCdf);
  copy_array(c.skip, kSkipCdf);
  copy_array(c.skip_mode, kSkipModeCdf);
  copy_array(c.is_inter, kIsInterCdf);
  copy_array(c.single_ref, kSingleRefCdf);
  copy_array(c.drl_mode, kDrlModeCdf);
  copy_array(c.shell_set, kJointShellSetCdf);
  copy_array(c.shell6_class0, kJointShell6Class0Cdf);
  copy_array(c.shell6_class1, kJointShell6Class1Cdf);
  copy_array(c.shell_last_two, kJointShellLastTwoClassesCdf);
  copy_array(c.offset_low, kShellOffsetLowClassCdf);
  copy_array(c.offset_class2, kShellOffsetClass2Cdf);
  copy_array(c.offset_other, kShellOffsetOtherClassCdf);
  copy_array(c.col_mv_greater, kColMvGreaterCdf);
  copy_array(c.col_mv_index, kColMvIndexCdf);
  copy_array(c.inter_tx_type_set1, kInterTxTypeSet1Cdf);
  copy_array(c.inter_tx_type_set2, kInterTxTypeSet2Cdf);
  copy_array(c.inter_tx_type_set3, kInterTxTypeSet3Cdf);
  copy_array(c.inter_tx_type_long, kInterTxTypeLongCdf);
  copy_array(c.inter_tx_type_index_set1, kInterTxTypeIndexSet1Cdf);
  copy_array(c.inter_tx_type_offset_set1, kInterTxTypeOffsetSet1Cdf);
  copy_array(c.inter_tx_type_index_set2, kInterTxTypeIndexSet2Cdf);
  copy_array(c.inter_tx_type_offset_set2, kInterTxTypeOffsetSet2Cdf);
}

// One in-frame child slot of a partition (§9.2 Partition_Subsize).
struct SlotInfo {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
};

int enumerate_slots(int mi_row, int mi_col, BlockSize size,
                    PartitionType partition, int mi_rows, int mi_cols,
                    SlotInfo slots[4]) {
  if (partition == PartitionType::kNone) return 0;
  const BlockSize subsize = partition::partition_subsize(partition, size);
  if (!partition::is_valid_block_size(subsize)) return 0;
  const int w4 = partition::num_4x4_blocks_wide(size);
  const int h4 = partition::num_4x4_blocks_high(size);
  const int half_w = w4 >> 1;
  const int half_h = h4 >> 1;
  int count = 0;
  const auto push = [&](int r, int c) {
    // decode_partition() returns before coding a child whose origin is outside
    // the frame, so only in-frame slots exist in the stream.
    if (r < mi_rows && c < mi_cols) {
      slots[count++] = SlotInfo{r, c, subsize};
    }
  };
  switch (partition) {
    case PartitionType::kSplit:
      push(mi_row, mi_col);
      push(mi_row, mi_col + half_w);
      push(mi_row + half_h, mi_col);
      push(mi_row + half_h, mi_col + half_w);
      break;
    case PartitionType::kHorz:
      push(mi_row, mi_col);
      push(mi_row + half_h, mi_col);
      break;
    case PartitionType::kVert:
      push(mi_row, mi_col);
      push(mi_row, mi_col + half_w);
      break;
    case PartitionType::kNone:
      break;
  }
  return count;
}

class Writer {
 public:
  Writer(const FramePlan& plan, std::span<const BlockSyntax> blocks,
         std::span<const std::int32_t> levels, const M2Config& config)
      : plan_(plan), blocks_(blocks), levels_(levels), config_(config) {
    geometry_.width = plan.width;
    geometry_.height = plan.height;
    geometry_.sb_size = plan.sb_size;
    geometry_.has_chroma = true;
    geometry_.chroma_subsampling_x = true;
    geometry_.chroma_subsampling_y = true;
    mi_cols_ = partition::mi_cols(geometry_);
    mi_rows_ = partition::mi_rows(geometry_);

    init_cdfs(cdfs_);
    tile_buffer_.resize(kTileCapacityBytes);
    enc_.init_symbol(tile_buffer_.data(), tile_buffer_.size());
    enc_.disable_cdf_update = config_.disable_cdf_update ? 1 : 0;

    above_mi_.assign(static_cast<std::size_t>(mi_cols_) + 16, 6);
    left_mi_.assign(static_cast<std::size_t>(mi_rows_) + 16, 6);
    intra_joint_.assign(
        static_cast<std::size_t>(mi_rows_) * static_cast<std::size_t>(mi_cols_),
        0);
    uvcfl_.assign(static_cast<std::size_t>(mi_rows_ / 2) *
                      static_cast<std::size_t>(mi_cols_ / 2),
                  0);
    const std::size_t mi_cells = static_cast<std::size_t>(mi_rows_) *
                                 static_cast<std::size_t>(mi_cols_);
    is_inter_grid_.assign(mi_cells, 0);
    skip_grid_.assign(mi_cells, 0);
    ymode_grid_.assign(mi_cells, 0);
    mv_grid_.assign(mi_cells, inter::mc::MiInfo{});
    for (int plane = 0; plane < kPlaneCount; ++plane) {
      const int n = (plane == 0 ? mi_cols_ : mi_cols_ / 2) + 16;
      const int m = (plane == 0 ? mi_rows_ : mi_rows_ / 2) + 16;
      above_level_[plane].assign(static_cast<std::size_t>(n), 0);
      left_level_[plane].assign(static_cast<std::size_t>(m), 0);
      above_dc_[plane].assign(static_cast<std::size_t>(n), 0);
      left_dc_[plane].assign(static_cast<std::size_t>(m), 0);
    }
    clear_above_context();
    trace_ = std::getenv("AV2ENC_M2_TRACE") != nullptr;
  }

  bool run(std::size_t* tile_bytes, std::string& error) {
    if (blocks_.empty()) {
      error = "syntax stream is empty";
      return false;
    }
    const int sb_cols = plan_.sb_cols();
    const int sb_rows = plan_.sb_rows();
    const int sb_mi = partition::sb_size_mi(geometry_);
    std::size_t index = 0;
    int sb_index = 0;
    while (index < blocks_.size()) {
      const BlockSyntax& node = blocks_[index];
      if ((node.flags & kFlagSbRoot) == 0) {
        error = "node " + std::to_string(index) +
                " is not rooted in a superblock (missing kFlagSbRoot)";
        return false;
      }
      if (sb_index >= sb_cols * sb_rows) {
        error = "more superblocks than the frame geometry allows";
        return false;
      }
      const int expect_row = (sb_index / sb_cols) * sb_mi;
      const int expect_col = (sb_index % sb_cols) * sb_mi;
      if (node.mi_row != expect_row || node.mi_col != expect_col) {
        error = "superblock " + std::to_string(sb_index) + " is at mi (" +
                std::to_string(node.mi_row) + "," +
                std::to_string(node.mi_col) + "), expected (" +
                std::to_string(expect_row) + "," +
                std::to_string(expect_col) + ")";
        return false;
      }
      if (node.mi_col == 0) {
        // New superblock row: the left partition context restarts.  Mirrors
        // `if (sb.mi_col == 0) clear_left_context()` in m1_still.
        clear_left_context();
      }
      std::size_t next = 0;
      if (!write_node(index, &next, error)) return false;
      index = next;
      ++sb_index;
    }
    if (sb_index != sb_cols * sb_rows) {
      error = "syntax stream covers " + std::to_string(sb_index) + " of " +
              std::to_string(sb_cols * sb_rows) + " superblocks";
      return false;
    }
    std::size_t payload_bytes = 0;
    const av2::Status status = enc_.exit_symbol(&payload_bytes);
    if (status != av2::Status::kOk) {
      error = std::string("entropy exit_symbol failed: ") +
              av2::status_name(status);
      return false;
    }
    payload_.assign(tile_buffer_.begin(),
                    tile_buffer_.begin() +
                        static_cast<std::ptrdiff_t>(payload_bytes));
    if (tile_bytes != nullptr) *tile_bytes = payload_bytes;
    return true;
  }

  const std::vector<std::uint8_t>& payload() const { return payload_; }

 private:
  // ---------------------------------------------------------------------
  // Entropy helpers
  // ---------------------------------------------------------------------
  bool sym(std::uint16_t* cdf, int n, int symbol) {
    const av2::Status status = enc_.encode_symbol(cdf, n, symbol);
    if (status != av2::Status::kOk) {
      error_ = std::string("entropy encode_symbol failed: ") +
               av2::status_name(status) + " n=" + std::to_string(n) +
               " sym=" + std::to_string(symbol) + " cdf=";
      for (int i = 0; i <= n; ++i) {
        error_ += " " + std::to_string(static_cast<int>(cdf[i]));
      }
      error_ += " block=(" + std::to_string(cur_r_) + "," +
                std::to_string(cur_c_) + ")";
      return false;
    }
    return true;
  }
  bool raw_bit(bool bit) {
    const av2::Status status = enc_.encode_bool(bit);
    if (status != av2::Status::kOk) {
      error_ = std::string("entropy encode_bool failed: ") +
               av2::status_name(status);
      return false;
    }
    return true;
  }
  bool write_literal(int bits, std::uint32_t value) {
    for (int i = bits - 1; i >= 0; --i) {
      if (!raw_bit(((value >> static_cast<unsigned>(i)) & 1U) != 0U)) {
        return false;
      }
    }
    return true;
  }

  // ---------------------------------------------------------------------
  // Context grids (all derived from the stream, never sent by the producer)
  // ---------------------------------------------------------------------
  void clear_above_context() {
    for (int plane = 0; plane < kPlaneCount; ++plane) {
      std::fill(above_level_[plane].begin(), above_level_[plane].end(), 0);
      std::fill(above_dc_[plane].begin(), above_dc_[plane].end(), 0);
    }
    std::fill(above_mi_.begin(), above_mi_.end(), 6);
  }
  void clear_left_context() {
    for (int plane = 0; plane < kPlaneCount; ++plane) {
      std::fill(left_level_[plane].begin(), left_level_[plane].end(), 0);
      std::fill(left_dc_[plane].begin(), left_dc_[plane].end(), 0);
    }
    std::fill(left_mi_.begin(), left_mi_.end(), 6);
  }

  int& above_mi(int x) { return above_mi_[static_cast<std::size_t>(x)]; }
  int& left_mi(int y) { return left_mi_[static_cast<std::size_t>(y)]; }
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
  int& intra_joint(int r, int c) {
    return intra_joint_[static_cast<std::size_t>(r) *
                            static_cast<std::size_t>(mi_cols_) +
                        static_cast<std::size_t>(c)];
  }
  std::uint8_t& uvcfl(int r, int c) {
    return uvcfl_[static_cast<std::size_t>(r) *
                      static_cast<std::size_t>(mi_cols_ / 2) +
                  static_cast<std::size_t>(c)];
  }
  bool inside_mi(int r, int c) const {
    return r >= 0 && r < mi_rows_ && c >= 0 && c < mi_cols_;
  }
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
    if (inside_mi(mv_row, mv_col)) {
      return intra_joint_[static_cast<std::size_t>(mv_row) *
                              static_cast<std::size_t>(mi_cols_) +
                          static_cast<std::size_t>(mv_col)];
    }
    return 0;
  }

  // ---------------------------------------------------------------------
  // Partition syntax
  // ---------------------------------------------------------------------
  bool code_do_split(PartitionType partition, int r, int c, BlockSize size) {
    const int bsw = std::max(kMiWidthLog2[static_cast<int>(size)], 1);
    const int bsh = std::max(kMiHeightLog2[static_cast<int>(size)], 1);
    const int ctx = kPartitionSizeAdjust[static_cast<int>(size)] * 4 +
                    (left_mi(r) < bsh ? 2 : 0) +
                    (above_mi(c) < bsw ? 1 : 0);
    return sym(&cdfs_.do_split[0][ctx][0], 2,
               partition == PartitionType::kNone ? 0 : 1);
  }

  bool code_rect_type(PartitionType partition, int r, int c, BlockSize size) {
    const int bsw = std::max(kMiWidthLog2[static_cast<int>(size)], 1);
    const int bsh = std::max(kMiHeightLog2[static_cast<int>(size)], 1);
    const int ctx = kPartitionSizeAdjustRect[static_cast<int>(size)] * 4 +
                    (left_mi(r) < bsh ? 2 : 0) +
                    (above_mi(c) < bsw ? 1 : 0);
    return sym(&cdfs_.rect_type[0][ctx][0], 2,
               partition == PartitionType::kVert ? kRectVert : kRectHorz);
  }

  // Emit the syntax for the node at `index` (and, for non-leaves, its whole
  // subtree); `next` receives the index one past the subtree.
  bool write_node(std::size_t index, std::size_t* next, std::string& error) {
    const BlockSyntax& node = blocks_[index];
    const auto size = static_cast<BlockSize>(node.bsize);
    const auto partition = static_cast<PartitionType>(node.partition);
    const std::size_t subtree_end = index + 1 + node.subtree;
    if (subtree_end > blocks_.size()) {
      error = "node " + std::to_string(index) + " has a subtree past the end";
      return false;
    }
    if (!partition::is_valid_block_size(size)) {
      error = "node " + std::to_string(index) + " has an invalid block size";
      return false;
    }
    const int r = node.mi_row;
    const int c = node.mi_col;
    if (r < 0 || c < 0 || r >= mi_rows_ || c >= mi_cols_) {
      error = "node " + std::to_string(index) + " is outside the frame";
      return false;
    }
    const bool forced = (node.flags & kFlagForced) != 0;
    const auto allowed = [&node](PartitionType p) {
      return (node.allowed_mask & (1u << static_cast<unsigned>(p))) != 0;
    };

    if (trace_) {
      std::cerr << "m1 node r=" << r << " c=" << c
                << " size=" << partition::block_size_name(size)
                << " part=" << partition::partition_name(partition)
                << " forced=" << (forced ? 1 : 0) << " allowed="
                << allowed(PartitionType::kNone) << allowed(PartitionType::kSplit)
                << allowed(PartitionType::kHorz)
                << allowed(PartitionType::kVert) << "\n";
    }

    if (!forced) {
      if (allowed(PartitionType::kNone)) {
        if (!code_do_split(partition, r, c, size)) {
          error = error_;
          return false;
        }
        if (partition == PartitionType::kNone) {
          if (node.subtree != 0) {
            error = "leaf node " + std::to_string(index) +
                    " has a non-zero subtree";
            return false;
          }
          if (!write_block(node, error)) return false;
          *next = index + 1;
          return true;
        }
      }
      if (partition == PartitionType::kNone) {
        // No do_split symbol was coded (NONE is the implied partition), but
        // the tree solver only reaches this for a forced node.
        if (node.subtree != 0) {
          error = "leaf node " + std::to_string(index) +
                  " has a non-zero subtree";
          return false;
        }
        if (!write_block(node, error)) return false;
        *next = index + 1;
        return true;
      }
      const partition::RectTypeImplied implied =
          partition::rect_type_implied_by_bsize(size);
      bool need_rect = !implied.implied;
      if (need_rect) {
        const bool allow_h = allowed(PartitionType::kHorz);
        const bool allow_v = allowed(PartitionType::kVert);
        if (!allow_h || !allow_v) need_rect = false;
      }
      if (need_rect) {
        if (!code_rect_type(partition, r, c, size)) {
          error = error_;
          return false;
        }
      }
    }
    if (partition == PartitionType::kNone) {
      if (node.subtree != 0) {
        error = "leaf node " + std::to_string(index) +
                " has a non-zero subtree";
        return false;
      }
      if (!write_block(node, error)) return false;
      *next = index + 1;
      return true;
    }

    // Descend into the children (preorder) and validate them against the
    // Partition_Subsize / boundary rules.
    SlotInfo slots[4];
    const int slot_count = enumerate_slots(r, c, size, partition, mi_rows_,
                                           mi_cols_, slots);
    if (slot_count == 0) {
      error = "node " + std::to_string(index) + " has no in-frame children";
      return false;
    }
    std::size_t cursor = index + 1;
    int slot_cursor = 0;
    while (cursor < subtree_end) {
      const BlockSyntax& child = blocks_[cursor];
      while (slot_cursor < slot_count &&
             (slots[slot_cursor].mi_row != static_cast<int>(child.mi_row) ||
              slots[slot_cursor].mi_col != static_cast<int>(child.mi_col))) {
        ++slot_cursor;
      }
      if (slot_cursor >= slot_count) {
        error = "node " + std::to_string(cursor) +
                " does not match a partition subsize slot of node " +
                std::to_string(index);
        return false;
      }
      if (child.bsize != static_cast<std::uint8_t>(slots[slot_cursor].size)) {
        error = "node " + std::to_string(cursor) +
                " has the wrong block size for its partition";
        return false;
      }
      ++slot_cursor;
      std::size_t after = 0;
      if (!write_node(cursor, &after, error)) return false;
      if (after > subtree_end) {
        error = "node " + std::to_string(index) +
                " has an inconsistent subtree count";
        return false;
      }
      cursor = after;
    }
    if (cursor != subtree_end) {
      error = "node " + std::to_string(index) +
              " has an inconsistent subtree count";
      return false;
    }
    *next = subtree_end;
    return true;
  }

  // ---------------------------------------------------------------------
  // Mode info syntax
  // ---------------------------------------------------------------------
  int joint_context() const {
    if (std::getenv("AV2ENC_M1_YMODE_CTX0") != nullptr) return 0;
    if (std::getenv("AV2ENC_M1_YMODE_CTX2") != nullptr) return 2;
    // Uses the block-start MiRow/MiCol of the current node.
    return (get_joint_mode(cur_r_, cur_c_, cur_w4_, cur_h4_, 0) >= 5 ? 1 : 0) +
           (get_joint_mode(cur_r_, cur_c_, cur_w4_, cur_h4_, 1) >= 5 ? 1 : 0);
  }

  // §5.20.5.5 get_intra_y_mode_set: the signalled mode index is resolved
  // through a neighbour-dependent list.  Directional candidates are seeded with
  // the IntraJointModes of the below-left (dir=0) and above-right (dir=1)
  // neighbours, then expanded around them for blocks larger than 8x8/64 pels,
  // and finally filled from Default_Mode_List_Y.  The encoder must invert that
  // list; writing the static-list position makes dependent blocks decode a
  // different direction.
  int y_mode_index_for(int joint) const {
    if (joint < 5) {
      return joint;
    }
    const int target = joint - 5;  // directional index 0..55
    int mode_index = 5;
    bool selected[56] = {};
    int dir_modes[2] = {0, 0};
    int count = 0;
    const int w = cur_w4_ * 4;
    const int h = cur_h4_ * 4;
    if (w >= 8 && h >= 8) {
      for (int dir = 0; dir < 2; ++dir) {
        int mode = get_joint_mode(cur_r_, cur_c_, cur_w4_, cur_h4_, dir);
        if (mode >= 5) {
          mode -= 5;
          if (count == 0 || mode != dir_modes[0]) {
            if (mode == target) return mode_index;
            ++mode_index;
            selected[mode] = true;
            dir_modes[count++] = mode;
          }
        }
      }
      if (w * h > 64) {
        for (int i = 1; i <= 4; ++i) {
          for (int j = 0; j < count; ++j) {
            for (int sgn = -1; sgn <= 1; sgn += 2) {
              int mode = dir_modes[j] + i * sgn;
              if (mode < 0) {
                mode += 56;
              } else if (mode >= 56) {
                mode -= 56;
              }
              if (!selected[mode]) {
                if (mode == target) return mode_index;
                ++mode_index;
                selected[mode] = true;
              }
            }
          }
        }
      }
    }
    for (int i = 0; i < 56; ++i) {
      const int mode = kDefaultModeListY[i];
      if (!selected[mode]) {
        if (mode == target) return mode_index;
        ++mode_index;
      }
    }
    return mode_index;
  }

  bool code_y_mode(const BlockSyntax& node) {
    const YModeCode code =
        y_mode_code(node.y_mode, node.y_angle_delta);
    // Directional modes use the dynamic set; non-directional indices are the
    // reordered mode itself and are unaffected by neighbours.
    const int mode_index =
        code.joint < 5 ? code.mode_index : y_mode_index_for(code.joint);
    const int ctx = joint_context();
    if (mode_index < kIntraModesCount) {
      if (!sym(&cdfs_.y_mode_set[0], 4, 0)) return false;
      if (mode_index <= 6) {
        return sym(&cdfs_.y_mode_index[ctx][0], 8, mode_index);
      }
      if (!sym(&cdfs_.y_mode_index[ctx][0], 8, 7)) return false;
      return sym(&cdfs_.y_mode_offset[ctx][0], 6, mode_index - 7);
    }
    const int set = 1 + (mode_index - kIntraModesCount) / 16;
    const int second = (mode_index - kIntraModesCount) % 16;
    if (!sym(&cdfs_.y_mode_set[0], 4, set)) return false;
    return write_literal(4, static_cast<std::uint32_t>(second));
  }

  // is_cfl context: neighbouring chroma CfL flags (§8.3.x).
  int cfl_context() const {
    const int cr = cur_r_ / 2;
    const int cc = cur_c_ / 2;
    const int cmi_cols = mi_cols_ / 2;
    int ctx = 0;
    if (cr > 0 && cc < cmi_cols &&
        uvcfl_[static_cast<std::size_t>(cr - 1) *
                   static_cast<std::size_t>(cmi_cols) +
               static_cast<std::size_t>(cc)] != 0) {
      ++ctx;
    }
    if (cc > 0 && cr < mi_rows_ / 2 &&
        uvcfl_[static_cast<std::size_t>(cr) *
                   static_cast<std::size_t>(cmi_cols) +
               static_cast<std::size_t>(cc - 1)] != 0) {
      ++ctx;
    }
    return ctx;
  }

  bool code_uv_mode(bool is_cfl, int uv_mode, int y_mode) {
    if (is_cfl) {
      // is_cfl derives its context from neighbouring chroma CfL flags.
      if (!sym(&cdfs_.is_cfl[cfl_context()][0], 2, 1)) return false;
      // CFL_DERIVED_ALPHA = 1.
      return sym(&cdfs_.cfl_index[0], 2, 1);
    }
    const int ctx = y_mode_is_directional(y_mode) ? 1 : 0;
    // is_cfl = 0 is only coded when CfL is allowed.
    if (config_.enable_cfl_intra) {
      if (!sym(&cdfs_.is_cfl[cfl_context()][0], 2, 0)) return false;
    }
    const int idx = uv_mode_index(uv_mode, y_mode);
    if (idx <= 6) {
      return sym(&cdfs_.uv_mode[ctx][0], 8, idx);
    }
    if (!sym(&cdfs_.uv_mode[ctx][0], 8, 7)) return false;
    return write_literal(3, static_cast<std::uint32_t>(idx - 7));
  }

  // ---------------------------------------------------------------------
  // Residual syntax
  // ---------------------------------------------------------------------
  bool write_eob(int eob, int plane, int coded_w, int coded_h, bool is_inter) {
    int eob_pt = 1;
    if (eob == 2) {
      eob_pt = 2;
    } else if (eob >= 3) {
      eob_pt = floor_log2(eob - 1) + 2;
    }
    const int multisize = std::min(5, floor_log2(coded_w)) +
                          std::min(5, floor_log2(coded_h)) - 4;
    std::uint16_t* cdf = nullptr;
    int escape_bits = 0;
    int cdf_n = 8;
    const int cdf_ctx = plane > 0 ? 2 : (is_inter ? 1 : 0);
    switch (multisize) {
      case 0:
        // §5.20.7.27 / §8.3.2: eobMultisize 0 (TX_4X4) codes eob_pt_16,
        // whose default CDF has five symbols (eobPt 1..5) and no escape.
        cdf = &cdfs_.eob_pt_16[cdf_ctx][0];
        escape_bits = 0;
        cdf_n = 5;
        break;
      case 1:
        // eobMultisize 1 (TX_4X8 / TX_8X4) codes eob_pt_32: six symbols
        // (eobPt 1..6), no escape.
        cdf = &cdfs_.eob_pt_32[cdf_ctx][0];
        escape_bits = 0;
        cdf_n = 6;
        break;
      case 2:
        cdf = &cdfs_.eob_pt_64[cdf_ctx][0];
        escape_bits = 0;
        cdf_n = 7;
        break;
      case 3:
        cdf = &cdfs_.eob_pt_128[cdf_ctx][0];
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
    if (eob_pt <= cdf_n) {
      symbol = eob_pt - 1;
    } else {
      // The last CDF symbol escapes to eobPt = cdf_n + escape.
      symbol = cdf_n - 1;
      escape = eob_pt - cdf_n;
    }
    if (!sym(cdf, cdf_n, symbol)) return false;
    if (symbol == cdf_n - 1 && escape_bits > 0) {
      if (!write_literal(escape_bits, static_cast<std::uint32_t>(escape))) {
        return false;
      }
    }
    if (eob_pt >= 3) {
      const int base = (1 << (eob_pt - 2)) + 1;
      const int extra = eob - base;
      const int eob_extra = (extra >> (eob_pt - 3)) & 1;
      if (!sym(&cdfs_.eob_extra[0], 2, eob_extra)) return false;
      for (int i = eob_pt - 4; i >= 0; --i) {
        if (!raw_bit(((extra >> i) & 1) != 0)) return false;
      }
    }
    return true;
  }

  bool write_quant_escape(int x, int& hr_level_avg) {
    const int pred_level = hr_level_avg;
    const int m = clip3(1, 6, pred_level <= 0 ? 0 : floor_log2(pred_level));
    const int k = m + 1;
    const int c_max = std::min(m + 4, 6);
    if (x < (c_max << m)) {
      const int q = x >> m;
      for (int i = 0; i < q; ++i) {
        if (!raw_bit(false)) return false;
      }
      if (!raw_bit(true)) return false;
      if (!write_literal(m, static_cast<std::uint32_t>(x & ((1 << m) - 1)))) {
        return false;
      }
    } else {
      for (int i = 0; i < c_max; ++i) {
        if (!raw_bit(false)) return false;
      }
      int g = 0;
      while (true) {
        const int length = k + g;
        const int x_base = (c_max << m) + (1 << length) - (1 << k);
        if (x < x_base + (1 << length)) {
          for (int i = 0; i < g; ++i) {
            if (!raw_bit(false)) return false;
          }
          if (!raw_bit(true)) return false;
          if (!write_literal(length,
                             static_cast<std::uint32_t>(x - x_base))) {
            return false;
          }
          break;
        }
        ++g;
      }
    }
    hr_level_avg = (x + hr_level_avg) >> 1;
    return true;
  }

  int dc_sign_context(int x4, int y4, int w4, int h4) const {
    if (const char* forced = std::getenv("AV2ENC_M1_DCSIGN_CTX")) {
      return std::atoi(forced);
    }
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
    if (dc_sign < 0) return 1;
    if (dc_sign > 0) return 2;
    return 0;
  }

  // Luma inter transform-type syntax (§5.20.8.2).  M2 signals DCT_DCT for
  // every codable inter TX block.  The symbol mapping is set-dependent:
  //   INTER_1: type group 0 + offset 7  -> Tx_Type_Inter_Inv_Set1[7] = DCT_DCT
  //   INTER_2: type group 0 + offset 3  -> Inv_Set2[3]          = DCT_DCT
  //   DCT_IDTX: symbol 1                -> Inv_Set3[1]          = DCT_DCT
  //   long sets: is_long_side_dct = 1, type 0 -> Inv_Long[1][*][0]
  bool write_inter_tx_type(int tx_size, int eob, int coded_w, int coded_h) {
    const InterTxSet set = inter_tx_set(tx_size);
    if (set == InterTxSet::kDctOnly) return true;
    const int ctx = inter_tx_type_context(eob, coded_w, coded_h, true);
    const int sqr = kTxSizeSqr[tx_size];
    switch (set) {
      case InterTxSet::kWide64:
      case InterTxSet::kHigh64:
        // WIDE_64/HIGH_64 imply is_long_side_dct = 1; only the type is coded.
        return sym(&cdfs_.inter_tx_type_long[ctx][sqr][0], 4, 0);
      case InterTxSet::kWide32:
      case InterTxSet::kHigh32:
        if (!sym(&cdfs_.is_long_side_dct[1][0], 2, 1)) return false;
        return sym(&cdfs_.inter_tx_type_long[ctx][sqr][0], 4, 0);
      case InterTxSet::kInter1: {
        if (!sym(&cdfs_.inter_tx_type_set1[ctx][sqr][0], 2, 0)) return false;
        const int off_ctx =
            inter_tx_type_context(eob, coded_w, coded_h, false);
        return sym(&cdfs_.inter_tx_type_index_set1[off_ctx][0], 8, 7);
      }
      case InterTxSet::kInter2: {
        if (!sym(&cdfs_.inter_tx_type_set2[ctx][0], 2, 0)) return false;
        const int off_ctx =
            inter_tx_type_context(eob, coded_w, coded_h, false);
        return sym(&cdfs_.inter_tx_type_index_set2[off_ctx][0], 8, 3);
      }
      case InterTxSet::kDctIdtx:
        return sym(&cdfs_.inter_tx_type_set3[ctx][sqr][0], 2, 1);
      case InterTxSet::kDctOnly:
        return true;
    }
    return false;
  }

  // `level_base` is the index of this plane's first scan-order level.
  bool code_tx(const BlockSyntax& node, int plane, std::size_t level_base,
               int bsize_w, int bsize_h, std::size_t levels_end,
               bool is_inter, std::string& error) {
    const int tx_size = node.tx_size[plane];
    if (tx_size >= 25) {
      error = "invalid tx_size " + std::to_string(tx_size) + " on plane " +
              std::to_string(plane);
      return false;
    }
    const auto params_opt =
        txv::make_tx_params(static_cast<txv::TxSize>(tx_size),
                            txv::TxType::kDctDct);
    if (!params_opt.has_value()) {
      error = "unsupported transform size";
      return false;
    }
    const txv::TxParams& params = *params_opt;
    const int coded_w = params.coded_w;
    const int coded_h = params.coded_h;
    const int adj_w = params.w;
    const int adj_h = params.h;
    const int area = adj_w * adj_h;
    const int eob = static_cast<int>(node.eob[plane]);
    if (eob < 0 || eob > area) {
      error = "eob " + std::to_string(eob) + " outside the plane " +
              std::to_string(plane) + " transform area";
      return false;
    }
    if (level_base + static_cast<std::size_t>(eob) > levels_end) {
      error = "coefficient levels overflow the flat level array";
      return false;
    }
    const int w4 = coded_w >> 2;
    const int h4 = coded_h >> 2;
    const int sub_x = plane > 0 ? 1 : 0;
    const int sub_y = plane > 0 ? 1 : 0;
    const int x4 = node.mi_col >> sub_x;
    const int y4 = node.mi_row >> sub_y;
    const int max_x4 = plane == 0 ? mi_cols_ : mi_cols_ / 2;
    const int max_y4 = plane == 0 ? mi_rows_ : mi_rows_ / 2;

    int ctx = 0;
    if (plane == 0) {
      int top = 0;
      int left = 0;
      for (int k = 0; k < w4; ++k) {
        if (x4 + k < max_x4) top |= above_level(0, x4 + k);
      }
      for (int k = 0; k < h4; ++k) {
        if (y4 + k < max_y4) left |= left_level(0, y4 + k);
      }
      top = std::min(top, 4);
      left = std::min(left, 4);
      if (bsize_w == coded_w && bsize_h == coded_h) {
        ctx = 0;
      } else {
        ctx = (top + left + 3) >> 1;
      }
    } else {
      int above = 0;
      int left = 0;
      for (int i = 0; i < w4; ++i) {
        if (x4 + i < max_x4) {
          above |= above_level(plane, x4 + i);
          above |= above_dc(plane, x4 + i);
        }
      }
      for (int i = 0; i < h4; ++i) {
        if (y4 + i < max_y4) {
          left |= left_level(plane, y4 + i);
          left |= left_dc(plane, y4 + i);
        }
      }
      ctx = (above != 0 ? 1 : 0) + (left != 0 ? 1 : 0);
      if (plane == 2) {
        if (bsize_w * bsize_h > coded_w * coded_h) ctx += 3;
        if (eob_u_ != 0) ctx += 6;
      } else {
        ctx += 6;
      }
    }

    const int tsc = tx_sz_ctx(tx_size);
    std::uint16_t* skip_cdf =
        plane == 2 ? &cdfs_.v_txb_skip[ctx][0]
                   : &cdfs_.txb_skip[is_inter ? 1 : 0][tsc][ctx][0];
    if (!sym(skip_cdf, 2, eob == 0 ? 1 : 0)) return false;
    if (eob == 0) {
      for (int i = 0; i < w4; ++i) {
        above_level(plane, x4 + i) = 0;
        above_dc(plane, x4 + i) = 0;
      }
      for (int i = 0; i < h4; ++i) {
        left_level(plane, y4 + i) = 0;
        left_dc(plane, y4 + i) = 0;
      }
      if (plane == 1) eob_u_ = 0;
      return true;
    }

    if (!write_eob(eob, plane, coded_w, coded_h, is_inter)) {
      error = error_;
      return false;
    }
    if (plane == 0) {
      if (is_inter) {
        if (!write_inter_tx_type(tx_size, eob, coded_w, coded_h)) {
          return false;
        }
      } else {
        switch (luma_tx_type_syntax(tx_size, eob)) {
        case TxTypeSyntax::kNone:
          break;
        case TxTypeSyntax::kIntraSet1:
          // DCT_DCT is symbol 0 in every TX_SET_INTRA_1 mode table.
          if (!sym(&cdfs_.intra_tx_type_set1[kTxSizeSqr[tx_size]][0], 7, 0)) {
            return false;
          }
          break;
        case TxTypeSyntax::kLong32:
          // A wide/high 32 TX set codes both the long-side flag and the type.
          if (!sym(&cdfs_.is_long_side_dct[0][0], 2, 1)) return false;
          if (!sym(&cdfs_.intra_tx_type_long[kTxSizeSqr[tx_size]][0], 4, 0)) {
            return false;
          }
          break;
        case TxTypeSyntax::kLong64:
          // WIDE_64/HIGH_64 imply is_long_side_dct = 1; only the type is coded.
          if (!sym(&cdfs_.intra_tx_type_long[kTxSizeSqr[tx_size]][0], 4, 0)) {
            return false;
          }
          break;
        }
      }
    }

    static constexpr int kSigRef[5][2] = {
        {0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
    static constexpr int kMagRef[3][2] = {{0, 1}, {1, 0}, {1, 1}};

    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    build_scan(adj_w, adj_h, scan);
    std::vector<int> level_pre(static_cast<std::size_t>(area), 0);
    // Levels arrive in scan order: levels_[level_base + c] is scan position c.
    const auto level_at = [&](int c) {
      return static_cast<int>(
          levels_[level_base + static_cast<std::size_t>(c)]);
    };
    for (int c = eob - 1; c >= 0; --c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      const int row = pos / adj_w;
      const int col = pos % adj_w;
      const int abs_value = std::abs(level_at(c));
      const bool is_lf = plane == 0 ? (row + col) < 4 : (row + col) < 1;
      const int base_levels = is_lf ? 4 : 2;
      const int base_max = is_lf ? 5 : 3;
      const int br_max = (is_lf && plane > 0) ? 0 : 3;
      const int pre_max = base_max + br_max;
      int base_level = 0;
      int br_level = 0;
      if (abs_value <= base_levels) {
        base_level = abs_value;
      } else {
        base_level = std::min(base_levels + 1, base_max);
        const int pre = std::min(abs_value, pre_max);
        br_level = pre - base_level;
      }

      int ctx_eob = 0;
      if (c != 0) {
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
            if (!sym(&cdfs_.coeff_base_lf_eob[tsc][ctx_eob][0], 5, symbol)) {
              return false;
            }
          } else {
            if (!sym(&cdfs_.coeff_base_eob[tsc][ctx_eob][0], 3, symbol)) {
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
          if (rr < adj_h && cc < adj_w) {
            const int limit = is_lf ? 5 : 3;
            mag += std::min(
                level_pre[static_cast<std::size_t>(rr) * adj_w + cc], limit);
          }
        }
        const int mctx = (mag + 1) >> 1;
        if (plane == 0) {
          if (is_lf) {
            int lf_ctx = 0;
            if (c == 0) {
              lf_ctx = std::min(mctx, 8);
            } else if (row + col < 2) {
              lf_ctx = std::min(mctx, 6) + 9;
            } else {
              lf_ctx = std::min(mctx, 4) + 16;
            }
            if (!sym(&cdfs_.coeff_base_lf[tsc][lf_ctx][0][0], 6, base_level)) {
              return false;
            }
          } else {
            const int ctx2 = std::min(mctx, 4);
            int hf_ctx = ctx2;
            if (row + col >= 6 && row + col < 8) {
              hf_ctx = ctx2 + 5;
            } else if (row + col >= 8) {
              hf_ctx = ctx2 + 10;
            }
            if (!sym(&cdfs_.coeff_base[tsc][hf_ctx][0][0], 4, base_level)) {
              return false;
            }
          }
        } else {
          const int ctx2 = std::min(mctx, 3);
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
          if (rr < adj_h && cc < adj_w) {
            mag += std::min(
                level_pre[static_cast<std::size_t>(rr) * adj_w + cc], 5);
          }
        }
        mag = std::min((mag + 1) >> 1, 6);
        if (plane == 0) {
          int br_ctx = mag;
          if (is_lf) {
            if (pos != 0) br_ctx = mag + 7;
            if (!sym(&cdfs_.coeff_br_lf[br_ctx][0], 4, br_level)) {
              return false;
            }
          } else {
            if (!sym(&cdfs_.coeff_br[mag][0], 4, br_level)) return false;
          }
        } else {
          if (!sym(&cdfs_.coeff_br_uv[std::min(mag, 3)][0], 4, br_level)) {
            return false;
          }
        }
      }
      level_pre[static_cast<std::size_t>(pos)] = pre_level;
    }

    int cul_level = 0;
    int dc_category = 0;
    int hr_level_avg = 0;
    for (int c = eob - 1; c >= 0; --c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      const int row = pos / adj_w;
      const int col = pos % adj_w;
      const int pre_level = level_pre[static_cast<std::size_t>(pos)];
      const int level = level_at(c);
      const int abs_value = std::abs(level);
      const bool negative = level < 0;
      if (pre_level != 0) {
        if (row == 0 && col == 0 && plane == 0) {
          const int sign_ctx = dc_sign_context(x4, y4, w4, h4);
          if (trace_) {
            std::cerr << "  dc_sign ctx=" << sign_ctx
                      << " sign=" << (negative ? 1 : 0) << "\n";
          }
          if (!sym(&cdfs_.dc_sign[0][0][sign_ctx][0], 2, negative ? 1 : 0)) {
            return false;
          }
        } else {
          if (!raw_bit(negative)) return false;
        }
        const bool is_lf = plane == 0 ? (row + col) < 4 : (row + col) < 1;
        const int max_level = is_lf ? (plane == 0 ? 8 : 5) : 6;
        if (abs_value >= max_level) {
          if (!write_quant_escape(abs_value - pre_level, hr_level_avg)) {
            return false;
          }
        }
      }
      if (pos == 0 && abs_value > 0) dc_category = negative ? 1 : 2;
      cul_level = std::min(4, cul_level + abs_value);
    }

    for (int i = 0; i < w4; ++i) {
      above_level(plane, x4 + i) = static_cast<std::uint8_t>(cul_level);
      above_dc(plane, x4 + i) = static_cast<std::uint8_t>(dc_category);
    }
    for (int i = 0; i < h4; ++i) {
      left_level(plane, y4 + i) = static_cast<std::uint8_t>(cul_level);
      left_dc(plane, y4 + i) = static_cast<std::uint8_t>(dc_category);
    }
    if (plane == 1) eob_u_ = 1;
    if (trace_ && plane == 0) {
      std::cerr << "  tx p0 eob=" << eob << " cul=" << cul_level
                << " dcCat=" << dc_category << " dcLevel="
                << level_at(0) << " x4=" << x4 << " y4=" << y4 << " w4=" << w4
                << " h4=" << h4 << "\n";
    }
    return true;
  }

  // ---------------------------------------------------------------------
  // Inter block layer (§5.20.7)
  // ---------------------------------------------------------------------
  std::size_t cell_index(int r, int c) const {
    return static_cast<std::size_t>(r) *
               static_cast<std::size_t>(mi_cols_) +
           static_cast<std::size_t>(c);
  }

  // add_neighbor() from decode_block(). NPosBuf keeps the first two valid
  // positions in the order below-left, above-right, left, above.
  int neighbor_positions(int pos[2][2]) const {
    int n = 0;
    const auto add = [&](int nr, int nc) {
      if (n < 2 && inside_mi(nr, nc)) {
        pos[n][0] = nr;
        pos[n][1] = nc;
        ++n;
      }
    };
    add(cur_r_ + cur_h4_ - 1, cur_c_ - 1);
    add(cur_r_ - 1, cur_c_ + cur_w4_ - 1);
    add(cur_r_, cur_c_ - 1);
    add(cur_r_ - 1, cur_c_);
    return n;
  }

  bool has_newmv_mode(int ymode) const { return ymode == 16; }

  // find_mode_ctx() §7.11.2 for single prediction.
  int new_mv_context() const {
    int new_mv_count = 0;
    const bool sb_border = (cur_r_ & 15) == 0;
    const auto scan = [&](int dr, int dc) -> int {
      const int mr = cur_r_ + dr;
      const int mc = cur_c_ + dc;
      if (!inside_mi(mr, mc)) return 0;
      const std::size_t idx = cell_index(mr, mc);
      if (is_inter_grid_[idx] == 0) return 0;
      if (has_newmv_mode(ymode_grid_[idx])) {
        new_mv_count = std::min(3, new_mv_count + 1);
      }
      return 1;
    };
    const int left_a = scan(cur_h4_ - 1, -1);
    const int above_a = scan(-1, cur_w4_ - 1);
    const int left_b = scan(0, -1);
    const int above_b =
        (cur_w4_ >= (sb_border ? 4 : 2)) ? scan(-1, 0) : 0;
    const int nearest = ((above_a || above_b) ? 1 : 0) +
                        ((left_a || left_b) ? 1 : 0);
    return nearest + (new_mv_count > 0 ? 2 : 0);
  }

  int is_inter_context() const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos);
    const auto intra_at = [&](int i) {
      return is_inter_grid_[cell_index(pos[i][0], pos[i][1])] == 0;
    };
    if (n == 2) {
      return (intra_at(0) && intra_at(1)) ? 3 : (intra_at(0) || intra_at(1));
    }
    if (n == 1) {
      return 2 * (intra_at(0) ? 1 : 0);
    }
    return 0;
  }

  int skip_context() const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos);
    int ctx = 0;
    for (int i = 0; i < n; ++i) {
      ctx += skip_grid_[cell_index(pos[i][0], pos[i][1])] != 0 ? 1 : 0;
    }
    return ctx;  // skip_mode == 0, so the +3 half-context is absent
  }

  bool code_is_inter(bool value) {
    return sym(&cdfs_.is_inter[is_inter_context()][0], 2, value ? 1 : 0);
  }

  bool code_skip_flag(bool value) {
    return sym(&cdfs_.skip[skip_context()][0], 2, value ? 1 : 0);
  }

  bool code_single_mode(int ctx, int single_mode) {
    return sym(&cdfs_.single_mode[ctx][0], 3, single_mode);
  }

  bool code_drl_index(int ctx, int index) {
    const int m = static_cast<int>(config_.max_drl_bits_minus_1) + 1;
    if (index < 0 || index > m) return false;
    for (int idx = 0; idx < m; ++idx) {
      const int mode = idx < index ? 1 : 0;
      if (!sym(&cdfs_.drl_mode[std::min(idx, 2)][ctx][0], 2, mode)) {
        return false;
      }
      if (mode == 0) break;
    }
    return true;
  }

  bool write_ns_bits(std::uint32_t n, std::uint32_t value) {
    unsigned w = 0;
    while ((1U << w) < n) ++w;
    const std::uint32_t m = (1U << w) - n;
    if (value < m) {
      return write_literal(static_cast<int>(w - 1U), value);
    }
    const std::uint32_t x = value + m;
    if (!write_literal(static_cast<int>(w - 1U), x >> 1U)) return false;
    return raw_bit((x & 1U) != 0U);
  }

  // read_mv() §5.20.7.20 for MvPrecision = 1/8 pel (P = 6).
  bool write_mv(std::int32_t dy, std::int32_t dx) {
    const std::int32_t ay = std::abs(dy);
    const std::int32_t ax = std::abs(dx);
    const std::int32_t shell = ay + ax;
    // The shell index is bounded by 2^17 - 1 in the spec; MV_UPP/MV_LOW keep
    // each component in [-2^16, 2^16], so this is a defensive check.
    if (shell >= (1 << 17)) return false;
    const int cls = shell == 0 ? 0 : floor_log2(static_cast<int>(shell));
    const std::int32_t base = cls == 0 ? 0 : (1 << cls);
    const std::int32_t off = shell - base;

    const int setsym = cls >= 8 ? 1 : 0;
    const int class_sym = cls >= 8 ? cls - 8 : cls;
    if (!sym(&cdfs_.shell_set[0], 2, setsym)) return false;
    std::uint16_t* class_cdf =
        setsym ? &cdfs_.shell6_class1[0] : &cdfs_.shell6_class0[0];
    if (!sym(class_cdf, 8, class_sym)) return false;
    if (setsym && class_sym == 7) {
      if (!sym(&cdfs_.shell_last_two[0], 2, cls - 15)) return false;
    }

    if (cls < 2) {
      if (!sym(&cdfs_.offset_low[cls][0], 2, off)) return false;
    } else if (cls == 2) {
      if (off == 0) {
        if (!sym(&cdfs_.offset_class2[0], 2, 0)) return false;
      } else {
        if (!sym(&cdfs_.offset_class2[0], 2, 1)) return false;
        if (off == 1) {
          if (!raw_bit(false)) return false;
        } else {
          if (!raw_bit(true)) return false;
          if (!raw_bit(off == 3)) return false;
        }
      }
    } else {
      for (int i = 0; i < cls; ++i) {
        if (!sym(&cdfs_.offset_other[i][0], 2, (off >> i) & 1)) {
          return false;
        }
      }
    }

    if (shell > 0) {
      const std::int32_t max_pair = shell >> 1;
      bool skip_col_bit = false;
      std::int32_t col = 0;
      if (ax == max_pair && ((shell & 1) == 0)) {
        skip_col_bit = true;
        col = max_pair;
      } else if (ax <= max_pair) {
        col = ax;
      } else {
        col = ay;
      }

      if (max_pair > 0) {
        const int max_bits = std::min<std::int32_t>(max_pair, 2);
        const std::int32_t prefix_col = std::min<std::int32_t>(col, 2);
        for (int i = 0; i < max_bits; ++i) {
          const int greater = prefix_col > i ? 1 : 0;
          if (!sym(&cdfs_.col_mv_greater[i][0], 2, greater)) return false;
          if (greater == 0) break;
        }
        if (max_pair > 2 && prefix_col == 2) {
          if (!write_ns_bits(static_cast<std::uint32_t>(max_pair - 1),
                             static_cast<std::uint32_t>(col - 2))) {
            return false;
          }
        }
      }

      if (!skip_col_bit) {
        const int index = ax <= max_pair ? 0 : 1;
        if (!sym(&cdfs_.col_mv_index[std::min(cls, 3)][0], 2, index)) {
          return false;
        }
      }
    }

    if (dy != 0 && !raw_bit(dy < 0)) return false;
    if (dx != 0 && !raw_bit(dx < 0)) return false;
    return true;
  }

  bool write_inter_mode_info(const BlockSyntax& node, std::string& error) {
    if (node.inter_mode > kInterNewMv) {
      error = "invalid inter mode";
      return false;
    }

    // Validate the decision against the spec-exact merge/MV-stack layer.  This
    // catches producer bugs before they desynchronise the decoder.
    merge::StackParams params;
    params.mi_row = node.mi_row;
    params.mi_col = node.mi_col;
    params.bw4 = cur_w4_;
    params.bh4 = cur_h4_;
    params.sb_h4 = 16;
    params.mi_cols = mi_cols_;
    params.mi_rows = mi_rows_;
    params.mi = mv_grid_.data();
    params.ref_frame[0] = kRefLastAv2;
    params.ref_frame[1] = merge::kRefNone;
    params.is_compound = false;
    params.global_mv[0] = inter::mc::Mv{0, 0};
    params.global_mv[1] = inter::mc::Mv{0, 0};
    // DRL reordering is not gated; the sequence header signals DISABLED.
    params.reorder = merge::DrlReorder::kDisabled;
    params.max_drl_bits_minus_1 =
        static_cast<int>(config_.max_drl_bits_minus_1);
    merge::MvStack stack;
    if (merge::build_stack_checked(params, &stack) != merge::Status::kOk) {
      error = "build_mv_stack failed";
      return false;
    }

    if (node.inter_mode == kInterGlobalMv) {
      if (node.mv_y != 0 || node.mv_x != 0) {
        error = "GLOBALMV block has a non-zero MV";
        return false;
      }
    } else {
      if (stack.count <= 0) {
        error = "empty MV stack";
        return false;
      }
      const int drl = node.inter_mode == kInterNewMv ? 0 : node.drl_index;
      const merge::SingleMode mode = node.inter_mode == kInterNearMv
                                         ? merge::SingleMode::kNearMv
                                         : merge::SingleMode::kNewMv;
      merge::ResolvedMode resolved;
      if (merge::resolve_single_mode(
              stack, params, mode, drl,
              inter::mc::Mv{node.mvd_y, node.mvd_x}, &resolved) !=
          merge::Status::kOk) {
        error = "resolve_mode failed";
        return false;
      }
      if (resolved.mv[0].y != node.mv_y || resolved.mv[0].x != node.mv_x) {
        error = "block MV does not match the signalled inter mode";
        return false;
      }
      if (node.inter_mode == kInterNearMv &&
          node.drl_index >= stack.count) {
        error = "DRL index outside the MV stack";
        return false;
      }
    }

    const int ctx = new_mv_context();
    // YMode = NEARMV + single_mode. YMode 14 = NEARMV, 15 = GLOBALMV,
    // 16 = NEWMV, so single_mode is 0 / 1 / 2 in that order.
    const int single_mode =
        node.inter_mode == kInterGlobalMv
            ? 1
            : (node.inter_mode == kInterNearMv ? 0 : 2);
    if (!code_single_mode(ctx, single_mode)) {
      error = error_;
      return false;
    }
    if (node.inter_mode == kInterNearMv || node.inter_mode == kInterNewMv) {
      // §5.20.7.6: read_drl_idx() is present for both NEARMV and NEWMV.
      // M2 NEWMV always uses DRL index 0 (the predictor resolve_mode used).
      const int drl = node.inter_mode == kInterNewMv ? 0 : node.drl_index;
      if (!code_drl_index(ctx, drl)) {
        error = "write drl_idx failed";
        return false;
      }
    }
    if (node.inter_mode == kInterNewMv) {
      if (!write_mv(node.mvd_y, node.mvd_x)) {
        error = error_;
        return false;
      }
    }
    return true;
  }

  void update_mode_grids(const BlockSyntax& node) {
    for (int yy = 0; yy < cur_h4_; ++yy) {
      const int rr = cur_r_ + yy;
      if (rr >= mi_rows_) break;
      for (int xx = 0; xx < cur_w4_; ++xx) {
        const int cc = cur_c_ + xx;
        if (cc >= mi_cols_) break;
        const std::size_t idx = cell_index(rr, cc);
        is_inter_grid_[idx] = node.is_inter ? 1 : 0;
        skip_grid_[idx] = node.skip_flag ? 1 : 0;
        ymode_grid_[idx] =
            node.is_inter
                ? static_cast<std::uint8_t>(
                      (node.inter_mode == kInterGlobalMv
                           ? 15
                           : (node.inter_mode == kInterNearMv ? 14 : 16)))
                : node.y_mode;
        inter::mc::MiInfo& mi = mv_grid_[idx];
        mi = inter::mc::MiInfo{};
        mi.ref[0] = node.is_inter ? kRefLastAv2 : kRefIntraAv2;
        mi.ref[1] = merge::kRefNone;
        mi.mv[0][0] = node.mv_y;
        mi.mv[0][1] = node.mv_x;
        mi.mv[1][0] = node.mv_y;
        mi.mv[1][1] = node.mv_x;
        mi.is_inter = node.is_inter ? 1 : 0;
        mi.w4 = static_cast<std::int8_t>(cur_w4_);
        mi.h4 = static_cast<std::int8_t>(cur_h4_);
        mi.col_base = static_cast<std::int16_t>(cur_c_);
        mi.row_base = static_cast<std::int16_t>(cur_r_);
      }
    }
  }

  // reset_block_context(bw4, bh4) from decode_block(): a skip block resets
  // the coefficient level/DC contexts over its area. Inter blocks in this
  // milestone are always skip blocks, so the next intra block must see zero
  // contexts even if an earlier intra block populated them.
  void reset_coeff_contexts(int r, int c, int w4, int h4) {
    const int x4 = c;
    const int y4 = r;
    for (int i = 0; i < w4; ++i) {
      above_level(0, x4 + i) = 0;
      above_dc(0, x4 + i) = 0;
    }
    for (int i = 0; i < h4; ++i) {
      left_level(0, y4 + i) = 0;
      left_dc(0, y4 + i) = 0;
    }
    const int cx4 = c >> 1;
    const int cy4 = r >> 1;
    const int cw4 = w4 >> 1;
    const int ch4 = h4 >> 1;
    for (int plane = 1; plane < kPlaneCount; ++plane) {
      for (int i = 0; i < cw4; ++i) {
        above_level(plane, cx4 + i) = 0;
        above_dc(plane, cx4 + i) = 0;
      }
      for (int i = 0; i < ch4; ++i) {
        left_level(plane, cy4 + i) = 0;
        left_dc(plane, cy4 + i) = 0;
      }
    }
  }

  // Intra mode + residual syntax. Used by keyframe blocks and by intra
  // fallback blocks inside an inter frame.
  bool write_intra_leaf(const BlockSyntax& node, int w, int h, int cw, int ch,
                        std::string& error) {
    // Validate the declared transform sizes against TX_MODE_LARGEST geometry
    // before emitting anything.
    const int coded_w[kPlaneCount] = {w, cw, cw};
    const int coded_h[kPlaneCount] = {h, ch, ch};
    for (int plane = 0; plane < kPlaneCount; ++plane) {
      const auto params_opt = txv::make_tx_params(
          static_cast<txv::TxSize>(node.tx_size[plane]),
          txv::TxType::kDctDct);
      if (!params_opt.has_value() ||
          params_opt->coded_w != coded_w[plane] ||
          params_opt->coded_h != coded_h[plane]) {
        error = "tx_size on plane " + std::to_string(plane) +
                " does not match the TX_MODE_LARGEST block size";
        return false;
      }
    }

    const std::size_t levels_end = levels_.size();
    if (!code_y_mode(node)) {
      error = error_;
      return false;
    }
    if (!code_uv_mode((node.flags & kFlagIsCfl) != 0, node.uv_mode,
                      node.y_mode)) {
      error = error_;
      return false;
    }
    std::size_t level_base = node.levels_offset;
    if (!code_tx(node, 0, level_base, w, h, levels_end, false, error)) {
      return false;
    }
    level_base += node.eob[0];
    if (!code_tx(node, 1, level_base, cw, ch, levels_end, false, error)) {
      return false;
    }
    level_base += node.eob[1];
    if (!code_tx(node, 2, level_base, cw, ch, levels_end, false, error)) {
      return false;
    }
    if (error.empty() && !error_.empty()) error = error_;
    return true;
  }

  // Inter residual syntax: transform coefficients for a non-skip inter leaf.
  // Mirrors write_intra_leaf's residual half; prediction is already applied by
  // the producer.  Chroma never signals a transform type and M2 does not
  // enable CCTX/parity hiding/TCQ, so the plane order is luma, U, V.
  bool write_inter_residual(const BlockSyntax& node, int w, int h, int cw,
                            int ch, std::string& error) {
    const int coded_w[kPlaneCount] = {w, cw, cw};
    const int coded_h[kPlaneCount] = {h, ch, ch};
    for (int plane = 0; plane < kPlaneCount; ++plane) {
      const auto params_opt = txv::make_tx_params(
          static_cast<txv::TxSize>(node.tx_size[plane]),
          txv::TxType::kDctDct);
      if (!params_opt.has_value() ||
          params_opt->coded_w != coded_w[plane] ||
          params_opt->coded_h != coded_h[plane]) {
        error = "inter tx_size on plane " + std::to_string(plane) +
                " does not match the TX_MODE_LARGEST block size";
        return false;
      }
    }

    const std::size_t levels_end = levels_.size();
    std::size_t level_base = node.levels_offset;
    if (!code_tx(node, 0, level_base, w, h, levels_end, true, error)) {
      return false;
    }
    level_base += node.eob[0];
    if (!code_tx(node, 1, level_base, cw, ch, levels_end, true, error)) {
      return false;
    }
    level_base += node.eob[1];
    if (!code_tx(node, 2, level_base, cw, ch, levels_end, true, error)) {
      return false;
    }
    if (error.empty() && !error_.empty()) error = error_;
    return true;
  }


  // ---------------------------------------------------------------------
  // Coding a leaf block
  // ---------------------------------------------------------------------
  bool write_block(const BlockSyntax& node, std::string& error) {
    const int r = node.mi_row;
    const int c = node.mi_col;
    const auto size = static_cast<BlockSize>(node.bsize);
    const int w = partition::block_width(size);
    const int h = partition::block_height(size);
    const int w4 = w >> 2;
    const int h4 = h >> 2;
    cur_r_ = r;
    cur_c_ = c;
    cur_w4_ = w4;
    cur_h4_ = h4;

    const BlockSize chroma_size =
        partition::chroma_plane_residual_size(geometry_, size);
    if (!partition::is_valid_block_size(chroma_size)) {
      error = "invalid chroma residual size";
      return false;
    }
    const int cw = partition::block_width(chroma_size);
    const int ch = partition::block_height(chroma_size);

    if (plan_.inter_frame) {
      if (!code_is_inter(node.is_inter != 0)) {
        error = error_;
        return false;
      }
      if (node.is_inter) {
        if (!code_skip_flag(node.skip_flag != 0)) {
          error = error_;
          return false;
        }
        if (!write_inter_mode_info(node, error)) return false;
        if (node.skip_flag) {
          reset_coeff_contexts(r, c, w4, h4);
        } else if (!write_inter_residual(node, w, h, cw, ch, error)) {
          return false;
        }
      } else {
        if (!write_intra_leaf(node, w, h, cw, ch, error)) return false;
      }
    } else {
      // Intra keyframe: intra_frame_mode_info() does not code is_inter.
      if (node.is_inter) {
        error = "keyframe block marked inter";
        return false;
      }
      if (!write_intra_leaf(node, w, h, cw, ch, error)) return false;
    }

    // Update partition context.
    for (int yy = 0; yy < h4; ++yy) {
      left_mi(r + yy) = kMiHeightLog2[static_cast<int>(size)];
    }
    for (int xx = 0; xx < w4; ++xx) {
      above_mi(c + xx) = kMiWidthLog2[static_cast<int>(size)];
    }

    update_mode_grids(node);

    if (!node.is_inter) {
      const YModeCode ycode = y_mode_code(node.y_mode, node.y_angle_delta);
      for (int yy = 0; yy < h4; ++yy) {
        const int rr = r + yy;
        if (rr >= mi_rows_) break;
        for (int xx = 0; xx < w4; ++xx) {
          const int cc = c + xx;
          if (cc >= mi_cols_) break;
          intra_joint(rr, cc) = ycode.joint;
        }
      }
    } else {
      for (int yy = 0; yy < h4; ++yy) {
        const int rr = r + yy;
        if (rr >= mi_rows_) break;
        for (int xx = 0; xx < w4; ++xx) {
          const int cc = c + xx;
          if (cc >= mi_cols_) break;
          intra_joint(rr, cc) = 0;  // inter_block_mode_info: DC_PRED
        }
      }
    }

    const int cr = r / 2;
    const int cc0 = c / 2;
    const int cw4 = cw / 4;
    const int ch4 = ch / 4;
    const int chroma_mi_rows = mi_rows_ / 2;
    const int chroma_mi_cols = mi_cols_ / 2;
    const bool is_cfl = !node.is_inter && (node.flags & kFlagIsCfl) != 0;
    for (int yy = 0; yy < ch4; ++yy) {
      const int rr = cr + yy;
      if (rr >= chroma_mi_rows) break;
      for (int xx = 0; xx < cw4; ++xx) {
        const int cc = cc0 + xx;
        if (cc >= chroma_mi_cols) break;
        uvcfl(rr, cc) = is_cfl ? 1 : 0;
      }
    }
    return true;
  }

  // Members
  const FramePlan& plan_;
  std::span<const BlockSyntax> blocks_;
  std::span<const std::int32_t> levels_;
  const M2Config& config_;
  FrameGeometry geometry_;
  int mi_cols_ = 0;
  int mi_rows_ = 0;

  CdfState cdfs_{};
  av2::Encoder enc_{};
  std::vector<std::uint8_t> tile_buffer_;
  std::vector<std::uint8_t> payload_;
  std::string error_;

  std::array<std::vector<std::uint8_t>, kPlaneCount> above_level_;
  std::array<std::vector<std::uint8_t>, kPlaneCount> left_level_;
  std::array<std::vector<std::uint8_t>, kPlaneCount> above_dc_;
  std::array<std::vector<std::uint8_t>, kPlaneCount> left_dc_;
  std::vector<int> above_mi_;
  std::vector<int> left_mi_;
  std::vector<int> intra_joint_;
  std::vector<std::uint8_t> uvcfl_;

  // Inter context grids. One entry per 4x4 luma cell, updated as blocks are
  // coded; the syntax writer derives every inter context from these exactly
  // like the decoder's IsInters/Skips/YModes/RefFrames arrays.
  std::vector<std::uint8_t> is_inter_grid_;
  std::vector<std::uint8_t> skip_grid_;
  std::vector<std::uint8_t> ymode_grid_;
  std::vector<inter::mc::MiInfo> mv_grid_;

  int cur_r_ = 0;
  int cur_c_ = 0;
  int cur_w4_ = 0;
  int cur_h4_ = 0;
  int eob_u_ = 0;
  bool trace_ = false;
};

}  // namespace

int FramePlan::sb_cols() const {
  FrameGeometry geometry;
  geometry.width = width;
  geometry.height = height;
  geometry.sb_size = sb_size;
  const int sb_mi = partition::sb_size_mi(geometry);
  return (partition::mi_cols(geometry) + sb_mi - 1) / sb_mi;
}

int FramePlan::sb_rows() const {
  FrameGeometry geometry;
  geometry.width = width;
  geometry.height = height;
  geometry.sb_size = sb_size;
  const int sb_mi = partition::sb_size_mi(geometry);
  return (partition::mi_rows(geometry) + sb_mi - 1) / sb_mi;
}

int FramePlan::superblock_count() const { return sb_cols() * sb_rows(); }

bool assemble_m2_frame(const FramePlan& plan,
                             std::span<const BlockSyntax> blocks,
                             std::span<const std::int32_t> levels,
                             const M2Config& config,
                             std::vector<std::uint8_t>& tile_payload,
                             std::string& error, std::size_t* tile_bytes) {
  error.clear();
  tile_payload.clear();
  if (plan.width <= 0 || plan.height <= 0) {
    error = "empty frame plan";
    return false;
  }
  if ((plan.width & 1) != 0 || (plan.height & 1) != 0) {
    error = "M2 4:2:0 requires even dimensions";
    return false;
  }
  if ((plan.width & 7) != 0 || (plan.height & 7) != 0) {
    error = "M2 baseline requires width and height divisible by 8";
    return false;
  }
  if (plan.sb_size != 64 && plan.sb_size != 128) {
    error = "M2 supports 64x64 or 128x128 superblocks";
    return false;
  }
  if (config.max_frame_width != static_cast<std::uint16_t>(plan.width) ||
      config.max_frame_height != static_cast<std::uint16_t>(plan.height)) {
    error = "frame plan size does not match the sequence/frame header config";
    return false;
  }

  Writer writer(plan, blocks, levels, config);
  if (!writer.run(tile_bytes, error)) return false;
  tile_payload = writer.payload();
  return true;
}

int y_mode_joint(int av2_mode, int angle_delta) {
  return y_mode_code(av2_mode, angle_delta).joint;
}

void build_scan(int w, int h, std::span<std::uint16_t> scan) {
  int x = 0;
  int y = 0;
  for (int c = 0; c < w * h; ++c) {
    scan[static_cast<std::size_t>(c)] =
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

}  // namespace av2enc::m2::syntax
