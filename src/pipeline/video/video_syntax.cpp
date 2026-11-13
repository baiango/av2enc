// AV2 M2 P-frame tile syntax writer.  See video_syntax.h for the contract.
//
// The partition, intra-mode and residual code is the M1 still-picture writer
// (byte-exact against dav2d for the M1 gate).  The inter block layer is new:
// is_inter/skip flags, single_mode, DRL index and shell-coded MVD, with all
// §8.3.2 contexts re-derived from the coded stream.
#include "pipeline/video/video_syntax.h"

#include "codec/inter/mv_precision/mv_precision.h"
#include "codec/inter/drl_reorder/video_stack_adapter.h"
#include "codec/filters/stream/filters_stream.h"
#include "codec/filters/pc_wiener/pc_wiener_stream.h"
#if defined(AV2ENC_VIDEO_DELTAQ)
#include "codec/entropy/deltaq/video/video_deltaq.h"
#endif

#include "pipeline/still/basic_cdf_tables.h"
#include "pipeline/still/basic_tables.h"
#include "pipeline/still/still_cdf_tables.h"
#include "pipeline/still/still_tables.h"
#include "codec/entropy/qctx/qctx.h"
#include "codec/entropy/lossless/lossless_cdf_tables.h"
#include "codec/intra/cfl/mhccp/mhccp.h"
#include "codec/intra/cfl/mhccp/mhccp_syntax.h"
#include "codec/intra/mrl/stream/mrl_interaction.h"
#include "codec/intra/mrl/stream/mrl_syntax.h"
#include "codec/transform/secondary/tx_secondary.h"
#include "pipeline/video/video_cdf_tables.h"
#include "codec/inter/multiref/gate/compound_cdf_tables.h"
#include "codec/inter/multiref/compound_predict.h"
#include "codec/partition/tree/partition.h"
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
#include "codec/partition/ext/partition_ext.h"
#include "codec/partition/stream/partition_stream.h"
#endif
#include "codec/partition/sdp/sdp.h"
#include "codec/transform/inter_residual/inter_tx_type.h"
#include "codec/transform/tcq/cctx.h"
#include "codec/transform/tcq/tcq.h"
#include "codec/transform/tcq/tcq_cdf_tables.h"
#include "codec/transform/tx_partition/tx_partition.h"
#include "codec/transform/tx_type/tx_type.h"
#include "codec/transform/variants/tx_variants.h"
#include "codec/entropy/av2/av2_entropy.h"
#include "codec/inter/mc/mc.h"
#include "codec/inter/merge/merge.h"
#include "codec/inter/morph/morph_common.h"
#include "codec/inter/advanced/video/advanced_syntax.h"
#include "codec/inter/morph/morph_syntax.h"
#include "codec/intra/ibc/core/ibc.h"
#include "codec/intra/ibc/stream/ibc_stack.h"
#include "codec/intra/ibc/stream/ibc_syntax.h"
#include "codec/intra/ibc/stream/still_ibc_stream.h"
#include "codec/intra/palette/core/palette.h"
#include "codec/intra/palette/stream/palette_stream.h"
#include "codec/inter/temporal/opfl.h"
#include "codec/inter/temporal/refinemv.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace av2enc::video::syntax {
namespace {
namespace sdpp = av2enc::sdp;

namespace drl = inter::drl;
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
namespace pe = partition_ext;
namespace ps = still::partition_stream;
#endif
namespace ibcs = still::ibc_stream;
namespace morph = inter::morph;
namespace adv = video::adv;
namespace pals = still::palette_stream;
namespace mrl_stream = still::mrl_stream;
namespace txsec = ::av2enc::txsec;

using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionType;
// The M1 table headers are generated and live in av2enc::still. They are the
// same §9.2/§9.3 defaults the M1 syntax writer uses, so share them rather than
// duplicating 180 CDF arrays.
using namespace av2enc::still;

// AV2 RefFrame labels (§6.19.7.9): LAST_FRAME = 0, INTRA_FRAME = 8.  The
// merge layer's MI grid uses these, not the MC-internal labels.
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

// §9.2 Fsc_Bsize_Groups[BLOCK_SIZES] (same table as the still/lossless writer).
constexpr int kFscBsizeGroups[29] = {
    0, 1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 3, 3, 4, 4, 6, 6, 4, 4, 6, 6};

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

struct CdfState {
  std::uint16_t do_split[2][64][3];
  std::uint16_t do_square_split[2][8][3];
  std::uint16_t rect_type[2][64][3];
  std::uint16_t y_mode_set[5];
  std::uint16_t y_mode_index[3][9];
  std::uint16_t y_mode_offset[3][7];
  std::uint16_t uv_mode[2][9];
  std::uint16_t is_cfl[3][3];
  std::uint16_t cfl_index[3];
  std::uint16_t cfl_sign[9];
  std::uint16_t cfl_alpha[6][9];
  // §5.20.7.32 cfl_mhccp / cfl_mh_dir CDFs (MHCCP).
  cfl::mhccp::CflCdfs mh_cdfs;
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
  // §9.3 Default_Cctx_Type_Cdf (§5.20.7.27 cctx_type, enable_cctx only).
  std::uint16_t cctx_type[8];
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
  // §5.20.5.5 mrl_index / mrl_sec_index (--video-mrl).
  still::mrl_stream::MrlCdfs mrl;
  // §5.20.8.2 IST symbols (--video-ist).
  std::uint16_t sec_tx_type[2][5][5];
  std::uint16_t most_probable_stx_set[8];
  std::uint16_t most_probable_stx_set_adst[5];

  // Inter block layer (§8.3.2 CDF selection).
  std::uint16_t single_mode[5][4];
  // §5.20.7.6 BAWP symbols (--video-bawp).
  std::uint16_t use_bawp[3];
  std::uint16_t use_bawp_chroma[3];
  std::uint16_t explicit_bawp[3][3];
  std::uint16_t explicit_bawp_scale[3];
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
  // TIP tile-block layer (§5.20.7.10, §9.3).  Selected only when a block
  // signals RefFrame[0] == TIP_FRAME (TIP_FRAME_AS_REF); the AS_OUTPUT frames
  // of --video-tip code no tile, but the state is part of the shared CDF image.
  std::uint16_t tip_mode[3][3];
  std::uint16_t tip_drl_mode[3][3];
  std::uint16_t tip_pred_mode[3];

  // Compound block layer (§5.20.7.10 / §5.20.7.6).
  std::uint16_t comp_mode[5][3];
  std::uint16_t is_joint[2][3];
  std::uint16_t compound_mode_non_joint[5][6];
  // §5.20.7.6 use_optflow (REFINE_SWITCHABLE only; ctx = YMode != NEAR_NEARMV).
  std::uint16_t use_optflow[2][3];
  // §5.20.7.16 masked compound (enable_masked_compound = 1).
  std::uint16_t comp_group_idx[12][3];
  std::uint16_t compound_type[3];
  std::uint16_t wedge_quad[5];
  std::uint16_t wedge_angle[4][6];
  std::uint16_t wedge_dist1[5];
  std::uint16_t wedge_dist2[4];

  // Inter residual luma transform-type symbols (§5.20.8.2).
  std::uint16_t inter_tx_type_set1[3][2][3];
  std::uint16_t inter_tx_type_set2[3][3];
  std::uint16_t inter_tx_type_set3[3][4][3];
  std::uint16_t inter_tx_type_long[3][4][5];
  std::uint16_t inter_tx_type_index_set1[3][9];
  std::uint16_t inter_tx_type_offset_set1[3][9];
  std::uint16_t inter_tx_type_index_set2[3][9];
  std::uint16_t inter_tx_type_offset_set2[3][5];

  // §5.20.6.3 TX-partition CDFs (opt-in TX_MODE_SELECT only).  Same default
  // tables the decoder installs; `init_cdfs()` resets them per tile.
  txpartition::TxPartitionCdfs tx_partition;

  // Precision-aware shell MVD tables (1/4 and 1/2 pel frame precision).
  inter::mvprec::MvPrecisionCdfState mv_precision;

  // §5.20.5.5 fsc_mode (enable_idtx_intra/--video-idtx).  Default_Fsc_Mode_Cdf
  // [4][6][3]; the product codes 0 with context [0][group].
  std::uint16_t fsc_mode[4][6][3];
};

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

// `qctx_slice` is §8.3.1 init_coeff_cdfs()'s COEFF_CDF_Q_CTXS slice, selected
// from the raw frame base_q_idx (0..3).  Slice 0 is byte-identical to the
// historical pinned tables.
void init_cdfs(CdfState& c, int qctx_slice,
               bool reduced_tx_part_set = false) {
  const qctx::CoeffCdfSlice& q = qctx::tables_for_slice(qctx_slice);
  copy_array(c.do_split, av2enc::basic::kDoSplitCdf);
  copy_array(c.do_square_split, av2enc::basic::kDoSquareSplitCdf);
  copy_array(c.rect_type, av2enc::basic::kRectTypeCdf);
  copy_array(c.y_mode_set, kYModeSetCdf);
  copy_array(c.y_mode_index, kYModeIndexCdf);
  copy_array(c.y_mode_offset, kYModeOffsetCdf);
  copy_array(c.uv_mode, kUvModeCflNotAllowedCdf);
  copy_array(c.is_cfl, kIsCflCdf);
  copy_array(c.cfl_index, kCflIndexCdf);
  copy_array(c.cfl_sign, kCflSignCdf);
  copy_array(c.cfl_alpha, kCflAlphaCdf);
  copy_array(c.fsc_mode, av2enc::lossless::kFscModeCdf);  // §5.20.5.5
  c.mh_cdfs.init();
  copy_array(c.txb_skip, *q.txb_skip);
  copy_array(c.v_txb_skip, *q.v_txb_skip);
  copy_array(c.eob_pt_1024, *q.eob_pt_1024);
  copy_array(c.eob_pt_512, *q.eob_pt_512);
  copy_array(c.eob_pt_256, *q.eob_pt_256);
  copy_array(c.eob_pt_128, *q.eob_pt_128);
  copy_array(c.eob_pt_64, *q.eob_pt_64);
  copy_array(c.eob_pt_32, *q.eob_pt_32);
  copy_array(c.eob_pt_16, *q.eob_pt_16);
  copy_array(c.eob_extra, *q.eob_extra);
  copy_array(c.coeff_base, *q.coeff_base);
  copy_array(c.coeff_base_lf, *q.coeff_base_lf);
  copy_array(c.cctx_type, av2enc::tcq::cdf::kCctxTypeCdf);
  copy_array(c.coeff_base_eob, *q.coeff_base_eob);
  copy_array(c.coeff_base_lf_eob, *q.coeff_base_lf_eob);
  copy_array(c.coeff_br, *q.coeff_br);
  copy_array(c.coeff_br_lf, *q.coeff_br_lf);
  copy_array(c.coeff_base_uv, *q.coeff_base_uv);
  copy_array(c.coeff_base_lf_uv, *q.coeff_base_lf_uv);
  copy_array(c.coeff_base_eob_uv, *q.coeff_base_eob_uv);
  copy_array(c.coeff_base_lf_eob_uv, *q.coeff_base_lf_eob_uv);
  copy_array(c.coeff_br_uv, *q.coeff_br_uv);
  copy_array(c.dc_sign, *q.dc_sign);
  copy_array(c.intra_tx_type_set1, av2enc::basic::kIntraTxTypeSet1Cdf);
  copy_array(c.intra_tx_type_long, kIntraTxTypeLongCdf);
  copy_array(c.is_long_side_dct, kIsLongSideDctCdf);
  mrl_stream::init_mrl_cdfs(c.mrl);
  for (int is_inter = 0; is_inter < 2; ++is_inter) {
    for (int sqr = 0; sqr < 5; ++sqr) {
      copy_array(c.sec_tx_type[is_inter][sqr],
                 txsec::kSecTxTypeCdf[is_inter][sqr]);
    }
  }
  copy_array(c.most_probable_stx_set, txsec::kMostProbableStxSetCdf);
  copy_array(c.most_probable_stx_set_adst, txsec::kMostProbableStxSetAdstCdf);
  copy_array(c.single_mode, kSingleModeCdf);
  copy_array(c.use_bawp, kUseBawpCdf);
  copy_array(c.use_bawp_chroma, kUseBawpChromaCdf);
  copy_array(c.explicit_bawp, kExplicitBawpCdf);
  copy_array(c.explicit_bawp_scale, kExplicitBawpScaleCdf);
  copy_array(c.skip, kSkipCdf);
  copy_array(c.skip_mode, kSkipModeCdf);
  copy_array(c.is_inter, kIsInterCdf);
  copy_array(c.single_ref, kSingleRefCdf);
  copy_array(c.drl_mode, kDrlModeCdf);
  copy_array(c.comp_mode, multiref::gate::kCompModeCdf);
  copy_array(c.is_joint, multiref::gate::kIsJointCdf);
  copy_array(c.compound_mode_non_joint,
             multiref::gate::kCompoundModeNonJointCdf);
  // §9.3 TileUseOptflowCdf[2] defaults (not generated into
  // video_cdf_tables.h; the compound component pins the same values).
  static constexpr std::uint16_t kUseOptflowCdf[2][3] = {
      11582, 31, 0, 24076, 26, 0,
  };
  copy_array(c.use_optflow, kUseOptflowCdf);
  copy_array(c.comp_group_idx, multiref::gate::kCompGroupIdxCdf);
  copy_array(c.compound_type, multiref::gate::kCompoundTypeCdf);
  copy_array(c.wedge_quad, multiref::gate::kWedgeQuadCdf);
  copy_array(c.wedge_angle, multiref::gate::kWedgeAngleCdf);
  copy_array(c.wedge_dist1, multiref::gate::kWedgeDist1Cdf);
  copy_array(c.wedge_dist2, multiref::gate::kWedgeDist2Cdf);
  copy_array(c.shell_set, kJointShellSetCdf);
  copy_array(c.shell6_class0, kJointShell6Class0Cdf);
  copy_array(c.shell6_class1, kJointShell6Class1Cdf);
  copy_array(c.shell_last_two, kJointShellLastTwoClassesCdf);
  copy_array(c.offset_low, kShellOffsetLowClassCdf);
  copy_array(c.offset_class2, kShellOffsetClass2Cdf);
  copy_array(c.offset_other, kShellOffsetOtherClassCdf);
  copy_array(c.tip_mode, kTipModeCdf);
  copy_array(c.tip_drl_mode, kTipDrlModeCdf);
  copy_array(c.tip_pred_mode, kTipPredModeCdf);
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
  c.tx_partition.reset(reduced_tx_part_set);
  inter::mvprec::init_default_mv_precision_cdfs(&c.mv_precision);
}

// ---------------------------------------------------------------------------
// Packed CDF image (§8.2 tile state without context state)
// ---------------------------------------------------------------------------
//
// CdfState here is trivially copyable, but the packed image keeps the seed /
// export / bank-table surface identical to the M1 still writer.  One descriptor
// list drives export, import and the avg_cdf::CdfBank table layout.
struct CdfArrayDesc {
  std::size_t struct_offset = 0;
  std::size_t packed_offset = 0;
  std::size_t words = 0;
  std::uint16_t symbols = 0;
};

const std::vector<CdfArrayDesc>& cdf_arrays() {
  static const std::vector<CdfArrayDesc> arrays = [] {
    const CdfState probe{};
    std::vector<CdfArrayDesc> out;
    std::size_t packed = 0;
    const auto* base = reinterpret_cast<const std::uint8_t*>(&probe);
    const auto add = [&](const void* member, std::size_t bytes,
                         std::uint16_t symbols) {
      CdfArrayDesc desc;
      desc.struct_offset = static_cast<std::size_t>(
          reinterpret_cast<const std::uint8_t*>(member) - base);
      desc.packed_offset = packed;
      desc.words = bytes / sizeof(std::uint16_t);
      desc.symbols = symbols;
      packed += desc.words;
      out.push_back(desc);
    };
#define AV2ENC_ADD_CDF(field, symbols) \
  add(probe.field, sizeof(probe.field), symbols)
    AV2ENC_ADD_CDF(do_split, 2);
    AV2ENC_ADD_CDF(do_square_split, 2);
    AV2ENC_ADD_CDF(rect_type, 2);
    AV2ENC_ADD_CDF(y_mode_set, 4);
    AV2ENC_ADD_CDF(y_mode_index, 8);
    AV2ENC_ADD_CDF(y_mode_offset, 6);
    AV2ENC_ADD_CDF(uv_mode, 8);
    AV2ENC_ADD_CDF(is_cfl, 2);
    AV2ENC_ADD_CDF(cfl_index, 2);
    AV2ENC_ADD_CDF(cfl_sign, 8);
    AV2ENC_ADD_CDF(cfl_alpha, 8);
    AV2ENC_ADD_CDF(fsc_mode, 2);  // §5.20.5.5 (enable_idtx_intra)
    AV2ENC_ADD_CDF(txb_skip, 2);
    AV2ENC_ADD_CDF(v_txb_skip, 2);
    AV2ENC_ADD_CDF(eob_pt_1024, 8);
    AV2ENC_ADD_CDF(eob_pt_512, 8);
    AV2ENC_ADD_CDF(eob_pt_256, 8);
    AV2ENC_ADD_CDF(eob_pt_128, 8);
    AV2ENC_ADD_CDF(eob_pt_64, 7);
    AV2ENC_ADD_CDF(eob_pt_32, 6);
    AV2ENC_ADD_CDF(eob_pt_16, 5);
    AV2ENC_ADD_CDF(eob_extra, 2);
    AV2ENC_ADD_CDF(coeff_base, 4);
    AV2ENC_ADD_CDF(coeff_base_lf, 6);
    AV2ENC_ADD_CDF(cctx_type, 7);
    AV2ENC_ADD_CDF(coeff_base_eob, 3);
    AV2ENC_ADD_CDF(coeff_base_lf_eob, 5);
    AV2ENC_ADD_CDF(coeff_br, 4);
    AV2ENC_ADD_CDF(coeff_br_lf, 4);
    AV2ENC_ADD_CDF(coeff_base_uv, 4);
    AV2ENC_ADD_CDF(coeff_base_lf_uv, 6);
    AV2ENC_ADD_CDF(coeff_base_eob_uv, 3);
    AV2ENC_ADD_CDF(coeff_base_lf_eob_uv, 5);
    AV2ENC_ADD_CDF(coeff_br_uv, 4);
    AV2ENC_ADD_CDF(dc_sign, 2);
    AV2ENC_ADD_CDF(intra_tx_type_set1, 7);
    AV2ENC_ADD_CDF(intra_tx_type_long, 4);
    AV2ENC_ADD_CDF(is_long_side_dct, 2);
    AV2ENC_ADD_CDF(sec_tx_type, 4);
    AV2ENC_ADD_CDF(most_probable_stx_set, 7);
    AV2ENC_ADD_CDF(most_probable_stx_set_adst, 4);
    add(probe.mrl.index, sizeof(probe.mrl.index), 4);
    add(probe.mrl.sec, sizeof(probe.mrl.sec), 2);
    AV2ENC_ADD_CDF(single_mode, 3);
    AV2ENC_ADD_CDF(use_bawp, 2);
    AV2ENC_ADD_CDF(use_bawp_chroma, 2);
    AV2ENC_ADD_CDF(explicit_bawp, 2);
    AV2ENC_ADD_CDF(explicit_bawp_scale, 2);
    AV2ENC_ADD_CDF(skip, 2);
    AV2ENC_ADD_CDF(skip_mode, 2);
    AV2ENC_ADD_CDF(is_inter, 2);
    AV2ENC_ADD_CDF(single_ref, 2);
    AV2ENC_ADD_CDF(drl_mode, 2);
    AV2ENC_ADD_CDF(shell_set, 2);
    AV2ENC_ADD_CDF(shell6_class0, 8);
    AV2ENC_ADD_CDF(shell6_class1, 8);
    AV2ENC_ADD_CDF(shell_last_two, 2);
    AV2ENC_ADD_CDF(offset_low, 2);
    AV2ENC_ADD_CDF(offset_class2, 2);
    AV2ENC_ADD_CDF(offset_other, 2);
    AV2ENC_ADD_CDF(col_mv_greater, 2);
    AV2ENC_ADD_CDF(col_mv_index, 2);
    AV2ENC_ADD_CDF(inter_tx_type_set1, 2);
    AV2ENC_ADD_CDF(inter_tx_type_set2, 2);
    AV2ENC_ADD_CDF(inter_tx_type_set3, 2);
    AV2ENC_ADD_CDF(inter_tx_type_long, 4);
    AV2ENC_ADD_CDF(inter_tx_type_index_set1, 8);
    AV2ENC_ADD_CDF(inter_tx_type_offset_set1, 8);
    AV2ENC_ADD_CDF(inter_tx_type_index_set2, 8);
    AV2ENC_ADD_CDF(inter_tx_type_offset_set2, 4);
    add(probe.mv_precision.use_amvd, sizeof(probe.mv_precision.use_amvd), 2);
    add(probe.mv_precision.mv_joint, sizeof(probe.mv_precision.mv_joint), 4);
    add(probe.mv_precision.amvd_index, sizeof(probe.mv_precision.amvd_index),
        8);
    add(probe.mv_precision.jmvd_scale_mode,
        sizeof(probe.mv_precision.jmvd_scale_mode), 5);
    add(probe.mv_precision.jmvd_adaptive_scale_mode,
        sizeof(probe.mv_precision.jmvd_adaptive_scale_mode), 3);
    add(probe.mv_precision.use_most_probable_precision,
        sizeof(probe.mv_precision.use_most_probable_precision), 2);
    add(probe.mv_precision.pb_mv_precision,
        sizeof(probe.mv_precision.pb_mv_precision), 3);
    add(probe.mv_precision.interp_filter,
        sizeof(probe.mv_precision.interp_filter), 3);
    add(probe.mv_precision.shell_set, sizeof(probe.mv_precision.shell_set), 2);
    add(probe.mv_precision.shell_last_two,
        sizeof(probe.mv_precision.shell_last_two), 2);
    add(probe.mv_precision.shell_offset_low,
        sizeof(probe.mv_precision.shell_offset_low), 2);
    add(probe.mv_precision.shell_offset_class2,
        sizeof(probe.mv_precision.shell_offset_class2), 2);
    add(probe.mv_precision.shell_offset_other,
        sizeof(probe.mv_precision.shell_offset_other), 2);
    add(probe.mv_precision.col_mv_greater,
        sizeof(probe.mv_precision.col_mv_greater), 2);
    add(probe.mv_precision.col_mv_index,
        sizeof(probe.mv_precision.col_mv_index), 2);
    add(probe.mv_precision.shell_class,
        sizeof(probe.mv_precision.shell_class), 8);
    // §5.20.6.3 TX-partition CDFs.  They adapt per tile and must ride the
    // §8.2.4 average-CDF bank like every other tile CDF; omitting them made
    // --video-tx-search + --avg-cdf/--adaptive-cdf desynchronise the second
    // tile/frame.
    AV2ENC_ADD_CDF(tx_partition.tx_do_partition, 2);
    AV2ENC_ADD_CDF(tx_partition.tx_partition_type, 7);
    AV2ENC_ADD_CDF(tx_partition.tx_partition_type_reduced, 7);
    AV2ENC_ADD_CDF(tx_partition.tx_2or3_partition_type, 2);
#undef AV2ENC_ADD_CDF
    return out;
  }();
  return arrays;
}

std::size_t cdf_image_words_impl() {
  const std::vector<CdfArrayDesc>& arrays = cdf_arrays();
  if (arrays.empty()) return 0;
  return arrays.back().packed_offset + arrays.back().words;
}

void export_cdf_image(const CdfState& state, std::uint16_t* out) {
  const auto* base = reinterpret_cast<const std::uint8_t*>(&state);
  for (const CdfArrayDesc& desc : cdf_arrays()) {
    std::memcpy(out + desc.packed_offset, base + desc.struct_offset,
                desc.words * sizeof(std::uint16_t));
  }
}

void import_cdf_image(CdfState& state, const std::uint16_t* in) {
  auto* base = reinterpret_cast<std::uint8_t*>(&state);
  for (const CdfArrayDesc& desc : cdf_arrays()) {
    std::memcpy(base + desc.struct_offset, in + desc.packed_offset,
                desc.words * sizeof(std::uint16_t));
  }
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
         std::span<const std::int32_t> levels, const VideoConfig& config,
         std::span<const std::uint16_t> cdf_seed,
         std::vector<std::uint16_t>* cdf_final,
         const post::VideoFilterPlan* filters = nullptr)
      : plan_(plan), blocks_(blocks), levels_(levels), config_(config),
        cdf_seed_(cdf_seed), cdf_final_(cdf_final), filters_(filters) {
    geometry_.width = plan.width;
    geometry_.height = plan.height;
    geometry_.sb_size = plan.sb_size;
    // The M2 8-bit 4:2:0 default keeps the historical geometry.  The format
    // path (10-bit / 4:2:2 / 4:4:4 / 4:0:0) carries its chroma layout in the
    // sequence header config.
    chroma_ = config_.chroma;
    depth_ = config_.depth;
    has_chroma_ = fmt::has_chroma(chroma_);
    sub_x_ = fmt::subsampling_x(chroma_);
    sub_y_ = fmt::subsampling_y(chroma_);
    geometry_.has_chroma = has_chroma_;
    geometry_.chroma_subsampling_x = sub_x_ != 0;
    geometry_.chroma_subsampling_y = sub_y_ != 0;
    mi_cols_ = partition::mi_cols(geometry_);
    mi_rows_ = partition::mi_rows(geometry_);
    frame_mi_cols_ = plan_.frame_mi_cols > 0 ? plan_.frame_mi_cols : mi_cols_;
    frame_mi_rows_ = plan_.frame_mi_rows > 0 ? plan_.frame_mi_rows : mi_rows_;
    tile_mi_row_ = plan_.tile_mi_row;
    tile_mi_col_ = plan_.tile_mi_col;

    init_cdfs(cdfs_,
              qctx::slice_for_base_q_idx(static_cast<int>(config_.base_q_idx)),
              plan_.reduced_tx_part_set);
    mrl_context_.reset(mi_rows_, mi_cols_, partition::sb_size_mi(geometry_));
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
    ps::init_tile_partition_cdfs(ext_cdfs_);
    tools_.enable_ext_partitions = config_.enable_ext_partitions;
    tools_.enable_uneven_4way_partitions =
        config_.enable_uneven_4way_partitions;
#endif
    if (!cdf_seed_.empty()) {
      import_cdf_image(cdfs_, cdf_seed_.data());
    }
    tile_buffer_.resize(kTileCapacityBytes);
    enc_.init_symbol(tile_buffer_.data(), tile_buffer_.size());
    enc_.disable_cdf_update = config_.disable_cdf_update ? 1 : 0;

    above_mi_.assign(static_cast<std::size_t>(mi_cols_) + 16, 6);
    left_mi_.assign(static_cast<std::size_t>(mi_rows_) + 16, 6);
    intra_joint_.assign(
        static_cast<std::size_t>(mi_rows_) * static_cast<std::size_t>(mi_cols_),
        0);
    uvcfl_.assign(static_cast<std::size_t>(chroma_mi_rows()) *
                      static_cast<std::size_t>(chroma_mi_cols()),
                  0);
    const std::size_t mi_cells = static_cast<std::size_t>(mi_rows_) *
                                 static_cast<std::size_t>(mi_cols_);
    is_inter_grid_.assign(mi_cells, 0);
    skip_grid_.assign(mi_cells, 0);
    ymode_grid_.assign(mi_cells, 0);
    comp_group_grid_.assign(mi_cells, 0);
    mv_grid_.assign(mi_cells, inter::mc::MiInfo{});
    if (plan_.allow_intrabc) {
      ibcs::init_ibc_cdfs(ibc_cdfs_);
      ibc_grid_.assign(mi_cells, 0);
      ibc_skip_grid_.assign(mi_cells, 0);
      ibc_neighbours_.reset(mi_cols_, mi_rows_);
      ibc_buffer_.reset(mi_cols_, mi_rows_);
      ibc_config_ = ibc::IbcConfig{};
      ibc_config_.sb_size = plan_.sb_size;
      ibc_config_.allow_global = true;
      ibc_config_.allow_local = false;
      ibc_config_.max_bvp_drl_bits_minus_1 = 0;
      ibc_content_.force_integer_mv = plan_.force_integer_mv;
      ibc_content_.max_bvp_drl_bits_minus_1 = 0;
    }
    if (plan_.morph_pred_present) {
      morph::init_morph_cdfs(morph_cdfs_);
      morph_grid_.assign(mi_cells, 0);
    }
    if (plan_.enable_interintra) {
      adv_cdfs_.reset();
    }
    if (plan_.allow_screen_content_tools) {
      // §5.20.5.5 palette_mode_info() is parsed for every eligible DC block
      // when screen-content tools are enabled, even if this encoder never
      // chooses a palette.  The adapter writes has_palette_y = 0 in that case.
      pals::ScreenContentFlags flags;
      flags.single_picture_header = false;
      flags.allow_screen_content_tools = true;
      flags.allow_palette = true;
      flags.force_integer_mv = plan_.force_integer_mv;
      flags.allow_intrabc = plan_.allow_intrabc;
      flags.allow_global_intrabc = plan_.allow_intrabc;
      flags.allow_local_intrabc = false;
      palette_writer_.emplace(flags);
      if (!palette_writer_->reset(mi_cols_, mi_rows_, error_)) {
        palette_reset_error_ = error_;
      }
      palette::init_palette_cdfs(palette_cdfs_);
    }
    for (int plane = 0; plane < kPlaneCount; ++plane) {
      const int n = (plane == 0 ? mi_cols_ : chroma_mi_cols()) + 16;
      const int m = (plane == 0 ? mi_rows_ : chroma_mi_rows()) + 16;
      above_level_[plane].assign(static_cast<std::size_t>(n), 0);
      left_level_[plane].assign(static_cast<std::size_t>(m), 0);
      above_dc_[plane].assign(static_cast<std::size_t>(n), 0);
      left_dc_[plane].assign(static_cast<std::size_t>(m), 0);
    }
    clear_above_context();
#if defined(AV2ENC_VIDEO_DELTAQ)
    if (plan_.qp_enabled()) {
      video::deltaq::FrameQpPlan qp_plan;
      qp_plan.base_q_idx = static_cast<int>(config_.base_q_idx);
      qp_plan.segmentation = plan_.segmentation;
      qp_plan.delta_q = plan_.delta_q;
      qp_plan.enable_ext_seg = false;
      qp_plan.primary_ref_is_none = plan_.primary_ref_is_none;
      VideoFrameHeader frame_header;
      frame_header.keyframe = !plan_.inter_frame;
      std::string qp_error;
      if (!video::deltaq::begin_frame(dqp_, qp_plan, frame_header, mi_cols_,
                                      mi_rows_, {}, &qp_error)) {
        dqp_init_error_ = "delta-Q setup: " + qp_error;
      } else if (plan_.delta_q.present &&
                 static_cast<int>(plan_.delta_steps.size()) !=
                     plan_.superblock_count()) {
        dqp_init_error_ =
            "delta-Q plan does not cover every superblock of this tile";
      } else {
        dqp_active_ = true;
      }
    }
#endif
    trace_ = std::getenv("AV2ENC_VIDEO_TRACE") != nullptr;
  }

  bool run(std::size_t* tile_bytes, std::string& error) {
#if defined(AV2ENC_VIDEO_DELTAQ)
    if (!dqp_init_error_.empty()) {
      error = dqp_init_error_;
      return false;
    }
#endif
    if (blocks_.empty()) {
      error = "syntax stream is empty";
      return false;
    }
    const int sb_cols = plan_.sb_cols();
    const int sb_rows = plan_.sb_rows();
    const int sb_mi = partition::sb_size_mi(geometry_);
    const bool cdef_active = filters_ != nullptr && filters_->cdef.enabled;
    if (cdef_active &&
        !filters_stream::cdef_tile_init(
            &cdef_state_, mi_cols_, mi_rows_, filters_->cdef.num_strengths,
            /*sb_width_log2=*/plan_.sb_size == 64
                ? 4
                : (plan_.sb_size == 128 ? 5 : 6),
            &error)) {
      return false;
    }
    const bool lr_active = filters_ != nullptr && filters_->lr.enabled;
    if (lr_active && filters_->lr.pc_wiener) {
      // §5.20.10.4 with FrameRestorationType[0] = RESTORE_PC_WIENER: the luma
      // units code only use_pc_wiener; chroma Wiener NS keeps the shared
      // filters_stream writer (its plane-0 state is RESTORE_NONE so it codes
      // no luma bits).
      namespace pcw = ::av2enc::filters::pc_wiener_stream;
      pcw::LrParams pc_params;
      pc_params.frame_type[0] = pcw::kRestorePcWiener;
      pc_params.frame_type[1] = pcw::kRestoreNone;
      pc_params.frame_type[2] = pcw::kRestoreNone;
      pc_params.luma_shift = filters_->lr.plane[0].shift;
      pc_params.chroma_shift = filters_->lr.chroma_shift;
      pc_ready_ = pcw::InitTileState(&pc_state_, pc_params, mi_cols_, mi_rows_,
                                     /*sub_x=*/1, /*sub_y=*/1, &error);
      if (!pc_ready_) return false;
      pc_state_.unit_type[0] = filters_->lr.plane[0].unit_type;
      const bool chroma_lr = filters_->lr.plane[1].enabled ||
                             filters_->lr.plane[2].enabled;
      if (chroma_lr) {
        filters_stream::LrParams params;
        params.frame_type[0] = filters_stream::kRestoreNone;
        for (int plane = 1; plane < 3; ++plane) {
          params.frame_type[plane] = filters_->lr.plane[plane].enabled
                                         ? filters_stream::kRestoreWienerNs
                                         : filters_stream::kRestoreNone;
          params.frame_filters_on[plane] = false;
        }
        params.frame_filters_on[0] = false;
        params.num_filter_classes = 1;
        params.num_filter_classes_idx = 0;
        params.luma_shift = filters_->lr.luma_shift;
        params.chroma_shift = filters_->lr.chroma_shift;
        if (!filters_stream::lr_state_init(&lr_state_, params, mi_cols_,
                                           mi_rows_, 1, 1, plan_.sb_size,
                                           &error)) {
          return false;
        }
        for (int plane = 1; plane < 3; ++plane) {
          const post::VideoLrPlane& src = filters_->lr.plane[plane];
          if (!src.enabled) continue;
          if (static_cast<int>(lr_state_.unit_type[plane].size()) !=
                  src.unit_rows * src.unit_cols ||
              static_cast<int>(lr_state_.unit_coeffs[plane].size()) !=
                  src.unit_rows * src.unit_cols *
                      static_cast<int>(
                          ::av2enc::restoration::kWienerNsCoeffStride)) {
            error = "video LR: unit grid does not match the plan";
            return false;
          }
          lr_state_.unit_type[plane] = src.unit_type;
          lr_state_.unit_coeffs[plane] = src.unit_coeffs;
        }
        lr_chroma_ready_ = true;
      }
    } else if (lr_active) {
      filters_stream::LrParams params;
      for (int plane = 0; plane < 3; ++plane) {
        params.frame_type[plane] =
            filters_->lr.plane[plane].enabled ? filters_stream::kRestoreWienerNs
                                              : filters_stream::kRestoreNone;
        params.frame_filters_on[plane] = false;
      }
      params.num_filter_classes = 1;
      params.num_filter_classes_idx = 0;
      params.luma_shift = filters_->lr.luma_shift;
      params.chroma_shift = filters_->lr.chroma_shift;
      if (!filters_stream::lr_state_init(&lr_state_, params, mi_cols_,
                                         mi_rows_, 1, 1, plan_.sb_size,
                                         &error)) {
        return false;
      }
      // Seed the per-unit decision grids from the encoder plan: read_lr() must
      // signal the same LrType/coefficients that were applied to the pixels.
      for (int plane = 0; plane < 3; ++plane) {
        const post::VideoLrPlane& src = filters_->lr.plane[plane];
        if (!src.enabled) continue;
        if (static_cast<int>(lr_state_.unit_type[plane].size()) !=
                lr_state_.unit_rows[plane] * lr_state_.unit_cols[plane] ||
            static_cast<int>(lr_state_.unit_coeffs[plane].size()) !=
                lr_state_.unit_rows[plane] * lr_state_.unit_cols[plane] *
                    static_cast<int>(
                        ::av2enc::restoration::kWienerNsCoeffStride)) {
          error = "video LR: unit grid does not match the tile";
          return false;
        }
        // The plan is frame-level; copy the units that overlap this tile into
        // the tile-local state, translated by the tile's unit origin.
        const int sub_x = plane == 0 ? 0 : 1;
        const int sub_y = plane == 0 ? 0 : 1;
        const int unit_size = lr_state_.unit_size[plane];
        const int row_off =
            (tile_mi_row_ * 4 >> sub_y) / std::max(1, unit_size);
        const int col_off =
            (tile_mi_col_ * 4 >> sub_x) / std::max(1, unit_size);
        const int coeff_stride =
            static_cast<int>(::av2enc::restoration::kWienerNsCoeffStride);
        for (int r = 0; r < lr_state_.unit_rows[plane]; ++r) {
          const int fr = r + row_off;
          if (fr < 0 || fr >= src.unit_rows) continue;
          for (int c = 0; c < lr_state_.unit_cols[plane]; ++c) {
            const int fc = c + col_off;
            if (fc < 0 || fc >= src.unit_cols) continue;
            const std::size_t dst =
                static_cast<std::size_t>(r) *
                    static_cast<std::size_t>(lr_state_.unit_cols[plane]) +
                static_cast<std::size_t>(c);
            const std::size_t from =
                static_cast<std::size_t>(fr) *
                    static_cast<std::size_t>(src.unit_cols) +
                static_cast<std::size_t>(fc);
            lr_state_.unit_type[plane][dst] = src.unit_type[from];
            std::copy_n(src.unit_coeffs.begin() +
                            static_cast<std::ptrdiff_t>(from * coeff_stride),
                        coeff_stride,
                        lr_state_.unit_coeffs[plane].begin() +
                            static_cast<std::ptrdiff_t>(dst * coeff_stride));
          }
        }
      }
      lr_chroma_ready_ = true;
    }
    const bool ccso_active = filters_ != nullptr && filters_->ccso.enabled;
    if (ccso_active) {
      filters_stream::CcsoParams params;
      params.frame_flag = true;
      params.ccso_luma_size_log2 = filters_->ccso.ccso_luma_size_log2;
      for (int plane = 0; plane < 3; ++plane) {
        filters_stream::CcsoPlaneParams& dst = params.plane[plane];
        const post::VideoCcsoPlane& src = filters_->ccso.plane[plane];
        dst.enabled = src.enabled;
        dst.bo_only = src.bo_only;
        dst.scale_idx = src.scale_idx;
        dst.quant_idx = src.quant_idx;
        dst.ext_filter = src.ext_filter;
        dst.edge_clf = src.edge_clf;
        dst.max_band_log2 = src.max_band_log2;
        dst.offset_idx = src.offset_idx;
      }
      if (!filters_stream::ccso_tile_init(&ccso_state_, params, mi_cols_,
                                          mi_rows_, 1, 1, &error)) {
        return false;
      }
    }
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
        // `if (sb.mi_col == 0) clear_left_context()` in still_encoder.
        clear_left_context();
      }
#if defined(AV2ENC_VIDEO_DELTAQ)
      if (dqp_active_) {
        // §5.20.2.1: ReadDeltas = delta_q_present at every superblock start;
        // the segment-prediction left context restarts at a superblock row.
        video::deltaq::begin_superblock(dqp_, node.mi_col);
        sb_step_ = nullptr;
        if (plan_.delta_q.present) {
          if (sb_index < 0 ||
              static_cast<std::size_t>(sb_index) >= plan_.delta_steps.size()) {
            error = "delta-Q plan is missing superblock " +
                    std::to_string(sb_index);
            return false;
          }
          sb_step_ = &plan_.delta_steps[static_cast<std::size_t>(sb_index)];
        }
        sb_first_leaf_ = true;
      }
#endif
      if (cdef_active) {
        // §5.20.9.6 clear_cdef(): once per superblock, before read_partition.
        filters_stream::cdef_clear_superblock(&cdef_state_, node.mi_row,
                                              node.mi_col);
      }
      // §5.20.10.4 read_lr(): before read_partition(), once per superblock.
      if (!write_lr_superblock(node, error)) return false;
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
    if (cdf_final_ != nullptr) {
      cdf_final_->resize(cdf_image_words_impl());
      export_cdf_image(cdfs_, cdf_final_->data());
    }
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

  bool code_do_square_split(PartitionType partition, int r, int c,
                            BlockSize size) {
    const int bsw = kMiWidthLog2[static_cast<int>(size)];
    const int bsh = kMiHeightLog2[static_cast<int>(size)];
    int ctx = (left_mi(r) < bsh ? 2 : 0) +
              (above_mi(c) < bsw ? 1 : 0);
    if (size == BlockSize::k256x256) ctx += 4;
    return sym(&cdfs_.do_square_split[0][ctx][0], 2,
               partition == PartitionType::kSplit ? 1 : 0);
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

  // §5.20.3.2 partition syntax for one SDP node.  The producer stores the
  // reference planner's exact symbol sequence (implied / single-candidate
  // partitions code nothing), including the PlaneStart and context derived
  // from the luma/chroma MiSize grids, so the writer cannot disagree with the
  // solver about the tree.
  bool code_sdp_partition(const BlockSyntax& node, std::string& error) {
    for (int i = 0; i < static_cast<int>(node.sdp_symbol_count); ++i) {
      const BlockSyntax::SdpSymbol& symbol = node.sdp_symbols[i];
      switch (static_cast<sdpp::SymbolKind>(symbol.kind)) {
        case sdpp::SymbolKind::kDoSplit:
          if (!sym(&cdfs_.do_split[symbol.plane][symbol.context][0], 2,
                   symbol.value)) {
            return false;
          }
          break;
        case sdpp::SymbolKind::kRectType:
          if (!sym(&cdfs_.rect_type[symbol.plane][symbol.context][0], 2,
                   symbol.value)) {
            return false;
          }
          break;
        case sdpp::SymbolKind::kDoSquareSplit:
        case sdpp::SymbolKind::kDoExtPartition:
        case sdpp::SymbolKind::kDoUneven4wayPartition:
        case sdpp::SymbolKind::kUneven4wayType:
        case sdpp::SymbolKind::kRegionType:
          error = "SDP frame codes " +
                  std::string(sdpp::symbol_name(
                      static_cast<sdpp::SymbolKind>(symbol.kind))) +
                  " at a 64x64 superblock stream; only rect-only sub-trees "
                  "are wired";
          return false;
      }
    }
    return true;
  }

  // SDP writer (--video-sdp on the intra keyframe).  The 64x64 split node
  // codes no partition syntax and carries the luma root followed by the chroma
  // root.  Each sub-tree node codes its own PlaneStart-selected partition
  // symbols; leaf blocks gate the tree-type-dependent mode/residual syntax in
  // write_block().
  bool write_node_sdp(std::size_t index, std::size_t* next,
                      std::string& error) {
    const BlockSyntax& node = blocks_[index];
    const auto size = static_cast<BlockSize>(node.bsize);
    const auto partition = static_cast<PartitionType>(node.partition);
    const std::size_t subtree_end = index + 1 + node.subtree;
    if (subtree_end > blocks_.size()) {
      error = "node " + std::to_string(index) + " has a subtree past the end";
      return false;
    }
    if ((node.flags & kFlagSdpSplit) != 0) {
      if (node.subtree == 0) {
        error = "SDP split node " + std::to_string(index) +
                " has no luma/chroma children";
        return false;
      }
      std::size_t cursor = index + 1;
      while (cursor < subtree_end) {
        std::size_t after = 0;
        if (!write_node_sdp(cursor, &after, error)) return false;
        if (after > subtree_end) {
          error = "SDP node " + std::to_string(index) +
                  " has an inconsistent subtree count";
          return false;
        }
        cursor = after;
      }
      if (cursor != subtree_end) {
        error = "SDP node " + std::to_string(index) +
                " has an inconsistent subtree count";
        return false;
      }
      *next = subtree_end;
      return true;
    }
    if (!partition::is_valid_block_size(size)) {
      error = "node " + std::to_string(index) + " has an invalid block size";
      return false;
    }
    if (!inside_mi(node.mi_row, node.mi_col)) {
      error = "node " + std::to_string(index) + " is outside the frame";
      return false;
    }
    if (!code_sdp_partition(node, error)) return false;

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

    // Descend into the children (preorder).  The tree solver produced one
    // child per in-frame slot, so the subtree count is authoritative.
    std::size_t cursor = index + 1;
    while (cursor < subtree_end) {
      std::size_t after = 0;
      if (!write_node_sdp(cursor, &after, error)) return false;
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

  // Emit the syntax for the node at `index` (and, for non-leaves, its whole
  // subtree); `next` receives the index one past the subtree.
  bool write_node(std::size_t index, std::size_t* next, std::string& error) {
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
    if (config_.enable_ext_partitions) {
      return write_node_ext(index, next, error);
    }
#endif
    if (plan_.frame_sdp) {
      return write_node_sdp(index, next, error);
    }
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
      // §5.20.3.2: do_square_split follows do_split and precedes rect_type.
      // It is present only when PARTITION_SPLIT is allowed, which can happen
      // only at a 128x128 (or 256x256) node; the 64x64 path never reaches it.
      if (allowed(PartitionType::kSplit)) {
        if (!code_do_square_split(partition, r, c, size)) {
          error = error_;
          return false;
        }
      }
      const partition::RectTypeImplied implied =
          partition::rect_type_implied_by_bsize(size);
      bool need_rect = partition != PartitionType::kSplit && !implied.implied;
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

#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
  // ---------------------------------------------------------------------
  // Extended-partition writer (--video-ext-partitions).  The symbol sequence
  // comes from partition_stream/partition_ext so the implication rules for
  // do_ext_partition / do_uneven_4way_partition and the raw L(1) type bit are
  // the reference module's, not re-derived here.
  // ---------------------------------------------------------------------
  bool write_node_ext(std::size_t index, std::size_t* next,
                      std::string& error) {
    const BlockSyntax& node = blocks_[index];
    const auto size = static_cast<partition::BlockSize>(node.bsize);
    const auto partition = static_cast<pe::PartitionType>(node.partition);
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
    if (!inside_mi(r, c)) {
      error = "node " + std::to_string(index) + " is outside the frame";
      return false;
    }
    if (partition != pe::PartitionType::kNone && node.subtree == 0) {
      error = "node " + std::to_string(index) +
              " has no children for a split";
      return false;
    }

    ps::StillNode sn;
    sn.mi_row = node.mi_row;
    sn.mi_col = node.mi_col;
    sn.bsize = node.bsize;
    sn.partition = node.partition;
    sn.flags = node.flags;
    sn.subtree = node.subtree;
    sn.allowed_mask = node.allowed_mask;
    // The video leaf coder owns its chroma block; collect_node_ext() rejects a
    // tree that needs shared ChromaMi ownership.
    sn.has_chroma = true;
    sn.chroma_offset = false;
    sn.chroma = pe::ChromaBlock{r, c, size};

    const ps::NeighbourContext nc = ps::neighbours_from_arrays(
        std::span<const int>(left_mi_), std::span<const int>(above_mi_), r, c,
        size, inside_mi(r, c - 1), inside_mi(r - 1, c));
    const ps::NodePlan plan = ps::plan_node(geometry_, tools_, sn, nc);
    if (!plan.ok) {
      error = "node " + std::to_string(index) + ": " + plan.error;
      return false;
    }
    if (!ps::emit_symbols(plan.symbols, ext_cdfs_, enc_, nullptr, error)) {
      return false;
    }

    if (partition == pe::PartitionType::kNone) {
      if (node.subtree != 0) {
        error = "leaf node " + std::to_string(index) +
                " has a non-zero subtree";
        return false;
      }
      if (!write_block(node, error)) return false;
      *next = index + 1;
      return true;
    }

    pe::ChildSlot raw[4];
    const int raw_count = pe::enumerate_children(
        geometry_, r, c, size, partition, /*has_chroma=*/true,
        /*chroma_offset=*/false, pe::ChromaBlock{r, c, size}, raw);
    pe::ChildSlot slots[4];
    int slot_count = 0;
    for (int i = 0; i < raw_count; ++i) {
      if (pe::child_is_coded(geometry_, raw[i])) slots[slot_count++] = raw[i];
    }
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
                " does not match a partition child slot of node " +
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
      if (!write_node_ext(cursor, &after, error)) return false;
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
#endif  // AV2ENC_VIDEO_EXT_PARTITIONS

  // ---------------------------------------------------------------------
  // Mode info syntax
  // ---------------------------------------------------------------------
  int joint_context() const {
    if (std::getenv("AV2ENC_STILL_YMODE_CTX0") != nullptr) return 0;
    if (std::getenv("AV2ENC_STILL_YMODE_CTX2") != nullptr) return 2;
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
    const int cr = cur_chroma_r_;
    const int cc = cur_chroma_c_;
    const int cmi_cols = chroma_mi_cols();
    int ctx = 0;
    if (cr > 0 && cc < cmi_cols &&
        uvcfl_[static_cast<std::size_t>(cr - 1) *
                   static_cast<std::size_t>(cmi_cols) +
               static_cast<std::size_t>(cc)] != 0) {
      ++ctx;
    }
    if (cc > 0 && cr < chroma_mi_rows() &&
        uvcfl_[static_cast<std::size_t>(cr) *
                   static_cast<std::size_t>(cmi_cols) +
               static_cast<std::size_t>(cc - 1)] != 0) {
      ++ctx;
    }
    return ctx;
  }

  bool code_uv_mode(const BlockSyntax& node, int cw, int ch) {
    const bool is_cfl = (node.flags & kFlagIsCfl) != 0;
    const int uv_mode = node.uv_mode;
    const int y_mode = node.y_mode;
    if (is_cfl) {
      // is_cfl derives its context from neighbouring chroma CfL flags.
      if (!sym(&cdfs_.is_cfl[cfl_context()][0], 2, 1)) return false;
      if (config_.enable_mhccp) {
        // §5.20.7.32 with MHCCP: cfl_mhccp, cfl_index, cfl_mh_dir and the
        // explicit alpha symbols all live in the gated adapter.
        const bool mhccp_allowed = cfl::mhccp::allowed(
            cw, ch, config_.enable_mhccp, /*is_chroma_part=*/false,
            /*frame_is_intra=*/true, /*cfl_allowed_in_sdp=*/false,
            /*lossless=*/false);
        if (node.cfl_index == 2 && !mhccp_allowed) {
          error_ = "CFL_MULTI signalled for a block where MHCCP is not allowed";
          return false;
        }
        cfl::mhccp::CflSyntax syntax;
        syntax.cfl_mhccp = node.cfl_index == 2 ? 1 : 0;
        syntax.cfl_index = node.cfl_index;
        syntax.cfl_mh_dir = node.cfl_mh_dir;
        syntax.cfl_alpha_signs = node.cfl_alpha_signs;
        syntax.cfl_alpha_u = node.cfl_alpha_u;
        syntax.cfl_alpha_v = node.cfl_alpha_v;
        std::string alpha_error;
        // §8.3: TileCflMhDirCdf[Size_Group[MiSize]] uses the coding block.
        const int mh_size_group =
            node.bsize < 29 ? cfl::mhccp::kSizeGroup[node.bsize] : -1;
        if (mh_size_group < 0) {
          error_ = "invalid block size for the cfl_mh_dir context";
          return false;
        }
        if (!cfl::mhccp::write_cfl_alphas(
                enc_, cdfs_.mh_cdfs, syntax, config_.enable_cfl_intra,
                mhccp_allowed, mh_size_group, &alpha_error)) {
          error_ = "read_cfl_alphas: " + alpha_error;
          return false;
        }
        return true;
      }
      // CFL_DERIVED_ALPHA = 1 (the non-MHCCP path).
      return sym(&cdfs_.cfl_index[0], 2, node.cfl_index);
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
    if (const char* forced = std::getenv("AV2ENC_STILL_DCSIGN_CTX")) {
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

  // Luma inter transform-type syntax (§5.20.8.2).  §5.20.8.2 codes a TX_TYPE
  // for every inter TX block whose set is not TX_SET_DCTONLY, including
  // eob == 1 (the eob <= 1 shortcut is intra-only).  `tx_type` is a TX_TYPE
  // value from the TX search; with the search off it is always DCT_DCT and the
  // emitted symbols are the historical ones.
  bool write_inter_tx_type(int tx_size, int tx_type, int eob, int coded_w,
                           int coded_h) {
    const auto ts = static_cast<txv::TxSize>(tx_size);
    const residual::InterTxSet set =
        residual::inter_tx_set(ts, plan_.reduced_tx_set);
    if (set == residual::InterTxSet::kDctOnly) return true;
    residual::InterTxCandidate cand;
    if (!residual::inter_tx_symbol(ts, plan_.reduced_tx_set,
                                   /*enable_inter_ddt=*/false,
                                   static_cast<txv::TxType>(tx_type), &cand)) {
      error_ = "inter tx_type is not codable for this size";
      return false;
    }
    const int ctx = inter_tx_type_context(eob, coded_w, coded_h, true);
    const int sqr = kTxSizeSqr[tx_size];
    switch (set) {
      case residual::InterTxSet::kWide64:
      case residual::InterTxSet::kHigh64:
        // WIDE_64/HIGH_64 imply is_long_side_dct = 1; only the type is coded.
        return sym(&cdfs_.inter_tx_type_long[ctx][sqr][0], 4, cand.symbol);
      case residual::InterTxSet::kWide32:
      case residual::InterTxSet::kHigh32:
        if (!sym(&cdfs_.is_long_side_dct[1][0], 2,
                 cand.is_long_side_dct ? 1 : 0)) {
          return false;
        }
        return sym(&cdfs_.inter_tx_type_long[ctx][sqr][0], 4, cand.symbol);
      case residual::InterTxSet::kInter1: {
        if (!sym(&cdfs_.inter_tx_type_set1[ctx][sqr][0], 2, cand.symbol)) {
          return false;
        }
        const int off_ctx =
            inter_tx_type_context(eob, coded_w, coded_h, false);
        if (cand.symbol == 0) {
          return sym(&cdfs_.inter_tx_type_index_set1[off_ctx][0], 8,
                     cand.offset_symbol);
        }
        return sym(&cdfs_.inter_tx_type_offset_set1[off_ctx][0], 8,
                   cand.offset_symbol);
      }
      case residual::InterTxSet::kInter2: {
        if (!sym(&cdfs_.inter_tx_type_set2[ctx][0], 2, cand.symbol)) {
          return false;
        }
        const int off_ctx =
            inter_tx_type_context(eob, coded_w, coded_h, false);
        if (cand.symbol == 0) {
          return sym(&cdfs_.inter_tx_type_index_set2[off_ctx][0], 8,
                     cand.offset_symbol);
        }
        return sym(&cdfs_.inter_tx_type_offset_set2[off_ctx][0], 4,
                   cand.offset_symbol);
      }
      case residual::InterTxSet::kDctIdtx:
        return sym(&cdfs_.inter_tx_type_set3[ctx][sqr][0], 2, cand.symbol);
      case residual::InterTxSet::kDctIdtxIddct:
        error_ = "TX_SET_DCT_IDTX_IDDCT is not signalled by this writer";
        return false;
      case residual::InterTxSet::kDctOnly:
        return true;
    }
    error_ = "unknown inter TX set";
    return false;
  }

  // One transform block of a leaf.  TX_MODE_LARGEST fills it from the plane
  // fields; TX_MODE_SELECT passes the per-part tx size/type/EOB and the part's
  // 4x4 origin so the §8.3.2 contexts are derived from the real transform
  // block, exactly like the decoder.
  struct TxView {
    int tx_size = 0;
    int tx_type = 0;
    int eob = 0;
    int mi_row = 0;
    int mi_col = 0;
    // §5.20.8.2 IST symbols; chroma/inter leave them 0.
    int sec_tx_type = 0;
    int stx_set = 0;
  };

  // `level_base` is the index of this plane's first scan-order level.
  bool code_tx_view(const BlockSyntax& node, int plane, const TxView& view,
                    std::size_t level_base, int bsize_w, int bsize_h,
                    std::size_t levels_end, bool is_inter,
                    std::string& error) {
    const int tx_size = view.tx_size;
    if (tx_size >= 25) {
      error = "invalid tx_size " + std::to_string(tx_size) + " on plane " +
              std::to_string(plane);
      return false;
    }
    // §5.20.7.27/.30: the one-axis 1-D classes need the class-specific scan,
    // get_lf_limits() contexts and the dc_sign_horz_vert element.  This writer
    // (and the TX search's 1-D opt-in) does not signal them; pinned dav2d also
    // rejects any negative H_DCT/V_DCT coefficient under --strict 1.  Fail
    // loudly instead of coding a 2-D scan for a 1-D transform.
    if (view.tx_type >= 10 && view.tx_type <= 15) {
      error = "1-D TX types (V/H_DCT, V/H_ADST, V/H_FLIPADST) are not "
              "signalled: dc_sign_horz_vert (§5.20.7.27) needs the "
              "class-specific scan, and pinned dav2d rejects negative "
              "H_DCT/V_DCT coefficients under --strict 1";
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
    const int eob = view.eob;
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
    const int sub_x = plane_sub_x(plane);
    const int sub_y = plane_sub_y(plane);
    const int x4 = view.mi_col >> sub_x;
    const int y4 = view.mi_row >> sub_y;
    const int max_x4 = plane == 0 ? mi_cols_ : chroma_mi_cols();
    const int max_y4 = plane == 0 ? mi_rows_ : chroma_mi_rows();

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
    if (plane == 1 && plan_.enable_cctx) {
      // §5.20.7.27: cctx_type follows the U EOB and precedes the coefficients.
      // The decoder reads the symbol when (is_inter || eob != 1); it is absent
      // (inferred 0) for an intra block with U eob == 1.
      if (is_inter || eob != 1) {
        if (!sym(&cdfs_.cctx_type[0], av2enc::tcq::kCctxTypeCount,
                 node.cctx_type)) {
          return false;
        }
      } else if (node.cctx_type != 0) {
        error = "cctx_type must be 0 when the U plane has eob == 1";
        return false;
      }
    }
    if (plane == 0) {
      if (is_inter) {
        if (!write_inter_tx_type(tx_size, view.tx_type, eob, coded_w,
                                 coded_h)) {
          return false;
        }
      } else {
        switch (luma_tx_type_syntax(tx_size, eob)) {
        case TxTypeSyntax::kNone:
          break;
        case TxTypeSyntax::kIntraSet1:
        case TxTypeSyntax::kLong32:
        case TxTypeSyntax::kLong64: {
          // §5.20.7.29 / §5.20.8.2: invert the signalled set to the symbol for
          // this TX_TYPE.  DCT_DCT is symbol 0 in every table, so the
          // TX_MODE_LARGEST path emits the historical bits.
          txtype::TxTypeQuery query;
          query.coded_w = coded_w;
          query.coded_h = coded_h;
          const int mapped = mrl_stream::mapped_intra_mode_for_mrl(
              node.y_mode, node.y_angle_delta, node.mrl_index, coded_w,
              coded_h);
          query.mapped_mode = mapped < 0 ? 0 : mapped;
          query.reduced_tx_set = plan_.reduced_tx_set;
          query.dct_only = false;
          int symbol = -1;
          bool is_long_side_dct = false;
          if (!txtype::tx_type_symbol(
                  query, static_cast<txtype::TxType>(view.tx_type), &symbol,
                  &is_long_side_dct)) {
            error_ = "intra tx_type is not codable for this size/mode";
            return false;
          }
          if (luma_tx_type_syntax(tx_size, eob) == TxTypeSyntax::kIntraSet1) {
            if (!sym(&cdfs_.intra_tx_type_set1[kTxSizeSqr[tx_size]][0], 7,
                     symbol)) {
              return false;
            }
          } else if (luma_tx_type_syntax(tx_size, eob) ==
                     TxTypeSyntax::kLong32) {
            if (!sym(&cdfs_.is_long_side_dct[0][0], 2,
                     is_long_side_dct ? 1 : 0)) {
              return false;
            }
            if (!sym(&cdfs_.intra_tx_type_long[kTxSizeSqr[tx_size]][0], 4,
                     symbol)) {
              return false;
            }
          } else {
            if (!sym(&cdfs_.intra_tx_type_long[kTxSizeSqr[tx_size]][0], 4,
                     symbol)) {
              return false;
            }
          }
          break;
        }
        }
      }
    }

    if (plane == 0 && !is_inter && config_.enable_intra_ist) {
      // §5.20.8.2: sec_tx_type follows transform_type() and precedes the
      // coefficients.  Luma only; chroma never codes it.
      txsec::SecondaryTransformConfig stx_config;
      stx_config.enable_intra_ist = true;
      txsec::SecondarySyntax stx;
      stx.plane_tx_type = static_cast<txv::TxType>(view.tx_type);
      stx.coded_w = coded_w;
      stx.coded_h = coded_h;
      stx.y_mode = node.y_mode;
      stx.angle_delta = node.y_angle_delta;
      stx.mrl_index = node.mrl_index;
      stx.most_probable_stx_set = view.stx_set;
      stx.sec_tx_type = view.sec_tx_type;
      stx.is_inter = false;
      stx.lossless = false;
      stx.bit_depth = 8;
      if (txsec::secondary_syntax_applicable(0, eob, stx, stx_config)) {
        if (!sym(&cdfs_.sec_tx_type[0][kTxSizeSqr[tx_size]][0], 4,
                 view.sec_tx_type)) {
          return false;
        }
        if (view.sec_tx_type != 0) {
          if (!txsec::secondary_syntax_valid(0, eob, stx, stx_config)) {
            error_ = "sec_tx_type is not valid for this block";
            return false;
          }
          const int set_count = txsec::most_probable_stx_set_count(
              stx.plane_tx_type, coded_w, coded_h);
          std::uint16_t* set_cdf = set_count == 4
                                       ? cdfs_.most_probable_stx_set_adst
                                       : cdfs_.most_probable_stx_set;
          if (!sym(set_cdf, set_count, view.stx_set)) return false;
        }
      } else if (view.sec_tx_type != 0) {
        error_ = "sec_tx_type set on a block where IST is not applicable";
        return false;
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
    // §5.20.7.27 useTcq: luma-only, TX_CLASS_2D-only, lossy and non-FSC.  A
    // 1-D TX type (10..15) is rejected above, IDTX (9) is TX_CLASS_2D.
    const bool use_tcq = plan_.allow_tcq && plane == 0 && view.tx_type <= 9;
    int tcq_state = 0;
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
      // Under TCQ the writer levels are full levels `q`; the syntax codes
      // syntax_level_for(q, pre_max) and read_quant carries the escape
      // increment x = (q - l) / 2.
      const int syntax_level =
          use_tcq ? av2enc::tcq::syntax_level_for(abs_value, pre_max)
                  : abs_value;
      int base_level = 0;
      int br_level = 0;
      if (syntax_level <= base_levels) {
        base_level = syntax_level;
      } else {
        base_level = std::min(base_levels + 1, base_max);
        const int pre = std::min(syntax_level, pre_max);
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
        // §8.3.2: coeff_base / coeff_base_lf select their CDF with
        // (tcqState >> 1) & 1.  The state is 0 when TCQ is off.
        const int tcq_ctx = use_tcq ? ((tcq_state >> 1) & 1) : 0;
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
            if (!sym(&cdfs_.coeff_base_lf[tsc][lf_ctx][tcq_ctx][0], 6,
                     base_level)) {
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
            if (!sym(&cdfs_.coeff_base[tsc][hf_ctx][tcq_ctx][0], 4,
                     base_level)) {
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
      if (use_tcq) {
        // First pass: Tcq_Next_State[state][syntax_level & 1].
        tcq_state =
            av2enc::tcq::next_state(tcq_state, syntax_level & 1);
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
        if (use_tcq) {
          // §5.20.7.27 read_quant under TCQ escapes at `maxLevel - 1` and adds
          // `x << 1`, where x = (full_level - syntax_level) / 2.  `pre_level`
          // is the syntax level the first pass coded.
          if (pre_level >= max_level - 1) {
            if (!write_quant_escape((abs_value - pre_level) / 2,
                                    hr_level_avg)) {
              return false;
            }
          }
        } else if (abs_value >= max_level) {
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

  // Chroma coded plane extents in 4x4 units.  The 4:2:0 default is exactly
  // `mi_{cols,rows} / 2`; 4:2:2 halves only columns, 4:4:4 keeps both and
  // 4:0:0 has no chroma.
  int chroma_mi_cols() const {
    return has_chroma_ ? (mi_cols_ >> sub_x_) : 0;
  }
  int chroma_mi_rows() const {
    return has_chroma_ ? (mi_rows_ >> sub_y_) : 0;
  }
  int plane_sub_x(int plane) const { return plane == 0 ? 0 : sub_x_; }
  int plane_sub_y(int plane) const { return plane == 0 ? 0 : sub_y_; }

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

  // §5.20.7.6 use_bawp + §7.13.3.25 sub-flags.  The encoder codes the fitted
  // form (explicit_bawp = 0); `use_bawp_chroma` is always 0 for this path.
  bool code_bawp(const BlockSyntax& node) {
    const int single_ctx =
        node.inter_mode == kInterNearMv
            ? 0
            : (node.inter_mode == kInterNewMv ? 2 : 0);
    if (!sym(&cdfs_.use_bawp[0], 2, node.use_bawp ? 1 : 0)) return false;
    if (!node.use_bawp) {
      if (node.explicit_bawp || node.explicit_bawp_scale ||
          node.use_bawp_chroma) {
        error_ = "BAWP sub-flags set while use_bawp is 0";
        return false;
      }
      return true;
    }
    if (!sym(&cdfs_.explicit_bawp[single_ctx][0], 2,
             node.explicit_bawp ? 1 : 0)) {
      return false;
    }
    if (node.explicit_bawp) {
      if (!sym(&cdfs_.explicit_bawp_scale[0], 2,
               node.explicit_bawp_scale ? 1 : 0)) {
        return false;
      }
    } else if (node.explicit_bawp_scale) {
      error_ = "explicit_bawp_scale set while explicit_bawp is 0";
      return false;
    }
    // HasChroma is always true in this 4:2:0 product path.
    if (!sym(&cdfs_.use_bawp_chroma[0], 2, node.use_bawp_chroma ? 1 : 0)) {
      return false;
    }
    return true;
  }

  int count_refs(int ref) const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos);
    int count = 0;
    for (int i = 0; i < n; ++i) {
      const inter::mc::MiInfo& mi =
          mv_grid_[cell_index(pos[i][0], pos[i][1])];
      if (mi.ref[0] == ref) ++count;
      if (mi.ref[1] == ref) ++count;
    }
    return count;
  }

  // §5.20.7.10 read_single_ref(): NumTotalRefs - 1 single_ref bits, decoded in
  // reference order; the first set bit wins and the last reference is implied.
  bool code_single_ref(int want) {
    const int num_refs = plan_.num_total_refs;
    if (num_refs <= 1) return true;
    if (want < 0 || want >= num_refs) return false;
    for (int ref = 0; ref < num_refs - 1; ++ref) {
      const int this_count = count_refs(ref);
      int next_count = 0;
      for (int i = ref + 1; i < num_refs; ++i) {
        next_count += count_refs(i);
      }
      const int ctx = this_count == next_count ? 1
                      : (this_count < next_count ? 0 : 2);
      const int bit = ref == want ? 1 : 0;
      if (trace_) {
        std::cerr << "  single_ref r=" << cur_r_ << " c=" << cur_c_
                  << " " << cur_w4_ * 4 << "x" << cur_h4_ * 4
                  << " want=" << want << " ref=" << ref << " bit=" << bit
                  << " ctx=" << ctx << " this=" << this_count
                  << " next=" << next_count << "\n";
      }
      if (!sym(&cdfs_.single_ref[ctx][ref][0], 2, bit)) return false;
      if (bit) return true;
    }
    return want == num_refs - 1;
  }

  bool code_drl_index(int ctx, int index, int start = 0) {
    const int m = static_cast<int>(config_.max_drl_bits_minus_1) + 1;
    if (index < start || index > m) return false;
    for (int idx = start; idx < m; ++idx) {
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

  static bool is_inter_ref_value(int ref) { return ref >= 0 && ref < 8; }

  bool ref_backward(int ref) const {
    return is_inter_ref_value(ref) && ref < plan_.num_total_refs &&
           plan_.frame_distance[static_cast<std::size_t>(ref)] < 0;
  }

  // §5.20.7.10 comp_mode context (AV2 v1.0.0):
  //   two neighbours with single prediction -> backward-ness mismatch
  //   one single neighbour                  -> 2 + (backward or intra)
  //   both compound                         -> 4
  int comp_mode_context() const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos);
    const auto nsingle = [&](int i) {
      return !is_inter_ref_value(
          mv_grid_[cell_index(pos[i][0], pos[i][1])].ref[1]);
    };
    const auto nref0 = [&](int i) {
      return mv_grid_[cell_index(pos[i][0], pos[i][1])].ref[0];
    };
    const auto nintra = [&](int i) {
      return is_inter_grid_[cell_index(pos[i][0], pos[i][1])] == 0;
    };
    if (n == 2) {
      if (nsingle(0) && nsingle(1)) {
        return (ref_backward(nref0(0)) ? 1 : 0) ^
               (ref_backward(nref0(1)) ? 1 : 0);
      }
      if (nsingle(0)) {
        return 2 + ((ref_backward(nref0(0)) || nintra(0)) ? 1 : 0);
      }
      if (nsingle(1)) {
        return 2 + ((ref_backward(nref0(1)) || nintra(1)) ? 1 : 0);
      }
      return 4;
    }
    if (n == 1) {
      if (nsingle(0)) return ref_backward(nref0(0)) ? 1 : 0;
      return 3;
    }
    return 1;
  }

  int is_joint_context(int ref0, int ref1) const {
    const int d0 = std::abs(plan_.frame_distance[static_cast<std::size_t>(ref0)]);
    const int d1 = std::abs(plan_.frame_distance[static_cast<std::size_t>(ref1)]);
    const bool same_side =
        (plan_.frame_distance[static_cast<std::size_t>(ref0)] >= 0) ==
        (plan_.frame_distance[static_cast<std::size_t>(ref1)] >= 0);
    return (same_side || d0 != d1) ? 1 : 0;
  }

  // §7.8 FurthestFuture: the future reference (FrameDistance < 0) with the
  // largest order hint.  With frame_distance = OrderHint - OrderHints[ref] the
  // largest order hint is the largest (least negative) distance.
  int furthest_future_ref() const {
    int best = -1;
    for (int ref = 0; ref < plan_.num_total_refs; ++ref) {
      const int dist = plan_.frame_distance[static_cast<std::size_t>(ref)];
      if (dist >= 0) continue;
      if (best < 0 ||
          dist > plan_.frame_distance[static_cast<std::size_t>(best)]) {
        best = ref;
      }
    }
    return best;
  }

  // §9.3 comp_group_idx context, matching the decoder's
  // CompGroupIdxs/RefFrames neighbour scan.
  int comp_group_context(int ref0, int ref1) const {
    const int bck = std::abs(plan_.frame_distance[static_cast<std::size_t>(ref1)]);
    const int fwd = std::abs(plan_.frame_distance[static_cast<std::size_t>(ref0)]);
    const int offset = (fwd == bck) ? 1 : 0;
    int pos[2][2] = {};
    const int n = neighbor_positions(pos);
    int ctxs[2] = {0, 0};
    const int furthest = furthest_future_ref();
    for (int i = 0; i < n && i < 2; ++i) {
      const std::size_t idx = cell_index(pos[i][0], pos[i][1]);
      const inter::mc::MiInfo& mi = mv_grid_[idx];
      if (mi.ref[1] >= 0 && mi.ref[1] < 8) {
        ctxs[i] = comp_group_grid_[idx] ? 1 : 0;
      } else if (furthest >= 0 && mi.ref[0] == furthest) {
        ctxs[i] = 2;
      }
    }
    return ctxs[1] + ctxs[0] +
           (std::min(ctxs[1], ctxs[0]) > 0 ? 1 : 0) + offset * 6;
  }

  // §5.20.7.10 read_ref_frames() + §5.20.7.6 compound mode for the gated
  // subset: NumTotalRefs == 2 with num_same_ref_compound == 0, so the (0,1)
  // pair is implied and no comp_ref bit is coded.  The only signalled compound
  // mode is non-joint GLOBAL_GLOBALMV (compound_mode_non_joint = 3) with the
  // AVERAGE blend: enable_masked_compound == 0 makes read_compound_type
  // bitless and enable_cwp == 0 suppresses cwp_idx.
  // Builds the §7.12.2 stack the decoder derives for one reference.  With
  // `is_compound` the two-list stack (skip_mode / use_optflow form) is built;
  // otherwise the single-reference stack for `ref`.  The global MVs come from
  // the same §7.12.2.2 projection the writer stores in the MI grid.
  bool build_mode_stack(const BlockSyntax& node, int ref, bool is_compound,
                        const BlockSyntax& mv_source,
                        merge::StackParams& params,
                        merge::MvStack& stack) const {
    params.mi_row = node.mi_row;
    params.mi_col = node.mi_col;
    params.bw4 = cur_w4_;
    params.bh4 = cur_h4_;
    params.sb_h4 = 16;
    params.mi_cols = mi_cols_;
    params.mi_rows = mi_rows_;
    params.mi = mv_grid_.data();
    params.ref_frame[0] = static_cast<std::int8_t>(ref);
    params.is_compound = is_compound;
    params.ref_frame[1] = is_compound
                              ? static_cast<std::int8_t>(mv_source.ref_index1)
                              : merge::kRefNone;
    params.global_mv[0] = block_global_mv_for_ref(mv_source, ref);
    params.global_mv[1] = is_compound
                              ? block_global_mv_for_ref(
                                    mv_source, mv_source.ref_index1)
                              : params.global_mv[0];
    params.reorder = merge::DrlReorder::kDisabled;
    params.max_drl_bits_minus_1 =
        static_cast<int>(config_.max_drl_bits_minus_1);
    if (plan_.num_total_refs > 1) {
      params.num_refs = plan_.num_total_refs;
      params.frame_distance = plan_.frame_distance.data();
    }
    return merge::build_stack_checked(params, &stack) == merge::Status::kOk;
  }

  // §7.13.3.8 opfl_allowed_for_refs() for this frame's reference pair.
  bool opfl_allowed_for_pair(int ref0, int ref1) const {
    if (plan_.frame_is_switch) return false;
    const int d0 = plan_.frame_distance[static_cast<std::size_t>(ref0)];
    const int d1 = plan_.frame_distance[static_cast<std::size_t>(ref1)];
    return inter::temporal::opfl_allowed_for_refs(d0, d1, false, false, false);
  }

  // §5.20.7.17 non-GLOBALMV compound refine-MV decision (no bit: NEAR_NEARMV
  // is not switchable).
  bool refine_mv_derived(const BlockSyntax& node, bool* use) const {
    inter::temporal::RefineMvConfig ctx;
    ctx.enable_refinemv = plan_.enable_refinemv;
    ctx.opfl_refine_type =
        static_cast<inter::temporal::RefineType>(plan_.opfl_refine_type);
    ctx.frame_is_switch = plan_.frame_is_switch;
    ctx.is_compound = true;
    ctx.skip_mode = false;
    ctx.use_optflow = node.use_optflow != 0;
    ctx.is_global_global_mv = node.compound_mode == 3;
    ctx.motion_mode_simple = true;
    ctx.has_newmv = false;
    ctx.y_mode_near_near_mv = node.compound_mode == 0;
    ctx.y_mode_joint_new_mv = false;
    ctx.block_width = cur_w4_ * 4;
    ctx.block_height = cur_h4_ * 4;
    const int d0 = plan_.frame_distance[static_cast<std::size_t>(node.ref_index)];
    const int d1 =
        plan_.frame_distance[static_cast<std::size_t>(node.ref_index1)];
    // Synthetic origin: only the relative distances matter, and -1 is the
    // §5.18.3.1 restricted sentinel.
    ctx.order_hint = 64;
    ctx.ref0_order_hint = 64 - d0;
    ctx.ref1_order_hint = 64 - d1;
    inter::temporal::RefineMvSyntax syntax;
    if (inter::temporal::refine_mv_syntax(ctx, &syntax) !=
        inter::temporal::TemporalStatus::kOk) {
      return false;
    }
    *use = syntax.use_refinemv;
    return true;
  }

  bool write_compound_mode_info(const BlockSyntax& node, std::string& error) {
    if (!plan_.reference_select) {
      error = "compound block without reference_select";
      return false;
    }
    if (plan_.num_total_refs != 2) {
      error = "compound syntax is wired for NumTotalRefs == 2";
      return false;
    }
    if (node.ref_index != 0 || node.ref_index1 != 1) {
      error = "compound reference pair must be (0, 1)";
      return false;
    }
    if (node.compound_joint != 0) {
      error = "joint compound is not wired";
      return false;
    }
    if (!node.skip_flag) {
      error = "compound inter residual is not wired";
      return false;
    }
    const bool near_near = node.compound_mode == 0;
    const bool global_global = node.compound_mode == 3;
    if (!near_near && !global_global) {
      error = "unsupported compound mode " +
              std::to_string(static_cast<int>(node.compound_mode));
      return false;
    }
    if (!near_near && (node.inter_mode != kInterGlobalMv || node.mv_y != 0 ||
                       node.mv_x != 0 || node.mv1_y != 0 || node.mv1_x != 0 ||
                       node.use_optflow || node.use_refinemv)) {
      error = "only GLOBAL_GLOBALMV compound is wired without MVs";
      return false;
    }
    // Validate the decision against the decoder's MV stack(s).  This is the
    // only place a compound MV can silently desynchronise.
    if (near_near) {
      merge::MvStack stacks[2];
      merge::StackParams params[2];
      inter::mc::Mv resolve[2];
      if (node.use_optflow) {
        merge::MvStack compound;
        merge::StackParams cparams;
        if (!build_mode_stack(node, 0, true, node, cparams, compound)) {
          error = "build_mv_stack (compound) failed";
          return false;
        }
        if (node.drl_index >= compound.count) {
          error = "compound DRL index outside the MV stack";
          return false;
        }
        resolve[0] = compound.cand[node.drl_index].mv[0];
        resolve[1] = compound.cand[node.drl_index].mv[1];
      } else {
        if (!build_mode_stack(node, 0, false, node, params[0], stacks[0]) ||
            !build_mode_stack(node, 1, false, node, params[1], stacks[1])) {
          error = "build_mv_stack (per-list) failed";
          return false;
        }
        if (merge::resolve_compound_near_near(stacks, node.drl_index,
                                              node.drl_index1, resolve) !=
            merge::Status::kOk) {
          error = "compound per-list DRL index outside the MV stack";
          return false;
        }
      }
      if (resolve[0].y != node.mv_y || resolve[0].x != node.mv_x ||
          resolve[1].y != node.mv1_y || resolve[1].x != node.mv1_x) {
        error = "compound MV does not match the stack candidates";
        return false;
      }
    }
    // §5.20.7.17: the derived refine-MV decision must match the prediction the
    // encoder applied.  NEAR_NEARMV is never switchable, so it codes no bit.
    bool derived_refinemv = false;
    if (near_near && !refine_mv_derived(node, &derived_refinemv)) {
      error = "refine_mv_syntax failed";
      return false;
    }
    if (derived_refinemv != (node.use_refinemv != 0)) {
      error = "compound refine-MV decision does not match §5.20.7.17";
      return false;
    }
    // §5.20.7.6 use_optflow is coded for a switchable frame over an allowed
    // reference pair, before the DRL indices and compound type.
    const bool optflow_eligible =
        near_near && plan_.opfl_refine_type == 1 && opfl_allowed_for_pair(0, 1) &&
        cur_w4_ * 4 >= 8 && cur_h4_ * 4 >= 8 && !plan_.frame_is_switch;
    if (!optflow_eligible && node.use_optflow) {
      error = "use_optflow set outside the §5.20.7.6 gate";
      return false;
    }
    if (!sym(&cdfs_.comp_mode[comp_mode_context()][0], 2, 1)) {
      error = error_;
      return false;
    }
    // (0, 1) is implied: no comp_ref symbols for NumTotalRefs == 2.
    if (!sym(&cdfs_.is_joint[is_joint_context(0, 1)][0], 2,
             node.compound_joint)) {
      error = error_;
      return false;
    }
    merge::NewMvContextParams nparams;
    nparams.mi_row = node.mi_row;
    nparams.mi_col = node.mi_col;
    nparams.bw4 = cur_w4_;
    nparams.bh4 = cur_h4_;
    nparams.sb_h4 = 16;
    nparams.mi_cols = mi_cols_;
    nparams.mi_rows = mi_rows_;
    nparams.mi = mv_grid_.data();
    nparams.y_mode = ymode_grid_.data();
    nparams.is_compound = true;
    nparams.ref_frame[0] = static_cast<std::int8_t>(node.ref_index);
    nparams.ref_frame[1] = static_cast<std::int8_t>(node.ref_index1);
    int ctx = 0;
    if (merge::derive_new_mv_context(nparams, &ctx) != merge::Status::kOk) {
      error = "derive_new_mv_context failed";
      return false;
    }
    if (!sym(&cdfs_.compound_mode_non_joint[ctx][0], 5,
             node.compound_mode)) {
      error = error_;
      return false;
    }
    if (optflow_eligible &&
        !sym(&cdfs_.use_optflow[node.compound_mode != 0 ? 1 : 0][0], 2,
             node.use_optflow ? 1 : 0)) {
      error = error_;
      return false;
    }
    if (near_near) {
      // §5.20.7.8 read_drl_idx(): has_second_drl() is true for a non-skip
      // NEAR_NEARMV block without use_optflow, so both list indices are coded.
      if (!code_drl_index(ctx, node.drl_index)) {
        error = "write drl_idx failed";
        return false;
      }
      if (!node.use_optflow) {
        int start = 0;
        if (merge::derive_second_drl_start(
                merge::kYModeNearNearMv,
                node.ref_index == node.ref_index1, node.drl_index,
                &start) != merge::Status::kOk) {
          error = "derive_second_drl_start failed";
          return false;
        }
        if (!code_drl_index(ctx, node.drl_index1, start)) {
          error = "write second drl_idx failed";
          return false;
        }
      }
      // §5.20.7.17 read_refinemv(): only the switchable form codes a bit, and
      // NEAR_NEARMV is not switchable, so the decision above is implicit.
    }
    // §5.20.7.16 read_compound_type().  use_optflow (and the switchable
    // refine-MV form) select COMPOUND_AVERAGE and code no symbols.  Otherwise,
    // with enable_masked_compound the comp_group_idx symbol is present for
    // every compound block (0 = AVERAGE); a masked block then codes
    // compound_type and the mask symbols.
    if (node.use_optflow) {
      if (node.compound_masked) {
        error = "use_optflow compound must be COMPOUND_AVERAGE";
        return false;
      }
    } else if (plan_.enable_masked_compound) {
      if (node.compound_masked && plan_.num_total_refs != 2) {
        error = "masked compound is wired for two references";
        return false;
      }
      if (!sym(&cdfs_.comp_group_idx[comp_group_context(0, 1)][0], 2,
               node.compound_masked ? 1 : 0)) {
        error = error_;
        return false;
      }
      if (node.compound_masked) {
        const bool wedge_ok = inter::adv::bsize_has_wedge(
            static_cast<int>(node.bsize));
        if (node.compound_type == 0) {
          if (!wedge_ok) {
            error = "wedge compound is not available for this block size";
            return false;
          }
          if (!sym(&cdfs_.compound_type[0], 2, 0)) {
            error = error_;
            return false;
          }
          multiref::WedgeSyntax syntax;
          if (!multiref::wedge_index_to_syntax(node.wedge_index, syntax)) {
            error = "wedge index is not signalable";
            return false;
          }
          if (!sym(&cdfs_.wedge_quad[0], 4, syntax.quad) ||
              !sym(&cdfs_.wedge_angle[syntax.quad][0], 5, syntax.angle)) {
            error = error_;
            return false;
          }
          const int angle = syntax.quad * 5 + syntax.angle;
          const bool dist2 = angle >= 10 || angle == 5 || angle == 0;
          if (dist2) {
            if (!sym(&cdfs_.wedge_dist2[0], 3, syntax.dist - 1)) {
              error = error_;
              return false;
            }
          } else if (!sym(&cdfs_.wedge_dist1[0], 4, syntax.dist)) {
            error = error_;
            return false;
          }
          if (!raw_bit(node.wedge_sign != 0)) {
            error = error_;
            return false;
          }
        } else if (node.compound_type == 1) {
          if (!sym(&cdfs_.compound_type[0], 2, 1)) {
            error = error_;
            return false;
          }
          if (!raw_bit(node.mask_type != 0)) {
            error = error_;
            return false;
          }
        } else {
          error = "invalid masked compound type";
          return false;
        }
      } else if (node.compound_type != 0) {
        error = "masked type without comp_group_idx";
        return false;
      }
    } else if (node.compound_masked) {
      error = "masked compound requires enable_masked_compound = 1";
      return false;
    }
    return true;
  }

  bool write_inter_mode_info(const BlockSyntax& node, std::string& error) {
    if (node.compound) {
      return write_compound_mode_info(node, error);
    }
    if (node.inter_mode > kInterNewMv) {
      error = "invalid inter mode";
      return false;
    }
    if (node.ref_index >= plan_.num_total_refs) {
      error = "block reference index outside the frame reference map";
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
    params.ref_frame[0] = static_cast<std::int8_t>(node.ref_index);
    params.ref_frame[1] = merge::kRefNone;
    params.is_compound = false;
    // §7.12.2.2 GlobalMvs[0] for the MV-stack candidates.  With global motion
    // enabled the decoder projects the accepted model at the frame precision;
    // the encoder must feed the same value or a NEARMV/derived candidate can
    // resolve differently.
    const inter::mc::Mv global_mv = block_global_mv(node);
    params.global_mv[0] = global_mv;
    params.global_mv[1] = global_mv;
    // DRL reordering is not gated; the sequence header signals DISABLED.
    params.reorder = merge::DrlReorder::kDisabled;
    params.max_drl_bits_minus_1 =
        static_cast<int>(config_.max_drl_bits_minus_1);
    if (plan_.num_total_refs > 1) {
      params.num_refs = plan_.num_total_refs;
      params.frame_distance = plan_.frame_distance.data();
    }
    // §7.12.2 steps 5/12 temporal scan: the writer must rebuild the same
    // stack the decision path used, including the §7.9 motion-field
    // candidates.
    params.use_ref_frame_mvs = plan_.temporal.use_ref_frame_mvs;
    params.motion_field = plan_.temporal.field;
    params.temporal_scan = plan_.temporal.scan;
    params.order_hint = plan_.temporal.order_hint;
    params.order_hints = plan_.temporal.ref_order_hints.data();
    params.num_past_refs = plan_.temporal.num_past_refs;
    params.num_future_refs = plan_.temporal.num_future_refs;
    params.closest_past = plan_.temporal.closest_past;
    params.closest_future = plan_.temporal.closest_future;
    params.tile_mi_col_start = plan_.tile_mi_col;
    params.tile_mi_row_start = plan_.tile_mi_row;
    params.tile_mi_col_end = plan_.frame_mi_cols;
    params.tile_mi_row_end = plan_.frame_mi_rows;
    merge::MvStack stack;
    const drl::Mode reorder_mode =
        plan_.drl_reorder == 2
            ? drl::Mode::kAlways
            : (plan_.drl_reorder == 1 ? drl::Mode::kConstraint
                                      : drl::Mode::kDisabled);
    // §7.12.2 step 16 as the pinned decoder evaluates it: its
    // high-priority-TMVP term is not gated by use_ref_frame_mvs, so
    // DRL_REORDER_CONSTRAINT does not sort a single reference within two POCs.
    // See the matching gate in video_frame.cpp (load_stack).
    bool suppress = false;
    if (reorder_mode == drl::Mode::kConstraint) {
      bool past = false;
      bool future = false;
      for (int r = 0; r < plan_.num_total_refs && r < 7; ++r) {
        const int d = plan_.frame_distance[static_cast<std::size_t>(r)];
        if (d > 0) {
          past = true;
        } else if (d < 0) {
          future = true;
        }
      }
      if (!(past && future) && node.ref_index >= 0 && node.ref_index < 7) {
        const int d =
            plan_.frame_distance[static_cast<std::size_t>(node.ref_index)];
        suppress = (d < 0 ? -d : d) <= 2;
      }
    }
    const merge::Status stack_status =
        reorder_mode == drl::Mode::kDisabled
            ? merge::build_stack_checked(params, &stack)
            : drl::build_reordered_stack(params, reorder_mode, suppress,
                                         &stack);
    if (stack_status != merge::Status::kOk) {
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
      if (plan_.mv_precision != 0 && node.inter_mode == kInterNewMv) {
        // The MVD is coded in precision units but the predictor is used
        // unchanged (only precisions coarser than 1/2 pel lower it).
        const inter::mc::Mv pred = stack.cand[drl].mv[0];
        const inter::mc::Mv expected{pred.y + node.mvd_y, pred.x + node.mvd_x};
        if (expected.y != node.mv_y || expected.x != node.mv_x) {
          error = "block MV does not match the signalled inter mode";
          return false;
        }
      } else {
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
      }
      if (node.inter_mode == kInterNearMv &&
          node.drl_index >= stack.count) {
        error = "DRL index outside the MV stack";
        return false;
      }
    }

    if (plan_.num_total_refs > 1) {
      // §5.20.7.10 read_ref_frames(): when reference_select is on and the
      // block may be compound, comp_mode is coded first (0 = single ref).
      if (plan_.reference_select && cur_w4_ * 4 >= 8 && cur_h4_ * 4 >= 8) {
        if (!sym(&cdfs_.comp_mode[comp_mode_context()][0], 2, 0)) {
          error = error_;
          return false;
        }
      }
      if (!code_single_ref(node.ref_index)) {
        error = "write single_ref failed";
        return false;
      }
    }

    int ctx = 0;
    if (plan_.num_total_refs > 1) {
      merge::NewMvContextParams nparams;
      nparams.mi_row = node.mi_row;
      nparams.mi_col = node.mi_col;
      nparams.bw4 = cur_w4_;
      nparams.bh4 = cur_h4_;
      nparams.sb_h4 = 16;
      nparams.mi_cols = mi_cols_;
      nparams.mi_rows = mi_rows_;
      nparams.mi = mv_grid_.data();
      nparams.y_mode = ymode_grid_.data();
      nparams.is_compound = false;
      nparams.ref_frame[0] = static_cast<std::int8_t>(node.ref_index);
      if (merge::derive_new_mv_context(nparams, &ctx) != merge::Status::kOk) {
        error = "derive_new_mv_context failed";
        return false;
      }
    } else {
      ctx = new_mv_context();
    }
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
    // §5.20.7.6: use_bawp follows single_mode for a single-prediction block
    // that is at least 8x8, not GLOBALMV, and not a switch frame.
    const bool bawp_available =
        plan_.allow_bawp && cur_w4_ * 4 >= 8 && cur_h4_ * 4 >= 8 &&
        node.inter_mode != kInterGlobalMv;
    if (bawp_available) {
      if (!code_bawp(node)) {
        error = error_;
        return false;
      }
    } else if (node.use_bawp || node.explicit_bawp ||
               node.use_bawp_chroma) {
      error = "BAWP symbols set outside the §5.20.7.6 gate";
      return false;
    }
    // §5.20.7.14 read_motion_mode(isCompound = 0).  It follows single_mode /
    // use_bawp and precedes read_drl_idx().  INTERINTRA is allowed only for a
    // non-compound block of 8x8..64x64 that does not use BAWP.
    if (!node.compound) {
      const int bw = cur_w4_ * 4;
      const int bh = cur_h4_ * 4;
      const bool interintra_allowed =
          plan_.enable_interintra && !node.use_bawp && bw >= 8 && bh >= 8 &&
          bw <= 64 && bh <= 64;
      if (!interintra_allowed && node.motion_mode != 0) {
        error = "INTERINTRA motion mode outside the §5.20.7.14 gate";
        return false;
      }
      if (plan_.enable_interintra) {
        adv::MotionModeSymbols symbols;
        symbols.is_warp_newmv = false;
        symbols.extend_allowed = false;
        symbols.local_allowed = false;
        symbols.inter_intra_allowed = interintra_allowed;
        symbols.inter_intra = node.motion_mode == 1;
        symbols.interintra_mode = node.interintra_mode;
        symbols.wedge_allowed = adv::kWedgeBits[node.bsize] != 0;
        symbols.wedge_interintra = node.wedge_interintra != 0;
        symbols.wedge_index = node.wedge_index;
        std::string mm_error;
        if (!adv::write_motion_mode(enc_, adv_cdfs_, node.bsize, symbols,
                                    &mm_error)) {
          error = "motion mode: " + mm_error;
          return false;
        }
      }
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
      if (plan_.mv_precision == 0) {
        if (!write_mv(node.mvd_y, node.mvd_x)) {
          error = error_;
          return false;
        }
      } else {
        const int precision = plan_.mv_precision == 1
                                  ? av2enc::me2::kMvPrecisionQuarterPel
                                  : av2enc::me2::kMvPrecisionHalfPel;
        std::string shell_error;
        if (!inter::mvprec::write_shell_mv(enc_, cdfs_.mv_precision,
                                           node.mvd_y, node.mvd_x, precision,
                                           &shell_error)) {
          error = "write_shell_mv failed: " + shell_error;
          return false;
        }
        if (node.mvd_y != 0 && !raw_bit(node.mvd_y < 0)) {
          error = error_;
          return false;
        }
        if (node.mvd_x != 0 && !raw_bit(node.mvd_x < 0)) {
          error = error_;
          return false;
        }
      }
    }
    return true;
  }

  // §7.12.2.2 GlobalMvs for this block at the frame precision.  The decoder
  // stores the projected value in MiMv for GLOBALMV blocks and seeds the merge
  // stack with it, so both the stack validation and the writer's own MI grid
  // must use it instead of the (zero) coded mv.
  inter::mc::Mv block_global_mv_for_ref(const BlockSyntax& node,
                                        int ref) const {
#if defined(AV2ENC_VIDEO_GME)
    if (ref >= 0 && plan_.gm_num_refs > ref) {
      inter::gme::MvPrecision precision = inter::gme::MvPrecision::kEighthPel;
      if (plan_.mv_precision == 1) {
        precision = inter::gme::MvPrecision::kQuarterPel;
      } else if (plan_.mv_precision == 2) {
        precision = inter::gme::MvPrecision::kHalfPel;
      }
      int projected[2] = {0, 0};
      inter::gme::get_warp_motion_vector(
          plan_.gm_params[static_cast<std::size_t>(ref)], node.mi_col,
          node.mi_row, cur_w4_, cur_h4_, mi_cols_, mi_rows_, precision,
          projected);
      return inter::mc::Mv{projected[0], projected[1]};
    }
#else
    (void)node;
    (void)ref;
#endif
    return inter::mc::Mv{0, 0};
  }

  inter::mc::Mv block_global_mv(const BlockSyntax& node) const {
    return block_global_mv_for_ref(node, node.ref_index);
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
        comp_group_grid_[idx] =
            (node.compound && node.compound_masked) ? 1 : 0;
        ymode_grid_[idx] =
            node.is_inter
                ? static_cast<std::uint8_t>(
                      node.compound
                          ? 19 + node.compound_mode
                          : (node.inter_mode == kInterGlobalMv
                                 ? 15
                                 : (node.inter_mode == kInterNearMv ? 14
                                                                    : 16)))
                : node.y_mode;
        inter::mc::MiInfo& mi = mv_grid_[idx];
        mi = inter::mc::MiInfo{};
        mi.ref[0] = node.is_inter
                        ? static_cast<std::int8_t>(node.ref_index)
                        : kRefIntraAv2;
        mi.ref[1] = node.compound
                        ? static_cast<std::int8_t>(node.ref_index1)
                        : (node.is_inter && node.motion_mode == 1
                               ? kRefIntraAv2
                               : merge::kRefNone);
        // A GLOBALMV block stores GlobalMvs[0]; every other mode stores its
        // coded MV.  A compound block stores the per-list MVs.  This mirrors
        // decode_block() and keeps later merge-stack candidates identical to
        // the decoder's.
        inter::mc::Mv mv0 = inter::mc::Mv{node.mv_y, node.mv_x};
        inter::mc::Mv mv1 = inter::mc::Mv{node.mv1_y, node.mv1_x};
        if (node.is_inter && node.inter_mode == kInterGlobalMv) {
          mv0 = block_global_mv_for_ref(node, node.ref_index);
          mv1 = node.compound ? block_global_mv_for_ref(node, node.ref_index1)
                              : mv0;
        }
        mi.mv[0][0] = mv0.y;
        mi.mv[0][1] = mv0.x;
        mi.mv[1][0] = mv1.y;
        mi.mv[1][1] = mv1.x;
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
    if (!has_chroma_) return;
    const int cx4 = c >> sub_x_;
    const int cy4 = r >> sub_y_;
    const int cw4 = w4 >> sub_x_;
    const int ch4 = h4 >> sub_y_;
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

  // Adapter that forwards §5.20.6.3 partition symbols to this writer's range
  // coder.  The TX-partition component owns the syntax and contexts.
  struct TxPartitionSink final : txpartition::SymbolSink {
    Writer* writer = nullptr;
    bool symbol(std::uint16_t* cdf, int n, int value,
                std::string& error) override {
      if (!writer->sym(cdf, n, value)) {
        error = writer->error_;
        return false;
      }
      return true;
    }
  };

  // Total number of luma levels the leaf's plan contributed.
  int luma_level_count(const BlockSyntax& node) const {
    if (!plan_.tx_mode_select || node.tx_part_count == 0) {
      return static_cast<int>(node.eob[0]);
    }
    int total = 0;
    for (int i = 0; i < node.tx_part_count; ++i) {
      total += static_cast<int>(node.tx_part_eob[i]);
    }
    return total;
  }

  // Codes the luma residual of one leaf.  In TX_MODE_LARGEST this is the
  // historical single `code_tx()` call; with TX_MODE_SELECT it first emits the
  // §5.20.6.3 partition (including the "NONE" do_partition symbol) and then one
  // transform block per returned part, using the per-part TX size/type/EOB and
  // the part's own 4x4 origin for the §8.3.2 contexts.
  bool code_luma_residual(const BlockSyntax& node, int w, int h,
                          std::size_t levels_end, bool is_inter,
                          std::size_t level_base, std::string& error) {
    if (!plan_.tx_mode_select) {
      const TxView view{node.tx_size[0], node.tx_type[0],
                        static_cast<int>(node.eob[0]), node.mi_row,
                        node.mi_col,
                        static_cast<int>(node.sec_tx_type[0]),
                        static_cast<int>(node.most_probable_stx_set[0])};
      return code_tx_view(node, 0, view, level_base, w, h, levels_end,
                          is_inter, error);
    }
    const int part_count = node.tx_part_count != 0 ? node.tx_part_count : 1;
    if (part_count < 1 || part_count > 4) {
      error = "luma TX part count out of range";
      return false;
    }
    // BLOCK_4X4 codes no partition syntax; every other M2 coding block does.
    if (node.bsize != static_cast<std::uint8_t>(txpartition::kBlock4x4)) {
      txpartition::BlockTxContext ctx;
      ctx.mi_size = node.bsize;
      ctx.is_inter = is_inter;
      ctx.fsc_mode = false;
      ctx.reduced_tx_part_set = plan_.reduced_tx_part_set;
      TxPartitionSink sink;
      sink.writer = this;
      txpartition::SplitGeometry relative;
      if (!txpartition::write_tx_partition(
              sink, cdfs_.tx_partition, ctx,
              static_cast<txpartition::TxPartition>(node.tx_partition),
              &relative, error)) {
        return false;
      }
      if (relative.count != part_count) {
        error = "luma TX part count does not match the signalled partition";
        return false;
      }
      for (int i = 0; i < part_count; ++i) {
        if (relative.parts[i].tx_size != node.tx_part_tx_size[i]) {
          error = "luma TX part size does not match the signalled geometry";
          return false;
        }
      }
    } else if (part_count != 1) {
      error = "BLOCK_4X4 cannot signal more than one transform";
      return false;
    }

    std::size_t base = level_base;
    for (int i = 0; i < part_count; ++i) {
      const TxView view{
          node.tx_part_tx_size[i], node.tx_part_tx_type[i],
          static_cast<int>(node.tx_part_eob[i]),
          static_cast<int>(node.mi_row) + static_cast<int>(node.tx_part_mi_row[i]),
          static_cast<int>(node.mi_col) + static_cast<int>(node.tx_part_mi_col[i]),
          static_cast<int>(node.tx_part_sec_tx_type[i]),
          static_cast<int>(node.tx_part_stx_set[i])};
      if (!code_tx_view(node, 0, view, base, w, h, levels_end, is_inter,
                        error)) {
        return false;
      }
      base += node.tx_part_eob[i];
    }
    return true;
  }

  // Intra mode + residual syntax. Used by keyframe blocks and by intra
  // fallback blocks inside an inter frame.
  // ---------------------------------------------------------------------
  // §5.20.5.3/.4 intra block copy + §5.20.5.4 morph_pred on an intra frame.
  // ---------------------------------------------------------------------
  // `is_bv_valid` only needs the plane geometry, so the writer builds a view
  // with null sample pointers; it never copies pixels.
  ibc::FrameView ibc_frame_view() const {
    ibc::FrameView view;
    view.y = ibc::PlaneView{nullptr, mi_cols_ * 4, mi_cols_ * 4,
                            mi_rows_ * 4};
    view.u = ibc::PlaneView{nullptr, chroma_mi_cols() * 4,
                            chroma_mi_cols() * 4, chroma_mi_rows() * 4};
    view.v = view.u;
    view.has_chroma = has_chroma_;
    view.sub_x = sub_x_;
    view.sub_y = sub_y_;
    return view;
  }

  // §5.20.5.3 use_intrabc (+ read_skip when set).  Present for every leaf when
  // allow_intrabc is on; only eligible block sizes carry the symbol.
  bool write_keyframe_mode_info(const BlockSyntax& node, bool is_ibc,
                                std::string& error) {
    if (!plan_.allow_intrabc) {
      if (is_ibc) {
        error = "IBC leaf supplied but the frame disables intrabc";
        return false;
      }
      return true;
    }
    if (!ibcs::intrabc_available(plan_.allow_intrabc, node.bsize)) {
      if (is_ibc) {
        error = "IBC leaf is not signalable at this block size";
        return false;
      }
      return true;
    }
    if (is_ibc) {
      // read_intrabc_info() forces DC luma/chroma and clears palette/CfL.
      if (node.y_mode != kDcPred || (node.flags & kFlagIsCfl) != 0) {
        error = "IBC leaf carries intra/palette mode state";
        return false;
      }
    }
    const int ctx = std::min(
        ibcs::ibc_symbol_context(ibc_grid_, mi_cols_, mi_rows_, node.mi_row,
                                 node.mi_col, node.bsize, plan_.sb_size),
        2);
    if (!ibcs::write_use_intrabc(enc_, ibc_cdfs_, ctx, is_ibc, error)) {
      return false;
    }
    if (!is_ibc) return true;
    const int skip_ctx = std::min(
        ibcs::skip_symbol_context(ibc_skip_grid_, mi_cols_, mi_rows_,
                                  node.mi_row, node.mi_col, node.bsize,
                                  plan_.sb_size),
        2);
    return ibcs::write_skip_flag(enc_, ibc_cdfs_, skip_ctx,
                                 node.ibc_skip != 0, error);
  }

  // §5.20.5.4 intrabc_info() (intrabc_mode/DRL/assign_mv) and, when the
  // sequence enables it, morph_pred.  The BV stack is rebuilt from the same
  // neighbour map and ring buffer the decoder maintains.
  bool write_ibc_body(const BlockSyntax& node, std::string& error) {
    const ibc::FrameView frame = ibc_frame_view();
    const ibcs::BvStack stack = ibcs::build_bv_stack(
        frame, ibc_config_, ibc_buffer_, node.mi_row, node.mi_col, node.bsize,
        ibc_neighbours_);
    ibcs::BvSyntax syntax;
    syntax.mode = node.ibc_mode == 1 ? ibcs::IntrabcMode::kNearMv
                                     : ibcs::IntrabcMode::kNewMv;
    syntax.ref_mv_idx = node.ibc_ref_mv_idx;
    syntax.intrabc_precision_quarter = false;
    const ibc::BlockVector target{node.ibc_bv_row, node.ibc_bv_col};
    if (!ibcs::write_intrabc_info(enc_, ibc_cdfs_, ibc_content_, stack, syntax,
                                  target, error)) {
      return false;
    }
    if (plan_.morph_pred_present) {
      const int w4 = std::max(1, cur_w4_);
      const int h4 = std::max(1, cur_h4_);
      const int ctx = morph::morph_symbol_context(
          morph_grid_.data(), mi_cols_, mi_rows_, node.mi_row, node.mi_col, w4,
          h4, plan_.sb_size);
      if (!morph::write_morph_pred(enc_, morph_cdfs_, ctx,
                                   node.morph_pred != 0, error)) {
        return false;
      }
    } else if (node.morph_pred != 0) {
      error = "morph_pred set but the frame does not signal it";
      return false;
    }
    if (node.ibc_skip != 0) {
      // §5.20.5.1: an IBC skip block codes no residual, but the decoder still
      // resets the level/DC contexts over the block.
      reset_coeff_contexts(node.mi_row, node.mi_col, cur_w4_, cur_h4_);
    }
    return true;
  }

  // §5.20.8 palette_mode_info() for a keyframe leaf.  When screen-content tools
  // are enabled the symbol is present for every DC block that satisfies the
  // §5.20.5.5 predicate, so a non-palette leaf still codes has_palette_y = 0.
  bool write_palette(const BlockSyntax& node, int w, int h,
                     std::string& error) {
    if (!plan_.allow_screen_content_tools) return true;
    if ((node.flags & kFlagIsIbc) != 0) return true;
    if (!palette_writer_.has_value()) {
      error = "palette writer is not initialised: " + palette_reset_error_;
      return false;
    }
    const int onscreen_w = std::min(w, mi_cols_ * 4 - node.mi_col * 4);
    const int onscreen_h = std::min(h, mi_rows_ * 4 - node.mi_row * 4);
    const palette::PaletteResult* result = nullptr;
    if (node.palette_index != BlockSyntax::kNoPalette) {
      if (node.palette_index >= plan_.palettes.size()) {
        error = "palette index is outside the frame palette list";
        return false;
      }
      result = &plan_.palettes[node.palette_index];
      if (!result->has_palette) {
        error = "palette leaf carries an empty palette result";
        return false;
      }
    }
    pals::PaletteModeInfo info;
    std::string palette_error;
    if (!palette_writer_->write(enc_, palette_cdfs_, node.mi_row, node.mi_col,
                                node.bsize, w, h, onscreen_w, onscreen_h,
                                result, node.y_mode, &info, palette_error)) {
      error = "palette syntax: " + palette_error;
      return false;
    }
    const bool want_palette = result != nullptr;
    if (info.has_palette_y != want_palette) {
      error = "palette syntax did not match the supplied decision";
      return false;
    }
    return true;
  }

  bool write_intra_leaf(const BlockSyntax& node, int w, int h, int cw, int ch,
                        std::string& error) {
    const int tree_type = node.tree_type;
    const bool chroma_part = tree_type == 2;
    const bool chroma_info = (node.flags & kFlagHasChromaInfo) != 0;
    const bool node_has_chroma = !chroma_info || node.has_chroma != 0;
    const int chroma_mi_row =
        chroma_info ? static_cast<int>(node.chroma_mi_row) : node.mi_row;
    const int chroma_mi_col =
        chroma_info ? static_cast<int>(node.chroma_mi_col) : node.mi_col;
    // Validate the declared transform sizes against TX_MODE_LARGEST geometry
    // before emitting anything.  Under TX_MODE_SELECT the luma size is carried
    // by the per-part list (validated against the signalled partition below);
    // chroma keeps the TX_MODE_LARGEST geometry.  A CHROMA_PART leaf has no
    // luma transform; a non-owning luma leaf has no chroma transforms.
    const int coded_w[kPlaneCount] = {w, cw, cw};
    const int coded_h[kPlaneCount] = {h, ch, ch};
    const int plane_count =
        node_has_chroma ? kPlaneCount : (chroma_part ? 0 : 1);
    for (int plane = plan_.tx_mode_select ? 1 : 0; plane < plane_count;
         ++plane) {
      if ((plane == 0 && chroma_part) || (plane > 0 && !node_has_chroma)) {
        continue;
      }
      const auto params_opt = txv::make_tx_params(
          static_cast<txv::TxSize>(node.tx_size[plane]),
          txv::TxType::kDctDct);
      if (!params_opt.has_value() ||
          params_opt->coded_w != coded_w[plane] ||
          params_opt->coded_h != coded_h[plane]) {
        error = "tx_size on plane " + std::to_string(plane) + " (" +
                std::to_string(node.tx_size[plane]) + ") does not match the " +
                "TX_MODE_LARGEST block size " +
                std::to_string(coded_w[plane]) + "x" +
                std::to_string(coded_h[plane]) + " tree=" +
                std::to_string(node.tree_type) + " bsize=" +
                std::to_string(node.bsize) + " chroma_bsize=" +
                std::to_string(node.chroma_bsize) + " has_chroma=" +
                std::to_string(node.has_chroma);
        return false;
      }
    }

    const std::size_t levels_end = levels_.size();
    std::size_t level_base = node.levels_offset;
    if (!chroma_part) {
      if (!code_y_mode(node)) {
        error = error_;
        return false;
      }
      // §5.20.5.5: after the Y mode/angle delta, every luma coding block with
      // Block_Width/Height <= FSC_MAX codes fsc_mode when enable_idtx_intra is
      // 1.  This encoder never uses FSC (enable_fsc = 0), so it codes 0 and all
      // neighbours stay 0 (CDF context [0][group]).
      if (config_.enable_idtx_intra && w <= 32 && h <= 32) {
        const int group = kFscBsizeGroups[node.bsize];
        if (!sym(&cdfs_.fsc_mode[0][group][0], 2, 0)) return false;
      }
      if (config_.enable_mrls) {
        const mrl_stream::MrlBlockSyntax mrl_block{
            node.mi_row, node.mi_col, w >> 2, h >> 2};
        const mrl_stream::MrlDecision mrl_decision{
            static_cast<int>(node.mrl_index), node.mrl_sec_index != 0};
        const mrl_stream::MrlSyntaxStatus status = mrl_stream::write_mrl_syntax(
            enc_, cdfs_.mrl, mrl_context_, mrl_block, node.y_mode, mrl_decision,
            &error_);
        if (status != mrl_stream::MrlSyntaxStatus::kOk) {
          error = "mrl syntax: " +
                  std::string(mrl_stream::mrl_syntax_status_name(status));
          if (!error_.empty()) error += ": " + error_;
          return false;
        }
      } else if (node.mrl_index != 0 || node.mrl_sec_index != 0) {
        error = "MRL decisions supplied but the sequence header disables MRL";
        return false;
      }
    }
    if (node_has_chroma && !code_uv_mode(node, cw, ch)) {
      error = error_;
      return false;
    }
    if (!chroma_part && !write_palette(node, w, h, error)) return false;
    if (!chroma_part) {
      if (!code_luma_residual(node, w, h, levels_end, false, level_base,
                              error)) {
        return false;
      }
      level_base += luma_level_count(node);
    }
    if (!node_has_chroma) {
      if (error.empty() && !error_.empty()) error = error_;
      return true;
    }
    const TxView u_view{node.tx_size[1], node.tx_type[1],
                        static_cast<int>(node.eob[1]), chroma_mi_row,
                        chroma_mi_col};
    if (!code_tx_view(node, 1, u_view, level_base, cw, ch, levels_end, false,
                      error)) {
      return false;
    }
    level_base += node.eob[1];
    const TxView v_view{node.tx_size[2], node.tx_type[2],
                        static_cast<int>(node.eob[2]), chroma_mi_row,
                        chroma_mi_col};
    if (!code_tx_view(node, 2, v_view, level_base, cw, ch, levels_end, false,
                      error)) {
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
    const int plane_count = has_chroma_ ? kPlaneCount : 1;
    for (int plane = plan_.tx_mode_select ? 1 : 0; plane < plane_count;
         ++plane) {
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
    if (!code_luma_residual(node, w, h, levels_end, true, level_base, error)) {
      return false;
    }
    if (!has_chroma_) {
      if (error.empty() && !error_.empty()) error = error_;
      return true;
    }
    level_base += luma_level_count(node);
    const TxView u_view{node.tx_size[1], node.tx_type[1],
                        static_cast<int>(node.eob[1]), node.mi_row,
                        node.mi_col};
    if (!code_tx_view(node, 1, u_view, level_base, cw, ch, levels_end, true,
                      error)) {
      return false;
    }
    level_base += node.eob[1];
    const TxView v_view{node.tx_size[2], node.tx_type[2],
                        static_cast<int>(node.eob[2]), node.mi_row,
                        node.mi_col};
    if (!code_tx_view(node, 2, v_view, level_base, cw, ch, levels_end, true,
                      error)) {
      return false;
    }
    if (error.empty() && !error_.empty()) error = error_;
    return true;
  }


  // ---------------------------------------------------------------------
  // Coding a leaf block
  // ---------------------------------------------------------------------
  // §5.20.10.1 read_cdef().  A no-op for CdefStrengths == 1, for a 64x64
  // whose origin was already coded, and for a skipped inter block when
  // cdef_on_skip_txfm is off (the decoder returns before reading cdef_idx).
  // §5.20.10.4 read_lr(): once per superblock, before read_partition().  The
  // unit grid is at least one superblock, so only a superblock origin starts
  // a restoration unit.
  bool write_lr_superblock(const BlockSyntax& node, std::string& error) {
    if (filters_ == nullptr || !filters_->lr.enabled) return true;
    if (filters_->lr.pc_wiener) {
      // Luma PC-Wiener units first (plane order 0,1,2 in read_lr()).
      if (!pc_ready_) return error.empty() ? true : false;
      namespace pcw = ::av2enc::filters::pc_wiener_stream;
      const post::VideoLrPlane& cfg = filters_->lr.plane[0];
      if (cfg.enabled) {
        const int unit_size = pc_state_.unit_size[0];
        const int origin_x = node.mi_col * 4;
        const int origin_y = node.mi_row * 4;
        if (unit_size > 0 && origin_x % unit_size == 0 &&
            origin_y % unit_size == 0) {
          const int unit_row = origin_y / unit_size;
          const int unit_col = origin_x / unit_size;
          if (unit_row < cfg.unit_rows && unit_col < cfg.unit_cols) {
            const std::size_t index =
                static_cast<std::size_t>(unit_row) *
                    static_cast<std::size_t>(cfg.unit_cols) +
                static_cast<std::size_t>(unit_col);
            if (index >= cfg.unit_type.size()) {
              error = "PC-Wiener unit decision vector is too small";
              return false;
            }
            pcw::UnitDecision decision;
            decision.type = cfg.unit_type[index];
            if (!pcw::WriteUnit(enc_, pc_state_, 0, unit_row, unit_col,
                                decision, &error)) {
              return false;
            }
          }
        }
      }
      if (lr_chroma_ready_) {
        std::vector<filters_stream::LrUnitDecision> decisions;
        if (!filters_stream::lr_build_decisions(
                lr_state_, node.mi_row, node.mi_col, /*w4=*/16, /*h4=*/16,
                /*tile_mi_row_start=*/0, /*tile_mi_col_start=*/0, &decisions,
                &error)) {
          return false;
        }
        if (!filters_stream::lr_write_block(
                enc_, lr_state_, node.mi_row, node.mi_col, /*w4=*/16,
                /*h4=*/16, /*tile_mi_row_start=*/0, /*tile_mi_col_start=*/0,
                decisions, &error)) {
          return false;
        }
      }
      return true;
    }
    std::vector<filters_stream::LrUnitDecision> decisions;
    if (!filters_stream::lr_build_decisions(lr_state_, node.mi_row,
                                            node.mi_col, /*w4=*/16, /*h4=*/16,
                                            /*tile_mi_row_start=*/0,
                                            /*tile_mi_col_start=*/0,
                                            &decisions, &error)) {
      return false;
    }
    if (!filters_stream::lr_write_block(
            enc_, lr_state_, node.mi_row, node.mi_col, /*w4=*/16, /*h4=*/16,
            /*tile_mi_row_start=*/0, /*tile_mi_col_start=*/0, decisions,
            &error)) {
      return false;
    }
    return true;
  }

  bool write_cdef_index(const BlockSyntax& node, std::string& error) {
    if (filters_ == nullptr || !filters_->cdef.enabled) return true;
    if (plan_.inter_frame && node.is_inter && node.skip_flag != 0 &&
        !filters_->cdef.on_skip_txfm) {
      return true;
    }
    const int br = (node.mi_row + tile_mi_row_) >> 4;
    const int bc = (node.mi_col + tile_mi_col_) >> 4;
    if (br < 0 || bc < 0 || br >= filters_->cdef.n64_rows ||
        bc >= filters_->cdef.n64_cols) {
      error = "cdef_idx grid does not cover a coding block";
      return false;
    }
    const int idx = filters_->cdef.index[static_cast<std::size_t>(br) *
                                            static_cast<std::size_t>(
                                                filters_->cdef.n64_cols) +
                                        static_cast<std::size_t>(bc)];
    if (!filters_stream::cdef_write_block(
            enc_, cdef_state_, /*tile_mi_col_start=*/0,
            /*tile_mi_row_start=*/0, node.mi_row, node.mi_col, idx, &error)) {
      return false;
    }
    return true;
  }

  // §5.20.10.2 read_ccso().  Called for every leaf; the shared writer is a
  // no-op unless the leaf origin is a CCSO unit origin.
  bool write_ccso_index(const BlockSyntax& node, std::string& error) {
    if (filters_ == nullptr || !filters_->ccso.enabled) return true;
    int blk[3] = {0, 0, 0};
    const int shift = filters_->ccso.ccso_luma_size_log2 - 2;
    for (int plane = 0; plane < 3; ++plane) {
      const post::VideoCcsoPlane& src = filters_->ccso.plane[plane];
      if (!src.enabled || shift < 0) continue;
      const int row = (node.mi_row + tile_mi_row_) >> shift;
      const int col = (node.mi_col + tile_mi_col_) >> shift;
      if (row < 0 || col < 0 || row >= src.blk_rows || col >= src.blk_cols) {
        continue;
      }
      blk[plane] = src.blks[static_cast<std::size_t>(row) *
                                static_cast<std::size_t>(src.blk_cols) +
                            static_cast<std::size_t>(col)];
    }
    if (!filters_stream::ccso_write_block(enc_, ccso_state_, node.mi_row,
                                          node.mi_col, blk, &error)) {
      return false;
    }
    return true;
  }

  bool write_block(const BlockSyntax& node, std::string& error) {
#if defined(AV2ENC_VIDEO_DELTAQ)
    // Mirrors src/codec/entropy/deltaq/video/video_deltaq.cpp::
    // write_superblock_delta_q(): verify the planned tile state before the
    // symbol, emit §5.20.5.11 (or no symbol for a full-SB skip), then verify
    // the achieved qindex.  §5.20.5.11 codes nothing when
    // MiSize == SbSize && skip_flag, so such a superblock can only realize a
    // zero step; the planner must have produced one.
    const auto write_delta_step = [this](const rc::DeltaQPlanStep& step,
                                         const BlockSyntax& leaf,
                                         bool skip_flag,
                                         std::string& step_error) -> bool {
      std::string dq_error;
      if (!av2enc::deltaq::check_plan_step(dqp_.tile_qp, step, &dq_error)) {
        step_error = "delta-Q plan: " + dq_error;
        return false;
      }
      if (!video::deltaq::write_delta_qindex(dqp_, enc_, leaf, skip_flag,
                                             step.reduced_delta, &dq_error)) {
        step_error = "delta_qindex: " + dq_error;
        return false;
      }
      if (dqp_.tile_qp.current_q_index != step.qindex_after) {
        step_error =
            "delta_qindex: superblock (" + std::to_string(step.sb_y) + "," +
            std::to_string(step.sb_x) + ") did not reach the planned qindex " +
            std::to_string(step.qindex_after) + " (got " +
            std::to_string(dqp_.tile_qp.current_q_index) +
            "); a full-SB skip cannot code a delta";
        return false;
      }
      return true;
    };
#endif
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
    // §6.19.3.1 TreeType.  A CHROMA_PART leaf codes no luma mode, TX size,
    // CDEF/CCSO/delta-q or luma residual; a LUMA_PART leaf codes no chroma.
    const int tree_type = node.tree_type;
    const bool chroma_part = tree_type == 2;
    const bool chroma_info = (node.flags & kFlagHasChromaInfo) != 0;
    const bool node_has_chroma = !chroma_info || node.has_chroma != 0;
    if (chroma_part && !node_has_chroma) {
      // §5.20.3.1: a chroma-tree leaf with HasChroma == 0 codes nothing.
      return true;
    }
    // §5.20.4.1 decode_block(): update_ibc_buffers() runs at the start of the
    // block, before mode_info() builds the BV stack.
    const bool is_ibc_block = (node.flags & kFlagIsIbc) != 0;
    if (plan_.allow_intrabc) {
      ibcs::begin_ibc_block(ibc_buffer_, r, c);
    }

    int cw = 0;
    int ch = 0;
    const auto chroma_size = chroma_info
                                 ? fmt::chroma_residual_size(
                                       partition::block_width(
                                           static_cast<BlockSize>(
                                               node.chroma_bsize)),
                                       partition::block_height(
                                           static_cast<BlockSize>(
                                               node.chroma_bsize)),
                                       chroma_)
                                 : fmt::chroma_residual_size(w, h, chroma_);
    if (node_has_chroma) {      // §5.20.7.26 get_plane_residual_size() via the Subsampled_Size table.
      // `fmt::chroma_residual_size()` models 4:2:0/4:2:2/4:4:4; the 4:2:0
      // values are the historical `block_width/2` pair.  For an SDP chroma
      // leaf the owned ChromaMi block, not the luma leaf, selects the size.
      if (!chroma_size.has_value()) {
        error = "invalid chroma residual size";
        return false;
      }
      cw = fmt::block_width(*chroma_size);
      ch = fmt::block_height(*chroma_size);
    }
    cur_chroma_r_ =
        chroma_info ? (node.chroma_mi_row >> 1) : (r >> sub_y_);
    cur_chroma_c_ =
        chroma_info ? (node.chroma_mi_col >> 1) : (c >> sub_x_);

#if defined(AV2ENC_VIDEO_DELTAQ)
    // §5.20.5.7/.8 / §5.20.7.2 / §5.20.5.11 ordering state for this leaf.
    // `first_leaf` is set at each superblock root: only that leaf calls
    // read_delta_qindex(), and read_delta_qindex() clears ReadDeltas whether
    // or not it coded a symbol.
    const bool dqp = dqp_active_;
    const bool seg_pre_skip = dqp && dqp_.segmentation.seg_id_pre_skip;
    const int segment_id = dqp ? static_cast<int>(node.segment_id) : 0;
    const rc::DeltaQPlanStep* delta_step = sb_first_leaf_ ? sb_step_ : nullptr;
    sb_first_leaf_ = false;
    std::string dq_error;
#endif

    if (plan_.inter_frame) {
#if defined(AV2ENC_VIDEO_DELTAQ)
      if (dqp && seg_pre_skip &&
          !video::deltaq::write_inter_segment_id_pre(
              dqp_, enc_, node, segment_id, /*skip_flag=*/false, &dq_error)) {
        error = "inter_segment_id(preSkip): " + dq_error;
        return false;
      }
#endif
      if (!code_is_inter(node.is_inter != 0)) {
        error = error_;
        return false;
      }
      if (node.is_inter) {
        if (!code_skip_flag(node.skip_flag != 0)) {
          error = error_;
          return false;
        }
#if defined(AV2ENC_VIDEO_DELTAQ)
        if (dqp && !seg_pre_skip &&
            !video::deltaq::write_inter_segment_id_post(
                dqp_, enc_, node, segment_id, node.skip_flag != 0,
                &dq_error)) {
          error = "inter_segment_id: " + dq_error;
          return false;
        }
#endif
        if (!write_cdef_index(node, error)) return false;
        if (!write_ccso_index(node, error)) return false;
#if defined(AV2ENC_VIDEO_DELTAQ)
        if (dqp && delta_step != nullptr &&
            !write_delta_step(*delta_step, node, node.skip_flag != 0, error)) {
          return false;
        }
#endif
        if (!write_inter_mode_info(node, error)) return false;
        if (node.skip_flag) {
          reset_coeff_contexts(r, c, w4, h4);
        } else if (!write_inter_residual(node, w, h, cw, ch, error)) {
          return false;
        }
      } else {
#if defined(AV2ENC_VIDEO_DELTAQ)
        if (dqp && !seg_pre_skip &&
            !video::deltaq::write_inter_segment_id_post(
                dqp_, enc_, node, segment_id, /*skip_flag=*/false, &dq_error)) {
          error = "inter_segment_id: " + dq_error;
          return false;
        }
#endif
        if (!write_cdef_index(node, error)) return false;
        if (!write_ccso_index(node, error)) return false;
#if defined(AV2ENC_VIDEO_DELTAQ)
        if (dqp && delta_step != nullptr &&
            !write_delta_step(*delta_step, node, /*skip_flag=*/false, error)) {
          return false;
        }
#endif
        if (!write_intra_leaf(node, w, h, cw, ch, error)) return false;
      }
    } else {
      // Intra keyframe: intra_frame_mode_info() does not code is_inter.
      const bool is_ibc = (node.flags & kFlagIsIbc) != 0;
      if (node.is_inter) {
        error = "keyframe block marked inter";
        return false;
      }
#if defined(AV2ENC_VIDEO_DELTAQ)
      // §5.20.5.3: intra_segment_id() precedes the keyframe mode info.
      if (dqp && !chroma_part &&
          !video::deltaq::write_intra_segment_id(
              dqp_, enc_, node, segment_id, /*skip_flag=*/false,
              &dq_error)) {
        error = "intra_segment_id: " + dq_error;
        return false;
      }
#endif
      if (!chroma_part &&
          !write_keyframe_mode_info(node, is_ibc, error)) {
        return false;
      }
      // §5.20.10.1 read_cdef(): after the skip group, before the modes.  An
      // IBC skip block returns before cdef_idx unless CDEF filters skip blocks.
      // §5.20.10.2 read_ccso() is a separate reader with no such early return,
      // so the CCSO unit symbol is always coded here.  A CHROMA_PART leaf
      // codes no CDEF/CCSO.
      const bool cdef_skip =
          chroma_part ||
          (is_ibc && node.ibc_skip != 0 && filters_ != nullptr &&
           filters_->cdef.enabled && !filters_->cdef.on_skip_txfm);
      if (!cdef_skip) {
        if (!write_cdef_index(node, error)) return false;
      }
      if (!write_ccso_index(node, error)) return false;
#if defined(AV2ENC_VIDEO_DELTAQ)
      if (dqp && !chroma_part && delta_step != nullptr &&
          !write_delta_step(*delta_step, node, /*skip_flag=*/false, error)) {
        return false;
      }
#endif
      if (chroma_part) {
        if (!write_intra_leaf(node, w, h, cw, ch, error)) return false;
      } else if (is_ibc) {
        if (!write_ibc_body(node, error)) return false;
      } else if (!write_intra_leaf(node, w, h, cw, ch, error)) {
        return false;
      }
    }

#if defined(AV2ENC_VIDEO_DELTAQ)
    // §5.20.5.7: record SegmentIds for the following leaves' prediction.
    if (dqp &&
        !video::deltaq::commit_segment_block(dqp_, node, segment_id,
                                             &dq_error)) {
      error = "commit_segment_block: " + dq_error;
      return false;
    }
#endif

    // Update partition context.  A CHROMA_PART leaf updates only the chroma
    // plane grid; a LUMA_PART/shared leaf updates the luma grid and, when it
    // owns chroma, the chroma grid.
    if (!chroma_part) {
      for (int yy = 0; yy < h4; ++yy) {
        left_mi(r + yy) = kMiHeightLog2[static_cast<int>(size)];
      }
      for (int xx = 0; xx < w4; ++xx) {
        above_mi(c + xx) = kMiWidthLog2[static_cast<int>(size)];
      }
    }

    if (!chroma_part) update_mode_grids(node);

    if (!chroma_part) {
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
    }

    if (!node_has_chroma) return true;
    const int cr = cur_chroma_r_;
    const int cc0 = cur_chroma_c_;
    const int cw4 = cw / 4;
    const int ch4 = ch / 4;
    const int chroma_rows = chroma_mi_rows();
    const int chroma_cols = chroma_mi_cols();
    const bool is_cfl = !node.is_inter && (node.flags & kFlagIsCfl) != 0;
    for (int yy = 0; yy < ch4; ++yy) {
      const int rr = cr + yy;
      if (rr >= chroma_rows) break;
      for (int xx = 0; xx < cw4; ++xx) {
        const int cc = cc0 + xx;
        if (cc >= chroma_cols) break;
        uvcfl(rr, cc) = is_cfl ? 1 : 0;
      }
    }
    if (plan_.allow_intrabc) {
      // §5.20.4.1: MorphPreds = use_intrabc && morph_pred and UseIntrabc over
      // every 4x4 of the decoded block.  The BV neighbour map/ring must be
      // advanced for every leaf, IBC or not.
      const std::uint8_t ibc_value = is_ibc_block ? 1 : 0;
      const std::uint8_t skip_value =
          (is_ibc_block && node.ibc_skip != 0) ? 1 : 0;
      const std::uint8_t morph_value =
          (is_ibc_block && node.morph_pred != 0) ? 1 : 0;
      for (int yy = 0; yy < h4; ++yy) {
        const int rr = r + yy;
        if (rr >= mi_rows_) break;
        for (int xx = 0; xx < w4; ++xx) {
          const int cc = c + xx;
          if (cc >= mi_cols_) break;
          const std::size_t idx = cell_index(rr, cc);
          ibc_grid_[idx] = ibc_value;
          ibc_skip_grid_[idx] = skip_value;
          if (plan_.morph_pred_present) morph_grid_[idx] = morph_value;
        }
      }
      ibcs::finish_ibc_block(
          ibc_buffer_, ibc_neighbours_, r, c, node.bsize, is_ibc_block,
          ibc::BlockVector{node.ibc_bv_row, node.ibc_bv_col});
    }
    return true;
  }

  // Members
  const FramePlan& plan_;
  std::span<const BlockSyntax> blocks_;
  std::span<const std::int32_t> levels_;
  const VideoConfig& config_;
  std::span<const std::uint16_t> cdf_seed_;
  std::vector<std::uint16_t>* cdf_final_ = nullptr;
  const post::VideoFilterPlan* filters_ = nullptr;
  filters_stream::CdefTileState cdef_state_{};
  filters_stream::CcsoTileState ccso_state_{};
  // Frame-level filter-plan geometry for a tile-local block list.
  int frame_mi_cols_ = 0;
  int frame_mi_rows_ = 0;
  int tile_mi_row_ = 0;
  int tile_mi_col_ = 0;
  filters_stream::LrState lr_state_{};
  ::av2enc::filters::pc_wiener_stream::TileState pc_state_{};
  bool pc_ready_ = false;
  bool lr_chroma_ready_ = false;
  FrameGeometry geometry_;
  fmt::ChromaFormat chroma_ = fmt::ChromaFormat::k420;
  fmt::BitDepth depth_ = fmt::BitDepth::k8;
  bool has_chroma_ = true;
  int sub_x_ = 1;
  int sub_y_ = 1;
  int mi_cols_ = 0;
  int mi_rows_ = 0;

#if defined(AV2ENC_VIDEO_DELTAQ)
  // §5.18.7.1/.8 tile-local segmentation + delta-Q state.  `dqp_active_` is
  // false on every default stream, so none of the symbols below are emitted.
  video::deltaq::FrameQpContext dqp_{};
  bool dqp_active_ = false;
  std::string dqp_init_error_;
  bool sb_first_leaf_ = false;
  const rc::DeltaQPlanStep* sb_step_ = nullptr;
#endif

  CdfState cdfs_{};
  // §5.20.5.5 UsesMrls grid for the current tile (decoder clears it per tile).
  mrl_stream::MrlContext mrl_context_;
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
  ps::TilePartitionCdfs ext_cdfs_{};
  pe::Tools tools_{};
#endif
  // §5.20.5.3/.4 IBC + §7.13.3.26 morph state (keyframe carrier).
  ibc::IbcBufferState ibc_buffer_;
  ibcs::BvNeighbourMap ibc_neighbours_;
  ibc::IbcConfig ibc_config_{};
  ibcs::IbcCdfs ibc_cdfs_{};
  ibcs::ScreenContentConfig ibc_content_{};
  std::vector<std::uint8_t> ibc_grid_;
  std::vector<std::uint8_t> ibc_skip_grid_;
  morph::MorphCdfs morph_cdfs_{};
  std::vector<std::uint8_t> morph_grid_;
  // §5.20.7.14-.15 inter-intra motion-mode CDFs.
  adv::AdvancedCdfState adv_cdfs_{};
  // §5.20.8.1 palette_mode_info() state: the video path never signals a
  // palette, but the has_palette_y=0 symbol is mandatory when screen tools are
  // on.
  std::optional<pals::PaletteSyntaxWriter> palette_writer_;
  palette::PaletteCdfState palette_cdfs_{};
  std::string palette_reset_error_;
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
  std::vector<std::uint8_t> comp_group_grid_;
  std::vector<inter::mc::MiInfo> mv_grid_;

  int cur_r_ = 0;
  int cur_c_ = 0;
  int cur_w4_ = 0;
  int cur_h4_ = 0;
  int cur_chroma_r_ = 0;
  int cur_chroma_c_ = 0;
  int eob_u_ = 0;
  bool trace_ = false;
};

}  // namespace

int intra_joint_mode(int y_mode, int angle_delta) {
  return y_mode_code(y_mode, angle_delta).joint;
}

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

bool assemble_video_frame(const FramePlan& plan,
                             std::span<const BlockSyntax> blocks,
                             std::span<const std::int32_t> levels,
                             const VideoConfig& config,
                             std::vector<std::uint8_t>& tile_payload,
                             std::string& error, std::size_t* tile_bytes,
                             std::span<const std::uint16_t> cdf_seed,
                             std::vector<std::uint16_t>* cdf_final,
                             const post::VideoFilterPlan* filters) {
  error.clear();
  tile_payload.clear();
  if (!cdf_seed.empty() && cdf_seed.size() != cdf_image_words_impl()) {
    error = "CDF seed has the wrong number of entries";
    return false;
  }
  if (plan.width <= 0 || plan.height <= 0) {
    error = "empty frame plan";
    return false;
  }
#if !defined(AV2ENC_VIDEO_EXT_PARTITIONS)
  if (config.enable_ext_partitions) {
    error = "extended partitions need the full build (partition_ext adapter)";
    return false;
  }
#endif
  if ((plan.width & 7) != 0 || (plan.height & 7) != 0) {
    error = "M2 baseline requires width and height divisible by 8";
    return false;
  }
  if (plan.sb_size != 64 && plan.sb_size != 128 && plan.sb_size != 256) {
    error = "M2 supports 64x64, 128x128 or 256x256 superblocks";
    return false;
  }
  if (config.max_frame_width != static_cast<std::uint16_t>(plan.width) ||
      config.max_frame_height != static_cast<std::uint16_t>(plan.height)) {
    error = "frame plan size does not match the sequence/frame header config";
    return false;
  }

  Writer writer(plan, blocks, levels, config, cdf_seed, cdf_final, filters);
  if (!writer.run(tile_bytes, error)) return false;
  tile_payload = writer.payload();
  return true;
}

std::size_t cdf_image_words() { return cdf_image_words_impl(); }

bool cdf_image_defaults_for_base_q_idx(int base_q_idx,
                                       std::vector<std::uint16_t>& out,
                                       std::string* error) {
  if (base_q_idx < 1 || base_q_idx > 255) {
    if (error != nullptr) {
      *error = "base_q_idx must be in [1, 255] for a CDF default image";
    }
    return false;
  }
  CdfState state{};
  init_cdfs(state, qctx::slice_for_base_q_idx(base_q_idx));
  out.resize(cdf_image_words_impl());
  export_cdf_image(state, out.data());
  return true;
}

bool cdf_image_tables(std::vector<std::uint16_t>& frame,
                      std::vector<std::uint16_t>& tile,
                      std::vector<std::uint16_t>& saved,
                      std::vector<std::uint16_t>& defaults,
                      std::vector<entropy::avg_cdf::CdfTable>& tables,
                      std::string* error) {
  const std::size_t words = cdf_image_words_impl();
  if (frame.size() != words || tile.size() != words ||
      saved.size() != words || defaults.size() != words) {
    if (error != nullptr) {
      *error = "CDF images must all have cdf_image_words() entries";
    }
    return false;
  }
  tables.clear();
  for (const CdfArrayDesc& desc : cdf_arrays()) {
    const std::size_t entries = static_cast<std::size_t>(desc.symbols) + 1U;
    if (entries == 0U || desc.words % entries != 0U) {
      if (error != nullptr) *error = "CDF array size is not a multiple of N+1";
      return false;
    }
    const std::size_t instances = desc.words / entries;
    for (std::size_t i = 0; i < instances; ++i) {
      entropy::avg_cdf::CdfTable table;
      table.symbols = desc.symbols;
      table.frame = frame.data() + desc.packed_offset + i * entries;
      table.tile = tile.data() + desc.packed_offset + i * entries;
      table.saved = saved.data() + desc.packed_offset + i * entries;
      table.defaults = defaults.data() + desc.packed_offset + i * entries;
      tables.push_back(table);
    }
  }
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

}  // namespace av2enc::video::syntax
