// Private to the video_syntax writer translation units.
//
// The M2 P-frame tile syntax writer is one state machine: a Writer that owns
// the tile range coder, the CDF state and every §8.3 context grid.  It is split
// across four TUs by syntax role:
//
//   video_syntax.cpp        entry point, per-frame/tile assembly, state setup
//   video_syntax_block.cpp  partition tree, mode info, intra leaf, filter units
//   video_syntax_inter.cpp  inter block layer (§5.20.7)
//   video_syntax_coeff.cpp  EOB/coefficients, TX type/partition (§5.20.6/.8)
//
// Nothing here is part of the encoder contract; include video_syntax.h instead.
#pragma once

#include "pipeline/video/video_syntax.h"

#include "codec/inter/mv_precision/mv_precision.h"
#include "codec/inter/drl_reorder/video_stack_adapter.h"
#include "codec/filters/stream/filters_stream.h"
#include "codec/filters/pc_wiener/pc_wiener_stream.h"
#include "cuda/pipeline/still/cdef/still_cdef_cdf.h"
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
#include "codec/intra/dip/dip_syntax.h"
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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace av2enc::video::syntax {
// The writer internals keep the single-TU structure they had before the
// split; `detail` replaces the file-local anonymous namespace so the four
// TUs share one definition of the class and its helpers.
namespace detail {
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

inline int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}
inline int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

inline bool y_mode_is_directional(int mode) { return mode >= kV && mode <= kD67; }

// §9.2 Fsc_Bsize_Groups[BLOCK_SIZES] (same table as the still/lossless writer).
constexpr int kFscBsizeGroups[29] = {
    0, 1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 3, 3, 4, 4, 6, 6, 4, 4, 6, 6};

// Reordered index of a YMode (0..12).
inline int reordered_index(int mode) {
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

inline YModeCode y_mode_code(int mode, int angle_delta) {
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

inline int uv_mode_index(int uv_mode, int y_mode) {
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

inline int tx_sz_ctx(int tx_size) {
  return (kTxSizeSqr[tx_size] + kTxSizeSqrUp[tx_size] + 1) >> 1;
}

// Transform-type syntax class for an intra luma TX block.  Mirrors
// get_tx_set() for reduced_tx_set = 0 and enable_idtx_intra = 0.
enum class TxTypeSyntax { kNone, kIntraSet1, kLong32, kLong64 };
inline TxTypeSyntax luma_tx_type_syntax(int tx_size, int eob) {
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
inline int inter_tx_type_context(int eob, int coded_w, int coded_h, bool use_min) {
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
  // §5.20.3.2 extended-partition rows (§9.3).  They adapt like every other
  // tile CDF, so they belong in the §8.2 image whenever --video-ext-partitions
  // selects them.  With the tool off they never code a symbol and stay at
  // their defaults, so the image is unchanged.
  std::uint16_t do_ext_partition[2][64][3];
  std::uint16_t do_uneven_4way_partition[2][64][3];
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
  // §5.20.7.9 dip_mode_info() (--video-dip).
  ::av2enc::dip::DipCdfState dip_cdfs;
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
  std::uint16_t comp_ref0[3][6][3];
  std::uint16_t comp_ref1[3][2][6][3];
  std::uint16_t is_joint[2][3];
  std::uint16_t compound_mode_non_joint[5][6];
  // §5.4.6 num_same_ref_compound > 0 (--video-same-ref-compound).
  std::uint16_t compound_mode_same_refs[5][5];
  // §5.20.7.6 use_optflow (REFINE_SWITCHABLE only; ctx = YMode != NEAR_NEARMV).
  std::uint16_t use_optflow[2][3];
  // §5.20.7.16 masked compound (enable_masked_compound = 1).
  std::uint16_t comp_group_idx[12][3];
  std::uint16_t compound_type[3];
  // §5.20.7.6 cwp_idx (enable_cwp = 1; NEAR_NEARMV + COMPOUND_AVERAGE).
  std::uint16_t cwp_idx[4][3];
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

  // §6.17.2 tile CDFs of the §5.20.10 filter symbols.  They adapt per tile
  // like every other §8.2 CDF, so they must ride the §8.2.4 / §7.5 image:
  // omitting them desynchronised the first frame that carries CDFs whenever
  // CDEF or CCSO coded an index symbol.
  //   cdef_index0   §8.3 cdef_index0, 4 contexts x 2 symbols.
  //   cdef_minus1_* §8.3 cdef_idx_minus_1 for CdefStrengths 3..8; each is a
  //                 separate §9.3 array with its own length (N-1 symbols).
  //   ccso_index    §8.3 ccso_blk, 3 planes x 4 contexts x 2 symbols.
  std::uint16_t cdef_index0[4][3];
  std::uint16_t cdef_minus1_3[3];
  std::uint16_t cdef_minus1_4[4];
  std::uint16_t cdef_minus1_5[5];
  std::uint16_t cdef_minus1_6[6];
  std::uint16_t cdef_minus1_7[7];
  std::uint16_t cdef_minus1_8[8];
  std::uint16_t ccso_index[3][4][3];
};

template <typename T, std::size_t N>
inline void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

// `qctx_slice` is §8.3.1 init_coeff_cdfs()'s COEFF_CDF_Q_CTXS slice, selected
// from the raw frame base_q_idx (0..3).  Slice 0 is byte-identical to the
// historical pinned tables.
inline void init_cdfs(CdfState& c, int qctx_slice,
               bool reduced_tx_part_set = false) {
  const qctx::CoeffCdfSlice& q = qctx::tables_for_slice(qctx_slice);
  copy_array(c.do_split, av2enc::basic::kDoSplitCdf);
  copy_array(c.do_square_split, av2enc::basic::kDoSquareSplitCdf);
  copy_array(c.rect_type, av2enc::basic::kRectTypeCdf);
  // §5.20.3.2 extended-partition rows.  The module is always compiled, so the
  // §9.3 defaults are available even in builds without the video carrier.
  copy_array(c.do_ext_partition, av2enc::partition_ext::kDoExtPartitionCdf);
  copy_array(c.do_uneven_4way_partition,
             av2enc::partition_ext::kDoUneven4wayPartitionCdf);
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
  c.dip_cdfs.reset();
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
  copy_array(c.comp_ref0, multiref::gate::kCompRef0Cdf);
  copy_array(c.comp_ref1, multiref::gate::kCompRef1Cdf);
  copy_array(c.is_joint, multiref::gate::kIsJointCdf);
  copy_array(c.compound_mode_non_joint,
             multiref::gate::kCompoundModeNonJointCdf);
  copy_array(c.compound_mode_same_refs,
             multiref::gate::kCompoundModeSameRefsCdf);
  // §9.3 TileUseOptflowCdf[2] defaults (not generated into
  // video_cdf_tables.h; the compound component pins the same values).
  static constexpr std::uint16_t kUseOptflowCdf[2][3] = {
      11582, 31, 0, 24076, 26, 0,
  };
  copy_array(c.use_optflow, kUseOptflowCdf);
  copy_array(c.comp_group_idx, multiref::gate::kCompGroupIdxCdf);
  copy_array(c.compound_type, multiref::gate::kCompoundTypeCdf);
  copy_array(c.cwp_idx, multiref::gate::kCwpIdxCdf);
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
  // §6.17.2: the filter tile CDFs reset with everything else.
  copy_array(c.cdef_index0, av2enc::cuda::still::kCdefIndex0Cdf);
  copy_array(c.cdef_minus1_3,
             av2enc::cuda::still::kCdefIndexMinus1With3Cdf);
  copy_array(c.cdef_minus1_4,
             av2enc::cuda::still::kCdefIndexMinus1With4Cdf);
  copy_array(c.cdef_minus1_5,
             av2enc::cuda::still::kCdefIndexMinus1With5Cdf);
  copy_array(c.cdef_minus1_6,
             av2enc::cuda::still::kCdefIndexMinus1With6Cdf);
  copy_array(c.cdef_minus1_7,
             av2enc::cuda::still::kCdefIndexMinus1With7Cdf);
  copy_array(c.cdef_minus1_8,
             av2enc::cuda::still::kCdefIndexMinus1With8Cdf);
  copy_array(c.ccso_index, still::filters_stream::kCcsoBlkCdf);
}

// §9.3 `Default_Cdef_Index_Minus1_With{N}_Cdf` slot of the frame image for a
// given CdefStrengths.  §5.20.10.1 codes no minus-1 symbol below 3 strengths.
inline std::uint16_t* cdef_minus1_slot(CdfState& c, int strengths) {
  switch (strengths) {
    case 3:
      return c.cdef_minus1_3;
    case 4:
      return c.cdef_minus1_4;
    case 5:
      return c.cdef_minus1_5;
    case 6:
      return c.cdef_minus1_6;
    case 7:
      return c.cdef_minus1_7;
    case 8:
      return c.cdef_minus1_8;
    default:
      return nullptr;
  }
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

inline const std::vector<CdfArrayDesc>& cdf_arrays() {
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
    // §5.20.3.2 extended partition rows (--video-ext-partitions).  They adapt
    // per tile like the rect rows; without them in the §8.2 image the carried
    // CDF state of the ext writer would differ from the decoder's from the
    // second frame on.
    AV2ENC_ADD_CDF(do_ext_partition, 2);
    AV2ENC_ADD_CDF(do_uneven_4way_partition, 2);
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
    add(probe.dip_cdfs.use_dip.data(), sizeof(probe.dip_cdfs.use_dip), 2);
    add(probe.dip_cdfs.dip_mode.data(), sizeof(probe.dip_cdfs.dip_mode), 6);
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
    // §6.17.2 filter tile CDFs (§5.20.10.1 read_cdef / §5.20.10.2 read_ccso).
    // A frame carries its §8.2 CDF image, so every symbol these tools code
    // adapts across frames like the block layer.
    AV2ENC_ADD_CDF(cdef_index0, 2);
    add(probe.cdef_minus1_3, sizeof(probe.cdef_minus1_3), 2);
    add(probe.cdef_minus1_4, sizeof(probe.cdef_minus1_4), 3);
    add(probe.cdef_minus1_5, sizeof(probe.cdef_minus1_5), 4);
    add(probe.cdef_minus1_6, sizeof(probe.cdef_minus1_6), 5);
    add(probe.cdef_minus1_7, sizeof(probe.cdef_minus1_7), 6);
    add(probe.cdef_minus1_8, sizeof(probe.cdef_minus1_8), 7);
    AV2ENC_ADD_CDF(ccso_index, 2);
#undef AV2ENC_ADD_CDF
    return out;
  }();
  return arrays;
}

inline std::size_t cdf_image_words_impl() {
  const std::vector<CdfArrayDesc>& arrays = cdf_arrays();
  if (arrays.empty()) return 0;
  return arrays.back().packed_offset + arrays.back().words;
}

inline void export_cdf_image(const CdfState& state, std::uint16_t* out) {
  const auto* base = reinterpret_cast<const std::uint8_t*>(&state);
  for (const CdfArrayDesc& desc : cdf_arrays()) {
    std::memcpy(out + desc.packed_offset, base + desc.struct_offset,
                desc.words * sizeof(std::uint16_t));
  }
}

inline void import_cdf_image(CdfState& state, const std::uint16_t* in) {
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

// ---------------------------------------------------------------------------
// Tile symbol cost meter (AV2ENC_SYMBOL_COST=1).
//
// Diagnostic only: it never touches the bitstream.  Every tile symbol is
// attributed to a category, either by the CDF array `sym()` was handed or, for
// the §5.20.10 filter writers (which call the entropy encoder directly), by the
// bit-prefix delta across the call.  The table shows what a frame still pays
// symbol group by symbol group; on a fully-skipped picture it is the floor.
// ---------------------------------------------------------------------------
enum CostTag : int {
  kCostPartition = 0,
  kCostIsInter,
  kCostSkip,
  kCostInterMode,
  kCostResidual,
  kCostCdef,
  kCostCcso,
  kCostDeltaQ,
  kCostIntra,
  kCostOther,
  kCostTagCount
};

struct CostMeter {
  std::uint64_t count[kCostTagCount] = {};
  double bits[kCostTagCount] = {};
  bool active = false;

  void add(int tag, double value) {
    ++count[tag];
    bits[tag] += value;
  }
  void add_prefix(int tag, std::uint64_t before, std::uint64_t after) {
    ++count[tag];
    bits[tag] += static_cast<double>(after - before);
  }
  double total_bits() const {
    double sum = 0.0;
    for (double b : bits) sum += b;
    return sum;
  }
};

inline const char* cost_tag_name(int tag) {
  switch (tag) {
    case kCostPartition: return "partition";
    case kCostIsInter: return "is_inter";
    case kCostSkip: return "skip";
    case kCostInterMode: return "inter_mode";
    case kCostResidual: return "residual";
    case kCostCdef: return "cdef_index";
    case kCostCcso: return "ccso_blk";
    case kCostDeltaQ: return "delta_q";
    case kCostIntra: return "intra_mode";
    default: return "other";
  }
}

inline int enumerate_slots(int mi_row, int mi_col, BlockSize size,
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
    geometry_.max_pb_aspect_ratio = plan.max_pb_aspect_ratio;
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
    if (config_.enable_dip) {
      usedip_.assign(static_cast<std::size_t>(mi_rows_) *
                         static_cast<std::size_t>(mi_cols_),
                     0);
    }
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
    ps::init_tile_partition_cdfs(ext_cdfs_);
    tools_.enable_ext_partitions = config_.enable_ext_partitions;
    tools_.enable_uneven_4way_partitions =
        config_.enable_uneven_4way_partitions;
#endif
    if (!cdf_seed_.empty()) {
      import_cdf_image(cdfs_, cdf_seed_.data());
    }
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
    // §8.2.2 tile CDF seed: the extended-partition rows live in the same image
    // as the rect rows, so the reused still-stream symbol planner starts from
    // the carried state (and --cdf-continuity holds across frames).  Only the
    // ext writer owns those rows; the rect writer adapts cdfs_ directly.
    if (tools_.enable_ext_partitions) import_ext_cdfs_from_image();
#endif
    tile_buffer_.resize(kTileCapacityBytes);
    enc_.init_symbol(tile_buffer_.data(), tile_buffer_.size());
    enc_.disable_cdf_update = config_.disable_cdf_update ? 1 : 0;
    enc_.minimal_trailer = config_.minimal_tile_trailer ? 1 : 0;

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
    prec_grid_.assign(mi_cells, inter::mvprec::NeighborMvInfo{});
    for (std::size_t i = 0; i < mi_cells; ++i) {
      prec_grid_[i].available = false;
      prec_grid_[i].ref[0] = merge::kRefNone;
      prec_grid_[i].ref[1] = merge::kRefNone;
    }
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
      qp_plan.enable_ext_seg = plan_.enable_ext_seg;
      qp_plan.primary_ref_is_none = plan_.primary_ref_is_none;
      VideoFrameHeader frame_header;
      frame_header.keyframe = !plan_.inter_frame;
      std::string qp_error;
      if (!video::deltaq::begin_frame(dqp_, qp_plan, frame_header, mi_cols_,
                                      mi_rows_, plan_.prev_segment_ids,
                                      &qp_error)) {
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
    cost_meter_.active = std::getenv("AV2ENC_SYMBOL_COST") != nullptr;
  }

  bool run(std::size_t* tile_bytes, std::string& error);

  const std::vector<std::uint8_t>& payload() const { return payload_; }

 private:
  // ---------------------------------------------------------------------
  // Entropy helpers
  // ---------------------------------------------------------------------
  // §8.2 symbol information cost in bits.  AV2 tile CDFs store forward
  // cumulative boundaries cdf[0..n-2] plus the adaptation parameters, so the
  // probability of symbol s is (cdf[s] - cdf[s-1]) / 32768 with cdf[-1] = 0
  // and cdf[n-1] = 32768 implicit.  Diagnostic only (AV2ENC_SYMBOL_COST); it
  // never feeds the bitstream.
  static double symbol_cost_bits(const std::uint16_t* cdf, int n, int symbol) {
    if (n < 2 || symbol < 0 || symbol >= n) return 0.0;
    const int lo = symbol == 0 ? 0 : static_cast<int>(cdf[symbol - 1]);
    const int hi = symbol == n - 1 ? static_cast<int>(av2::kCdfScale)
                                   : static_cast<int>(cdf[symbol]);
    const int freq = hi - lo;
    if (freq <= 0) return 0.0;
    return std::log2(static_cast<double>(av2::kCdfScale) /
                     static_cast<double>(freq));
  }

  int cost_tag_for(const std::uint16_t* p) const;

  bool sym(std::uint16_t* cdf, int n, int symbol) {
    const double cost =
        cost_meter_.active ? symbol_cost_bits(cdf, n, symbol) : 0.0;
    const int tag = cost_meter_.active ? cost_tag_for(cdf) : kCostOther;
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
    if (cost_meter_.active) cost_meter_.add(tag, cost);
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
  bool code_do_split(PartitionType partition, int r, int c, BlockSize size);
  bool code_do_square_split(PartitionType partition, int r, int c,
                            BlockSize size);
  bool code_rect_type(PartitionType partition, int r, int c, BlockSize size);
  bool code_sdp_partition(const BlockSyntax& node, std::string& error);
  bool write_node_sdp(std::size_t index, std::size_t* next,
                      std::string& error);
  bool write_node(std::size_t index, std::size_t* next, std::string& error);
  void import_ext_cdfs_from_image();
  void export_ext_cdfs_to_image();
  bool write_node_ext(std::size_t index, std::size_t* next,
                      std::string& error);

  // ---------------------------------------------------------------------
  // Mode info syntax
  // ---------------------------------------------------------------------
  int joint_context() const;
  int y_mode_index_for(int joint, bool sub8_leaf) const;
  bool code_y_mode(const BlockSyntax& node);
  int cfl_context() const;
  bool code_uv_mode(const BlockSyntax& node, int cw, int ch);

  // ---------------------------------------------------------------------
  // Residual syntax
  // ---------------------------------------------------------------------
  bool write_eob(int eob, int plane, int coded_w, int coded_h, bool is_inter);
  bool write_quant_escape(int x, int& hr_level_avg);
  int dc_sign_context(int x4, int y4, int w4, int h4) const;
  bool write_inter_tx_type(int tx_size, int tx_type, int eob, int coded_w,
                           int coded_h);

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

  bool code_tx_view(const BlockSyntax& node, int plane, const TxView& view,
                    std::size_t level_base, int bsize_w, int bsize_h,
                    std::size_t levels_end, bool is_inter,
                    std::string& error);

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

  int neighbor_positions(int pos[2][2]) const;
  bool has_newmv_mode(int ymode) const;
  bool tip_mode_present() const;
  int tip_context() const;
  int new_mv_context() const;
  int is_inter_context() const;
  int skip_context() const;
  bool code_is_inter(bool value);
  bool code_skip_flag(bool value);
  bool code_single_mode(int ctx, int single_mode);
  bool code_bawp(const BlockSyntax& node);
  int count_refs(int ref) const;
  bool code_single_ref(int want);
  bool code_compound_ref(const BlockSyntax& node);
  bool code_drl_index(int ctx, int index, int start = 0);
  bool code_tip_drl_index(int index, int start = 0);
  bool write_ns_bits(std::uint32_t n, std::uint32_t value);
  bool write_mv(std::int32_t dy, std::int32_t dx);
  static bool is_inter_ref_value(int ref);
  bool ref_backward(int ref) const;
  int comp_mode_context() const;
  int is_joint_context(int ref0, int ref1) const;
  int furthest_future_ref() const;
  int comp_group_context(int ref0, int ref1) const;
  bool build_mode_stack(const BlockSyntax& node, int ref, bool is_compound,
                        const BlockSyntax& mv_source,
                        merge::StackParams& params,
                        merge::MvStack& stack) const;
  bool opfl_allowed_for_pair(int ref0, int ref1) const;
  bool refine_mv_derived(const BlockSyntax& node, bool* use) const;
  bool write_compound_mode_info(const BlockSyntax& node, std::string& error);
  bool write_tip_block_mode_info(const BlockSyntax& node, std::string& error);
  bool write_inter_mode_info(const BlockSyntax& node, std::string& error);
  inter::mc::Mv block_global_mv_for_ref(const BlockSyntax& node,
                                        int ref) const;
  inter::mc::Mv block_global_mv(const BlockSyntax& node) const;
  me2::FramePrecisionSyntax frame_mv_precision() const;
  int block_mv_precision(const BlockSyntax& node) const;
  inter::mvprec::NeighborSet mv_res_neighbours() const;
  void update_mode_grids(const BlockSyntax& node);

  void reset_coeff_contexts(int r, int c, int w4, int h4);

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

  int luma_level_count(const BlockSyntax& node) const;
  bool code_luma_residual(const BlockSyntax& node, int w, int h,
                          std::size_t levels_end, bool is_inter,
                          std::size_t level_base, std::string& error);

  // Intra mode + residual syntax. Used by keyframe blocks and by intra
  // fallback blocks inside an inter frame.
  // ---------------------------------------------------------------------
  // §5.20.5.3/.4 intra block copy + §5.20.5.4 morph_pred on an intra frame.
  // ---------------------------------------------------------------------
  ibc::FrameView ibc_frame_view() const;
  bool write_keyframe_mode_info(const BlockSyntax& node, bool is_ibc,
                                std::string& error);
  bool write_ibc_body(const BlockSyntax& node, std::string& error);
  bool write_palette(const BlockSyntax& node, int w, int h,
                     std::string& error);
  bool write_dip(const BlockSyntax& node, int w, int h, std::string& error);
  void update_usedip(const BlockSyntax& node, int w4, int h4);
  bool write_intra_leaf(const BlockSyntax& node, int w, int h, int cw, int ch,
                        std::string& error);

  bool write_inter_residual(const BlockSyntax& node, int w, int h, int cw,
                            int ch, std::string& error);

  // ---------------------------------------------------------------------
  // Coding a leaf block
  // ---------------------------------------------------------------------
  bool write_lr_superblock(const BlockSyntax& node, std::string& error);
  bool write_cdef_index(const BlockSyntax& node, std::string& error);
  bool write_ccso_index(const BlockSyntax& node, std::string& error);
  bool write_block(const BlockSyntax& node, std::string& error);

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
  // §8.3.1 UseDip flag per MI cell (--video-dip only).
  std::vector<std::uint8_t> usedip_;
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
  // §8.3 neighbour state for the MV-resolution contexts (use_amvd,
  // use_most_probable_precision, pb_mv_precision).  One entry per MI cell,
  // updated with the same block origin/extent as mv_grid_.
  std::vector<inter::mvprec::NeighborMvInfo> prec_grid_;

  int cur_r_ = 0;
  int cur_c_ = 0;
  int cur_w4_ = 0;
  int cur_h4_ = 0;
  int cur_chroma_r_ = 0;
  int cur_chroma_c_ = 0;
  int eob_u_ = 0;
  bool trace_ = false;
  CostMeter cost_meter_;
};

}  // namespace detail
}  // namespace av2enc::video::syntax
