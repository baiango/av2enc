// AV2 M1 still-picture syntax writer.  See still_syntax.h for the contract.
//
// Everything in the anonymous namespace below was extracted verbatim (modulo
// state ownership) from the inline writer in src/encoder/m1/still_encoder.cpp.  The
// symbol order, the CDF selection, the context derivation, and the raw-bit
// layout must not change: the CPU path is the byte-exact reference for the GPU
// path.
#include "pipeline/still/still_syntax.h"

#include "pipeline/still/basic_cdf_tables.h"
#include "pipeline/still/basic_tables.h"
#include "pipeline/still/still_cdf_tables.h"
#include "pipeline/still/still_tables.h"
#include "codec/entropy/qctx/qctx.h"
#include "codec/entropy/deltaq/still/deltaq.h"
#include "codec/entropy/lossless/lossless_cdf_tables.h"
#include "codec/intra/cfl/mhccp/mhccp.h"
#include "codec/intra/cfl/mhccp/mhccp_syntax.h"
#include "codec/intra/dip/dip_syntax.h"
#include "codec/intra/mrl/stream/mrl_interaction.h"
#include "codec/intra/mrl/stream/mrl_syntax.h"
#include "codec/intra/palette/core/palette.h"
#include "codec/intra/palette/stream/palette_stream.h"
#include "codec/intra/ibc/stream/ibc_stack.h"
#include "codec/intra/ibc/stream/ibc_syntax.h"
#include "codec/intra/ibc/stream/still_ibc_stream.h"
#include "codec/filters/loop_restoration/classes/wiener_syntax.h"
#include "codec/filters/pc_wiener/pc_wiener_stream.h"
#include "codec/partition/tree/partition.h"
#include "codec/partition/stream/partition_stream.h"
#include "codec/transform/tx_type/tx_type.h"
#include "codec/transform/secondary/tx_secondary.h"
#include "codec/transform/secondary/tx_secondary_tables.h"
#include "codec/transform/tcq/tcq.h"
#include "codec/transform/tcq/tcq_cdf_tables.h"
#include "codec/transform/tx_partition/tx_partition.h"
#include "codec/transform/variants/tx_variants.h"
#include "codec/entropy/av2/av2_entropy.h"

#include "cuda/pipeline/still/cdef/still_cdef_cdf.h"
#include "cuda/pipeline/still/cdef/still_cdef_index_syntax.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace av2enc::still::syntax {
namespace {

using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionType;
namespace ps = still::partition_stream;
namespace pe = partition_ext;
namespace ibcs = ibc_stream;

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

// Sub-8 leaves (MiSize < BLOCK_8X8) omit the neighbour-derived directional
// candidate from the UV list, so the index is the plain Default_Mode_List_Uv
// position (§5.20.5.6).
int uv_mode_index_sub8(int uv_mode) {
  int index = 0;
  for (int i = 0; i < kIntraModesCount; ++i) {
    const int mode = kDefaultModeListUv[i];
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

// §9.2 Fsc_Bsize_Groups[BLOCK_SIZES] (same table as the lossless writer).
constexpr int kFscBsizeGroups[29] = {
    0, 1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 3, 3, 4, 4, 6, 6, 4, 4, 6, 6};

// Default_Ccso_Blk_Cdf[3][CCSO_CONTEXT=4][3] (§9.3), copied verbatim.
constexpr std::uint16_t kCcsoBlkCdf[3][4][3] = {
    {{18469, 62, 0}, {16384, 0, 0}, {4949, 37, 0}, {16384, 0, 0}},
    {{23470, 37, 0}, {16384, 0, 0}, {6666, 37, 0}, {16384, 0, 0}},
    {{22914, 37, 0}, {16384, 0, 0}, {6993, 37, 0}, {16384, 0, 0}},
};

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
  std::uint16_t ccso_blk[3][4][3];
  std::uint16_t sec_tx_type[2][5][5];
  std::uint16_t most_probable_stx_set[8];
  std::uint16_t most_probable_stx_set_adst[5];
  // §5.20.7.27 cctx_type: Default_Cctx_Type_Cdf (7 symbols).
  std::uint16_t cctx_type[8];
  // §5.20.6.3 tx_partition CDFs (full or reduced set per the sequence bit).
  av2enc::video::txpartition::TxPartitionCdfs tx_partition;
  // §5.20.5.5 fsc_mode: Default_Fsc_Mode_Cdf[4][6][3].  This encoder always
  // codes 0, so the neighbour context is 0.
  std::uint16_t fsc_mode[4][6][3];
  mrl_stream::MrlCdfs mrl;
  mrl_stream::MrlContext mrl_context;
  dip::DipCdfState dip_cdfs;
  // §5.20.7.32 read_cfl_alphas() including cfl_mhccp/cfl_mh_dir (MHCCP).
  cfl::mhccp::CflCdfs mh_cdfs;
};

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

// `qctx_slice` is §8.3.1 init_coeff_cdfs()'s COEFF_CDF_Q_CTXS slice, selected
// from the raw frame base_q_idx (0..3).  Slice 0 reproduces the historical
// pinned tables byte for byte.
void init_cdfs(CdfState& c, int qctx_slice, bool reduced_tx_part_set) {
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
  copy_array(c.ccso_blk, kCcsoBlkCdf);
  for (int is_inter = 0; is_inter < 2; ++is_inter) {
    for (int sqr = 0; sqr < 5; ++sqr) {
      copy_array(c.sec_tx_type[is_inter][sqr],
                 av2enc::txsec::kSecTxTypeCdf[is_inter][sqr]);
    }
  }
  copy_array(c.most_probable_stx_set,
             av2enc::txsec::kMostProbableStxSetCdf);
  copy_array(c.most_probable_stx_set_adst,
             av2enc::txsec::kMostProbableStxSetAdstCdf);
  copy_array(c.cctx_type, av2enc::tcq::cdf::kCctxTypeCdf);
  copy_array(c.fsc_mode, av2enc::lossless::kFscModeCdf);
  c.tx_partition.reset(reduced_tx_part_set);
  mrl_stream::init_mrl_cdfs(c.mrl);
  c.mh_cdfs.init();
}

// ---------------------------------------------------------------------------
// Packed CDF image (§8.2 tile state without neighbour/context state)
// ---------------------------------------------------------------------------
//
// The writer's CdfState also carries MrlContext (a std::vector), so it is not
// trivially copyable.  A seed/export therefore copies only the CDF arrays, in a
// fixed order, into a flat uint16_t image.  The same descriptor list builds the
// avg_cdf::CdfBank tables, so the bank and the seed image cannot drift.
struct CdfArrayDesc {
  std::size_t struct_offset = 0;  // bytes from the start of CdfState
  std::size_t packed_offset = 0;  // uint16_t entries from the image start
  std::size_t words = 0;          // total entries over all array instances
  std::uint16_t symbols = 0;      // N; each instance has N+1 entries
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
    AV2ENC_ADD_CDF(ccso_blk, 2);
    AV2ENC_ADD_CDF(sec_tx_type, 4);
    AV2ENC_ADD_CDF(most_probable_stx_set, 7);
    AV2ENC_ADD_CDF(most_probable_stx_set_adst, 4);
    add(probe.mrl.index, sizeof(probe.mrl.index), 4);
    add(probe.mrl.sec, sizeof(probe.mrl.sec), 2);
    add(probe.dip_cdfs.use_dip.data(), sizeof(probe.dip_cdfs.use_dip), 2);
    add(probe.dip_cdfs.dip_mode.data(), sizeof(probe.dip_cdfs.dip_mode), 6);
    add(probe.mh_cdfs.is_cfl, sizeof(probe.mh_cdfs.is_cfl), 2);
    add(probe.mh_cdfs.cfl_index, sizeof(probe.mh_cdfs.cfl_index), 2);
    add(probe.mh_cdfs.cfl_sign, sizeof(probe.mh_cdfs.cfl_sign), 8);
    add(probe.mh_cdfs.cfl_alpha, sizeof(probe.mh_cdfs.cfl_alpha), 8);
    add(probe.mh_cdfs.cfl_mhccp, sizeof(probe.mh_cdfs.cfl_mhccp), 2);
    add(probe.mh_cdfs.cfl_mh_dir, sizeof(probe.mh_cdfs.cfl_mh_dir), 3);
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
         std::span<const std::int32_t> levels, const StillConfig& config,
         std::span<const palette::PaletteResult> palette_results,
         std::span<const std::uint16_t> cdf_seed,
         std::vector<std::uint16_t>* cdf_final)
      : plan_(plan), blocks_(blocks), levels_(levels), config_(config),
        palette_results_(palette_results), cdf_seed_(cdf_seed),
        cdf_final_(cdf_final) {
    geometry_.width = plan.width;
    geometry_.height = plan.height;
    geometry_.sb_size = plan.sb_size;
    geometry_.has_chroma = true;
    geometry_.chroma_subsampling_x = true;
    geometry_.chroma_subsampling_y = true;
    mi_cols_ = partition::mi_cols(geometry_);
    mi_rows_ = partition::mi_rows(geometry_);

    init_cdfs(cdfs_,
              qctx::slice_for_base_q_idx(static_cast<int>(config_.base_q_idx)),
              config_.reduced_tx_part_set != 0);
    if (!cdf_seed_.empty()) {
      import_cdf_image(cdfs_, cdf_seed_.data());
    }
    ps::init_tile_partition_cdfs(ext_cdfs_);
    tools_.enable_ext_partitions = config_.enable_ext_partitions;
    tools_.enable_uneven_4way_partitions =
        config_.enable_uneven_4way_partitions;
    cdfs_.mrl_context.reset(mi_rows_, mi_cols_,
                            partition::sb_size_mi(geometry_));
    tile_buffer_.resize(kTileCapacityBytes);
    enc_.init_symbol(tile_buffer_.data(), tile_buffer_.size());
    enc_.disable_cdf_update = config_.disable_cdf_update ? 1 : 0;

    above_mi_.assign(static_cast<std::size_t>(mi_cols_) + 16, 6);
    left_mi_.assign(static_cast<std::size_t>(mi_rows_) + 16, 6);
    above_mi_chroma_.assign(static_cast<std::size_t>(mi_cols_) + 16, 6);
    left_mi_chroma_.assign(static_cast<std::size_t>(mi_rows_) + 16, 6);
    intra_joint_.assign(
        static_cast<std::size_t>(mi_rows_) * static_cast<std::size_t>(mi_cols_),
        0);
    uvcfl_.assign(static_cast<std::size_t>(mi_rows_ / 2) *
                      static_cast<std::size_t>(mi_cols_ / 2),
                  0);
    usedip_.assign(static_cast<std::size_t>(mi_rows_) *
                       static_cast<std::size_t>(mi_cols_),
                   0);
    // §5.20.5.5: palette_mode_info() is parsed for every DC block when the
    // frame enables screen-content tools, even if this encoder never chooses a
    // palette.  The adapter writes has_palette_y = 0 in that case.
    if (config_.allow_screen_content_tools) {
      palette_stream::ScreenContentFlags flags;
      flags.single_picture_header = true;
      flags.allow_screen_content_tools = config_.allow_screen_content_tools;
      flags.allow_palette = true;
      flags.force_integer_mv = config_.force_integer_mv;
      flags.allow_intrabc = false;
      palette_writer_.emplace(flags);
      std::string palette_error;
      if (!palette_writer_->reset(mi_cols_, mi_rows_, palette_error)) {
        palette_reset_error_ = palette_error;
      }
      palette::init_palette_cdfs(palette_cdfs_);
    }
    if (config_.allow_intrabc) {
      ibcs::init_ibc_cdfs(ibc_cdfs_);
      ibc_grid_.assign(static_cast<std::size_t>(mi_rows_) *
                           static_cast<std::size_t>(mi_cols_),
                       0);
      skip_grid_.assign(ibc_grid_.size(), 0);
      ibc_neighbours_.reset(mi_cols_, mi_rows_);
      ibc_buffer_.reset(mi_cols_, mi_rows_);
      ibc_config_ = ibc::IbcConfig{};
      ibc_config_.sb_size = plan_.sb_size;
      ibc_config_.allow_global = config_.allow_global_intrabc;
      ibc_config_.allow_local = config_.allow_local_intrabc;
      ibc_config_.max_bvp_drl_bits_minus_1 = 0;
      ibc_content_.force_integer_mv = config_.force_integer_mv;
      ibc_content_.max_bvp_drl_bits_minus_1 = 0;
    }
    for (int plane = 0; plane < kPlaneCount; ++plane) {
      const int n = (plane == 0 ? mi_cols_ : mi_cols_ / 2) + 16;
      const int m = (plane == 0 ? mi_rows_ : mi_rows_ / 2) + 16;
      above_level_[plane].assign(static_cast<std::size_t>(n), 0);
      left_level_[plane].assign(static_cast<std::size_t>(m), 0);
      above_dc_[plane].assign(static_cast<std::size_t>(n), 0);
      left_dc_[plane].assign(static_cast<std::size_t>(m), 0);
    }
    clear_above_context();
    init_delta_q();
    trace_ = std::getenv("AV2ENC_STILL_TRACE") != nullptr;

    // §5.20.9.6 clear_cdef() at frame start: one tile, so a whole-grid reset
    // covers every superblock.  CDEF is opt-in; the grid stays empty otherwise.
    cdef_grid_.reset(mi_cols_, mi_rows_);
    std::memcpy(cdef_index0_, av2enc::cuda::still::kCdefIndex0Cdf,
                sizeof(cdef_index0_));
    cdef_cdfs_.index0 = &cdef_index0_[0][0];
    cdef_cdfs_.minus1 = cdef_minus1_;
    if (const std::uint16_t* minus1 =
            av2enc::cuda::still::cdef_index_minus1_default(
                config_.cdef.num_strengths)) {
      std::memcpy(cdef_minus1_, minus1,
                  sizeof(std::uint16_t) *
                      static_cast<std::size_t>(config_.cdef.num_strengths));
    }

    // §5.20.10.4/.5/.6 loop-restoration tile state.  M1 is intra with one
    // tile; luma is either per-unit Wiener NS or per-unit PC-Wiener.  Chroma
    // is independent, so PC-Wiener luma + Wiener NS chroma initialises both
    // states.
    if (config_.lr.enable && config_.lr.pc_wiener) {
      namespace pcw = av2enc::filters::pc_wiener_stream;
      pcw::LrParams pc_params;
      pc_params.frame_type[0] = pcw::kRestorePcWiener;
      pc_params.frame_type[1] = pcw::kRestoreNone;
      pc_params.frame_type[2] = pcw::kRestoreNone;
      pc_params.luma_shift = config_.lr.plane[0].shift;
      pc_params.chroma_shift = config_.lr.plane[1].shift;
      pc_ready_ = pcw::InitTileState(&pc_state_, pc_params, mi_cols_,
                                     mi_rows_, /*sub_x=*/1, /*sub_y=*/1,
                                     &error_);
      if (config_.lr.plane[1].enabled) {
        restoration::classes::WienerLrHeaderParams params;
        params.frame_type[0] = restoration::kRestoreNone;
        params.frame_type[1] = restoration::kRestoreWienerNs;
        params.frame_type[2] = restoration::kRestoreWienerNs;
        for (int plane = 0; plane < kPlaneCount; ++plane) {
          params.frame_filters_on[plane] = false;
        }
        params.num_filter_classes = 1;
        params.num_filter_classes_idx = 0;
        params.luma_shift = config_.lr.plane[0].shift;
        params.chroma_shift = config_.lr.plane[1].shift;
        lr_ready_ = restoration::classes::wiener_tile_init(
            &lr_state_, params, mi_cols_, mi_rows_, /*sub_x=*/1, /*sub_y=*/1,
            &error_);
      }
    } else if (config_.lr.enable) {
      restoration::classes::WienerLrHeaderParams params;
      for (int plane = 0; plane < kPlaneCount; ++plane) {
        params.frame_type[plane] = config_.lr.plane[plane].enabled
                                       ? restoration::kRestoreWienerNs
                                       : restoration::kRestoreNone;
        params.frame_filters_on[plane] = false;
      }
      params.num_filter_classes = 1;
      params.num_filter_classes_idx = 0;
      params.luma_shift = config_.lr.plane[0].shift;
      params.chroma_shift = config_.lr.plane[1].shift;
      lr_ready_ = restoration::classes::wiener_tile_init(
          &lr_state_, params, mi_cols_, mi_rows_, /*sub_x=*/1, /*sub_y=*/1,
          &error_);
    }
  }

  bool run(std::size_t* tile_bytes, std::string& error) {
    if (!deltaq_init_error_.empty()) {
      error = deltaq_init_error_;
      return false;
    }
    if (blocks_.empty()) {
      error = "syntax stream is empty";
      return false;
    }
    if (!error_.empty()) {
      error = error_;
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
        // `if (sb.mi_col == 0) clear_left_context()` in still_encoder.
        clear_left_context();
      }
      // §5.20.2.1: ReadDeltas = delta_q_present at every superblock start.
      deltaq::begin_superblock(tile_qp_);
      if (node.mi_col == 0) deltaq::clear_left_context(seg_grid_);
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
  void init_delta_q() {
    rc::SegmentationParams seg_params;
    rc::DeltaQParams dq_params;
    deltaq::FrameQpHeader qp_header;
    qp_header.primary_ref_is_none = true;
    std::string err;
    const int base_q_idx = static_cast<int>(config_.base_q_idx);
    if (!make_delta_q_params(config_.per_sb_delta_q, seg_params, dq_params, &err) ||
        !deltaq::derive_frame_segmentation(seg_params, dq_params, base_q_idx,
                                           qp_header, seg_, &err) ||
        !deltaq::begin_tile(tile_qp_, base_q_idx, dq_params, &err) ||
        !deltaq::init_segment_grid(
            seg_grid_, mi_cols_, mi_rows_, 0, mi_cols_, 0, mi_rows_,
            /*enable_ext_seg=*/false, deltaq::max_segments(false), {}, &err)) {
      deltaq_init_error_ = "delta-Q setup: " + err;
      return;
    }
    deltaq_cdfs_.init();
    deltaq::clear_above_context(seg_grid_);
    deltaq::clear_left_context(seg_grid_);
  }

  bool sym(std::uint16_t* cdf, int n, int symbol) {
    const av2::Status status = enc_.encode_symbol(cdf, n, symbol);
    if (status != av2::Status::kOk) {
      error_ = std::string("entropy encode_symbol failed: ") +
               av2::status_name(status);
      return false;
    }
    return true;
  }
  // §5.20.6.3 writer adapter: the transform-partition component owns the
  // syntax and contexts, this sink forwards its symbols to the tile coder.
  class TxSink final : public av2enc::video::txpartition::SymbolSink {
   public:
    explicit TxSink(Writer* writer) : writer_(writer) {}
    bool symbol(std::uint16_t* cdf, int n, int value,
                std::string& error) override {
      if (!writer_->sym(cdf, n, value)) {
        error = writer_->error_;
        return false;
      }
      return true;
    }

   private:
    Writer* writer_ = nullptr;
  };
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
    std::fill(above_mi_chroma_.begin(), above_mi_chroma_.end(), 6);
  }
  void clear_left_context() {
    for (int plane = 0; plane < kPlaneCount; ++plane) {
      std::fill(left_level_[plane].begin(), left_level_[plane].end(), 0);
      std::fill(left_dc_[plane].begin(), left_dc_[plane].end(), 0);
    }
    std::fill(left_mi_.begin(), left_mi_.end(), 6);
    std::fill(left_mi_chroma_.begin(), left_mi_chroma_.end(), 6);
  }

  int& above_mi(int x, int plane = 0) {
    return (plane == 1 ? above_mi_chroma_ : above_mi_)[
        static_cast<std::size_t>(x)];
  }
  int& left_mi(int y, int plane = 0) {
    return (plane == 1 ? left_mi_chroma_ : left_mi_)[
        static_cast<std::size_t>(y)];
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
  bool code_do_split(PartitionType partition, int r, int c, BlockSize size,
                     int plane = 0) {
    const int bsw = std::max(kMiWidthLog2[static_cast<int>(size)], 1);
    const int bsh = std::max(kMiHeightLog2[static_cast<int>(size)], 1);
    const int ctx = kPartitionSizeAdjust[static_cast<int>(size)] * 4 +
                    (left_mi(r, plane) < bsh ? 2 : 0) +
                    (above_mi(c, plane) < bsw ? 1 : 0);
    return sym(&cdfs_.do_split[plane][ctx][0], 2,
               partition == PartitionType::kNone ? 0 : 1);
  }

  // §5.20.3.2 do_square_split.  Only legal at a 128x128 (or 256x256) node, so
  // a 64x64-superblock stream never codes it and its bytes are unchanged.
  bool code_do_square_split(PartitionType partition, int r, int c,
                            BlockSize size, int plane = 0) {
    const int bsw = kMiWidthLog2[static_cast<int>(size)];
    const int bsh = kMiHeightLog2[static_cast<int>(size)];
    int ctx = (left_mi(r, plane) < bsh ? 2 : 0) +
              (above_mi(c, plane) < bsw ? 1 : 0);
    if (size == BlockSize::k256x256) ctx += 4;
    return sym(&cdfs_.do_square_split[plane][ctx][0], 2,
               partition == PartitionType::kSplit ? 1 : 0);
  }

  bool code_rect_type(PartitionType partition, int r, int c, BlockSize size,
                      int plane = 0) {
    const int bsw = std::max(kMiWidthLog2[static_cast<int>(size)], 1);
    const int bsh = std::max(kMiHeightLog2[static_cast<int>(size)], 1);
    const int ctx = kPartitionSizeAdjustRect[static_cast<int>(size)] * 4 +
                    (left_mi(r, plane) < bsh ? 2 : 0) +
                    (above_mi(c, plane) < bsw ? 1 : 0);
    return sym(&cdfs_.rect_type[plane][ctx][0], 2,
               partition == PartitionType::kVert ? kRectVert : kRectHorz);
  }

  // Emit the syntax for the node at `index` (and, for non-leaves, its whole
  // subtree); `next` receives the index one past the subtree.
  bool write_node(std::size_t index, std::size_t* next, std::string& error) {
    if (config_.enable_sdp) {
      return write_node_sdp(index, next, error);
    }
    if (config_.enable_ext_partitions) {
      return write_node_ext(index, next, error);
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
      if (partition != PartitionType::kSplit) {
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

  // SDP writer (--still-sdp).  Intra SDP replaces every 64x64 SHARED node with
  // a luma tree and then a chroma tree (§5.20.3.1).  The split node itself
  // codes nothing.  The luma/chroma trees are currently forced PARTITION_NONE:
  // that is the first decoder-gated configuration and the writer rejects any
  // other partition instead of guessing the tree-specific implied rules.
  bool write_node_sdp(std::size_t index, std::size_t* next,
                      std::string& error) {
    const BlockSyntax& node = blocks_[index];
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

    const auto size = static_cast<BlockSize>(node.bsize);
    const auto partition = static_cast<PartitionType>(node.partition);
    if (!partition::is_valid_block_size(size)) {
      error = "node " + std::to_string(index) + " has an invalid block size";
      return false;
    }
    if (!inside_mi(node.mi_row, node.mi_col)) {
      error = "node " + std::to_string(index) + " is outside the frame";
      return false;
    }
    const int plane = node.tree_type == 2 ? 1 : 0;
    const bool forced = (node.flags & kFlagForced) != 0;
    const bool allowed_none =
        (node.allowed_mask & (1u << static_cast<unsigned>(
                                  PartitionType::kNone))) != 0;
    if (partition != PartitionType::kNone) {
      error = "SDP sub-tree partitions are not wired: node " +
              std::to_string(index) + " at " +
              partition::block_size_name(size) + " codes " +
              partition::partition_name(partition);
      return false;
    }
    if (!forced && allowed_none) {
      if (!code_do_split(partition, node.mi_row, node.mi_col, size, plane)) {
        error = error_;
        return false;
      }
    }
    if (node.subtree != 0) {
      error = "SDP leaf node " + std::to_string(index) +
              " has a non-zero subtree";
      return false;
    }
    if (!write_block(node, error)) return false;
    *next = index + 1;
    return true;
  }

  // Extended-partition writer (--still-ext-partitions).  The symbol sequence
  // comes from partition_stream/partition_ext, so the implication rules for
  // do_ext_partition / do_uneven_4way_partition and the raw L(1) type bit are
  // the reference module's, not re-derived here.
  //
  // Choma ownership: the search policy only produces nodes whose every child is
  // >= 8x8, so `chromaOffset` is 0 throughout and each leaf owns its own chroma
  // block.  A tree that violates that is rejected below instead of coding a
  // desynchronised stream.
  bool write_node_ext(std::size_t index, std::size_t* next,
                      std::string& error) {
    const BlockSyntax& node = blocks_[index];
    const auto size = static_cast<BlockSize>(node.bsize);
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
    if (static_cast<pe::PartitionType>(node.partition) != pe::PartitionType::kNone &&
        node.subtree == 0) {
      error = "node " + std::to_string(index) + " has no children for a split";
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
    const bool chroma_info = (node.flags & kFlagHasChromaInfo) != 0;
    sn.has_chroma = chroma_info ? (node.has_chroma != 0) : true;
    sn.chroma_offset = chroma_info ? (node.chroma_offset != 0) : false;
    sn.chroma = chroma_info
                    ? pe::ChromaBlock{node.chroma_mi_row, node.chroma_mi_col,
                                      static_cast<BlockSize>(node.chroma_bsize)}
                    : pe::ChromaBlock{r, c, size};

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

  // is_cfl context: neighbouring chroma CfL flags (§8.3.x).  Indexed at the
  // owned chroma block origin (ChromaMi), not the luma leaf.
  int cfl_context() const {
    const int cr = cur_chroma_r_;
    const int cc = cur_chroma_c_;
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

  bool code_uv_mode(const BlockSyntax& node, int cw, int ch) {
    const bool is_cfl = (node.flags & kFlagIsCfl) != 0;
    if (is_cfl) {
      // is_cfl derives its context from neighbouring chroma CfL flags.
      if (!sym(&cdfs_.is_cfl[cfl_context()][0], 2, 1)) return false;
      if (config_.enable_mhccp) {
        // §5.20.7.32 with MHCCP: cfl_mhccp, cfl_index, cfl_mh_dir and the
        // explicit alpha symbols all live in the gated mhccp adapter.
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
        if (!cfl::mhccp::write_cfl_alphas(
                enc_, cdfs_.mh_cdfs, syntax, config_.enable_cfl_intra,
                mhccp_allowed,
                cfl::mhccp::size_group_for_chroma_block(cw, ch),
                &alpha_error)) {
          error_ = "read_cfl_alphas: " + alpha_error;
          return false;
        }
        return true;
      }
      // §5.20.7.32: cfl_index 0 = CFL_EXPLICIT, 1 = CFL_DERIVED_ALPHA
      // (2 = CFL_MULTI/MHCCP, wired separately).
      if (node.cfl_index > 1) {
        error_ = "cfl_index 2 (MHCCP) is not wired";
        return false;
      }
      if (!sym(&cdfs_.cfl_index[0], 2, node.cfl_index)) return false;
      if (node.cfl_index == 0) {
        const int signs = node.cfl_alpha_signs;
        if (signs < 0 || signs > 7) {
          error_ = "cfl_alpha_signs outside 0..7";
          return false;
        }
        const int sign_u = (signs + 1) / 3;
        const int sign_v = (signs + 1) % 3;
        if (!sym(&cdfs_.cfl_sign[0], 8, signs)) return false;
        // §8.3 Tables 8.4/8.5 alpha CDF contexts.
        static constexpr int kAlphaCtxU[8] = {-1, -1, 0, 1, 2, 3, 4, 5};
        static constexpr int kAlphaCtxV[8] = {0, 3, -1, 1, 4, -1, 2, 5};
        if (sign_u != 0) {
          const int ctx = kAlphaCtxU[signs];
          if (ctx < 0 || node.cfl_alpha_u > 7) {
            error_ = "invalid cfl_alpha_u";
            return false;
          }
          if (!sym(&cdfs_.cfl_alpha[ctx][0], 8, node.cfl_alpha_u)) return false;
        }
        if (sign_v != 0) {
          const int ctx = kAlphaCtxV[signs];
          if (ctx < 0 || node.cfl_alpha_v > 7) {
            error_ = "invalid cfl_alpha_v";
            return false;
          }
          if (!sym(&cdfs_.cfl_alpha[ctx][0], 8, node.cfl_alpha_v)) return false;
        }
      }
      return true;
    }
    const int y_mode = node.y_mode;
    const int uv_mode = node.uv_mode;
    const int ctx = y_mode_is_directional(y_mode) ? 1 : 0;
    // is_cfl = 0 is only coded when CfL is allowed.
    if (config_.enable_cfl_intra) {
      if (!sym(&cdfs_.is_cfl[cfl_context()][0], 2, 0)) return false;
    }
    const int idx = ((cur_w4_ * 4 < 8) || (cur_h4_ * 4 < 8))
                        ? uv_mode_index_sub8(uv_mode)
                        : uv_mode_index(uv_mode, y_mode);
    if (idx <= 6) {
      return sym(&cdfs_.uv_mode[ctx][0], 8, idx);
    }
    if (!sym(&cdfs_.uv_mode[ctx][0], 8, 7)) return false;
    return write_literal(3, static_cast<std::uint32_t>(idx - 7));
  }

  // ---------------------------------------------------------------------
  // Residual syntax
  // ---------------------------------------------------------------------
  bool write_eob(int eob, int plane, int coded_w, int coded_h) {
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
    const int cdf_ctx = plane > 0 ? 2 : 0;
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

  // One transform block of a leaf: luma parts come from LumaTxPart, chroma
  // from the historical plane fields.
  struct TxBlockView {
    int tx_size = 0;
    int tx_type = 0;
    int sec_tx_type = 0;
    int stx_set = 0;
    int eob = 0;
    int mi_row = 0;
    int mi_col = 0;
  };

  // `level_base` is the index of this plane's first scan-order level.
  bool code_tx(const BlockSyntax& node, int plane, const TxBlockView& view,
               std::size_t level_base, int bsize_w, int bsize_h,
               std::size_t levels_end, std::string& error) {
    const int tx_size = view.tx_size;
    if (tx_size >= 25) {
      error = "invalid tx_size " + std::to_string(tx_size) + " on plane " +
              std::to_string(plane);
      return false;
    }
    // §5.20.7.27 selects dc_sign_horz_vert for the HORIZ/VERT TX classes.  With
    // enable_idtx_intra = 0 no 1-D (identity-axis) transform is in any intra
    // TX-type set, so fail loudly instead of coding a raw sign bit if a plan
    // ever supplies one.  `--still-idtx` enables those sets, and the sign coder
    // below matches the gated dav2d behaviour for them.
    if (!config_.enable_idtx_intra) {
      txv::Tx1dType row_type = txv::Tx1dType::kDct;
      txv::Tx1dType col_type = txv::Tx1dType::kDct;
      txv::tx_1d_types(static_cast<txv::TxType>(view.tx_type), &row_type,
                       &col_type);
      if (tx_size < 25 &&
          (row_type == txv::Tx1dType::kIdt || col_type == txv::Tx1dType::kIdt)) {
        error =
            "1-D intra TX types are not signalled: dc_sign_horz_vert "
            "(§5.20.7.27) requires enable_idtx_intra, which is 0";
        return false;
      }
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
    const int sub_x = plane > 0 ? 1 : 0;
    const int sub_y = plane > 0 ? 1 : 0;
    // Chroma coefficient contexts are indexed at the owned ChromaMi origin,
    // which can differ from the luma leaf origin for sub-8 blocks.  Luma
    // contexts follow the transform part origin.
    const bool chroma_info = (node.flags & kFlagHasChromaInfo) != 0;
    const int x4 = plane > 0
                       ? (chroma_info ? (node.chroma_mi_col >> sub_x)
                                      : (node.mi_col >> sub_x))
                       : view.mi_col;
    const int y4 = plane > 0
                       ? (chroma_info ? (node.chroma_mi_row >> sub_y)
                                      : (node.mi_row >> sub_y))
                       : view.mi_row;
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
    std::uint16_t* skip_cdf = plane == 2 ? &cdfs_.v_txb_skip[ctx][0]
                                         : &cdfs_.txb_skip[0][tsc][ctx][0];
    if (!sym(skip_cdf, 2, eob == 0 ? 1 : 0)) return false;
    if (eob == 0) {
      if (plane == 1 && config_.enable_cctx && node.cctx_type != 0) {
        error = "cctx_type must be 0 when the U plane is all zero";
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
      if (plane == 1) eob_u_ = 0;
      return true;
    }

    if (!write_eob(eob, plane, coded_w, coded_h)) {
      error = error_;
      return false;
    }
    if (plane == 1 && config_.enable_cctx) {
      // §5.20.7.27: cctx_type follows the U EOB and precedes the
      // coefficients.  (is_inter || eob != 1) reduces to eob != 1 for an
      // intra picture, and 4:2:0 makes is_cctx_allowed() true whenever the
      // sequence enables CCTX.
      if (eob != 1) {
        if (!sym(&cdfs_.cctx_type[0], 7, node.cctx_type)) return false;
      } else if (node.cctx_type != 0) {
        error = "cctx_type must be 0 when the U plane has eob == 1";
        return false;
      }
    }
    if (plane == 0) {
      const TxTypeSyntax type_syntax = luma_tx_type_syntax(tx_size, eob);
      if (type_syntax != TxTypeSyntax::kNone) {
        // §5.20.7.29: the mapped mode adds Mrl_Index_To_Delta before the
        // wide-angle map.  At mrl_index 0 this equals txtype::map_intra_mode().
        // The set is selected by the transform size, not the coding block.
        int mapped_mode = mrl_stream::mapped_intra_mode_for_mrl(
            node.y_mode, node.y_angle_delta, node.mrl_index, coded_w, coded_h);
        if (mapped_mode < 0) mapped_mode = 0;
        txtype::TxTypeQuery query;
        query.coded_w = coded_w;
        query.coded_h = coded_h;
        query.mapped_mode = mapped_mode;
        query.reduced_tx_set = config_.reduced_tx_set;
        query.dct_only = false;
        int symbol = -1;
        bool is_long_side_dct = false;
        if (!txtype::tx_type_symbol(query, static_cast<txtype::TxType>(
                                                view.tx_type),
                                     &symbol, &is_long_side_dct)) {
          error = "tx_type is not codable for this size/mode";
          return false;
        }
        switch (type_syntax) {
          case TxTypeSyntax::kNone:
            break;
          case TxTypeSyntax::kIntraSet1:
            if (!sym(&cdfs_.intra_tx_type_set1[kTxSizeSqr[tx_size]][0], 7,
                     symbol)) {
              return false;
            }
            break;
          case TxTypeSyntax::kLong32:
            // A wide/high 32 set codes both the long-side flag and the type.
            if (!sym(&cdfs_.is_long_side_dct[0][0], 2,
                     is_long_side_dct ? 1 : 0)) {
              return false;
            }
            if (!sym(&cdfs_.intra_tx_type_long[kTxSizeSqr[tx_size]][0], 4,
                     symbol)) {
              return false;
            }
            break;
          case TxTypeSyntax::kLong64:
            // WIDE_64/HIGH_64 imply is_long_side_dct = 1; only the type is
            // coded.
            if (!sym(&cdfs_.intra_tx_type_long[kTxSizeSqr[tx_size]][0], 4,
                     symbol)) {
              return false;
            }
            break;
        }
      }
    }

    if (plane == 0 && config_.enable_intra_ist) {
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
      stx.mrl_index = 0;  // MRL is not wired yet
      stx.most_probable_stx_set = view.stx_set;
      stx.sec_tx_type = view.sec_tx_type;
      stx.is_inter = false;
      stx.lossless = false;
      stx.bit_depth = 8;
      if (txsec::secondary_syntax_applicable(0, eob, stx, stx_config)) {
        // sec_tx_type is coded for every applicable block, including the
        // "no IST" symbol 0.
        if (!sym(&cdfs_.sec_tx_type[0][kTxSizeSqr[tx_size]][0], 4,
                 view.sec_tx_type)) {
          return false;
        }
        if (view.sec_tx_type != 0) {
          if (!txsec::secondary_syntax_valid(0, eob, stx, stx_config)) {
            error = "sec_tx_type is not valid for this block";
            return false;
          }
          const int set_count = txsec::most_probable_stx_set_count(
              stx.plane_tx_type, coded_w, coded_h);
          std::uint16_t* set_cdf = set_count == 4
                                       ? cdfs_.most_probable_stx_set_adst
                                       : cdfs_.most_probable_stx_set;
          if (!sym(set_cdf, set_count, view.stx_set)) {
            return false;
          }
        }
      } else if (view.sec_tx_type != 0) {
        error = "sec_tx_type set on a block where IST is not applicable";
        return false;
      }
    }

    static constexpr int kSigRef[5][2] = {
        {0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
    static constexpr int kMagRef[3][2] = {{0, 1}, {1, 0}, {1, 1}};

    // §5.20.7.27: useTcq is luma-only, TX_CLASS_2D-only and disabled for
    // lossless/FSC.  An intra still is never lossless/FSC here.  TX_TYPE
    // 10..15 are the 1-D classes; IDTX (9) is TX_CLASS_2D.
    const bool use_tcq =
        config_.enable_tcq && plane == 0 && view.tx_type <= 9;
    int tcq_state = 0;

    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    build_scan(adj_w, adj_h, scan);
    std::vector<int> level_pre(static_cast<std::size_t>(area), 0);
    // Levels arrive in scan order: levels_[level_base + c] is scan position c.
    // Under TCQ they are the full levels `q`; the syntax level is
    // syntax_level_for(q, pre_max) and read_quant carries the escape.
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
        tcq_state = av2enc::tcq::next_state(tcq_state, syntax_level & 1);
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
          // read_quant under TCQ escapes at `maxLevel - 1` and adds `x << 1`
          // where x = (full_level - syntax_level) / 2.  `pre_level` is the
          // syntax level the first pass coded.
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

  // §5.20.10.2 read_ccso(): one binary symbol per enabled plane at every CCSO
  // unit origin (64x64 for M1's 64x64 superblocks).  Each plane's unit grid
  // carries the decision; the context is the left unit's coded value.
  bool write_ccso(const BlockSyntax& node, std::string& error) {
    if (!config_.ccso.enable) return true;
    // §5.18.7.12: CcsoLumaSizeLog2 is 6 for a 64x64 frame superblock and 7 for
    // a 128x128 one (ccso_unit_matches_sb_size = 1).
    const int ccso_shift = plan_.sb_size == 128 ? 5 : 4;
    const int kCcsoUnitMi = 1 << ccso_shift;
    const int kCcsoShift = ccso_shift;
    const int mi_row = node.mi_row;
    const int mi_col = node.mi_col;
    if ((mi_row & (kCcsoUnitMi - 1)) != 0 ||
        (mi_col & (kCcsoUnitMi - 1)) != 0) {
      return true;
    }
    const int unit_row = mi_row >> kCcsoShift;
    const int unit_col = mi_col >> kCcsoShift;
    for (int plane = 0; plane < kPlaneCount; ++plane) {
      if (!config_.ccso.plane[plane].enabled) continue;
      const std::uint8_t* grid = plan_.ccso_blks[plane];
      const int cols = plan_.ccso_blk_cols[plane];
      const int rows = plan_.ccso_blk_rows[plane];
      if (unit_row >= rows || unit_col >= cols) {
        error = "CCSO unit grid does not cover a coding block";
        return false;
      }
      const int value =
          grid != nullptr
              ? (grid[static_cast<std::size_t>(unit_row) *
                          static_cast<std::size_t>(cols) +
                      static_cast<std::size_t>(unit_col)] != 0
                     ? 1
                     : 0)
              : 1;
      int ctx = 0;
      if (mi_col - kCcsoUnitMi >= 0) {
        const int left_col = (mi_col - kCcsoUnitMi) >> kCcsoShift;
        const std::uint8_t left =
            grid != nullptr
                ? grid[static_cast<std::size_t>(unit_row) *
                           static_cast<std::size_t>(cols) +
                       static_cast<std::size_t>(left_col)]
                : 1;
        ctx = 2 * (left != 0 ? 1 : 0);
      }
      if (!sym(&cdfs_.ccso_blk[plane][ctx][0], 2, value)) {
        return false;
      }
    }
    return true;
  }

  // §5.20.8 palette_mode_info() + palette_tokens().  Called for every leaf when
  // the frame enables palette (the adapter writes has_palette_y = 0 itself when
  // the position is reachable and no palette was chosen).
  bool write_palette(const BlockSyntax& node, int w, int h,
                     std::string& error) {
    const bool is_palette = (node.flags & kFlagIsPalette) != 0;
    if (!config_.allow_screen_content_tools) {
      if (is_palette) {
        error = "palette leaf supplied but the frame disables screen tools";
        return false;
      }
      return true;
    }
    if (!palette_writer_.has_value()) {
      error = "palette writer is not initialised: " + palette_reset_error_;
      return false;
    }
    const palette::PaletteResult* result = nullptr;
    if (is_palette) {
      if (node.palette_index >= palette_results_.size()) {
        error = "palette side-array index out of range";
        return false;
      }
      result = &palette_results_[node.palette_index];
      if (!result->has_palette) {
        error = "palette leaf references a result without a palette";
        return false;
      }
    }
    // §5.20.8.4 onscreen dimensions use MiCols/MiRows, not the picture size.
    const int onscreen_w = std::min(w, mi_cols_ * 4 - node.mi_col * 4);
    const int onscreen_h = std::min(h, mi_rows_ * 4 - node.mi_row * 4);
    palette_stream::PaletteModeInfo info;
    std::string palette_error;
    if (!palette_writer_->write(enc_, palette_cdfs_, node.mi_row, node.mi_col,
                                node.bsize, w, h, onscreen_w, onscreen_h,
                                result, node.y_mode, &info, palette_error)) {
      error = "palette syntax: " + palette_error;
      return false;
    }
    if ((info.has_palette_y ? true : false) != is_palette) {
      error = "palette writer disagrees with the planned palette decision";
      return false;
    }
    return true;
  }

  // One Wiener NS unit symbol (plus coefficients) at a 64x64 luma / 32x32
  // chroma unit origin.
  bool write_lr_plane_units(const BlockSyntax& node, int plane,
                            std::string& error) {
    const StillLrPlane& cfg = config_.lr.plane[plane];
    if (!cfg.enabled) return true;
    const int sub = plane == 0 ? 0 : 1;
    const int unit_size = lr_state_.unit_size[plane];
    const int origin_x = (node.mi_col * 4) >> sub;
    const int origin_y = (node.mi_row * 4) >> sub;
    if (unit_size <= 0 || origin_x % unit_size != 0 ||
        origin_y % unit_size != 0) {
      return true;
    }
    const int unit_row = origin_y / unit_size;
    const int unit_col = origin_x / unit_size;
    if (unit_row >= cfg.unit_rows || unit_col >= cfg.unit_cols) {
      error = "LR unit grid does not cover a coding block";
      return false;
    }
    const std::size_t index =
        static_cast<std::size_t>(unit_row) *
            static_cast<std::size_t>(cfg.unit_cols) +
        static_cast<std::size_t>(unit_col);
    if (index >= cfg.unit_type.size() ||
        index * restoration::kWienerNsCoeffStride >=
            cfg.unit_coeffs.size()) {
      error = "LR unit decision vector is too small";
      return false;
    }
    restoration::classes::WienerUnitSyntax unit;
    unit.type = cfg.unit_type[index];
    unit.merged = cfg.unit_merged[index] != 0;
    unit.use_bank_k = cfg.unit_bank_k[index];
    unit.subset = cfg.unit_subset[index];
    unit.uv_sym = cfg.unit_uv_sym[index] != 0;
    for (int j = 0; j < restoration::kWienerNsCoeffStride; ++j) {
      unit.coeffs[j] =
          cfg.unit_coeffs[index * restoration::kWienerNsCoeffStride +
                         static_cast<std::size_t>(j)];
    }
    return restoration::classes::wiener_write_unit(
        enc_, lr_state_, plane, unit_row, unit_col, unit, &error);
  }

  // §5.20.10.4 read_lr(): called once per coding block, before
  // read_partition().  M1's LoopRestorationSize is at least one superblock, so
  // only a superblock origin can start a unit; child blocks are no-ops.
  bool write_lr_superblock(const BlockSyntax& node, std::string& error) {
    if (config_.lr.enable && config_.lr.pc_wiener) {
      if (!pc_ready_) return error.empty() ? true : false;
      const StillLrPlane& cfg = config_.lr.plane[0];
      if (cfg.enabled) {
        const int unit_size = pc_state_.unit_size[0];
        const int origin_x = node.mi_col * 4;
        const int origin_y = node.mi_row * 4;
        if (unit_size > 0 && origin_x % unit_size == 0 &&
            origin_y % unit_size == 0) {
          const int unit_row = origin_y / unit_size;
          const int unit_col = origin_x / unit_size;
          if (unit_row >= cfg.unit_rows || unit_col >= cfg.unit_cols) {
            error = "PC-Wiener unit grid does not cover a coding block";
            return false;
          }
          const std::size_t index =
              static_cast<std::size_t>(unit_row) *
                  static_cast<std::size_t>(cfg.unit_cols) +
              static_cast<std::size_t>(unit_col);
          if (index >= cfg.unit_type.size()) {
            error = "PC-Wiener unit decision vector is too small";
            return false;
          }
          av2enc::filters::pc_wiener_stream::UnitDecision decision;
          decision.type = cfg.unit_type[index];
          if (!av2enc::filters::pc_wiener_stream::WriteUnit(
                  enc_, pc_state_, 0, unit_row, unit_col, decision,
                  &error)) {
            return false;
          }
        }
      }
      // Chroma Wiener NS units follow luma in the same read_lr() call.
      if (lr_ready_) {
        for (int plane = 1; plane < kPlaneCount; ++plane) {
          if (!write_lr_plane_units(node, plane, error)) return false;
        }
      }
      return true;
    }
    if (!lr_ready_) return true;
    for (int plane = 0; plane < kPlaneCount; ++plane) {
      if (!write_lr_plane_units(node, plane, error)) return false;
    }
    return true;
  }

  // §5.20.7.9 dip_mode_info(): after the intra Y/UV mode and palette syntax,
  // before the residual.  Only luma DC blocks with the size predicate can
  // signal it.
  bool write_dip(const BlockSyntax& node, int w, int h, std::string& error) {
    const bool present = config_.enable_dip && node.y_mode == kDcPred && w > 4 &&
                         h > 4 && w * h >= 128 &&
                         (node.flags & kFlagIsPalette) == 0;
    if (!present) {
      if (node.use_dip != 0) {
        error = "use_dip is not signalable for this block";
        return false;
      }
      return true;
    }
    const int r = node.mi_row;
    const int c = node.mi_col;
    // §8.3.1 use_dip context: the sum of UseDip over the §5.20.4.1 NPos set
    // (below-left, above-right, left, above — first two valid), not just the
    // immediate above/left neighbours.
    const mrl_stream::MrlBlockSyntax mrl_block{r, c, w >> 2, h >> 2};
    mrl_stream::NeighborPos npos[2];
    const int nnum = mrl_stream::collect_npos(
        mrl_block, mi_rows_, mi_cols_, partition::sb_size_mi(geometry_), npos);
    int use_dip_ctx = 0;
    for (int i = 0; i < nnum; ++i) {
      if (usedip_[static_cast<std::size_t>(npos[i].row) *
                      static_cast<std::size_t>(mi_cols_) +
                  static_cast<std::size_t>(npos[i].col)] != 0) {
        ++use_dip_ctx;
      }
    }
    dip::SyntaxContext context;
    context.enable_dip = true;
    context.y_mode = node.y_mode;
    context.palette_size_y = 0;
    context.tree_is_chroma_part = false;
    context.w = w;
    context.h = h;
    context.use_dip_ctx = use_dip_ctx;
    dip::DipModeInfo info;
    info.use_dip = node.use_dip != 0;
    info.dip_transpose = node.dip_transpose != 0;
    info.dip_mode = node.dip_mode;
    std::string dip_error;
    if (!dip::write_dip_mode_info(context, info, enc_, cdfs_.dip_cdfs,
                                  &dip_error)) {
      error = "dip_mode_info: " + dip_error;
      return false;
    }
    return true;
  }

  void update_usedip(const BlockSyntax& node, int w4, int h4) {
    if (!config_.enable_dip) return;
    const std::uint8_t value = node.use_dip != 0 ? 1 : 0;
    for (int yy = 0; yy < h4; ++yy) {
      const int rr = node.mi_row + yy;
      if (rr >= mi_rows_) break;
      for (int xx = 0; xx < w4; ++xx) {
        const int cc = node.mi_col + xx;
        if (cc >= mi_cols_) break;
        usedip_[static_cast<std::size_t>(rr) *
                    static_cast<std::size_t>(mi_cols_) +
                static_cast<std::size_t>(cc)] = value;
      }
    }
  }

  // ---------------------------------------------------------------------
  // Coding a leaf block
  // ---------------------------------------------------------------------
  // §5.20.5.3 use_intrabc + read_skip.  Called for every leaf before the
  // segment/cdef/ccso/delta-q group (SegIdPreSkip = 0).
  bool write_ibc_mode_info(const BlockSyntax& node, std::string& error) {
    if (!config_.allow_intrabc) {
      if ((node.flags & kFlagIsIbc) != 0) {
        error = "IBC leaf supplied but the sequence/frame disables intrabc";
        return false;
      }
      return true;
    }
    const bool is_ibc = (node.flags & kFlagIsIbc) != 0;
    // §5.20.5.3: the symbol is present only for the exact predicate the decoder
    // uses (allow_intrabc, not CHROMA_PART, w/h <= 64, MiSize != 64x64).
    if (!ibcs::intrabc_available(config_.allow_intrabc, node.bsize)) {
      if (is_ibc) {
        error = "IBC leaf is not signalable at this block size";
        return false;
      }
      return true;
    }
    if (is_ibc) {
      // read_intrabc_info() forces these values; anything else is a plan bug.
      if (node.y_mode != kDcPred || node.use_dip != 0 ||
          (node.flags & (kFlagIsPalette | kFlagIsCfl)) != 0) {
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
        ibcs::skip_symbol_context(skip_grid_, mi_cols_, mi_rows_, node.mi_row,
                                  node.mi_col, node.bsize, plan_.sb_size),
        2);
    return ibcs::write_skip_flag(enc_, ibc_cdfs_, skip_ctx,
                                 node.ibc_skip != 0, error);
  }

  // §5.20.5.4 read_intrabc_info(): intrabc_mode / DRL / (precision) /
  // assign_mv.  The stack is rebuilt from the same neighbour map the decoder
  // maintains, so a NEARMV index always names the same RefStackMv.
  bool write_intrabc_info(const BlockSyntax& node, std::string& error) {
    ibc::FrameView frame{};
    frame.has_chroma = true;
    frame.sub_x = 1;
    frame.sub_y = 1;
    const ibcs::BvStack stack = ibcs::build_bv_stack(
        frame, ibc_config_, ibc_buffer_, node.mi_row, node.mi_col, node.bsize,
        ibc_neighbours_);
    ibcs::BvSyntax syntax;
    syntax.mode = node.ibc_mode == 1 ? ibcs::IntrabcMode::kNearMv
                                     : ibcs::IntrabcMode::kNewMv;
    syntax.ref_mv_idx = node.ibc_ref_mv_idx;
    syntax.intrabc_precision_quarter = false;
    const ibc::BlockVector target{node.ibc_bv_row, node.ibc_bv_col};
    return ibcs::write_intrabc_info(enc_, ibc_cdfs_, ibc_content_, stack,
                                    syntax, target, error);
  }

  // §5.20.5.1 reset_block_context() for skip_flag = 1: the decoder zeroes the
  // level/DC contexts over the whole coding block, and code_tx() does the same
  // for an eob = 0 plane.  An IBC skip block codes no residual, so the reset
  // must still happen or later blocks derive different coefficient contexts.
  void reset_skip_block_context(const BlockSyntax& node, int w, int h, int cw,
                                int ch) {
    for (int plane = 0; plane < 3; ++plane) {
      const int coded_w = plane == 0 ? w : cw;
      const int coded_h = plane == 0 ? h : ch;
      const int w4 = coded_w >> 2;
      const int h4 = coded_h >> 2;
      const int sub = plane > 0 ? 1 : 0;
      const int x4 = node.mi_col >> sub;
      const int y4 = node.mi_row >> sub;
      for (int i = 0; i < w4; ++i) {
        above_level(plane, x4 + i) = 0;
        above_dc(plane, x4 + i) = 0;
      }
      for (int i = 0; i < h4; ++i) {
        left_level(plane, y4 + i) = 0;
        left_dc(plane, y4 + i) = 0;
      }
    }
    eob_u_ = 0;
  }

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

    // §5.20.3.1 chroma ownership.  A producer that sets kFlagHasChromaInfo
    // carries the decoded HasChroma and the ChromaMi block the leaf owns; the
    // historical producers leave it clear and keep the leaf-derived behaviour.
    const bool chroma_info = (node.flags & kFlagHasChromaInfo) != 0;
    const bool has_chroma = !chroma_info || node.has_chroma != 0;
    const BlockSize chroma_mi_size =
        chroma_info ? static_cast<BlockSize>(node.chroma_bsize) : size;
    const bool is_ibc = (node.flags & kFlagIsIbc) != 0;
    if (config_.allow_intrabc) {
      ibcs::begin_ibc_block(ibc_buffer_, r, c);
    }
    const BlockSize chroma_size =
        partition::chroma_plane_residual_size(geometry_, chroma_mi_size);
    if (!partition::is_valid_block_size(chroma_size)) {
      error = "invalid chroma residual size";
      return false;
    }
    const int cw = partition::block_width(chroma_size);
    const int ch = partition::block_height(chroma_size);
    cur_chroma_r_ = chroma_info ? (node.chroma_mi_row >> 1) : (r / 2);
    cur_chroma_c_ = chroma_info ? (node.chroma_mi_col >> 1) : (c / 2);

    // §6.19.3.1 TreeType.  A CHROMA_PART leaf codes no luma mode, no TX size,
    // no CDEF/CCSO/delta-q/segment id, and no luma residual; it codes UV only
    // when it owns a chroma block.  Shared and LUMA_PART use plane 0.
    const int tree_type = node.tree_type;
    const bool chroma_part = tree_type == 2;
    if (chroma_part && !has_chroma) {
      // §5.20.3.1: a chroma-tree leaf with HasChroma == 0 codes nothing.
      return true;
    }

    // Validate the declared transform sizes against TX_MODE_LARGEST geometry
    // before emitting anything, so a bad GPU plan fails loudly instead of
    // desynchronizing the CDF state.  A non-owning sub-8 leaf codes luma only;
    // a CHROMA_PART leaf codes luma not at all.  Under TX_MODE_SELECT the luma
    // size is carried by the per-part list and validated against the signalled
    // partition below; chroma keeps the TX_MODE_LARGEST geometry.
    const int coded_w[kPlaneCount] = {w, cw, cw};
    const int coded_h[kPlaneCount] = {h, ch, ch};
    for (int plane = config_.tx_mode_select ? 1 : 0; plane < kPlaneCount;
         ++plane) {
      if (plane > 0 && !has_chroma) continue;
      if (plane == 0 && chroma_part) continue;
      const auto params_opt = txv::make_tx_params(
          static_cast<txv::TxSize>(node.tx_size[plane]), txv::TxType::kDctDct);
      if (!params_opt.has_value() || params_opt->coded_w != coded_w[plane] ||
          params_opt->coded_h != coded_h[plane]) {
        error = "tx_size on plane " + std::to_string(plane) +
                " does not match the TX_MODE_LARGEST block size";
        return false;
      }
    }

    const std::size_t levels_end = levels_.size();
    // §5.20.5.3: use_intrabc (and read_skip when set) precede
    // intra_segment_id/cdef/ccso/delta_qindex for the default SegIdPreSkip = 0.
    if (!write_ibc_mode_info(node, error)) return false;
    // §5.20.5.7 intra_segment_id(): first symbol of the leaf when segmentation
    // is enabled.  The neutral table uses segment 0; the per-segment-Q planner
    // fills node.segment_id.
    deltaq::SegmentBlock segment_block;
    segment_block.mi_row = r;
    segment_block.mi_col = c;
    segment_block.bw4 = w4;
    segment_block.bh4 = h4;
    const int segment_id = seg_.enabled ? node.segment_id : 0;
    std::string dq_error;
    if (!chroma_part) {
      if (!deltaq::write_intra_segment_id(enc_, deltaq_cdfs_, seg_, seg_grid_,
                                          segment_block, segment_id,
                                          /*skip_flag=*/is_ibc &&
                                              node.ibc_skip != 0,
                                          &dq_error)) {
        error = "intra_segment_id: " + dq_error;
        return false;
      }
      // §5.20.10.1 read_cdef(): after intra_segment_id(), before
      // read_intra_y_mode().  Repeats within one 64x64 block are no-ops.
      if (config_.cdef.enable &&
          !av2enc::cuda::still::cdef_encode_block_index(
              &cdef_grid_, config_.cdef.num_strengths,
              /*sb_width_log2=*/(plan_.sb_size == 128 ? 5 : 4),
              /*tile_mi_col_start=*/0, /*tile_mi_row_start=*/0, r, c,
              node.cdef_idx, cdef_cdfs_,
              [this](std::uint16_t* cdf, int n, int symbol) {
                return sym(cdf, n, symbol);
              })) {
        error = "cdef_idx is not signallable (block mi " +
                std::to_string(r) + "," + std::to_string(c) + ", idx " +
                std::to_string(static_cast<int>(node.cdef_idx)) + ")";
        return false;
      }
      if (!write_ccso(node, error)) return false;
      // §5.20.5.11 read_delta_qindex(): after read_cdef/read_ccso, before the
      // mode info.  Only the first leaf of a superblock actually codes it; the
      // module clears ReadDeltas and later calls are no-ops.  The neutral plan
      // always codes a zero reduced delta, so qindex does not move.
      if (!deltaq::write_delta_qindex(
              enc_, deltaq_cdfs_.delta_q, tile_qp_,
              w == plan_.sb_size && h == plan_.sb_size, /*skip_flag=*/false,
              /*reduced_delta=*/0, &dq_error)) {
        error = "delta_qindex: " + dq_error;
        return false;
      }
    }
    if (!chroma_part) {
      if (is_ibc) {
        // §5.20.5.4 intrabc_info(): intrabc_mode, DRL and assign_mv.
        if (!write_intrabc_info(node, error)) return false;
      } else {
        if (!code_y_mode(node)) {
          error = error_;
          return false;
        }
        // §5.20.5.5: after the Y mode/angle delta and before mrl_index, every
        // luma coding block with Block_Width/Height <= FSC_MAX codes fsc_mode
        // when enable_idtx_intra is 1.  This encoder never uses FSC
        // (enable_fsc = 0), so it always codes 0; all neighbours are therefore
        // 0 and the CDF context is 0.  The symbol is independent of the TX
        // partition and coded once per coding block.
        if (config_.enable_idtx_intra && w <= 32 && h <= 32) {
          const int group = kFscBsizeGroups[static_cast<int>(size)];
          if (!sym(&cdfs_.fsc_mode[0][group][0], 2, 0)) return false;
        }
        // §5.20.5.5: mrl_index (and mrl_sec_index when > 0) follow the Y-mode
        // symbols and precede read_intra_uv_mode().  The adapter also updates
        // the UsesMrls grid for non-directional blocks, exactly like
        // decode_block().
        if (config_.enable_mrls) {
          const mrl_stream::MrlBlockSyntax mrl_block{r, c, w4, h4};
          const mrl_stream::MrlDecision mrl_decision{
              static_cast<int>(node.mrl_index), node.mrl_sec_index != 0};
          const mrl_stream::MrlSyntaxStatus status =
              mrl_stream::write_mrl_syntax(enc_, cdfs_.mrl, cdfs_.mrl_context,
                                           mrl_block, node.y_mode, mrl_decision,
                                           &error_);
          if (status != mrl_stream::MrlSyntaxStatus::kOk) {
            error = "mrl syntax: " + std::string(
                                        mrl_stream::mrl_syntax_status_name(
                                            status));
            if (!error_.empty()) error += ": " + error_;
            return false;
          }
        } else if (node.mrl_index != 0 || node.mrl_sec_index != 0) {
          error = "MRL decisions supplied but the sequence header disables MRL";
          return false;
        }
      }
    }
    // read_intrabc_info() bypasses read_intra_uv_mode/palette/DIP.
    if (has_chroma && !is_ibc && !code_uv_mode(node, cw, ch)) {
      error = error_;
      return false;
    }
    if (!chroma_part && !is_ibc) {
      if (!write_palette(node, w, h, error)) return false;
      if (!write_dip(node, w, h, error)) return false;
    }
    // §5.20.5.3: skip_flag = 1 codes no residual at all; skip_flag = 0 uses the
    // normal three-plane residual on top of the IBC prediction.
    if (!is_ibc || node.ibc_skip == 0) {
      std::size_t level_base = node.levels_offset;
      if (!chroma_part) {
        if (config_.tx_mode_select) {
          // §5.20.6.1 read_tx_size(allowSelect): intra always passes
          // allowSelect = 1, and a BLOCK_4X4 coding block skips the partition
          // syntax entirely.
          TxBlockView luma_views[kMaxLumaTxParts];
          int luma_parts = 0;
          using av2enc::video::txpartition::SplitGeometry;
          using av2enc::video::txpartition::TxPartition;
          const auto decision = static_cast<TxPartition>(node.tx_partition);
          if (node.bsize != static_cast<std::uint8_t>(
                                  av2enc::video::txpartition::kBlock4x4)) {
            av2enc::video::txpartition::BlockTxContext ctx;
            ctx.mi_size = node.bsize;
            ctx.is_inter = false;
            ctx.fsc_mode = false;
            ctx.reduced_tx_part_set = config_.reduced_tx_part_set;
            TxSink sink(this);
            SplitGeometry relative;
            if (!av2enc::video::txpartition::write_tx_partition(
                    sink, cdfs_.tx_partition, ctx, decision, &relative,
                    error)) {
              return false;
            }
            SplitGeometry geometry;
            std::string geometry_error;
            if (!av2enc::video::txpartition::tx_partition_geometry(
                    node.bsize, node.mi_row, node.mi_col, decision, &geometry,
                    &geometry_error)) {
              error = "tx_partition_geometry: " + geometry_error;
              return false;
            }
            luma_parts = geometry.count;
            if (node.tx_part_count != 0 && node.tx_part_count != luma_parts) {
              error =
                  "luma TX part count does not match the signalled partition";
              return false;
            }
            for (int i = 0; i < luma_parts; ++i) {
              const LumaTxPart& part =
                  node.tx_part_count != 0 ? node.tx_parts[i] : LumaTxPart{};
              if (part.tx_size != geometry.parts[i].tx_size &&
                  node.tx_part_count != 0) {
                error =
                    "luma TX part size does not match the signalled geometry";
                return false;
              }
              luma_views[i].tx_size = geometry.parts[i].tx_size;
              luma_views[i].tx_type =
                  node.tx_part_count != 0 ? part.tx_type : node.tx_type[0];
              luma_views[i].sec_tx_type =
                  node.tx_part_count != 0 ? part.sec_tx_type : 0;
              luma_views[i].stx_set =
                  node.tx_part_count != 0 ? part.most_probable_stx_set : 0;
              luma_views[i].eob =
                  node.tx_part_count != 0 ? static_cast<int>(part.eob)
                                          : static_cast<int>(node.eob[0]);
              luma_views[i].mi_row = geometry.parts[i].mi_row;
              luma_views[i].mi_col = geometry.parts[i].mi_col;
            }
          } else {
            luma_parts = 1;
            luma_views[0].tx_size = node.tx_size[0];
            luma_views[0].tx_type = node.tx_type[0];
            luma_views[0].sec_tx_type = node.sec_tx_type;
            luma_views[0].stx_set = node.most_probable_stx_set;
            luma_views[0].eob = static_cast<int>(node.eob[0]);
            luma_views[0].mi_row = node.mi_row;
            luma_views[0].mi_col = node.mi_col;
          }
          for (int i = 0; i < luma_parts; ++i) {
            if (!code_tx(node, 0, luma_views[i], level_base, w, h, levels_end,
                         error)) {
              if (error.empty()) error = error_;
              return false;
            }
            level_base += static_cast<std::size_t>(luma_views[i].eob);
          }
        } else {
          const TxBlockView luma_view{node.tx_size[0], node.tx_type[0],
                                      node.sec_tx_type,
                                      node.most_probable_stx_set,
                                      static_cast<int>(node.eob[0]),
                                      node.mi_row, node.mi_col};
          if (!code_tx(node, 0, luma_view, level_base, w, h, levels_end,
                       error)) {
            if (error.empty()) error = error_;
            return false;
          }
          level_base += node.eob[0];
        }
      }
      if (has_chroma) {
        const TxBlockView u_view{node.tx_size[1], node.tx_type[1], 0, 0,
                                 static_cast<int>(node.eob[1]), node.mi_row,
                                 node.mi_col};
        if (!code_tx(node, 1, u_view, level_base, cw, ch, levels_end, error)) {
          if (error.empty()) error = error_;
          return false;
        }
        level_base += node.eob[1];
        const TxBlockView v_view{node.tx_size[2], node.tx_type[2], 0, 0,
                                 static_cast<int>(node.eob[2]), node.mi_row,
                                 node.mi_col};
        if (!code_tx(node, 2, v_view, level_base, cw, ch, levels_end, error)) {
          if (error.empty()) error = error_;
          return false;
        }
        level_base += node.eob[2];
      }
    } else {
      reset_skip_block_context(node, w, h, cw, ch);
    }
    if (error.empty() && !error_.empty()) error = error_;

    // Record the segment id for the neighbours' prediction contexts (§5.20.5.7).
    if (!chroma_part &&
        !deltaq::commit_segment_block(seg_grid_, segment_block, segment_id,
                                      &dq_error)) {
      error = "commit_segment_block: " + dq_error;
      return false;
    }

    // Update partition/mode state after the block.
    if (!chroma_part) {
      update_usedip(node, w4, h4);
      for (int yy = 0; yy < h4; ++yy) {
        left_mi(r + yy, 0) = kMiHeightLog2[static_cast<int>(size)];
      }
      for (int xx = 0; xx < w4; ++xx) {
        above_mi(c + xx, 0) = kMiWidthLog2[static_cast<int>(size)];
      }
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
    }
    const int cr = cur_chroma_r_;
    const int cc0 = cur_chroma_c_;
    const int cw4 = cw / 4;
    const int ch4 = ch / 4;
    const int chroma_mi_rows = mi_rows_ / 2;
    const int chroma_mi_cols = mi_cols_ / 2;
    const bool is_cfl = (node.flags & kFlagIsCfl) != 0;
    if (!has_chroma) {
      // Non-owning sub-8 leaf: the owning leaf writes the whole inherited
      // chroma block (including the CfL flag grid).
      return true;
    }
    if (chroma_part) {
      // The chroma tree's neighbour contexts are derived from the chroma plane
      // grid; fill the owned ChromaMi extent with ChromaMiSize's log2.
      const int cbw4 = partition::block_width(chroma_mi_size) / 4;
      const int cbh4 = partition::block_height(chroma_mi_size) / 4;
      const int log2_w = kMiWidthLog2[static_cast<int>(chroma_mi_size)];
      const int log2_h = kMiHeightLog2[static_cast<int>(chroma_mi_size)];
      for (int yy = 0; yy < cbh4; ++yy) {
        const int rr = node.chroma_mi_row + yy;
        if (rr >= mi_rows_) break;
        left_mi(rr, 1) = log2_h;
      }
      for (int xx = 0; xx < cbw4; ++xx) {
        const int cc = node.chroma_mi_col + xx;
        if (cc >= mi_cols_) break;
        above_mi(cc, 1) = log2_w;
      }
    }
    for (int yy = 0; yy < ch4; ++yy) {
      const int rr = cr + yy;
      if (rr >= chroma_mi_rows) break;
      for (int xx = 0; xx < cw4; ++xx) {
        const int cc = cc0 + xx;
        if (cc >= chroma_mi_cols) break;
        // uv_smooth_ is prediction state and is not part of the syntax; only
        // the CfL flag grid feeds the is_cfl context.
        uvcfl(rr, cc) = is_cfl ? 1 : 0;
      }
    }
    if (config_.allow_intrabc) {
      const std::uint8_t ibc_value = is_ibc ? 1 : 0;
      const std::uint8_t skip_value =
          (is_ibc && node.ibc_skip != 0) ? 1 : 0;
      for (int yy = 0; yy < h4; ++yy) {
        const int rr = r + yy;
        if (rr >= mi_rows_) break;
        for (int xx = 0; xx < w4; ++xx) {
          const int cc = c + xx;
          if (cc >= mi_cols_) break;
          const std::size_t idx = static_cast<std::size_t>(rr) *
                                      static_cast<std::size_t>(mi_cols_) +
                                  static_cast<std::size_t>(cc);
          ibc_grid_[idx] = ibc_value;
          skip_grid_[idx] = skip_value;
        }
      }
      ibcs::finish_ibc_block(ibc_buffer_, ibc_neighbours_, r, c, node.bsize,
                             is_ibc,
                             ibc::BlockVector{node.ibc_bv_row,
                                              node.ibc_bv_col});
    }
    return true;
  }

  // Members
  const FramePlan& plan_;
  std::span<const BlockSyntax> blocks_;
  std::span<const std::int32_t> levels_;
  const StillConfig& config_;
  FrameGeometry geometry_;
  int mi_cols_ = 0;
  int mi_rows_ = 0;

  CdfState cdfs_{};
  // Extended-partition CDF copies (five §9.3 arrays) and the sequence-header
  // tool flags.  Used only when config_.enable_ext_partitions is set.
  ps::TilePartitionCdfs ext_cdfs_{};
  pe::Tools tools_{};
  // §5.18.7.1 / §5.20.5.7/.11: neutral delta-Q / segmentation state.  Everything
  // is inert when both tools are disabled.
  deltaq::BlockCdfs deltaq_cdfs_{};
  deltaq::TileQp tile_qp_{};
  deltaq::SegmentGrid seg_grid_{};
  deltaq::FrameSegmentation seg_{};
  std::string deltaq_init_error_;
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
  // Partition-context grids for the CHROMA_PART tree (PlaneStart 1).  Sizes are
  // ChromaMiSize in luma 4x4 units, matching the spec's chroma-plane MiSizes.
  std::vector<int> above_mi_chroma_;
  std::vector<int> left_mi_chroma_;
  std::vector<int> intra_joint_;
  std::vector<std::uint8_t> uvcfl_;
  // §5.20.7.9 UseDip flag grid (luma MI) for the TileUseDipCdf context.
  std::vector<std::uint8_t> usedip_;
  // §5.20.8 palette state: side array, per-frame writer (owns the per-MI
  // palette sizes/colours for get_palette_cache) and its CDF copies.
  std::span<const palette::PaletteResult> palette_results_;
  std::span<const std::uint16_t> cdf_seed_;
  std::vector<std::uint16_t>* cdf_final_ = nullptr;
  std::optional<palette_stream::PaletteSyntaxWriter> palette_writer_;
  palette::PaletteCdfState palette_cdfs_{};
  std::string palette_reset_error_;

  // §5.20.5.3/.4 intra block copy state: CDFs, per-MI mode/skip grids, the
  // exact §7.12.2 BV stack and the §5.20.5.1 buffer bookkeeping.
  ibcs::IbcCdfs ibc_cdfs_{};
  ibcs::ScreenContentConfig ibc_content_{};
  ibc::IbcConfig ibc_config_{};
  ibc::IbcBufferState ibc_buffer_;
  ibcs::BvNeighbourMap ibc_neighbours_;
  std::vector<std::uint8_t> ibc_grid_;
  std::vector<std::uint8_t> skip_grid_;

  // §5.20.10.1 CDEF index state (fresh CDF copies so adaptive mode can update).
  av2enc::cuda::still::CdefIndexGrid cdef_grid_;
  av2enc::cuda::still::CdefIndexCdfs cdef_cdfs_{};
  std::uint16_t cdef_index0_[4][3] = {};
  std::uint16_t cdef_minus1_[8] = {};

  // §5.20.10.4 read_lr() state (one tile, intra, per-unit Wiener NS).
  restoration::classes::WienerTileState lr_state_;
  bool lr_ready_ = false;
  // §7.20.4 PC-Wiener uses the self-contained stream module instead.
  av2enc::filters::pc_wiener_stream::TileState pc_state_;
  bool pc_ready_ = false;

  int cur_r_ = 0;
  int cur_c_ = 0;
  int cur_w4_ = 0;
  int cur_h4_ = 0;
  // Chroma 4x4 origin of the block whose chroma is being coded (ChromaMi for
  // sub-8 owners, luma/2 for the historical producers).
  int cur_chroma_r_ = 0;
  int cur_chroma_c_ = 0;
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
  init_cdfs(state, qctx::slice_for_base_q_idx(base_q_idx),
            /*reduced_tx_part_set=*/false);
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

bool assemble_still_frame(const FramePlan& plan,
                       std::span<const BlockSyntax> blocks,
                       std::span<const std::int32_t> levels,
                       const StillConfig& config, std::vector<std::uint8_t>& out,
                       std::string& error, std::size_t* tile_bytes,
                       std::span<const palette::PaletteResult> palette_results,
                       std::span<const std::uint16_t> cdf_seed,
                       std::vector<std::uint16_t>* cdf_final) {
  error.clear();
  out.clear();
  if (!cdf_seed.empty() && cdf_seed.size() != cdf_image_words_impl()) {
    error = "CDF seed has the wrong number of entries";
    return false;
  }
  if (plan.width <= 0 || plan.height <= 0) {
    error = "empty frame plan";
    return false;
  }
  if ((plan.width & 1) != 0 || (plan.height & 1) != 0) {
    error = "M1 4:2:0 requires even dimensions";
    return false;
  }
  if (plan.sb_size != 64 && plan.sb_size != 128) {
    error = "M1 supports 64x64 or 128x128 superblocks";
    return false;
  }
  // The frame header is written from the config; a plan/config mismatch would
  // produce a stream whose MiCols/MiRows disagree with the coded tree.
  if (config.max_frame_width != static_cast<std::uint16_t>(plan.width) ||
      config.max_frame_height != static_cast<std::uint16_t>(plan.height)) {
    error = "frame plan size does not match the sequence/frame header config";
    return false;
  }
  if (plan.sb_size != superblock_size_px(config)) {
    error = "frame plan superblock size does not match the sequence header";
    return false;
  }

  Writer writer(plan, blocks, levels, config, palette_results, cdf_seed,
                cdf_final);
  if (!writer.run(tile_bytes, error)) return false;
  out = assemble_still_picture(config, writer.payload());
  if (out.empty()) {
    error = "temporal-unit assembly failed";
    return false;
  }
  return true;
}

int y_mode_joint(int av2_mode, int angle_delta) {
  return y_mode_code(av2_mode, angle_delta).joint;
}

int partition_child_slots(int mi_row, int mi_col, int bsize, int partition,
                          int mi_rows, int mi_cols, ChildSlot out[4]) {
  SlotInfo slots[4];
  const int count = enumerate_slots(mi_row, mi_col,
                                    static_cast<BlockSize>(bsize),
                                    static_cast<PartitionType>(partition),
                                    mi_rows, mi_cols, slots);
  for (int i = 0; i < count; ++i) {
    out[i].mi_row = static_cast<std::uint16_t>(slots[i].mi_row);
    out[i].mi_col = static_cast<std::uint16_t>(slots[i].mi_col);
    out[i].bsize = static_cast<std::uint8_t>(slots[i].size);
  }
  return count;
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

}  // namespace av2enc::still::syntax
