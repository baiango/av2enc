#include "formats/chroma/chroma_still.h"

#include "formats/chroma/chroma_headers.h"
#include "formats/chroma/chroma_filters.h"
#include "formats/chroma/chroma_residual.h"
#include "pipeline/still/basic_cdf_tables.h"
#include "codec/entropy/av2/av2_entropy.h"
#include "codec/filters/stream/filter_cdfs.h"
#include "cuda/pipeline/still/cdef/still_cdef_cdf.h"
#include "cuda/pipeline/still/cdef/still_cdef_index_syntax.h"
#include "formats/high_bitdepth/hbd_lr_syntax.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::chroma {
namespace {

constexpr std::size_t kTileCapacityBytes = 8U * 1024U * 1024U;

// ---------------------------------------------------------------------------
// CDF state.  Same layout as the M0 still writer; only the qctx-0 slice and
// frozen CDFs are used (base_q_idx = 90, disable_cdf_update = 1).
// ---------------------------------------------------------------------------

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

void init_cdfs(CdfState& c) {
  copy_array(c.do_split, av2enc::basic::kDoSplitCdf);
  copy_array(c.rect_type, av2enc::basic::kRectTypeCdf);
  copy_array(c.y_mode_set, av2enc::basic::kYModeSetCdf);
  copy_array(c.y_mode_index, av2enc::basic::kYModeIndexCdf);
  copy_array(c.uv_mode, av2enc::basic::kUvModeCflNotAllowedCdf);
  copy_array(c.txb_skip, av2enc::basic::kTxbSkipCdf);
  copy_array(c.v_txb_skip, av2enc::basic::kVTxbSkipCdf);
  copy_array(c.eob_pt_1024, av2enc::basic::kEobPt1024Cdf);
  copy_array(c.eob_pt_512, av2enc::basic::kEobPt512Cdf);
  copy_array(c.eob_pt_256, av2enc::basic::kEobPt256Cdf);
  copy_array(c.eob_pt_128, av2enc::basic::kEobPt128Cdf);
  copy_array(c.eob_pt_64, av2enc::basic::kEobPt64Cdf);
  copy_array(c.eob_extra, av2enc::basic::kEobExtraCdf);
  copy_array(c.coeff_base, av2enc::basic::kCoeffBaseCdf);
  copy_array(c.coeff_base_lf, av2enc::basic::kCoeffBaseLfCdf);
  copy_array(c.coeff_base_eob, av2enc::basic::kCoeffBaseEobCdf);
  copy_array(c.coeff_base_lf_eob, av2enc::basic::kCoeffBaseLfEobCdf);
  copy_array(c.coeff_br, av2enc::basic::kCoeffBrCdf);
  copy_array(c.coeff_br_lf, av2enc::basic::kCoeffBrLfCdf);
  copy_array(c.coeff_base_uv, av2enc::basic::kCoeffBaseUvCdf);
  copy_array(c.coeff_base_lf_uv, av2enc::basic::kCoeffBaseLfUvCdf);
  copy_array(c.coeff_base_eob_uv, av2enc::basic::kCoeffBaseEobUvCdf);
  copy_array(c.coeff_base_lf_eob_uv, av2enc::basic::kCoeffBaseLfEobUvCdf);
  copy_array(c.coeff_br_uv, av2enc::basic::kCoeffBrUvCdf);
  copy_array(c.dc_sign, av2enc::basic::kDcSignCdf);
  copy_array(c.intra_tx_type_set1, av2enc::basic::kIntraTxTypeSet1Cdf);
  copy_array(c.intra_tx_type_long, kDefaultIntraTxTypeLongCdf);
  copy_array(c.is_long_side_dct, kDefaultIsLongSideDctCdf);
}

struct Plane {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> samples;

  void resize(int w, int h, std::uint8_t value) {
    width = w;
    height = h;
    samples.assign(static_cast<std::size_t>(w) *
                       static_cast<std::size_t>(h),
                   value);
  }
  std::uint8_t& at(int x, int y) {
    return samples[static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)];
  }
  std::uint8_t at(int x, int y) const {
    return samples[static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)];
  }
};

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}
int clip1(int value) { return clip3(0, 255, value); }
int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// §9.2 transform size index from the nominal dimensions.  The index order is
// the shared `txv::TxSize` / `deblock::TxSize` enumeration.
int tx_size_index_from_dims(int w, int h) {
  struct Entry {
    int w;
    int h;
  };
  static constexpr Entry kSizes[25] = {
      {4, 4},   {8, 8},   {16, 16}, {32, 32}, {64, 64}, {4, 8},   {8, 4},
      {8, 16},  {16, 8},  {16, 32}, {32, 16}, {32, 64}, {64, 32}, {4, 16},
      {16, 4},  {8, 32},  {32, 8},  {16, 64}, {64, 16}, {4, 32},  {32, 4},
      {8, 64},  {64, 8},  {4, 64},  {64, 4}};
  for (int i = 0; i < 25; ++i) {
    if (kSizes[i].w == w && kSizes[i].h == h) return i;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Tile writer
// ---------------------------------------------------------------------------

class TileWriter {
 public:
  TileWriter(const ChromaPicture& picture, const ChromaStillOptions& options,
             const FrameShape& shape, const ChromaFilterPlan* plan)
      : picture_(picture), shape_(shape), gpu_(options.gpu), plan_(plan) {
    mi_cols_ = shape.mi_cols;
    mi_rows_ = shape.mi_rows;
    padded_mi_cols_ = ((mi_cols_ + 15) / 16) * 16;
    padded_mi_rows_ = ((mi_rows_ + 15) / 16) * 16;

    init_cdfs(cdfs_);
    tile_buffer_.resize(kTileCapacityBytes);
    enc_.init_symbol(tile_buffer_.data(), tile_buffer_.size());
    enc_.disable_cdf_update = options.disable_cdf_update ? 1 : 0;

    // §7.18/§7.19 filter symbol CDFs.  Frozen, like the rest of this writer.
    std::memcpy(ccso_cdf_, av2enc::still::filters_stream::kCcsoBlkCdf,
                sizeof(ccso_cdf_));
    if (plan_ != nullptr && plan_->cdef.enabled &&
        plan_->cdef.num_strengths > 1) {
      cdef_grid_.reset(mi_cols_, mi_rows_);
      std::memcpy(cdef_index0_, av2enc::cuda::still::kCdefIndex0Cdf,
                  sizeof(cdef_index0_));
      cdef_cdfs_.index0 = &cdef_index0_[0][0];
      cdef_cdfs_.minus1 = cdef_minus1_;
      const std::uint16_t* minus1 =
          av2enc::cuda::still::cdef_index_minus1_default(
              plan_->cdef.num_strengths);
      if (minus1 != nullptr) {
        std::memcpy(cdef_minus1_, minus1,
                    sizeof(std::uint16_t) *
                        static_cast<std::size_t>(plan_->cdef.num_strengths));
      }
    }
    if (plan_ != nullptr && plan_->lr.config.enabled) {
      namespace lrs = av2enc::hbd::lr_syntax;
      lrs::LrParams params;
      av2enc::hbd::fill_lr_params(plan_->lr.config, shape.num_planes, &params);
      lrs::LrHeaderContext context;
      av2enc::hbd::fill_lr_header_context(
          plan_->lr.config, shape.num_planes, shape.info.subsampling_x,
          shape.info.subsampling_y, options.base_q_idx, &context);
      std::string lr_error;
      lr_ready_ = lrs::InitTileState(&lr_state_, params, mi_cols_, mi_rows_,
                                     shape.info.subsampling_x,
                                     shape.info.subsampling_y, &lr_error);
      if (!lr_ready_) error_ = "LR tile init failed: " + lr_error;
    }

    for (int plane = 0; plane < 3; ++plane) {
      const PlaneShape& p = shape_.planes[static_cast<std::size_t>(plane)];
      if (!p.present) {
        continue;
      }
      recon_[plane].resize(p.padded_width, p.padded_height, 128);
      const int plane_mi_cols = p.padded_width >> 2;
      const int plane_mi_rows = p.padded_height >> 2;
      above_level_[plane].assign(static_cast<std::size_t>(plane_mi_cols) + 8U,
                                 0);
      left_level_[plane].assign(static_cast<std::size_t>(plane_mi_rows) + 8U,
                                0);
      above_dc_[plane].assign(static_cast<std::size_t>(plane_mi_cols) + 8U, 0);
      left_dc_[plane].assign(static_cast<std::size_t>(plane_mi_rows) + 8U, 0);
    }
    above_mi_.assign(static_cast<std::size_t>(mi_cols_) + 8U, 6);
    left_mi_.assign(static_cast<std::size_t>(mi_rows_) + 8U, 6);
    intra_joint_.assign(static_cast<std::size_t>(padded_mi_rows_) *
                            static_cast<std::size_t>(padded_mi_cols_),
                        0);
    clear_above_context();
  }

  bool encode(std::vector<std::uint8_t>* tile_payload, std::string& error) {
    if (gpu_ != nullptr && !gpu_->begin(picture_, error)) {
      return false;
    }
    clear_above_context();
    for (int r = 0; r < mi_rows_; r += 16) {
      clear_left_context();
      for (int c = 0; c < mi_cols_; c += 16) {
        if (!encode_superblock(r, c)) {
          error = error_;
          return false;
        }
        ++leaves_;
      }
    }
    std::size_t payload_bytes = 0;
    const av2::Status status = enc_.exit_symbol(&payload_bytes);
    if (status != av2::Status::kOk) {
      error = std::string("entropy exit_symbol failed: ") +
              av2::status_name(status);
      return false;
    }
    if (gpu_ != nullptr) {
      for (int plane = 0; plane < 3; ++plane) {
        if (!shape_.planes[static_cast<std::size_t>(plane)].present) {
          continue;
        }
        if (!gpu_->download_recon(plane, recon_[plane].samples.data(),
                                  error)) {
          return false;
        }
      }
    }
    tile_payload->assign(tile_buffer_.begin(),
                         tile_buffer_.begin() +
                             static_cast<std::ptrdiff_t>(payload_bytes));
    error.clear();
    return true;
  }

  const Plane& recon(int plane) const { return recon_[plane]; }
  Plane& recon_mut(int plane) { return recon_[plane]; }
  int leaves() const { return leaves_; }
  const std::vector<ChromaFilterBlock>& leaf_info() const {
    return leaves_info_;
  }

 private:
  bool sym(std::uint16_t* cdf, int n, int symbol) {
    const av2::Status status = enc_.encode_symbol(cdf, n, symbol);
    if (status != av2::Status::kOk) {
      error_ = std::string("entropy encode_symbol failed: ") +
               av2::status_name(status);
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

  void clear_above_context() {
    for (int p = 0; p < 3; ++p) {
      std::fill(above_level_[p].begin(), above_level_[p].end(), 0);
      std::fill(above_dc_[p].begin(), above_dc_[p].end(), 0);
    }
    std::fill(above_mi_.begin(), above_mi_.end(), 6);
  }
  void clear_left_context() {
    for (int p = 0; p < 3; ++p) {
      std::fill(left_level_[p].begin(), left_level_[p].end(), 0);
      std::fill(left_dc_[p].begin(), left_dc_[p].end(), 0);
    }
    std::fill(left_mi_.begin(), left_mi_.end(), 6);
  }

  std::uint8_t& above_level(int p, int x) {
    return above_level_[p][static_cast<std::size_t>(x)];
  }
  std::uint8_t& left_level(int p, int y) {
    return left_level_[p][static_cast<std::size_t>(y)];
  }
  std::uint8_t& above_dc(int p, int x) {
    return above_dc_[p][static_cast<std::size_t>(x)];
  }
  std::uint8_t& left_dc(int p, int y) {
    return left_dc_[p][static_cast<std::size_t>(y)];
  }
  std::uint8_t& above_mi(int x) {
    return above_mi_[static_cast<std::size_t>(x)];
  }
  std::uint8_t& left_mi(int y) {
    return left_mi_[static_cast<std::size_t>(y)];
  }
  std::uint8_t& intra_joint(int r, int c) {
    return intra_joint_[static_cast<std::size_t>(r) *
                            static_cast<std::size_t>(padded_mi_cols_) +
                        static_cast<std::size_t>(c)];
  }

  bool is_inside(int r, int c) const {
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
    if (!is_inside(mv_row, mv_col)) {
      return 0;
    }
    return intra_joint_[static_cast<std::size_t>(mv_row) *
                            static_cast<std::size_t>(padded_mi_cols_) +
                        static_cast<std::size_t>(mv_col)];
  }

  // §5.20.3.2 PARTITION_NONE at BLOCK_64X64, then §5.20.5.3 mode info, then
  // §5.20.7 residual for plane 0 and (if present) planes 1/2.
  bool encode_superblock(int r, int c) {
    // §5.20.10.4 read_lr() precedes read_partition().
    if (!write_lr(r, c)) return false;
    // do_split context: Partition_Size_Adjust[BLOCK_64X64] = 3, bsw/bsh = 4.
    const int ctx = 3 * 4 + (left_mi(r) < 4 ? 2 : 0) +
                    (above_mi(c) < 4 ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][ctx][0], 2, 0)) {
      return false;
    }
    // §5.20.10.1/§5.20.10.2: read_cdef()/read_ccso() come after
    // intra_segment_id (absent: no segmentation) and before the mode info.
    if (!write_cdef(r, c)) return false;
    if (!write_ccso(r, c)) return false;

    // intra_frame_mode_info() for DC_PRED.
    if (!sym(&cdfs_.y_mode_set[0], 4, 0)) {
      return false;
    }
    const int joint_ctx = (get_joint_mode(r, c, 16, 16, 0) >= 5 ? 1 : 0) +
                          (get_joint_mode(r, c, 16, 16, 1) >= 5 ? 1 : 0);
    if (!sym(&cdfs_.y_mode_index[joint_ctx][0], 8, 0)) {
      return false;
    }
    if (!shape_.monochrome) {
      if (!sym(&cdfs_.uv_mode[0][0], 8, 0)) {
        return false;
      }
    }

    // TX_MODE_LARGEST: luma residual is BLOCK_64X64.
    if (!encode_residual_plane(0, r, c, 64, 64)) {
      return false;
    }
    if (!shape_.monochrome) {
      const auto chroma_size = chroma_mi_size(BlockSize::k64x64, shape_.chroma);
      if (!chroma_size) {
        error_ = "Subsampled_Size has no legal BLOCK_64X64 chroma residual";
        return false;
      }
      const int cw = fmt::block_width(*chroma_size);
      const int ch = fmt::block_height(*chroma_size);
      if (!encode_residual_plane(1, r, c, cw, ch)) {
        return false;
      }
      if (!encode_residual_plane(2, r, c, cw, ch)) {
        return false;
      }
    }

    for (int y = 0; y < 16; ++y) {
      left_mi(r + y) = 4;
    }
    for (int x = 0; x < 16; ++x) {
      above_mi(c + x) = 4;
    }
    for (int y = 0; y < 16; ++y) {
      for (int x = 0; x < 16; ++x) {
        intra_joint(r + y, c + x) = 0;
      }
    }
    ChromaFilterBlock leaf;
    leaf.mi_row = r;
    leaf.mi_col = c;
    leaf.luma_size = 64;
    leaf.luma_eob_zero = (luma_eob_ == 0);
    leaf.tx_size[0] = tx_size_index_from_dims(64, 64);
    if (!shape_.monochrome) {
      const auto chroma_size = chroma_mi_size(BlockSize::k64x64, shape_.chroma);
      if (chroma_size) {
        const int cw = fmt::block_width(*chroma_size);
        const int ch = fmt::block_height(*chroma_size);
        leaf.tx_size[1] = tx_size_index_from_dims(cw, ch);
        leaf.tx_size[2] = leaf.tx_size[1];
      }
    }
    leaves_info_.push_back(leaf);
    return true;
  }

  std::uint8_t source_sample(int plane, int x, int y) const {
    const PlaneShape& p = shape_.planes[static_cast<std::size_t>(plane)];
    const int cx = clip3(0, p.padded_width - 1, x);
    const int cy = clip3(0, p.padded_height - 1, y);
    return picture_.planes[plane][static_cast<std::size_t>(cy) *
                                     static_cast<std::size_t>(p.padded_width) +
                                 static_cast<std::size_t>(cx)];
  }

  // §5.20.10.4 read_lr(): before read_partition() at the coding-block origin.
  bool write_lr(int r, int c) {
    if (!lr_ready_) return error_.empty();
    namespace lrs = av2enc::hbd::lr_syntax;
    const av2enc::hbd::HbdLrConfig& cfg = plan_->lr.config;
    for (int plane = 0; plane < shape_.num_planes; ++plane) {
      const av2enc::hbd::HbdLrPlaneConfig& pc = cfg.plane[plane];
      if (pc.frame_type == 0) continue;
      const int sub_x = plane == 0 ? 0 : shape_.info.subsampling_x;
      const int sub_y = plane == 0 ? 0 : shape_.info.subsampling_y;
      const int ox = (c * 4) >> sub_x;
      const int oy = (r * 4) >> sub_y;
      const int unit_size = lr_state_.unit_size[plane];
      if (unit_size <= 0 || ox % unit_size != 0 || oy % unit_size != 0) {
        continue;
      }
      const int unit_row = oy / unit_size;
      const int unit_col = ox / unit_size;
      if (unit_row >= lr_state_.unit_rows[plane] ||
          unit_col >= lr_state_.unit_cols[plane]) {
        continue;
      }
      const std::size_t index =
          static_cast<std::size_t>(unit_row) *
              static_cast<std::size_t>(lr_state_.unit_cols[plane]) +
          static_cast<std::size_t>(unit_col);
      if (index >= pc.unit_type.size()) {
        error_ = "LR unit decision grid is too small";
        return false;
      }
      lrs::UnitDecision decision;
      decision.type = pc.unit_type[index];
      if (decision.type == av2enc::restoration::kRestoreWienerNs) {
        if (index >= pc.unit_subset.size() ||
            index * av2enc::restoration::kWienerNsCoeffStride >=
                pc.unit_coeffs.size()) {
          error_ = "LR unit coefficient grid is too small";
          return false;
        }
        decision.subset = pc.unit_subset[index];
        decision.uv_sym = index < pc.unit_uv_sym.size() &&
                          pc.unit_uv_sym[index] != 0;
        for (int j = 0; j < av2enc::restoration::kWienerNsCoeffStride; ++j) {
          decision.coeffs[j] =
              pc.unit_coeffs[index * static_cast<std::size_t>(
                                           av2enc::restoration::
                                               kWienerNsCoeffStride) +
                             static_cast<std::size_t>(j)];
        }
      }
      std::string lr_error;
      if (!lrs::WriteUnit(enc_, lr_state_, plane, unit_row, unit_col, decision,
                          &lr_error)) {
        error_ = "LR unit symbol failed: " + lr_error;
        return false;
      }
    }
    return true;
  }

  // §5.20.10.1 read_cdef(): one symbol per 64x64 block origin when
  // CdefStrengths >= 2.
  bool write_cdef(int r, int c) {
    if (plan_ == nullptr || !plan_->cdef.enabled ||
        plan_->cdef.num_strengths <= 1) {
      return true;
    }
    const int n64_cols = (mi_cols_ + 15) / 16;
    const std::size_t index =
        static_cast<std::size_t>(r / 16) * static_cast<std::size_t>(n64_cols) +
        static_cast<std::size_t>(c / 16);
    if (index >= plan_->cdef.index.size()) {
      error_ = "cdef_idx grid is too small";
      return false;
    }
    const int idx = static_cast<int>(plan_->cdef.index[index]);
    const bool ok = av2enc::cuda::still::cdef_encode_block_index(
        &cdef_grid_, plan_->cdef.num_strengths,
        /*sb_width_log2=*/4, /*tile_mi_col_start=*/0,
        /*tile_mi_row_start=*/0, r, c, idx, cdef_cdfs_,
        [this](std::uint16_t* cdf, int n, int symbol) {
          return sym(cdf, n, symbol);
        });
    if (!ok) {
      error_ = "cdef_idx is not signallable";
      return false;
    }
    return true;
  }

  // §5.20.10.2 read_ccso(): one binary symbol per enabled plane at a CCSO unit
  // origin; the context is the left unit's already-coded value.
  bool write_ccso(int r, int c) {
    if (plan_ == nullptr || !plan_->ccso.config.enabled) return true;
    const int unit_mi = 1 << (plan_->ccso.config.ccso_luma_size_log2 - 2);
    if (unit_mi <= 0 || (r & (unit_mi - 1)) != 0 ||
        (c & (unit_mi - 1)) != 0) {
      return true;
    }
    for (int plane = 0; plane < shape_.num_planes; ++plane) {
      if (!plan_->ccso.config.plane[plane].enabled) continue;
      const std::vector<std::uint8_t>& grid =
          plan_->ccso.config.blks[plane];
      const int cols = plan_->ccso.config.blk_cols[plane];
      const int rows = plan_->ccso.config.blk_rows[plane];
      if (cols <= 0 || rows <= 0) {
        error_ = "ccso unit grid is missing";
        return false;
      }
      const int shift = plan_->ccso.config.ccso_luma_size_log2 - 2;
      const int unit_row = r >> shift;
      const int unit_col = c >> shift;
      if (unit_row >= rows || unit_col >= cols) {
        error_ = "ccso unit grid does not cover a coding block";
        return false;
      }
      const int value =
          grid[static_cast<std::size_t>(unit_row) *
                   static_cast<std::size_t>(cols) +
               static_cast<std::size_t>(unit_col)] != 0
              ? 1
              : 0;
      int ctx = 0;
      if (unit_col > 0) {
        const std::uint8_t left =
            grid[static_cast<std::size_t>(unit_row) *
                     static_cast<std::size_t>(cols) +
                 static_cast<std::size_t>(unit_col - 1)];
        ctx = 2 * (left != 0 ? 1 : 0);
      }
      if (!sym(&ccso_cdf_[plane][ctx][0], 2, value)) return false;
    }
    return true;
  }

  bool write_literal(int bits, std::uint32_t value) {    for (int i = bits - 1; i >= 0; --i) {
      if (!raw_bit(((value >> static_cast<unsigned>(i)) & 1U) != 0U)) {
        return false;
      }
    }
    return true;
  }

  // Encoder inverse of read_quant() (§5.20.7.28) for allowTcq = 0 and
  // isHidden = 0.
  bool write_quant_escape(int x) {
    const int pred_level = hr_level_avg_;
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
      for (;;) {
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
    hr_level_avg_ = (x + hr_level_avg_) >> 1;
    return true;
  }

  static int tx_sqr_index(int w, int h) {
    const int m = std::min(w, h);
    if (m >= 32) return 3;
    if (m >= 16) return 2;
    if (m >= 8) return 1;
    return 0;
  }
  static int tx_sqr_up_index(int w, int h) {
    const int m = std::max(w, h);
    if (m >= 64) return 4;
    if (m >= 32) return 3;
    if (m >= 16) return 2;
    if (m >= 8) return 1;
    return 0;
  }

  // §5.20.8.2 transform_type() for intra luma, DC_PRED, DCT_DCT.  The 64x64
  // leaf maps to TX_SET_DCTONLY and codes no bits; the rectangular branches are
  // kept for completeness.
  bool write_luma_tx_type(int w, int h, int eob) {
    if (eob <= 1) {
      return true;
    }
    const int sqr = tx_sqr_index(w, h);
    const int sqr_up = tx_sqr_up_index(w, h);
    int set = 5;  // TX_SET_INTRA_1
    if (sqr_up > 3) {
      set = sqr >= 3 ? 0 : (w > h ? 1 : 2);
    } else if (sqr_up == 3) {
      set = sqr == 3 ? 0 : (w > h ? 3 : 4);
    }
    if (set == 0) {
      return true;
    }
    if (set == 3 || set == 4) {
      if (!sym(&cdfs_.is_long_side_dct[0][0], 2, 1)) return false;
      return sym(&cdfs_.intra_tx_type_long[sqr][0], 4, 0);
    }
    if (set == 1 || set == 2) {
      return sym(&cdfs_.intra_tx_type_long[sqr][0], 4, 0);
    }
    return sym(&cdfs_.intra_tx_type_set1[sqr][0], 7, 0);
  }

  bool write_eob(int eob, int plane, const TxShape& shape) {
    int eob_pt = 1;
    if (eob == 2) {
      eob_pt = 2;
    } else if (eob >= 3) {
      eob_pt = floor_log2(eob - 1) + 2;
    }
    std::uint16_t* cdf = nullptr;
    int escape_bits = 0;
    int cdf_n = 8;
    const int cdf_ctx = plane > 0 ? 2 : 0;
    switch (shape.eob_multisize) {
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
    if (dc_sign < 0) return 1;
    if (dc_sign > 0) return 2;
    return 0;
  }

  bool encode_residual_plane(int plane, int mi_row, int mi_col,
                             int nominal_w, int nominal_h) {
    TxShape tx;
    if (!make_tx_shape(nominal_w, nominal_h, tx)) {
      error_ = "unsupported transform shape " + std::to_string(nominal_w) +
               "x" + std::to_string(nominal_h);
      return false;
    }
    if (tx.eob_multisize < 2) {
      error_ = "the minimal still writer does not have the EOB CDF for a " +
               std::to_string(nominal_w) + "x" + std::to_string(nominal_h) +
               " transform (eob_multisize < 2)";
      return false;
    }

    const PlaneShape& ps = shape_.planes[static_cast<std::size_t>(plane)];
    const int sub_x = plane == 0 ? 0 : ps.subsampling_x;
    const int sub_y = plane == 0 ? 0 : ps.subsampling_y;
    const int x4 = mi_col >> sub_x;
    const int y4 = mi_row >> sub_y;
    const int w4 = nominal_w >> 2;
    const int h4 = nominal_h >> 2;
    // §5.20.7.27 bounds the neighbour-context scans by MiCols>>subX and
    // MiRows>>subY (the coded plane), not by the 64x64-padded plane.
    const int max_x4 = ps.coded_width >> 2;
    const int max_y4 = ps.coded_height >> 2;

    int ctx = 0;
    if (plane == 0) {
      // With TX_MODE_LARGEST the residual block equals the transform block, so
      // §5.20.7.27's `bw == w && bh == h` branch wins and ctx = 0.
      ctx = 0;
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
        // bsize == the residual size here, so bw*bh == w*h and the +3 branch
        // of §5.20.7.27 never fires.
        if (eob_u_ != 0) {
          ctx += 6;
        }
      } else {
        ctx += 6;
      }
    }

    hr_level_avg_ = 0;
    std::uint16_t* skip_cdf =
        plane == 2 ? &cdfs_.v_txb_skip[ctx][0]
                   : &cdfs_.txb_skip[0][tx.tx_size_ctx][ctx][0];

    // Prediction origin in plane samples.
    const int x0 = (mi_col >> sub_x) * 4;
    const int y0 = (mi_row >> sub_y) * 4;
    const bool have_left = is_inside(mi_row, mi_col - 1);
    const bool have_above = is_inside(mi_row - 1, mi_col);
    const std::size_t adj_area =
        static_cast<std::size_t>(tx.adj_w) * static_cast<std::size_t>(tx.adj_h);
    int pred_value = 128;
    std::vector<int> levels(adj_area, 0);
    int eob = 0;
    std::vector<std::uint16_t> scan;
    build_diagonal_scan(tx.adj_w, tx.adj_h, scan);

    if (gpu_ != nullptr) {
      // The 8-bit chroma product GPU block stage owns the reconstructed
      // planes: it runs §7.13.2.10 DC prediction, the forward residual /
      // transform / quantiser, and returns the same levels the CPU writer
      // would produce.
      if (!gpu_->forward(plane, x0, y0, nominal_w, nominal_h, tx, have_left,
                         have_above, &pred_value, levels.data(), &eob,
                         error_)) {
        return false;
      }
    } else {
      // §7.13.2.1 clamps the above/left reference reads to maxX/maxY, which
      // are the *coded* plane extents (MiCols*4 >> subX, MiRows*4 >> subY),
      // not the 64x64-padded reconstruction buffer.  Using the padded extent
      // changes the DC average for right/bottom edge blocks whose transform
      // extends past the coded plane.
      const int plane_max_x = ps.coded_width - 1;
      const int plane_max_y = ps.coded_height - 1;
      int sum_left = 0;
      int sum_above = 0;
      if (have_left) {
        for (int i = 0; i < nominal_h; ++i) {
          sum_left += recon_[plane].at(x0 - 1, clip3(0, plane_max_y, y0 + i));
        }
      }
      if (have_above) {
        for (int j = 0; j < nominal_w; ++j) {
          sum_above += recon_[plane].at(clip3(0, plane_max_x, x0 + j), y0 - 1);
        }
      }
      if (have_left && have_above) {
        pred_value = clip1(approx_divide(sum_left + sum_above,
                                         nominal_w + nominal_h));
      } else if (have_left) {
        pred_value = round2(sum_left, floor_log2(nominal_h));
      } else if (have_above) {
        pred_value = round2(sum_above, floor_log2(nominal_w));
      }

      // Forward residual: average the upscale_x * upscale_y source samples
      // that the decoder's §7.15.4 sample-duplication step will expand into.
      std::vector<std::int16_t> residual_adj(adj_area, 0);
      const int upscale_count = tx.upscale_x * tx.upscale_y;
      const int upscale_shift = floor_log2(upscale_count);
      for (int j = 0; j < tx.adj_h; ++j) {
        for (int i = 0; i < tx.adj_w; ++i) {
          int sum = 0;
          for (int dy = 0; dy < tx.upscale_y; ++dy) {
            for (int dx = 0; dx < tx.upscale_x; ++dx) {
              sum += static_cast<int>(source_sample(
                         plane, x0 + i * tx.upscale_x + dx,
                         y0 + j * tx.upscale_y + dy)) -
                     pred_value;
            }
          }
          residual_adj[static_cast<std::size_t>(j) * tx.adj_w + i] =
              static_cast<std::int16_t>(round2(sum, upscale_shift));
        }
      }

      std::vector<double> coeff(adj_area, 0.0);
      forward_dct(residual_adj.data(), tx.adj_w, tx.adj_h, tx.odd_rect,
                  coeff.data());
      const double quant_factor = quant_factor_for(tx);
      quantize_coefficients(coeff.data(), tx.seg_eob, quant_factor,
                            levels.data());
      for (int c = tx.seg_eob - 1; c >= 0; --c) {
        if (levels[scan[static_cast<std::size_t>(c)]] != 0) {
          eob = c + 1;
          break;
        }
      }
    }

    last_eob_ = eob;
    if (plane == 0) luma_eob_ = eob;
    if (!sym(skip_cdf, 2, eob == 0 ? 1 : 0)) {
      return false;
    }
    if (eob == 0) {
      if (gpu_ != nullptr) {
        if (!gpu_->reconstruct(plane, x0, y0, nominal_w, nominal_h, tx,
                               pred_value, levels.data(), eob, error_)) {
          return false;
        }
      } else {
        for (int j = 0; j < nominal_h; ++j) {
          for (int i = 0; i < nominal_w; ++i) {
            recon_[plane].at(x0 + i, y0 + j) =
                static_cast<std::uint8_t>(pred_value);
          }
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

    if (!write_eob(eob, plane, tx)) {
      return false;
    }
    if (plane == 0 && !write_luma_tx_type(tx.adj_w, tx.adj_h, eob)) {
      return false;
    }

    static constexpr int kSigRef[5][2] = {
        {0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
    static constexpr int kMagRef[3][2] = {{0, 1}, {1, 0}, {1, 1}};

    std::vector<int> level_pre(static_cast<std::size_t>(tx.adj_w) *
                                   static_cast<std::size_t>(tx.adj_h),
                               0);
    for (int c = eob - 1; c >= 0; --c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      const int row = pos / tx.adj_w;
      const int col = pos % tx.adj_w;
      const int abs_value = std::abs(levels[static_cast<std::size_t>(pos)]);
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
        const int area = tx.adj_w * tx.adj_h;
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
            if (!sym(&cdfs_.coeff_base_lf_eob[tx.tx_size_ctx][ctx_eob][0], 5,
                     symbol)) {
              return false;
            }
          } else {
            if (!sym(&cdfs_.coeff_base_eob[tx.tx_size_ctx][ctx_eob][0], 3,
                     symbol)) {
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
          if (rr < tx.adj_h && cc < tx.adj_w) {
            const int limit = is_lf ? 5 : 3;
            mag += std::min(
                level_pre[static_cast<std::size_t>(rr) * tx.adj_w + cc],
                limit);
          }
        }
        const int mag_ctx = (mag + 1) >> 1;
        if (plane == 0) {
          if (is_lf) {
            int lf_ctx = 0;
            if (c == 0) {
              lf_ctx = std::min(mag_ctx, 8);
            } else if (row + col < 2) {
              lf_ctx = std::min(mag_ctx, 6) + 9;
            } else {
              lf_ctx = std::min(mag_ctx, 4) + 16;
            }
            if (!sym(&cdfs_.coeff_base_lf[tx.tx_size_ctx][lf_ctx][0][0], 6,
                     base_level)) {
              return false;
            }
          } else {
            const int ctx2 = std::min(mag_ctx, 4);
            int hf_ctx = ctx2;
            if (row + col >= 6 && row + col < 8) {
              hf_ctx = ctx2 + 5;
            } else if (row + col >= 8) {
              hf_ctx = ctx2 + 10;
            }
            if (!sym(&cdfs_.coeff_base[tx.tx_size_ctx][hf_ctx][0][0], 4,
                     base_level)) {
              return false;
            }
          }
        } else {
          const int ctx2 = std::min(mag_ctx, 3);
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
          if (rr < tx.adj_h && cc < tx.adj_w) {
            mag += std::min(
                level_pre[static_cast<std::size_t>(rr) * tx.adj_w + cc], 5);
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
    }

    // Sign and read_quant() pass, in reverse scan order (§5.20.7.27).
    int cul_level = 0;
    int dc_category = 0;
    for (int c = eob - 1; c >= 0; --c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      const int row = pos / tx.adj_w;
      const int col = pos % tx.adj_w;
      const int pre_level = level_pre[static_cast<std::size_t>(pos)];
      const int abs_value = std::abs(levels[static_cast<std::size_t>(pos)]);
      const bool negative = levels[static_cast<std::size_t>(pos)] < 0;
      if (pre_level != 0) {
        if (row == 0 && col == 0 && plane == 0) {
          const int sign_ctx = dc_sign_context(x4, y4, w4, h4);
          if (!sym(&cdfs_.dc_sign[0][0][sign_ctx][0], 2,
                   negative ? 1 : 0)) {
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

    // Dequant + §7.15.4 inverse transform, then duplicate the adjusted
    // residual back to the nominal residual block size.  The GPU executor
    // runs the same three stages into its device reconstruction plane.
    if (gpu_ != nullptr) {
      if (!gpu_->reconstruct(plane, x0, y0, nominal_w, nominal_h, tx,
                             pred_value, levels.data(), eob, error_)) {
        return false;
      }
    } else {
      std::vector<int> dequant(static_cast<std::size_t>(tx.adj_w) *
                                   static_cast<std::size_t>(tx.adj_h),
                               0);
      dequantize_block(levels.data(), tx.seg_eob, tx.dq_denom, dequant.data());
      std::vector<int> residual_back(static_cast<std::size_t>(tx.adj_w) *
                                         static_cast<std::size_t>(tx.adj_h),
                                     0);
      inverse_dct(dequant.data(), tx, residual_back.data());

      for (int j = 0; j < tx.adj_h; ++j) {
        for (int i = 0; i < tx.adj_w; ++i) {
          const int value = clip1(
              pred_value +
              residual_back[static_cast<std::size_t>(j) * tx.adj_w + i]);
          for (int dy = 0; dy < tx.upscale_y; ++dy) {
            for (int dx = 0; dx < tx.upscale_x; ++dx) {
              recon_[plane].at(x0 + i * tx.upscale_x + dx,
                               y0 + j * tx.upscale_y + dy) =
                  static_cast<std::uint8_t>(value);
            }
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

  const ChromaPicture& picture_;
  FrameShape shape_;
  ChromaGpuExecutor* gpu_ = nullptr;
  const ChromaFilterPlan* plan_ = nullptr;

  int mi_cols_ = 0;
  int mi_rows_ = 0;
  int padded_mi_cols_ = 0;
  int padded_mi_rows_ = 0;

  CdfState cdfs_{};
  av2::Encoder enc_{};
  std::vector<std::uint8_t> tile_buffer_;
  std::string error_;

  Plane recon_[3];
  std::array<std::vector<std::uint8_t>, 3> above_level_;
  std::array<std::vector<std::uint8_t>, 3> left_level_;
  std::array<std::vector<std::uint8_t>, 3> above_dc_;
  std::array<std::vector<std::uint8_t>, 3> left_dc_;
  std::vector<std::uint8_t> above_mi_;
  std::vector<std::uint8_t> left_mi_;
  std::vector<std::uint8_t> intra_joint_;
  int eob_u_ = 0;
  int hr_level_avg_ = 0;
  int last_eob_ = 0;
  int luma_eob_ = 0;
  int leaves_ = 0;
  std::vector<ChromaFilterBlock> leaves_info_;

  // §7.18/§7.19/§7.20 tile symbol state.
  av2enc::cuda::still::CdefIndexGrid cdef_grid_;
  av2enc::cuda::still::CdefIndexCdfs cdef_cdfs_{};
  std::uint16_t cdef_index0_[4][3] = {};
  std::uint16_t cdef_minus1_[8] = {};
  std::uint16_t ccso_cdf_[3][4][3] = {};
  av2enc::hbd::lr_syntax::TileState lr_state_{};
  bool lr_ready_ = false;
};

}  // namespace

// ---------------------------------------------------------------------------
// Input conversion
// ---------------------------------------------------------------------------

bool make_chroma_picture(int width, int height, ChromaFormat chroma,
                         std::span<const std::uint8_t> y,
                         std::span<const std::uint8_t> u,
                         std::span<const std::uint8_t> v,
                         ChromaPicture& picture, std::string& error) {
  const std::uint32_t chroma_width = (static_cast<std::uint32_t>(width) + 1U) / 2U;
  const std::uint32_t chroma_height =
      (static_cast<std::uint32_t>(height) + 1U) / 2U;
  if (y.size() < static_cast<std::size_t>(width) * height ||
      u.size() < static_cast<std::size_t>(chroma_width) * chroma_height ||
      v.size() < static_cast<std::size_t>(chroma_width) * chroma_height) {
    error = "source planes are smaller than the requested dimensions";
    return false;
  }

  fmt::Source420_8 source;
  source.width = static_cast<std::uint32_t>(width);
  source.height = static_cast<std::uint32_t>(height);
  source.y = fmt::ConstPlane8{y.data(), static_cast<std::uint32_t>(width)};
  source.u = fmt::ConstPlane8{u.data(), chroma_width};
  source.v = fmt::ConstPlane8{v.data(), chroma_width};

  fmt::Picture converted;
  const fmt::ConvertResult result =
      fmt::convert_420_8_to_picture(source, chroma, fmt::BitDepth::k8,
                                    converted);
  if (!result.ok()) {
    error = result.error;
    return false;
  }
  return make_chroma_picture(converted, picture, error);
}

bool make_chroma_picture(const fmt::Picture& source, ChromaPicture& picture,
                         std::string& error) {
  if (!fmt::is_internally_supported(source.format)) {
    error = "unsupported source format";
    return false;
  }
  const FrameShape shape =
      make_frame_shape(static_cast<int>(source.format.width),
                       static_cast<int>(source.format.height),
                       source.format.chroma);

  picture = ChromaPicture{};
  picture.width = shape.width;
  picture.height = shape.height;
  picture.chroma = shape.chroma;

  for (int plane = 0; plane < 3; ++plane) {
    const PlaneShape& p = shape.planes[static_cast<std::size_t>(plane)];
    if (!p.present) {
      continue;
    }
    picture.plane_width[plane] = p.padded_width;
    picture.plane_height[plane] = p.padded_height;
    picture.planes[plane].assign(
        static_cast<std::size_t>(p.padded_width) *
            static_cast<std::size_t>(p.padded_height),
        0);
    const fmt::PlaneBuffer& src = source.planes[plane];
    for (int y = 0; y < p.padded_height; ++y) {
      const int sy = std::min(y, p.visible_height - 1);
      for (int x = 0; x < p.padded_width; ++x) {
        const int sx = std::min(x, p.visible_width - 1);
        picture.planes[plane][static_cast<std::size_t>(y) *
                                   static_cast<std::size_t>(p.padded_width) +
                               static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>(src.at(static_cast<std::uint32_t>(sx),
                                             static_cast<std::uint32_t>(sy)));
      }
    }
  }
  error.clear();
  return true;
}

// ---------------------------------------------------------------------------
// Encoder entry point
// ---------------------------------------------------------------------------

bool encode_chroma_still_picture(const ChromaPicture& picture,
                                 const ChromaStillOptions& options,
                                 ChromaStillResult& result, std::string& error) {
  result = ChromaStillResult{};

  std::string shape_error;
  const auto shape = make_frame_shape_checked(picture.width, picture.height,
                                              picture.chroma, shape_error);
  if (!shape) {
    error = shape_error;
    return false;
  }
  if (!supports_64x64_none(*shape, shape_error)) {
    error = shape_error;
    return false;
  }
  if (picture.plane_width[0] != shape->planes[0].padded_width ||
      picture.plane_height[0] != shape->planes[0].padded_height) {
    error = "luma plane dimensions do not match the frame shape";
    return false;
  }
  if (!shape->monochrome) {
    for (int plane = 1; plane < 3; ++plane) {
      if (picture.plane_width[plane] !=
              shape->planes[static_cast<std::size_t>(plane)].padded_width ||
          picture.plane_height[plane] !=
              shape->planes[static_cast<std::size_t>(plane)].padded_height) {
        error = "chroma plane dimensions do not match the frame shape";
        return false;
      }
    }
  }

  const ChromaFilterOptions& requested = options.filters;
  const bool any_filter =
      requested.enable_deblock || requested.enable_cdef ||
      requested.enable_ccso || requested.enable_wiener ||
      requested.enable_chroma_lr || requested.enable_pc_wiener ||
      requested.enable_gdf;
  if (any_filter) {
    if (!options.disable_cdf_update) {
      error = "in-loop filters require disable_cdf_update = 1 (the filter "
              "symbol CDFs are frozen in this writer)";
      return false;
    }
    if (options.gpu != nullptr) {
      error = "in-loop filters are not wired for the 8-bit chroma GPU block "
              "stage; use the CPU path";
      return false;
    }
  }

  // First pass: reconstruct (and collect the leaf metadata the filter plan
  // needs).  Its tile payload is discarded when filters are enabled.
  TileWriter writer(picture, options, *shape, nullptr);
  std::vector<std::uint8_t> tile;
  if (!writer.encode(&tile, error)) {
    return false;
  }

  ChromaFilterPlan filter_plan;
  if (any_filter) {
    ChromaFilterOptions filter_options = requested;
    filter_options.base_q_idx = options.base_q_idx;
    ChromaFilterPlanes planes;
    for (int plane = 0; plane < shape->num_planes; ++plane) {
      Plane& recon = writer.recon_mut(plane);
      planes.plane[plane] = recon.samples.data();
      planes.stride[plane] = recon.width;
      planes.source[plane] = picture.planes[plane].data();
      planes.source_stride[plane] = picture.plane_width[plane];
      planes.visible_w[plane] =
          shape->planes[static_cast<std::size_t>(plane)].visible_width;
      planes.visible_h[plane] =
          shape->planes[static_cast<std::size_t>(plane)].visible_height;
    }
    const fmt::Format format{static_cast<std::uint32_t>(picture.width),
                             static_cast<std::uint32_t>(picture.height),
                             picture.chroma, fmt::BitDepth::k8};
    if (!run_chroma_filters(format, planes, writer.leaf_info(),
                            filter_options, &filter_plan, error)) {
      return false;
    }
    // Second pass: code the same coefficients with the filter syntax inserted
    // at the §5.20.10 hook points.  The reconstruction is identical, so the
    // filtered first-pass planes stay authoritative.
    TileWriter syntax_writer(picture, options, *shape, &filter_plan);
    std::vector<std::uint8_t> filtered_tile;
    if (!syntax_writer.encode(&filtered_tile, error)) {
      return false;
    }
    tile.swap(filtered_tile);
  }

  ChromaSeqConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(picture.width);
  config.max_frame_height = static_cast<std::uint16_t>(picture.height);
  config.seq_level_idx = options.seq_level_idx;
  config.chroma = picture.chroma;
  config.base_q_idx = options.base_q_idx;
  config.disable_cdf_update = options.disable_cdf_update;

  // §5.4.10 / §5.18 filter signalling derived from the same plan the tile
  // writer used.
  ChromaFilterHeaderConfig header;
  const bool header_filters = any_filter && filter_plan.any();
  if (header_filters) {
    if (filter_plan.deblock) {
      header.apply_deblock_y_vert = true;
      header.apply_deblock_y_horz = true;
      header.apply_deblock_u = shape->num_planes > 1;
      header.apply_deblock_v = shape->num_planes > 2;
    }
    if (filter_plan.cdef.enabled) {
      header.enable_cdef = true;
      header.cdef_damping = filter_plan.cdef.damping;
      header.cdef_num_strengths = filter_plan.cdef.num_strengths;
      for (int i = 0; i < filter_plan.cdef.num_strengths; ++i) {
        header.cdef_strengths[i] = filter_plan.cdef.strengths[i];
      }
      config.enable_cdef = true;
    }
    if (filter_plan.ccso.config.enabled) {
      header.enable_ccso = true;
      header.ccso = filter_plan.ccso.config;
      config.enable_ccso = true;
      config.ccso_unit_matches_sb_size =
          filter_plan.ccso.config.unit_matches_sb_size;
    }
    if (filter_plan.lr.config.enabled) {
      header.enable_restoration = true;
      header.lr = filter_plan.lr.config;
      config.enable_restoration = true;
      config.lr_luma_pc = filter_plan.lr.config.plane[0].frame_type ==
                          av2enc::restoration::kRestorePcWiener;
      config.lr_luma_ns = filter_plan.lr.config.plane[0].frame_type ==
                          av2enc::restoration::kRestoreWienerNs;
      config.lr_chroma_ns =
          shape->num_planes > 2 &&
          (filter_plan.lr.config.plane[1].frame_type ==
               av2enc::restoration::kRestoreWienerNs ||
           filter_plan.lr.config.plane[2].frame_type ==
               av2enc::restoration::kRestoreWienerNs);
      config.lr_tools_uv_present = !shape->monochrome;
    }
    if (filter_plan.gdf.config.enabled) {
      header.enable_gdf = true;
      header.gdf_pic_qc_idx = filter_plan.gdf.config.pic_qc_idx;
      header.gdf_pic_scale_idx = filter_plan.gdf.config.pic_scale_idx;
      config.enable_gdf = true;
      config.gdf_unit_matches_sb_size = true;
    }
  }

  if (!validate_chroma_seq_config(config, error)) {
    return false;
  }

  result.frame = assemble_chroma_still(
      config, tile, header_filters ? &header : nullptr);
  result.tile_bytes = static_cast<std::uint32_t>(tile.size());
  result.frame_bytes = static_cast<std::uint32_t>(result.frame.size());
  result.shape = *shape;
  result.leaves = writer.leaves();
  for (int plane = 0; plane < 3; ++plane) {
    if (!shape->planes[static_cast<std::size_t>(plane)].present) {
      continue;
    }
    const Plane& recon = writer.recon(plane);
    result.recon_width[plane] = recon.width;
    result.recon_height[plane] = recon.height;
    result.recon[plane] = recon.samples;
  }
  error.clear();
  return true;
}

bool copy_visible_plane(const ChromaStillResult& result, int plane,
                        std::vector<std::uint8_t>& out) {
  if (plane < 0 || plane >= 3) {
    return false;
  }
  const PlaneShape& p = result.shape.planes[static_cast<std::size_t>(plane)];
  if (!p.present) {
    out.clear();
    return true;
  }
  out.assign(static_cast<std::size_t>(p.visible_width) *
                 static_cast<std::size_t>(p.visible_height),
             0);
  for (int y = 0; y < p.visible_height; ++y) {
    for (int x = 0; x < p.visible_width; ++x) {
      out[static_cast<std::size_t>(y) *
              static_cast<std::size_t>(p.visible_width) +
          static_cast<std::size_t>(x)] =
          result.recon[plane][static_cast<std::size_t>(y) *
                                  static_cast<std::size_t>(result.recon_width[plane]) +
                              static_cast<std::size_t>(x)];
    }
  }
  return true;
}

bool write_recon_y4m(const ChromaStillResult& result, std::FILE* file,
                     std::uint32_t fps_num, std::uint32_t fps_den,
                     std::string& error) {
  if (file == nullptr) {
    error = "null file";
    return false;
  }
  const char* tag = "C420";
  switch (result.shape.chroma) {
    case ChromaFormat::k400:
      tag = "Cmono";
      break;
    case ChromaFormat::k420:
      tag = "C420";
      break;
    case ChromaFormat::k422:
      tag = "C422";
      break;
    case ChromaFormat::k444:
      tag = "C444";
      break;
  }
  if (std::fprintf(file, "YUV4MPEG2 W%d H%d F%u:%u Ip A1:1 %s\n",
                   result.shape.width, result.shape.height, fps_num, fps_den,
                   tag) < 0) {
    error = "failed to write Y4M header";
    return false;
  }
  if (std::fprintf(file, "FRAME\n") < 0) {
    error = "failed to write Y4M frame marker";
    return false;
  }
  const int planes = result.shape.monochrome ? 1 : 3;
  for (int plane = 0; plane < planes; ++plane) {
    std::vector<std::uint8_t> visible;
    if (!copy_visible_plane(result, plane, visible)) {
      error = "invalid plane index";
      return false;
    }
    if (!visible.empty() &&
        std::fwrite(visible.data(), 1, visible.size(), file) != visible.size()) {
      error = "failed to write Y4M plane";
      return false;
    }
  }
  error.clear();
  return true;
}

}  // namespace av2enc::chroma
