#include "codec/entropy/deltaq/still/deltaq.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace av2enc::deltaq {
namespace {

void set_error(std::string* error, const char* message) {
  if (error != nullptr) *error = message;
}

void set_error(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
}

int clip3(int lo, int hi, int value) noexcept {
  return std::max(lo, std::min(hi, value));
}

bool entropy_ok(std::string* error, av2::Status status, const char* what) {
  if (status == av2::Status::kOk) return true;
  set_error(error, std::string(what) + ": " + av2::status_name(status));
  return false;
}

// f(n) / L(n) on the range coder's raw-bit path (§8.2.3/§8.2.5).
bool write_literal(av2::Encoder& encoder, unsigned bits, std::uint32_t value,
                   std::string* error) {
  for (unsigned i = bits; i-- > 0;) {
    const bool bit = ((value >> i) & 1U) != 0U;
    if (!entropy_ok(error, encoder.encode_bool(bit), "encode_bool")) {
      return false;
    }
  }
  return true;
}

bool read_literal(av2::Decoder& decoder, unsigned bits, std::uint32_t& value,
                  std::string* error) {
  std::uint32_t out = 0;
  for (unsigned i = 0; i < bits; ++i) {
    bool bit = false;
    if (!entropy_ok(error, decoder.read_bool(&bit), "read_bool")) return false;
    out = (out << 1U) | (bit ? 1U : 0U);
  }
  value = out;
  return true;
}

bool symbol(av2::Encoder& encoder, std::uint16_t* cdf, int n, int value,
            std::string* error) {
  return entropy_ok(error, encoder.encode_symbol(cdf, n, value),
                    "encode_symbol");
}

bool symbol(av2::Decoder& decoder, std::uint16_t* cdf, int n, int& value,
            std::string* error) {
  return entropy_ok(error, decoder.decode_symbol(cdf, n, &value),
                    "decode_symbol");
}

bool features_empty(const rc::SegmentFeatures& features) noexcept {
  for (int j = 0; j < kSegLvlMax; ++j) {
    if (features.enabled[static_cast<std::size_t>(j)] ||
        features.data[static_cast<std::size_t>(j)] != 0) {
      return false;
    }
  }
  return true;
}

bool validate_seg_info(const rc::SegmentationParams& params, int num_segments,
                       std::string* error) {
  if (num_segments != kMaxSegmentsBase && num_segments != kMaxSegmentsSpec) {
    set_error(error, "segmentation: MaxSegments must be 8 or 16");
    return false;
  }
  for (int i = num_segments; i < kMaxSegmentsSpec; ++i) {
    if (!features_empty(params.segments[static_cast<std::size_t>(i)])) {
      set_error(error,
                "segmentation: segment id >= MaxSegments carries features");
      return false;
    }
  }
  for (int i = 0; i < num_segments; ++i) {
    const rc::SegmentFeatures& features =
        params.segments[static_cast<std::size_t>(i)];
    for (int j = 0; j < kSegLvlMax; ++j) {
      if (!features.enabled[static_cast<std::size_t>(j)]) continue;
      const std::int32_t data = features.data[static_cast<std::size_t>(j)];
      if (j == kSegLvlAltQ) {
        if (data < -kSegAltQLimit || data > kSegAltQLimit) {
          set_error(error, "segmentation: ALT_Q data out of [-351, 351]");
          return false;
        }
      } else if (data != 0) {
        set_error(error, "segmentation: SKIP/GLOBALMV data must be zero");
        return false;
      }
    }
  }
  return true;
}

// seg_info() §5.4.9.  `num_segments` is MaxSegments (8 or 16).
// The feature values have already been validated.
void write_seg_info(BitWriter& writer, const rc::SegmentationParams& params,
                    int num_segments) {
  for (int i = 0; i < num_segments; ++i) {
    const rc::SegmentFeatures& features =
        params.segments[static_cast<std::size_t>(i)];
    for (int j = 0; j < kSegLvlMax; ++j) {
      const bool enabled = features.enabled[static_cast<std::size_t>(j)];
      writer.put_bit(enabled);
      if (!enabled) continue;
      if (j == kSegLvlAltQ) {
        const std::int32_t data = features.data[static_cast<std::size_t>(j)];
        // su(1 + Segmentation_Feature_Bits[ALT_Q]) = su(10), two's complement.
        const std::uint32_t raw =
            static_cast<std::uint32_t>(data) & ((1U << 10U) - 1U);
        writer.put_bits(raw, 10U);
      }
      // Segmentation_Feature_Bits[j] == 0 for SKIP/GLOBALMV: no bits.
    }
  }
}

bool read_seg_info(BitReader& reader, int num_segments,
                   rc::SegmentationParams& params, std::string* error) {
  params.segments = {};
  for (int i = 0; i < num_segments; ++i) {
    rc::SegmentFeatures& features =
        params.segments[static_cast<std::size_t>(i)];
    for (int j = 0; j < kSegLvlMax; ++j) {
      bool enabled = false;
      if (!reader.read_bit(enabled)) {
        set_error(error, "segmentation: truncated seg_info");
        return false;
      }
      features.enabled[static_cast<std::size_t>(j)] = enabled;
      features.data[static_cast<std::size_t>(j)] = 0;
      if (!enabled) continue;
      if (j == kSegLvlAltQ) {
        std::uint32_t raw = 0;
        if (!reader.read_bits(10U, raw)) {
          set_error(error, "segmentation: truncated ALT_Q");
          return false;
        }
        int value = static_cast<int>(raw);
        if ((raw & (1U << 9U)) != 0U) value -= 1024;
        features.data[static_cast<std::size_t>(j)] =
            clip3(-kSegAltQLimit, kSegAltQLimit, value);
      }
      // SKIP / GLOBALMV: zero-bit features, data = 0.
    }
  }
  return true;
}

void segment_neighbours(const SegmentGrid& grid, int mi_row, int mi_col,
                        int& prev_ul, int& prev_u, int& prev_l) noexcept {
  const bool avail_u =
      mi_row - 1 >= grid.tile_mi_row_start && mi_row - 1 < grid.tile_mi_row_end &&
      mi_col >= grid.tile_mi_col_start && mi_col < grid.tile_mi_col_end;
  const bool avail_l =
      mi_row >= grid.tile_mi_row_start && mi_row < grid.tile_mi_row_end &&
      mi_col - 1 >= grid.tile_mi_col_start && mi_col - 1 < grid.tile_mi_col_end;
  prev_ul = -1;
  prev_u = -1;
  prev_l = -1;
  if (avail_u && avail_l) {
    prev_ul = grid.current[static_cast<std::size_t>(mi_row - 1) *
                               static_cast<std::size_t>(grid.mi_cols) +
                           static_cast<std::size_t>(mi_col - 1)];
  }
  if (avail_u) {
    prev_u = grid.current[static_cast<std::size_t>(mi_row - 1) *
                              static_cast<std::size_t>(grid.mi_cols) +
                          static_cast<std::size_t>(mi_col)];
  }
  if (avail_l) {
    prev_l = grid.current[static_cast<std::size_t>(mi_row) *
                              static_cast<std::size_t>(grid.mi_cols) +
                          static_cast<std::size_t>(mi_col - 1)];
  }
}

int segment_prediction_from(int prev_ul, int prev_u, int prev_l) noexcept {
  if (prev_u == -1) return prev_l == -1 ? 0 : prev_l;
  if (prev_l == -1) return prev_u;
  return prev_ul == prev_u ? prev_u : prev_l;
}

void set_pred_contexts(SegmentGrid& grid, const SegmentBlock& block,
                       int value) noexcept {
  for (int i = 0; i < block.bw4; ++i) {
    const int c = block.mi_col + i;
    if (c >= 0 && c < grid.mi_cols) {
      grid.above_pred[static_cast<std::size_t>(c)] =
          static_cast<std::uint8_t>(value);
    }
  }
  for (int i = 0; i < block.bh4; ++i) {
    const int r = block.mi_row + i;
    if (r >= 0 && r < grid.mi_rows) {
      grid.left_pred[static_cast<std::size_t>(r)] =
          static_cast<std::uint8_t>(value);
    }
  }
}

bool write_segment_id_core(av2::Encoder& encoder, BlockCdfs& cdfs,
                           const FrameSegmentation& seg,
                           const SegmentGrid& grid, const SegmentBlock& block,
                           int segment_id, bool skip_flag,
                           std::string* error) {
  int prev_ul = -1;
  int prev_u = -1;
  int prev_l = -1;
  segment_neighbours(grid, block.mi_row, block.mi_col, prev_ul, prev_u, prev_l);
  const int pred = segment_prediction_from(prev_ul, prev_u, prev_l);

  if (skip_flag && !seg.has_lossless_segment) {
    if (segment_id != pred) {
      set_error(error,
                "segmentation: skipped block must reuse the predicted id");
      return false;
    }
    return true;
  }

  int coded = 0;
  if (!neg_interleave(segment_id, pred, seg.last_active_seg_id + 1, coded)) {
    set_error(error, "segmentation: segment id is not codeable for this pred");
    return false;
  }
  const int ctx = segment_id_context(prev_ul, prev_u, prev_l);
  int ext = 0;
  if (grid.enable_ext_seg) {
    ext = coded >= 8 ? 1 : 0;
    if (!symbol(encoder, &cdfs.seg_id_ext_flag[ctx][0], 2, ext, error)) {
      return false;
    }
    coded -= ext * 8;
  } else if (coded >= 8) {
    set_error(error, "segmentation: coded id >= 8 without enable_ext_seg");
    return false;
  }
  // §8.3.2: segment_id uses TileSegmentIdExtCdf[ctx] when the ext flag is 1,
  // and TileSegmentIdCdf[ctx] otherwise.
  std::uint16_t* cdf =
      ext != 0 ? &cdfs.segment_id_ext[ctx][0] : &cdfs.segment_id[ctx][0];
  return symbol(encoder, cdf, 8, coded, error);
}

bool read_segment_id_core(av2::Decoder& decoder, BlockCdfs& cdfs,
                          const FrameSegmentation& seg,
                          const SegmentGrid& grid, const SegmentBlock& block,
                          bool skip_flag, int& segment_id,
                          std::string* error) {
  int prev_ul = -1;
  int prev_u = -1;
  int prev_l = -1;
  segment_neighbours(grid, block.mi_row, block.mi_col, prev_ul, prev_u, prev_l);
  const int pred = segment_prediction_from(prev_ul, prev_u, prev_l);

  if (skip_flag && !seg.has_lossless_segment) {
    segment_id = pred;
    return true;
  }

  const int ctx = segment_id_context(prev_ul, prev_u, prev_l);
  int ext = 0;
  if (grid.enable_ext_seg) {
    if (!symbol(decoder, &cdfs.seg_id_ext_flag[ctx][0], 2, ext, error)) {
      return false;
    }
  }
  // §8.3.2: TileSegmentIdExtCdf[ctx] when the ext flag is 1.
  std::uint16_t* cdf =
      ext != 0 ? &cdfs.segment_id_ext[ctx][0] : &cdfs.segment_id[ctx][0];
  int coded = 0;
  if (!symbol(decoder, cdf, 8, coded, error)) {
    return false;
  }
  coded += ext * 8;
  segment_id = neg_deinterleave(coded, pred, seg.last_active_seg_id + 1);
  if (segment_id < 0 || segment_id >= seg.num_segments) {
    set_error(error, "segmentation: decoded segment id out of range");
    return false;
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Constants and CDF selection
// ---------------------------------------------------------------------------

int max_segments(bool enable_ext_seg) noexcept {
  return enable_ext_seg ? kMaxSegmentsSpec : kMaxSegmentsBase;
}

int coeff_cdf_qctx(int base_q_idx) noexcept {
  // §6.17.2: raw base_q_idx, never an effective/segment/delta qindex.
  const int q = clip3(0, 255, base_q_idx);
  if (q <= 90) return 0;
  if (q <= 140) return 1;
  if (q <= 190) return 2;
  return 3;
}

void BlockCdfs::init() noexcept {
  static_assert(sizeof(delta_q) == sizeof(cdf::kDeltaQCdf));
  std::memcpy(delta_q, cdf::kDeltaQCdf, sizeof(delta_q));
  std::memcpy(segment_id, cdf::kSegmentIdCdf, sizeof(segment_id));
  std::memcpy(segment_id_ext, cdf::kSegmentIdExtCdf, sizeof(segment_id_ext));
  std::memcpy(seg_id_ext_flag, cdf::kSegIdExtFlagCdf,
              sizeof(seg_id_ext_flag));
  std::memcpy(segment_id_predicted, cdf::kSegmentIdPredictedCdf,
              sizeof(segment_id_predicted));
}

CoeffCdfSlice coeff_cdf_slice(int base_q_idx) noexcept {
  const int idx = coeff_cdf_qctx(base_q_idx);
  CoeffCdfSlice slice;
  slice.txb_skip = &coeff::kTxbSkipCdf[idx];
  slice.v_txb_skip = &coeff::kVTxbSkipCdf[idx];
  slice.eob_pt_16 = &coeff::kEobPt16Cdf[idx];
  slice.eob_pt_32 = &coeff::kEobPt32Cdf[idx];
  slice.eob_pt_64 = &coeff::kEobPt64Cdf[idx];
  slice.eob_pt_128 = &coeff::kEobPt128Cdf[idx];
  slice.eob_pt_256 = &coeff::kEobPt256Cdf[idx];
  slice.eob_pt_512 = &coeff::kEobPt512Cdf[idx];
  slice.eob_pt_1024 = &coeff::kEobPt1024Cdf[idx];
  slice.eob_extra = &coeff::kEobExtraCdf[idx];
  slice.dc_sign = &coeff::kDcSignCdf[idx];
  slice.coeff_base = &coeff::kCoeffBaseCdf[idx];
  slice.coeff_base_lf = &coeff::kCoeffBaseLfCdf[idx];
  slice.coeff_base_eob = &coeff::kCoeffBaseEobCdf[idx];
  slice.coeff_base_lf_eob = &coeff::kCoeffBaseLfEobCdf[idx];
  slice.coeff_br = &coeff::kCoeffBrCdf[idx];
  slice.coeff_br_lf = &coeff::kCoeffBrLfCdf[idx];
  slice.coeff_base_uv = &coeff::kCoeffBaseUvCdf[idx];
  slice.coeff_base_lf_uv = &coeff::kCoeffBaseLfUvCdf[idx];
  slice.coeff_base_eob_uv = &coeff::kCoeffBaseEobUvCdf[idx];
  slice.coeff_base_lf_eob_uv = &coeff::kCoeffBaseLfEobUvCdf[idx];
  slice.coeff_br_uv = &coeff::kCoeffBrUvCdf[idx];
  return slice;
}

// ---------------------------------------------------------------------------
// Frame header syntax
// ---------------------------------------------------------------------------

std::size_t write_segmentation_params(BitWriter& writer,
                                      const rc::SegmentationParams& params,
                                      const FrameQpHeader& header,
                                      std::string* error) {
  std::string local;
  std::string* err = &local;
  if (!rc::segmentation_params_valid(params, err)) {
    set_error(error, local);
    return 0;
  }
  const int num_segments = max_segments(header.enable_ext_seg);
  if (!validate_seg_info(params, num_segments, err)) {
    set_error(error, local);
    return 0;
  }

  if (params.enabled && header.primary_ref_is_none &&
      params.temporal_update) {
    set_error(error,
              "segmentation: key frame infers temporal_update = 0");
    return 0;
  }
  if (params.enabled && header.frame_type_switch && params.temporal_update) {
    set_error(error,
              "segmentation: SWITCH_FRAME infers temporal_update = 0");
    return 0;
  }

  const std::size_t start = writer.bit_size();
  writer.put_bit(params.enabled);
  if (!params.enabled) return writer.bit_size() - start;

  // cur_mfh_id == 0 and seq_seg_info_present_flag == 0, so haveSegParams = 0,
  // allowChange = 0, reuse_seg_info = 0: seg_info(MaxSegments) is present.
  write_seg_info(writer, params, num_segments);

  if (header.primary_ref_is_none) {
    // segmentation_update_map = 1; segmentation_temporal_update = 0 (no bits).
  } else {
    writer.put_bit(params.update_map);
    if (params.update_map && !header.frame_type_switch) {
      writer.put_bit(params.temporal_update);
    }
  }
  return writer.bit_size() - start;
}

std::size_t write_delta_q_params(BitWriter& writer,
                                 const rc::DeltaQParams& params,
                                 int base_q_idx, std::string* error) {
  std::string local;
  std::string* err = &local;
  if (base_q_idx < 0 || base_q_idx > 255) {
    set_error(error, "delta_q: base_q_idx must be in [0, 255]");
    return 0;
  }
  if (!rc::delta_q_params_valid(params, err)) {
    set_error(error, local);
    return 0;
  }
  if (base_q_idx == 0 && params.present) {
    set_error(error, "delta_q: base_q_idx == 0 forces delta_q_present = 0");
    return 0;
  }
  const std::size_t start = writer.bit_size();
  if (base_q_idx > 0) {
    writer.put_bit(params.present);
    if (params.present) {
      writer.put_bits(static_cast<std::uint64_t>(params.res), 2U);
    }
  }
  return writer.bit_size() - start;
}

std::size_t write_segmentation_and_delta_q(BitWriter& writer,
                                           const rc::SegmentationParams& seg,
                                           const rc::DeltaQParams& delta_q,
                                           int base_q_idx,
                                           const FrameQpHeader& header,
                                           std::string* error) {
  if (header.using_qmatrix) {
    set_error(error, "using_qmatrix = 1 is not supported by this module");
    return 0;
  }
  std::string local;
  const std::size_t start = writer.bit_size();
  write_segmentation_params(writer, seg, header, &local);
  if (!local.empty()) {
    set_error(error, local);
    return 0;
  }
  writer.put_bit(false);  // using_qmatrix (setup_qm_params)
  write_delta_q_params(writer, delta_q, base_q_idx, &local);
  if (!local.empty()) {
    set_error(error, local);
    return 0;
  }
  return writer.bit_size() - start;
}

bool read_segmentation_params(BitReader& reader, const FrameQpHeader& header,
                              rc::SegmentationParams& params,
                              std::string* error) {
  params = rc::SegmentationParams{};
  bool enabled = false;
  if (!reader.read_bit(enabled)) {
    set_error(error, "segmentation: truncated segmentation_enabled");
    return false;
  }
  params.enabled = enabled;
  if (!enabled) return true;

  const int num_segments = max_segments(header.enable_ext_seg);
  if (!read_seg_info(reader, num_segments, params, error)) return false;

  if (header.primary_ref_is_none) {
    params.update_map = true;
    params.temporal_update = false;
  } else {
    bool update_map = false;
    if (!reader.read_bit(update_map)) {
      set_error(error, "segmentation: truncated segmentation_update_map");
      return false;
    }
    params.update_map = update_map;
    params.temporal_update = false;
    if (update_map && !header.frame_type_switch) {
      bool temporal = false;
      if (!reader.read_bit(temporal)) {
        set_error(error, "segmentation: truncated segmentation_temporal_update");
        return false;
      }
      params.temporal_update = temporal;
    }
  }
  if (!rc::segmentation_params_valid(params, error)) return false;
  return validate_seg_info(params, num_segments, error);
}

bool read_delta_q_params(BitReader& reader, int base_q_idx,
                         rc::DeltaQParams& params, std::string* error) {
  params = rc::DeltaQParams{};
  if (base_q_idx < 0 || base_q_idx > 255) {
    set_error(error, "delta_q: base_q_idx must be in [0, 255]");
    return false;
  }
  if (base_q_idx == 0) return true;
  bool present = false;
  if (!reader.read_bit(present)) {
    set_error(error, "delta_q: truncated delta_q_present");
    return false;
  }
  params.present = present;
  if (!present) return true;
  std::uint32_t res = 0;
  if (!reader.read_bits(2U, res)) {
    set_error(error, "delta_q: truncated delta_q_res");
    return false;
  }
  params.res = static_cast<int>(res);
  return true;
}

bool read_segmentation_and_delta_q(BitReader& reader, int base_q_idx,
                                   const FrameQpHeader& header,
                                   rc::SegmentationParams& seg,
                                   rc::DeltaQParams& delta_q,
                                   bool& using_qmatrix,
                                   std::string* error) {
  if (!read_segmentation_params(reader, header, seg, error)) return false;
  bool qmatrix = false;
  if (!reader.read_bit(qmatrix)) {
    set_error(error, "setup_qm_params: truncated using_qmatrix");
    return false;
  }
  using_qmatrix = qmatrix;
  if (qmatrix) {
    set_error(error, "using_qmatrix = 1 is not supported by this module");
    return false;
  }
  return read_delta_q_params(reader, base_q_idx, delta_q, error);
}

// ---------------------------------------------------------------------------
// Derived frame state
// ---------------------------------------------------------------------------

bool derive_frame_segmentation(const rc::SegmentationParams& params,
                               const rc::DeltaQParams& delta_q,
                               int base_q_idx,
                               const FrameQpHeader& header,
                               FrameSegmentation& out, std::string* error) {
  out = FrameSegmentation{};
  out.enable_ext_seg = header.enable_ext_seg;
  out.num_segments = max_segments(header.enable_ext_seg);
  if (!rc::segmentation_params_valid(params, error)) return false;
  if (!rc::delta_q_params_valid(delta_q, error)) return false;
  if (base_q_idx < 0 || base_q_idx > 255) {
    set_error(error, "segmentation: base_q_idx must be in [0, 255]");
    return false;
  }
  if (!params.enabled) return true;
  if (header.primary_ref_is_none && params.temporal_update) {
    set_error(error, "segmentation: key frame infers temporal_update = 0");
    return false;
  }
  if (header.frame_type_switch && params.temporal_update) {
    set_error(error, "segmentation: SWITCH_FRAME infers temporal_update = 0");
    return false;
  }
  if (!validate_seg_info(params, out.num_segments, error)) return false;

  out.enabled = true;
  out.features = params.segments;
  out.update_map =
      header.primary_ref_is_none ? true : params.update_map;
  out.temporal_update =
      header.primary_ref_is_none
          ? false
          : (params.update_map ? params.temporal_update : false);

  for (int i = 0; i < out.num_segments; ++i) {
    const rc::SegmentFeatures& features =
        params.segments[static_cast<std::size_t>(i)];
    for (int j = 0; j < kSegLvlMax; ++j) {
      if (!features.enabled[static_cast<std::size_t>(j)]) continue;
      out.last_active_seg_id = i;
      if (j >= kSegLvlSkip) out.seg_id_pre_skip = true;
    }

    // §5.18.2: LosslessArray[i] uses get_qindex(1, i).  This module has no
    // UV/DC quantizer deltas, so those terms are <= 0 exactly when the qindex
    // itself is 0 and delta_q_present is 0.
    int qindex = base_q_idx;
    if (features.enabled[static_cast<std::size_t>(kSegLvlAltQ)]) {
      qindex += static_cast<int>(
          features.data[static_cast<std::size_t>(kSegLvlAltQ)]);
    }
    qindex = clip3(0, 255, qindex);
    if (qindex == 0 && !delta_q.present) out.has_lossless_segment = true;
  }
  return true;
}

bool effective_qindex(const FrameSegmentation& seg, int base_q_idx,
                      int current_q_index, bool delta_q_present,
                      int segment_id, bool ignore_delta_q, int& qindex,
                      std::string* error) {
  if (base_q_idx < 0 || base_q_idx > 255 || current_q_index < 0 ||
      current_q_index > 255) {
    set_error(error, "get_qindex: qindex out of range");
    return false;
  }
  if (segment_id < 0 || segment_id >= seg.num_segments) {
    set_error(error, "get_qindex: segment id out of range");
    return false;
  }
  if (seg.enabled &&
      seg.features[static_cast<std::size_t>(segment_id)]
          .enabled[static_cast<std::size_t>(kSegLvlAltQ)]) {
    const int data = static_cast<int>(
        seg.features[static_cast<std::size_t>(segment_id)]
            .data[static_cast<std::size_t>(kSegLvlAltQ)]);
    int value = base_q_idx + data;
    if (!ignore_delta_q && delta_q_present) value = current_q_index + data;
    qindex = clip3(0, 255, value);
    return true;
  }
  if (!ignore_delta_q && delta_q_present) {
    qindex = current_q_index;
  } else {
    qindex = base_q_idx;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Tile QP state
// ---------------------------------------------------------------------------

bool begin_tile(TileQp& state, int base_q_idx, const rc::DeltaQParams& delta_q,
                std::string* error) {
  if (base_q_idx < 0 || base_q_idx > 255) {
    set_error(error, "delta_q: base_q_idx must be in [0, 255]");
    return false;
  }
  if (!rc::delta_q_params_valid(delta_q, error)) return false;
  if (delta_q.present && base_q_idx == 0) {
    set_error(error, "delta_q: delta_q_present requires base_q_idx >= 1");
    return false;
  }
  state.base_q_idx = base_q_idx;
  state.delta_q = delta_q;
  state.current_q_index = base_q_idx;  // §5.20.1, per tile
  state.read_deltas = false;
  state.initialized = true;
  return true;
}

void begin_superblock(TileQp& state) noexcept {
  state.read_deltas = state.delta_q.present;  // §5.20.2.1
}

bool codes_delta_q(const TileQp& state, bool mi_size_is_sb,
                   bool skip_flag) noexcept {
  return state.initialized && !(mi_size_is_sb && skip_flag) &&
         state.read_deltas;
}

int reduced_delta_for_target(int current_q_index, int target_q_index,
                             int res) noexcept {
  if (res < 0) return 0;
  const int step = 1 << res;
  const int diff = target_q_index - current_q_index;
  if (diff >= 0) return (diff + step / 2) / step;
  return -((-diff + step / 2) / step);
}

bool apply_reduced_delta(TileQp& state, int reduced_delta,
                         std::string* error) {
  if (!state.initialized) {
    set_error(error, "delta_q: tile state is not initialized");
    return false;
  }
  if (!state.delta_q.present) {
    set_error(error, "delta_q: no deltas are present in this frame");
    return false;
  }
  const std::int64_t offset =
      static_cast<std::int64_t>(reduced_delta) << state.delta_q.res;
  const std::int64_t next =
      static_cast<std::int64_t>(state.current_q_index) + offset;
  state.current_q_index =
      static_cast<int>(std::clamp<std::int64_t>(next, 1, 255));
  return true;
}

bool write_delta_q_abs(av2::Encoder& encoder, std::uint16_t cdf[9], int value,
                       std::string* error) {
  if (value < -kDeltaQMaxAbs || value > kDeltaQMaxAbs) {
    set_error(error, "delta_q: reduced delta magnitude is not codeable");
    return false;
  }
  const int magnitude = value < 0 ? -value : value;
  const int head = magnitude < kDeltaQSmall ? magnitude : kDeltaQSmall;
  if (!symbol(encoder, cdf, kDeltaQAlphabet, head, error)) return false;

  if (head == kDeltaQSmall) {
    // Escape: delta_q_abs = bits + (1 << rem_bits) + DELTA_Q_SMALL - 2.
    int rem_bits = 0;
    for (int r = 1; r <= 8; ++r) {
      const int base = (1 << r) + kDeltaQSmall - 2;
      const int top = base + (1 << r) - 1;
      if (magnitude >= base && magnitude <= top) {
        rem_bits = r;
        break;
      }
    }
    if (rem_bits == 0) {
      set_error(error, "delta_q: could not encode the escape value");
      return false;
    }
    const int base = (1 << rem_bits) + kDeltaQSmall - 2;
    const int bits = magnitude - base;
    if (!write_literal(encoder, 3U,
                       static_cast<std::uint32_t>(rem_bits - 1), error)) {
      return false;
    }
    if (!write_literal(encoder, static_cast<unsigned>(rem_bits),
                       static_cast<std::uint32_t>(bits), error)) {
      return false;
    }
  }

  if (magnitude != 0) {
    if (!entropy_ok(error, encoder.encode_bool(value < 0), "encode_bool")) {
      return false;
    }
  }
  return true;
}

bool read_delta_q_abs(av2::Decoder& decoder, std::uint16_t cdf[9], int& value,
                      std::string* error) {
  int head = 0;
  if (!symbol(decoder, cdf, kDeltaQAlphabet, head, error)) return false;
  int magnitude = head;
  if (head == kDeltaQSmall) {
    std::uint32_t rem_code = 0;
    if (!read_literal(decoder, 3U, rem_code, error)) return false;
    const unsigned rem_bits = rem_code + 1U;
    std::uint32_t bits = 0;
    if (!read_literal(decoder, rem_bits, bits, error)) return false;
    magnitude = static_cast<int>(bits) + (1 << rem_bits) + kDeltaQSmall - 2;
  }
  if (magnitude != 0) {
    bool negative = false;
    if (!entropy_ok(error, decoder.read_bool(&negative), "read_bool")) {
      return false;
    }
    value = negative ? -magnitude : magnitude;
  } else {
    value = 0;
  }
  return true;
}

bool write_delta_qindex(av2::Encoder& encoder, std::uint16_t cdf[9],
                        TileQp& state, bool mi_size_is_sb, bool skip_flag,
                        int reduced_delta, std::string* error) {
  if (!state.initialized) {
    set_error(error, "delta_q: tile state is not initialized");
    return false;
  }
  if (codes_delta_q(state, mi_size_is_sb, skip_flag)) {
    if (!write_delta_q_abs(encoder, cdf, reduced_delta, error)) return false;
    if (reduced_delta != 0) {
      if (!apply_reduced_delta(state, reduced_delta, error)) return false;
    }
  }
  if (state.delta_q.present) {
    state.current_q_index = clip3(1, 255, state.current_q_index);
  }
  state.read_deltas = false;  // ReadDeltas = 0 after read_delta_qindex()
  return true;
}

bool read_delta_qindex(av2::Decoder& decoder, std::uint16_t cdf[9],
                       TileQp& state, bool mi_size_is_sb, bool skip_flag,
                       int& reduced_delta, std::string* error) {
  reduced_delta = 0;
  if (!state.initialized) {
    set_error(error, "delta_q: tile state is not initialized");
    return false;
  }
  if (codes_delta_q(state, mi_size_is_sb, skip_flag)) {
    if (!read_delta_q_abs(decoder, cdf, reduced_delta, error)) return false;
    if (reduced_delta != 0) {
      if (!apply_reduced_delta(state, reduced_delta, error)) return false;
    }
  }
  if (state.delta_q.present) {
    state.current_q_index = clip3(1, 255, state.current_q_index);
  }
  state.read_deltas = false;
  return true;
}

// ---------------------------------------------------------------------------
// Segmentation map state
// ---------------------------------------------------------------------------

bool init_segment_grid(SegmentGrid& grid, int mi_cols, int mi_rows,
                       int tile_mi_col_start, int tile_mi_col_end,
                       int tile_mi_row_start, int tile_mi_row_end,
                       bool enable_ext_seg, int num_segments,
                       std::span<const std::uint8_t> previous_ids,
                       std::string* error) {
  if (mi_cols <= 0 || mi_rows <= 0) {
    set_error(error, "segmentation grid: dimensions must be positive");
    return false;
  }
  if (tile_mi_col_start < 0 || tile_mi_col_end > mi_cols ||
      tile_mi_row_start < 0 || tile_mi_row_end > mi_rows ||
      tile_mi_col_start >= tile_mi_col_end ||
      tile_mi_row_start >= tile_mi_row_end) {
    set_error(error, "segmentation grid: invalid tile bounds");
    return false;
  }
  if (num_segments != max_segments(enable_ext_seg)) {
    set_error(error, "segmentation grid: num_segments does not match ext flag");
    return false;
  }
  grid = SegmentGrid{};
  grid.mi_cols = mi_cols;
  grid.mi_rows = mi_rows;
  grid.tile_mi_col_start = tile_mi_col_start;
  grid.tile_mi_col_end = tile_mi_col_end;
  grid.tile_mi_row_start = tile_mi_row_start;
  grid.tile_mi_row_end = tile_mi_row_end;
  grid.enable_ext_seg = enable_ext_seg;
  grid.num_segments = num_segments;
  const std::size_t cells =
      static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows);
  grid.current.assign(cells, 0);
  grid.previous.assign(cells, 0);
  grid.above_pred.assign(static_cast<std::size_t>(mi_cols), 0);
  grid.left_pred.assign(static_cast<std::size_t>(mi_rows), 0);
  return set_previous_ids(grid, previous_ids, error);
}

void clear_above_context(SegmentGrid& grid) noexcept {
  std::fill(grid.above_pred.begin(), grid.above_pred.end(), 0);
}

void clear_left_context(SegmentGrid& grid) noexcept {
  std::fill(grid.left_pred.begin(), grid.left_pred.end(), 0);
}

bool set_previous_ids(SegmentGrid& grid, std::span<const std::uint8_t> ids,
                      std::string* error) {
  const std::size_t cells =
      static_cast<std::size_t>(grid.mi_cols) * static_cast<std::size_t>(grid.mi_rows);
  if (ids.empty()) {
    std::fill(grid.previous.begin(), grid.previous.end(), 0);
    return true;
  }
  if (ids.size() != cells) {
    set_error(error, "segmentation grid: previous map size mismatch");
    return false;
  }
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (static_cast<int>(ids[i]) >= grid.num_segments) {
      set_error(error, "segmentation grid: previous segment id out of range");
      return false;
    }
  }
  std::copy(ids.begin(), ids.end(), grid.previous.begin());
  return true;
}

int segment_prediction(const SegmentGrid& grid, int mi_row,
                       int mi_col) noexcept {
  int prev_ul = -1;
  int prev_u = -1;
  int prev_l = -1;
  segment_neighbours(grid, mi_row, mi_col, prev_ul, prev_u, prev_l);
  return segment_prediction_from(prev_ul, prev_u, prev_l);
}

int segment_id_context(int prev_ul, int prev_u, int prev_l) noexcept {
  if (prev_ul < 0) return 0;
  if (prev_ul == prev_u && prev_ul == prev_l) return 2;
  if (prev_ul == prev_u || prev_ul == prev_l || prev_u == prev_l) return 1;
  return 0;
}

int previous_segment_prediction(const SegmentGrid& grid, int mi_row, int mi_col,
                                int bw4, int bh4) noexcept {
  const int x_mis = std::min(grid.mi_cols - mi_col, bw4);
  const int y_mis = std::min(grid.mi_rows - mi_row, bh4);
  int seg = kMaxSegmentsSpec - 1;
  for (int y = 0; y < y_mis; ++y) {
    for (int x = 0; x < x_mis; ++x) {
      const std::size_t index =
          static_cast<std::size_t>(mi_row + y) *
              static_cast<std::size_t>(grid.mi_cols) +
          static_cast<std::size_t>(mi_col + x);
      seg = std::min(seg, static_cast<int>(grid.previous[index]));
    }
  }
  return seg;
}

int neg_deinterleave(int diff, int ref, int max) noexcept {
  if (ref == 0) return diff;
  if (ref >= max - 1) return max - diff - 1;
  if (2 * ref < max) {
    if (diff <= 2 * ref) {
      if (diff & 1) return ref + ((diff + 1) >> 1);
      return ref - (diff >> 1);
    }
    return diff;
  }
  if (diff <= 2 * (max - ref - 1)) {
    if (diff & 1) return ref + ((diff + 1) >> 1);
    return ref - (diff >> 1);
  }
  return max - (diff + 1);
}

bool neg_interleave(int target, int ref, int max, int& coded) noexcept {
  if (max <= 0 || target < 0 || target >= max) return false;
  for (int diff = 0; diff < max; ++diff) {
    if (neg_deinterleave(diff, ref, max) == target) {
      coded = diff;
      return true;
    }
  }
  return false;
}

bool commit_segment_block(SegmentGrid& grid, const SegmentBlock& block,
                          int segment_id, std::string* error) {
  if (segment_id < 0 || segment_id >= grid.num_segments) {
    set_error(error, "segmentation: segment id out of range");
    return false;
  }
  const int row_end = std::min(block.mi_row + block.bh4, grid.mi_rows);
  const int col_end = std::min(block.mi_col + block.bw4, grid.mi_cols);
  for (int r = std::max(block.mi_row, 0); r < row_end; ++r) {
    for (int c = std::max(block.mi_col, 0); c < col_end; ++c) {
      grid.current[static_cast<std::size_t>(r) *
                       static_cast<std::size_t>(grid.mi_cols) +
                   static_cast<std::size_t>(c)] =
          static_cast<std::uint8_t>(segment_id);
    }
  }
  return true;
}

bool write_intra_segment_id(av2::Encoder& encoder, BlockCdfs& cdfs,
                            const FrameSegmentation& seg, SegmentGrid& grid,
                            const SegmentBlock& block, int segment_id,
                            bool skip_flag, std::string* error) {
  if (!seg.enabled) {
    if (segment_id != 0) {
      set_error(error, "segmentation: disabled frame forces segment id 0");
      return false;
    }
    return true;
  }
  return write_segment_id_core(encoder, cdfs, seg, grid, block, segment_id,
                               skip_flag, error);
}

bool read_intra_segment_id(av2::Decoder& decoder, BlockCdfs& cdfs,
                           const FrameSegmentation& seg, SegmentGrid& grid,
                           const SegmentBlock& block, bool skip_flag,
                           int& segment_id, std::string* error) {
  if (!seg.enabled) {
    segment_id = 0;
    return true;
  }
  return read_segment_id_core(decoder, cdfs, seg, grid, block, skip_flag,
                              segment_id, error);
}

bool write_inter_segment_id(av2::Encoder& encoder, BlockCdfs& cdfs,
                            const FrameSegmentation& seg, SegmentGrid& grid,
                            const SegmentBlock& block, bool pre_skip,
                            int segment_id, bool skip_flag,
                            bool& coded_symbols, std::string* error) {
  coded_symbols = false;
  if (!seg.enabled) {
    if (segment_id != 0) {
      set_error(error, "segmentation: disabled frame forces segment id 0");
      return false;
    }
    return true;
  }
  if (!pre_skip && seg.seg_id_pre_skip) {
    set_error(error,
              "segmentation: post-skip id is absent when SegIdPreSkip = 1");
    return false;
  }

  const int predicted = previous_segment_prediction(
      grid, block.mi_row, block.mi_col, block.bw4, block.bh4);

  if (seg.update_map) {
    if (pre_skip && !seg.seg_id_pre_skip) {
      // segment_id = 0; the real call is the post-skip one.
      return true;
    }
    if (!pre_skip && skip_flag) {
      // seg_id_predicted = 0; contexts cleared; read_segment_id() follows.
      set_pred_contexts(grid, block, 0);
      if (!write_segment_id_core(encoder, cdfs, seg, grid, block, segment_id,
                                 /*skip_flag=*/true, error)) {
        return false;
      }
      coded_symbols = seg.has_lossless_segment;
      return true;
    }
    if (seg.temporal_update) {
      const int ctx = static_cast<int>(
          grid.left_pred[static_cast<std::size_t>(block.mi_row)]) +
                      static_cast<int>(
                          grid.above_pred[static_cast<std::size_t>(block.mi_col)]);
      const int predicted_flag = segment_id == predicted ? 1 : 0;
      if (!symbol(encoder, &cdfs.segment_id_predicted[ctx][0], 2,
                  predicted_flag, error)) {
        return false;
      }
      if (!predicted_flag) {
        if (!write_segment_id_core(encoder, cdfs, seg, grid, block, segment_id,
                                   skip_flag, error)) {
          return false;
        }
      }
      set_pred_contexts(grid, block, predicted_flag);
      coded_symbols = true;
      return true;
    }
    if (!write_segment_id_core(encoder, cdfs, seg, grid, block, segment_id,
                               skip_flag, error)) {
      return false;
    }
    coded_symbols = !(skip_flag && !seg.has_lossless_segment);
    return true;
  }

  // segmentation_update_map == 0: the map is inherited; no bits are coded.
  if (segment_id != predicted) {
    set_error(error,
              "segmentation: update_map = 0 forces the previous-map id");
    return false;
  }
  return true;
}

bool read_inter_segment_id(av2::Decoder& decoder, BlockCdfs& cdfs,
                           const FrameSegmentation& seg, SegmentGrid& grid,
                           const SegmentBlock& block, bool pre_skip,
                           bool skip_flag, int& segment_id,
                           bool& coded_symbols, std::string* error) {
  coded_symbols = false;
  segment_id = 0;
  if (!seg.enabled) return true;
  if (!pre_skip && seg.seg_id_pre_skip) {
    set_error(error,
              "segmentation: post-skip id is absent when SegIdPreSkip = 1");
    return false;
  }

  const int predicted = previous_segment_prediction(
      grid, block.mi_row, block.mi_col, block.bw4, block.bh4);

  if (seg.update_map) {
    if (pre_skip && !seg.seg_id_pre_skip) {
      return true;  // segment_id = 0; post-skip call does the real work
    }
    if (!pre_skip && skip_flag) {
      set_pred_contexts(grid, block, 0);
      if (!read_segment_id_core(decoder, cdfs, seg, grid, block,
                                /*skip_flag=*/true, segment_id, error)) {
        return false;
      }
      coded_symbols = seg.has_lossless_segment;
      return true;
    }
    if (seg.temporal_update) {
      const int ctx = static_cast<int>(
          grid.left_pred[static_cast<std::size_t>(block.mi_row)]) +
                      static_cast<int>(
                          grid.above_pred[static_cast<std::size_t>(block.mi_col)]);
      int predicted_flag = 0;
      if (!symbol(decoder, &cdfs.segment_id_predicted[ctx][0], 2,
                  predicted_flag, error)) {
        return false;
      }
      if (predicted_flag) {
        segment_id = predicted;
      } else if (!read_segment_id_core(decoder, cdfs, seg, grid, block,
                                       skip_flag, segment_id, error)) {
        return false;
      }
      set_pred_contexts(grid, block, predicted_flag);
      coded_symbols = true;
      return true;
    }
    if (!read_segment_id_core(decoder, cdfs, seg, grid, block, skip_flag,
                              segment_id, error)) {
      return false;
    }
    coded_symbols = !(skip_flag && !seg.has_lossless_segment);
    return true;
  }

  segment_id = predicted;
  return true;
}

// ---------------------------------------------------------------------------
// Rate-control planner integration
// ---------------------------------------------------------------------------

bool make_delta_q_plan_view(const rc::DeltaQPlan& plan, int sb_cols,
                            int sb_rows, DeltaQPlanView& view,
                            std::string* error) {
  view = DeltaQPlanView{};
  if (sb_cols <= 0 || sb_rows <= 0) {
    set_error(error, "delta_q plan: tile dimensions must be positive");
    return false;
  }
  const std::size_t expected =
      static_cast<std::size_t>(sb_cols) * static_cast<std::size_t>(sb_rows);
  if (plan.steps.size() != expected) {
    set_error(error, "delta_q plan: step count does not match the tile");
    return false;
  }
  for (int y = 0; y < sb_rows; ++y) {
    for (int x = 0; x < sb_cols; ++x) {
      const rc::DeltaQPlanStep& step =
          plan.steps[static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(sb_cols) +
                     static_cast<std::size_t>(x)];
      if (step.sb_x != x || step.sb_y != y) {
        set_error(error, "delta_q plan: steps are not in raster order");
        return false;
      }
      if (y == 0 && x == 0) continue;
      const rc::DeltaQPlanStep& prev =
          plan.steps[static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(sb_cols) +
                     static_cast<std::size_t>(x) - 1];
      if (step.qindex_before != prev.qindex_after) {
        set_error(error, "delta_q plan: qindex chain is broken");
        return false;
      }
    }
  }
  view.sb_cols = sb_cols;
  view.sb_rows = sb_rows;
  view.plan = &plan;
  return true;
}

const rc::DeltaQPlanStep* delta_q_plan_step(const DeltaQPlanView& view, int sb_x,
                                            int sb_y) noexcept {
  if (view.plan == nullptr || sb_x < 0 || sb_y < 0 || sb_x >= view.sb_cols ||
      sb_y >= view.sb_rows) {
    return nullptr;
  }
  return &view.plan->steps[static_cast<std::size_t>(sb_y) *
                               static_cast<std::size_t>(view.sb_cols) +
                           static_cast<std::size_t>(sb_x)];
}

bool check_plan_step(const TileQp& state, const rc::DeltaQPlanStep& step,
                     std::string* error) {
  if (state.current_q_index != step.qindex_before) {
    set_error(error, "delta_q plan: qindex_before does not match tile state");
    return false;
  }
  TileQp probe = state;
  if (!apply_reduced_delta(probe, step.reduced_delta, error)) return false;
  if (probe.current_q_index != step.qindex_after) {
    set_error(error, "delta_q plan: qindex_after does not match the apply rule");
    return false;
  }
  return true;
}

bool load_segmentation_map(const rc::SegmentationMap& map, SegmentGrid& grid,
                           std::string* error) {
  if (!rc::segmentation_map_valid(map, error)) return false;
  if (map.mi_cols != grid.mi_cols || map.mi_rows != grid.mi_rows) {
    set_error(error, "segmentation grid: rc map dimensions do not match");
    return false;
  }
  return set_previous_ids(
      grid, std::span<const std::uint8_t>(map.segment_id), error);
}

int block_qindex(const FrameSegmentation& seg, const TileQp& qp,
                 int segment_id) noexcept {
  int qindex = qp.base_q_idx;
  if (!effective_qindex(seg, qp.base_q_idx, qp.current_q_index,
                        qp.delta_q.present, segment_id,
                        /*ignore_delta_q=*/false, qindex, nullptr)) {
    return qp.base_q_idx;
  }
  return qindex;
}

}  // namespace av2enc::deltaq
