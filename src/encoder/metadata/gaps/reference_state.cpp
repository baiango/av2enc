#include "encoder/metadata/gaps/reference_state.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace av2enc::metadata::gaps {
namespace {

std::string size_text(std::size_t value) { return std::to_string(value); }

bool check_grid(const char* name, std::size_t rows, std::size_t cols,
                std::size_t entries, std::string& error) {
  if (cols == 0U) {
    if (rows != 0U || entries != 0U) {
      error = std::string(name) + " has " + size_text(rows) + " rows and " +
              size_text(entries) +
              " entries; a zero-column grid must be empty";
      return false;
    }
    return true;
  }
  if (rows > std::numeric_limits<std::size_t>::max() / cols) {
    error = std::string(name) + " dimensions overflow";
    return false;
  }
  if (rows * cols != entries) {
    error = std::string(name) + " is " + size_text(rows) + "x" +
            size_text(cols) + " but holds " + size_text(entries) +
            " entries";
    return false;
  }
  return true;
}

bool fits_unsigned(std::uint32_t value, unsigned bits) {
  return bits >= 32U || value < (1U << bits);
}

}  // namespace

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

bool validate_saved_mvs(const SavedMvs& value, std::string& error) {
  error.clear();
  return check_grid("SavedMvs", value.rows, value.cols, value.units.size(),
                    error);
}

bool validate_saved_ref_frames(const SavedRefFrames& value,
                               std::string& error) {
  error.clear();
  return check_grid("SavedRefFrames", value.rows, value.cols,
                    value.units.size(), error);
}

bool validate_saved_gm_params(const SavedGmParams&, std::string& error) {
  // REFS_PER_FRAME x 6 is a fixed-shape array; every int32 is representable.
  error.clear();
  return true;
}

bool validate_saved_segment_ids(const SavedSegmentIds& value,
                                std::string& error) {
  error.clear();
  if (!check_grid("SavedSegmentIds", value.rows, value.cols, value.ids.size(),
                  error)) {
    return false;
  }
  // §5.18.6.2: at most 16 segments; a stored id therefore fits in 4 bits.
  for (const std::uint8_t id : value.ids) {
    if (id > 15U) {
      error = "SavedSegmentIds entry " + size_text(id) +
              " exceeds the 16-segment maximum";
      return false;
    }
  }
  return true;
}

bool validate_reference_slot(const ReferenceSlot& value, std::string& error) {
  error.clear();
  if (!validate_saved_mvs(value.mvs, error) ||
      !validate_saved_ref_frames(value.ref_frames, error) ||
      !validate_saved_gm_params(value.gm_params, error) ||
      !validate_saved_segment_ids(value.segment_ids, error)) {
    return false;
  }
  if (value.mvs.rows != value.ref_frames.rows ||
      value.mvs.cols != value.ref_frames.cols) {
    error = "SavedMvs and SavedRefFrames must share the 8x8 grid";
    return false;
  }
  // §7.23: SavedMvs/SavedRefFrames use (MiRows>>1, MiCols>>1) while
  // SavedSegmentIds uses (MiRows, MiCols).
  if (value.mvs.rows != value.segment_ids.rows / 2U ||
      value.mvs.cols != value.segment_ids.cols / 2U) {
    error = "SavedMvs grid must be MiRows>>1 x MiCols>>1 of "
            "SavedSegmentIds";
    return false;
  }
  return true;
}

bool validate_grain_params(const GrainParams& value, std::string& error) {
  error.clear();
  const FilmGrainModel& model = value.model;
  const auto fail = [&error](std::string message) {
    error = std::move(message);
    return false;
  };

  if (value.fgm_chroma_idc > 3U) {
    return fail("fgm_chroma_idc must be one of the four CHROMA_FORMAT values");
  }
  if (value.config.fgm_id > 7U) {
    return fail("fgm_id must fit in 3 bits (§5.18.10.1)");
  }
  if (model.num_y_points > 15U) {
    return fail("num_y_points must fit in 4 bits (§5.18.10.2)");
  }
  if (model.num_cb_points > 15U || model.num_cr_points > 15U) {
    return fail("num_cb_points/num_cr_points must fit in 4 bits (§5.18.10.2)");
  }
  if (model.y_points.size() != model.num_y_points ||
      model.cb_points.size() != model.num_cb_points ||
      model.cr_points.size() != model.num_cr_points) {
    return fail("grain point arrays do not match their num_*_points counts");
  }

  if (!fits_unsigned(model.point_value_increment_bits_minus_1, 3U) ||
      !fits_unsigned(model.point_scaling_bits_minus_5, 2U)) {
    return fail("y point-curve bit widths do not fit their f(3)/f(2) fields");
  }
  if (!fits_unsigned(model.cb_point_value_increment_bits_minus_1, 3U) ||
      !fits_unsigned(model.cb_point_scaling_bits_minus_5, 2U) ||
      !fits_unsigned(model.cr_point_value_increment_bits_minus_1, 3U) ||
      !fits_unsigned(model.cr_point_scaling_bits_minus_5, 2U)) {
    return fail("chroma point-curve bit widths do not fit their f(3)/f(2) "
                "fields");
  }
  if (!fits_unsigned(model.grain_scaling_minus_8, 2U) ||
      !fits_unsigned(model.ar_coeff_lag, 2U) ||
      !fits_unsigned(model.bits_per_ar_coeff_y_minus_5, 2U) ||
      !fits_unsigned(model.bits_per_ar_coeff_cb_minus_5, 2U) ||
      !fits_unsigned(model.bits_per_ar_coeff_cr_minus_5, 2U) ||
      !fits_unsigned(model.ar_coeff_shift_minus_6, 2U) ||
      !fits_unsigned(model.grain_scale_shift, 2U)) {
    return fail("a film-grain f(2) field does not fit in 2 bits");
  }
  if (!fits_unsigned(model.cb_offset, 9U) ||
      !fits_unsigned(model.cr_offset, 9U)) {
    return fail("cb_offset/cr_offset must fit in 9 bits (§5.18.10.2)");
  }
  if (model.num_y_points == 0U &&
      (model.point_value_increment_bits_minus_1 != 0U ||
       model.point_scaling_bits_minus_5 != 0U)) {
    return fail("y point-curve widths are not coded when num_y_points == 0");
  }
  if (model.num_cb_points == 0U &&
      (model.cb_point_value_increment_bits_minus_1 != 0U ||
       model.cb_point_scaling_bits_minus_5 != 0U)) {
    return fail("Cb point-curve widths are not coded when num_cb_points == 0");
  }
  if (model.num_cr_points == 0U &&
      (model.cr_point_value_increment_bits_minus_1 != 0U ||
       model.cr_point_scaling_bits_minus_5 != 0U)) {
    return fail("Cr point-curve widths are not coded when num_cr_points == 0");
  }
  if (model.chroma_scaling_from_luma &&
      (model.num_cb_points != 0U || model.num_cr_points != 0U)) {
    return fail("num_cb_points/num_cr_points are not coded when "
                "chroma_scaling_from_luma == 1");
  }
  if (model.num_cb_points == 0U &&
      (model.cb_mult != 0U || model.cb_luma_mult != 0U ||
       model.cb_offset != 0U)) {
    return fail("cb_mult/cb_luma_mult/cb_offset are not coded when "
                "num_cb_points == 0");
  }
  if (model.num_cr_points == 0U &&
      (model.cr_mult != 0U || model.cr_luma_mult != 0U ||
       model.cr_offset != 0U)) {
    return fail("cr_mult/cr_luma_mult/cr_offset are not coded when "
                "num_cr_points == 0");
  }
  if (!model.clip_to_restricted_range && model.fg_mc_identity) {
    return fail("fg_mc_identity is not coded when "
                "clip_to_restricted_range == 0");
  }

  const std::size_t num_pos_luma =
      2U * model.ar_coeff_lag * (model.ar_coeff_lag + 1U);
  const std::size_t num_pos_chroma =
      num_pos_luma + (model.num_y_points != 0U ? 1U : 0U);
  if (model.num_y_points == 0U) {
    if (!model.ar_coeffs_y.empty() || model.bits_per_ar_coeff_y_minus_5 != 0U) {
      return fail("luma AR coefficients are not coded when num_y_points == 0");
    }
  } else if (model.ar_coeffs_y.size() != num_pos_luma) {
    return fail("ar_coeffs_y has " + size_text(model.ar_coeffs_y.size()) +
                " entries; ar_coeff_lag requires " + size_text(num_pos_luma));
  }
  const bool cb_coded =
      model.chroma_scaling_from_luma || model.num_cb_points != 0U;
  if (!cb_coded) {
    if (!model.ar_coeffs_cb.empty() ||
        model.bits_per_ar_coeff_cb_minus_5 != 0U) {
      return fail("Cb AR coefficients are not coded without Cb points or "
                  "chroma_scaling_from_luma");
    }
  } else if (model.ar_coeffs_cb.size() != num_pos_chroma) {
    return fail("ar_coeffs_cb has " + size_text(model.ar_coeffs_cb.size()) +
                " entries; expected " + size_text(num_pos_chroma));
  }
  const bool cr_coded =
      model.chroma_scaling_from_luma || model.num_cr_points != 0U;
  if (!cr_coded) {
    if (!model.ar_coeffs_cr.empty() ||
        model.bits_per_ar_coeff_cr_minus_5 != 0U) {
      return fail("Cr AR coefficients are not coded without Cr points or "
                  "chroma_scaling_from_luma");
    }
  } else if (model.ar_coeffs_cr.size() != num_pos_chroma) {
    return fail("ar_coeffs_cr has " + size_text(model.ar_coeffs_cr.size()) +
                " entries; expected " + size_text(num_pos_chroma));
  }

  const auto check_coeffs = [&fail](const std::vector<std::int16_t>& coeffs,
                                    unsigned bits_minus_5,
                                    const char* name) -> bool {
    const unsigned bits = bits_minus_5 + 5U;
    const std::int32_t minimum = -(1 << (bits - 1U));
    const std::int32_t maximum = (1 << (bits - 1U)) - 1;
    for (const std::int16_t coeff : coeffs) {
      if (coeff < minimum || coeff > maximum) {
        return fail(std::string(name) + " coefficient " +
                    std::to_string(coeff) + " does not fit in " +
                    std::to_string(bits) + " signed bits");
      }
    }
    return true;
  };
  if (model.num_y_points != 0U &&
      !check_coeffs(model.ar_coeffs_y, model.bits_per_ar_coeff_y_minus_5,
                    "ar_coeffs_y")) {
    return false;
  }
  if (cb_coded &&
      !check_coeffs(model.ar_coeffs_cb, model.bits_per_ar_coeff_cb_minus_5,
                    "ar_coeffs_cb")) {
    return false;
  }
  if (cr_coded &&
      !check_coeffs(model.ar_coeffs_cr, model.bits_per_ar_coeff_cr_minus_5,
                    "ar_coeffs_cr")) {
    return false;
  }
  return true;
}

bool validate_ccso_plane(const CcsoPlaneParams& value, std::string& error) {
  error.clear();
  const auto fail = [&error](std::string message) {
    error = std::move(message);
    return false;
  };
  if (value.luma_size_log2 < 6U || value.luma_size_log2 > 8U) {
    return fail("CcsoLumaSizeLog2 must be 6, 7 or 8 (§5.18.7.12)");
  }
  if (!value.enabled) {
    if (value.scale_idx != 0U || value.bo_only || value.quant_idx != 0U ||
        value.ext_filter != 0U || value.max_band_log2 != 0U ||
        value.edge_clf != 0U || value.blk_rows != 0U ||
        value.blk_cols != 0U || !value.blks.empty()) {
      return fail("a disabled CCSO plane must keep every coded field at its "
                  "default (§5.18.7.12)");
    }
    return true;
  }
  if (!fits_unsigned(value.scale_idx, 2U)) {
    return fail("ccso_scale_idx must fit in 2 bits");
  }
  if (!fits_unsigned(value.quant_idx, 2U)) {
    return fail("ccso_quant_idx must fit in 2 bits");
  }
  if (!fits_unsigned(value.ext_filter, 3U)) {
    return fail("ccso_ext_filter must fit in 3 bits");
  }
  if (!fits_unsigned(value.edge_clf, 1U)) {
    return fail("ccso_edge_clf must fit in 1 bit");
  }
  const unsigned band_bits = 2U + (value.bo_only ? 1U : 0U);
  if (!fits_unsigned(value.max_band_log2, band_bits)) {
    return fail("ccso_max_band_log2 must fit in f(2 + ccso_bo_only) bits");
  }
  if (value.bo_only &&
      (value.quant_idx != 0U || value.ext_filter != 0U ||
       value.edge_clf != 0U)) {
    return fail("ccso_quant_idx/ccso_ext_filter/ccso_edge_clf are inferred 0 "
                "when ccso_bo_only == 1");
  }
  if (!check_grid("CcsoBlks", value.blk_rows, value.blk_cols,
                  value.blks.size(), error)) {
    return false;
  }
  for (const std::uint8_t blk : value.blks) {
    if (blk > 1U) {
      return fail("CcsoBlks entries must be 0 or 1");
    }
  }
  return true;
}

bool validate_ccso_params(const CcsoParams& value, std::string& error) {
  error.clear();
  if (value.luma_size_log2 < 6U || value.luma_size_log2 > 8U) {
    error = "CcsoLumaSizeLog2 must be 6, 7 or 8 (§5.18.7.12)";
    return false;
  }
  for (std::size_t plane = 0; plane < kCcsoPlanes; ++plane) {
    if (value.planes[plane].luma_size_log2 != value.luma_size_log2) {
      error = "plane " + size_text(plane) +
              " CcsoLumaSizeLog2 does not match the frame value";
      return false;
    }
    if (!validate_ccso_plane(value.planes[plane], error)) {
      error = "plane " + size_text(plane) + ": " + error;
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Store: §7.23 Saved* arrays
// ---------------------------------------------------------------------------

bool ReferenceStateStore::save_reference(std::size_t index,
                                         const ReferenceSlot& slot,
                                         std::string& error) {
  error.clear();
  if (index >= kNumRefFrames) {
    error = "reference index " + size_text(index) + " is outside [0, " +
            size_text(kNumRefFrames) + ")";
    return false;
  }
  if (!validate_reference_slot(slot, error)) {
    return false;
  }
  slots_[index] = slot;
  reference_complete_[index] = true;
  return true;
}

bool ReferenceStateStore::load_reference(std::size_t index,
                                         ReferenceSlot& slot) const {
  if (index >= kNumRefFrames || !reference_complete_[index]) {
    return false;
  }
  slot = *slots_[index];
  return true;
}

bool ReferenceStateStore::save_mvs(std::size_t index, const SavedMvs& value,
                                   std::string& error) {
  error.clear();
  if (index >= kNumRefFrames) {
    error = "reference index " + size_text(index) + " is outside [0, " +
            size_text(kNumRefFrames) + ")";
    return false;
  }
  if (!validate_saved_mvs(value, error)) {
    return false;
  }
  if (!slots_[index].has_value()) {
    slots_[index] = ReferenceSlot{};
  }
  slots_[index]->mvs = value;
  return true;
}

bool ReferenceStateStore::load_mvs(std::size_t index, SavedMvs& value) const {
  if (index >= kNumRefFrames || !slots_[index].has_value()) {
    return false;
  }
  value = slots_[index]->mvs;
  return true;
}

bool ReferenceStateStore::save_ref_frames(std::size_t index,
                                          const SavedRefFrames& value,
                                          std::string& error) {
  error.clear();
  if (index >= kNumRefFrames) {
    error = "reference index " + size_text(index) + " is outside [0, " +
            size_text(kNumRefFrames) + ")";
    return false;
  }
  if (!validate_saved_ref_frames(value, error)) {
    return false;
  }
  if (!slots_[index].has_value()) {
    slots_[index] = ReferenceSlot{};
  }
  slots_[index]->ref_frames = value;
  return true;
}

bool ReferenceStateStore::load_ref_frames(std::size_t index,
                                          SavedRefFrames& value) const {
  if (index >= kNumRefFrames || !slots_[index].has_value()) {
    return false;
  }
  value = slots_[index]->ref_frames;
  return true;
}

bool ReferenceStateStore::save_gm_params(std::size_t index,
                                         const SavedGmParams& value,
                                         std::string& error) {
  error.clear();
  if (index >= kNumRefFrames) {
    error = "reference index " + size_text(index) + " is outside [0, " +
            size_text(kNumRefFrames) + ")";
    return false;
  }
  if (!validate_saved_gm_params(value, error)) {
    return false;
  }
  if (!slots_[index].has_value()) {
    slots_[index] = ReferenceSlot{};
  }
  slots_[index]->gm_params = value;
  return true;
}

bool ReferenceStateStore::load_gm_params(std::size_t index,
                                         SavedGmParams& value) const {
  if (index >= kNumRefFrames || !slots_[index].has_value()) {
    return false;
  }
  value = slots_[index]->gm_params;
  return true;
}

bool ReferenceStateStore::save_segment_ids(std::size_t index,
                                           const SavedSegmentIds& value,
                                           std::string& error) {
  error.clear();
  if (index >= kNumRefFrames) {
    error = "reference index " + size_text(index) + " is outside [0, " +
            size_text(kNumRefFrames) + ")";
    return false;
  }
  if (!validate_saved_segment_ids(value, error)) {
    return false;
  }
  if (!slots_[index].has_value()) {
    slots_[index] = ReferenceSlot{};
  }
  slots_[index]->segment_ids = value;
  return true;
}

bool ReferenceStateStore::load_segment_ids(std::size_t index,
                                           SavedSegmentIds& value) const {
  if (index >= kNumRefFrames || !slots_[index].has_value()) {
    return false;
  }
  value = slots_[index]->segment_ids;
  return true;
}

// ---------------------------------------------------------------------------
// Store: save_grain_params / save_ccso_params
// ---------------------------------------------------------------------------

bool ReferenceStateStore::save_grain_params(std::size_t index,
                                            const GrainParams& value,
                                            std::string& error) {
  error.clear();
  if (index > kNumRefFrames) {
    error = "grain index " + size_text(index) + " is outside [0, " +
            size_text(kNumRefFrames) + "] (NUM_REF_FRAMES is the mixed-output "
            "slot, §7.21.2)";
    return false;
  }
  if (!validate_grain_params(value, error)) {
    return false;
  }
  grain_[index] = value;
  return true;
}

bool ReferenceStateStore::load_grain_params(std::size_t index,
                                            GrainParams& value) const {
  if (index > kNumRefFrames || !grain_[index].has_value()) {
    return false;
  }
  value = *grain_[index];
  return true;
}

bool ReferenceStateStore::save_ccso_params(std::size_t index,
                                           std::size_t plane,
                                           const CcsoPlaneParams& value,
                                           std::string& error) {
  error.clear();
  if (index >= kNumRefFrames) {
    error = "reference index " + size_text(index) + " is outside [0, " +
            size_text(kNumRefFrames) + ")";
    return false;
  }
  if (plane >= kCcsoPlanes) {
    error = "plane " + size_text(plane) + " is outside [0, " +
            size_text(kCcsoPlanes) + ")";
    return false;
  }
  if (!validate_ccso_plane(value, error)) {
    return false;
  }
  ccso_[index][plane] = value;
  return true;
}

bool ReferenceStateStore::load_ccso_params(std::size_t index,
                                           std::size_t plane,
                                           CcsoPlaneParams& value) const {
  if (index >= kNumRefFrames || plane >= kCcsoPlanes ||
      !ccso_[index][plane].has_value()) {
    return false;
  }
  value = *ccso_[index][plane];
  return true;
}

bool ReferenceStateStore::save_ccso_frame(std::size_t index,
                                          const CcsoParams& value,
                                          std::string& error) {
  error.clear();
  if (index >= kNumRefFrames) {
    error = "reference index " + size_text(index) + " is outside [0, " +
            size_text(kNumRefFrames) + ")";
    return false;
  }
  if (!validate_ccso_params(value, error)) {
    return false;
  }
  for (std::size_t plane = 0; plane < kCcsoPlanes; ++plane) {
    CcsoPlaneParams stored = value.planes[plane];
    stored.luma_size_log2 = value.luma_size_log2;
    ccso_[index][plane] = stored;
  }
  return true;
}

bool ReferenceStateStore::load_ccso_frame(std::size_t index,
                                          CcsoParams& value) const {
  if (index >= kNumRefFrames) {
    return false;
  }
  value = CcsoParams{};
  value.luma_size_log2 = ccso_[index][0].has_value()
                             ? ccso_[index][0]->luma_size_log2
                             : 6U;
  for (std::size_t plane = 0; plane < kCcsoPlanes; ++plane) {
    if (!ccso_[index][plane].has_value()) {
      return false;
    }
    value.planes[plane] = *ccso_[index][plane];
  }
  return true;
}

// ---------------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------------

bool ReferenceStateStore::has_reference(std::size_t index) const {
  return index < kNumRefFrames && slots_[index].has_value();
}

bool ReferenceStateStore::has_grain_params(std::size_t index) const {
  return index <= kNumRefFrames && grain_[index].has_value();
}

bool ReferenceStateStore::has_ccso_params(std::size_t index,
                                          std::size_t plane) const {
  return index < kNumRefFrames && plane < kCcsoPlanes &&
         ccso_[index][plane].has_value();
}

void ReferenceStateStore::clear() {
  slots_.fill(std::nullopt);
  reference_complete_.fill(false);
  grain_.fill(std::nullopt);
  for (auto& frame : ccso_) {
    frame.fill(std::nullopt);
  }
}

}  // namespace av2enc::metadata::gaps
