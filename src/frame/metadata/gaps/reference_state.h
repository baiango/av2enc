#pragma once

// AV2 v1.0.0 §7.23 reference-frame-update save/restore state that the encoder
// currently does not model at all:
//
//   SavedMvs[i][y8][x8][list][comp]        (§7.23 p577)
//   SavedRefFrames[i][y8][x8][list]        (§7.23 p577)
//   SavedGmParams[i][ref][j]               (§7.23 p577)
//   SavedSegmentIds[i][row][col]           (§7.23 p577)
//   save_grain_params(i) / load_grain_params(idx)     (§7.23 p577-578,
//                                                     §7.21.2 p562)
//   save_ccso_params(i, plane) / load_ccso_params(i, plane) (§7.23 p577-578)
//
// Normative reference: spec/AV2_Spec_v1.0.0.pdf.
//
// Scope
// -----
// This is a pure state container. It does not add syntax writers and does not
// change any encoder decision. It exists so the reference-update path can call
// the §7.23 saves at the exact point the spec requires and so later temporal
// prediction / film-grain / CCSO-reuse work has a tested store to read from.
// `README.md` in this directory documents the call sites in the reference
// update path.
//
// Shape rules (all enforced by the savers)
// ----------------------------------------
// * The motion-field grid is (MiRows >> 1) x (MiCols >> 1) 8x8 units;
//   SavedMvs and SavedRefFrames share it.
// * SavedSegmentIds uses the luma MI grid, MiRows x MiCols.
// * SavedGmParams always has REFS_PER_FRAME rows of six parameters.
// * Grain/CCSO stores are indexed independently of the four Saved* arrays,
//   because §7.23 invokes those functions separately and with their own index
//   domains (grain also uses NUM_REF_FRAMES as the mixed-output slot).
// * A slot that has never been saved cannot be loaded; load returns false.
// * Saving is a deep copy; later mutation of the caller's snapshot cannot
//   change the stored state.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace av2enc::metadata::gaps {

inline constexpr std::size_t kNumRefFrames = 16;  // NUM_REF_FRAMES (§3)
inline constexpr std::size_t kRefsPerFrame = 7;  // REFS_PER_FRAME
inline constexpr std::size_t kMaxFilmGrainSlots = 8;  // MAX_FILM_GRAIN
inline constexpr std::size_t kCcsoPlanes = 3;         // NumPlanes
// §5.18.7.12 / §7.23: CcsoFilterOffset[plane][band][cls0][cls1],
// 64 bands x 3 x 3 classes.
inline constexpr std::size_t kCcsoBandNum = 64;
inline constexpr std::size_t kCcsoMaxClasses = 3;
inline constexpr std::size_t kCcsoOffsetEntries =
    kCcsoBandNum * kCcsoMaxClasses * kCcsoMaxClasses;

// ---------------------------------------------------------------------------
// §7.23 Saved* arrays
// ---------------------------------------------------------------------------

// Motion-vector component pair. `comp` 0/1 follows the spec order; the encoder
// convention is row then column.
struct MotionVector {
  std::int16_t row = 0;
  std::int16_t col = 0;

  friend bool operator==(const MotionVector&, const MotionVector&) = default;
};

// One 8x8 unit: [list][comp]. List 0 is past, list 1 is future.
struct MvPair {
  std::array<MotionVector, 2> list{};

  friend bool operator==(const MvPair&, const MvPair&) = default;
};

// SavedMvs[ i ][ y8 ][ x8 ][ list ][ comp ].
struct SavedMvs {
  std::size_t rows = 0;  // MiRows >> 1
  std::size_t cols = 0;  // MiCols >> 1
  std::vector<MvPair> units;  // rows * cols, index y8 * cols + x8

  friend bool operator==(const SavedMvs&, const SavedMvs&) = default;
};

// One 8x8 unit: MfRefFrames[list] (a reference index or -1).
struct RefFramePair {
  std::array<std::int16_t, 2> list{};

  friend bool operator==(const RefFramePair&, const RefFramePair&) = default;
};

// SavedRefFrames[ i ][ y8 ][ x8 ][ list ].
struct SavedRefFrames {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<RefFramePair> units;  // rows * cols

  friend bool operator==(const SavedRefFrames&, const SavedRefFrames&) =
      default;
};

// SavedGmParams[ i ][ ref ][ j ], j = 0..5.
struct GlobalMotionParams {
  std::array<std::int32_t, 6> params{};

  friend bool operator==(const GlobalMotionParams&,
                         const GlobalMotionParams&) = default;
};

struct SavedGmParams {
  std::array<GlobalMotionParams, kRefsPerFrame> refs{};

  friend bool operator==(const SavedGmParams&, const SavedGmParams&) = default;
};

// SavedSegmentIds[ i ][ row ][ col ], luma MI grid.
struct SavedSegmentIds {
  std::size_t rows = 0;  // MiRows
  std::size_t cols = 0;  // MiCols
  std::vector<std::uint8_t> ids;  // rows * cols, index row * cols + col

  friend bool operator==(const SavedSegmentIds&,
                         const SavedSegmentIds&) = default;
};

// The four arrays saved together for one refreshed reference slot.
struct ReferenceSlot {
  SavedMvs mvs;
  SavedRefFrames ref_frames;
  SavedGmParams gm_params;
  SavedSegmentIds segment_ids;

  friend bool operator==(const ReferenceSlot&, const ReferenceSlot&) = default;
};

// ---------------------------------------------------------------------------
// §7.23 save_grain_params / load_grain_params
// ---------------------------------------------------------------------------

// A point of a film-grain scaling curve (§5.18.10.2). `value` is the decoded
// (delta-accumulated) value, `scaling` the decoded scaling.
struct GrainPoint {
  std::uint16_t value = 0;
  std::uint16_t scaling = 0;

  friend bool operator==(const GrainPoint&, const GrainPoint&) = default;
};

// All syntax elements of §5.18.10.2 film_grain_model().
struct FilmGrainModel {
  bool chroma_scaling_from_luma = false;
  std::uint8_t num_y_points = 0;                    // f(4)
  std::uint8_t point_value_increment_bits_minus_1 = 0;  // f(3)
  std::uint8_t point_scaling_bits_minus_5 = 0;          // f(2)
  std::vector<GrainPoint> y_points;

  std::uint8_t num_cb_points = 0;  // f(4)
  std::uint8_t cb_point_value_increment_bits_minus_1 = 0;
  std::uint8_t cb_point_scaling_bits_minus_5 = 0;
  std::vector<GrainPoint> cb_points;

  std::uint8_t num_cr_points = 0;  // f(4)
  std::uint8_t cr_point_value_increment_bits_minus_1 = 0;
  std::uint8_t cr_point_scaling_bits_minus_5 = 0;
  std::vector<GrainPoint> cr_points;

  std::uint8_t grain_scaling_minus_8 = 0;  // f(2)
  std::uint8_t ar_coeff_lag = 0;           // f(2)
  std::uint8_t bits_per_ar_coeff_y_minus_5 = 0;  // f(2)
  std::vector<std::int16_t> ar_coeffs_y;
  std::uint8_t bits_per_ar_coeff_cb_minus_5 = 0;
  std::vector<std::int16_t> ar_coeffs_cb;
  std::uint8_t bits_per_ar_coeff_cr_minus_5 = 0;
  std::vector<std::int16_t> ar_coeffs_cr;
  std::uint8_t ar_coeff_shift_minus_6 = 0;  // f(2)
  std::uint8_t grain_scale_shift = 0;       // f(2)

  std::uint8_t cb_mult = 0;      // f(8)
  std::uint8_t cb_luma_mult = 0;  // f(8)
  std::uint16_t cb_offset = 0;   // f(9)
  std::uint8_t cr_mult = 0;      // f(8)
  std::uint8_t cr_luma_mult = 0;  // f(8)
  std::uint16_t cr_offset = 0;   // f(9)

  bool overlap_flag = false;              // f(1)
  bool clip_to_restricted_range = false;   // f(1)
  bool fg_mc_identity = false;             // f(1) when clip is set
  bool film_grain_block_size = false;      // f(1)

  friend bool operator==(const FilmGrainModel&, const FilmGrainModel&) =
      default;
};

// All syntax elements of §5.18.10.1 film_grain_config().
struct FilmGrainConfig {
  bool apply_grain = false;
  std::uint8_t fgm_id = 0;       // f(3)
  std::uint16_t grain_seed = 0;  // f(16)

  friend bool operator==(const FilmGrainConfig&,
                         const FilmGrainConfig&) = default;
};

// A complete saved film-grain unit. `fgm_chroma_idc` is the film grain OBU
// field that selected subX/subY/monochrome while parsing the model; it is not
// itself part of film_grain_model()/film_grain_config(), but a lossless store
// keeps it so the model can be re-serialized with the same interpretation.
struct GrainParams {
  std::uint8_t fgm_chroma_idc = 0;  // CHROMA_FORMAT_* (§5.14)
  FilmGrainConfig config;
  FilmGrainModel model;

  friend bool operator==(const GrainParams&, const GrainParams&) = default;
};

// §5.18.7.12 one-plane CCSO state as listed by save_ccso_params. The
// CcsoLumaSizeLog2 scalar is saved per (i, plane) by the spec, so each plane
// carries its own copy.
struct CcsoPlaneParams {
  std::uint8_t luma_size_log2 = 6;  // CcsoLumaSizeLog2, 6..8
  bool enabled = false;       // ccso_planes[plane]
  std::uint8_t scale_idx = 0;  // f(2)
  bool bo_only = false;        // f(1)
  std::uint8_t quant_idx = 0;  // f(2)
  std::uint8_t ext_filter = 0;  // f(3)
  std::uint8_t max_band_log2 = 0;  // f(2 + bo_only)
  std::uint8_t edge_clf = 0;       // f(1)
  std::array<std::int16_t, kCcsoOffsetEntries> filter_offset{};
  std::size_t blk_rows = 0;
  std::size_t blk_cols = 0;
  std::vector<std::uint8_t> blks;  // CcsoBlks[plane]

  friend bool operator==(const CcsoPlaneParams&,
                         const CcsoPlaneParams&) = default;
};

// CcsoLumaSizeLog2 plus the three planes saved by save_ccso_params.
struct CcsoParams {
  std::uint8_t luma_size_log2 = 6;  // CcsoLumaSizeLog2, 6..8
  std::array<CcsoPlaneParams, kCcsoPlanes> planes{};

  friend bool operator==(const CcsoParams&, const CcsoParams&) = default;
};

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

// Return false and set `error` when the snapshot cannot be a §7.23 save of a
// real frame (shape mismatch, out-of-range coded field, or a field that is
// inferred when a flag is clear but left non-zero).
bool validate_saved_mvs(const SavedMvs& value, std::string& error);
bool validate_saved_ref_frames(const SavedRefFrames& value,
                               std::string& error);
bool validate_saved_gm_params(const SavedGmParams& value, std::string& error);
bool validate_saved_segment_ids(const SavedSegmentIds& value,
                                std::string& error);
bool validate_reference_slot(const ReferenceSlot& value, std::string& error);

bool validate_grain_params(const GrainParams& value, std::string& error);
bool validate_ccso_plane(const CcsoPlaneParams& value, std::string& error);
bool validate_ccso_params(const CcsoParams& value, std::string& error);

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

// §7.23 state for one CVS. All save/load functions are bounds-checked and
// return false (leaving the store unchanged) on invalid input.
class ReferenceStateStore {
 public:
  // §7.23 per refreshed slot. `save_reference()` validates and deep-copies the
  // four Saved* arrays and marks the slot complete. The individual savers may
  // be used instead, but then `load_reference()` returns false: a partial slot
  // must not masquerade as a full snapshot.
  bool save_reference(std::size_t index, const ReferenceSlot& slot,
                      std::string& error);
  bool load_reference(std::size_t index, ReferenceSlot& slot) const;

  bool save_mvs(std::size_t index, const SavedMvs& value, std::string& error);
  bool load_mvs(std::size_t index, SavedMvs& value) const;
  bool save_ref_frames(std::size_t index, const SavedRefFrames& value,
                       std::string& error);
  bool load_ref_frames(std::size_t index, SavedRefFrames& value) const;
  bool save_gm_params(std::size_t index, const SavedGmParams& value,
                      std::string& error);
  bool load_gm_params(std::size_t index, SavedGmParams& value) const;
  bool save_segment_ids(std::size_t index, const SavedSegmentIds& value,
                        std::string& error);
  bool load_segment_ids(std::size_t index, SavedSegmentIds& value) const;

  // §7.23 save_grain_params(i) / §7.21.2 load_grain_params(idx).
  // `index` is 0..NUM_REF_FRAMES inclusive: the spec also loads
  // load_grain_params(NUM_REF_FRAMES) for mixed output, so the store exposes
  // that slot. Both ends are bounds-checked.
  bool save_grain_params(std::size_t index, const GrainParams& value,
                         std::string& error);
  bool load_grain_params(std::size_t index, GrainParams& value) const;

  // §7.23 save_ccso_params(i, plane) / load_ccso_params(i, plane).
  bool save_ccso_params(std::size_t index, std::size_t plane,
                        const CcsoPlaneParams& value, std::string& error);
  bool load_ccso_params(std::size_t index, std::size_t plane,
                        CcsoPlaneParams& value) const;

  // Convenience wrappers over the per-plane calls above, in plane order 0..2.
  bool save_ccso_frame(std::size_t index, const CcsoParams& value,
                       std::string& error);
  bool load_ccso_frame(std::size_t index, CcsoParams& value) const;

  // Observability for tests / diagnostics.
  bool has_reference(std::size_t index) const;
  bool has_grain_params(std::size_t index) const;
  bool has_ccso_params(std::size_t index, std::size_t plane) const;
  void clear();

 private:
  std::array<std::optional<ReferenceSlot>, kNumRefFrames> slots_{};
  // True only after save_reference() stored a complete four-array snapshot.
  std::array<bool, kNumRefFrames> reference_complete_{};
  // NUM_REF_FRAMES + 1: index kNumRefFrames is the §7.21.2 mixed-output slot.
  std::array<std::optional<GrainParams>, kNumRefFrames + 1> grain_{};
  std::array<std::array<std::optional<CcsoPlaneParams>, kCcsoPlanes>,
             kNumRefFrames>
      ccso_{};
};

}  // namespace av2enc::metadata::gaps
