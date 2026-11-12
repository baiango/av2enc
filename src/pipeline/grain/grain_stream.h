// Encoder-side film grain model estimation, slot (fgm_id) selection, and
// stream assembly helpers.
//
// The estimator is deliberately simple and deterministic: a high-pass
// (4-neighbour Laplacian) activity estimate drives a piecewise-linear scaling
// curve and an intensity gate, and the AR taps come from a regularised
// least-squares fit of the residual against the §7.21.7.3 raster template.
// The goal is a *valid, stable* model, not a perceptual grain matcher.
//
// fgm_id selection is an 8-slot LRU cache over exactly-equal models: a frame
// whose estimated model is already resident reuses the slot and emits no
// film grain OBU; otherwise the least-recently-used slot is overwritten and a
// single-slot film grain OBU is returned with the plan.
//
// Stream helpers implement the §7.3.8.8 availability flow: the model OBU is
// inserted before the coded frame OBU in the same temporal unit (§7.3.3 /
// §7.3.4), and film_grain_params_present is flipped on in the sequence header
// only for streams that actually carry grain.
#ifndef AV2ENC_ENCODER_FILTERS_FILM_GRAIN_STREAM_GRAIN_STREAM_H_
#define AV2ENC_ENCODER_FILTERS_FILM_GRAIN_STREAM_GRAIN_STREAM_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "pipeline/grain/stream/film_grain_syntax.h"

namespace av2enc::filters::grain {

// Planar 8-bit input.  `u`/`v` may be null when only luma is analysed (the
// estimator then still produces a luma-only model).
struct GrainFrameView {
  const std::uint8_t* y = nullptr;
  const std::uint8_t* u = nullptr;
  const std::uint8_t* v = nullptr;
  int width = 0;
  int height = 0;
  int y_stride = 0;
  int c_stride = 0;
};

struct GrainEstimateOptions {
  bool enable = true;
  int ar_coeff_lag = 2;              // 0..3
  double scaling_gain = 2.0;         // scaling = mean|laplacian| * gain
  double min_activity = 1.0;         // below this, apply_grain = 0
  int max_y_points = 9;              // <= 14; 0 disables the curve
  bool chroma_scaling_from_luma = true;
  bool overlap_flag = true;
  bool use_32x32_blocks = false;
  int ar_coeff_shift_minus_6 = 2;    // 8 fractional bits
  int bits_per_ar_coeff_minus_5 = 3; // 8-bit signed taps
};

struct GrainEstimate {
  bool apply = false;
  double activity = 0.0;  // mean |4-neighbour Laplacian| on interior luma
  GrainModel model;
};

GrainEstimate EstimateGrainModel(const GrainFrameView& frame,
                                 const GrainEstimateOptions& options = {});

// Deterministic per-frame seed; every value is legal (§6.17.10.1).
std::uint16_t GrainSeedForFrame(std::uint32_t frame_index);

struct GrainPlanOptions {
  GrainEstimateOptions estimate;
  GrainChromaIdc chroma_idc = GrainChromaIdc::k420;
  // Cancellation: plan apply_grain = 0 without touching slot state.
  bool disable = false;
};

struct GrainFramePlan {
  bool apply_grain = false;
  int fgm_id = 0;
  std::uint16_t grain_seed = 0;
  // True when the model is new/changed and the caller must emit `obu` before
  // the coded frame OBU (otherwise the slot is already present).
  bool emit_model = false;
  FilmGrainObu obu;  // valid when emit_model
  GrainModel model;  // model the decoder will load from fgm_id
  GrainChromaIdc chroma_idc = GrainChromaIdc::k420;
  double activity = 0.0;
};

class GrainStreamPlanner {
 public:
  explicit GrainStreamPlanner(const GrainPlanOptions& options = {});

  // Resets all FilmGrainPresent slots and the LRU stamps.
  void Reset();

  const GrainSlotState& slots() const { return slots_; }
  const GrainPlanOptions& options() const { return options_; }

  // Estimates the model, selects a slot, updates §7.23 save_grain_params
  // state, and returns the per-frame decision.  `frame_index` is the coding
  // order index (it only seeds `grain_seed`).
  bool PlanFrame(const GrainFrameView& frame, std::uint32_t frame_index,
                 int tlayer_id, int mlayer_id, GrainFramePlan* plan,
                 std::string* error);

  // Explicit "no grain on this frame" plan (§5.18.10.1 apply_grain = 0).
  GrainFramePlan CancelFrame(std::uint32_t frame_index) const;

 private:
  GrainPlanOptions options_;
  GrainSlotState slots_;
  std::uint32_t stamps_[kNumGrainSlots] = {};
  std::uint32_t tick_ = 0;
};

// --- Stream assembly --------------------------------------------------------

struct GrainObuView {
  std::uint8_t type = 0;
  std::size_t offset = 0;  // byte offset of the leb128 prefix
  std::size_t size = 0;    // total bytes including prefix + header
  std::span<const std::uint8_t> payload;
};

// True for OBU types that carry a coded frame (§7.3.3/§7.3.4).
bool IsFrameObuType(std::uint8_t type);

// Splits an Annex B temporal unit into OBUs.
bool ParseTemporalUnit(std::span<const std::uint8_t> unit,
                       std::vector<GrainObuView>* obus, std::string* error);

// Inserts a film grain OBU immediately before the coded-frame OBU.  Fails if
// the unit already contains a film grain OBU (at most one model OBU per
// frame unit in this encoder) or has no frame OBU.
bool InsertFilmGrainObu(std::span<const std::uint8_t> unit,
                        std::span<const std::uint8_t> obu_bytes,
                        std::vector<std::uint8_t>* out, std::string* error);

// Reads/writes the sequence-level film_grain_params_present bit in place.
//
// Layout assumption (documented, and checked): the repo's sequence-header
// writers emit sequence syntax ending with
//   seq_tile_info_present_flag, film_grain_params_present, obu_extension_flag
// followed by trailing_bits().  The last set bit in the payload is therefore
// trailing_one_bit, the bit before it is obu_extension_flag, and the bit
// before that is film_grain_params_present.
bool ReadSequenceHeaderFilmGrainPresent(std::span<const std::uint8_t> unit,
                                        bool* present, std::string* error);
bool PatchSequenceHeaderFilmGrainPresent(std::vector<std::uint8_t>* unit,
                                         bool present, std::string* error);

}  // namespace av2enc::filters::grain

#endif  // AV2ENC_ENCODER_FILTERS_FILM_GRAIN_STREAM_GRAIN_STREAM_H_
