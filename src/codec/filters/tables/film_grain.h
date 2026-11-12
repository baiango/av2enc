// §7.21.7 Film grain synthesis process, using the recovered normative
// Gaussian_Sequence table (film_grain_tables.h).
//
// The table comes from the official gaussian_sequence.h attachment; see
// raw/SHA256SUMS and generate_spec_tables.py.  Scalar CPU reference only; film
// grain is an output-only process and is not part of reconstruction.
//
// Scope: 4:0:0, 4:2:0, 4:2:2 and 4:4:4 with BitDepth 8, 10 or 12.

#ifndef AV2ENC_ENCODER_FILTERS_SPEC_TABLES_FILM_GRAIN_H_
#define AV2ENC_ENCODER_FILTERS_SPEC_TABLES_FILM_GRAIN_H_

#include <cstdint>
#include <string>

namespace av2enc::filters::spec {

constexpr int kMaxFilmGrainPoints = 14;
constexpr int kMaxFilmGrainArCoeffs = 25;  // numPosLuma(3) + 1

// §5.18.10.2 film_grain_model syntax elements (after the sign recentring of
// the ar_coeffs_* arrays performed by the syntax parser).
struct FilmGrainModel {
  bool chroma_scaling_from_luma = false;

  int num_y_points = 0;
  int point_y_value[kMaxFilmGrainPoints] = {};
  int point_y_scaling[kMaxFilmGrainPoints] = {};

  int num_cb_points = 0;
  int point_cb_value[kMaxFilmGrainPoints] = {};
  int point_cb_scaling[kMaxFilmGrainPoints] = {};

  int num_cr_points = 0;
  int point_cr_value[kMaxFilmGrainPoints] = {};
  int point_cr_scaling[kMaxFilmGrainPoints] = {};

  int grain_scaling_minus_8 = 0;   // 0..3
  int ar_coeff_lag = 0;            // 0..3
  int ar_coeffs_y[kMaxFilmGrainArCoeffs] = {};
  int ar_coeffs_cb[kMaxFilmGrainArCoeffs] = {};
  int ar_coeffs_cr[kMaxFilmGrainArCoeffs] = {};
  int ar_coeff_shift_minus_6 = 0;  // 0..3
  int grain_scale_shift = 0;       // 0..3

  int cb_mult = 0;
  int cb_luma_mult = 0;
  int cb_offset = 0;
  int cr_mult = 0;
  int cr_luma_mult = 0;
  int cr_offset = 0;

  bool overlap_flag = false;
  bool clip_to_restricted_range = false;
  bool fg_mc_identity = false;
  bool film_grain_block_size = false;  // false = 16x16, true = 32x32
};

// Input/output arrays.  `w` and `h` are CropWidth/CropHeight in luma samples.
struct FilmGrainPlane {
  std::uint16_t* data = nullptr;
  int stride = 0;
};

struct FilmGrainFrame {
  FilmGrainPlane plane[3];
  int w = 0;
  int h = 0;
  int sub_x = 0;
  int sub_y = 0;
  int bit_depth = 8;
  int num_planes = 3;
};

// Checks the §6.17.10.2 bitstream conformance requirements that matter to the
// synthesis process.  Returns false and fills `why` on failure.
bool FilmGrainModelValid(const FilmGrainModel& model, int sub_x, int sub_y,
                         int bit_depth, int num_planes, std::string* why);

// §7.21.7 film grain synthesis.  Modifies frame.plane[0..num_planes-1].
void ApplyFilmGrain(const FilmGrainModel& model, FilmGrainFrame& frame,
                    std::uint16_t grain_seed);

}  // namespace av2enc::filters::spec

#endif  // AV2ENC_ENCODER_FILTERS_SPEC_TABLES_FILM_GRAIN_H_
