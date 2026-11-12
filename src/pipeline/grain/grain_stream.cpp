#include "pipeline/grain/stream/grain_stream.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace av2enc::filters::grain {
namespace {

bool Fail(std::string* error, const char* message) {
  if (error != nullptr) *error = message;
  return false;
}

int ClampInt(int value, int lo, int hi) {
  return std::max(lo, std::min(hi, value));
}

int AbsInt(int value) { return value < 0 ? -value : value; }

// 4-neighbour Laplacian on interior luma samples.
int Laplacian(const GrainFrameView& f, int x, int y) {
  const std::uint8_t* row = f.y + static_cast<std::ptrdiff_t>(y) * f.y_stride;
  const std::uint8_t* up =
      f.y + static_cast<std::ptrdiff_t>(y - 1) * f.y_stride;
  const std::uint8_t* down =
      f.y + static_cast<std::ptrdiff_t>(y + 1) * f.y_stride;
  return 4 * static_cast<int>(row[x]) - static_cast<int>(row[x - 1]) -
         static_cast<int>(row[x + 1]) - static_cast<int>(up[x]) -
         static_cast<int>(down[x]);
}

// Cholesky solve of A x = b for a small symmetric positive-definite system.
// A is row-major n x n.  Returns false when the matrix is not positive
// definite (caller falls back to zero taps).
bool SolveSpd(int n, const double* a_in, const double* b, double* x) {
  double l[25 * 25] = {};
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = a_in[i * n + j];
      for (int k = 0; k < j; ++k) sum -= l[i * n + k] * l[j * n + k];
      if (i == j) {
        if (sum <= 1e-12) return false;
        l[i * n + j] = std::sqrt(sum);
      } else {
        l[i * n + j] = sum / l[j * n + j];
      }
    }
  }
  double y[25] = {};
  for (int i = 0; i < n; ++i) {
    double sum = b[i];
    for (int k = 0; k < i; ++k) sum -= l[i * n + k] * y[k];
    y[i] = sum / l[i * n + i];
  }
  for (int i = n - 1; i >= 0; --i) {
    double sum = y[i];
    for (int k = i + 1; k < n; ++k) sum -= l[k * n + i] * x[k];
    x[i] = sum / l[i * n + i];
  }
  return true;
}

// Number of neighbours in the §7.21.7.3 raster template for `lag`.
int NumPositions(int lag) { return 2 * lag * (lag + 1); }

// Collects the template offsets in raster order: deltaRow -lag..0,
// deltaCol -lag..lag, skipping (0, 0).
void TemplateOffsets(int lag, int* dx, int* dy, int* count) {
  int n = 0;
  for (int dyv = -lag; dyv <= 0; ++dyv) {
    for (int dxv = -lag; dxv <= lag; ++dxv) {
      if (dyv == 0 && dxv == 0) continue;
      dx[n] = dxv;
      dy[n] = dyv;
      ++n;
    }
  }
  *count = n;
}

// Regularised least-squares AR fit on the high-pass residual.
void FitArCoefficients(const GrainFrameView& f, int lag, int ar_shift,
                       int coeff_bits, int* coeffs) {
  for (int i = 0; i < kMaxArPositions; ++i) coeffs[i] = 0;
  const int n = NumPositions(lag);
  if (n == 0) return;
  const int margin = lag + 1;
  if (f.width <= 2 * margin + 1 || f.height <= 2 * margin + 1) return;

  int dx[kMaxArPositions] = {};
  int dy[kMaxArPositions] = {};
  int count = 0;
  TemplateOffsets(lag, dx, dy, &count);

  double a[25 * 25] = {};
  double b[25] = {};
  double trace = 0.0;
  for (int y = margin; y < f.height - margin; ++y) {
    for (int x = margin; x < f.width - margin; ++x) {
      const double e = static_cast<double>(Laplacian(f, x, y)) * 0.25;
      double taps[25] = {};
      for (int p = 0; p < count; ++p) {
        taps[p] =
            static_cast<double>(Laplacian(f, x + dx[p], y + dy[p])) * 0.25;
      }
      for (int p = 0; p < count; ++p) {
        b[p] += taps[p] * e;
        for (int q = 0; q <= p; ++q) a[p * count + q] += taps[p] * taps[q];
      }
    }
  }
  for (int p = 0; p < count; ++p) trace += a[p * count + p];
  const double ridge =
      (trace > 0.0 ? trace / static_cast<double>(count) : 1.0) * 1e-4 + 1e-9;
  for (int p = 0; p < count; ++p) a[p * count + p] += ridge;
  // Mirror the lower triangle.
  for (int p = 1; p < count; ++p) {
    for (int q = 0; q < p; ++q) a[q * count + p] = a[p * count + q];
  }

  double solution[25] = {};
  if (!SolveSpd(count, a, b, solution)) return;

  const int bits = coeff_bits + 5;
  const int lo = -(1 << (bits - 1));
  const int hi = (1 << (bits - 1)) - 1;
  const double scale = std::ldexp(1.0, ar_shift);
  for (int p = 0; p < count; ++p) {
    coeffs[p] = ClampInt(static_cast<int>(std::lround(solution[p] * scale)), lo,
                         hi);
  }
}

// Builds the piecewise-linear scaling curve from per-bucket luma activity.
// Returns the number of points written.
int BuildScalingCurve(const GrainFrameView& f,
                      const GrainEstimateOptions& opt, int num_points,
                      int* values, int* scalings) {
  constexpr int kBuckets = 9;
  const int bucket_x[kBuckets] = {0, 32, 64, 96, 128, 160, 192, 224, 255};
  double sum[kBuckets] = {};
  long long count[kBuckets] = {};
  double total = 0.0;
  long long total_count = 0;

  for (int y = 1; y < f.height - 1; ++y) {
    for (int x = 1; x < f.width - 1; ++x) {
      const int sample = static_cast<int>(
          f.y[static_cast<std::ptrdiff_t>(y) * f.y_stride + x]);
      int bucket = 0;
      while (bucket + 1 < kBuckets && sample > bucket_x[bucket + 1]) ++bucket;
      const int activity = AbsInt(Laplacian(f, x, y));
      sum[bucket] += static_cast<double>(activity);
      ++count[bucket];
      total += static_cast<double>(activity);
      ++total_count;
    }
  }

  const double global_mean =
      total_count > 0 ? total / static_cast<double>(total_count) : 0.0;
  int point_count = 0;
  const int max_points = ClampInt(num_points, 1, kMaxGrainPoints);
  int last_scaling = -1;
  for (int i = 0; i < kBuckets; ++i) {
    if (point_count >= max_points) break;
    double mean = global_mean;
    if (count[i] >= 8) mean = sum[i] / static_cast<double>(count[i]);
    int scaling = ClampInt(
        static_cast<int>(std::lround(mean * opt.scaling_gain)), 0, 255);
    // Collapse a flat curve to one point; the last bucket always terminates.
    if (i > 0 && scaling == last_scaling && i + 1 < kBuckets &&
        point_count < max_points) {
      continue;
    }
    values[point_count] = bucket_x[i];
    scalings[point_count] = scaling;
    last_scaling = scaling;
    ++point_count;
  }
  if (point_count == 0) {
    values[0] = 0;
    scalings[0] = ClampInt(static_cast<int>(std::lround(global_mean *
                                                        opt.scaling_gain)),
                           0, 255);
    point_count = 1;
  }
  // Zero the unused tail.
  for (int i = point_count; i < kMaxGrainPoints; ++i) {
    values[i] = 0;
    scalings[i] = 0;
  }
  return point_count;
}

int ModelScalingMax(const GrainModel& m) {
  int best = 0;
  for (int i = 0; i < m.num_y_points; ++i) {
    best = std::max(best, m.point_y_scaling[i]);
  }
  return best;
}

bool ModelsEqual(const GrainModel& a, GrainChromaIdc a_idc,
                 const GrainModel& b, GrainChromaIdc b_idc) {
  return a_idc == b_idc && a == b;
}

bool ParseLeb128(std::span<const std::uint8_t> data, std::size_t* pos,
                 std::uint64_t* value, std::string* error) {
  std::uint64_t result = 0;
  int shift = 0;
  while (true) {
    if (*pos >= data.size()) return Fail(error, "truncated leb128");
    const std::uint8_t byte = data[(*pos)++];
    result |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
    if ((byte & 0x80U) == 0) break;
    shift += 7;
    if (shift > 28) return Fail(error, "leb128 value too large");
  }
  *value = result;
  return true;
}

}  // namespace

GrainEstimate EstimateGrainModel(const GrainFrameView& frame,
                                 const GrainEstimateOptions& options) {
  GrainEstimate estimate;
  if (!options.enable || frame.y == nullptr || frame.width < 8 ||
      frame.height < 8) {
    return estimate;
  }

  double activity_sum = 0.0;
  long long activity_count = 0;
  for (int y = 1; y < frame.height - 1; ++y) {
    for (int x = 1; x < frame.width - 1; ++x) {
      activity_sum += static_cast<double>(AbsInt(Laplacian(frame, x, y)));
      ++activity_count;
    }
  }
  estimate.activity =
      activity_count > 0 ? activity_sum / static_cast<double>(activity_count)
                         : 0.0;

  GrainModel model;
  model.chroma_scaling_from_luma = options.chroma_scaling_from_luma;
  model.ar_coeff_lag = ClampInt(options.ar_coeff_lag, 0, 3);
  model.grain_scaling_minus_8 = 0;
  model.grain_scale_shift = 0;
  model.ar_coeff_shift_minus_6 = ClampInt(options.ar_coeff_shift_minus_6, 0, 3);
  model.overlap_flag = options.overlap_flag;
  model.clip_to_restricted_range = false;
  model.fg_mc_identity = false;
  model.film_grain_block_size = options.use_32x32_blocks;
  model.point_y_increment_bits_minus_1 = 7;  // 8-bit increments
  model.point_y_scaling_bits_minus_5 = 3;    // 8-bit scaling values

  model.num_y_points =
      BuildScalingCurve(frame, options, options.max_y_points,
                        model.point_y_value, model.point_y_scaling);

  // Chroma point curves are not estimated.  With
  // chroma_scaling_from_luma == 0 and no chroma points the model carries no
  // chroma grain at all, which is legal for every layout.

  if (ModelScalingMax(model) == 0 ||
      estimate.activity < options.min_activity) {
    estimate.apply = false;
    return estimate;
  }

  const int ar_shift = model.ar_coeff_shift_minus_6 + 6;
  const int coeff_bits = ClampInt(options.bits_per_ar_coeff_minus_5, 0, 3);
  int luma[25] = {};
  FitArCoefficients(frame, model.ar_coeff_lag, ar_shift, coeff_bits, luma);
  model.bits_per_ar_coeff_y_minus_5 = coeff_bits;
  model.bits_per_ar_coeff_cb_minus_5 = coeff_bits;
  model.bits_per_ar_coeff_cr_minus_5 = coeff_bits;
  for (int i = 0; i < kMaxArPositions; ++i) {
    model.ar_coeffs_y[i] = luma[i];
    // Chroma reuses the luma spatial taps; the extra (numPosLuma) tap is the
    // averaged luma-grain term, left at zero.
    model.ar_coeffs_cb[i] = luma[i];
    model.ar_coeffs_cr[i] = luma[i];
  }
  model.ar_coeffs_cb[NumPositions(model.ar_coeff_lag)] = 0;
  model.ar_coeffs_cr[NumPositions(model.ar_coeff_lag)] = 0;

  model = NormalizeModel(model, GrainChromaIdc::k420);
  std::string why;
  if (!ValidateModel(model, GrainChromaIdc::k420, &why)) {
    return estimate;
  }
  estimate.apply = true;
  estimate.model = model;
  return estimate;
}

std::uint16_t GrainSeedForFrame(std::uint32_t frame_index) {
  // Odd multiplier keeps consecutive seeds far apart; any 16-bit value is
  // legal (§6.17.10.1).
  const std::uint32_t mixed = 0x9E37U * (frame_index + 1U) + 0x1D2FU;
  return static_cast<std::uint16_t>(mixed & 0xFFFFU);
}

GrainStreamPlanner::GrainStreamPlanner(const GrainPlanOptions& options)
    : options_(options) {
  Reset();
}

void GrainStreamPlanner::Reset() {
  slots_.Reset();
  for (int i = 0; i < kNumGrainSlots; ++i) stamps_[i] = 0;
  tick_ = 0;
}

GrainFramePlan GrainStreamPlanner::CancelFrame(
    std::uint32_t frame_index) const {
  GrainFramePlan plan;
  plan.apply_grain = false;
  plan.fgm_id = 0;
  plan.grain_seed = GrainSeedForFrame(frame_index);
  plan.chroma_idc = options_.chroma_idc;
  return plan;
}

bool GrainStreamPlanner::PlanFrame(const GrainFrameView& frame,
                                   std::uint32_t frame_index, int tlayer_id,
                                   int mlayer_id, GrainFramePlan* plan,
                                   std::string* error) {
  if (plan == nullptr) return Fail(error, "null grain plan");
  if (options_.disable) {
    *plan = CancelFrame(frame_index);
    return true;
  }
  GrainEstimate estimate = EstimateGrainModel(frame, options_.estimate);
  if (!estimate.apply) {
    *plan = CancelFrame(frame_index);
    plan->activity = estimate.activity;
    return true;
  }

  // The estimator always produces a 4:2:0 layout model; re-validate against
  // the frame's chroma format before using it.
  const GrainChromaIdc idc = options_.chroma_idc;
  std::string why;
  if (!ValidateModel(estimate.model, idc, &why)) {
    // For non-4:2:0 layouts the chroma point constraints change; fall back to
    // cancelling rather than emitting an invalid model.
    *plan = CancelFrame(frame_index);
    plan->activity = estimate.activity;
    return true;
  }

  int slot = -1;
  for (int i = 0; i < kNumGrainSlots; ++i) {
    if (!slots_.slot(i).present) continue;
    if (ModelsEqual(slots_.slot(i).model, slots_.slot(i).chroma_idc,
                    estimate.model, idc)) {
      slot = i;
      break;
    }
  }

  bool emit_model = false;
  if (slot < 0) {
    // Never-used slots first (ascending), then least-recently-used.
    slot = 0;
    for (int i = 0; i < kNumGrainSlots; ++i) {
      if (stamps_[i] == 0) {
        slot = i;
        break;
      }
      if (stamps_[i] < stamps_[slot]) slot = i;
    }
    if (!slots_.SaveModel(slot, estimate.model, idc, tlayer_id, mlayer_id,
                          error)) {
      return false;
    }
    ++tick_;
    stamps_[slot] = tick_;
    emit_model = true;
  } else {
    ++tick_;
    stamps_[slot] = tick_;
  }

  GrainFramePlan result;
  result.apply_grain = true;
  result.fgm_id = slot;
  result.grain_seed = GrainSeedForFrame(frame_index);
  result.emit_model = emit_model;
  result.model = slots_.slot(slot).model;
  result.chroma_idc = idc;
  result.activity = estimate.activity;
  if (emit_model) {
    result.obu = FilmGrainObu{};
    result.obu.update_flags = static_cast<std::uint8_t>(1U << slot);
    result.obu.chroma_idc = idc;
    result.obu.models[slot] = result.model;
  }
  // §7.3.8.8 / §6.17.10.1: the model must be present and layer-compatible.
  if (!ValidateConfigAvailability(slots_, FilmGrainConfig{true, slot,
                                                          result.grain_seed},
                                  idc, tlayer_id, mlayer_id, error)) {
    return false;
  }
  *plan = result;
  return true;
}

bool IsFrameObuType(std::uint8_t type) {
  switch (type) {
    case 4:   // OBU_CLOSED_LOOP_KEY
    case 5:   // OBU_OPEN_LOOP_KEY
    case 6:   // OBU_LEADING_TILE_GROUP
    case 7:   // OBU_REGULAR_TILE_GROUP
    case 10:  // OBU_SWITCH
    case 11:  // OBU_LEADING_SEF
    case 12:  // OBU_REGULAR_SEF
    case 13:  // OBU_LEADING_TIP
    case 14:  // OBU_REGULAR_TIP
    case 19:  // OBU_BRIDGE_FRAME
    case 21:  // OBU_RAS_FRAME
      return true;
    default:
      return false;
  }
}

bool ParseTemporalUnit(std::span<const std::uint8_t> unit,
                       std::vector<GrainObuView>* obus, std::string* error) {
  if (obus == nullptr) return Fail(error, "null OBU list");
  obus->clear();
  std::size_t pos = 0;
  while (pos < unit.size()) {
    const std::size_t start = pos;
    std::uint64_t size = 0;
    if (!ParseLeb128(unit, &pos, &size, error)) return false;
    if (size == 0 || pos + size > unit.size()) {
      return Fail(error, "truncated OBU in temporal unit");
    }
    const std::uint8_t header = unit[pos];
    const bool extension = (header & 0x80U) != 0;
    GrainObuView view;
    view.type = static_cast<std::uint8_t>((header >> 2U) & 0x1FU);
    view.offset = start;
    view.size = pos + static_cast<std::size_t>(size) - start;
    const std::size_t header_bytes = extension ? 2U : 1U;
    if (size < header_bytes) return Fail(error, "OBU smaller than header");
    view.payload = unit.subspan(
        pos + header_bytes,
        static_cast<std::size_t>(size) - header_bytes);
    obus->push_back(view);
    pos += static_cast<std::size_t>(size);
  }
  return true;
}

bool InsertFilmGrainObu(std::span<const std::uint8_t> unit,
                        std::span<const std::uint8_t> obu_bytes,
                        std::vector<std::uint8_t>* out, std::string* error) {
  if (out == nullptr) return Fail(error, "null stream output");
  std::vector<GrainObuView> obus;
  if (!ParseTemporalUnit(unit, &obus, error)) return false;
  std::size_t frame_index = obus.size();
  for (std::size_t i = 0; i < obus.size(); ++i) {
    if (obus[i].type == kFilmGrainObuType) {
      return Fail(error, "unit already has a film grain OBU");
    }
    if (IsFrameObuType(obus[i].type) && frame_index == obus.size()) {
      frame_index = i;
    }
  }
  if (frame_index == obus.size()) {
    return Fail(error, "unit has no coded-frame OBU");
  }
  const std::size_t insert_at = obus[frame_index].offset;
  const auto split = static_cast<std::ptrdiff_t>(insert_at);
  out->clear();
  out->reserve(unit.size() + obu_bytes.size());
  out->insert(out->end(), unit.begin(), unit.begin() + split);
  out->insert(out->end(), obu_bytes.begin(), obu_bytes.end());
  out->insert(out->end(), unit.begin() + split, unit.end());
  return true;
}

namespace {

// Locates the sequence-header OBU payload and the bit position of
// film_grain_params_present.
bool FindFilmGrainPresentBit(std::span<const std::uint8_t> unit,
                             std::size_t* payload_start,
                             std::size_t* payload_bytes,
                             std::size_t* bit_in_payload, std::string* error) {
  std::vector<GrainObuView> obus;
  if (!ParseTemporalUnit(unit, &obus, error)) return false;
  for (const GrainObuView& obu : obus) {
    if (obu.type != 1) continue;  // OBU_SEQUENCE_HEADER
    const std::span<const std::uint8_t> payload = obu.payload;
    if (payload.size() < 2) return Fail(error, "short sequence header");
    // Find the trailing_one_bit: the highest logical bit index in the
    // payload.  Within a byte, logical index increases as the byte-local bit
    // position decreases (MSB first), so scan from the LSB up and take the
    // first set bit.
    int last_one = -1;
    for (std::size_t index = payload.size(); index-- > 0;) {
      if (payload[index] == 0) continue;
      for (int bit = 0; bit <= 7; ++bit) {
        if ((payload[index] >> bit) & 1U) {
          last_one = static_cast<int>(index * 8U) + (7 - bit);
          break;
        }
      }
      break;
    }
    if (last_one < 2) return Fail(error, "sequence header has no trailing bit");
    const int extension_bit = last_one - 1;
    if (((payload[static_cast<std::size_t>(extension_bit / 8)] >>
          (7 - (extension_bit % 8))) &
         1U) != 0U) {
      return Fail(error, "sequence header obu_extension_flag is set");
    }
    *payload_start = obu.offset + obu.size - payload.size();
    *payload_bytes = payload.size();
    *bit_in_payload = static_cast<std::size_t>(last_one - 2);
    return true;
  }
  return Fail(error, "unit has no sequence header OBU");
}

}  // namespace

bool ReadSequenceHeaderFilmGrainPresent(std::span<const std::uint8_t> unit,
                                        bool* present, std::string* error) {
  std::size_t payload_start = 0;
  std::size_t payload_bytes = 0;
  std::size_t bit = 0;
  if (!FindFilmGrainPresentBit(unit, &payload_start, &payload_bytes, &bit,
                               error)) {
    return false;
  }
  if (present != nullptr) {
    *present = ((unit[payload_start + bit / 8] >> (7 - (bit % 8))) & 1U) != 0U;
  }
  return true;
}

bool PatchSequenceHeaderFilmGrainPresent(std::vector<std::uint8_t>* unit,
                                         bool present, std::string* error) {
  if (unit == nullptr) return Fail(error, "null temporal unit");
  std::size_t payload_start = 0;
  std::size_t payload_bytes = 0;
  std::size_t bit = 0;
  const std::span<const std::uint8_t> view(*unit);
  if (!FindFilmGrainPresentBit(view, &payload_start, &payload_bytes, &bit,
                               error)) {
    return false;
  }
  const std::size_t byte = payload_start + bit / 8;
  const std::uint8_t mask =
      static_cast<std::uint8_t>(1U << (7 - (bit % 8)));
  if (present) {
    (*unit)[byte] = static_cast<std::uint8_t>((*unit)[byte] | mask);
  } else {
    (*unit)[byte] = static_cast<std::uint8_t>((*unit)[byte] &
                                              static_cast<std::uint8_t>(~mask));
  }
  return true;
}

}  // namespace av2enc::filters::grain
