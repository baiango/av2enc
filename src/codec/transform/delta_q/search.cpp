// Encoder-side RD search for the frame DC/AC delta values.
//
// This is deliberately *not* a normative process.  It is a small, deterministic
// coordinate-descent search over the legal frame delta values.  The cost model
// uses the module's own §7.14.2/§7.14.4 quantizer on 8x8 luma / 4x4 chroma
// transforms, a scalar rate proxy, and SSE.  QP 0 with all total deltas <= 0
// stays in the §5.18.2 CodedLossless state, where every candidate has zero
// distortion; the deterministic tie-break then selects delta 0.
#include "codec/transform/delta_q/dcdeltaq.h"

#include "codec/transform/variants/tx_variants.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace av2enc::dcdeltaq {
namespace {

void set_error(std::string* error, const char* what) {
  if (error != nullptr) *error = what;
}

int clip3(int lo, int hi, int v) {
  return v < lo ? lo : (v > hi ? hi : v);
}

int clip1(int v) { return clip3(0, 255, v); }

std::size_t at(int stride, int row, int col) {
  return static_cast<std::size_t>(row) * static_cast<std::size_t>(stride) +
         static_cast<std::size_t>(col);
}

struct Plane {
  int w = 0;
  int h = 0;
  std::vector<std::uint8_t> samples;
};

int dc_predict(const Plane& recon, int x, int y, int w, int h) {
  int sum = 0;
  int count = 0;
  if (x > 0) {
    const int cx = x - 1;
    for (int j = 0; j < h; ++j) {
      sum += recon.samples[at(recon.w, y + j, cx)];
      ++count;
    }
  }
  if (y > 0) {
    const int cy = y - 1;
    for (int i = 0; i < w; ++i) {
      sum += recon.samples[at(recon.w, cy, x + i)];
      ++count;
    }
  }
  if (count == 0) return 128;
  return (sum + count / 2) / count;
}

double level_bits(const std::vector<std::int32_t>& levels, int n) {
  int eob = 0;
  for (int i = n - 1; i >= 0; --i) {
    if (levels[static_cast<std::size_t>(i)] != 0) {
      eob = i + 1;
      break;
    }
  }
  if (eob == 0) return 3.0;
  double bits = 4.0 + std::log2(static_cast<double>(eob + 1));
  for (int i = 0; i < eob; ++i) {
    const int level = levels[static_cast<std::size_t>(i)];
    if (level == 0) {
      bits += 0.35;
    } else {
      const int mag = level < 0 ? -level : level;
      bits += 1.4 + std::log2(1.0 + static_cast<double>(mag));
    }
  }
  return bits;
}

struct DeltaVector {
  int y_dc = 0;
  int u_dc = 0;
  int v_dc = 0;
  int u_ac = 0;
  int v_ac = 0;
};

void normalize(DeltaVector& d, bool equal_ac_dc, bool separate_uv) {
  if (equal_ac_dc) {
    d.u_dc = d.u_ac;
    d.v_dc = d.v_ac;
  }
  if (!separate_uv) {
    d.v_dc = d.u_dc;
    d.v_ac = d.u_ac;
  }
}

enum class Knob {
  kYDc,
  kUvDc,
  kUvAc,
  kUDc,
  kVDc,
  kUAc,
  kVAc,
  kUv,
  kU,
  kV,
};

void apply_knob(DeltaVector& d, Knob knob, int value) {
  switch (knob) {
    case Knob::kYDc:
      d.y_dc = value;
      break;
    case Knob::kUvDc:
      d.u_dc = value;
      d.v_dc = value;
      break;
    case Knob::kUvAc:
      d.u_ac = value;
      d.v_ac = value;
      break;
    case Knob::kUDc:
      d.u_dc = value;
      break;
    case Knob::kVDc:
      d.v_dc = value;
      break;
    case Knob::kUAc:
      d.u_ac = value;
      break;
    case Knob::kVAc:
      d.v_ac = value;
      break;
    case Knob::kUv:
      d.u_dc = d.u_ac = d.v_dc = d.v_ac = value;
      break;
    case Knob::kU:
      d.u_dc = d.u_ac = value;
      break;
    case Knob::kV:
      d.v_dc = d.v_ac = value;
      break;
  }
}

int knob_value(const DeltaVector& d, Knob knob) {
  switch (knob) {
    case Knob::kYDc:
      return d.y_dc;
    case Knob::kUvDc:
      return d.u_dc;
    case Knob::kUvAc:
      return d.u_ac;
    case Knob::kUDc:
      return d.u_dc;
    case Knob::kVDc:
      return d.v_dc;
    case Knob::kUAc:
      return d.u_ac;
    case Knob::kVAc:
      return d.v_ac;
    case Knob::kUv:
      return d.u_dc;
    case Knob::kU:
      return d.u_dc;
    case Knob::kV:
      return d.v_dc;
  }
  return 0;
}

std::vector<Knob> make_knobs(bool equal_ac_dc, bool separate_uv) {
  std::vector<Knob> out;
  if (!equal_ac_dc) out.push_back(Knob::kYDc);
  if (equal_ac_dc) {
    if (separate_uv) {
      out.push_back(Knob::kU);
      out.push_back(Knob::kV);
    } else {
      out.push_back(Knob::kUv);
    }
  } else {
    if (separate_uv) {
      out.push_back(Knob::kUDc);
      out.push_back(Knob::kVDc);
      out.push_back(Knob::kUAc);
      out.push_back(Knob::kVAc);
    } else {
      out.push_back(Knob::kUvDc);
      out.push_back(Knob::kUvAc);
    }
  }
  return out;
}

struct EvalResult {
  double rd = std::numeric_limits<double>::infinity();
  double sse = 0.0;
  double bits = 0.0;
  QuantFrameState quant;
  bool lossless = false;
};

class CostModel {
 public:
  CostModel(const FramePlanesView& planes, const DeltaQSearchConfig& config)
      : config_(config) {
    luma_ = {planes.width, planes.height,
             std::vector<std::uint8_t>(planes.y.begin(), planes.y.end())};
    chroma_w_ = planes.width / 2;
    chroma_h_ = planes.height / 2;
    u_ = {chroma_w_, chroma_h_,
          std::vector<std::uint8_t>(planes.u.begin(), planes.u.end())};
    v_ = {chroma_w_, chroma_h_,
          std::vector<std::uint8_t>(planes.v.begin(), planes.v.end())};
    recon_y_ = {planes.width, planes.height,
                std::vector<std::uint8_t>(
                    static_cast<std::size_t>(planes.width) *
                        static_cast<std::size_t>(planes.height),
                    128)};
    recon_u_ = {chroma_w_, chroma_h_,
                std::vector<std::uint8_t>(
                    static_cast<std::size_t>(chroma_w_) *
                        static_cast<std::size_t>(chroma_h_),
                    128)};
    recon_v_ = recon_u_;

    const int luma_cols = planes.width / 8;
    const int luma_rows = planes.height / 8;
    step_ = 1;
    const int total = luma_cols * luma_rows;
    if (config.max_luma_blocks > 0 && total > config.max_luma_blocks) {
      const double ratio =
          static_cast<double>(total) / static_cast<double>(config.max_luma_blocks);
      step_ = std::max(1, static_cast<int>(std::ceil(std::sqrt(ratio))));
    }
    for (int by = 0; by < luma_rows; by += step_) {
      for (int bx = 0; bx < luma_cols; bx += step_) {
        blocks_.push_back({bx * 8, by * 8});
      }
    }
    if (config.use_qm) {
      qm_state_.using_qmatrix = true;
      qm_state_.pic_qm_num_minus_1 = 0;
      qm_state_.qm_y[0] = 0;
      qm_state_.qm_u[0] = 0;
      qm_state_.qm_v[0] = 0;
      for (int p = 0; p < 3; ++p) {
        for (int s = 0; s < qm::kMaxSegments; ++s) {
          qm_state_.seg_qm_level[p][s] = 0;
        }
      }
    }
  }

  EvalResult evaluate(const DeltaVector& input) {
    DeltaVector d = input;
    normalize(d, config_.equal_ac_dc_q, config_.separate_uv_delta_q);

    QuantFrameInputs in;
    in.base_q_idx = config_.base_q_idx;
    in.delta_q_present = false;
    in.current_q_index = config_.base_q_idx;
    in.seq = make_sequence();
    in.frame.delta_q_y_dc = d.y_dc;
    in.frame.delta_q_u_dc = d.u_dc;
    in.frame.delta_q_v_dc = d.v_dc;
    in.frame.delta_q_u_ac = d.u_ac;
    in.frame.delta_q_v_ac = d.v_ac;
    in.num_segments = kBaseSegments;
    const QuantFrameState quant = derive_quant(in, 0);

    EvalResult out;
    out.quant = quant;
    out.lossless = quant.lossless;
    q_ac_for_lambda_ = quant.plane[0].ac;
    if (quant.lossless) {
      // §5.18.2 CodedLossless: the reversible path is exact.  Every candidate
      // has SSD 0; the rate term is constant, so the tie-break selects the
      // smallest |delta| (0).
      out.sse = 0.0;
      out.bits = 8.0 * static_cast<double>(blocks_.size());
      out.rd = lambda() * out.bits;
      ++evaluations_;
      return out;
    }

    std::fill(recon_y_.samples.begin(), recon_y_.samples.end(), 128);
    std::fill(recon_u_.samples.begin(), recon_u_.samples.end(), 128);
    std::fill(recon_v_.samples.begin(), recon_v_.samples.end(), 128);

    double sse = 0.0;
    double bits = 0.0;
    for (const BlockPos& pos : blocks_) {
      sse += code_block(luma_, recon_y_, 0, pos.x, pos.y, 8, 8, quant, &bits);
      sse += code_block(u_, recon_u_, 1, pos.x / 2, pos.y / 2, 4, 4, quant,
                        &bits);
      sse += code_block(v_, recon_v_, 2, pos.x / 2, pos.y / 2, 4, 4, quant,
                        &bits);
    }
    out.sse = sse;
    out.bits = bits;
    out.rd = sse + lambda() * bits;
    ++evaluations_;
    return out;
  }

  int evaluations() const { return evaluations_; }

 private:
  struct BlockPos {
    int x = 0;
    int y = 0;
  };

  SequenceDeltaQ make_sequence() const {
    SequenceDeltaQ seq;
    seq.monochrome = false;
    seq.separate_uv_delta_q = config_.separate_uv_delta_q;
    seq.equal_ac_dc_q = config_.equal_ac_dc_q;
    seq.base_y_dc_delta_q = 0;
    seq.base_uv_dc_delta_q = 0;
    seq.base_uv_ac_delta_q = 0;
    if (!config_.equal_ac_dc_q) {
      seq.y_dc_delta_q_enabled = true;
      seq.uv_dc_delta_q_enabled = true;
    }
    seq.uv_ac_delta_q_enabled = true;
    return seq;
  }

  double lambda() const {
    // AV1-style lambda from the AC quantizer step, scaled to the rate proxy.
    // q_ac is a step (40..~16k), not a QP.
    const double q = static_cast<double>(std::max(1, q_ac_for_lambda_));
    return config_.lambda_scale * 0.85 * (q / 8.0) * (q / 8.0);
  }

  double code_block(const Plane& src, Plane& recon, int plane, int x, int y,
                    int w, int h, const QuantFrameState& quant,
                    double* bits) {
    const int pred = dc_predict(recon, x, y, w, h);
    auto params_opt = txv::make_tx_params(w, h, txv::TxType::kDctDct);
    if (!params_opt.has_value()) return 0.0;
    const txv::TxParams& params = *params_opt;
    std::vector<std::int16_t> residual(static_cast<std::size_t>(w * h));
    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        const int s = src.samples[at(src.w, y + j, x + i)];
        residual[static_cast<std::size_t>(j * w + i)] =
            static_cast<std::int16_t>(s - pred);
      }
    }
    std::vector<double> coeff(static_cast<std::size_t>(w * h));
    txv::forward_block(residual.data(), params, coeff.data());

    std::vector<std::int32_t> levels(static_cast<std::size_t>(w * h), 0);
    QuantBlock qb;
    qb.plane = plane;
    qb.coded_w = w;
    qb.coded_h = h;
    qb.tx_size = params.tx_size;
    qb.tx_type = static_cast<int>(txv::TxType::kDctDct);
    qb.lossless = false;
    qb.allow_tcq = config_.tcq;
    qb.using_qmatrix = config_.use_qm;
    qb.qm_state = config_.use_qm ? &qm_state_ : nullptr;
    qb.q = quant.plane[plane];
    qb.coefficient_scale = params.quant_factor;
    std::string error;
    if (!quantize_block(coeff.data(), qb, levels.data(), &error)) return 0.0;
    *bits += level_bits(levels, w * h);

    std::vector<std::int32_t> dequant(static_cast<std::size_t>(w * h));
    if (!dequantize_block(levels.data(), qb, dequant.data(), &error)) {
      return 0.0;
    }
    std::vector<std::int16_t> rec_residual(static_cast<std::size_t>(w * h));
    txv::inverse_block(dequant.data(), params, rec_residual.data());
    double sse = 0.0;
    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        const int s = src.samples[at(src.w, y + j, x + i)];
        const int r = pred +
                      rec_residual[static_cast<std::size_t>(j * w + i)];
        const int v = clip1(r);
        recon.samples[at(recon.w, y + j, x + i)] =
            static_cast<std::uint8_t>(v);
        const int d = s - v;
        sse += static_cast<double>(d) * d;
      }
    }
    return sse;
  }

  DeltaQSearchConfig config_;
  int chroma_w_ = 0;
  int chroma_h_ = 0;
  Plane luma_;
  Plane u_;
  Plane v_;
  Plane recon_y_;
  Plane recon_u_;
  Plane recon_v_;
  std::vector<BlockPos> blocks_;
  int step_ = 1;
  int evaluations_ = 0;
  int q_ac_for_lambda_ = 64;
  qm::QmFrameState qm_state_;
};

bool delta_less(int a, int b) {
  const int aa = a < 0 ? -a : a;
  const int bb = b < 0 ? -b : b;
  if (aa != bb) return aa < bb;
  return a < b;
}

std::vector<int> make_candidates(const DeltaQSearchConfig& config) {
  static const int kLadder[] = {-24, -16, -12, -8, -6, -4, -3, -2, -1, 0,
                                1,   2,   3,   4,  6,  8,  12, 16, 24};
  const int bound =
      std::max(0, std::min(config.max_delta_magnitude, kFrameDeltaMax));
  std::vector<int> out;
  for (int value : kLadder) {
    if (value < -bound || value > bound) continue;
    if (config.base_q_idx == 0 && !config.allow_lossy_at_qp0 && value > 0) {
      continue;
    }
    out.push_back(value);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

bool to_frame(const DeltaVector& d, const SequenceDeltaQ& seq,
              FrameDeltaQ* frame, std::string* error) {
  FrameDeltaQ f;
  f.delta_q_y_dc = d.y_dc;
  f.delta_q_u_dc = d.u_dc;
  f.delta_q_v_dc = d.v_dc;
  f.delta_q_u_ac = d.u_ac;
  f.delta_q_v_ac = d.v_ac;
  if (!validate_frame_delta_q(seq, f, error)) return false;
  *frame = f;
  return true;
}

}  // namespace

bool search_frame_delta_q(const FramePlanesView& planes,
                          const DeltaQSearchConfig& config,
                          DeltaQSearchResult* result,
                          std::string* error) {
  if (result == nullptr) {
    set_error(error, "search_frame_delta_q: null result");
    return false;
  }
  *result = DeltaQSearchResult{};
  if (planes.width <= 0 || planes.height <= 0 ||
      (planes.width & 7) != 0 || (planes.height & 7) != 0) {
    set_error(error, "search_frame_delta_q: width/height must be multiples of 8");
    return false;
  }
  const std::size_t luma = static_cast<std::size_t>(planes.width) *
                           static_cast<std::size_t>(planes.height);
  const std::size_t chroma =
      static_cast<std::size_t>(planes.width / 2) *
      static_cast<std::size_t>(planes.height / 2);
  if (planes.y.size() < luma || planes.u.size() < chroma ||
      planes.v.size() < chroma) {
    set_error(error, "search_frame_delta_q: short plane buffer");
    return false;
  }
  if (config.base_q_idx < 0 || config.base_q_idx > kMaxQ) {
    set_error(error, "search_frame_delta_q: base_q_idx outside [0, 255]");
    return false;
  }
  if (config.max_passes < 0 || config.max_passes > 64) {
    set_error(error, "search_frame_delta_q: max_passes outside [0, 64]");
    return false;
  }
  if (config.max_delta_magnitude < 0 ||
      config.max_delta_magnitude > kFrameDeltaMax) {
    set_error(error, "search_frame_delta_q: max_delta_magnitude outside [0, 63]");
    return false;
  }

  SequenceDeltaQ seq;
  seq.monochrome = false;
  seq.separate_uv_delta_q = config.separate_uv_delta_q;
  seq.equal_ac_dc_q = config.equal_ac_dc_q;
  seq.base_y_dc_delta_q = 0;
  seq.base_uv_dc_delta_q = 0;
  seq.base_uv_ac_delta_q = 0;
  if (!config.equal_ac_dc_q) {
    seq.y_dc_delta_q_enabled = true;
    seq.uv_dc_delta_q_enabled = true;
  }
  seq.uv_ac_delta_q_enabled = true;
  if (!validate_sequence_delta_q(seq, error)) return false;

  CostModel model(planes, config);
  const std::vector<int> candidates = make_candidates(config);
  const std::vector<Knob> knobs =
      make_knobs(config.equal_ac_dc_q, config.separate_uv_delta_q);

  DeltaVector current;
  EvalResult best = model.evaluate(current);
  for (int pass = 0; pass < config.max_passes; ++pass) {
    bool improved = false;
    for (Knob knob : knobs) {
      const int start_value = knob_value(current, knob);
      int best_value = start_value;
      EvalResult best_here = best;
      for (int candidate : candidates) {
        if (candidate == start_value) continue;
        DeltaVector trial = current;
        apply_knob(trial, knob, candidate);
        normalize(trial, config.equal_ac_dc_q, config.separate_uv_delta_q);
        const EvalResult eval = model.evaluate(trial);
        const bool better_rd = eval.rd < best_here.rd - 1e-9;
        const bool tie = std::fabs(eval.rd - best_here.rd) <= 1e-9;
        if (better_rd ||
            (tie && delta_less(candidate, best_value))) {
          best_here = eval;
          best_value = candidate;
        }
      }
      if (best_value != start_value) {
        apply_knob(current, knob, best_value);
        normalize(current, config.equal_ac_dc_q, config.separate_uv_delta_q);
        best = best_here;
        improved = true;
      }
    }
    if (!improved) break;
  }

  FrameDeltaQ frame;
  if (!to_frame(current, seq, &frame, error)) return false;

  QuantFrameInputs in;
  in.base_q_idx = config.base_q_idx;
  in.current_q_index = config.base_q_idx;
  in.seq = seq;
  in.frame = frame;
  in.num_segments = kBaseSegments;
  const QuantFrameState quant = derive_quant(in, 0);

  result->seq = seq;
  result->frame = frame;
  result->quant = quant;
  result->rd = best.rd;
  result->sse = best.sse;
  result->bits = best.bits;
  result->evaluations = model.evaluations();
  result->lossless = best.lossless;
  return true;
}

}  // namespace av2enc::dcdeltaq
