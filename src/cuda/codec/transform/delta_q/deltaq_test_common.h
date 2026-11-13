// Host-side case generator shared by the delta-Q emulation and GPU tests.
//
// Every case carries the CPU oracle result (`quantize_block` /
// `dequantize_block` from `src/codec/transform/delta_q/`), so both tests
// compare the device core against the reference bit for bit.  This header is
// test-only and is never included by the kernel translation unit.
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include "codec/transform/delta_q/dcdeltaq.h"
#include "codec/transform/qm/qm.h"
#include "cuda/codec/transform/delta_q/deltaq_types.h"

namespace av2enc::cuda::deltaq::test {

struct Case {
  Job job;
  std::vector<double> coeff;
  std::vector<std::int32_t> expected_levels;
  std::vector<std::int32_t> expected_dequant;
  std::vector<std::int32_t> q2;  // non-empty when job.q2_offset >= 0
  bool expect_reject = false;
};

inline int adjusted(int coded) {
  return coded < 32 ? coded : 32;
}

inline std::uint32_t next_random(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

// Deterministic coefficients: zeros, lround ties, extremes and a spread that
// produces non-zero levels over several qindex values.
inline std::vector<double> make_coeff(std::size_t count, int variant) {
  std::vector<double> out(count, 0.0);
  std::uint32_t state = 0x9e3779b9u ^ static_cast<std::uint32_t>(variant * 7919);
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint32_t r = next_random(&state);
    const int bucket = static_cast<int>(r % 16u);
    double value = 0.0;
    if (bucket == 0) {
      value = 0.0;
    } else if (bucket == 1) {
      value = (i & 1u) ? 0.5 : -0.5;
    } else if (bucket == 2) {
      value = (i & 1u) ? 1.5 : -1.5;
    } else if (bucket == 3) {
      value = static_cast<double>(static_cast<int>(r % 2001u) - 1000) * 97.0;
    } else if (bucket == 4) {
      value = (i & 1u) ? 1.0e7 : -1.0e7;
    } else {
      value = static_cast<double>(static_cast<int>(r % 4001u) - 2000) / 8.0;
    }
    out[i] = value;
  }
  return out;
}

inline void fill_expected(const dcdeltaq::QuantBlock& block,
                          const std::vector<double>& coeff,
                          Case* out) {
  out->expected_levels.assign(coeff.size(), 0);
  out->expected_dequant.assign(coeff.size(), 0);
  std::string error;
  if (!dcdeltaq::quantize_block(coeff.data(), block,
                                out->expected_levels.data(), &error)) {
    out->expect_reject = true;
    return;
  }
  if (!dcdeltaq::dequantize_block(out->expected_levels.data(), block,
                                  out->expected_dequant.data(), &error)) {
    out->expect_reject = true;
  }
}

// Builds the same job fields the device consumes from the CPU-side delta-Q
// state.  `r` is the CPU `resolve_delta_q()` result for this plane.
inline void fill_job(Job* job, int plane, int coded_w, int coded_h,
                     int qindex, int tx_type, int delta_dc, int delta_ac,
                     bool lossless, bool allow_tcq, bool use_fsc,
                     double coefficient_scale) {
  job->in_offset = 0;
  job->out_offset = 0;
  job->plane = plane;
  job->coded_w = coded_w;
  job->coded_h = coded_h;
  job->qindex = qindex;
  job->delta_dc = delta_dc;
  job->delta_ac = delta_ac;
  job->tx_type = tx_type;
  job->q2_offset = -1;
  job->lossless = lossless ? 1 : 0;
  job->allow_tcq = allow_tcq ? 1 : 0;
  job->use_fsc = use_fsc ? 1 : 0;
  job->coefficient_scale = coefficient_scale;
}

inline dcdeltaq::SequenceDeltaQ make_sequence(int slot) {
  dcdeltaq::SequenceDeltaQ seq;
  seq.equal_ac_dc_q = false;
  seq.separate_uv_delta_q = (slot & 1) != 0;
  seq.y_dc_delta_q_enabled = true;
  seq.uv_dc_delta_q_enabled = true;
  seq.uv_ac_delta_q_enabled = true;
  seq.base_y_dc_delta_q = -8 + (slot % 5) * 4;      // [-8, 8]
  seq.base_uv_dc_delta_q = -5 + (slot % 4) * 3;     // [-5, 4]
  seq.base_uv_ac_delta_q = -3 + (slot % 3) * 3;     // [-3, 3]
  return seq;
}

inline dcdeltaq::FrameDeltaQ make_frame(int slot) {
  dcdeltaq::FrameDeltaQ frame;
  frame.delta_q_y_dc = (slot % 7) - 3;
  frame.delta_q_u_dc = (slot % 5) - 2;
  frame.delta_q_v_dc = (slot % 3) - 1;
  frame.delta_q_u_ac = (slot % 4) - 1;
  frame.delta_q_v_ac = (slot % 6) - 3;
  return frame;
}

// The main systematic matrix: sizes x planes x TX_TYPEs x TCQ/FSC x qindex.
inline std::vector<Case> make_cases() {
  using namespace dcdeltaq;
  std::vector<Case> cases;

  const int sizes[][2] = {{4, 4},  {8, 8},  {16, 16}, {32, 32},
                          {64, 64}, {4, 8},  {16, 32}, {32, 8}};
  const int types[] = {0, 3, 9, 10, 15};  // DCT, ADST_ADST, IDTX, V_DCT, H_FLIPST
  const int qps[] = {1, 8, 32, 64, 90, 141, 191, 255};
  int counter = 0;

  for (const auto& size : sizes) {
    for (int plane = 0; plane < 3; ++plane) {
      for (int type : types) {
        for (int tcq = 0; tcq < 2; ++tcq) {
          for (int fsc = 0; fsc < 2; ++fsc) {
            const int slot = counter++;
            const int base_q_idx = qps[slot % 8];
            const SequenceDeltaQ seq = make_sequence(slot);
            const FrameDeltaQ frame = make_frame(slot);
            const ResolvedDeltaQ r = resolve_delta_q(seq, frame);

            QuantFrameInputs in;
            in.base_q_idx = base_q_idx;
            in.current_q_index = base_q_idx;
            in.seq = seq;
            in.frame = frame;
            const QuantFrameState state = derive_quant(in, 0);

            Case c;
            fill_job(&c.job, plane, size[0], size[1], state.qindex, type,
                     plane == 0 ? r.y_dc : (plane == 1 ? r.u_dc : r.v_dc),
                     plane == 0 ? 0 : (plane == 1 ? r.u_ac : r.v_ac),
                     false, tcq != 0, fsc != 0,
                     8.0 / (static_cast<double>(1 + (slot % 4)) * 528.0));

            QuantBlock block;
            block.plane = plane;
            block.coded_w = size[0];
            block.coded_h = size[1];
            block.tx_size = 0;
            block.tx_type = type;
            block.segment_id = 0;
            block.lossless = false;
            block.allow_tcq = tcq != 0;
            block.use_fsc = fsc != 0;
            block.q = state.plane[plane];
            block.coefficient_scale = c.job.coefficient_scale;

            const std::size_t count = static_cast<std::size_t>(
                adjusted(size[0]) * adjusted(size[1]));
            c.coeff = make_coeff(count, slot);
            fill_expected(block, c.coeff, &c);
            cases.push_back(std::move(c));
          }
        }
      }
    }
  }

  // Effective qindex from segments / delta_q_present.  The quantizers use
  // get_qindex(), not the raw base_q_idx the coefficient CDFs select from.
  for (int variant = 0; variant < 6; ++variant) {
    QuantFrameInputs in;
    in.base_q_idx = 90;
    in.current_q_index = 100 + variant * 17;
    in.delta_q_present = (variant % 2) != 0;
    in.num_segments = 8;
    in.alt_q_enabled[1] = true;
    in.alt_q[1] = -255;  // lossless segment when delta_q_present = 0
    in.seq = make_sequence(3 + variant);
    in.frame = make_frame(2 + variant);
    const QuantFrameState state = derive_quant(in, 1);
    const ResolvedDeltaQ r = resolve_delta_q(in.seq, in.frame);

    Case c;
    fill_job(&c.job, 0, 8, 8, state.qindex, 0, r.y_dc, 0, false,
             variant % 3 == 0, false, 8.0 / (3.0 * 528.0));
    QuantBlock block;
    block.plane = 0;
    block.coded_w = 8;
    block.coded_h = 8;
    block.allow_tcq = variant % 3 == 0;
    block.q = state.plane[0];
    block.coefficient_scale = c.job.coefficient_scale;
    c.coeff = make_coeff(64, 300 + variant);
    fill_expected(block, c.coeff, &c);
    cases.push_back(std::move(c));
  }

  // Lossless pass-through: quantize is llround and dequant is a copy.
  for (int size_index = 0; size_index < 3; ++size_index) {
    const int w = sizes[size_index][0];
    const int h = sizes[size_index][1];
    Case c;
    fill_job(&c.job, 0, w, h, 0, 0, 0, 0, true, false, false, 1.0);
    QuantBlock block;
    block.plane = 0;
    block.coded_w = w;
    block.coded_h = h;
    block.lossless = true;
    block.q = {64, 64};
    const std::size_t count =
        static_cast<std::size_t>(adjusted(w) * adjusted(h));
    c.coeff = make_coeff(count, 100 + size_index);
    fill_expected(block, c.coeff, &c);
    cases.push_back(std::move(c));
  }

  // Quantizer-matrix interplay: CPU build_q2_matrix() output is uploaded and
  // the device consumes it verbatim.  One identity user matrix and one default
  // matrix level.
  for (int variant = 0; variant < 2; ++variant) {
    qm::QmFrameState qm_state;
    qm_state.using_qmatrix = true;
    qm_state.seg_qm_level[0][0] = 3;
    qm_state.seg_qm_level[1][0] = 3;
    qm_state.seg_qm_level[2][0] = 3;
    qm_state.qm_data_present[3] = variant == 0;
    if (variant == 0) {
      for (int p = 0; p < 3; ++p) {
        qm::fill_user_qm_constant(&qm_state.user_qm[3], 32);
      }
    }
    Case c;
    fill_job(&c.job, 1, 8, 8, 90, 0, -4, 6, false, false, false,
             8.0 / (2.0 * 528.0));
    QuantBlock block;
    block.plane = 1;
    block.coded_w = 8;
    block.coded_h = 8;
    block.tx_size = 1;  // 8x8
    block.tx_type = 0;
    block.using_qmatrix = true;
    block.qm_state = &qm_state;
    block.q = {dcdeltaq::get_q(90, c.job.delta_dc),
               dcdeltaq::get_q(90, c.job.delta_ac)};
    block.coefficient_scale = c.job.coefficient_scale;
    c.q2.assign(64, 0);
    std::string error;
    if (!dcdeltaq::build_q2_matrix(block, c.q2.data(), &error)) {
      c.expect_reject = true;
    }
    c.job.q2_offset = 0;
    c.coeff = make_coeff(64, 200 + variant);
    fill_expected(block, c.coeff, &c);
    cases.push_back(std::move(c));
  }

  // Rejected jobs: bad plane, bad coded size, bad TX type, and a non-positive
  // q2 entry.  The device must report -1 and leave the output untouched.
  const int invalid[][4] = {
      {3, 8, 8, 0},   // plane
      {0, 0, 8, 0},   // coded width
      {0, 8, 0, 0},   // coded height
      {0, 8, 8, 16},  // TX type
  };
  for (const auto& spec : invalid) {
    Case c;
    fill_job(&c.job, spec[0], spec[1], spec[2], 90, spec[3], 0, 0, false,
             false, false, 1.0);
    c.coeff.assign(64, 3.0);
    c.expected_levels.assign(64, -12345);
    c.expected_dequant.assign(64, -12345);
    c.expect_reject = true;
    cases.push_back(std::move(c));
  }
  {
    Case c;
    fill_job(&c.job, 0, 8, 8, 90, 0, 0, 0, false, false, false, 1.0);
    c.q2.assign(64, 33);
    c.q2[17] = 0;  // non-positive q2 -> rejected
    c.job.q2_offset = 0;
    c.coeff.assign(64, 3.0);
    c.expected_levels.assign(64, -12345);
    c.expected_dequant.assign(64, -12345);
    c.expect_reject = true;
    cases.push_back(std::move(c));
  }

  return cases;
}

}  // namespace av2enc::cuda::deltaq::test
