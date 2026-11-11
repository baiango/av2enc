#ifndef AV2ENC_INTER_ADVANCED_COMPOUND_COMMON_H_
#define AV2ENC_INTER_ADVANCED_COMPOUND_COMMON_H_

// Compound prediction and inter-intra blending — AV2 v1.0.0 §7.13.3.14,
// §7.13.3.27–.30.
//
// `run_compound_job()` is the single transcription of the blending rules. The
// CPU reference and the CUDA kernel both call it, so they are bit-exact by
// construction.
//
// Prediction precision: `Preds[]` after `block_inter_prediction()` with
// `isCompound = 1` are at 1/16 sample precision (InterRound0 = 3,
// InterRound1 = 7, InterPostRound = 4). The intra prediction used by
// inter-intra is an integer sample value, and `dst` holds the already-clipped
// inter prediction for the inter-intra branch.

#include "advanced_common.h"

namespace av2enc::inter::adv {

// InterPostRound for compound prediction (§7.13.3.16:
// 2*FILTER_BITS - (3 + 7) = 4).
inline constexpr int kCompoundPostRound = 4;

struct CompoundJob {
  // --- output geometry, in samples of `plane` -----------------------------
  int plane = 0;
  int sub_x = 0;
  int sub_y = 0;
  BlockRect rect{};
  int bsize = tables::kBlock64x64;  // luma block size; selects the wedge mask
  int bit_depth = 8;

  // --- compound predictions (1/16 sample precision) -----------------------
  const std::int16_t* p0 = nullptr;
  int p0_stride = 0;
  const std::int16_t* p1 = nullptr;
  int p1_stride = 0;

  // --- inter-intra --------------------------------------------------------
  const std::uint8_t* intra = nullptr;
  int intra_stride = 0;

  // --- luma mask for chroma wedge/diffwtd jobs ----------------------------
  // For plane > 0 with a wedge/diffwtd mask, pass the luma mask produced by the
  // plane-0 job. `luma_mask_stride` is in samples (the luma block width).
  const std::uint8_t* luma_mask = nullptr;
  int luma_mask_stride = 0;

  // Optional: write this plane's mask here (plane 0 only).
  std::uint8_t* mask_out = nullptr;
  int mask_out_stride = 0;

  // --- output -------------------------------------------------------------
  std::uint8_t* dst = nullptr;
  int dst_stride = 0;

  // --- mode ---------------------------------------------------------------
  CompoundType type = CompoundType::kAverage;
  int cwp_weight = kCwpEqual;  // 0..16
  int wedge_index = 0;
  int wedge_sign = 0;
  int mask_type = 0;

  // Implicit mask blend (§7.13.3.14): enable_imp_msk_bld && CWP_EQUAL &&
  // YMode != GLOBAL_GLOBALMV && both refs unscaled && !compoundWarp.
  int imp_mask_eligible = 0;
  int ref_start_x[2] = {0, 0};
  int ref_start_y[2] = {0, 0};
  int mi_cols = 0;
  int mi_rows = 0;

  // --- inter-intra --------------------------------------------------------
  int is_inter_intra = 0;
  InterIntraMode interintra_mode = InterIntraMode::kDc;
};

// §7.13.3.14 get_mask(): 2 when only reference 0 lands on-screen, 0 when only
// reference 1 does, 1 otherwise.
AV2ENC_ADV_HD inline int implicit_mask_value(const CompoundJob& job, int i, int j) {
  const int last_x = (job.mi_cols * kMiSize >> job.sub_x) - 1;
  const int last_y = (job.mi_rows * kMiSize >> job.sub_y) - 1;
  const int ref_x0 = job.ref_start_x[0] + j;
  const int ref_x1 = job.ref_start_x[1] + j;
  const int ref_y0 = job.ref_start_y[0] + i;
  const int ref_y1 = job.ref_start_y[1] + i;
  const bool on0 = ref_x0 >= 0 && ref_x0 <= last_x && ref_y0 >= 0 && ref_y0 <= last_y;
  const bool on1 = ref_x1 >= 0 && ref_x1 <= last_x && ref_y1 >= 0 && ref_y1 <= last_y;
  if (on0 && !on1) {
    return 2;
  }
  if (on1 && !on0) {
    return 0;
  }
  return 1;
}

// Mask value for luma sample (i, j) of the job's luma block.
AV2ENC_ADV_HD inline int compound_luma_mask(const CompoundJob& job, int i, int j) {
  switch (job.type) {
    case CompoundType::kWedge:
      return wedge_mask_value(job.bsize, job.wedge_sign, job.wedge_index, i, j);
    case CompoundType::kDiffWtd:
      return diffwtd_mask_value(job.p0[i * job.p0_stride + j], job.p1[i * job.p1_stride + j],
                                job.mask_type, kCompoundPostRound, job.bit_depth);
    case CompoundType::kIntra:
      return intra_mode_mask_value(job.interintra_mode, job.rect.w, job.rect.h, i, j);
    case CompoundType::kAverage:
    default:
      return 32;
  }
}

// Mask value at plane sample (i, j). Luma evaluates the mask directly; chroma
// subsamples the luma mask for wedge/diffwtd, and regenerates the intra-mode
// mask on the chroma grid (§7.13.3.30).
AV2ENC_ADV_HD inline int compound_mask_sample(const CompoundJob& job, int i, int j) {
  const bool mask_from_luma =
      job.type == CompoundType::kWedge || job.type == CompoundType::kDiffWtd;
  if (job.plane == 0 || !mask_from_luma) {
    return compound_luma_mask(job, i, j);
  }
  if (job.luma_mask == nullptr) {
    return 32;  // caller error; neutral weight keeps the output defined
  }
  const int stride = job.luma_mask_stride;
  if (!job.sub_x) {
    return job.luma_mask[i * stride + j];
  }
  if (!job.sub_y) {
    return mask_subsample(job.luma_mask[i * stride + 2 * j], job.luma_mask[i * stride + 2 * j + 1],
                          0, 0, job.sub_x, job.sub_y);
  }
  return mask_subsample(job.luma_mask[(2 * i) * stride + 2 * j],
                        job.luma_mask[(2 * i) * stride + 2 * j + 1],
                        job.luma_mask[(2 * i + 1) * stride + 2 * j],
                        job.luma_mask[(2 * i + 1) * stride + 2 * j + 1], job.sub_x, job.sub_y);
}

// Blends one sample. `pred0`/`pred1` are the two compound predictions at
// 1/16 sample precision (unused for inter-intra); `intra_pred` is the integer
// intra prediction and `inter_pred` the already-clipped inter prediction
// currently in the destination (inter-intra only). This is the single
// per-sample transcription used by the CPU loop and the CUDA kernel.
AV2ENC_ADV_HD inline int compound_sample_at(const CompoundJob& job, int i, int j, int pred0,
                                            int pred1, int intra_pred, int inter_pred) {
  if (job.is_inter_intra) {
    const int m = job.type == CompoundType::kWedge
                      ? compound_mask_sample(job, i, j)
                      : intra_mode_mask_value(job.interintra_mode, job.rect.w, job.rect.h, i, j);
    return inter_intra_blend_sample(m, intra_pred, inter_pred);
  }
  if (job.type == CompoundType::kAverage) {
    if (job.imp_mask_eligible) {
      const int m = implicit_mask_value(job, i, j);
      return clip1(round2(m * pred0 + (2 - m) * pred1, 1 + kCompoundPostRound), 8);
    }
    const int cw = job.cwp_weight;
    return clip1(round2(cw * pred0 + (16 - cw) * pred1, 4 + kCompoundPostRound), 8);
  }
  const int m = compound_mask_sample(job, i, j);
  return mask_blend_sample(m, pred0, pred1, kCompoundPostRound);
}

// Mask value written to `mask_out` for a luma job.
AV2ENC_ADV_HD inline int compound_mask_out_at(const CompoundJob& job, int i, int j) {
  return job.type == CompoundType::kWedge
             ? wedge_mask_value(job.bsize, job.wedge_sign, job.wedge_index, i, j)
             : compound_luma_mask(job, i, j);
}

// Runs one compound/inter-intra job. Writes the blended samples into `dst`.
AV2ENC_ADV_HD inline void run_compound_job(const CompoundJob& job) {
  const int w = job.rect.w;
  const int h = job.rect.h;

  for (int i = 0; i < h; ++i) {
    std::uint8_t* dst_row =
        job.dst + static_cast<std::ptrdiff_t>(job.rect.y + i) * job.dst_stride + job.rect.x;
    const std::int16_t* p0_row =
        job.p0 ? job.p0 + static_cast<std::ptrdiff_t>(i) * job.p0_stride : nullptr;
    const std::int16_t* p1_row =
        job.p1 ? job.p1 + static_cast<std::ptrdiff_t>(i) * job.p1_stride : nullptr;
    const std::uint8_t* intra_row =
        job.intra ? job.intra + static_cast<std::ptrdiff_t>(i) * job.intra_stride : nullptr;
    for (int j = 0; j < w; ++j) {
      const int pred0 = p0_row ? p0_row[j] : 0;
      const int pred1 = p1_row ? p1_row[j] : 0;
      const int intra_pred = intra_row ? intra_row[j] : 0;
      dst_row[j] = static_cast<std::uint8_t>(
          compound_sample_at(job, i, j, pred0, pred1, intra_pred, dst_row[j]));
    }
    if (job.mask_out != nullptr && job.plane == 0) {
      std::uint8_t* mask_row = job.mask_out + static_cast<std::ptrdiff_t>(i) * job.mask_out_stride;
      for (int j = 0; j < w; ++j) {
        mask_row[j] = static_cast<std::uint8_t>(compound_mask_out_at(job, i, j));
      }
    }
  }
}

// Structural check: is this compound type legal for the block?
AV2ENC_ADV_HD inline bool compound_type_valid(CompoundType type, bool inter_intra,
                                              bool wedge_interintra, int bsize) {
  switch (type) {
    case CompoundType::kAverage:
      return !inter_intra;
    case CompoundType::kWedge:
      return bsize_has_wedge(bsize);
    case CompoundType::kDiffWtd:
      return !inter_intra;
    case CompoundType::kIntra:
      return inter_intra && !wedge_interintra;
  }
  return false;
}

}  // namespace av2enc::inter::adv

#endif  // AV2ENC_INTER_ADVANCED_COMPOUND_COMMON_H_
