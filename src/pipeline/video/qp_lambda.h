// QP-scaled RD lambdas for the video encoder.
//
// Five decisions price a rate term against an SSE-domain distortion and used a
// constant calibrated at one operating point:
//
//   site                        old constant              old QP dependence
//   --video-leaf-rd-lambda      125 at QP104              sqrt(step) only
//   --video-tx-lambda           136 at QP110              none
//   --video-key-tx-lambda       0 (inherits 136)          none
//   --video-rdoq-scale          0.1 at QP110              step^2 (per-TX)
//   partition fallback price    700 at QP190              none
//
// `lambda = c * step^2` is the SSE-domain relation, but the measured optimum
// does not follow it for every site: the leaf weight, the TX-search weight and
// the coefficient trellis each have their own measured exponent.  This header
// keeps the fits in one place so the encoder and the docs cannot drift.
//
// All fits are anchored at the quantiser step where the legacy constant was
// measured, so `qp_lambda_scale` off (the default) reproduces the historical
// streams byte for byte.  With scaling on, a site whose existing CLI flag was
// given explicitly keeps that exact value: the flags remain escape hatches.
//
// Measured at the two operating points (see docs/qp-scaled-lambda.md):
//   240p/100 kbps  nasa_480x240, QP 110, step 944
//   720p/300 kbps  space8k_1440x720, QP 217, step 20480, key offset -57
#pragma once

#include <cmath>

namespace av2enc::video {

// lambda(step) = anchor * (step / anchor_step)^exponent.
struct LambdaPowerFit {
  double anchor;       // legacy constant at the anchor quantiser step
  double anchor_step;  // §7.14.2 qlookup() step at the anchor QP
  double exponent;     // measured power law
};

inline double lambda_from_fit(double step, const LambdaPowerFit& fit) {
  if (!(step > 0.0) || !(fit.anchor_step > 0.0)) return 0.0;
  return fit.anchor * std::pow(step / fit.anchor_step, fit.exponent);
}

// --video-leaf-rd-lambda.  Anchor 125 at QP104 (luma step qlookup/8 = 98).
// Measured optimum: ~130 at QP110 (flat 125..150), ~200..300 in the legacy
// anchor's units at QP217.  The legacy sqrt() form (exponent 0.5) reaches
// 125*sqrt(26.1) = 639 there, about half the measured optimum; exponent 0.72
// puts the two points at 143 and 1322 effective SSE-per-bit.
inline constexpr LambdaPowerFit kLeafLambdaFit{125.0, 98.0, 0.72};

// --video-tx-lambda (TX search and the inter leaf guard).  Anchor 136 at
// QP217, the 720p optimum and the existing nightly constant, so the 720p
// stream is bit-identical; the measured exponent puts QP110 at 48, the 240p
// optimum measured at iso-rate.
inline constexpr LambdaPowerFit kTxLambdaFit{136.0, 2560.0, 0.3385};

  // --video-key-tx-lambda.  The intra TX search wants ~4..8 at both keyframe
  // points (QP 93 and QP 160): the weight is content-calibrated, not
  // QP-calibrated, so the measured exponent is zero.
inline constexpr LambdaPowerFit kKeyTxLambdaFit{8.0, 66.0, 0.0};

// --video-rdoq-scale.  The stored quantity is the scale s in
// lambda_coeff = s * quant_step^2, so an exponent of -0.5 makes the
// coefficient-domain lambda scale as step^1.5: the 240p optimum 0.1 becomes
// 0.0215 at QP217, matching the measured 720p optimum (0.02).
inline constexpr LambdaPowerFit kRdoqSseScaleFit{0.1, 118.0, -0.5};

// --video-partition-rd-lambda.  Same shape: s in lambda = s * step^2, with
// step the full qlookup() value (944 at QP110, 9216 at QP190).  The measured
// scale doubles over the operating range (2.75e-6 -> 5.5e-6), i.e. the probe
// lambda itself scales as step^2.22.
inline constexpr LambdaPowerFit kPartitionProbeFit{2.75e-6, 944.0, 0.22};

// The variance + syntax-bit fallback used for the keyframe tree and for the
// P-frame nodes the probe cannot model.  The legacy constant is 700 SSE/bit
// at QP190; the measured two-point fit is 7.34 at QP110 (the 240p keyframe
// wants fine leaves) and ~620 at QP217, i.e. step^1.44, not step^2: a single
// 700 at both points has the wrong sign on some nodes.
inline constexpr LambdaPowerFit kPartitionModelFit{7.344, 944.0, 1.44};

}  // namespace av2enc::video
