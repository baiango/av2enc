/-
  Cause — can the *cause* of the pricing error be proved?

  Short answer: not positively.  A discrepancy tells you that the model is wrong;
  it cannot tell you which of many mechanisms produced it, and "the numbers fit
  mechanism X, therefore X caused them" is abduction, not proof -- any
  sufficiently flexible hypothesis can be fitted to three points.

  What *can* be proved is elimination: show that a candidate mechanism is
  inconsistent with the measurements, and it is genuinely ruled out.  Three
  eliminations are available, two already measured and one proved here.

  1. NOT "the model is static, contexts would fix it".  Measured: the writer's
     own write-time -log2(p) under the real adaptive contexts carries the same
     bias as the analytic surrogate (-7.86 vs -9.00 bits at QP198, -6.76 vs
     -7.43 at QP180, -3.28 vs -3.33 at QP220).  Adaptive and static pricing
     agree with each other and disagree with the coder.

  2. NOT "a wrong constant in the additive family".  That family prices a leaf
        price = overhead + magnitude + position * eob
     so its bias against the coder is affine in `eob`: bias = alpha + beta*eob.
     Three measured rates over-determine (alpha, beta), and the three points are
     *not* collinear, so no (alpha, beta) reproduces them.  `no_affine_bias_fits`
     proves exactly that from the measurements.  This is the useful result: it
     says the defect is not a tuning constant at all, so further constant-fitting
     (which is what the first two attempts at a fix were) cannot succeed by
     construction.

  3. NOT "4x4 partitions are intrinsically worse" (Partition.lean, from a
     different measurement).

  What remains, and what this file does NOT establish: that the residual cost is
  non-additive over scan positions -- that it depends on context state, TX shape
  or the significance-map structure in a way no `overhead + magnitude +
  position*eob` form can express.  That is the surviving hypothesis, and it is
  supported by eliminations rather than by direct measurement.

  What *would* establish a cause positively: a predicted-delta intervention.
  Change one term, predict how the bias must move, measure that it moves by
  exactly that.  The 0.5 -> 0 position change did move the bias by the predicted
  -0.31*eob (16.0 -> 3.0 mean absolute), which is causal evidence that the
  double-counted position term was *a* cause.  It was not the whole cause: the
  surviving -3.5..+4.6 bit spread is what elimination 2 says no additive form can
  absorb.
-/

namespace Cause

set_option linter.unusedVariables false

/-- Eliminations 2, as arithmetic.  The measured bias of the additive family
    against the coder, in hundredths of a bit, at the measured mean `eob` for
    each rate (also in hundredths):

      QP220   eob 10.72   bias  +9.76
      QP198   eob 26.91   bias +14.32
      QP180   eob 54.73   bias +23.82

    An affine bias `bias = alpha + beta * eob` requires the secant slopes between
    consecutive points to agree.  Written over a common denominator, with
    `beta = beta_num / beta_den`:

      beta_num * 1619 =   456 * beta_den     (QP220 -> QP198)
      beta_num * 4401 =  1406 * beta_den     (QP220 -> QP180)

    These have no solution with `beta_den ≠ 0`: cross-multiplying gives
    `beta_den * (456*4401 - 1406*1619) = 0` and the bracket is -269458.  So the
    additive family provably cannot reproduce the measurements, at any constants.
-/
theorem no_affine_bias_fits :
    ¬ (∃ (beta_num beta_den : Int),
          beta_den ≠ 0 ∧
          beta_num * 1619 = 456 * beta_den ∧
          beta_num * 4401 = 1406 * beta_den) := by
  rintro ⟨bn, bd, hbd, h1, h2⟩
  -- scale the two secant equations to a common left-hand side
  have e1 : bn * 1619 * 4401 = 456 * bd * 4401 := by omega
  have e2 : bn * 4401 * 1619 = 1406 * bd * 1619 := by omega
  have hsq : bn * 1619 * 4401 = bn * 4401 * 1619 := by omega
  rw [hsq] at e1
  -- now 456*bd*4401 = 1406*bd*1619, i.e. bd * (-269458) = 0
  have hzero : bd * (456 * 4401 - 1406 * 1619) = 0 := by omega
  have hbracket : (456 * 4401 - 1406 * 1619 : Int) ≠ 0 := by decide
  have : bd = 0 := by
    rcases Int.mul_eq_zero.mp hzero with h | h
    · exact h
    · exact absurd h hbracket
  exact hbd this

end Cause
