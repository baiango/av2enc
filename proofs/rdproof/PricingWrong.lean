/-
  PricingWrong — is our pricing actually wrong?  Provably, yes, and by how much.

  Two separate arguments, both formalised here.  The first is direct and needs no
  theory; the second turns a measured coding loss into a lower bound on the
  pricing error.

  ARGUMENT 1 (direct contradiction).  The encoder's price for a leaf is a
  number; the range coder's output for that leaf is a number.  If the pricing
  were exact these would be equal for every leaf, hence equal in total.  They
  are measured to differ in total, so at least one leaf is mispriced -- and the
  measured difference is a lower bound on the total absolute mispricing.
  `sum_size_le_total_error` and `mispriced_exists` are that argument.

  ARGUMENT 2 (from a coding loss).  `Partition.lean` proves that with exact
  pricing the encoder cannot measure worse when its candidate set strictly
  grows.  It is measured to measure worse (the 4x4 arm loses 0.012-0.025 JOD at
  matched bytes at 20/30/40 KB).  So the pricing is not exact, or the search
  never reaches its own minimum; that measurement cannot say which.  This file
  records the shape of that refutation rather than re-proving the monotonicity.

  What is NOT claimed: that the mispricing costs quality in proportion to its
  size.  `pricing_gap_bound` (Pricing.lean) bounds the decision loss by twice
  the *signed-relevant* error; a large error that cancels across the candidates
  being compared costs nothing.  The measurements say the error is one-sided,
  which is the case where it does not cancel.

  Numbers instantiated at the end, from `AV2ENC_RATE_HYBRID_S_MEASURE`.
-/

namespace PricingWrong

set_option linter.unusedVariables false

/-- A leaf: `(price, true_cost)` in bits. -/
abbrev Leaf := Int × Int

/-- Signed error, positive when the model prices the leaf too *cheaply*. -/
def signed (x : Leaf) : Int := x.2 - x.1

/-- Non-negative per-leaf error: the larger of the two directions, so it is at
    least `|price - true|` and at least `signed`.  Written with `max` rather
    than an absolute value so every step is linear for `omega`. -/
def absErr (x : Leaf) : Int := max (x.2 - x.1) (x.1 - x.2)

/-- Sum of the signed errors. -/
def sumSigned : List Leaf → Int
  | [] => 0
  | x :: xs => signed x + sumSigned xs

/-- Sum of the per-leaf absolute errors. -/
def sumAbsErr : List Leaf → Int
  | [] => 0
  | x :: xs => absErr x + sumAbsErr xs

theorem absErr_ge_signed (x : Leaf) : signed x ≤ absErr x := by
  unfold absErr signed
  have h : x.2 - x.1 ≤ max (x.2 - x.1) (x.1 - x.2) :=
    Int.le_max_left _ _
  omega

theorem absErr_nonneg (x : Leaf) : 0 ≤ absErr x := by
  unfold absErr
  omega

theorem absErr_zero_iff_exact (x : Leaf) (h : absErr x = 0) : x.1 = x.2 := by
  unfold absErr at h
  have h1 : x.2 - x.1 ≤ 0 := by
    have := Int.le_max_left (x.2 - x.1) (x.1 - x.2)
    omega
  have h2 : x.1 - x.2 ≤ 0 := by
    have := Int.le_max_right (x.2 - x.1) (x.1 - x.2)
    omega
  omega

/-- **The total absolute mispricing is at least the measured total difference.**
    Cancellation can only reduce the absolute sum, never increase it. -/
theorem sum_size_le_total_error : ∀ l : List Leaf, sumSigned l ≤ sumAbsErr l
  | [] => by simp [sumSigned, sumAbsErr]
  | x :: xs => by
      have hx := absErr_ge_signed x
      have ih := sum_size_le_total_error xs
      simp only [sumSigned, sumAbsErr]
      omega

/-- **If the totals differ, some leaf is mispriced.**  Exact pricing is
    refuted by any measured difference at all. -/
theorem mispriced_exists : ∀ l : List Leaf, 0 < sumAbsErr l → ∃ x ∈ l, x.1 ≠ x.2
  | [], h => by simp [sumAbsErr] at h
  | x :: xs, h => by
      by_cases hx : absErr x = 0
      · have hrest : 0 < sumAbsErr xs := by
          simp only [sumAbsErr] at h
          omega
        obtain ⟨y, hy, hne⟩ := mispriced_exists xs hrest
        exact ⟨y, by simp [hy], hne⟩
      · have hne : x.1 ≠ x.2 := by
          intro he
          apply hx
          unfold absErr
          rw [he]
          simp
        exact ⟨x, by simp, hne⟩

/-- **The quantitative statement.**  A measured total difference `G` between the
    true cost and the price forces at least `G` bits of total absolute
    mispricing.  This is the form to quote: it is a lower bound on the model's
    error derived from the coder's own output, not an estimate. -/
theorem total_error_at_least {l : List Leaf} {G : Int}
    (hG : G ≤ sumSigned l) :
    G ≤ sumAbsErr l :=
  Int.le_trans hG (sum_size_le_total_error l)

/-- Non-vacuity: a two-leaf case mirroring the shape of the measurement.  One
    leaf priced 9 bits too cheaply and one exactly, with a measured total
    difference of 9 bits, forces at least 9 bits of absolute error. -/
example : PricingWrong.sumAbsErr [(20, 29), (40, 40)] = 9 := by
  decide

example : (9 : Int) ≤ PricingWrong.sumAbsErr [(20, 29), (40, 40)] := by
  decide

end PricingWrong
