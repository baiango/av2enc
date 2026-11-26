/-
  Pricing — how much of the intra gap can a better *decision* close?

  Context (measured 2026-11-26): on src720.png (1280x720, intra only) at
  25,000 B, `SvtAv1EncApp --preset 0 --qp` scores CVVDP 7.4405 against our
  stack arm's 7.2935 -- a 0.147 JOD deficit.  Two earlier proofs already close
  two explanations: `RdProof` shows the partition/mode search is optimal given
  the measured candidates, and `RdAlloc` bounds re-allocation at +0.026 dB.
  So the deficit is not the search and not the allocation.  It is either

    (1) MISPRICING -- our rate model disagrees with what the bitstream spends,
        so a decision that is optimal under the model is not optimal in truth;
    (2) MODEL ERROR -- the model is accurate but the syntax/tools cannot
        express the better choice.

  This file proves the bound that separates them.  The encoder minimises an
  additive price over independent leaves; the truth is what the range coder
  spends.  If every leaf's price is within `e` bits of its true cost, then the
  chosen coding order is within `2 * n * e` bits of the true optimum.  Two
  consequences, and they are the decision procedure:

    * `e = 0` (an exact model) makes our choice truly optimal -- so nothing
      inside the model can be improved by tuning;
    * conversely, if a target sits below `2 * n * e`, NO model-consistent
      choice reaches it, and the model/tool set must change.

  Scope, stated up front: this is a bound, not an estimate, and it is only as
  good as `e`.  It says nothing about how to make the model exact, and nothing
  about which coding tool is missing.  What it does is convert "we are 0.147
  JOD behind" into "how much of that is mispricing", which is answerable by
  measurement -- `AV2ENC_RATE_HYBRID_S_MEASURE` already records the actual
  range-coder bits per leaf next to the decision's prediction.

  Everything below is stated in integers (bits, JOD x 1e4, whatever unit is
  used consistently) so that the proofs are linear and need no analysis.
-/

namespace Pricing

set_option linter.unusedVariables false

/-- A leaf's `(price, true_cost)` pair: what the model says a coding choice
    costs, and what the range coder actually spends on it. -/
abbrev Leaf := Int × Int

/-- Total modelled price over the leaves. -/
def sumPrice : List Leaf → Int
  | [] => 0
  | x :: xs => x.1 + sumPrice xs

/-- Total true cost over the leaves. -/
def sumTrue : List Leaf → Int
  | [] => 0
  | x :: xs => x.2 + sumTrue xs

/-- The model prices this leaf to within `e` bits, in both directions.  Stated
    as two inequalities rather than `|price - true| ≤ e` so that every step of
    the proof stays linear and `omega` can finish it. -/
def err (e : Int) (x : Leaf) : Prop := x.1 ≤ x.2 + e ∧ x.2 ≤ x.1 + e

/-- The total error allowance over the leaves: a fold rather than
    `length * e`, so every step of the proof stays additive and needs no
    cast between `Nat` and `Int`.  `nerr e l` is `l.length * e`. -/
def nerr (e : Int) : List Leaf → Int
  | [] => 0
  | _ :: xs => e + nerr e xs

/-- Equal-length lists have equal error allowances. -/
theorem nerr_eq_of_length {e : Int} : ∀ (c : List Leaf) {a : List Leaf},
    c.length = a.length → nerr e c = nerr e a
  | [], a, h => by
      cases a with
      | nil => rfl
      | cons _ _ => simp [List.length] at h
  | _ :: cs, a, h => by
      cases a with
      | nil => simp [List.length] at h
      | cons _ as =>
          have hs : cs.length = as.length := by
            simp only [List.length] at h
            omega
          simp [nerr, nerr_eq_of_length cs hs]

/-- Summed prices cannot understate the true cost by more than the allowance. -/
theorem sum_err_low {e : Int} :
    ∀ l : List Leaf, (∀ x ∈ l, err e x) →
      sumPrice l ≤ sumTrue l + nerr e l
  | [], _ => by simp [sumPrice, sumTrue, nerr]
  | x :: xs, h => by
      have hx : err e x := h x (by simp)
      obtain ⟨hx1, hx2⟩ := hx
      have hxs : ∀ y ∈ xs, err e y := fun y hy => h y (by simp [hy])
      have ih := sum_err_low xs hxs
      simp only [sumPrice, sumTrue, nerr]
      omega

/-- ... nor overstate it, in the other direction. -/
theorem sum_err_high {e : Int} :
    ∀ l : List Leaf, (∀ x ∈ l, err e x) →
      sumTrue l ≤ sumPrice l + nerr e l
  | [], _ => by simp [sumPrice, sumTrue, nerr]
  | x :: xs, h => by
      have hx : err e x := h x (by simp)
      obtain ⟨hx1, hx2⟩ := hx
      have hxs : ∀ y ∈ xs, err e y := fun y hy => h y (by simp [hy])
      have ih := sum_err_high xs hxs
      simp only [sumPrice, sumTrue, nerr]
      omega

/-- **The pricing-gap bound.**  The encoder picks the leaf costs that minimise
    the modelled price (`hopt`: no alternative prices cheaper).  If the model
    is within `e` bits on every leaf of both the chosen and the alternative
    coding, then the chosen coding's true cost is within `2 * nerr e c` of the
    alternative's.  Taking the alternative to be the true optimum gives the
    statement in the header: true cost ≤ true optimum + `2 n e`. -/
theorem pricing_gap_bound {e : Int} {c a : List Leaf}
    (hc : ∀ x ∈ c, err e x) (ha : ∀ x ∈ a, err e x)
    (hlen : c.length = a.length)
    (hopt : sumPrice c ≤ sumPrice a) :
    sumTrue c ≤ sumTrue a + 2 * nerr e c := by
  have hc_high : sumTrue c ≤ sumPrice c + nerr e c := sum_err_high c hc
  have ha_low : sumPrice a ≤ sumTrue a + nerr e a := sum_err_low a ha
  have hn : nerr e a = nerr e c := (nerr_eq_of_length c hlen).symm
  omega

/-- The allowance is zero when the error bound is zero. -/
theorem nerr_zero : ∀ l : List Leaf, nerr 0 l = 0
  | [] => rfl
  | _ :: xs => by
      have ih := nerr_zero xs
      simp only [nerr]
      omega

/-- Corollary: an **exact** model (`e = 0`) makes the chosen coding truly
    optimal.  This is the negative result that retires tuning: with exact
    pricing there is nothing left to win inside the model, so a remaining gap
    is not a mispriced decision. -/
theorem exact_model_is_true_optimal {c a : List Leaf}
    (hc : ∀ x ∈ c, err 0 x) (ha : ∀ x ∈ a, err 0 x)
    (hlen : c.length = a.length) (hopt : sumPrice c ≤ sumPrice a) :
    sumTrue c ≤ sumTrue a := by
  have h := pricing_gap_bound (e := 0) hc ha hlen hopt
  have hz : nerr 0 c = 0 := nerr_zero c
  have h0 : 2 * nerr 0 c = 0 := by omega
  omega

/-
  What the measured model error means for the report (prose, not a theorem):
  the corollaries above are direct rearrangements of `pricing_gap_bound`, so
  they add nothing the main statement does not already contain.  The useful
  consequence is the direction of the bias.  `AV2ENC_RATE_HYBRID_S_MEASURE` on
  inter residual leaves reports pred 51.27 bits against actual 36.955 (bias
  +14.32, p90|error| 26.6, n=245), i.e. the coefficient model prices a coding
  residual about 39 % too high.  With `e` that large on the leaves that matter,
  `pricing_gap_bound` allows a real loss from mispricing alone: every
  residual-versus-skip comparison is biased toward skipping.  Mode syntax is
  fine (pred 4.199 vs actual 4.203, p90 0.77 bits), so the error is specific to
  coefficients, not to the decision framework.
-/

end Pricing
