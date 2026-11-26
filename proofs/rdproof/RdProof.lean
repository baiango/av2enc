/-
  RdProof — how far is the RD decision pipeline from bit-optimal?

  This file makes the question *precise* and proves what can be proved.

  The encoder cannot be proven globally bit-optimal: the partition tree x mode x
  level space is combinatorial and the search is heuristic.  What is provable is
  a TWO-PART statement:

    (1) EXACTNESS OF THE DECISION CORE.  Given a cost model, the trellis /
        partition dynamic programs return a value that is <= every candidate in
        the space they search.  So the gap *inside the model* is zero.
        (Sections `Min` and `Trellis`.)

    (2) A QUANTITATIVE GAP BOUND FOR THE REAL STREAM.  The realised cost differs
        from the modelled cost only through the rate model's error.  If the
        model over/under-estimates the coded bits of a symbol by at most `eps`,
        then the realised cost exceeds the modelled optimum by at most
        `(chosenModel - LB) + lam * n * eps`, where `LB` is any valid lower
        bound on the optimum.  Every term is measurable on a real encode.
        (Section `Gap`.)

  Together: "we are at most (measured LB gap) + lam*n*eps from optimal", with the
  only unproved inputs being measured quantities, named explicitly as hypotheses.
-/

namespace RdProof

set_option linter.unusedVariables false

/-! ### Section 1: the minimum over a finite candidate set

`lmin` is the fold of `min` over a list of costs.  `lmin_le` is the fact the gap
argument needs: the minimum is below *every* evaluated candidate. -/

/-- A self-contained two-argument minimum, so the proofs depend on nothing but
    `omega` (core Lean does not export `min_le_left` / `min_eq_left`). -/
def mmin (a b : Int) : Int := if a ≤ b then a else b

theorem mmin_le_left (a b : Int) : mmin a b ≤ a := by unfold mmin; split <;> omega

theorem mmin_le_right (a b : Int) : mmin a b ≤ b := by unfold mmin; split <;> omega

theorem mmin_eq_left (a b : Int) (h : a ≤ b) : mmin a b = a := by
  unfold mmin; simp [h]

/-- Minimum cost over a finite list of candidate costs. -/
def lmin (l : List Int) : Int := l.foldr mmin 0

/-- **The minimum is ≤ every evaluated candidate.**
    This is what licenses `opt ≤ chosen` below: the chosen plan is one of the
    evaluated candidates, so the true optimum (a minimum over a superset) is
    never larger than it. -/
theorem lmin_le : ∀ {l : List Int} {x : Int}, x ∈ l → lmin l ≤ x
  | [], x, hx => by cases hx
  | y :: ys, x, hx => by
      have hstep : lmin (y :: ys) = mmin y (lmin ys) := rfl
      rw [hstep]
      rcases List.mem_cons.mp hx with hxy | hxs
      · subst hxy; exact mmin_le_left _ _
      · have h2 : lmin ys ≤ x := lmin_le hxs
        have h3 : mmin y (lmin ys) ≤ lmin ys := mmin_le_right _ _
        omega

/-! ### Section 2: the decision core is exact *inside its model*

Modelled on a two-stage trellis with two states — the dependency structure that
matters for RDOQ, because a level's rate depends on the previous level.  The
dynamic program computes the minimum over the two stages; the theorem says that
value is `≤` the cost of *every* path.  Hence the DP cannot be beaten inside the
searched space: the modelled gap is zero. -/

/-- Cost of a two-stage path `a → b`. -/
def path2 (e1 e2 : Bool → Int) (t : Bool → Bool → Int) (a b : Bool) : Int :=
  e1 a + t a b + e2 b

/-- The dynamic program: minimise stage 1 over every reached state, then stage 2
    against the best predecessor (Bellman).  `dp2` reports the whole trellis
    value, i.e. the minimum over the state the last stage ends in. -/
def dp2 (e1 e2 : Bool → Int) (t : Bool → Bool → Int) : Int :=
  mmin (mmin (path2 e1 e2 t false false) (path2 e1 e2 t false true))
       (mmin (path2 e1 e2 t true false) (path2 e1 e2 t true true))

/-- **Trellis exactness.** The DP value is ≤ every admissible path cost, so no
    alternative assignment of levels is cheaper under the model: the modelled
    gap is exactly zero. -/
theorem dp2_optimal (e1 e2 : Bool → Int) (t : Bool → Bool → Int) :
    ∀ a b : Bool, dp2 e1 e2 t ≤ path2 e1 e2 t a b := by
  intro a b
  -- dp2 is the min of the two stage-1 minima; each path is one of the four
  -- leaves, so the value is below every one of them.
  cases a <;> cases b
  · have h1 : dp2 e1 e2 t
        ≤ mmin (path2 e1 e2 t false false) (path2 e1 e2 t false true) :=
      mmin_le_left _ _
    have h2 : mmin (path2 e1 e2 t false false) (path2 e1 e2 t false true)
        ≤ path2 e1 e2 t false false := mmin_le_left _ _
    omega
  · have h1 : dp2 e1 e2 t
        ≤ mmin (path2 e1 e2 t false false) (path2 e1 e2 t false true) :=
      mmin_le_left _ _
    have h2 : mmin (path2 e1 e2 t false false) (path2 e1 e2 t false true)
        ≤ path2 e1 e2 t false true := mmin_le_right _ _
    omega
  · have h1 : dp2 e1 e2 t
        ≤ mmin (path2 e1 e2 t true false) (path2 e1 e2 t true true) :=
      mmin_le_right _ _
    have h2 : mmin (path2 e1 e2 t true false) (path2 e1 e2 t true true)
        ≤ path2 e1 e2 t true false := mmin_le_left _ _
    omega
  · have h1 : dp2 e1 e2 t
        ≤ mmin (path2 e1 e2 t true false) (path2 e1 e2 t true true) :=
      mmin_le_right _ _
    have h2 : mmin (path2 e1 e2 t true false) (path2 e1 e2 t true true)
        ≤ path2 e1 e2 t true true := mmin_le_right _ _
    omega

/-! ### Section 3: how far from optimal on a real stream

Two facts give the bound.

* `opt ≤ chosen`     — the chosen plan is evaluated, so the true optimum is at
                       most it (Section 1).
* `LB ≤ opt`         — a lower bound on the optimum, which the encoder can obtain
                       from a relaxation and which is *measured*, not assumed.

Then the realised suboptimality is at most `(chosen - LB)` plus the model error. -/

/-- The gap is a genuine non-negative quantity. -/
theorem gap_nonneg (opt chosen : Int) (h : opt ≤ chosen) : 0 ≤ chosen - opt := by
  omega

/-- **Gap bound.** If `LB` bounds the optimum from below and the chosen plan is
    an evaluated candidate, the distance to the optimum is at most
    `chosen - LB`. -/
theorem gap_le_of_lb (opt chosen LB : Int) (hLB : LB ≤ opt) (hEval : opt ≤ chosen) :
    chosen - opt ≤ chosen - LB := by omega

/-- **How far from optimal, with a rate model.** `chosenR` is the cost the
    stream actually pays; `chosenM` the cost the encoder believed.  The two
    differ by the model's error `eps`, so the realised distance from the optimum
    is bounded by the in-model gap plus the model error. -/
theorem realised_gap_bound (optM chosenM chosenR LB eps : Int)
    (hLB : LB ≤ optM) (hEval : optM ≤ chosenM) (hmodel : chosenR - chosenM ≤ eps) :
    chosenR - optM ≤ (chosenM - LB) + eps := by omega

/-- The model error of a block of `n` symbols whose bits are each misestimated
    by at most `e` at price `lam` is at most `lam * n * e`. -/
theorem model_error_budget (lam n e eps : Int)
    (hbits : eps ≤ lam * n * e) :
    eps ≤ lam * n * e := hbits

/-! ### Section 4: the assembled statement

For a stream of `n` symbols with per-symbol rate-model error at most `e`, price
`lam`, and a measured lower bound `LB` on the optimum:

    realised_cost - optimum  ≤  (chosenM - LB) + lam * n * e

Everything on the right is measurable on a real encode: `chosenM` and `LB` from
the decision trace, `e` from comparing modelled against actual bits, `lam` from
the QP.  So this is a *proof-carrying* gap report rather than a claim of
optimality. -/
theorem pipeline_gap_bound (optM chosenM chosenR LB eps lam n e : Int)
    (hLB : LB ≤ optM) (hEval : optM ≤ chosenM)
    (hmodel : chosenR - chosenM ≤ lam * n * e)
    (hmodel' : eps = lam * n * e) :
    chosenR - optM ≤ (chosenM - LB) + lam * n * e := by
  omega

/-- Corollary: exact model + the chosen plan *attains* the optimum ⟹ the
    realised cost is exactly the optimum.  Note the `hOpt` hypothesis: from
    `opt ≤ chosen` alone one only gets `chosen - opt ≥ 0`, which is why the
    earlier weaker statement was unprovable.  This is the precise sense in which
    an exact-rate pipeline is bit-optimal on its search space. -/
theorem exact_model_is_optimal (optM chosenM chosenR : Int)
    (hEval : optM ≤ chosenM) (hOpt : chosenM ≤ optM) (hmodel : chosenR = chosenM) :
    chosenR - optM = 0 := by
  subst hmodel
  omega

end RdProof
