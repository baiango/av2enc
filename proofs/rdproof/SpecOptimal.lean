/-
  SpecOptimal — what "our tools are better within the AV2 spec" can mean, and
  what it takes to prove it.

  "Better" is meaningless without a comparator.  For AV2 there are three, and
  only one of them is a theorem:

    1. Better than the SPEC's normative decoder — impossible and not the goal.
       Our stream must decode to exactly what §7 says; matching it is
       conformance, not superiority.
    2. Better than another encoder's CHOICES inside the same syntax — this is
       the certifiable one, and its certificate is a lower bound over the set of
       VALID STREAMS.  That set is defined by the spec (and checked by the
       reference decoder), so it is a real mathematical object.
    3. Better than an encoder that uses syntax we do not implement — not
       certifiable from our side at all, because its streams are outside the
       space we can enumerate.

  So the provable form is: *for a specified block class, our chosen coding is
  the cost-minimising element of the set of all streams the spec permits for it.*
  Two facts govern whether that statement is meaningful.

  (a) EXHAUSTIVENESS.  Optimality over the candidate list the search happened to
      visit says nothing unless that list contains every valid coding.
      `exhaustive_is_spec_optimal` is the positive direction, and
      `pruning_breaks_optimality` exhibits the negative: omit one valid coding
      and the claim silently becomes false while every internal check still
      passes.  This is why "we search partitions exhaustively" has to be
      checked against the spec's feasible set, not against the search's own
      notion of it -- and why the measured 4x4 loss matters: the search did not
      reach its own optimum even over the set it DID visit.

  (b) THE COST MUST BE THE TRUE COST.  Optimality over a wrong cost is
      optimality of the wrong problem; `PricingWrong.lean` measures exactly that
      failure on the coefficient model.

  What this does NOT give: a statement that our tools beat any particular
  encoder.  It gives a statement about the spec's feasible set, which is the only
  thing anyone can verify.  Beating SVT is an empirical claim about two
  artefacts and stays one.
-/

namespace SpecOptimal

set_option linter.unusedVariables false

def imin (a b : Int) : Int := if a ≤ b then a else b

theorem imin_le_left (a b : Int) : imin a b ≤ a := by unfold imin; split <;> omega
theorem imin_le_right (a b : Int) : imin a b ≤ b := by unfold imin; split <;> omega

/-- Minimum of a non-empty list, seeded by its head. -/
def minWith (acc : Int) : List Int → Int
  | [] => acc
  | x :: xs => minWith (imin acc x) xs

theorem minWith_le_acc (a : Int) : ∀ l : List Int, minWith a l ≤ a
  | [] => by simp [minWith]
  | x :: xs => by
      have ih := minWith_le_acc (imin a x) xs
      simp only [minWith] at ih ⊢
      exact Int.le_trans ih (imin_le_left a x)

theorem minWith_le_mem : ∀ (a : Int) (l : List Int), ∀ x ∈ l, minWith a l ≤ x := by
  intro a0 l0
  induction l0 generalizing a0 with
  | nil => intro x hx; simp at hx
  | cons y ys ih =>
      intro x hx
      have hstep : minWith (imin a0 y) ys ≤ x := by
        rcases (by simpa using hx : x = y ∨ x ∈ ys) with heq | hmem
        · rw [heq]
          exact Int.le_trans (minWith_le_acc (imin a0 y) ys) (imin_le_right a0 y)
        · exact ih (imin a0 y) x hmem
      simpa only [minWith] using hstep

def head : List Int → Int
  | [] => 0
  | x :: _ => x

def minList (l : List Int) : Int := minWith (head l) l

/-- **The positive direction.**  If the search's candidate list `C` contains
    every stream the spec permits for the block (`V`), and the cost is the true
    cost, then the cost the search settles on is at most the cost of any valid
    stream.  That is precisely "our choice is spec-optimal for this block", and
    it is checkable: `V` is enumerable on a small enough block, and each member
    is verified by the reference decoder. -/
theorem exhaustive_is_spec_optimal {C V : List Int}
    (hV : V ≠ []) (hsub : ∀ x ∈ V, x ∈ C) :
    ∀ x ∈ V, minList C ≤ x := by
  intro x hx
  exact minWith_le_mem (head C) C x (hsub x hx)

/-- **The negative direction, and the reason the claim is fragile.**  Omitting a
    single valid coding can make the search's result strictly worse than a
    stream the spec permits -- while every internal consistency check still
    passes, because nothing inside the encoder knows what it failed to visit.
    Concretely: the spec permits {2, 5}, the search visits only {5}; it returns
    5, and "5 is the minimum of everything I considered" is true and
    irrelevant, because 2 is valid and cheaper. -/
theorem pruning_breaks_optimality :
    minList [5] > (2 : Int) ∧
      (2 : Int) ∈ ([2, 5] : List Int) ∧
      ¬ (∀ x ∈ ([2, 5] : List Int), minList [5] ≤ x) := by
  decide

end SpecOptimal
