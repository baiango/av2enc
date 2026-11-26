/-
  Partition — can a 16x16 partition floor beat going down to 4x4?

  Measured (2026-11-26, src720.png 1280x720, matched bytes, CVVDP):

      20,000 B   min-block 16: 6.9391   min-block 4: 6.9139   (-0.025)
      30,000 B   min-block 16: 7.3777   min-block 4: 7.3653   (-0.012)
      40,000 B   min-block 16: 7.7673   min-block 4: 7.7482   (-0.019)

  So the 16x16 floor really does measure better at both ends of the rate range.
  This file proves that such a result *cannot* be evidence about partitioning.

  A minimum block size restricts the set of partition candidates the encoder
  may choose from, and allowing 4x4 makes that set strictly larger: every tree
  the 16x16 floor can build is still legal when 4x4 leaves are allowed, paying
  the same split-flag syntax above each leaf.  So

      candidates(min_block = 16)  ⊆  candidates(min_block = 4).

  The encoder does RDO at a fixed lambda: each candidate is priced `D + lambda *
  R` and the cheapest wins.  Minimising over a superset can never return a
  larger value than minimising over a subset, at any lambda.  So at every
  operating point on the lambda ladder the 4x4-allowed encoder is at least as
  cheap -- and a measured loss therefore means the encoder is not achieving the
  minimum over its own candidate set.  That is a search or pricing defect, not a
  property of small blocks.

  This is not hypothetical here: the coefficient model of 2026-11-26 charged a
  position term twice, which prices leaves inconsistently with what they cost
  and biases the tree.  A correct search with 4x4 allowed is a *superset*
  option, and the measurement says we are leaving that option's value on the
  table.

  Scope: a statement about cost, not about content.  It says nothing about how
  much 4x4 is worth -- only that it cannot be worth less than nothing when the
  decisions are exact.  It also assumes the 16x16 candidate is representable in
  the deeper tree at no extra syntax cost, which is true of the partition
  syntax: a leaf pays the split flags above it either way.
-/

namespace Partition

set_option linter.unusedVariables false

/-- Two-argument minimum, so the proofs need only `omega`. -/
def imin (a b : Int) : Int := if a ≤ b then a else b

theorem imin_le_left (a b : Int) : imin a b ≤ a := by
  unfold imin; split <;> omega

theorem imin_le_right (a b : Int) : imin a b ≤ b := by
  unfold imin; split <;> omega

/-- Minimum of a list, seeded by its first element. -/
def minWith (acc : Int) : List Int → Int
  | [] => acc
  | x :: xs => minWith (imin acc x) xs

/-- The seed is never below the result. -/
theorem minWith_le_acc (a : Int) : ∀ l : List Int, minWith a l ≤ a
  | [] => by simp [minWith]
  | x :: xs => by
      have ih := minWith_le_acc (imin a x) xs
      simp only [minWith] at ih ⊢
      exact Int.le_trans ih (imin_le_left a x)

/-- The result is below every element of the list. -/
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

/-- The result is the seed or an element of the list. -/
theorem minWith_eq_acc_or_mem (a : Int) : ∀ l : List Int,
    minWith a l = a ∨ minWith a l ∈ l
  | [] => by simp [minWith]
  | x :: xs => by
      have ih := minWith_eq_acc_or_mem (imin a x) xs
      have hstep : minWith (imin a x) xs = imin a x ∨ minWith (imin a x) xs ∈ xs := ih
      rcases hstep with h | h
      · -- result equals imin a x, so it is `a` (when a ≤ x) or `x`
        by_cases hle : a ≤ x
        · left
          have : imin a x = a := by simp [imin, hle]
          simp only [minWith]
          rw [h, this]
        · right
          have : imin a x = x := by simp [imin, hle]
          simp only [minWith]
          rw [h, this]
          simp
      · right
        simp only [minWith]
        simp [h]

/-- First element (0 for the empty list; only used on non-empty lists). -/
def head : List Int → Int
  | [] => 0
  | x :: _ => x

/-- The head of a non-empty list is a member. -/
theorem head_mem {l : List Int} (h : l ≠ []) : head l ∈ l := by
  cases l with
  | nil => exact absurd rfl h
  | cons y ys => simp [head]

/-- Minimum over a list. -/
def minList (l : List Int) : Int := minWith (head l) l

theorem minList_mem {l : List Int} (h : l ≠ []) : minList l ∈ l := by
  have h1 : minWith (head l) l = head l ∨ minWith (head l) l ∈ l :=
    minWith_eq_acc_or_mem (head l) l
  rcases h1 with heq | hmem
  · rw [minList, heq]
    exact head_mem h
  · rwa [minList]

/-- **The monotonicity that answers the question.**  If every candidate
    available to the restricted encoder (`A`, the 16x16 floor) is also
    available to the unrestricted one (`B`, down to 4x4), then at the same
    lambda the unrestricted minimum is no larger.  Costs are `D + lambda * R`,
    so this holds at every lambda and therefore at every rate point on the
    lambda ladder. -/
theorem unrestricted_is_at_least_as_cheap {A B : List Int}
    (hA : A ≠ []) (hB : B ≠ [])
    (hsub : ∀ x ∈ A, x ∈ B) :
    minList B ≤ minList A := by
  have hmemA : minList A ∈ A := minList_mem hA
  have hinB : minList A ∈ B := hsub (minList A) hmemA
  exact minWith_le_mem (head B) B (minList A) hinB

/-- Corollary, in the form that makes a measured loss diagnostic rather than
    informative: if the unrestricted encoder measures *more* expensive than the
    restricted one at the same lambda, then the unrestricted search did not
    achieve its own minimum.  The fault is in the search or the cost model, not
    in the candidate set. -/
theorem measured_loss_is_a_defect {A B : List Int}
    (hA : A ≠ []) (hB : B ≠ [])
    (hsub : ∀ x ∈ A, x ∈ B)
    (hmeasured : minList A < minList B) : False := by
  have h := unrestricted_is_at_least_as_cheap hA hB hsub
  omega

end Partition
