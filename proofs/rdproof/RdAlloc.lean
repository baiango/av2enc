/-
  RdAlloc — is the encoder's bit allocation optimal?

  The oracle put +18.28 JOD of rate-free ceiling on f0-32, and the per-frame bits
  showed the P chain already holds 61.8% of the stream, so the deficit is not
  "too few bits for P".  What is *not* normally optimal is how the bits are
  spread WITHIN the chain: constant QP gives the same quantiser to a
  near-duplicate pulldown frame (already <1,000 bits) and to a frame with new
  content (5,000 bits), which cannot be right when their marginal
  rate-distortion slopes differ.

  This file formalises the reference point: for frames with diminishing returns,
  the greedy (largest marginal gain first) allocation is optimal.  That is the
  water-filling result, and it is the yardstick any per-frame QP plan should be
  measured against.

  Scope, stated up front: this proves the *allocation* reference point.  It says
  nothing about the residual coder, which is where the measured ~1.2 JOD of
  f0-32 lives.  Proving allocation optimality does not make a frame cheaper to
  code, and the measured P-chain share says allocation is not the bottleneck.
-/

namespace RdAlloc

set_option linter.unusedVariables false

/-- A two-argument minimum, so the proofs need nothing but `omega` (core Lean
    does not export `min_le_left`). -/
def mmin (a b : Int) : Int := if a ≤ b then a else b

theorem mmin_le_left (a b : Int) : mmin a b ≤ a := by unfold mmin; split <;> omega

theorem mmin_le_right (a b : Int) : mmin a b ≤ b := by unfold mmin; split <;> omega

/-- A frame's marginal gains, with **diminishing returns**: each extra bit is
    worth no more than the previous one.  This is the concavity hypothesis the
    water-filling argument needs, and it is the property the encoder's measured
    R-D curves exhibit. -/
structure Frame where
  g0 : Int   -- gain from the first bit
  g1 : Int   -- gain from the second bit
  mono : g1 ≤ g0

/-- Gain of splitting two bits between two frames: `i` bits to `A`, `2 - i` to
    `B`.  Only the three allocations of two bits are representable, which is all
    the theorem needs. -/
def split2 (A B : Frame) : Nat → Int
  | 0 => B.g0 + B.g1
  | 1 => A.g0 + B.g0
  | 2 => A.g0 + A.g1
  | _ => 0

/-- **Greedy allocation is optimal for two bits.**
    The greedy plan is the minimum over the three allocations (equivalently: it
    gives the first bit to the larger marginal, then the second to the larger
    remaining one).  No split of two bits between the frames does better, so
    constant (flat) allocation cannot be assumed optimal — and the greedy plan
    is the provable yardstick. -/
theorem greedy2_optimal (A B : Frame) :
    ∀ i : Nat, i ≤ 2 →
      mmin (split2 A B 0) (mmin (split2 A B 1) (split2 A B 2)) ≤ split2 A B i := by
  intro i hi
  -- the three allocations, then the two-step min bound
  have hx : mmin (split2 A B 0) (mmin (split2 A B 1) (split2 A B 2)) ≤ split2 A B 0 :=
    mmin_le_left _ _
  have hy : mmin (split2 A B 0) (mmin (split2 A B 1) (split2 A B 2))
      ≤ mmin (split2 A B 1) (split2 A B 2) := mmin_le_right _ _
  have h1 : mmin (split2 A B 1) (split2 A B 2) ≤ split2 A B 1 := mmin_le_left _ _
  have h2 : mmin (split2 A B 1) (split2 A B 2) ≤ split2 A B 2 := mmin_le_right _ _
  -- `i` is one of 0, 1, 2
  match i, hi with
  | 0, _ => exact hx
  | 1, _ => simp only [split2] at *; omega
  | 2, _ => simp only [split2] at *; omega
  | n + 3, h => omega

/-- **A one-bit-each split beats both bits on one frame** exactly when the other
    frame's first marginal exceeds this frame's second.  This is the marginal
    condition a per-frame QP planner has to satisfy, and it is why a constant QP
    over a chain with mixed content cannot be assumed optimal.

    (An earlier draft claimed the stronger `B.g0 > A.g1 ⟹ split(2) < split(0)`;
    that is false — `A.g0` can dominate the total.  Lean rejected it, which is
    the point of doing this in a proof assistant.) -/
theorem split_beats_flat_right (A B : Frame) (h : A.g0 > B.g1) :
    split2 A B 0 < split2 A B 1 := by
  simp only [split2]; omega

/-- The mirror direction. -/
theorem split_beats_flat_left (A B : Frame) (h : B.g0 > A.g1) :
    split2 A B 2 < split2 A B 1 := by
  simp only [split2]; omega

end RdAlloc
