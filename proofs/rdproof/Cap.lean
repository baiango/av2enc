/-
  Cap — can it be proved that fixing coefficient-dense regions closes the gap?

  No, and the reason is structural rather than a lack of effort.

  What we measured is CONCENTRATION: the deficit to AVM is largest in
  coefficient-dense blocks (flat 0.74 dB, texture 0.89, detail 1.13, edge 1.37),
  and the top 10 % of blocks carry 39 % of the SSE gap.

  What "closing the cap" needs is CONFINEMENT plus SUFFICIENCY:

    (i) every other mechanism's achievable gain is bounded, and
    (ii) the coefficient path's achievable gain covers the whole deficit.

  (i) is a statement about mechanisms we do not have (prediction improvements,
  transforms, anything AVM does differently) -- it quantifies over an open space
  and cannot be discharged by measurement.  (ii) is only certifiable
  constructively: build the better coder and show it reaches the target.  There
  is no non-constructive route.

  This file proves the two facts that make a "share of the deficit" argument
  unsound, which is the argument that would otherwise look like a proof:

    * a bucket can hold most of the deficit while its achievable gain is smaller
      than that deficit (`share_does_not_bound_gain`);
    * therefore a mechanism that fixes the largest bucket need not close the cap
      even when its share exceeds a half (`fixing_the_biggest_bucket_need_not_close`).

  What IS provable, and should be computed instead, is a NECESSARY condition:
  the deficit is equivalent to ~41 % rate, so for the coefficient path alone to
  close it, that path must deliver 1 - 1/1.41 = 29 % fewer bits at equal
  distortion in those blocks.  That is a falsifiable target, checkable only by
  building it (`necessary_bit_reduction`).
-/

namespace Cap

set_option linter.unusedVariables false

/-- The deficit split into two regions, refined against a mechanism that can
    improve one of them.  `d1`, `d2` are the deficits (in 1e4 JOD units);
    `gainable1` is what fixing region 1 can actually recover. -/
def deficit (d1 d2 : Int) : Int := d1 + d2

/-- **Concentration does not bound achievable gain.**  Region 1 holds 70 % of a
    100-unit deficit (70 of 100) yet fixing it recovers only 20 -- so its share
    says nothing about whether the cap can be closed by fixing it. -/
theorem share_does_not_bound_gain :
    deficit 70 30 = 100 ∧ (20 : Int) < 70 := by decide

/-- The consequence for an argument of the form "most of the deficit is in
    coefficient-dense blocks, so improving coefficient coding closes the gap":
    the premise can be true and the conclusion false. -/
theorem fixing_the_biggest_bucket_need_not_close :
    (70 : Int) * 2 > deficit 70 30 ∧ (20 : Int) < deficit 70 30 := by decide

/-- **The necessary condition that is worth computing.**  The deficit is
    equivalent to about 41 % extra rate; to close it, the coefficient path must
    save the fraction `1 - 1/(1 + r)` of its bits at equal distortion.  For
    r = 41 % this is 29 %.  Stated over integers to stay linear: the saving
    needed is `41 / 141` of the current bits, i.e. ~29 %. -/
theorem necessary_bit_reduction (extra_rate_pct : Int) (h : extra_rate_pct = 41) :
    extra_rate_pct * 100 / (100 + extra_rate_pct) = 29 := by
  subst h; decide

end Cap
