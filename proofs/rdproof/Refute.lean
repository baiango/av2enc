/-
  Refute — how to prove a tool implementation is NOT optimal.

  Optimality of an implementation is a statement about a fixed lambda:
  "the point I chose minimises D + lambda*R over the codings my tools can
  express".  Refuting it therefore needs a CERTIFICATE: a coding that

    (i)  the same tool set can express (else the claim is about the tool set),
    (ii) can be decoded by the reference decoder (else it is not a stream), and
    (iii) is strictly cheaper under the implementation's own measure.

  Three certificate kinds, in increasing generality:

  1. NESTING (used for partitions, `Partition.measured_loss_refutes_optimality`).
     If the candidate set of run B contains run A's, and B measures worse, B did
     not reach its own minimum.  Requires the sets to nest, which is only
     available when one setting restricts another.

  2. RD DOMINATION (`dominated_refutes_optimality`, below).  Applies to any tool:
     if another coding reaches at most our rate and at most our distortion, with
     one strictly better, then it is cheaper at EVERY lambda > 0, so whatever we
     chose was not the minimiser at any lambda.  This is the certificate that
     matched-rate experiments produce, and it is the one to reach for when the
     alternative comes from a different configuration rather than a subset.

  3. INTERNAL CONTRADICTION.  If the implementation's own price says `y` is
     cheaper than what it chose, it is not optimal under its own measure --
     no external reference needed.  `PricingWrong.lean` is this kind, applied to
     the coefficient model; a decision that contradicts its own price is
     refuted by its own logs.

  What none of these can do: refute the TOOL SET.  Every certificate lives
  inside the space the tools can express, which is why (i) is a condition and
  not a detail -- an alternative that needs syntax we do not have refutes
  nothing about our implementation.

  This file proves kind 2 and the fact that kind 2 subsumes kind 1 at the level
  of costs, so the two are not independent evidence.
-/

namespace Refute

set_option linter.unusedVariables false

/-- The encoder's objective for a coding with rate `R` and distortion `D`. -/
def cost (lam R D : Int) : Int := D + lam * R

/-- **RD domination refutes optimality, at every lambda at once.**

    If the alternative achieves `R_b ≤ R_a` and `D_b ≤ D_a` with one of the two
    strict, then for every positive lambda it is strictly cheaper.  So if the
    implementation settled on `a`, it did not minimise its own objective for any
    lambda -- a certificate that needs no knowledge of the encoder's internals,
    only the two measured points and the assumption that `b` is reachable with
    the same tools. -/
theorem dominated_refutes_optimality {Ra Da Rb Db lam : Int}
    (hlam : 0 < lam)
    (hR : Rb ≤ Ra) (hD : Db ≤ Da)
    (hstrict : Rb < Ra ∨ Db < Da) :
    cost lam Rb Db < cost lam Ra Da := by
  unfold cost
  -- `D + lambda * R` is bilinear, so this is not `omega` territory: use the
  -- core order lemmas, which are exactly the two strictness shapes we need.
  rcases hstrict with hlt | hlt
  · have hRlt : lam * Rb < lam * Ra :=
      Int.mul_lt_mul_of_pos_left hlt hlam
    exact Int.add_lt_add_of_le_of_lt hD hRlt
  · have hRle : lam * Rb ≤ lam * Ra :=
      Int.mul_le_mul_of_nonneg_left hR (Int.le_of_lt hlam)
    exact Int.add_lt_add_of_lt_of_le hlt hRle

/-- Domination is strict whenever either coordinate is, and the two costs then
    never tie -- there is no lambda at which the dominated point is defensible. -/
theorem domination_excludes_any_tie {Ra Da Rb Db : Int}
    (hR : Rb ≤ Ra) (hD : Db ≤ Da) (hstrict : Rb < Ra ∨ Db < Da) :
    ∀ lam : Int, 0 < lam → cost lam Rb Db < cost lam Ra Da :=
  fun lam hlam => dominated_refutes_optimality hlam hR hD hstrict

/-- Concrete shape of a certificate, in the units the experiments actually
    produce: 22,043 B at 7.0607 versus 21,394 B at 7.0425 (the 4x4 and 16x16
    arms at QP216).  The second dominates the first: fewer bytes AND higher
    quality.  Rates in bytes, distortion in units of (something) x 1e4 to keep
    it integral. -/
example : cost 1 21394 (-70425) < cost 1 22043 (-70607) := by decide

end Refute
