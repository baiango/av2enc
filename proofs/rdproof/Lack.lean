/-
  Lack — can it be proved what av2enc lacks?

  Two directions, and only one is a theorem.

  (1) CONSTRUCTIVE (provable).  If encoding with a tool T is cheaper than
      encoding without it at matched rate, then the encoder without T lacks the
      value of T.  The certificate is the stream: anyone can reproduce it.  This
      is the only direction that is checkable, and it is what the DIP measurement
      is: DIP alone is worth +0.012 JOD, so av2enc-without-DIP lacks that.

  (2) CAUSAL (not provable).  "av2enc lacks X, and X is why it is 0.394 JOD
      behind AVM" additionally requires showing that no other mechanism produces
      the same gain -- a quantification over an open space of tools, which no
      measurement can discharge.  A tool set that is missing is not a fact about
      the world that can be witnessed; only a specific missing stream can.

  (3) THE HOLE THIS FILE PROVES.  The six-tool sweep tested each tool ALONE.
      Singleton elimination does not bound combinations: tools interact, and a
      cost function can make every singleton non-improving while a pair is
      strictly better.  So "we measured six tools and none helped" is NOT
      evidence that no subset helps.  `singleton_elimination_is_not_compositional`
      is that counterexample, which is why the combination runs exist.

  What this means for answering "what does av2enc lack": the provable content is
  a list of measured, reproducible gains (DIP: +0.012 JOD) plus internal defects
  proved elsewhere (`PricingWrong`, `Partition.measured_loss_refutes_optimality`).
  The claim "what it lacks versus AVM" stays an empirical hypothesis, and the
  deficit map in docs/negative-results.md is its evidence, not its proof.
-/

namespace Lack

/-- Encoding cost for a set of enabled tools.  Tools 1 and 2 are individually
    redundant and jointly useful -- the shape real coding tools have, because a
    predictor and the transform that models its residual only pay together. -/
def cost : List Int → Int
  | [] => 10
  | [1] => 10
  | [2] => 10
  | [1, 2] => 8
  | _ => 99

/-- **Singleton elimination is not compositional.**  Both tools are neutral
    alone (`cost [1] = cost [] = cost [2]`) and the pair is strictly better
    (`cost [1,2] = 8 < 10 = cost []`).  So a sweep of individual switches cannot
    bound what a subset would do. -/
theorem singleton_elimination_is_not_compositional :
    cost [1] ≤ cost [] ∧ cost [2] ≤ cost [] ∧ cost [1, 2] < cost [] := by
  decide

/-- The same fact stated as the logical point: "no singleton helps" does not
    imply "no subset helps". -/
theorem no_singleton_helps_does_not_imply_no_subset_helps :
    ¬ (∀ c : List Int → Int,
        (c [1] ≤ c [] ∧ c [2] ≤ c []) → c [1, 2] ≥ c []) := by
  intro h
  have := h cost (by decide)
  revert this
  decide

end Lack
