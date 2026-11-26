# How far is the RD pipeline from bit-optimal? A Lean 4 proof

Machine-checked answer in `RdProof.lean`. Build:

```sh
cd proofs/rdproof
elan toolchain install     # Lean 4.36.0, no Mathlib
lake build                 # 0 errors, 0 sorry
lake env lean Check.lean   # axioms audit (see below)
```

## The honest split

**Not provable here.** Global bit-optimality of the encoder. The space is
partition tree x mode x MV x transform x level; the search is heuristic, and
proving the C++ implements these DPs would need a refinement proof of the
3,913-line leaf coder. Anyone claiming "bit-optimal" without that is claiming
something they have not shown.

**Provable, and proved.** Two things that together bound the distance:

1. **The decision core is exact inside its model.** The trellis/partition DP
   returns a value below *every* candidate path, so the gap the DP leaves is
   zero. `dp2_optimal` states this for the two-stage, two-state trellis — the
   dependency shape that matters for RDOQ, because a level's rate depends on the
   previous level.
2. **The realised stream's distance from the optimum is bounded by measurable
   quantities.** `pipeline_gap_bound`:

   ```
   realised_cost - optimum  ≤  (chosenModel - LB) + lam * n * e
   ```

   where, on a real encode, every right-hand term is measurable:
   * `chosenModel` — the cost the encoder believed it paid (decision trace),
   * `LB` — a lower bound on the optimum (relaxation),
   * `e` — the per-symbol rate-model error, from modelled-vs-actual bits,
   * `lam`, `n` — the price and the symbol count.

So the deliverable is a **proof-carrying gap report**, not a claim of
optimality: "we are at most `(chosen - LB) + lam*n*e` from optimal", with the
two inputs that are not proved named as hypotheses and measured instead.

## Theorems

| theorem | statement |
|---|---|
| `lmin_le` | the minimum over a candidate set is `≤` every evaluated candidate |
| `dp2_optimal` | the trellis DP is `≤` every admissible path → modelled gap = 0 |
| `gap_nonneg` | the gap is non-negative |
| `gap_le_of_lb` | a lower bound on the optimum bounds the gap |
| `realised_gap_bound` | in-model gap plus model error bounds the realised gap |
| `exact_model_is_optimal` | exact model + chosen attains the optimum ⟹ realised cost **is** the optimum |

That last theorem carries the subtlety worth knowing: `opt ≤ chosen` alone only
gives `chosen - opt ≥ 0`. Optimality needs the chosen plan to *attain* the
optimum (`hOpt`), which is exactly the assumption a heuristic search cannot
discharge for the full space — only per superblock, by exhaustive enumeration.
The first draft of this file stated it without `hOpt` and Lean refused to prove
it, which is the proof doing its job.

## Axioms audit

```
'RdProof.lmin_le'                depends on axioms: [propext, Quot.sound]
'RdProof.dp2_optimal'            depends on axioms: [propext, Quot.sound]
'RdProof.gap_le_of_lb'           depends on axioms: [propext, Quot.sound]
'RdProof.realised_gap_bound'     depends on axioms: [propext, Quot.sound]
'RdProof.pipeline_gap_bound'     depends on axioms: [propext, Quot.sound]
'RdProof.exact_model_is_optimal' depends on axioms: [propext, Quot.sound]
```

No `sorryAx`: every theorem is complete.

## What this does NOT give you

* It does not connect the Lean model to the C++ — a gap between the proof and
  the binary remains, and closing it would need a refinement proof or a
  differential test harness that checks the C++ realises the same recurrence.
* It does not prove the candidate set contains the true optimum. That needs the
  per-superblock exhaustive enumeration (`docs/negative-results.md`, the
  partition-probe work) to establish `hOpt` for small instances.
* `e` (rate-model error) is an input, not a result. If it is large, the bound is
  loose and honest: it says the pipeline is at most that far off, which on the
  measured 720p streams is where the real headroom lives.
