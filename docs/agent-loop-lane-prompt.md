You are lane {{lane}} of a five-lane tool-improvement loop.
Work in {{worktree}}.

Your exclusive target: {{target}}.
Do not work on another lane's target.

Read {{ledger}}, docs/tool-quality.md, docs/tool-improvement-queue-2.md,
docs/tool-improvement-queue-3.md, docs/realization-gaps.md,
docs/negative-results.md and docs/tool-improvement-loop.md first.

Perform EXACTLY ONE iteration. Do not start a second one.
1. Pick one unblocked mechanism inside your exclusive target.
2. Baseline, then ONE contained default-off implementation change. No new
   syntax, no preset/pin change, no third-party code or source copying.
3. Measure the target probe and natural 240p QP104/106/108 with the AVM oracle.
4. Keep only a sign-consistent >= +0.15 K gain or a clear probe-gap closure;
   otherwise commit the diagnostic and revert the code.
5. Update {{ledger}} with target, patch, probe/K, kept/negative, and blockers.
   Commit the ledger and any kept patch.
6. Run the gate-off pins and tools/test.sh; they must pass.

Closed lanes: key/intra (+0.00 K), eff-perc/filter decision/refresh/rate knobs,
HF RDOQ/trellis, BAWP decision, chroma exact-rate, and the skip variants
already recorded. Do not repeat them. 4:2:0 only. Do not push or merge.
End with one line: ITERATION RESULT: ...
