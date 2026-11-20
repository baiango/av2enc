You are one iteration of the AV2ENC RD-engine improvement loop. Work in
{{worktree}}.

Read {{ledger}}, docs/rd-engine.md, docs/negative-results.md and the available
docs/rd-*.md reports first. Perform EXACTLY ONE iteration. Do not start a
second.

1. Pick the next unblocked target from {{ledger}}. Do not duplicate an item
   listed under "in flight". If the queue is exhausted, pick the next measured
   gap from the census and say why.
2. Baseline, then ONE contained default-off implementation. No new syntax; spec
   only; 4:2:0. Keep the gate-off pins byte-identical.
3. Write build/agent-loop/spec.json:
   {"ref":"testdata/nasa_480x240_30fps_60f.y4m","bin":"build-release/av2enc",
    "frames":60,
    "base_args":["--video","--frames","60","--cpu-threads","1","--nightly",
     "--video-drift-refresh-threshold","1.5","--video-drift-refresh-budget","3250"],
    "qp":118,"target_kbps":100.0,
    "arms":[{"name":"base","env":{}},
            {"name":"a1","env":{"AV2ENC_NEWGATE":"1"}},
            {"name":"a2","env":{"AV2ENC_NEWGATE":"1","AV2ENC_KNOB":"0.5"}},
            {"name":"a3","env":{"AV2ENC_NEWGATE":"1","AV2ENC_KNOB":"2"}}]}
   The runner merges the 86.013 base env into every arm. Base must stay empty.
4. Do NOT decide keep/revert. Leave source changes uncommitted; the runner
   measures and reverts on fail.
5. Update {{ledger}} with the target, patch, bracket and result.
6. Build release -Werror; keep the gate-off pins; decode any gate-on stream
   60/60 with tools/validate_dav2d.sh.

Do not push or merge. End with one line: ITERATION RESULT: <target> KEEP|FAIL|LEAD.
