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
3. Write build/agent-loop/spec.json for the script's envelope gate:
   {"ref": "testdata/nasa_480x240_30fps_60f.y4m",
    "bin": "build-release/av2enc", "frames": 60,
    "base_args": ["--video","--frames","60","--cpu-threads","8","--nightly"],
    "qp": 106, "target_kbps": 100.0,
    "arms": [{"name":"base","env":{}},
             {"name":"cand","env":{"AV2ENC_TOOL":"1"}}]}
   `base` must be present with an empty env. Put EVERY knob value of the
   candidate in arms: the envelope is the best K over those arms. Gates must be
   default-off, so the base arm is byte-identical to main on the same build.
4. Do NOT decide keep/revert from your own numbers. Leave source changes
   uncommitted. After this call the script runs tools/envelope.py, records
   base/best/delta, keeps src/ on PASS and reverts src/ on FAIL. Use
   tools/oracle.py yourself only to diagnose a segment (it is not the gate).
5. Update {{ledger}} with target, patch, the measured bracket, and blockers.
   Never claim an oracle gain the script did not measure.
6. Run the gate-off pins and tools/test.sh; they must pass.

Closed lanes: key/intra (+0.00 K), eff-perc/filter decision/refresh/rate knobs,
HF RDOQ/trellis, BAWP decision, chroma exact-rate, the skip variants already
recorded, and the frame-global interpolation filter (measured: EIGHTTAP 82.048,
SHARP 81.349, BILINEAR 79.293, SMOOTH 77.738; see docs/interp-filter.md).
Do not repeat them. 4:2:0 only. Do not push or merge.
End with one line: ITERATION RESULT: ...
