# Still decision stage status (2026-11-13)

Owner: `src/cuda/pipeline/still/` (orchestrator + `decide/`).

Wired and gated (`--still-gpu`): TCQ (`--still-tcq`) and RDOQ
(`--still-rdoq`).  Details, parity numbers and the device rate-model design are
in `src/cuda/pipeline/still/decide/README.md` and
`src/cuda/pipeline/still/README.md`.

Still rejected by name: `--still-tx-partition` and `--still-rdo-intra`.

## `--still-tx-partition` (§5.20.6.1–.3)

Not wired.  The device pieces exist (the §5.20.6.3 geometry/plan tables in
`src/cuda/codec/transform/tx_partition/`), but the product path needs four
things that are not in the orchestrator today:

1. **Per-part prediction/reconstruction feedback.**  `choose_luma_tx_partition`
   re-predicts every part from the reconstruction of the previous part
   (`encode_luma_parts`), including the §7.13.2.15/.16 smooth flags taken at the
   *coding-block* origin.  The GPU TX stage (`still_tx_batch`) cannot run inside
   a kernel, so a trial needs a predict -> transform -> recon launch chain per
   part (up to 4 parts x 4 candidates per block) plus the
   `count_top_right`/`count_bottom_left` replay and the per-part decoded-flag
   update the CPU performs through `decoded_.fill(...)`.
2. **Snapshot/restore** of the block's recon region and decoded cells between
   candidates (`take_luma_snapshot`/`restore_luma_snapshot`).
3. **Syntax parts.**  `BlockDecision` carries one luma TX; the writer
   (`still_syntax.cpp`) and the bridge (`still_stream_bridge.cpp`) need
   `tx_part_count`, `tx_partition` and up to four `LumaTxPart` records with
   per-part coefficients.
4. **Coefficient regions.**  A 64x64 coding block's single transform is the
   adjusted 32x32 (1024 elements), but a SPLIT/HORZ/VERT trial can carry up to
   4 x 1024 elements, so `fill_packed_offsets`/`plan.coeff_count_y` must reserve
   four times the per-block luma region when the flag is on.

## `--still-rdo-intra` (§7.13.2)

Implemented on this branch and **reverted**: the frame bytes were byte-identical
to `--still --still-rdo-intra` at 64x64 and 640x360, but at 640x360 the
encoder-side **deblocked** reconstruction differed from the CPU reference at
`(96, 0)` (925 of 230400 Y samples, +-1 LSB).  With `--no-deblock` the same
stream's recon was bit-exact, and the *pre-deblock* recon matched in both cases,
so the divergence is in the deblock stage's input/metadata for the modes this
decision selects (the same block data deblocks identically under
`--still-full-intra`).  The internal `verify_against_cpu` recon gate therefore
fails even though the bitstream matches; shipping it would hide a GPU/CPU
reconstruction difference.

What the implementation did (reusable):

* one device launch predicts all 61 candidates (batched `IntraJob`s into
  `61 x prediction-area` scratch), one `still_tx_batch` forward-transforms them,
  and a one-thread select kernel reproduces
  `intra_candidate_rate_bits` / `better_candidate` exactly (cost, SSE, rate,
  mode order, input index) using `SSE + lambda * bits` with the CPU's
  placeholder `default_rdoq_*` model (`lambda_sse_from_quant_step(528,
  rdo_sse_scale)`); `log2(eob+1)` comes from a host-built small-integer table,
  `floor(log2(level+1))` is an integer bit length;
* the chosen mode/angle/SSE replace `LumaSearchResult`, so the existing
  chosen-mode prediction, MRL/DIP ordering and palette/IBC compares keep
  working;
* the candidate SSEs come from the search's `LumaSearchDebug` channel (they
  matched the CPU's `sse_against_source` values exactly in testing).

Next step: compare the per-MI `DeblockMi` grid and the deblock output for a
block whose mode only `--still-rdo-intra` selects (directional modes with a
non-zero `AngleDeltaY`), then re-enable.
