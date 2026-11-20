# Per-frame reference signalling (§5.18.2)

Status: **probe + opt-in `--video-ref-signal-compact` implemented and gated.**
Against the nightly pin the compact form codes **417 instead of 948 reference
bits** over 60 frames and the stream is **61 B smaller (24,885 → 24,824,
−0.245%)** with an identical tile payload and an identical decoded picture
stream. Code: `src/pipeline/video/video_headers.cpp` (probe),
`src/pipeline/video/video_frame.cpp` (`VideoOptions::ref_signal_compact`),
`src/main.cpp` (`--video-ref-signal-compact`). Gate:
`tools/gate_video_ref_signalling.sh`.

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/compact.ivf \
    --video --frames 60 --qp 110 --cpu-threads 8 --fast-decode \
    --video-ref-signal-compact
# 24,824 B, md5 d4c151b25e8472b9688378a11a1fca9a
```

## What AV2 v1.0.0 actually codes

The AV1 names this task started from do not exist in AV2 v1.0.0:

| AV1 concept | AV2 v1.0.0 (§5.18.2) |
|---|---|
| `frame_refs_short_signaling` + `last_frame_idx` | **absent**. The map is either the full explicit list or the §7.7 derived list, selected by the *sequence* flag `explicit_ref_frame_map` |
| `ref_frame_idx[0..6]` | present, `f(CeilLog2(NumRefFrames))` each, after `num_total_refs f(3)` |
| `ref_order_hint[i]` (per-frame list) | **absent**. `RefOrderHint[]` is decoder slot state (§7.7 ranks slots by it); the frame codes only its own `order_hint f(OrderHintBits)` |
| `frame_refs_short_signaling` cost | 0 bits |

So there is no per-frame short-ref flag to flip. The only per-frame reference
syntax is the primary-ref block, `refresh_frame_flags`, and the explicit map
(flag + `num_total_refs` + indices).

## Measured field cost

`AV2ENC_TRACE_REF_BITS=1` makes the frame-header writer print one line per
coded frame:

```text
refsig key=<0|1> hint=<bits> primary=<bits> cdf=<bits> refresh=<bits> \
       map=<bits> nrefs=<bits> refidx=<bits> n=<total_refs> header=<bits> \
       total_ref=<bits>
```

`primary` is `signal_primary_ref_frame` + `primary_ref_frame`; `cdf` is
`disable_cross_frame_cdf_init`, which lives in the same §5.18.2 block and is
what the carried-CDF chain needs; `nrefs` is `num_total_refs`; `refidx` is the
`ref_frame_idx[]` list. Sum with awk on `=`/space:

```sh
AV2ENC_TRACE_REF_BITS=1 av2enc … --fast-decode 2>&1 >/dev/null | grep refsig |
  awk -F'[= ]' '{h+=$5; p+=$7; c+=$9; r+=$11; m+=$13; n+=$15; i+=$17; t+=$23}
                 END {print h,p,c,r,m,n,i,t}'
```

Nightly pin (`--fast-decode`, 60 frames, 59 P + 1 key):

| Field | § | baseline bits/frame | baseline total | compact bits/frame | compact total |
|---|---|---:|---:|---:|---:|
| `order_hint` | 5.18.2 | 4 (all 60 frames) | 240 | 4 | 240 |
| `signal_primary_ref_frame` | 5.18.2 | 1 (P) | 59 | 1 | 59 |
| `primary_ref_frame` | 5.18.2 | 3 (P, when signalled) | 177 | 0 (not signalled) | 0 |
| `disable_cross_frame_cdf_init` | 5.18.2 | 1 (P) | 59 | 1 | 59 |
| `refresh_frame_flags` | 5.18.2 | 2 (P, `f(2)`) | 118 | 1 (P, `f(1)`) | 59 |
| `frame_explicit_ref_frame_map` | 5.18.2 | 1 (P) | 59 | 0 (sequence flag 0) | 0 |
| `num_total_refs` | 5.18.2 | 3 (P) | 177 | 0 (derived) | 0 |
| `ref_frame_idx[0]` | 5.18.2 | 1 (P) | 59 | 0 (derived) | 0 |
| `ref_order_hint[]` | — | 0 (derived) | 0 | 0 | 0 |
| **total** | | **15.8 (avg)** | **948 = 118.5 B** | **7.0 (avg)** | **417 = 52.1 B** |

The key frame codes only its `order_hint` (4 bits) in both forms: with
`max_mlayer_id = 0`, `OBU_CLOSED_LOOP_KEY` implies `refresh_frame_flags =
allFrames` and `primary_ref_frame = PRIMARY_REF_NONE`.

Savings by field: `primary_ref_frame` 177, `num_total_refs` 177,
`refresh_frame_flags` 59, `frame_explicit_ref_frame_map` 59,
`ref_frame_idx` 59 → **531 bits = 66.4 B of syntax**. The IVF shrinks by
**61 B**, because the header is byte-aligned after §5.19's
`is_first_tile_group` bit and 5.4 B of the saving falls inside alignment
padding.

## What is derivable, and why

**Reference map (5 bits/frame).** `explicit_ref_frame_map = 0` makes the
decoder run §7.7 `get_ref_frames()`. With `NumRefFrames = 1` the only slot is
the one the key refreshed (`allFrames` = slot 0), so the derivation yields
exactly `NumTotalRefs = 1, ref_frame_idx[0] = 0` — the same list the encoder
codes explicitly today. This is the "omitting a ref that is implied" form: it
is the sequence header that removes the bits, not a per-frame flag. (With
`NumRefFrames = 2` the same derivation would return *two* distinct slots after
the first P frame, because slot 0 holds the key and slot 1 the previous P; that
is why the flag also pins `NumRefFrames = 1`.)

**`primary_ref_frame` (4 bits/frame).** §5.18.2
`choose_primary_secondary_ref_frame()` only counts a reference whose
`RefFrameType` is `INTER_FRAME`. With one reference:

* P frames 2..59 reference the previous P → derived primary is `ref_frame_idx`
  index 0, exactly the `primary_ref_frame = 0` coded today (so the §7.4.4 CDF
  seed is the same slot);
* frame 1 references the key → derived primary is `PRIMARY_REF_NONE` in both
  forms, and `set_primary_ref_frame_and_ctx()` forces
  `disable_cross_frame_cdf_init = 1`, which the encoder already codes for that
  frame.

The proof that the derived form installs the same CDF image at every frame is
observable: the **tile payload is byte-identical**
(`tile 22219 bytes` for 60 frames) and the carried §8.2 image is a function of
the coded symbols, so an identical tile means an identical seed.

**`refresh_frame_flags` (2 → 1 bit).** The compact form writes slot 0 on every
frame with `NumRefFrames = 1`, so the field is `f(1)`. The historical chain
alternates `i % 2` and codes `f(2)`; both forms reference the immediately
preceding reconstruction. `enable_short_refresh_frame_flags`
(`--video-short-refresh`) is 1 + `CeilLog2(1)` = 1 bit too, so it adds nothing
here (it is rejected with the compact flag because both own the field).

**`order_hint` (4 bits/frame) is not derivable.** It is the frame's position in
the (wrapping) hint domain; 240 bits = 30 B, 72% of the remaining reference
cost. It is also the one field a decoder uses to derive relative distance, so
the width is a semantic choice, not redundancy.

## Forms measured and rejected

| Form | Cost | Verdict |
|---|---:|---|
| `frame_refs_short_signaling` (AV1) | — | Not in AV2 v1.0.0 §5.18.2; nothing to cut |
| per-frame `ref_order_hint[]` (AV1) | 0 | Already derived from the slot state in AV2 |
| explicit map with `NumTotalRefs = 1` and one index | 5 bits/P | The baseline; `num_total_refs` is 3 bits even for 1, so the derived form wins |
| `enable_short_refresh_frame_flags` at `NumRefFrames = 1` | 1 bit/P | Same as `f(1)`; no gain (would save 4 bits at `NumRefFrames = 8`, unused by this chain) |
| `OrderHintBits` 4 → 2 | 2 bits/frame | **Rejected.** Measured 24,803 B (21 B more) and 60/60 pictures sample-identical, but §5.18.3.1 `get_relative_dist(a,b) = Clip3(-127,127, a-b)` has **no modulo**, so at every wrap (hint 0 against a hint-3 reference) the decoder reads distance **−3** and §7.8 classifies the only reference as a *future* reference (`check_backward()` feeds the skip/skip-mode CDF context). It is exact only because the gated carrier has no consumer of `FrameDistance` (no compound, TIP, temporal MVs or explicit BAWP). Not a reference-signalling redundancy; do not ship |
| `--video-implicit-ref-map` without `NumRefFrames = 1` | removes 5 bits, changes the list | The §7.7 list grows to the whole live pool (2 slots here), so decisions change; see `docs/negative-results.md` |

## Implementation

```text
VideoConfig::num_ref_frames = 1                    (was resolved_num_ref_frames: 2)
VideoConfig::explicit_ref_frame_map = 0            (was 1)
VideoFrameHeader::signal_primary_ref_frame = 0     (was 1 under --cdf-continuity)
VideoFrameHeader::frame_explicit_ref_frame_map =0  (was 1)
refresh_frame_flags = 1 << 0                       (f(1))
```

Everything else is unchanged: one tile, `order_hint_bits = 4`, the per-frame QP
plan, the carried CDF chain, the drift refresh and every tile-side decision.

Gating in `validate_video_options()`: `--video-refs 1`, the plain key+P chain
(no B/hierarchical/tip/long-term/leading/SEF/switch/RAS/morph), one tile, no
`--video-implicit-ref-map`, no `--video-short-refresh`, no
`--video-num-ref-frames`, no MFH/timing/multi-sequence, 8-bit 4:2:0. The GPU
backend rejects it by name (CPU-only), as do `--video-refs 2`, `--tiles`,
`--video-num-ref-frames` and `--video --still`.

## Verification

`tools/gate_video_ref_signalling.sh --bin build-t/av2enc`:

* flag off: default 25,009 B `a1d0dba1…`, quality 25,141 B `3984677b…`,
  nightly 24,885 B `5e316100…` — byte-identical;
* probe: exactly 948 reference bits baseline and 417 compact, with the
  per-field table above;
* 60-frame stream: **24,824 B md5 `d4c151b2…`**, `dav2d --strict 1` decodes
  **60/60** pictures (count, not exit status), the encoder `--video-recon`
  equals the decoded payload, and the decoded payload equals the flag-off
  decode — byte-identical output, so no CVVDP is needed;
* 2-frame and 6-frame prefixes: 18,993 → 18,992 B and 19,183 → 19,178 B,
  2/2 and 6/6 pictures, identical decoded output and identical tile bytes;
* the non-nightly composition (plain `--video` and `--video-rd-preset quality`
  on 640x360) decodes 4/4 pictures sample-exactly in both forms;
* rejections: `--video-refs 2`, `--tiles`, `--video-short-refresh`,
  `--video-num-ref-frames`, `--video-gpu`, `--video --still`.

At 24,885 B the cut is small (0.245%) because the reference syntax is only
2.0 B/frame of the ~27 B/frame OBU; the remaining 0.87 B/frame is the
`order_hint` (4 bits), the primary-ref signal bit and the CDF-init bit. The
change is worth keeping for the entropy of the header path and for streams
where `NumRefFrames` is larger, but it does not move the nightly operating
point on its own.
