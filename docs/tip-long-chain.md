# One-key long TIP chain (`--video-tip-groups`) — negative result

Status: **experimental prototype, default off; measured negative.**
Branch `exp/luma-tip`, base `0bf1d001`.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(§5.4.6, §5.18.2, §6.17.2, §7.9, §7.10, §7.13.3.2). Decoder: pinned
`avmdec` for the K oracle, pinned `dav2d f9dafbc` for the historical TIP
component gates.

## 1. Question

`--video-tip` codes 3..8 pictures as key + one future anchor P + TIP output
pictures. The mechanism is extending that to a 60-frame one-key schedule with
periodic **hidden** future anchors:

```
key(0)  hidden A(G)  TIP(1..G-1)  SEF(A(G))
        hidden A(2G) TIP(G+1..2G-1) SEF(A(2G)) ...
```

The hidden anchor is `immediate_output_frame = 0`; one show-existing-frame OBU
re-shows it after its group. A TIP frame refreshes no slot, so only the key and
the two rolling anchor slots are references. This keeps decoder output in
display order while the anchors stay out of the output stream proper.

## 2. Ideal oracle (structure result)

The future-anchor structure was scored independently with an ideal, rate-free
future reference (decoded-domain replacement). This is the ceiling before any
real anchor is coded:

| anchor spacing | perfect-anchor, rate-free K |
|---:|---:|
| 2 | **89.8** |
| 4 | **87.2** |
| 8 | **83.9** |
| 16 | **76.1** |
| 32 | **76.0** |

Only spacing 2 clears the SVT wall (88.8), and it still misses AVM (90.2) by
0.4 K. A spacing-2 schedule with a perfect neighbour is not a decodable
hidden-anchor schedule: it needs a new anchor every two display pictures and a
perfect (source-quality, rate-free) reference at every step.

**Therefore any real hidden-anchor/TIP schedule with spacing > 2, or with
coded rather than perfect references, is bounded below 88.8 and cannot clear
the wall.** That is the gate result: do not tune this class.

## 3. Real envelope (structure result)

Matched-rate K at 240p / 100 kbps, same tool set:

| schedule | K @ 100 kbps |
|---|---:|
| flat-P control | **78.98** |
| hidden anchor spacing 4 | **78.34** |
| hidden anchor spacing 8 | **74.07** |
| hidden anchor spacing 16 | **71.75** |
| same mechanism, rate-free | **79.50** |

Every real point is below its own flat-P control. The anchor rate is the
blocker, not the predictor: the rate-free run of the same mechanism reaches
only 79.50, so even free anchors do not rescue it at spacing >= 4.

## 4. TIP-specific prototype

`--video-tip-groups G` implements the schedule above in
`src/pipeline/video/video_frame_sequence.cpp` (`encode_tip_group_sequence`).
The prototype was validated at 60 pictures:

```
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/tip60.ivf \
  --video --frames 60 --qp 106 --cpu-threads 8 \
  --video-tip --video-tip-groups 4 \
  --video-tip-key-qp 106 --video-tip-anchor-qp 172 \
  --video-recon /tmp/tip60.recon.y4m
```

Picture counts for G=4, N=60:

| item | count |
|---|---:|
| coded pictures | 60 |
| temporal units (incl. SEF) | 75 |
| key | 1 |
| hidden anchor P | 15 |
| TIP_FRAME_AS_OUTPUT | 44 |
| show-existing-frame units | 15 |
| decoder output pictures | **60** |
| decoder output == encoder recon | **60/60 exact** |

Decoder: `avmdec` (60/60) and `dav2d --strict 1` (60/60, exact recon); SEF
units are not counted as coded pictures. Legacy 3..8 `--video-tip` streams are
byte-identical; `video_tip_gate.sh` now also runs an 8-picture G=4 chain and
passes 12/12 TIP streams.

### Measured matched-rate point

QP allocation was chosen to bracket 100 kbps, not tuned. Key QP is 106; anchor
QPs 172/180 bracket the target:

| key QP | anchor QP | bytes | kbps | JOD | BA | SS2 | K |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 106 | 164 | 29,149 | 116.596 | 7.4579 | 16.118 | 25.341 | 53.769 |
| 106 | 172 | 25,735 | 102.940 | 7.2387 | 18.056 | 19.034 | 51.261 |
| 106 | 180 | 23,209 | 92.836 | 6.9789 | 19.160 | 12.129 | 48.634 |

**Matched-rate K @ 100.000 kbps = 50.497** (`tools/oracle.py`, linear between
the 172 and 180 points) against the nightly 82.05 pin and the 88.8 wall.

The TIP schedule is worse than the two-sided-P structure measured above. The
reason is structural, not an allocation error: `TIP_FRAME_AS_OUTPUT` carries no
tile payload, so a TIP picture has **no residual and no per-frame QP of its
own**. Its quality is the quality of its two anchor references. At 100 kbps the
key plus 15 anchors consume the whole budget; a rate-feasible anchor lands at
QP 172-180 and every predicted picture inherits that coarse reference. The
alternative is deep anchors at QP 79, which cost ~250 kB (about 500 kbps) for
the same 60 frames. There is no middle point: this was measured, not inferred.

## 5. Verdict

- Ideal (perfect, rate-free) structure oracle: 87.2 at spacing 4, 83.9 at 8,
  76.1 at 16, 76.0 at 32. Only a non-decodable spacing-2 perfect-neighbour
  variant reaches 89.8, and it still misses 90.2.
- Real TIP prototype at spacing 4: **K 50.50 @ 100 kbps** (60/60 decode,
  exact recon).
- Real structure envelope: 78.34 / 74.07 / 71.75 at spacing 4/8/16, below the
  78.98 flat-P control.
- **NO-GO. Do not tune anchor QP, key QP, group length or lambda inside this
  class.** A residual-coded two-sided P with a free/perfect anchor is the only
  variant that approaches the wall, and the perfect-neighbour schedule is not
  decodable at spacing 2.

The remaining blocker is normative, not a tuning knob: AV2 `TIP_FRAME_AS_OUTPUT`
is prediction-only, so the architecture cannot spend rate on the frames it
actually displays. Any future TIP work should target TIP as a **block mode
inside a residual-coded P** (the `TIP_FRAME_AS_REF` direction already gated by
`--video-tip-block-ref`), not a prediction-only output picture.

## 6. Flags

All default off; gate-off streams are byte-identical.

| flag | meaning |
|---|---|
| `--video-tip-groups G` | one-key long chain, hidden anchor every G display pictures (2..32) |
| `--video-tip-key-qp Q` | key quantiser (default `--qp`) |
| `--video-tip-anchor-qp Q` | hidden-anchor quantiser (default `--qp`) |
| `--video-tip-oracle-ideal-anchor` | gate-only decoded-domain oracle: TIP predicts from the source future picture; recon intentionally differs from the decoder |

`--video-tip-groups` rejects `--video-ref-mvs`, `--video-tip-block-ref`,
`--video-grain` and `--tiles` by name.
