# Oracle check: kept P-tail stack

Status: diagnostic, 2026-11-19. The kept stack is the two default-off P-tail
patches in main plus:

```sh
AV2ENC_SKIP_DRIFT=1
AV2ENC_SKIP_DRIFT_FADE=1
AV2ENC_SKIP_DRIFT_ILLUM=1
AV2ENC_SKIP_DRIFT_REFDEF=1
```

Ladder: QP 112/114/116/118, 240p, 4:2:0, AVM oracle. Matched to 100.0 kbps by
interpolation between QP118 and QP116: **84.508 K**. The single QP116 point is
84.535 at 100.264 kbps.

## Segment ceilings (source replacement, pooling only)

| Segment | Stack QP116 | Old nightly | Old eff-perc |
|---|---:|---:|---:|
| key f0 | 84.648 | 82.239 | 85.305 |
| f0-32 | **89.603** | 85.855 | **90.197** |
| f33-43 | 86.539 | 84.367 | 87.299 |
| f44-59 | **88.654** | 88.066 | 88.995 |

The stack raises the nightly segment ceiling on f0-32 (85.86 → 89.60) and
f44-59 (88.07 → 88.65), but it remains below the eff-perc ceilings. A
segment/replacement ceiling is a pooling bound, not a decodable stream and not
a matched-rate oracle.

## Verdict

- Matched-rate K: **84.508**, below SVT 88.8 and eff-perc 85.10.
- `tools/oracle.py` verdict: **NO-GO (<88.8)**.
- The kept stack is a **realization gain**, not a higher matched-rate oracle.
  The source-replacement bound moved; the encodable point did not.
- Do not restart blind matched-K lanes. A kept patch must report a matched-rate
  envelope and its segment ceilings; a segment ceiling alone opens no path.

The 5-lane batch was stopped at this check. Lane 4's kept exact-rate patch
(84.414) is below the stack and was not merged.
