# M1 results — partition tree, full intra, transforms, CfL, deblock

Status: **bit-exact against `dav2d` f9dafbc**, verified per sample on all three
planes. M1 is opt-in (`--m1`); M0 remains the default until M1 is measured more
broadly.

## Verified streams (QP 90, one frame, one tile)

| Still | IVF | Y / U / V PSNR | Avg | Verify |
|---|---:|---|---:|---|
| 64×64 | 936 B | 42.50 / 45.80 / 45.12 | **43.28 dB** | BIT-EXACT |
| 640×360 | 13,355 B | 45.09 / 52.45 / 50.53 | **46.37 dB** | BIT-EXACT |
| 1080p (proxy) | 37,856 B | 50.68 / 52.59 / 54.29 | **51.41 dB** | BIT-EXACT |

M0 reference for the same stills:

| Still | M0 IVF | M0 avg PSNR |
|---|---:|---:|
| 64×64 | 1,138 B | 42.13 dB |
| 640×360 | 19,751 B | 47.20 dB |
| 1080p | 83,228 B | 49.79 dB |

M1 is 18% smaller at 64×64 (+1.15 dB), 32% smaller at 640×360 (−0.83 dB), and
54% smaller at 1080p (+1.62 dB).

## Verification method

`tools/m1_verify/`:

1. encode with the CLI,
2. dump `M1Result.recon_y/u/v` with the *same* `M1Options`,
3. decode the IVF with `dav2d`,
4. compare every sample in all three planes.

`BIT-EXACT` means zero differing samples, not a PSNR threshold. A harness/CLI
option mismatch is caught explicitly (`--expect-depth`, `--expect-deblock`,
`--expect-full-intra`).

Coverage: default DC path at 64×64/640×360/1080p; forced luma modes 0–12;
angle deltas ±1 and ±3 for the eight directional modes; full intra search at all
three sizes.

## Tools enabled

- recursive partition tree (8×8–64×64, rect allowed; `PARTITION_SPLIT` only where legal)
- full luma intra (13 modes, angle deltas, edge filtering, IDIF, wide-angle)
- variable TX sizes via `tx_variants`, including adjusted 64→32 sizes
- CfL with derived alpha
- deblocking filter

## Full intra is opt-in

Full directional intra is bit-exact but not a rate/quality win at QP 90 on this
corpus:

| Still | Default (DC) | Full intra |
|---|---|---|
| 64×64 | 936 B / 43.28 dB | 979 B / 43.26 dB |
| 640×360 | 13,355 B / 46.37 dB | 13,199 B / 46.30 dB |
| 1080p | 37,856 B / 51.41 dB | 38,559 B / 51.28 dB |

The mode search is SSE-only; it needs a rate term (RDO) before it can pay for
the mode signalling. Until then `--m1` stays DC-safe and `--m1-full-intra`
enables the search.

## Known limits

- Directional/smooth **chroma** modes are disabled in the default path; chroma
  uses DC or CfL. Re-enabling the full UV search still mismatches.
- Explicit CfL alpha signalling is not wired; derived alpha is used
  (`cfl_index = 1`).
- The 1080p row uses a Lanczos upscale of the 640×360 test frame; it is a proxy,
  not a natural 1080p corpus.
- Target validation still requires an RTX 3060 enumerating 8192 MiB; the 12 GB
  dev box is a proxy.
