# CUDA integration gap analysis

Snapshot: **2026-11-13**, `main` at `c3708ee` after the second integration
wave. Final merged-tree gates on the RTX 3060: `gate_video_gpu` (4 frames),
`gate_video_filters` and `gate_video_formats` all pass; unit ctest 10/10;
CPU baseline 19 streams / 18 gates / 30 features; `tools/check_arena.sh` PASS.

Spec: AV2 Bitstream & Decoding Process Specification v1.0.0, 1169 pages,
commit `38f28e7`, SHA-256
`e9916f091e4e83446aad6b4601641c5b292e569c144c4163b26a4497573b533f`.

## How to read this

- **Wired** — reachable from `--still-gpu` and/or `--video-gpu`, CPU↔GPU
  byte-identical, `dav2d --strict 1` decodes the expected picture count.
- **Kernel only** — device implementation and parity suite exist, but no
  product path consumes it.
- **Rejected** — a named usage error; never a silent CPU fallback.
- **By design** — syntax, entropy, headers or output processes intentionally
  run on the host.

## 0. Summary

| Class | Count | Meaning |
|---|---:|---|
| Video product integration still missing | 10 | kernel exists, `--video-gpu` does not consume it |
| Still product integration still missing | 5 | kernel exists, `--still-gpu` rejects it |
| No CUDA kernel | 4 areas | encoder search/framing work, not pixel kernels |
| Decoder-gated under pinned `dav2d` | 4 | AVM decodes; `dav2d` does not |
| CPU by design | 6 areas | entropy writer, headers, ref lists, output |

## 1. Wired product surface

### `--video-gpu`

keyframe + P/B frames, 1–7 references over a 4–8 slot `NumRefFrames` pool,
quarter/half/eighth pel, force-single, depth/min-block/ME-range, compound
AVERAGE, wedge and diffwtd masks, B frames, temporal layers + drop,
long-term/OLK/leading/SEF/SWITCH/RAS, MFH, timing + content interpretation,
average/adaptive CDF, per-frame CRF/VBR, multi-tile, deblock, CDEF, CCSO,
Wiener NS, chroma LR, GDF, PC-Wiener, BAWP, global motion + GLOBALMV warp,
full intra search (`--video-intra-full`), TX size/type/partition search
(`--video-tx-search`), CfL, MHCCP, QM default matrices, 10-bit
4:2:0/4:2:2/4:4:4/4:0:0 and 8-bit 4:2:2/4:4:4/4:0:0.

### `--still-gpu`

plan/search/recon/chroma/deblock, CDEF, CCSO, Wiener NS, PC-Wiener, GDF,
chroma LR, QM, DC/AC delta-Q, neutral segmentation / delta-Q, device per-SB
`SEG_LVL_ALT_Q`, IST, TX type, IDTX, CCTX, TCQ, RDOQ, CfL explicit and
downsampling, MHCCP, MRL, DIP, IBP, palette, IntraBC, full-UV chroma,
SB128/SB256 including CDEF/CCSO/LR unit grids, extended partitions, SDP,
sub-8, multi-tile, lossless, 10-bit deblock, adaptive film grain, content
interpretation, and the full 10-bit/chroma still formats.

## 2. Video — kernel exists, product integration missing

| Spec | Tool | State |
|---|---|---|
| §7.10, §7.13.3.2 | TIP | CPU `--video-tip` is dav2d-exact; `--video-gpu` rejected |
| §7.13.3.3–.12 | OPFL / refine-MV | component gate decodes; `--video-opfl` rejected (needs non-GLOBALMV compound) |
| §7.13.3.26 | morphological prediction | component gate decodes; `--video-morph` rejected (needs a video IBC keyframe path) |
| §7.13.3.15 | save sub-PU | kernel + parity; not wired |
| §7.22 | temporal MV / motion-field scan | storage driven by CPU TIP; §7.12.2 temporal scan not in the merge stack |
| §5.13, §5.20.7 | segmentation / delta-Q signalling | per-SB device state exists for still; video tile-side segment-id/delta-q ordering not wired |
| §5.20.7.27, §8.3 | `dc_sign` | kernel exists; mixed 1-D intra types are decoder-rejected |
| §5.18.2 | cross-frame CDF continuity | **decoder-gated**: AVM decodes, dav2d drops frames |
| §7.3.6, §6.8–.10 | SVC / layers / LCR | **decoder-gated**: AVM `--all-layers` decodes; dav2d emits fewer pictures |
| §7.8 | implicit reference lists | CPU path is **decoder-gated** (`dav2d --strict 1`, 6 carriers, recon exact); the GPU path rejects the flag |

## 3. Still — kernel exists, product integration missing

| Spec | Tool | State |
|---|---|---|
| §5.20.6.1–.3 | TX partition | rejected: needs per-part predict→transform→recon chains and 4× coefficient regions |
| §7.13.2 + §7.15 | RDO intra decision | rejected: implementation reverted after a deblocked-recon mismatch at 640×360 |
| §6.12, §9.4 | user-defined QM OBU | **decoder-gated**: AVM decodes; dav2d rejects QM OBUs |
| §6.4.1, §7.18–§7.20 | 10-bit CDEF / CCSO / LR | rejected: cores still clip at 8 bits; 10-bit deblock is wired |
| §5.20.7.27 | mixed 1-D intra TX / `dc_sign` | deferred: pinned dav2d rejects negative H/V_DCT |

## 4. No CUDA kernel

| Spec | Area | Why |
|---|---|---|
| §5.19 | still tile-group framing extensions | the product tile loop is wired; unverified layouts stay rejected |
| §5.4.12 | HRD / two-pass internals | host policy, fed into the wired per-frame QP path |
| §7.3.6, §6.8–.10 | SVC layer framing | host-only composition |
| §8 | range coder / coefficient CDFs | host writer; a byte-exact CUDA coder exists but is not used |

## 5. Decoder policy

`dav2d f9dafbc` is the product gate. AVM is used only where `dav2d` cannot
parse a spec tool. Verified with AVM:

| Stream | dav2d | AVM |
|---|---|---|
| user-defined QM OBU | rejects (`Invalid argument`) | decodes 1/1 |
| cross-frame CDF continuity, 3 frames | frame 1 then per-frame errors (exit 0) | 3/3 |
| SVC 2 layers × 3 TUs | 1 picture | 6/6 with `--all-layers` |

The project does not weaken the gate: a tool `dav2d` cannot decode stays
rejected by name in the product, even when the encoder-side syntax is
spec-conformant.

## 6. Merge history (this wave)

| Branch | Landed |
|---|---|
| `video-intra` + `-wire` | §7.13.2 61-candidate luma search |
| `video-tx` + `-wire` | §5.20.6/§5.20.8 TX search |
| `video-deblock` | §7.17 |
| `video-loopfilters` | §7.18–§7.20 (CDEF/CCSO/Wiener/chroma LR/GDF) |
| `video-refs` | NumRefFrames 4–8, NumTotalRefs 1–7 |
| `video-rc-tiles` | CRF/VBR, multi-tile |
| `video-advanced-inter` | GME + GLOBALMV warp |
| `video-adv-tip` | CPU TIP |
| `video-adv-bawp` | BAWP |
| `video-compound-masks` | wedge + diffwtd |
| `video-codec-tools` | CfL, MHCCP, PC-Wiener, QM |
| `video-10bit` | 10-bit/chroma video |
| `video-consistency` | GME multi-ref, filter/tile, TX-search combinations |
| `still-tx-tools` / `still-tx-final` | IST, TX type, IDTX, CCTX, TCQ, RDOQ |
| `still-intra-tools` | palette, IntraBC, full-UV |
| `still-formats` | 10-bit/chroma still |
| `still-seg-filters` | segmentation, SB128/256 filter grids |
| `still-side-final` | tiles, lossless, 10-bit deblock, metadata |
