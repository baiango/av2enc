# Static-superblock partition collapse (`--video-skip-collapse`)

A frame that changes nothing still has to signal its partition tree. On a
60-frame all-identical clip at the nightly operating point, a P frame whose
leaves all skip costs **32.4 information bits, 32.1 of them in partition
symbols** — a 7-byte tile. `--video-skip-collapse` rewrites a superblock whose
leaves are all plain zero-motion inter skips into one superblock-sized skip
leaf. The reconstruction is unchanged on static content; the tree bits go away.

Opt-in, CPU-only, whole-frame single tile, 8-bit 4:2:0, flat key+P chain. Off by
default, so the default, `quality` and `--nightly` streams and every GPU path
stay byte-identical (pins 25,009 `a1d0dba1…`, 25,141 `3984677b…`, 24,885
`5e316100…`). Gated by `src/pipeline/video/video_skip_collapse_gate.sh`.

## Why a static superblock still pays

One 64×64 superblock of an all-skip P frame codes, per leaf:

```text
read_partition(): do_split = 0           1 symbol per interior superblock
decode_block():   is_inter = 1          1 symbol per leaf
                  skip_flag = 1          1 symbol per leaf
                  single_ref / single_mode / drl_idx     per leaf
                  cdef_idx              absent for a skip leaf with
                                        CdefOnSkipTxfm = 0 (§5.20.10.1)
                  ccso_blk              no unit symbol in a skip-only plan
                  delta_qindex          absent for MiSize == SbSize &&
                                        skip_flag (§5.20.5.11)
```

Every one of those symbols is cheap once the CDFs have adapted — except the
partition symbols, whose cost is *how the tree splits*, not how many leaves it
has. `AV2ENC_SYMBOL_COST=1` prints the per-family table per tile:

| frame | tile | partition | is_inter | skip | mode | info bits |
|---|---:|---:|---:|---:|---:|---:|
| P, flag off (steady) | 7 B | 78 / 32.15 | 49 / 0.067 | 49 / 0.066 | 49 / 0.170 | 32.45 |
| P, flag on (steady) | 3 B | 35 / 7.196 | 33 / 0.045 | 33 / 0.044 | 33 / 0.122 | 7.41 |
| P, flag off (first P) | 22 B | 78 / 50.94 | 49 / 1.21 | 49 / 6.02 | 60 / 95.63 | 153.8 |
| P, flag on (first P) | 16 B | 35 / 16.15 | 33 / 1.03 | 33 / 5.39 | 44 / 87.93 | 110.5 |

(`partition` is `do_split` + `rect_type`; `mode` is `single_ref`, `single_mode`,
`drl_idx` and the BAWP/motion-mode symbols.)

The flag-off tree on that clip is the one `--video-partition-rd` solves: 17
optional splits in rows 2–3 of the picture, each coding `do_split = 1` at 1–5.5
bits, plus the boundary chains. Frames 3–60 code the identical tree because the
probe sees the same keyframe reconstruction error every frame, while every leaf
still decides to skip.

## What the rewrite does

For each superblock of a P frame, after the partition solve and the first
decision walk:

1. **Eligibility.** Every coded leaf of the superblock is `is_inter = 1`,
   `skip_flag = 1`, `ref_index = 0`, `mv = (0,0)`, and uses no mode that changes
   the predictor: no compound, no TIP, no BAWP (or chroma BAWP), no
   inter-intra, no morph, no optflow/refinemv. A GLOBALMV leaf also counts when
   the frame codes no global motion (its MV is then zero). Frames with
   segmentation / delta-Q / per-segment Q are excluded: §5.20.5.11 cannot code a
   delta on a full-SB skip.
2. **Rewrite.** `partition::decide_superblock()` re-solves that superblock with
   `leaf = 0`, `none_signal = 0` and every optional split at 1e12. Spec-forced
   splits (frame boundary) cost nothing and are still taken; anything else
   becomes `PARTITION_NONE`. The node keeps the solver's own
   boundary/allowed/forced metadata, so the writer codes exactly one
   `do_split = 0` for an interior superblock.
3. **Verify.** The frame is re-collected. A rewritten superblock must come back
   as one leaf with `skip_flag = 1`, `ref_index = 0` and `mv = (0,0)`;
   otherwise its original tree is restored and the frame is re-collected
   (three passes at most). This is what keeps the change reconstruction-exact on
   the static clip, and it is the only reason the steady-state tile is 3 B and
   not 2 B: one superblock at (32,0) has a 64×64 NEWMV candidate at 1/8-pel
   motion, so its split tree is kept (5.5 bits).

A superblock that already has `PARTITION_NONE` is left alone, and the keyframe
is never touched.

## What is not there to cut

The same table says the other groups are already at their floor, so the tree is
the only thing worth attacking:

* **CDEF.** §5.20.10.1 `read_cdef()` returns before `cdef_idx` for an inter
  skip block when `CdefOnSkipTxfm` is disabled, which is the product default.
  The meter records `cdef_index` 0 symbols on every all-skip P frame.
* **CCSO.** A skip-only P frame plans no CCSO units at all
  (`filters_->ccso.enabled` is false for those frames), so `ccso_blk` codes
  nothing; the keyframe pays 189 symbols / 29 bits for its unit flags, which is
  the filter's own grid, not skip overhead.
* **§5.20.5.11 delta-Q.** `read_delta_qindex()` codes nothing when
  `MiSize == SbSize && skip_flag`, so a full-SB skip already spends zero bits
  there. There is no cheaper skip representation to signal: a skip leaf codes
  `is_inter`, `skip_flag` and its mode regardless, and those are 0.05-0.12 bits
  once the CDFs have adapted.
* **Per-frame framing.** 12 B of IVF container and ~2 B of OBU/frame header per
  picture are outside the tile; the tile itself bottoms out at the 2-byte
  §8.2.4 marker.

## Measured floor

Static clip: `testdata/nasa_480x240_30fps_60f.y4m` frame 0 repeated 60 times,
`--nightly` (QP110, key offset −27, carried CDFs), `--cpu-threads 8`.

| stream | total | key | P mean | P steady | P tile steady |
|---|---:|---:|---:|---:|---:|
| `--nightly` | 20,915 B | 18,901 B | 21.39 B | 21 B | 7 B |
| `--nightly --video-skip-collapse` | **20,695 B** | 18,901 B | **17.66 B** | **17 B** | **3 B** |

Per P frame that is **−3.73 B (−17.4 %)**; at steady state 21 B → 17 B
(**−19 %**). The 17 B is 12 B of IVF framing, 2 B of OBU/frame header and the
3-byte tile.

The irreducible floor for a coded tile is §8.2.4 `exit_symbol()`: a 15-bit
marker plus byte padding, i.e. 2 bytes. `--video-force-single` (one 64×64 leaf
per superblock, keyframe included) reaches it — 2-byte tiles, 0.25 information
bits — and 16 B per P frame. The 1 B between that and the collapse stream is the
one rejected superblock above.

## Real clip

`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, `--nightly`:

| stream | bytes | kbps | PSNR-Y | CVVDP | MD5 |
|---|---:|---:|---:|---:|---|
| pin | 24,885 | 99.540 | 43.9358 | 9.7177 | `5e316100…` |
| `--video-skip-collapse` | 24,539 | 98.156 | 43.9096 | 9.6973 | differs |

The moving clip loses 346 B (−1.39 %) and 0.020 JOD at the same QP: the collapse
itself is reconstruction-exact, but the neighbouring-block *shape* changes, and
§7.12.2 reads block sizes when it builds the merge/DRL stack, so later blocks in
the same frame and later frames can pick different modes. On a static clip that
never happens (below). This is a floor tool, not a quality tool: it is a wash on
moving content, so it stays out of `--nightly`.

## Decode exactness

* Static clip, flag on: `dav2d --strict 1` exits 0 and the visible encoder
  reconstruction equals the decoded payload byte for byte (10,368,000 B).
* Static clip, flag on vs flag off: the two reconstructions are byte-identical
  frame for frame (10,368,000 B). The rewrite is a pure bit saving.
* Real clip, flag on: `dav2d --strict 1` exits 0, recon equals the decoded
  payload (10,368,000 B), 60/60 pictures.

## Instrumentation

`AV2ENC_SYMBOL_COST=1` (tile writer only, no effect on the bitstream) prints one
line per tile:

```text
symbol_cost tile_bytes=3 info_bits=7.40685 partition=35/7.19587 \
  is_inter=33/0.0450615 skip=33/0.0443125 inter_mode=33/0.121604
```

Each entry is `symbol count / information bits`, where the information cost is
`-log2(p)` from the live §8.2 CDF (`cdf[s] - cdf[s-1]` over 32768). `cdef_index`
and `ccso_blk` are measured by the bit-prefix delta across the filter writer
calls, which is why they are whole-bit estimates. `payload_bytes` also carries
the §8.2.4 marker and padding, so `tile_bytes` can exceed `info_bits / 8`.

## Usage

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/static.ivf \
    --video --frames 60 --qp 110 --cpu-threads 8 --nightly \
    --video-skip-collapse
```

Rejected by name with `--video-gpu` (the CUDA backend solves its own tree),
`--tiles` other than 1x1, and the segmentation / delta-Q / per-segment-Q tools.
