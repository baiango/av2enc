# M1 reconstruction verifier

Bit-exact check that the M1 encoder's own reconstruction (`StillResult.recon_y/u/v`)
matches what the pinned `dav2d` decoder produces from the IVF written by the
CLI. This exists because PSNR is not a correctness gate: a stream can sit at
50+ dB while one block has already desynced and the rest of the frame is only
being held together by the error mask.

The check is **exact**: any plane, any sample, any `±1` difference is a failure.

```
Y4M ──av2enc CLI──> IVF ──dav2d──> decoded Y/U/V ─┐
 │                                                ├─ compare byte-for-byte
 └──still_dump──> StillResult.recon_y/u/v ──────────────┘
```

## Files

| File | Purpose |
|---|---|
| `still_dump.cpp` | Encodes the first Y4M frame with `encode_still_picture` and writes the reconstruction + resolved `StillOptions` + visible `AV2ENC_STILL_*` env. |
| `verify.py` | Decodes the IVF with dav2d, compares all planes, prints per-plane counts/max, the first differing 4x4/8x8 block, superblock map and delta histogram. Exit 0 only on exactness. |
| `run_matrix.sh` | Builds the harness and runs resolutions × modes, printing a compact table. `--remote` drives the 3060 box. |

## Build and run (remote 3060 box)

The harness links the same static core library as the CLI and needs `cudart`:

```sh
ssh -p <port> <user>@<host>
cd /workspace/av2enc

g++ -O2 -std=c++20 -Isrc \
    tools/still_verify/still_dump.cpp \
    -o build-cuda/still_dump \
    build-cuda/libav2enc_core.a \
    -L/usr/local/cuda/lib64 -lcudart
```

```sh
# one frame -> IVF with the real CLI
build-cuda/av2enc src.y4m -o out.ivf --still --still-full-intra

# same encode through the harness -> encoder reconstruction
build-cuda/still_dump -i src.y4m -o out.dump --full-intra

# decode + exact compare
python3 tools/still_verify/verify.py --ivf out.ivf --dump out.dump
```

`run_matrix.sh` does all of the above (harness build included):

```sh
# from a laptop, sync tools/still_verify and run on the box
tools/still_verify/run_matrix.sh --remote

# directly on the box
tools/still_verify/run_matrix.sh

# subset, with an explicit depth and no deblock
tools/still_verify/run_matrix.sh --sizes "640x360" --modes "dc force:0 full" \
    --depth 4 --no-deblock

# luma-only bisect: force chroma DC, forward the env to the remote run
AV2ENC_STILL_UV_DC=1 tools/still_verify/run_matrix.sh --remote
```

Variables set in the caller (`AV2ENC_STILL_*`, `DAV2D`) are forwarded over ssh;
values already exported on the box are inherited as usual.

`--rebuild` re-runs `cmake --build build-cuda --target av2enc` first (serialized
with `flock`); without it the script warns when `src/` is newer than
`libav2enc_core.a`. The harness **must** be linked against the same library
that produced the IVF — rebuilding only one of the two compares two different
encoders.

## Options (must match the CLI)

`still_dump` defaults are exactly `av2enc::still::StillOptions{}`. The mapping:

| CLI | `still_dump` | `StillOptions` field | default |
|---|---|---|---|
| `--still-depth N` | `--depth N` | `max_partition_depth` | 4 |
| `--no-deblock` | `--no-deblock` | `enable_deblock` | true |
| `--still-full-intra` | `--full-intra` | `luma_dc_only = false` | true (DC only) |
| `--no-cfl` | `--no-cfl` | `enable_cfl` | true |
| `--still-single` | `--still-single` | `force_single_block` | false |
| `--still-dc` | `--still-dc` | `force_dc` | false |
| `--qp N` | `--qp N` | `base_q_idx` | 90 (M1 pins 90) |
| `--adaptive-cdf` | `--adaptive-cdf` | `adaptive_cdf` | false |
| — | `--min-block-size N` | `min_block_size` | 8 |
| — | `--no-edge-filter` | `enable_intra_edge_filter` | true |

Environment knobs are read by the library inside `encode_still_picture`, so
they only need to be exported for both commands:

```sh
AV2ENC_STILL_FORCE_Y=3:0 build-cuda/av2enc ... --still --still-full-intra
AV2ENC_STILL_FORCE_Y=3:0 build-cuda/still_dump ... --full-intra
```

`still_dump` records the visible knobs (`FORCE_Y`, `FORCE_Y2`, `BASE_ONLY`,
`NO_DIR`, `NO_EDGE`, `UV_DC`, `LUMA_DC`, `TRACE`, the debug `*_CTX*` knobs) in
the dump header and `verify.py` prints them, so a stale `export` is visible in
the report.

### ⚠️ Depth mismatch produces fake transform errors

A previous investigation set `--still-depth 0` on the CLI but left the harness at
the default depth 4. The harness then reconstructed a different partition tree
and `verify.py` reported `±1` differences clustered around transform edges —
which looked exactly like a transform rounding bug but was a harness bug. Keep
the options in sync, and use the guard flags when scripting:

```sh
verify.py --ivf out.ivf --dump out.dump \
    --expect-depth 4 --expect-deblock 1 --expect-full-intra 0
```

These fail with exit 2 if the dump's recorded options do not match.

## Output

```
dump:    640x360  qp=90 depth=4(default) deblock=1 cfl=1 full_intra=0 single=0 dc=0  tile=19203B frame=19227B
env:     UV_DC=1
ivf:     out.ivf
decoder: /workspace/av2enc/tools/bin/dav2d
  plane Y: exact (230400 samples, max|diff|=0)
  plane U: exact (57600 samples, max|diff|=0)
  plane V: exact (57600 samples, max|diff|=0)
RESULT:  BIT-EXACT (0 differing samples)
```

Mismatch (real output: `AV2ENC_STILL_FORCE_Y=3:0 AV2ENC_STILL_UV_DC=1`, 640x360,
one of the directional modes that desyncs on the current tree):

```
dump:    640x360  qp=90 depth=4(default) deblock=1 cfl=1 full_intra=1 single=0 dc=0  tile=25356B frame=25380B
env:     FORCE_Y=3:0 UV_DC=1
  plane Y: 206542/230400 differ (89.645%), max|diff|=91
    first sample:  (250, 0) enc=212 dec=213 d=-1
    diff bbox:     x 0..639, y 0..359
    first 4x4:     block (62, 0) at (248, 0) max|diff|=12
    first 8x8:     block (31, 0) at (248, 0) max|diff|=30
    64x64 superblocks with diffs: 56 (top 3)
      sb (4, 0): 4096 diffs, max|diff|=19
      sb (5, 0): 4096 diffs, max|diff|=17
      sb (6, 0): 4096 diffs, max|diff|=17
    deltas:        -17:54421, -5:24347, +49:20577, +81:10245, ...
  plane U: exact (57600 samples, max|diff|=0)
  plane V: exact (57600 samples, max|diff|=0)
RESULT:  MISMATCH 206542 differing samples (Y 206542, U 0, V 0), max|diff|=91
```

Coordinates printed without a suffix are in that plane's sample grid. Chroma
lines also print the luma-grid coordinate (`[luma x,y]`), because block
decisions are made on the luma grid. The first block is the raster-first block
containing a differing sample; its `max|diff|` covers the whole 4x4/8x8 block
(including samples that agree). The 64x64 map is the fastest way to see whether
a desync starts at a superblock/tile boundary, and the delta histogram
distinguishes a rounding bug (`±1`) from a wrong mode or wrong neighbour
(large, signed deltas).

Machine-readable one-liner (`--tsv`):

```
STATUS  Y_diffs  U_diffs  V_diffs  max|diff|  first_x  first_y  first_plane
```

`first_x/first_y` are luma coordinates of the first differing sample, or `-1`
when exact.

Exit codes: `0` bit-exact, `1` differing samples, `2` setup error (missing
files, decoder failure, dimension/option mismatch).

## Matrix

`run_matrix.sh` generates 64x64 and 640x360 with `ffmpeg` `testsrc2` and
1920x1080 by Lanczos-upscaling 640x360, then runs:

* `dc` — `--still`
* `force:0` … `force:12` — `--still --still-full-intra` with `AV2ENC_STILL_FORCE_Y=N:0`
* `full` — `--still --still-full-intra`

and prints one row per case:

```
size       mode       bytes      Y      U      V  max first(4x4)  result     time
------------------------------------------------------------------------------------------
64x64      dc          2174      0      0      0    0 -           EXACT      0s
64x64      F0          2174      0      0      0    0 -           EXACT      0s
...
640x360    dc         19271      0      0      0    0 -           EXACT      0s
640x360    F3         25424 206542      0      0   91 62,0        DIFF(Y)    0s
640x360    full       19258  90235      0      0   83 95,0        DIFF(Y)    0s
```

(the excerpt above is `AV2ENC_STILL_UV_DC=1`, so chroma is exact and only luma
mode correctness is on display; `bytes` is the IVF file size). `first(4x4)` is
the 4x4 block index of the raster-first differing luma sample (multiply by 4
for the sample coordinate); `-` means exact.

Exit 0 only if every row is `EXACT`.

Use `AV2ENC_STILL_UV_DC=1` to isolate luma (forces chroma DC) and
`AV2ENC_STILL_BASE_ONLY=1` / `NO_DIR=1` / `NO_EDGE=1` to bisect a directional or
edge-filter bug. Use `AV2ENC_STILL_TRACE=1` for the library's per-block trace.

## Notes / limitations

* Only the **first frame** is compared. The CLI may encode more; pass a
  single-frame Y4M or ignore later frames.
* `StillResult.leaves` exists in the header but is never assigned by
  `encode_still_picture` (always 0), and `StillResult` exposes no per-block
  partition/mode map. The verifier therefore infers the first bad block from
  pixels; if you need the actual partition node, that information has to be
  added to `StillResult` in encoder code.
* The dump is `"StillDUMP1\n" + one JSON line + Y/U/V raw planes` (tightly packed
  8-bit 4:2:0), so it is trivial to post-process from other tools.
* Observed on the current tree on 2026-11-11 (commit `aeca339`, remote
  `build-cuda`): with `AV2ENC_STILL_UV_DC=1`, DC is bit-exact at 64x64, 640x360
  and 1080p. The default **CfL chroma path desyncs** at every resolution (Y
  exact, U/V wrong; at 640x360 the first U diff is at luma x=320, max|diff|
  114), and luma forced modes 3, 7, 8, 9, 10 plus full search desync. Those are
  encoder bugs for the M1 correctness work, not harness bugs. Re-run the
  matrix after fixes; this note is a snapshot, not a gate.
