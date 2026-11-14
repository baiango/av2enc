# Keyframe intra TX partition (`--video-key-intra-txpart`)

`--video-tx-search` gives the luma residual the §5.18.8 `TX_MODE_SELECT`
syntax (per-leaf TX size, TX type, and §5.20.6.3 partition).  Its intra half is
deliberately single-predictor: [`video_tx_search.h`](../../src/pipeline/video/video_tx_search.h)
takes one prediction for the whole coding block, so an intra leaf keeps
`TX_PARTITION_NONE` and only searches the TX type.  That is correct but leaves a
hole: §5.20.7.24 predicts **per transform block**, and the decoder feeds every
part's reconstruction into the next part's prediction.  One predictor over a
64x16 leaf with one 64x16 transform is a poor model of a block with an edge.

The still encoder already implements the correct chain
(`still_encoder.cpp` `encode_luma_parts()` / `choose_luma_tx_partition()`), and
`src/codec/transform/tx_partition/still_tx_partition_gate.sh` proves it
sample-exact against `dav2d`.  This flag ports that chain into the video
keyframe.

## What it does

For every keyframe intra luma leaf, after the unchanged whole-block mode search
(`--video-intra-full`), it enumerates `TX_PARTITION_NONE` plus the
`txpartition::legal_splits()` SPLIT/HORZ/VERT options that fit inside the coded
plane.  For each candidate part, in decode order:

1. derive §5.20.9.1 availability from the encoder's §6.19.2.3 `BlockDecoded`
   map at the **part** origin, and the §7.13.2.15/.16 smooth-neighbour flags at
   the **coding-block** origin (the decoder reads them there);
2. predict the part with the block's signalled `(y_mode, angle_delta,
   mrl_index, mrl_sec_index)`, using the shared `intra::predict_intra` /
   `ibp::predict` / `mrl::predict_block`;
3. run the TX type/level search for the part alone
   (`search_luma_tx(..., search_partition=false)`, so the single-predictor
   contract is respected);
4. write the part reconstruction into the real luma plane and mark its 4x4
   cells, so the next part predicts from it.

The candidate cost is coded SSE (visible rows only under `--video-edge-mask`)
plus `tx_rd_lambda * (sum of per-part TX rate + 3 bits for the §5.20.6.3
partition)`.  The search snapshots/restores only the block's recon samples and
decoded cells, so losing trials leave no trace; the winner is re-committed.

Chroma is untouched: the sequence header keeps `enable_chroma_dctonly = 1` and
the §5.20.6.1 `get_tx_size(plane > 0, ...)` path ignores the luma partition, so
each chroma plane keeps one DCT.  Inter TX search is untouched.

`NONE` reuses the whole-block prediction the mode search already produced, so
when the partition loses the flag-on stream is byte-identical to the flag-off
stream for that leaf.  DIP-selected leaves stay `TX_PARTITION_NONE` (DIP is a
whole-block alternative prediction).  HORZ4/VERT4 are not enumerated; the port
matches the still encoder's NONE/SPLIT/HORZ/VERT set.

## Gate

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON
cmake --build build -j1
DAV2D=tools/bin/dav2d-macos \
    src/pipeline/video/video_key_intra_txpart_gate.sh --bin build/av2enc
```

The gate rejects the flag without `--video-tx-search` and with `--video-gpu`,
proves intra SPLIT/HORZ/VERT partitions are selected, decodes both streams with
`dav2d --strict 1`, checks the visible encoder recon equals the decoded payload
for every frame, requires the keyframe to get smaller and higher-PSNR, and
re-checks the 60-frame `--nightly` pin (`24,939 B`, MD5
`8f8713dac14eaa74af770e9142e5eae9`).

## Measured RD

Host: Apple M4, Release without LTO, `--cpu-threads 8`.  Clip:
`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames.  `tools/quality.py` (CVVDP
0.5.7, `standard_fhd`, MPS) with `dav2d-macos`.  The quality command is the
documented nightly point plus the flag:

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o out.ivf --video --frames 60 \
    --qp 117 --cpu-threads 8 --nightly --video-key-intra-txpart
```

| QP | Bytes | kbps | CVVDP | PSNR-Y | keyframe B | keyframe PSNR-Y |
|---:|---:|---:|---:|---:|---:|---:|
| 112 | 24,731 | 98.924 | 9.6730 | 42.8777 | 17,186 | 44.53 |
| 114 | 23,971 | 95.884 | 9.6661 | 42.7914 | 16,585 | 44.23 |
| 115 | 23,461 | 93.844 | 9.6448 | 42.5858 | 15,994 | 43.85 |
| 116 | 23,304 | 93.216 | 9.6373 | 42.4644 | 15,880 | 43.70 |
| 117 | 23,074 | 92.296 | 9.6451 | 42.4378 | 15,666 | 43.57 |
| 118 | 22,632 | 90.528 | 9.6375 | 42.2441 | 15,222 | 43.27 |
| 120 | 22,190 | 88.760 | 9.6122 | 42.1272 | 14,866 | 43.04 |

Flag-off nightly at QP 117 (unchanged pin): **24,939 B / 99.756 kbps / 9.5868
JOD / 41.6953 dB**, keyframe **16,429 B / 34.57 dB**.  (Keyframe bytes here are
the first IVF frame record, which is what the gate compares; the frame payload
is 55-56 B smaller.)

At the same QP the keyframe is **-4.6 % bytes and +9.00 dB**, and the whole
stream is **-7.5 % bytes and +0.74 dB** at a *lower* rate.  Bracketing the
baseline 100 kbps point with the documented QP 116/117 rows, the flag-on QP 112
point (98.924 kbps, 9.6730 JOD) is about **+0.065 JOD at iso-rate**; read
against the measured QP 117 baseline it is +0.086 JOD.

Keyframe leaf histogram (QP 117, flag off): 189 luma leaves - 8x64x64,
5x64x32, 15x64x16, 22x32x16, 6x32x32, 14x16x32, 118x16x16, 1x32x64.  With the
flag, 189 leaves pick 60 NONE, 70 SPLIT, 44 HORZ and 15 VERT.

## Decode and recon

The 60-frame flag-on stream decodes with `dav2d --strict 1`, and the
`--video-recon` payload equals the decoded payload byte for byte (60 x
172,800 B of planar samples), i.e. the per-part prediction/recon feedback
matches the decoder's own chain.

## Limits

* CPU CPU-reference path only; the CUDA device TX module keeps the
  single-predictor contract and `--video-gpu` rejects the flag by name.
* 8-bit 4:2:0 only (the format-video core rejects it).
* DIP leaves and blocks that would need a part outside `MiCols`/`MiRows` keep
  `TX_PARTITION_NONE`.
* Not part of the default, `quality` or `--nightly` presets.
