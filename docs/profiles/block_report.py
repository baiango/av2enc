#!/usr/bin/env python3
"""Block-level diagnostics for one encoded still (720p or any size).

Joins three sources and reports where the bits and the error go:

  --blocks   AV2ENC_BLOCKDUMP output:
             mi_row mi_col bsize partition is_inter mode ref drl mvx mvy
             skip eob0 eob1 eob2 levels_digest
  --leafbits AV2ENC_LEAFBITS output:
             mi_row mi_col w h is_inter skip eob0 bits
  --src / --dec  reference and decoded luma (y4m or raw I420)

Reports the intra-mode distribution, and error + emitted bits per
source-activity band (8x8 variance) for each decoded arm.

Usage:
  block_report.py --src src.y4m --dec ours.y4m --dec avm.yuv \
      --blocks bd.txt --leafbits lb.txt [--width 1280] [--height 720]
"""
import argparse
import collections
import os

MODE_NAMES = {
    0: 'DC', 1: 'V', 2: 'H', 3: 'D45', 4: 'D135', 5: 'D113', 6: 'D157',
    7: 'D203', 8: 'D67', 9: 'SMOOTH', 10: 'SMOOTH_V', 11: 'SMOOTH_H',
    12: 'PAETH',
}


def read_luma(path, width, height, frame=0):
    with open(path, 'rb') as f:
        data = f.read()
    frame_bytes = width * height * 3 // 2
    start = data.index(b'\n') + 1 if data.startswith(b'YUV4MPEG2') else 0
    off = start + frame * frame_bytes
    chunk = data[off:off + width * height]
    if len(chunk) != width * height:
        raise SystemExit(f'{path}: expected {width * height} luma bytes at '
                         f'offset {off}, got {len(chunk)}')
    return chunk


def variance(plane, width, height, bx, by, n):
    if bx + n > width or by + n > height:
        return -1.0
    vals = [plane[(by + y) * width + bx + x] for y in range(n) for x in range(n)]
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def band_of(v):
    if v < 0:
        return 'edge'
    if v < 50:
        return 'flat'
    if v < 500:
        return 'mid'
    return 'texture'


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dec', action='append', default=[],
                    help='decoded luma; repeat to compare arms')
    ap.add_argument('--blocks')
    ap.add_argument('--leafbits')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--block', type=int, default=8)
    return ap.parse_args()


def main():
    a = parse_args()
    W, H, N = a.width, a.height, a.block
    src = read_luma(a.src, W, H, a.frame)
    decs = {os.path.basename(p): read_luma(p, W, H, a.frame) for p in a.dec}

    # 8x8 activity grid and per-arm error.
    act, err = {}, {name: {} for name in decs}
    for by in range(0, H, N):
        for bx in range(0, W, N):
            v = variance(src, W, H, bx, by, N)
            act[(bx, by)] = band_of(v)
            if v < 0:
                continue
            for name, plane in decs.items():
                s = 0
                for y in range(by, by + N):
                    r = y * W + bx
                    for x in range(N):
                        e = src[r + x] - plane[r + x]
                        s += e * e
                err[name][(bx, by)] = s / (N * N)

    # Intra mode histogram (key picture: every leaf is intra).
    leaves = []
    if a.blocks:
        for line in open(a.blocks):
            f = line.split()
            if len(f) >= 15:
                leaves.append({'mi_row': int(f[0]), 'mi_col': int(f[1]),
                               'is_inter': int(f[4]), 'mode': int(f[5])})
    if leaves:
        intra = [lf for lf in leaves if not lf['is_inter']]
        hist = collections.Counter(MODE_NAMES.get(lf['mode'], str(lf['mode']))
                                   for lf in intra)
        tot = sum(hist.values())
        print(f'intra leaves: {tot} of {len(leaves)}')
        for name, n in hist.most_common():
            print(f'  {name:10} {n:6}  {100.0 * n / tot:5.1f}%')

    # Emitted bits per leaf, spread over the 8x8 blocks the leaf covers.
    spend = {}
    if a.leafbits:
        for line in open(a.leafbits):
            f = line.split()
            if len(f) < 8:
                continue
            r, c, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            b = float(f[7])
            nbx, nby = max(1, w // N), max(1, h // N)
            share = b / (nbx * nby)
            for j in range(nby):
                for i in range(nbx):
                    k = (c * 4 + i * N, r * 4 + j * N)
                    spend[k] = spend.get(k, 0.0) + share
        print(f'\nleaf bits: {sum(spend.values()) / 8:,.0f} B over '
              f'{len(spend)} blocks')
        agg = collections.defaultdict(lambda: [0.0, 0])
        for k, b in spend.items():
            band = band_of(variance(src, W, H, k[0], k[1], N))
            if band == 'edge':
                continue
            agg[band][0] += b / 8
            agg[band][1] += 1
        print(f"  {'band':10} {'blocks':>7} {'B/blk':>8} {'share':>7}")
        tot = sum(v[0] for v in agg.values())
        for band in ('flat', 'mid', 'texture'):
            b, n = agg[band]
            print(f'  {band:10} {n:7} {b / n:8.2f} {100.0 * b / tot:6.1f}%')

    if decs:
        print(f"\nerror by band (mean 8x8 MSE):")
        print(f"  {'arm':16} " + ' '.join(f'{b:>9}' for b in
                                          ('flat', 'mid', 'texture')))
        for name in decs:
            row = []
            for band in ('flat', 'mid', 'texture'):
                vals = [err[name][k] for k, bd in act.items()
                        if bd == band and k in err[name]]
                row.append(sum(vals) / len(vals) if vals else float('nan'))
            print(f'  {name:16} ' + ' '.join(f'{v:9.2f}' for v in row))


main()
