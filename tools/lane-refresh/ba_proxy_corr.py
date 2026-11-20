#!/usr/bin/env python3
"""Per-frame BA vs the first-party masked error (AV2ENC_PIPE_BA shape).

The perceptual refresh selector is the only measured +1 K lever in the tree,
but its metric was a third-party port.  This script asks whether the
first-party masked error the RD already uses ranks frames the way Butteraugli
does.  If it does, the existing two-pass planner can be driven by it.

  tools/lane-refresh/ba_proxy_corr.py \
      --ref testdata/nasa_480x240_30fps_60f.y4m \
      --test /tmp/base.dec.y4m --json /tmp/ba_corr.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import extra_metrics  # noqa: E402


def read_y4m(path: Path, limit: int = 0):
    frames = []
    with path.open("rb") as f:
        header = f.readline()
        if not header.startswith(b"YUV4MPEG2"):
            raise SystemExit(f"{path}: not a Y4M file")
        w = h = 0
        for tok in header.split():
            if tok.startswith(b"W"):
                w = int(tok[1:])
            elif tok.startswith(b"H"):
                h = int(tok[1:])
        while not limit or len(frames) < limit:
            line = f.readline()
            if not line or not line.startswith(b"FRAME"):
                break
            y = np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w)
            f.read(w * h // 2)
            frames.append(y.copy())
    return w, h, frames


def masked_error(src: np.ndarray, rec: np.ndarray) -> dict:
    y = src.astype(np.float64)
    r = rec.astype(np.float64)
    e = y - r

    # Frame mask, mirroring video_pipe_ba.h build_frame_mask() at weight 1.
    m1 = np.rint(uniform_filter(y, 3, mode="nearest"))
    m3 = np.rint(uniform_filter(y, 7, mode="nearest"))
    m7 = np.rint(uniform_filter(y, 15, mode="nearest"))
    high = np.abs(y - m1)
    mid = np.abs(m1 - m3)
    low = np.abs(m3 - m7)
    act = np.maximum(np.floor((high + 2 * mid + 2 * low + 2) / 5) - 2, 0)
    act = np.rint(uniform_filter(act, 3, mode="nearest"))
    mask = np.rint(256.0 * 24.0 / (24.0 + act))
    mask = np.clip(mask, 160, 256)
    mean = mask.mean()
    if mean > 0:
        mask = np.clip(np.rint(mask * (256.0 / mean)), 64, 512)

    # 3x3 mean of the error, count-limited at the borders (weighted_sse()).
    pad = np.pad(e, 1, mode="constant")
    acc = np.zeros_like(e)
    cnt = np.zeros_like(e)
    for dy in range(3):
        for dx in range(3):
            acc += pad[dy:dy + e.shape[0], dx:dx + e.shape[1]]
            valid = np.ones_like(e)
            if dy == 0:
                valid[0, :] = 0
            if dy == 2:
                valid[-1, :] = 0
            if dx == 0:
                valid[:, 0] = 0
            if dx == 2:
                valid[:, -1] = 0
            cnt += valid
    low3 = acc / cnt

    w = mask / 256.0
    return {
        "masked": float(np.sum(w * (e * e + 1.5 * low3 * low3))),
        "eff": float(np.sum(e * e + 1.5 * low3 * low3)),
        "sse": float(np.sum(e * e)),
        "mask_mean": float(mask.mean()),
    }


def rank(x):
    order = np.argsort(x)
    out = np.empty_like(order, dtype=np.float64)
    out[order] = np.arange(len(x), dtype=np.float64)
    return out


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3:
        return float("nan"), float("nan")
    pear = float(np.corrcoef(a, b)[0, 1])
    spear = float(np.corrcoef(rank(a), rank(b))[0, 1])
    return pear, spear


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    _, _, src = read_y4m(args.ref)
    _, _, rec = read_y4m(args.test)
    n = min(len(src), len(rec))
    if n == 0:
        raise SystemExit("no frames")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "ref").mkdir()
        (tmp / "test").mkdir()
        extra_metrics.extract(args.ref, tmp / "ref", n)
        extra_metrics.extract(args.test, tmp / "test", n)
        ba, ss2 = [], []
        for i in range(1, n + 1):
            b, s = extra_metrics.score_pair(tmp / "ref" / f"f{i:04d}.png",
                                            tmp / "test" / f"f{i:04d}.png")
            ba.append(b)
            ss2.append(s)

    rows = []
    for i in range(n):
        row = {"frame": i, "ba": ba[i], "ss2": ss2[i]}
        row.update(masked_error(src[i], rec[i]))
        rows.append(row)

    metrics = ("masked", "eff", "sse")
    segments = {
        "all": list(range(0, n)),
        "head 1-32": list(range(1, min(33, n))),
        "tail 33-59": list(range(33, min(60, n))),
        "late 44-59": list(range(44, min(60, n))),
    }
    report = {"frames": n, "segments": {}}
    for name, idx in segments.items():
        if len(idx) < 3:
            continue
        seg = {k: corr([rows[i][k] for i in idx], [rows[i]["ba"] for i in idx])
               for k in metrics}
        report["segments"][name] = {
            "n": len(idx),
            **{k: {"pearson": v[0], "spearman": v[1]}
               for k, v in seg.items()},
        }

    # Selector agreement: the frames the planner would pick are the largest
    # predicted error.  Compare the top-N sets.
    late = list(range(min(33, n), n))
    top = args.top
    ba_order = sorted(late, key=lambda i: rows[i]["ba"], reverse=True)[:top]
    for k in metrics:
        order = sorted(late, key=lambda i: rows[i][k], reverse=True)[:top]
        report.setdefault("top", {})[k] = {
            "overlap": len(set(order) & set(ba_order)),
            "of": len(ba_order),
        }

    if args.json:
        args.json.write_text(json.dumps({"report": report, "rows": rows},
                                        indent=2) + "\n")

    print(f"frames={n}")
    for name, seg in report["segments"].items():
        parts = "  ".join(
            f"{k} r={v['pearson']:+.3f} rho={v['spearman']:+.3f}"
            for k, v in seg.items() if isinstance(v, dict))
        print(f"{name:11s} {parts}")
    if "top" in report:
        print(f"top-{args.top} overlap on tail: " +
              "  ".join(f"{k}={v['overlap']}/{v['of']}"
                        for k, v in report["top"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
