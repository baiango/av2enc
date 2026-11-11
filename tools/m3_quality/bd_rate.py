#!/usr/bin/env python3
"""Bjoentegaard delta-rate / delta-PSNR helpers for M3 quality sweeps.

The standard Bjoentegaard metric integrates the difference between two
rate-quality curves over their common quality (or rate) range:

    BD-rate = 10 ** (mean(log10(R_test) - log10(R_anchor))) - 1

A fit needs at least three operating points on both curves and a non-empty
common range.  The current M1/M3 still-picture encoder pins ``base_q_idx = 90``
until the coefficient-CDF qctx work lands, so a real curve cannot be built yet.
This module reports that condition explicitly instead of inventing a number.

Two interpolation modes are available:

* ``pchip``  - monotone piecewise cubic Hermite (default; no overshoot)
* ``cubic``  - the classic cubic-polynomial least-squares fit

Both integrate on a dense grid, so the two modes differ only in interpolation.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

MIN_POINTS = 3


@dataclass(frozen=True)
class RDPoint:
    """One rate-quality operating point."""

    quality: float          # PSNR-Y in dB or CVVDP in JOD
    rate_kbps: float
    label: str = ""


def _clean(points: list[RDPoint]) -> list[RDPoint]:
    """Sort by quality and merge duplicate quality values (average rate)."""
    ordered = sorted(
        (p for p in points if math.isfinite(p.quality) and p.rate_kbps > 0.0),
        key=lambda p: p.quality,
    )
    merged: list[RDPoint] = []
    for point in ordered:
        if merged and abs(point.quality - merged[-1].quality) <= 1e-9:
            previous = merged[-1]
            merged[-1] = RDPoint(
                previous.quality,
                0.5 * (previous.rate_kbps + point.rate_kbps),
                previous.label,
            )
        else:
            merged.append(point)
    return merged


def _unavailable(
    reason: str,
    anchor_n: int,
    test_n: int,
    method: str,
) -> dict:
    return {
        "available": False,
        "method": method,
        "reason": reason,
        "anchor_points": anchor_n,
        "test_points": test_n,
        "min_points": MIN_POINTS,
    }


def _cubic(points: list[RDPoint], xs: np.ndarray) -> np.ndarray:
    x = np.asarray([p.quality for p in points], dtype=float)
    y = np.log10(np.asarray([p.rate_kbps for p in points], dtype=float))
    degree = min(3, len(points) - 1)
    coeffs = np.polyfit(x, y, degree)
    return np.polyval(coeffs, xs)


def _interp_log_rate(points: list[RDPoint], xs: np.ndarray, method: str) -> np.ndarray:
    if method == "cubic":
        return _cubic(points, xs)
    if method != "pchip":
        raise ValueError(f"unknown BD-rate method: {method}")
    from scipy.interpolate import PchipInterpolator

    x = np.asarray([p.quality for p in points], dtype=float)
    y = np.log10(np.asarray([p.rate_kbps for p in points], dtype=float))
    return PchipInterpolator(x, y, extrapolate=False)(xs)


def _common_range(
    anchor: list[RDPoint],
    test: list[RDPoint],
    *,
    log_rate: bool,
    method: str,
) -> tuple[float, float] | None:
    ax = np.log10([p.rate_kbps for p in anchor]) if log_rate else [p.quality for p in anchor]
    tx = np.log10([p.rate_kbps for p in test]) if log_rate else [p.quality for p in test]
    lo = max(float(np.min(ax)), float(np.min(tx)))
    hi = min(float(np.max(ax)), float(np.max(tx)))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return None
    return lo, hi


def bd_rate(
    anchor: list[RDPoint],
    test: list[RDPoint],
    *,
    method: str = "pchip",
    min_points: int = MIN_POINTS,
) -> dict:
    """Return Bjoentegaard delta-rate of ``test`` against ``anchor`` in percent.

    Negative is better (test spends less rate at equal quality).
    """
    a = _clean(anchor)
    t = _clean(test)
    if len(a) < min_points or len(t) < min_points:
        return _unavailable(
            f"need at least {min_points} operating points on each curve; "
            f"anchor has {len(a)}, test has {len(t)}",
            len(a),
            len(t),
            method,
        )
    common = _common_range(a, t, log_rate=False, method=method)
    if common is None:
        return _unavailable(
            "quality ranges do not overlap; cannot integrate a common range",
            len(a),
            len(t),
            method,
        )
    lo, hi = common
    xs = np.linspace(lo, hi, 1001)
    a_log = _interp_log_rate(a, xs, method)
    t_log = _interp_log_rate(t, xs, method)
    if not (np.all(np.isfinite(a_log)) and np.all(np.isfinite(t_log))):
        return _unavailable(
            "interpolation produced non-finite values (check quality ordering)",
            len(a),
            len(t),
            method,
        )
    delta = float(np.trapezoid(t_log - a_log, xs) / (hi - lo))
    return {
        "available": True,
        "method": method,
        "bd_rate_pct": (10.0**delta - 1.0) * 100.0,
        "mean_log10_rate_delta": delta,
        "common_quality_range": [lo, hi],
        "anchor_points": len(a),
        "test_points": len(t),
        "min_points": min_points,
    }


def bd_psnr(
    anchor: list[RDPoint],
    test: list[RDPoint],
    *,
    method: str = "pchip",
    min_points: int = MIN_POINTS,
) -> dict:
    """Return Bjoentegaard delta-PSNR of ``test`` against ``anchor`` in dB.

    Positive is better (test has higher PSNR at equal rate).
    """
    a = _clean(anchor)
    t = _clean(test)
    if len(a) < min_points or len(t) < min_points:
        return _unavailable(
            f"need at least {min_points} operating points on each curve; "
            f"anchor has {len(a)}, test has {len(t)}",
            len(a),
            len(t),
            method,
        )
    # Swap roles: fit quality as a function of log10(rate).  The transformed
    # list must be re-sorted because a monotone rate/quality curve runs in the
    # opposite direction after the swap.
    a_rate = _clean([RDPoint(math.log10(p.rate_kbps), p.quality, p.label) for p in a])
    t_rate = _clean([RDPoint(math.log10(p.rate_kbps), p.quality, p.label) for p in t])
    common = _common_range(a_rate, t_rate, log_rate=False, method=method)
    if common is None:
        return _unavailable(
            "rate ranges do not overlap; cannot integrate a common range",
            len(a),
            len(t),
            method,
        )
    lo, hi = common
    xs = np.linspace(lo, hi, 1001)

    def interp(points: list[RDPoint], grid: np.ndarray) -> np.ndarray:
        x = np.asarray([p.quality for p in points], dtype=float)
        y = np.asarray([p.rate_kbps for p in points], dtype=float)
        if method == "cubic":
            coeffs = np.polyfit(x, y, min(3, len(points) - 1))
            return np.polyval(coeffs, grid)
        from scipy.interpolate import PchipInterpolator

        return PchipInterpolator(x, y, extrapolate=False)(grid)

    a_psnr = interp(a_rate, xs)
    t_psnr = interp(t_rate, xs)
    if not (np.all(np.isfinite(a_psnr)) and np.all(np.isfinite(t_psnr))):
        return _unavailable(
            "interpolation produced non-finite values (check rate ordering)",
            len(a),
            len(t),
            method,
        )
    delta = float(np.trapezoid(t_psnr - a_psnr, xs) / (hi - lo))
    return {
        "available": True,
        "method": method,
        "bd_psnr_db": delta,
        "common_log10_rate_range": [lo, hi],
        "anchor_points": len(a),
        "test_points": len(t),
        "min_points": min_points,
    }


def _self_test() -> int:
    # Four points; test is exactly 10% below anchor at every quality.
    anchor = [
        RDPoint(30.0, 1000.0),
        RDPoint(33.0, 500.0),
        RDPoint(36.0, 250.0),
        RDPoint(39.0, 125.0),
    ]
    test = [RDPoint(p.quality, p.rate_kbps * 0.9) for p in anchor]
    for method in ("pchip", "cubic"):
        result = bd_rate(anchor, test, method=method)
        if not result["available"]:
            print(f"FAIL {method}: {result['reason']}")
            return 1
        got = result["bd_rate_pct"]
        if abs(got + 10.0) > 0.05:
            print(f"FAIL {method}: BD-rate {got:.4f}%, expected -10%")
            return 1
        print(f"PASS {method}: BD-rate {got:.4f}%")

    # Three points: quadratic fallback, test has +1 dB at equal rate.
    anchor3 = [RDPoint(30.0, 1000.0), RDPoint(33.0, 500.0), RDPoint(36.0, 250.0)]
    test3 = [RDPoint(q + 1.0, r) for q, r in ((p.quality, p.rate_kbps) for p in anchor3)]
    result = bd_psnr(anchor3, test3)
    if not result["available"] or abs(result["bd_psnr_db"] - 1.0) > 0.05:
        print(f"FAIL bd_psnr: {result}")
        return 1
    print(f"PASS bd_psnr: {result['bd_psnr_db']:.4f} dB")

    # One point must be reported as unavailable, never coerced to 0%.
    result = bd_rate(anchor[:1], test[:1])
    if result["available"]:
        print("FAIL single-point BD-rate should be unavailable")
        return 1
    print(f"PASS single-point: {result['reason']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
