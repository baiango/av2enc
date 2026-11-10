#!/usr/bin/env python3
"""Analytic multi-GPU planner for av2enc.

This tool is a *planning model*, not an encoder and not a benchmark. It needs
neither CUDA nor a multi-GPU host. It models tile-parallel and GOP-parallel
work distribution for N x RTX 3060 12 GB cards, each running the compat8g
7,500 MB cap, and reports:

  - tiles (or GOP segments) per GPU and the resulting load imbalance
  - duplicated reference memory per GPU and across the pool
  - host-mediated PCIe traffic per frame and its time at the target fps
  - estimated frame time and scaling versus one GPU
  - a per-GPU live-VRAM estimate under the 7,500 MB cap

The absolute timing is calibrated to the product floor (1080p at ~1 fps on one
8 GB-class RTX 3060). It is an assumption, not a measurement. Every number this
tool prints is SIMULATED. See README.md and docs/multi-gpu.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

SB = 64  # AV2 superblock edge used by av2enc (luma samples)
MIB = 1024 * 1024
CDF_KIB = 46.7  # full per-tile CDF working set, docs/architecture.md
# PCIe useful payload per lane per direction, GB/s. 128b/130b for gen 3/4,
# 128b/130b with FLIT for gen 5; these are raw line-rate values before the
# efficiency factor.
PCIE_PER_LANE_GBPS = {3: 0.985, 4: 1.969, 5: 3.938}

RESOLUTIONS = {
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
    "2160p": (3840, 2160),
}
DEFAULT_LAYOUT = {
    1: "4x2",
    2: "8x2",
    3: "16x2",
    4: "16x2",
    5: "16x2",
    6: "16x2",
    7: "16x2",
    8: "16x2",
}
LAYOUTS = ("4x2", "8x2", "16x2")


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def parse_layout(text: str) -> tuple[int, int]:
    try:
        cols_s, rows_s = text.lower().split("x", 1)
        cols, rows = int(cols_s), int(rows_s)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"bad tile layout {text!r}; expected CxR, e.g. 8x2") from exc
    if cols < 1 or rows < 1 or (cols & (cols - 1)) or (rows & (rows - 1)):
        raise ValueError(f"tile layout {text!r}: columns and rows must be powers of two")
    return cols, rows


def log2_exact(value: int) -> int:
    if value < 1 or (value & (value - 1)):
        raise ValueError(f"{value} is not a positive power of two")
    return value.bit_length() - 1


@dataclass(frozen=True)
class Tile:
    index: int
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int
    sbs_w: int
    sbs_h: int

    @property
    def pixels(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def sb_area(self) -> int:
        return self.sbs_w * self.sbs_h


def uniform_starts(full_sbs: int, sbs: int, tile_log2: int) -> list[int]:
    """AV2 uniform tile spacing, docs/tiles.md section 5.18.7.5.

    ``full_sbs`` is floor(dimension / 64), ``sbs`` is ceil(dimension / 64).
    The final tile extends to the end of the superblock grid, so a partial
    superblock row/column lands in the last tile.
    """
    tile_sb = full_sbs >> tile_log2
    extra = sbs if tile_sb == 0 else full_sbs - (tile_sb << tile_log2)
    starts: list[int] = []
    start = 0
    for i in range(1 << tile_log2):
        if start >= sbs:
            break
        starts.append(start)
        start += tile_sb
        if i < extra:
            start += 1
    return starts


def build_tiles(width: int, height: int, layout: tuple[int, int]) -> list[Tile]:
    cols, rows = layout
    col_starts = uniform_starts(width // SB, ceil_div(width, SB), log2_exact(cols))
    row_starts = uniform_starts(height // SB, ceil_div(height, SB), log2_exact(rows))
    if len(col_starts) != cols or len(row_starts) != rows:
        raise ValueError(
            f"layout {cols}x{rows} is not representable at {width}x{height}: "
            f"got {len(col_starts)}x{len(row_starts)} tiles"
        )
    sb_cols = ceil_div(width, SB)
    sb_rows = ceil_div(height, SB)
    tiles: list[Tile] = []
    index = 0
    for row in range(rows):
        y0 = row_starts[row] * SB
        y1 = min((row_starts[row + 1] if row + 1 < rows else sb_rows) * SB, height)
        for col in range(cols):
            x0 = col_starts[col] * SB
            x1 = min((col_starts[col + 1] if col + 1 < cols else sb_cols) * SB, width)
            tiles.append(
                Tile(
                    index=index,
                    row=row,
                    col=col,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    sbs_w=(x1 - x0 + SB - 1) // SB,
                    sbs_h=(y1 - y0 + SB - 1) // SB,
                )
            )
            index += 1
    if sum(t.pixels for t in tiles) != width * height:
        raise AssertionError("tile areas do not sum to the frame area")
    return tiles


def content_factors(count: int, skew: float) -> list[float]:
    """Deterministic synthetic content-complexity factors.

    Tile cost is not proportional to area: a detailed tile can cost noticeably
    more than a flat one. We use a stable hash of the tile index and normalize
    the mean to 1.0 so total work is unchanged. This is a modelling device,
    not measured content.
    """
    if skew <= 0.0:
        return [1.0] * count
    raw: list[float] = []
    for i in range(count):
        u = math.sin((i + 1) * 12.9898 + 78.233) * 43758.5453
        frac = u - math.floor(u)
        raw.append(1.0 + skew * (2.0 * frac - 1.0))
    mean = sum(raw) / count
    return [r / mean for r in raw]


def assign_balanced(tiles: list[Tile], gpus: int) -> list[list[int]]:
    """Longest-processing-time-first on tile pixel area.

    Assignment is deterministic: largest tile first, then lowest tile index;
    ties go to the lowest GPU index. The bitstream merge order is independent
    of this assignment and stays in raster TileNum order.
    """
    loads = [0] * gpus
    assign: list[list[int]] = [[] for _ in range(gpus)]
    order = sorted(range(len(tiles)), key=lambda i: (-tiles[i].pixels, i))
    for i in order:
        g = min(range(gpus), key=lambda k: (loads[k], k))
        assign[g].append(i)
        loads[g] += tiles[i].pixels
    for g in range(gpus):
        assign[g].sort()
    return assign


def assign_contiguous(tiles: list[Tile], gpus: int) -> list[list[int]]:
    """Optimal contiguous raster ranges.

    Dynamic programming over tile boundaries minimises the largest per-GPU
    pixel area. This mirrors the architecture note that each GPU owns a
    contiguous tile range. It cannot always beat balanced assignment: at
    1080p the final superblock row is only 56 px high, so a two-way split at a
    row boundary is inherently uneven.
    """
    n = len(tiles)
    prefix = [0] * (n + 1)
    for i, tile in enumerate(tiles):
        prefix[i + 1] = prefix[i] + tile.pixels

    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(gpus + 1)]
    cut = [[0] * (n + 1) for _ in range(gpus + 1)]
    dp[0][0] = 0.0
    for g in range(1, gpus + 1):
        for i in range(g, n + 1):
            for j in range(g - 1, i):
                value = max(dp[g - 1][j], prefix[i] - prefix[j])
                if value < dp[g][i]:
                    dp[g][i] = value
                    cut[g][i] = j

    assign: list[list[int]] = [[] for _ in range(gpus)]
    end = n
    for g in range(gpus, 0, -1):
        start = cut[g][end]
        assign[g - 1] = list(range(start, end))
        end = start
    return assign


def load_metrics(
    assign: list[list[int]], tiles: list[Tile], factors: list[float]
) -> tuple[list[int], list[float], float, float, float, float]:
    area_loads = [sum(tiles[i].pixels for i in group) for group in assign]
    cost_loads = [sum(tiles[i].pixels * factors[i] for i in group) for group in assign]
    area_mean = sum(area_loads) / len(area_loads)
    cost_mean = sum(cost_loads) / len(cost_loads)
    area_imbalance = max(area_loads) / area_mean if area_mean else 0.0
    cost_imbalance = max(cost_loads) / cost_mean if cost_mean else 0.0
    min_ratio = min(cost_loads) / cost_mean if cost_mean else 0.0
    cv = (
        math.sqrt(sum((x - cost_mean) ** 2 for x in cost_loads) / len(cost_loads))
        / cost_mean
        if cost_mean
        else 0.0
    )
    return area_loads, cost_loads, area_imbalance, cost_imbalance, min_ratio, cv


def pcie_effective_gbps(gen: int, width: int, efficiency: float) -> float:
    return PCIE_PER_LANE_GBPS[gen] * width * efficiency


def transfer_parts(
    mode: str,
    frame_bytes: int,
    gpus: int,
    tiles: int,
    bitrate_mbps: float,
    fps: float,
) -> dict[str, float]:
    payload = bitrate_mbps * 1e6 / 8.0 / fps
    if mode == "gop":
        return {
            "input_upload": float(frame_bytes),
            "tile_payload": payload,
            "recon_gather": 0.0,
            "ref_fill": 0.0,
            "cdf_gather": 0.0,
        }
    return {
        # Upper bound: each GPU receives the full source frame. An optimized
        # implementation could upload only each GPU's tile regions (~1 frame
        # total instead of G frames).
        "input_upload": float(gpus * frame_bytes),
        # Compressed tile payloads, D2H to the coordinator.
        "tile_payload": payload,
        # Each GPU's local reconstructions are gathered once to form the full
        # reference frame.
        "recon_gather": float(frame_bytes) if gpus > 1 else 0.0,
        # The coordinator fills the missing regions on every GPU. Each GPU
        # already owns 1/G of the new reference frame.
        "ref_fill": float((gpus - 1) * frame_bytes) if gpus > 1 else 0.0,
        # Only the first min(8, tiles) tile CDFs participate in the frame-level
        # CDF average (docs/tiles.md section 4).
        "cdf_gather": float(min(8, tiles) * CDF_KIB * 1024) if gpus > 1 else 0.0,
    }


def transfer_time_ms(total_bytes: float, gbps: float, phases: int) -> float:
    # One serialised host-mediated sequence plus a small per-phase latency.
    return total_bytes / (gbps * 1e9) * 1000.0 + 0.05 * phases


def vram_estimate_mb(
    width: int,
    height: int,
    frame_bytes: int,
    refs: int,
    lookahead: int,
    local_tiles: int,
    max_tile_pixels: int,
    rdo_multiplier: float,
    headroom_mb: float,
) -> dict[str, float]:
    # docs/vram.md arena split. state_bytes = max(1 MiB, 64 B per 16x16 block).
    state = max(1.0 * MIB, width * height / 4.0)
    persistent = (refs + lookahead + 2) * frame_bytes + state
    scratch = local_tiles * (max_tile_pixels * 24.0 + 2.0 * MIB)
    candidate = frame_bytes * rdo_multiplier
    headroom = headroom_mb * MIB
    total = persistent + scratch + candidate + headroom
    return {
        "persistent_mb": persistent / MIB,
        "scratch_mb": scratch / MIB,
        "candidate_mb": candidate / MIB,
        "headroom_mb": headroom / MIB,
        "live_mb": total / MIB,
    }


def simulate_tile(
    width: int,
    height: int,
    gpus: int,
    layout: tuple[int, int],
    cfg: argparse.Namespace,
    resolution_name: str,
) -> dict:
    tiles = build_tiles(width, height, layout)
    if len(tiles) < gpus:
        raise ValueError(
            f"{resolution_name} {layout[0]}x{layout[1]} has {len(tiles)} tiles "
            f"for {gpus} GPUs; use more tiles or fewer GPUs"
        )
    if cfg.assignment == "balanced":
        assign = assign_balanced(tiles, gpus)
    else:
        assign = assign_contiguous(tiles, gpus)
    factors = content_factors(len(tiles), cfg.content_skew)
    area_loads, cost_loads, area_imb, load_imb, min_ratio, cv = load_metrics(
        assign, tiles, factors
    )

    frame_bytes = int(width * height * 1.5)
    bits = transfer_parts(
        "tile", frame_bytes, gpus, len(tiles), cfg.bitrate_mbps, cfg.fps
    )
    total_xfer_bytes = sum(bits.values())
    phases = sum(1 for v in bits.values() if v > 0.0)
    gbps = pcie_effective_gbps(cfg.pcie_gen, cfg.pcie_width, cfg.pcie_efficiency)
    xfer_ms = transfer_time_ms(total_xfer_bytes, gbps, phases)

    # Calibrated so a 1080p single-GPU frame is about 1.00 s.
    compute_1gpu_ms = width * height * cfg.ms_per_mpx / 1e6
    frame_ms = (
        cfg.serial_fraction * compute_1gpu_ms
        + (1.0 - cfg.serial_fraction) * load_imb * compute_1gpu_ms / gpus
        + xfer_ms
    )
    baseline_ms = compute_1gpu_ms + transfer_time_ms(
        sum(
            transfer_parts(
                "tile", frame_bytes, 1, len(tiles), cfg.bitrate_mbps, cfg.fps
            ).values()
        ),
        gbps,
        2,
    )
    speedup = baseline_ms / frame_ms

    per_gpu_vram = []
    for group in assign:
        local_tiles = [tiles[i] for i in group]
        per_gpu_vram.append(
            vram_estimate_mb(
                width,
                height,
                frame_bytes,
                cfg.refs,
                cfg.lookahead,
                len(local_tiles),
                max(t.pixels for t in local_tiles),
                cfg.rdo_multiplier,
                cfg.headroom_mb,
            )
        )
    vram_live = max(v["live_mb"] for v in per_gpu_vram)
    vram_detail = max(per_gpu_vram, key=lambda v: v["live_mb"])

    ref_pool_mb = cfg.refs * frame_bytes / MIB
    ref_total_mb = gpus * ref_pool_mb
    ref_extra_mb = (gpus - 1) * ref_pool_mb

    return {
        "resolution": resolution_name,
        "width": width,
        "height": height,
        "mode": "tile",
        "scale_mode": cfg.scale_mode,
        "gpus": gpus,
        "layout": f"{layout[0]}x{layout[1]}",
        "assignment": cfg.assignment,
        "tile_count": len(tiles),
        "tiles_per_gpu": [len(g) for g in assign],
        "gpu_tile_indices": assign,
        "gpu_area_sb": [round(a / (SB * SB), 3) for a in area_loads],
        "gpu_cost_sb_equiv": [round(c / (SB * SB), 3) for c in cost_loads],
        "max_tile_area_sb": max(t.sb_area for t in tiles),
        "mean_tile_area_sb": round(sum(t.sb_area for t in tiles) / len(tiles), 3),
        "area_imbalance": round(area_imb, 4),
        "load_imbalance": round(load_imb, 4),
        "load_min_ratio": round(min_ratio, 4),
        "load_cv": round(cv, 4),
        "ref_pool_mb_per_gpu": round(ref_pool_mb, 2),
        "ref_dup_total_mb": round(ref_total_mb, 2),
        "ref_dup_extra_mb": round(ref_extra_mb, 2),
        "host_transfer_parts_mb": {
            k: round(v / MIB, 3) for k, v in bits.items()
        },
        "host_transfer_mb": round(total_xfer_bytes / MIB, 2),
        "host_transfer_ms": round(xfer_ms, 3),
        "host_transfer_pct_1fps": round(
            100.0 * xfer_ms / (1000.0 / cfg.fps), 3
        ),
        "pcie": f"{cfg.pcie_gen}.0 x{cfg.pcie_width}",
        "pcie_effective_gbps": round(gbps, 1),
        "compute_ms_1gpu": round(compute_1gpu_ms, 1),
        "frame_ms": round(frame_ms, 1),
        "throughput_fps": round(1000.0 / frame_ms, 3),
        "speedup_vs_1gpu": round(speedup, 3),
        "efficiency": round(speedup / gpus, 3),
        "vram_breakdown_mb": {
            k: round(v, 2) for k, v in vram_detail.items()
        },
        "vram_live_mb": round(vram_live, 1),
        "vram_cap_mb": cfg.vram_cap_mb,
        "vram_fits": vram_live <= cfg.vram_cap_mb,
    }


def simulate_gop(
    width: int,
    height: int,
    gpus: int,
    cfg: argparse.Namespace,
    resolution_name: str,
) -> dict:
    frames = max(1, cfg.frames)
    gop_len = max(1, cfg.gop_length)
    segments = ceil_div(frames, gop_len)
    if segments < gpus:
        raise ValueError(
            f"GOP mode needs at least {gpus} segments; have {segments} "
            f"({frames} frames / {gop_len}-frame GOP)"
        )
    seg_frames = [min(gop_len, frames - i * gop_len) for i in range(segments)]
    factors = content_factors(segments, cfg.content_skew)
    # Round-robin contiguous GOP assignment. Segment order stays fixed.
    assign = [list(range(g % gpus, segments, gpus)) for g in range(gpus)]
    loads = [
        sum(seg_frames[i] * factors[i] for i in group) for group in assign
    ]
    mean_load = sum(loads) / gpus
    load_imb = max(loads) / mean_load if mean_load else 1.0

    frame_bytes = int(width * height * 1.5)
    bits = transfer_parts(
        "gop", frame_bytes, gpus, segments, cfg.bitrate_mbps, cfg.fps
    )
    total_xfer_bytes = sum(bits.values()) * frames
    phases = sum(1 for v in bits.values() if v > 0.0)
    gbps = pcie_effective_gbps(cfg.pcie_gen, cfg.pcie_width, cfg.pcie_efficiency)
    xfer_ms_total = transfer_time_ms(total_xfer_bytes, gbps, phases)

    compute_total_ms = frames * width * height * cfg.ms_per_mpx / 1e6
    serial = 0.01  # segment assembly only; no per-frame barrier
    wall_ms = (
        serial * compute_total_ms
        + (1.0 - serial) * load_imb * compute_total_ms / gpus
        + xfer_ms_total
    )
    baseline_ms = compute_total_ms + transfer_time_ms(
        sum(
            transfer_parts(
                "gop", frame_bytes, 1, segments, cfg.bitrate_mbps, cfg.fps
            ).values()
        )
        * frames,
        gbps,
        2,
    )
    speedup = baseline_ms / wall_ms
    frame_ms = wall_ms / frames

    ref_pool_mb = cfg.refs * frame_bytes / MIB
    keyframe_penalty = (
        (cfg.keyframe_ratio + gop_len - 1) / gop_len
        / ((cfg.keyframe_ratio + cfg.baseline_keyframe_interval - 1)
           / cfg.baseline_keyframe_interval)
        - 1.0
    )

    # A single GPU still needs the same per-GPU arena; GOP mode does not
    # change the live set. Each frame is still encoded with a tile layout, so
    # size the scratch from that layout rather than from the GOP length.
    layout_name = DEFAULT_LAYOUT.get(gpus, "16x2")
    frame_tiles = build_tiles(width, height, parse_layout(layout_name))
    local_tiles = ceil_div(len(frame_tiles), gpus)
    max_tile_px = max(t.pixels for t in frame_tiles)
    v = vram_estimate_mb(
        width,
        height,
        frame_bytes,
        cfg.refs,
        cfg.lookahead,
        local_tiles,
        max_tile_px,
        cfg.rdo_multiplier,
        cfg.headroom_mb,
    )

    return {
        "resolution": resolution_name,
        "width": width,
        "height": height,
        "mode": "gop",
        "scale_mode": cfg.scale_mode,
        "gpus": gpus,
        "layout": "gop",
        "assignment": "round-robin gop",
        "frames": frames,
        "gop_length": gop_len,
        "segments": segments,
        "segments_per_gpu": [len(g) for g in assign],
        "gpu_segment_indices": assign,
        "load_imbalance": round(load_imb, 4),
        "ref_pool_mb_per_gpu": round(ref_pool_mb, 2),
        "ref_dup_total_mb": round(gpus * ref_pool_mb, 2),
        "ref_dup_extra_mb": round((gpus - 1) * ref_pool_mb, 2),
        "host_transfer_parts_mb": {k: round(v / MIB, 3) for k, v in bits.items()},
        "host_transfer_mb": round(sum(bits.values()) / MIB, 2),
        "host_transfer_ms": round(xfer_ms_total / frames, 3),
        "host_transfer_pct_1fps": round(
            100.0 * (xfer_ms_total / frames) / (1000.0 / cfg.fps), 3
        ),
        "pcie": f"{cfg.pcie_gen}.0 x{cfg.pcie_width}",
        "pcie_effective_gbps": round(gbps, 1),
        "compute_ms_1gpu": round(compute_total_ms / frames, 1),
        "frame_ms": round(frame_ms, 1),
        "throughput_fps": round(1000.0 / frame_ms, 3),
        "speedup_vs_1gpu": round(speedup, 3),
        "efficiency": round(speedup / gpus, 3),
        "keyframe_ratio": cfg.keyframe_ratio,
        "baseline_keyframe_interval": cfg.baseline_keyframe_interval,
        "keyframe_bitrate_penalty_pct": round(100.0 * keyframe_penalty, 2),
        "vram_breakdown_mb": {k: round(val, 2) for k, val in v.items()},
        "vram_live_mb": round(v["live_mb"], 1),
        "vram_cap_mb": cfg.vram_cap_mb,
        "vram_fits": v["live_mb"] <= cfg.vram_cap_mb,
    }


def parse_int_list(text: str) -> list[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("empty list")
    return values


def resolve_resolutions(cfg: argparse.Namespace) -> list[tuple[str, int, int]]:
    if cfg.resolution and (cfg.width or cfg.height):
        raise ValueError("use either --resolution or --width/--height, not both")
    if bool(cfg.width) != bool(cfg.height):
        raise ValueError("--width and --height must be given together")
    if cfg.width and cfg.height:
        if cfg.width <= 0 or cfg.height <= 0:
            raise ValueError("--width and --height must be positive")
        if cfg.width % 2 or cfg.height % 2:
            raise ValueError("4:2:0 input requires even width and height")
        return [(f"{cfg.width}x{cfg.height}", cfg.width, cfg.height)]
    if cfg.resolution:
        w, h = RESOLUTIONS[cfg.resolution]
        return [(cfg.resolution, w, h)]
    if cfg.matrix:
        return [("1080p", *RESOLUTIONS["1080p"]), ("4k", *RESOLUTIONS["4k"])]
    return [("1080p", *RESOLUTIONS["1080p"])]


def resolve_layouts(cfg: argparse.Namespace, gpu_count: int) -> list[tuple[int, int]]:
    if cfg.tiles == "auto":
        name = DEFAULT_LAYOUT.get(gpu_count)
        if name is None:
            raise ValueError(
                f"no default tile layout for {gpu_count} GPUs; pass --tiles"
            )
        return [parse_layout(name)]
    layouts = [parse_layout(name) for name in cfg.tiles.split(",") if name.strip()]
    if not layouts:
        raise ValueError("--tiles must be 'auto' or a layout such as 8x2")
    return layouts


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]
    lines = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |"
        )
    return "\n".join(lines)


def print_text(results: list[dict], cfg: argparse.Namespace) -> None:
    gbps = pcie_effective_gbps(cfg.pcie_gen, cfg.pcie_width, cfg.pcie_efficiency)
    print("av2enc multi-GPU planner - SIMULATION ONLY")
    print("No CUDA, no multi-GPU hardware, no measured scaling.")
    print(
        f"PCIe {cfg.pcie_gen}.0 x{cfg.pcie_width} "
        f"(effective {gbps:.1f} GB/s), refs={cfg.refs}, "
        f"target {cfg.fps:g} fps, serial fraction {cfg.serial_fraction:.1%}, "
        f"content skew +/-{cfg.content_skew:.0%}"
    )
    print(f"scale mode: {cfg.scale_mode} (default quality)")
    print()
    print("Scheduling and scaling")
    headers = [
        "res",
        "gpus",
        "layout",
        "tiles/GPU",
        "SB max/mean",
        "load imb",
        "work ms",
        "speedup",
        "eff",
    ]
    rows: list[list[str]] = []
    for r in results:
        rows.append(
            [
                r["resolution"],
                str(r["gpus"]),
                r["layout"],
                "/".join(str(n) for n in r["tiles_per_gpu"]),
                f"{r['max_tile_area_sb']}/{r['mean_tile_area_sb']:.1f}",
                f"{r['load_imbalance']:.3f}",
                f"{r['frame_ms']:.1f}",
                f"{r['speedup_vs_1gpu']:.3f}",
                f"{r['efficiency']:.3f}",
            ]
        )
    print(markdown_table(headers, rows) if cfg.markdown else _plain(headers, rows))
    print()
    print("Memory and host transfer")
    headers2 = [
        "res",
        "gpus",
        "ref/GPU MB",
        "dup extra MB",
        "xfer MB",
        "xfer ms",
        "xfer %1fps",
        "live MB",
        "cap MB",
        "fits",
    ]
    rows2: list[list[str]] = []
    for r in results:
        rows2.append(
            [
                r["resolution"],
                str(r["gpus"]),
                f"{r['ref_pool_mb_per_gpu']:.2f}",
                f"{r['ref_dup_extra_mb']:.2f}",
                f"{r['host_transfer_mb']:.2f}",
                f"{r['host_transfer_ms']:.3f}",
                f"{r['host_transfer_pct_1fps']:.2f}",
                f"{r['vram_live_mb']:.1f}",
                f"{r['vram_cap_mb']:.0f}",
                "yes" if r["vram_fits"] else "NO",
            ]
        )
    print(markdown_table(headers2, rows2) if cfg.markdown else _plain(headers2, rows2))
    print()
    print(
        "Notes: work ms is the same-quality parallel work time. speedup is "
        "parallel work-time scaling at the same quality. quality mode holds "
        "aggregate fps at the target and spends this budget on quality; "
        "throughput mode raises fps. MB means MiB."
    )
    print(
        "Top scaling risk: content-dependent tile cost (a straggler GPU gates "
        "every frame barrier). PCIe ref sync is negligible at 1 fps but becomes "
        "material in throughput mode."
    )
    print(
        "All live-VRAM figures are model estimates; the arena still reserves "
        f"the {cfg.vram_cap_mb:.0f} MB per-GPU cap."
    )


def _plain(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]
    lines = [
        "  ".join(h.ljust(w) for h, w in zip(headers, widths)),
        "  ".join("-" * w for w in widths),
    ]
    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)


def print_gop(results: list[dict], cfg: argparse.Namespace) -> None:
    gbps = pcie_effective_gbps(cfg.pcie_gen, cfg.pcie_width, cfg.pcie_efficiency)
    print("av2enc multi-GPU planner - SIMULATION ONLY (GOP mode)")
    print(
        f"PCIe {cfg.pcie_gen}.0 x{cfg.pcie_width} (effective {gbps:.1f} GB/s), "
        f"refs={cfg.refs}, {cfg.frames} frames, {cfg.gop_length}-frame GOP"
    )
    print()
    headers = [
        "res",
        "gpus",
        "segments/GPU",
        "load imb",
        "frame ms",
        "speedup",
        "eff",
        "keyframe penalty",
    ]
    rows: list[list[str]] = []
    for r in results:
        rows.append(
            [
                r["resolution"],
                str(r["gpus"]),
                "/".join(str(n) for n in r["segments_per_gpu"]),
                f"{r['load_imbalance']:.3f}",
                f"{r['frame_ms']:.1f}",
                f"{r['speedup_vs_1gpu']:.3f}",
                f"{r['efficiency']:.3f}",
                f"{r['keyframe_bitrate_penalty_pct']:+.1f}%",
            ]
        )
    print(markdown_table(headers, rows) if cfg.markdown else _plain(headers, rows))
    print()
    headers2 = [
        "res",
        "gpus",
        "ref/GPU MB",
        "xfer MB",
        "xfer ms",
        "live MB",
        "cap MB",
        "fits",
    ]
    rows2: list[list[str]] = []
    for r in results:
        rows2.append(
            [
                r["resolution"],
                str(r["gpus"]),
                f"{r['ref_pool_mb_per_gpu']:.2f}",
                f"{r['host_transfer_mb']:.2f}",
                f"{r['host_transfer_ms']:.3f}",
                f"{r['vram_live_mb']:.1f}",
                f"{r['vram_cap_mb']:.0f}",
                "yes" if r["vram_fits"] else "NO",
            ]
        )
    print(markdown_table(headers2, rows2) if cfg.markdown else _plain(headers2, rows2))
    print()
    print(
        "Keyframe penalty is relative to a single-GPU stream with a "
        f"{cfg.baseline_keyframe_interval}-frame keyframe interval and "
        f"keyframe/inter ratio {cfg.keyframe_ratio:g}. If the baseline already "
        "uses the same closed-GOP cadence, the penalty is zero."
    )
    print("MB means MiB. Every figure is modelled; none is measured on multi-GPU hardware.")


def self_test() -> int:
    checks = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        if not cond:
            raise AssertionError(msg)

    for name, (w, h) in RESOLUTIONS.items():
        for layout_name in LAYOUTS:
            tiles = build_tiles(w, h, parse_layout(layout_name))
            check(sum(t.pixels for t in tiles) == w * h, f"{name} {layout_name} area")
            check(len({t.index for t in tiles}) == len(tiles), "tile indices")
            for gpus in (1, 2, 4):
                if len(tiles) < gpus:
                    continue
                for policy in (assign_balanced, assign_contiguous):
                    assign = policy(tiles, gpus)
                    flat = sorted(i for group in assign for i in group)
                    check(flat == list(range(len(tiles))), "assignment coverage")
                    check(all(group for group in assign), "empty GPU assignment")
                    factors = content_factors(len(tiles), 0.15)
                    _, _, _, imb, _, _ = load_metrics(assign, tiles, factors)
                    check(imb >= 1.0, "imbalance below 1")
                    if gpus == 1:
                        check(abs(imb - 1.0) < 1e-9, "one GPU must have imbalance 1")

    cfg = argparse.Namespace(
        assignment="balanced",
        content_skew=0.15,
        bitrate_mbps=1.0,
        fps=1.0,
        pcie_gen=4,
        pcie_width=16,
        pcie_efficiency=0.85,
        ms_per_mpx=481.0,
        serial_fraction=0.035,
        refs=4,
        lookahead=4,
        rdo_multiplier=1.0,
        headroom_mb=512.0,
        vram_cap_mb=7500.0,
        scale_mode="quality",
        frames=240,
        gop_length=24,
        keyframe_ratio=6.0,
        baseline_keyframe_interval=240,
    )
    for name, (w, h) in (("1080p", RESOLUTIONS["1080p"]), ("4k", RESOLUTIONS["4k"])):
        prev = None
        for gpus in (1, 2, 4):
            r = simulate_tile(w, h, gpus, parse_layout(DEFAULT_LAYOUT[gpus]), cfg, name)
            check(r["vram_fits"], f"{name} {gpus}gpu vram")
            check(r["speedup_vs_1gpu"] > 0, "speedup")
            if gpus == 1:
                check(abs(r["speedup_vs_1gpu"] - 1.0) < 1e-3, "1 GPU speedup")
            if prev is not None:
                check(r["speedup_vs_1gpu"] > prev, "speedup must increase with GPUs")
            prev = r["speedup_vs_1gpu"]
        a = simulate_tile(w, h, 2, parse_layout(DEFAULT_LAYOUT[2]), cfg, name)
        b = simulate_tile(w, h, 2, parse_layout(DEFAULT_LAYOUT[2]), cfg, name)
        check(a == b, "simulation must be deterministic")

    gop = simulate_gop(1920, 1080, 2, cfg, "1080p")
    check(gop["segments"] == 10, "GOP segment count")
    check(gop["segments_per_gpu"] == [5, 5], "GOP segment split")
    check(gop["speedup_vs_1gpu"] > 1.8, "2-GPU GOP speedup")
    check(gop["keyframe_bitrate_penalty_pct"] > 0.0, "GOP keyframe penalty")
    check(gop["vram_fits"], "GOP VRAM fit")

    print(f"PASS: {checks} geometry, assignment, VRAM, and determinism checks")
    print("PASS: synthetic content factors are deterministic")
    print("No hardware was used.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Analytic multi-GPU planner for av2enc (no CUDA, no hardware).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--self-test", action="store_true", help="run model invariants")
    ap.add_argument("--resolution", choices=sorted(RESOLUTIONS), help="named input size")
    ap.add_argument("--width", type=int, help="frame width in pixels")
    ap.add_argument("--height", type=int, help="frame height in pixels")
    ap.add_argument(
        "--gpus",
        default="1,2,4",
        help="GPU counts to simulate (counts, not device IDs; e.g. 2)",
    )
    ap.add_argument(
        "--tiles",
        default="auto",
        help="tile layout CxR, 'auto', or a comma list (e.g. 4x2,8x2,16x2)",
    )
    ap.add_argument("--matrix", action="store_true", help="1080p and 4K matrix")
    ap.add_argument("--multi-gpu", choices=("tile", "gop"), default="tile")
    ap.add_argument(
        "--scale-mode", choices=("quality", "throughput"), default="quality"
    )
    ap.add_argument("--pcie-gen", type=int, choices=sorted(PCIE_PER_LANE_GBPS), default=4)
    ap.add_argument("--pcie-width", type=int, choices=(1, 2, 4, 8, 16), default=16)
    ap.add_argument("--pcie-efficiency", type=float, default=0.85)
    ap.add_argument("--refs", type=int, default=4)
    ap.add_argument("--lookahead", type=int, default=4)
    ap.add_argument("--fps", type=float, default=1.0, help="target fps for transfer ms")
    ap.add_argument("--bitrate-mbps", type=float, default=1.0)
    ap.add_argument(
        "--ms-per-mp",
        dest="ms_per_mpx",
        type=float,
        default=481.0,
        help="single-GPU ms per luma megapixel (calibrated to 1080p ~1 fps)",
    )
    ap.add_argument(
        "--serial-fraction",
        type=float,
        default=0.035,
        help="single-GPU frame time that does not shrink with more GPUs",
    )
    ap.add_argument(
        "--content-skew",
        type=float,
        default=0.15,
        help="synthetic per-tile cost variation, +/- fraction",
    )
    ap.add_argument(
        "--assignment", choices=("balanced", "contiguous"), default="balanced"
    )
    ap.add_argument("--vram-cap-mb", type=float, default=7500.0)
    ap.add_argument("--rdo-multiplier", type=float, default=1.0)
    ap.add_argument("--headroom-mb", type=float, default=512.0)
    ap.add_argument("--frames", type=int, default=240, help="GOP mode clip length")
    ap.add_argument("--gop-length", type=int, default=24)
    ap.add_argument("--keyframe-ratio", type=float, default=6.0)
    ap.add_argument("--baseline-keyframe-interval", type=int, default=240)
    ap.add_argument("--json", type=Path, help="write the model result as JSON")
    ap.add_argument("--markdown", action="store_true", help="print Markdown tables")
    cfg = ap.parse_args(argv)

    if cfg.self_test:
        return self_test()

    try:
        gpu_counts = parse_int_list(cfg.gpus)
        if any(g < 1 for g in gpu_counts):
            raise ValueError(
                "GPU counts must be positive; pass counts such as 2, "
                "not device IDs such as 0,1"
            )
        resolutions = resolve_resolutions(cfg)
    except ValueError as exc:
        ap.error(str(exc))
        return 2

    results: list[dict] = []
    try:
        for res_name, width, height in resolutions:
            for gpus in gpu_counts:
                layouts = resolve_layouts(cfg, gpus)
                if cfg.multi_gpu == "gop":
                    # GOP mode assigns whole segments; --tiles only changes the
                    # per-frame tile layout, which the GOP planner derives
                    # from its default layout. Do not emit one row per layout.
                    results.append(simulate_gop(width, height, gpus, cfg, res_name))
                    continue
                for layout in layouts:
                    results.append(
                        simulate_tile(width, height, gpus, layout, cfg, res_name)
                    )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if cfg.multi_gpu == "gop":
        print_gop(results, cfg)
    else:
        print_text(results, cfg)

    if cfg.json:
        payload = {
            "model": "analytic-multi-gpu-v1",
            "measured": False,
            "disclaimer": (
                "No multi-GPU hardware was used. All figures are model estimates, "
                "not measured validation."
            ),
            "config": {
                "pcie_gen": cfg.pcie_gen,
                "pcie_width": cfg.pcie_width,
                "pcie_efficiency": cfg.pcie_efficiency,
                "refs": cfg.refs,
                "lookahead": cfg.lookahead,
                "fps": cfg.fps,
                "bitrate_mbps": cfg.bitrate_mbps,
                "ms_per_mpx": cfg.ms_per_mpx,
                "serial_fraction": cfg.serial_fraction,
                "content_skew": cfg.content_skew,
                "assignment": cfg.assignment,
                "vram_cap_mb": cfg.vram_cap_mb,
                "scale_mode": cfg.scale_mode,
            },
            "scenarios": results,
        }
        cfg.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {cfg.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
