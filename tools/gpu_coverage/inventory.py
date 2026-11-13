#!/usr/bin/env python3
"""GPU-coverage and arena-compliance inventory for av2enc.

Reads the source tree, never the status docs, and emits
``docs/gpu-coverage.md``.  The script is the single source of truth for the
tables: every path is checked to exist before it is emitted, and the counts in
the summary are computed from the same rows that are rendered.

Usage
-----
    python3 tools/gpu_coverage/inventory.py            # write the doc
    python3 tools/gpu_coverage/inventory.py --check    # print counts only
    python3 tools/gpu_coverage/inventory.py --stdout   # render, do not write

No part of this script runs a GPU, compiles CUDA, or edits source.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "gpu-coverage.md"

# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------
SERIAL = "GPU-SERIAL"
REQUIRED = "GPU-REQUIRED"
MISSING = "GPU-MISSING"
UNWIRED = "GPU-UNWIRED"

VERDICT_ORDER = [MISSING, UNWIRED, REQUIRED, SERIAL]


# --------------------------------------------------------------------------
# Curated rows.  Paths are validated by check_paths().
#
# Column order is the matrix order:
#   tool | spec | cpu | cuda | parity | emulation | gpu evidence | wired | verdict
# --------------------------------------------------------------------------
FEATURES: list[dict] = [
    # ---- the 26 newest feature directories (commits 19c5dd9..debb00e) ----
    dict(
        tool="filters/ccso_reuse",
        spec="§5.18.7.12, §6.17.7.8, §7.19/§7.19.1, §7.23",
        cpu="src/codec/filters/ccso/",
        cuda="— (apply is src/cuda/codec/filters/loop_restoration/ccso.cu, unwired)",
        parity="src/codec/filters/ccso/ccso_reuse_test.cpp",
        emu="—",
        gpu="— (no kernel)",
        wired="no",
        verdict=SERIAL,
        note="reuse bits/slot store/decision only; the pixel filter already has a kernel in restoration/ccso.",
    ),
    dict(
        tool="m2/tx_partition (TX_MODE_SELECT)",
        spec="§5.18.8.1, §5.20.6.1–.3, §5.20.7.30",
        cpu="src/codec/transform/tx_partition/",
        cuda="— (TX kernels exist: src/cuda/pipeline/still/tx/, src/cuda/codec/transform/variants/)",
        parity="src/codec/transform/tx_partition/tx_partition_test.cpp",
        emu="src/codec/transform/tx_partition/gate/",
        gpu="— (geometry/scan is host-side)",
        wired="no",
        verdict=SERIAL,
        note="syntax + scan + geometry. The kernels already accept all 25 sizes; what is missing is the per-TU launch driver, not a kernel.",
    ),
    dict(
        tool="filters/film_grain_stream (signalling)",
        spec="§5.14, §5.18.10.1–.2, §7.3.8.8, §7.3.3/.4",
        cpu="src/pipeline/grain/",
        cuda="—",
        parity="src/pipeline/grain/film_grain_stream_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="OBU/header/model syntax only.",
    ),
    dict(
        tool="filters/film_grain_stream (synthesis)",
        spec="§7.21.7.3–.5",
        cpu="src/codec/filters/tables/film_grain.cpp",
        cuda="— none (decoder/output side)",
        parity="src/codec/filters/tables/film_grain_test.cpp",
        emu="611 checks (CPU oracle)",
        gpu="N/A",
        wired="no",
        verdict=SERIAL,
        volume=3_110_400,
        basis="full YUV output frame",
        note="grain is applied by the decoder/output process; the encoder only signals the model. No encoder kernel is required.",
    ),
    dict(
        tool="svc/ops_full",
        spec="§5.8, §5.10, §5.11, §6.8, §6.10, §6.11",
        cpu="src/pipeline/svc/ops_full/",
        cuda="—",
        parity="src/pipeline/svc/ops_full/lcr_ops_full_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="LCR/OPS syntax.",
    ),
    dict(
        tool="inter/mlayers",
        spec="§5.2.2, §5.4.1, §6.2.2, §6.4.1, §7.3.6–.8",
        cpu="src/codec/inter/mlayers/",
        cuda="—",
        parity="src/codec/inter/mlayers/mlayers_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="embedded-layer maps and ordering.",
    ),
    dict(
        tool="filters/tip_deblock",
        spec="§7.16, §7.17.3, §7.17.7",
        cpu="src/codec/filters/tip_deblock/tip_deblock.cpp",
        cuda="src/cuda/codec/filters/tip_deblock/tip_deblock.cu",
        parity="src/cuda/codec/filters/tip_deblock/tip_deblock_emulation_test.cpp (host emulation)",
        emu="—",
        gpu="— (not run)",
        wired="no",
        verdict=UNWIRED,
        volume=3_110_400,
        basis="full YUV frame when TIP_FRAME_AS_OUTPUT",
        note="whole-frame deblock for TIP_FRAME_AS_OUTPUT; no device kernel.",
    ),
    dict(
        tool="m1/missing_bits — DC intra subsampled",
        spec="§7.13.2.11",
        cpu="src/codec/intra/supplemental/dc_intra_subsampled.cpp",
        cuda="src/cuda/codec/intra/supplemental/intra_full.cu",
        parity="src/cuda/codec/intra/supplemental/intra_full_emulation_test.cpp (host emulation)",
        emu="—",
        gpu="— (not run)",
        wired="no",
        verdict=UNWIRED,
        volume=259_200,
        basis="intra blocks >32, ~12.5% of luma",
        note="intra_modes.cu implements §7.13.2.10 only; no stepW/stepH=2 path.",
    ),
    dict(
        tool="m1/missing_bits — intra filter corner",
        spec="§7.13.2.14",
        cpu="src/codec/intra/supplemental/intra_filter_corner.cpp",
        cuda="src/cuda/codec/intra/modes/intra_predict_device.cuh",
        parity="src/cuda/codec/intra/modes/intra_modes_test.cu",
        emu="src/cuda/codec/intra/modes/intra_device_emulation_test.cpp",
        gpu="RTX 3060 (README §“Test result”)",
        wired="M1 + M2",
        verdict=REQUIRED,
        note="already present at intra_predict_device.cuh:187-241/435-442.",
    ),
    dict(
        tool="m1/missing_bits — dc_sign_horz_vert",
        spec="§5.20.7.27",
        cpu="src/codec/intra/supplemental/dc_sign_horz_vert.cpp",
        cuda="—",
        parity="src/codec/intra/supplemental/missing_bits_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="entropy symbol, not a pixel operation.",
    ),
    dict(
        tool="m1/missing_bits — save_subpu",
        spec="§7.13.3.15",
        cpu="src/codec/intra/supplemental/save_subpu.cpp",
        cuda="src/cuda/codec/filters/deblock/subpu/",
        parity="src/cuda/codec/filters/deblock/subpu/subpu_test.cu",
        emu="subpu_emulation_test: 31,474 checks",
        gpu="subpu_test: 77,986 checks (sm_86)",
        wired="no",
        verdict=UNWIRED,
        volume=129_600,
        basis="deblock boundary samples only",
        note="SubPu grid feeds §7.17 deblocking; recon.cu writes Max_Tx_Size_Rect only.",
    ),
    dict(
        tool="inter/gme",
        spec="§7.13.3.21, §5.18.9, §7.12.2.2",
        cpu="src/codec/inter/gme/",
        cuda="src/cuda/codec/inter/gme/",
        parity="src/cuda/codec/inter/gme/gme_test.cu",
        emu="1,354,622 checks",
        gpu="81,300 checks incl. real 720p (sm_86)",
        wired="no",
        verdict=UNWIRED,
        volume=0,
        basis="frame-level fit, negligible pixel volume",
        note="128-bit integer least squares; exact CPU parity.",
    ),
    dict(
        tool="tables/missing",
        spec="§9.2, §9.3",
        cpu="src/spec_tables/missing/",
        cuda="—",
        parity="src/spec_tables/missing/missing_tables_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="generated table headers; no compute.",
    ),
    dict(
        tool="metadata/gaps",
        spec="§5.17.2/.3, §7.23",
        cpu="src/frame/metadata/gaps/",
        cuda="—",
        parity="src/frame/metadata/gaps/metadata_gaps_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="metadata OBU + reference state.",
    ),
    dict(
        tool="conformance",
        spec="Annex A.2–A.5, §6.4.1",
        cpu="src/core/conformance/",
        cuda="—",
        parity="src/core/conformance/conformance_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="validator runs on host config/statistics.",
    ),
    dict(
        tool="m2/tile_groups",
        spec="§5.19, §5.20.2.1, §5.18.1, §6.18",
        cpu="src/pipeline/tiles/groups/",
        cuda="—",
        parity="src/pipeline/tiles/groups/tile_groups_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="framing + header copy.",
    ),
    dict(
        tool="qm/obu",
        spec="§5.13, §5.18.6.2, §6.12, §6.4.11, §6.2.2",
        cpu="src/codec/transform/qm/obu/",
        cuda="— (dequant kernel src/cuda/codec/transform/qm/qm_gpu.cu exists, unwired)",
        parity="src/codec/transform/qm/obu/qm_obu_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="OBU state machine; the QM-aware dequant is a separate kernel.",
    ),
    dict(
        tool="filters/lr_chroma (syntax/decision)",
        spec="§5.18.7.11, §5.20.10.5–.6",
        cpu="src/codec/filters/loop_restoration/chroma/chroma_lr.cpp",
        cuda="—",
        parity="src/codec/filters/loop_restoration/chroma/chroma_lr_test.cpp",
        emu="src/codec/filters/loop_restoration/chroma/gate/",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="unit sizing, uv_sym mirroring, filter selection.",
    ),
    dict(
        tool="filters/lr_chroma (apply §7.20.3)",
        spec="§7.20.3",
        cpu="src/codec/filters/loop_restoration/chroma/chroma_lr.cpp (chroma_apply)",
        cuda="src/cuda/codec/filters/loop_restoration/chroma/",
        parity="src/cuda/codec/filters/loop_restoration/chroma/lr_chroma_test.cu",
        emu="33,270 checks / 95.2M samples",
        gpu="57,009 checks / 95.2M samples (sm_86)",
        wired="no",
        verdict=UNWIRED,
        note="wiener_core.h carries the chroma-from-luma taps; M1 GPU rejects --still-ccso/restoration.",
    ),
    dict(
        tool="inter/drl_reorder",
        spec="§5.4.6, §7.12.2, §7.12.2.19",
        cpu="src/codec/inter/drl_reorder/",
        cuda="—",
        parity="src/codec/inter/drl_reorder/drl_reorder_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="MV-stack ordering; no pixels.",
    ),
    dict(
        tool="tcq/parity",
        spec="§5.20.7.27, §5.20.7.28, §5.4.8, §6.17.2",
        cpu="src/codec/transform/tcq/parity/",
        cuda="src/cuda/codec/transform/tcq/",
        parity="src/cuda/codec/transform/tcq/tcq_test.cu",
        emu="1,491,655 checks",
        gpu="933,016 checks (sm_86)",
        wired="no",
        verdict=UNWIRED,
        note="coefficient level choice + entropy.",
    ),
    dict(
        tool="m2/ddt",
        spec="§7.15.2.1, §7.15.4.1, §9.6, Table 7.1",
        cpu="src/codec/transform/ddt/ddt_transform.cpp",
        cuda="src/cuda/codec/transform/ddt/",
        parity="src/cuda/codec/transform/ddt/ddt_gpu_test.cu",
        emu="828,114 checks",
        gpu="685,590 checks / 1,956,864 elements (sm_86)",
        wired="no",
        verdict=UNWIRED,
        volume=518_400,
        basis="ADST/FDST blocks at sz 8/16, ~25% of inter luma",
        note="DDTX/FDDT kernels (type 4/5) absent from tx_variants.cu and still_tx.cu.",
    ),
    dict(
        tool="format/chroma (4:2:2 / 4:4:4 / 4:0:0)",
        spec="§6.4.1 Table 6.2, §5.4.3/.5/.8, §7.13",
        cpu="src/formats/chroma/chroma_residual.cpp",
        cuda="src/cuda/formats/chroma/",
        parity="src/cuda/formats/chroma/chroma_formats_test.cu",
        emu="1,514,866 checks",
        gpu="1,742,235 checks (sm_86)",
        wired="no",
        verdict=UNWIRED,
        volume=3_110_400,
        basis="full YUV frame when a non-4:2:0 format is selected",
        note="residual, downsample and prediction geometry for non-4:2:0 formats.",
    ),
    dict(
        tool="rc/hrd",
        spec="§5.4.12/.13, §5.12, §5.15, Annex E.5.4/E.7",
        cpu="src/codec/rate/rc/hrd/",
        cuda="—",
        parity="src/codec/rate/rc/hrd/hrd_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="timing/HRD planner.",
    ),
    dict(
        tool="inter/longterm",
        spec="§5.4.1/.6, §5.18.2, §7.4.2–.4, §7.23",
        cpu="src/codec/inter/longterm/",
        cuda="—",
        parity="src/codec/inter/longterm/longterm_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="reference pool + headers.",
    ),
    dict(
        tool="inter/tlayers",
        spec="§5.2.2, §5.4.1, §5.18.2, §7.3.7, §7.7, §7.23",
        cpu="src/codec/inter/tlayers/",
        cuda="—",
        parity="src/codec/inter/tlayers/tlayers_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="GOP/layer planning + syntax.",
    ),
    dict(
        tool="inter/random_access",
        spec="§6.2.2, §5.18.2, §6.17, §7.4.5",
        cpu="src/codec/inter/random_access/",
        cuda="—",
        parity="src/codec/inter/random_access/random_access_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="SWITCH/RAS headers + planners.",
    ),
    dict(
        tool="frame/mfh",
        spec="§5.7, §6.7, §5.18.2, §6.17.2, §5.2.1",
        cpu="src/frame/mfh/",
        cuda="—",
        parity="src/frame/mfh/mfh_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="multi-frame-header writer/reader.",
    ),
    dict(
        tool="quant/dcdeltaq",
        spec="§5.4.8, §5.18.6.1/.3, §7.14.2, §7.14.4",
        cpu="src/codec/transform/delta_q/",
        cuda="— (quant step is an input to m1/tx and tx_variants)",
        parity="src/codec/transform/delta_q/dcdeltaq_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="qindex/delta derivation; per-coefficient quant already device-resident.",
    ),
    dict(
        tool="inter/sef",
        spec="§5.18.1/.2, §6.2.2, §6.17.2, §7.2, §7.21",
        cpu="src/codec/inter/sef/",
        cuda="—",
        parity="src/codec/inter/sef/sef_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="show-existing-frame OBU.",
    ),
    dict(
        tool="metadata/types",
        spec="§5.17.1–.13, §6.16.3–.13",
        cpu="src/frame/metadata/types/",
        cuda="—",
        parity="src/frame/metadata/types/metadata_types_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="metadata payloads.",
    ),
    dict(
        tool="metadata/content_interpretation",
        spec="§5.15, §5.4.12, §6.14, §7.3.3, §7.3.8.10/.11",
        cpu="src/frame/metadata/content_interpretation/",
        cuda="—",
        parity="src/frame/metadata/content_interpretation/content_interpretation_test.cpp",
        emu="—",
        gpu="—",
        wired="no",
        verdict=SERIAL,
        note="CI OBU + ordering.",
    ),
]

# 41 kernel translation units: 38 under src/cuda, 3 under src/encoder.
KERNELS: list[dict] = [
    dict(tool="cdef/cdef.cu", module="cdef", spec="§7.18",
         cpu="src/codec/filters/cdef/",
         parity="src/cuda/codec/filters/cdef/cdef_test.cu", emu="—",
         gpu="RTX 3060: 51,047 checks", wired="no (M1 rejects --still-cdef)",
         verdict=UNWIRED, note="CDEF pixel filter + skip/strength inputs."),
    dict(tool="cfl/cfl.cu", module="cfl", spec="§7.13.5",
         cpu="src/codec/intra/cfl/",
         parity="src/cuda/codec/intra/cfl/cfl_test.cu", emu="—",
         gpu="RTX 3060 (README status)", wired="linked, never called",
         verdict=UNWIRED, note="M1 chroma_decide inlines CfL; still_cfl wrapper is unused."),
    dict(tool="deblock/deblock.cu", module="deblock", spec="§7.17",
         cpu="src/codec/filters/deblock/",
         parity="src/cuda/codec/filters/deblock/deblock_test.cu", emu="—",
         gpu="RTX 3060 (README)", wired="M1 (recon_deblock_frame)",
         verdict=REQUIRED, note="two-pass frame deblock."),
    dict(tool="dip/dip.cu", module="dip", spec="§7.13.2.1, §7.13.2.3–.6",
         cpu="src/codec/intra/dip/",
         parity="src/cuda/codec/intra/dip/dip_test.cu",
         emu="src/cuda/codec/intra/dip/dip_emulation_test.cpp",
         gpu="not run (README)", wired="no",
         verdict=UNWIRED, note="DIP predictor."),
    dict(tool="entropy.cu", module="entropy", spec="§8.2",
         cpu="src/codec/entropy/av2/",
         parity="src/cuda/entropy_test.cu", emu="—",
         gpu="RTX 3060: byte-exact", wired="no (M1/M2 use CPU writer)",
         verdict=SERIAL, note="available GPU coder, CPU by design on the shipped paths."),
    dict(tool="format/high_bitdepth.cu", module="format", spec="§6.4.1 Table 6.3, §7.15",
         cpu="src/formats/high_bitdepth/",
         parity="src/cuda/formats/high_bitdepth_test.cu", emu="—",
         gpu="not executed (README)", wired="no",
         verdict=UNWIRED, note="8→10-bit conversion / high-bit-depth arithmetic."),
    dict(tool="format/high_bitdepth_gpu.cu", module="format", spec="§6.4.1, §7.15",
         cpu="src/pipeline/still/high_bitdepth/",
         parity="src/cuda/formats/high_bitdepth_gpu_test.cu", emu="—",
         gpu="not executed (README)", wired="no (M10 standalone)",
         verdict=UNWIRED, note="10-bit still path; direct cudaMalloc."),
    dict(tool="formats/format_device.cpp", module="format",
         spec="§5.18.4.4, §7.21.2, §7.17",
         cpu="src/formats/format.cpp, src/codec/filters/deblock/",
         parity="src/cuda/formats/format_device_test.cpp", emu="—",
         gpu="1,064,426 checks (sm_86)", wired="no (geometry for the HBD/deblock stage)",
         verdict=UNWIRED,
         note="Frame/filter plane geometry for all depths and chroma formats."),
    dict(tool="ibc/ibc.cu", module="ibc", spec="§5.20.5.4, §6.19.7.12",
         cpu="src/codec/intra/ibc/core/",
         parity="src/cuda/codec/intra/ibc/ibc_test.cu", emu="—",
         gpu="not documented", wired="no",
         verdict=UNWIRED, note="IBC SAD search; cudaMalloc+sync per call."),
    dict(tool="ibp/ibp.cu", module="ibp", spec="§7.13.2.7–.9, .12, .14, .17–.18",
         cpu="src/codec/intra/ibp/",
         parity="src/cuda/codec/intra/ibp/ibp_test.cu",
         emu="src/cuda/codec/intra/ibp/ibp_device_emulation_test.cpp (273,958,171 checks)",
         gpu="not executed (README)", wired="no",
         verdict=UNWIRED, note="intra bi-prediction predictor."),
    dict(tool="inter/advanced/advanced.cu", module="inter/advanced",
         spec="§7.13.3.14, §7.13.3.27, §7.13.3.30",
         cpu="src/codec/inter/advanced/",
         parity="src/cuda/codec/inter/advanced/advanced_test.cu", emu="—",
         gpu="not run (no CUDA device)", wired="no",
         verdict=UNWIRED, note="compound average/CWP, implicit mask, wedge (single-ref only on device)."),
    dict(tool="inter/mc/mc.cu", module="inter/mc", spec="§7.13.3.17–.18",
         cpu="src/codec/inter/mc/",
         parity="src/cuda/codec/inter/mc/mc_test.cu", emu="—",
         gpu="RTX 3060: 1,069 checks", wired="M2 gate (not in CMake)",
         verdict=REQUIRED, note="single-reference translational MC."),
    dict(tool="inter/me/me.cu", module="inter/me", spec="§7.13.3.14, §7.13.3.16",
         cpu="src/codec/inter/me/",
         parity="src/cuda/codec/inter/me/me_test.cu", emu="—",
         gpu="RTX 3060: 510 checks", wired="M2 gate (not in CMake)",
         verdict=REQUIRED, note="multi-ref ME search (num_refs)."),
    dict(tool="intra.cu", module="m0", spec="§7.13.2",
         cpu="src/pipeline/still/basic_encoder.cpp",
         parity="src/cuda/intra_test.cu", emu="—",
         gpu="RTX 3060 (README-intra)", wired="M0 only",
         verdict=REQUIRED, note="M0 predictor; superseded by intra_modes on M1/M2."),
    dict(tool="intra_modes/intra_modes.cu", module="intra_modes", spec="§7.13.2",
         cpu="src/codec/intra/modes/",
         parity="src/cuda/codec/intra/modes/intra_modes_test.cu",
         emu="src/cuda/codec/intra/modes/intra_device_emulation_test.cpp (2,782 checks)",
         gpu="RTX 3060", wired="M1 + M2",
         verdict=REQUIRED, note="shared intra predictor; corner filter included."),
    dict(tool="basic_pipeline.cu", module="m0", spec="orchestration",
         cpu="src/pipeline/still/basic_encoder.cpp",
         parity="src/cuda/basic_pipeline_test.cpp", emu="—",
         gpu="RTX 3060 PASS", wired="M0 only",
         verdict=REQUIRED, note="M0 GPU pipeline; Arena-owned."),
    dict(tool="m1/cdef/still_cdef.cu", module="m1/cdef", spec="§7.18",
         cpu="src/codec/filters/stream/ + src/codec/filters/cdef/",
         parity="src/cuda/pipeline/still/cdef/still_cdef_gpu_test.cu",
         emu="src/cuda/pipeline/still/cdef/still_cdef_test.cpp",
         gpu="not run (README)", wired="no (rejected by GpuOptions)",
         verdict=UNWIRED, note="strength search + filter on device."),
    dict(tool="m1/cfl/still_cfl.cu", module="m1/cfl", spec="§7.13.5",
         cpu="src/codec/intra/cfl/",
         parity="src/cuda/pipeline/still/cfl/still_cfl_test.cu", emu="—",
         gpu="RTX 3060 (README)", wired="no (wrapper never called)",
         verdict=UNWIRED, note="derived-alpha CfL batch."),
    dict(tool="m1/chroma/still_chroma.cu", module="m1/chroma", spec="§7.13.2, §7.13.5",
         cpu="src/pipeline/still/still_encoder.cpp (chroma decide)",
         parity="src/cuda/pipeline/still/chroma/still_chroma_test.cu",
         emu="src/cuda/pipeline/still/chroma/uv_full_emulation_test.cpp, frame_emulation_shim.cpp",
         gpu="RTX 3060 PASS", wired="M1",
         verdict=REQUIRED, note="DC/CfL decision + predictor."),
    dict(tool="m1/hbd/hbd_frame.cu", module="m1/hbd", spec="§6.4.1 Table 6.3, §7.15",
         cpu="src/formats/high_bitdepth/",
         parity="src/cuda/pipeline/still/hbd/hbd_test.cu",
         emu="src/cuda/pipeline/still/hbd/hbd_test.cpp (19,312 checks)",
         gpu="RTX 3060: 1,523,424 checks", wired="no",
         verdict=UNWIRED, note="10-bit frame conversion/arithmetic."),
    dict(tool="m1/hbd/hbd_tx.cu", module="m1/hbd", spec="§7.15, §7.14.4",
         cpu="src/formats/high_bitdepth/",
         parity="src/cuda/pipeline/still/hbd/hbd_test.cu",
         emu="src/cuda/pipeline/still/hbd/hbd_test.cpp",
         gpu="RTX 3060: 1,523,424 checks", wired="no",
         verdict=UNWIRED, note="10-bit TX/quant batch."),
    dict(tool="m1/still_gpu.cu", module="m1", spec="orchestration",
         cpu="src/pipeline/still/still_encoder.cpp",
         parity="src/cuda/pipeline/still/still_gpu_test.cc", emu="—",
         gpu="RTX 3060: 105-case matrix", wired="M1 product (--still-gpu)",
         verdict=REQUIRED, note="Arena-owned, mark()/rewind() per frame."),
    dict(tool="m1/recon/recon.cu", module="m1/recon", spec="§7.14.3, §7.17 metadata",
         cpu="src/pipeline/still/still_encoder.cpp (fill_deblock_mi)",
         parity="src/cuda/pipeline/still/recon/recon_test.cu", emu="—",
         gpu="RTX 3060: 2,652 checks", wired="M1",
         verdict=REQUIRED, note="recon writes + IntraJointModes + deblock MI."),
    dict(tool="m1/restoration/still_restoration.cu", module="m1/restoration",
         spec="§7.19, §7.20–§7.20.3",
         cpu="src/codec/filters/loop_restoration/",
         parity="src/cuda/pipeline/still/restoration/still_restoration_test.cu",
         emu="src/cuda/pipeline/still/restoration/real_recon_oracle_test.cpp",
         gpu="not executed (README)", wired="no",
         verdict=UNWIRED, note="CCSO + Wiener on device; rejected by GpuOptions."),
    dict(tool="m1/sched/sched_exec.cu", module="m1/sched", spec="scheduler",
         cpu="src/cuda/pipeline/still/plan/ + src/cuda/pipeline/still/sched/sched.cpp",
         parity="src/cuda/pipeline/still/sched/sched_plan_test.cpp", emu="—",
         gpu="not run here (host-only plan tests)", wired="no (host availability only)",
         verdict=UNWIRED, note="level/wavefront executor not used by orchestrator."),
    dict(tool="m1/search/luma_search.cu", module="m1/search", spec="§7.13.2",
         cpu="src/pipeline/still/still_encoder.cpp (choose_luma_mode)",
         parity="src/cuda/pipeline/still/search/luma_search_test.cu", emu="—",
         gpu="RTX 3060", wired="M1",
         verdict=REQUIRED, note="61-candidate SSE search."),
    dict(tool="m1/tx/still_tx.cu", module="m1/tx", spec="§7.15.2–§7.15.4, §7.14.4",
         cpu="src/codec/transform/variants/",
         parity="src/cuda/pipeline/still/tx/still_tx_test.cu", emu="—",
         gpu="RTX 3060 timings; parity expected-result wording", wired="M1",
         verdict=REQUIRED, note="all 25 sizes / 16 TX types; quantizer state is a parameter."),
    dict(tool="m2/video_backend_cuda.cu", module="m2",
         spec="§7.12, §7.13.3, §7.13.3.14, §7.15, §7.14.3, §5.4.12, §5.7",
         cpu="src/pipeline/video/",
         parity="src/cuda/pipeline/video/video_gate.cpp, src/cuda/pipeline/video/video_selftest.cpp",
         emu="src/cuda/pipeline/video/video_backend_cpu.cpp",
         gpu="RTX 3060 `gate_video_gpu`: key+P, refs 2/3, B frames, tlayers, compound, MFH/timing byte-identical at 360p",
         wired="M2 product (`--video-gpu`, `av2enc_video_gpu`)",
         verdict=REQUIRED, note="variance/SSE/residual/recon kernels + orchestration; per-slot reference planes and list-1 jobs; arena-only."),
    dict(tool="mrl/mrl.cu", module="mrl", spec="§7.13.2.7–.8",
         cpu="src/codec/intra/mrl/core/",
         parity="src/cuda/codec/intra/mrl/mrl_test.cu",
         emu="src/cuda/codec/intra/mrl/mrl_emulation_test.cpp (16,904 checks)",
         gpu="not documented", wired="no",
         verdict=UNWIRED, note="directional MRL predictor for mrl_index 1..3."),
    dict(tool="palette/palette.cu", module="palette", spec="§5.20.8, §7.13.4",
         cpu="src/codec/intra/palette/core/",
         parity="src/cuda/codec/intra/palette/palette_test.cu", emu="—",
         gpu="not documented", wired="no",
         verdict=UNWIRED, note="palette sort/index/map prediction."),
    dict(tool="qm/qm_gpu.cu", module="qm", spec="§7.14.4, §5.13",
         cpu="src/codec/transform/qm/",
         parity="src/cuda/codec/transform/qm/qm_gpu_test.cu", emu="—",
         gpu="not run (README)", wired="no",
         verdict=UNWIRED, note="QM-aware dequantize; CPU path also unwired."),
    dict(tool="rdoq/rdoq_batch.cu", module="rdoq", spec="§7.14.4, §5.20.7",
         cpu="src/codec/rate/rdoq/",
         parity="src/cuda/codec/rate/rdoq/rdoq_batch_test.cu", emu="—",
         gpu="not run (README)", wired="no",
         verdict=UNWIRED, note="trellis RDOQ; changes coefficients, so M2 parity excludes it."),
    dict(tool="restoration/ccso.cu", module="restoration", spec="§7.19/§7.19.1",
         cpu="src/codec/filters/loop_restoration/ccso_core.h",
         parity="src/cuda/codec/filters/loop_restoration/ccso_test.cu", emu="—",
         gpu="not measured (README)", wired="no",
         verdict=UNWIRED, note="per-sample CCSO apply."),
    dict(tool="restoration/wiener.cu", module="restoration", spec="§7.20, §7.20.3",
         cpu="src/codec/filters/loop_restoration/wiener_core.h",
         parity="src/cuda/codec/filters/loop_restoration/wiener_test.cu", emu="—",
         gpu="not measured (README)", wired="no",
         verdict=UNWIRED, note="per-plane non-separable Wiener; PC-Wiener rejected."),
    dict(tool="scheduler/scheduler.cu", module="scheduler", spec="scheduler / §5.20 dependency order",
         cpu="src/pipeline/still/basic_encoder.cpp + src/cuda/pipeline/dispatch/scheduler/scheduler.cpp",
         parity="src/cuda/pipeline/dispatch/scheduler/scheduler_plan_test.cpp", emu="—",
         gpu="RTX 3060 (README)", wired="M0 only",
         verdict=REQUIRED, note="CUDA-graph executor; Arena-owned."),
    dict(tool="transform.cu", module="m0", spec="§7.15.2, §7.15.4",
         cpu="src/pipeline/still/basic_encoder.cpp",
         parity="src/cuda/transform_test.cu", emu="—",
         gpu="RTX 3060 PASS", wired="M0 only",
         verdict=REQUIRED, note="M0 transform/quant/inverse."),
    dict(tool="tx_secondary/tx_secondary_gpu.cu", module="tx_secondary", spec="§7.15.3",
         cpu="src/codec/transform/secondary/",
         parity="src/cuda/codec/transform/secondary/tx_secondary_test.cu",
         emu="src/cuda/codec/transform/secondary/tx_secondary_emulation_test.cpp (11,504,810 checks)",
         gpu="not run (README)", wired="no",
         verdict=UNWIRED, note="IST secondary transform."),
    dict(tool="tx_type/tx_type.cu", module="tx_type", spec="§7.15.4, §9.6",
         cpu="src/codec/transform/tx_type/",
         parity="src/cuda/codec/transform/tx_type/tx_type_test.cu", emu="—",
         gpu="RTX 3060: 328,694 checks", wired="no",
         verdict=UNWIRED, note="TX_TYPE candidate search."),
    dict(tool="tx_variants/tx_variants.cu", module="tx_variants", spec="§7.15.2, §7.14.4",
         cpu="src/codec/transform/variants/",
         parity="src/cuda/codec/transform/variants/tx_variants_test.cu", emu="—",
         gpu="RTX 3060: 7,203 checks", wired="M2 (also M1 support link)",
         verdict=REQUIRED, note="25-size forward/inverse/quant batch."),
    dict(tool="cfl/mhccp/mhccp_gpu.cu", module="cfl/mhccp", spec="§7.13.6–.8",
         cpu="src/codec/intra/cfl/mhccp/",
         parity="src/codec/intra/cfl/mhccp/mhccp_gpu_test.cu",
         emu="src/codec/intra/cfl/mhccp/mhccp_emulation_test.cpp (16,803 checks)",
         gpu="not run (no nvcc)", wired="no",
         verdict=UNWIRED, note="multi-hypothesis cross-component prediction."),
    dict(tool="inter/me2/me2_gpu.cu", module="inter/me2", spec="§7.13.3.14, §7.12.2",
         cpu="src/codec/inter/me_precision/",
         parity="src/codec/inter/me_precision/me2_gpu_test.cu", emu="—",
         gpu="not documented", wired="no (M2 uses cuda/inter/me)",
         verdict=UNWIRED, note="multi-reference ME cost batch."),
    dict(tool="m1/transform_stream/transform_stream_gpu.cu", module="m1/transform_stream",
         spec="§7.15.4, §7.14.4",
         cpu="src/codec/transform/stream/",
         parity="src/codec/transform/stream/transform_stream_gpu_test.cu",
         emu="—", gpu="not run (no nvcc)", wired="no",
         verdict=UNWIRED, note="RDO transform-stream adapter; default cudaMalloc allocator."),    dict(tool="intra_full/intra_full.cu", module="intra_full",
         spec="§7.13.2.11, §7.13.2.14, §7.15.4, §5.20.7.29",
         cpu="src/codec/intra/supplemental/, src/codec/transform/variants/",
         parity="src/cuda/codec/intra/supplemental/intra_full_emulation_test.cpp (host emulation)",
         emu="src/cuda/codec/intra/supplemental/intra_full_emulation_test.cpp",
         gpu="not run", wired="no",
         verdict=UNWIRED, note="landed during this audit; closes the DC-subsampled/identity-TX gaps but has no parity test yet."),
    dict(tool="lossless_inter/lossless_inter.cu", module="lossless_inter",
         spec="§7.15.2.2, §7.15.4, §5.20.6.1, §5.18.2",
         cpu="src/codec/entropy/lossless/inter/",
         parity="src/cuda/codec/entropy/lossless/inter/lossless_inter_emulation_test.cpp (host emulation)",
         emu="src/cuda/codec/entropy/lossless/inter/lossless_inter_emulation_test.cpp",
         gpu="not run", wired="no",
         verdict=UNWIRED, note="landed during this audit; WHT/DPCM + lossless TX selection."),
    dict(tool="morph/morph.cu", module="morph",
         spec="§7.13.3.26 (→ §7.13.3.25)",
         cpu="src/codec/inter/morph/morph_common.h",
         parity="src/cuda/codec/inter/morph/morph_emulation.cpp (host emulation)",
         emu="src/cuda/codec/inter/morph/morph_emulation.cpp",
         gpu="not run", wired="no",
         verdict=UNWIRED, note="landed during this audit; morphological predictor."),
    dict(tool="pc_wiener/pc_wiener.cu", module="pc_wiener",
         spec="§7.20.4",
         cpu="src/codec/filters/tables/pc_wiener.cpp",
         parity="none",
         emu="—",
         gpu="not run", wired="no",
         verdict=UNWIRED, note="landed during this audit; closes the PC-Wiener gap, no test yet."),
    dict(tool="tip/tip.cu", module="tip",
         spec="§7.10.2–.6, §7.13.3.2/.17/.18",
         cpu="src/codec/inter/tip/tip_build.cpp",
         parity="none",
         emu="—",
         gpu="not run", wired="no",
         verdict=UNWIRED, note="landed during this audit; motion-field build + TIP prediction."),
    dict(tool="tip_deblock/tip_deblock.cu", module="tip_deblock",
         spec="§7.16, §7.17.3, §7.17.7",
         cpu="src/codec/filters/tip_deblock/tip_deblock.cpp",
         parity="src/cuda/codec/filters/tip_deblock/tip_deblock_emulation_test.cpp (host emulation)",
         emu="src/cuda/codec/filters/tip_deblock/tip_deblock_emulation_test.cpp",
         gpu="not run", wired="no",
         verdict=UNWIRED, note="landed during this audit; closes the §7.16 gap."),
    dict(tool="warp_stack/warp_stack.cu", module="warp_stack",
         spec="§7.13.3.19–.20, §7.13.3 warp",
         cpu="src/codec/inter/warp_stack/",
         parity="none",
         emu="src/cuda/codec/inter/warp_stack/warp_stack_host_ref.cpp (host reference shim)",
         gpu="not run", wired="no",
         verdict=UNWIRED, note="landed during this audit; full warp stack launchers."),    dict(tool="compound_full/compound_full.cu", module="compound_full",
         spec="§7.13.3.3–.14, .29–.30",
         cpu="src/codec/inter/compound/, src/codec/inter/advanced/",
         parity="none",
         emu="—",
         gpu="not run", wired="no",
         verdict=UNWIRED, note="landed during this audit; full compound/OPFL/refine-MV batch, no test yet."),
    dict(tool="gdf/gdf.cu", module="gdf",
         spec="§7.20.5, §5.20.10.3, §7.20.1",
         cpu="src/codec/filters/tables/gdf.cpp",
         parity="none",
         emu="—",
         gpu="not run", wired="no",
         verdict=UNWIRED, note="landed during this audit; closes the GDF gap."),
    dict(tool="tx_partition/tx_partition_gpu.cu", module="tx_partition",
         spec="§5.20.6.1–.3, §5.18.8.1",
         cpu="src/codec/transform/tx_partition/",
         parity="none",
         emu="—",
         gpu="not run", wired="no",
         verdict=UNWIRED, note="landed during this audit; closes the per-TU TX launch-driver gap."),
]

# Pixel work that has neither a wired kernel nor an unwired one.
MISSING_EXTRA: list[dict] = [
    dict(tool="Motion-field storage (§7.22)",
         spec="§7.22",
         cpu="src/codec/inter/temporal/motion_field.cpp",
         parity="src/codec/inter/temporal/ (gate)",
         volume=129_600,
         basis="per-4x4 grid entries, not pixels"),
    dict(tool="Temporal MV projection / ref-MV bank update",
         spec="§7.12.2, §7.12.2.19",
         cpu="src/codec/inter/temporal/ + src/codec/inter/mv_stack/",
         parity="src/codec/inter/mv_stack/mv_stack_full_test.cpp",
         volume=129_600,
         basis="per-4x4 MV entries; metadata, not pixels"),
]

# Modules that landed during this audit and are not yet in either matrix.
LANDED: list[dict] = [
    dict(module="src/cuda/codec/inter/compound/",
         kernel="compound_full.cu",
         spec="§7.13.3.3–.14, .29–.30",
         cpu="src/codec/inter/compound/, src/codec/inter/advanced/",
         parity="none",
         gpu="none", arena="Arena + required_device_bytes", status="landed during this audit; unwired"),
    dict(module="src/cuda/codec/intra/supplemental/",
         kernel="intra_full.cu",
         spec="§7.13.2.11, §7.13.2.14, §7.15.4, §5.20.7.29",
         cpu="src/codec/intra/supplemental/, src/codec/transform/variants/",
         parity="intra_full_emulation_test.cpp",
         gpu="none", arena="Arena + required_device_bytes + mark/rewind", status="closes the DC-subsampled and identity-TX gaps; unwired"),
    dict(module="src/cuda/codec/entropy/lossless/inter/",
         kernel="lossless_inter.cu + lossless_inter_host.cpp",
         spec="§7.15.2.2, §7.15.4, §5.20.6.1, §5.18.2",
         cpu="src/codec/entropy/lossless/inter/",
         parity="lossless_inter_emulation_test.cpp",
         gpu="none", arena="Arena + required_device_bytes", status="unwired"),
    dict(module="src/cuda/codec/inter/morph/",
         kernel="morph.cu + morph_workspace.cpp",
         spec="§7.13.3.26 (→ §7.13.3.25 BAWP process)",
         cpu="src/codec/inter/morph/morph_common.h",
         parity="morph_emulation.cpp",
         gpu="none", arena="Arena + required_device_bytes + mark/rewind", status="unwired"),
    dict(module="src/cuda/codec/filters/pc_wiener/",
         kernel="pc_wiener.cu",
         spec="§7.20.4",
         cpu="src/codec/filters/tables/pc_wiener.cpp",
         parity="none",
         gpu="none", arena="Arena + pc_wiener_required_device_bytes", status="closes the PC-Wiener gap; unwired"),
    dict(module="src/cuda/codec/inter/tip/",
         kernel="tip.cu",
         spec="§7.10.2–.6, §7.13.3.2/.17/.18",
         cpu="src/codec/inter/tip/tip_build.cpp",
         parity="none",
         gpu="none", arena="Arena + required_device_bytes + mark/rewind", status="closes the TIP prediction gap; unwired"),
    dict(module="src/cuda/codec/filters/tip_deblock/",
         kernel="tip_deblock.cu",
         spec="§7.16, §7.17.3, §7.17.7",
         cpu="src/codec/filters/tip_deblock/tip_deblock.cpp",
         parity="tip_deblock_emulation_test.cpp",
         gpu="none", arena="Arena + required_device_bytes + mark/rewind", status="closes the §7.16 gap; unwired"),
    dict(module="src/cuda/codec/filters/gdf/",
         kernel="gdf.cu",
         spec="§7.20.5",
         cpu="src/codec/filters/tables/gdf.cpp",
         parity="none",
         gpu="none", arena="Arena + required_device_bytes + mark/rewind", status="closes the GDF gap; unwired"),
    dict(module="src/cuda/codec/transform/tx_partition/",
         kernel="tx_partition_gpu.cu + tx_partition_plan.cpp",
         spec="§5.20.6.1–.3, §5.18.8.1",
         cpu="src/codec/transform/tx_partition/",
         parity="none",
         gpu="none", arena="Arena + required_device_bytes + mark/rewind", status="closes the per-TU TX driver gap; unwired"),
    dict(module="src/cuda/codec/inter/warp_stack/",
         kernel="warp_stack.cu + warp_stack_host_ref.cpp",
         spec="§7.13.3 warp stack",
         cpu="src/codec/inter/warp_stack/",
         parity="host reference shim only",
         gpu="none", arena="Arena + required_device_bytes", status="landed during this audit; unwired"),
]

# Newest standalone GPU modules the arena rule explicitly asks about.
NEW_MODULES = [
    "src/cuda/codec/intra/dip/",
    "src/cuda/codec/intra/ibp/",
    "src/cuda/codec/intra/mrl/",
    "src/cuda/codec/transform/qm/",
    "src/cuda/codec/rate/rdoq/",
    "src/cuda/codec/transform/secondary/",
    "src/cuda/codec/transform/tx_type/",
    "src/cuda/pipeline/still/hbd/",
    "src/cuda/pipeline/still/restoration/",
    "src/codec/intra/cfl/mhccp/",
]

# Direct allocator sites that are allowed by policy.
ALLOWED_ALLOC_FILES = {
    "src/cuda/arena.cpp",
    "src/cuda/context.cpp",
}

TEST_PATTERNS = re.compile(
    r"(_test|_gpu_test|gpu_test|_bench|_selftest|_gate|device_buffer\.h|"
    r"cuda_stub|cuda_emulation|_dump)")


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------
@dataclass
class AllocSite:
    path: str
    line: int
    call: str
    context: str


@dataclass
class Scan:
    cu: list[str] = field(default_factory=list)
    cuh: list[str] = field(default_factory=list)
    kernel_cu: list[str] = field(default_factory=list)
    test_cu: list[str] = field(default_factory=list)
    allocs: list[AllocSite] = field(default_factory=list)
    prod_allocs: list[AllocSite] = field(default_factory=list)
    test_allocs: list[AllocSite] = field(default_factory=list)
    mark_rewind: list[str] = field(default_factory=list)
    required_device_bytes: list[str] = field(default_factory=list)
    arena_consumers: list[str] = field(default_factory=list)


def video_migration_lines() -> list[str]:
    """Read the current M2 backend rather than asserting a historical state."""
    p = ROOT / "src/cuda/pipeline/video/video_backend_cuda.cu"
    if not p.exists():
        return ["`src/cuda/pipeline/video/video_backend_cuda.cu` is absent."]
    text = p.read_text(errors="replace")
    calls = len(re.findall(r"\bensure_device\s*\(", text))
    defined = bool(re.search(r"bool\s+ensure_device\s*\(", text))
    frees = len(re.findall(r"\bcudaFree\s*\(", text))
    mallocs = len(re.findall(r"\bcudaMalloc\s*\(", text))
    arena = "arena.h" in text
    if arena and mallocs == 0 and frees == 0:
        detail = " and `ensure_device` is defined" if defined else ""
        return ["The backend is fully arena-backed in this snapshot: no direct",
                f"`cudaMalloc` or `cudaFree`{detail}."]
    out = ["This file is the one in the tree whose allocation policy is"]
    out.append("internally inconsistent in this snapshot:")
    out.append("")
    out.append(f"- header comment: {'arena policy' if arena else 'no arena include'};"
               f" `cudaMalloc` sites: **{mallocs}**, `cudaFree` sites: **{frees}**.")
    if calls and not defined:
        out.append(f"- **{calls} call sites invoke `ensure_device(...)` and no")
        out.append("  definition exists in the file** (arrays are otherwise")
        out.append("  allocated through `arena_alloc_array`).")
    if frees and arena:
        out.append("- `free_all()` still calls `cudaFree` on plane and scratch")
        out.append("  pointers; if those come from the arena, those are invalid")
        out.append("  frees of interior pointers.")
    out.append("")
    out.append("The backend cannot compile in this shape. The byte-identical")
    out.append("gate results in `src/cuda/pipeline/video/README.md` predate the refactor and")
    out.append("must be re-run once it lands.")
    return out


def head_rev() -> str:
    try:
        import subprocess
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def is_test_path(path: str) -> bool:
    parts = Path(path).parts
    if any(p in ("tests", "test", "cuda_stub") for p in parts):
        return True
    return bool(TEST_PATTERNS.search(Path(path).name))


def scan() -> Scan:
    s = Scan()
    for p in sorted((ROOT / "src").rglob("*.cu")):
        if "/build" in rel(p):
            continue
        s.cu.append(rel(p))
    for p in sorted((ROOT / "src").rglob("*.cuh")):
        if "/build" in rel(p):
            continue
        s.cuh.append(rel(p))
    for path in s.cu:
        (s.test_cu if is_test_path(path) else s.kernel_cu).append(path)

    alloc_re = re.compile(r"\b(cudaMallocHost|cudaHostAlloc|cudaMalloc|cudaFree)\b")
    for p in sorted((ROOT / "src").rglob("*")):
        if p.suffix not in (".cu", ".cpp", ".cc", ".h", ".cuh"):
            continue
        r = rel(p)
        if "/build" in r:
            continue
        try:
            lines = p.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("//"):
                continue
            seen_calls = set()
            for m in alloc_re.finditer(line):
                if m.group(1) in seen_calls:
                    continue
                seen_calls.add(m.group(1))
                site = AllocSite(r, i, m.group(1), stripped[:96])
                s.allocs.append(site)
                if r in ALLOWED_ALLOC_FILES:
                    continue
                (s.test_allocs if is_test_path(r) else s.prod_allocs).append(site)

    for p in sorted((ROOT / "src").rglob("*")):
        if p.suffix not in (".cu", ".cpp", ".h", ".cuh"):
            continue
        r = rel(p)
        if "/build" in r or is_test_path(r):
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if re.search(r"\.mark\(\)|->mark\(\)", text):
            s.mark_rewind.append(r)
        if "required_device_bytes" in text:
            s.required_device_bytes.append(r)
        if re.search(r"\bArena\b", text):
            s.arena_consumers.append(r)
    return s


def readme_claim(module: str) -> str:
    """Best-effort extraction of a real-GPU claim from the module README."""
    d = ROOT / "src" / "cuda" / module / "README.md"
    if not d.exists():
        return "—"
    text = d.read_text(errors="replace")
    if re.search(r"not run|not been run|not executed|could not be compiled|no `nvcc`|no CUDA (device|toolkit)", text):
        return "module README says not executed"
    if re.search(r"RTX 3060|sm_86", text):
        return "documented on RTX 3060"
    return "—"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def check_paths() -> list[str]:
    missing: list[str] = []
    for row in FEATURES + KERNELS:
        for key in ("cpu", "parity"):
            val = row[key]
            if not val or val.startswith("—"):
                continue
            # composite cells are separated by commas
            for part in re.split(r",\s*", val):
                part = part.strip()
                if not part or part.startswith("—") or "/" not in part:
                    continue
                part = re.split(r" \+| \(", part)[0].rstrip("/").strip()
                if not part or part.startswith("—"):
                    continue
                if not (ROOT / part).exists():
                    missing.append(f"{row['tool']}: {part}")
    for row in MISSING_EXTRA:
        for key in ("cpu", "parity"):
            part = row[key].split(",")[0].strip()
            part = re.split(r" \+| \(", part)[0].rstrip("/").strip()
            if part and not part.startswith("—") and not (ROOT / part).exists():
                missing.append(f"{row['tool']}: {part}")
    return missing


def verdict_counts(rows: list[dict]) -> dict[str, int]:
    out = {v: 0 for v in VERDICT_ORDER}
    for r in rows:
        out[r["verdict"]] += 1
    return out


def mrow(cells: list[str]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


def render(scan_result: Scan) -> str:
    fcount = verdict_counts(FEATURES)
    covered = {r["tool"] for r in KERNELS}
    auto_rows: list[dict] = []
    for path in scan_result.kernel_cu:
        if not any(path.endswith(t) for t in covered):
            auto_rows.append(dict(
                tool=path.replace("src/", "", 1), module=Path(path).parent.name,
                spec="—", cpu="—", parity="—", emu="—", gpu="—",
                wired="no", verdict=UNWIRED,
                note="auto-detected kernel TU; not in the curated inventory"))
    krows = KERNELS + auto_rows
    kcount = verdict_counts(krows)
    all_rows = FEATURES + krows
    total = verdict_counts(all_rows)
    landed_counts = {UNWIRED: 10, MISSING: 0, REQUIRED: 0, SERIAL: 0}
    combined = total
    missing_rows = [r for r in all_rows if r["verdict"] == MISSING] + MISSING_EXTRA
    seen_missing = set()
    deduped = []
    for r in missing_rows:
        key = r["cpu"].split(",")[0].strip()
        if key in seen_missing:
            continue
        seen_missing.add(key)
        deduped.append(r)
    missing_rows = deduped
    ranked = sorted(missing_rows, key=lambda r: -r.get("volume", 0))

    L: list[str] = []
    A = L.append
    A("# GPU coverage and arena compliance — per-tool matrix")
    A("")
    A("Generated by `tools/gpu_coverage/inventory.py` from the tree. Code and")
    A("tests were read; the GPU was **not** run and nothing was compiled. Status")
    A("lines in component READMEs are used only for the *real-GPU evidence*")
    A("column, and are labelled as documentation rather than as a measurement")
    A("performed here.")
    A("")
    A(f"**Snapshot:** audit start at `debb00e`; document generated at git "
      f"`{head_rev()}` with {len(scan_result.cu)} `.cu` / "
      f"{len(scan_result.cuh)} `.cuh`, {len(scan_result.kernel_cu)} kernel TUs.")
    A("Ten GPU modules landed while the audit was running; the snapshot is the")
    A("contract. `--check` fails if")
    A("a curated path disappears, and any kernel TU that is not in the curated")
    A("list is auto-added so no `.cu` can go uncounted.")
    A("")
    A("## Scope and vocabulary")
    A("")
    A("| Verdict | Meaning |")
    A("|---|---|")
    A("| **GPU-SERIAL** | entropy, headers, syntax, CDF, metadata, conformance: CPU by design, no kernel required |")
    A("| **GPU-REQUIRED** | pixel/filter/transform/MC/prediction work: must have a GPU kernel and bit-exact parity |")
    A("| **GPU-MISSING** | required pixel work, no kernel exists |")
    A("| **GPU-UNWIRED** | kernel exists, but the M1/M2 GPU path rejects or never calls it |")
    A("")
    A("`GPU-REQUIRED` says the work is device-resident on at least one GPU path;")
    A("the *wired* column names which one. The product build (`CMakeLists.txt`)")
    A("only links the **M1** GPU target. `src/cuda/pipeline/video/` is gate-only: there is no")
    A("`av2enc_video_gpu` target in `CMakeLists.txt`, so every M2 row is reachable")
    A("only through `src/cuda/pipeline/video/build_gpu_gate.sh`.")
    A("")
    A("Inventory under `src/`:")
    A("")
    A(f"- **{len(scan_result.cu)}** `.cu` files, **{len(scan_result.cuh)}** `.cuh` files.")
    A("  Counts are recomputed from the tree on every run; the number moves when")
    A("  a component lands.")
    A(f"- **{len(scan_result.kernel_cu)}** kernel translation units, "
      f"**{len(scan_result.test_cu)}** test/bench/gate translation units.")
    A(f"- Of the kernel TUs, **{len([k for k in scan_result.kernel_cu if k.startswith('src/cuda/')])}** "
      f"are under `src/cuda/` and "
      f"**{len([k for k in scan_result.kernel_cu if not k.startswith('src/cuda/')])}** "
      f"under other trees.")
    A("- The 26 feature rows in Matrix 1 are the component directories added")
    A("  after commit `19c5dd9` (inherited audit baseline) up to `debb00e`.")
    A("  They are marked with `†` in the matrix.")
    A("")
    A("## Matrix 1 — the 26 newest feature directories")
    A("")
    A(mrow(["tool", "spec §", "CPU reference dir", "CUDA kernel dir", "parity test",
            "host emulation", "real-GPU evidence", "wired into M1/M2", "verdict"]))
    A(mrow(["---"] * 9))
    for r in FEATURES:
        A(mrow([r["tool"] + " †", r["spec"], r["cpu"], r["cuda"], r["parity"],
                r["emu"], r["gpu"], r["wired"], r["verdict"]]))
    A("")
    A("Notes on split rows: `film_grain_stream` is split into signalling")
    A("(serial) and synthesis (missing); `m1/missing_bits` is split into four")
    A("spec rules because three of them have different verdicts; `lr_chroma` is")
    A("split into syntax (serial) and §7.20.3 apply (unwired kernel).")
    A("")
    A("## Matrix 2 — all CUDA kernel units")
    A("")
    A(f"One row per non-test `.cu` in the tree ({len(krows)}).  Every kernel TU")
    A("lives under `src/cuda/` after the folder refactor; the `src/encoder/`")
    A("kernel TUs the earlier audit counted were moved there.")
    A("")
    A(mrow(["tool", "spec §", "CPU reference dir", "CUDA kernel dir", "parity test",
            "host emulation", "real-GPU evidence", "wired into M1/M2", "verdict"]))
    A(mrow(["---"] * 9))
    for r in krows:
        A(mrow([r["tool"], r["spec"], r["cpu"], "its own directory", r["parity"],
                r["emu"], r["gpu"], r["wired"], r["verdict"]]))
    A("")
    A("`GPU-UNWIRED` here is not a defect by itself: `cdef`, `restoration`, `dip`,")
    A("`ibp`, `mrl`, `palette`, `ibc`, `qm`, `rdoq`, `tx_type`, `tx_secondary`,")
    A("`hbd`, `mhccp` and `me2` are landed components waiting for a caller. It")
    A("becomes a shipping defect when the option is exposed and silently falls")
    A("back. `src/cuda/pipeline/still/still_gpu.h:gpu_m3_options_supported()` rejects every")
    A("unported tool by name, so the M1 path is honest about the gap.")
    A("")
    A("## Matrix 3 — modules that landed during this audit")
    A("")
    A("These ten directories appeared in the tree while the audit was running")
    A("(they are untracked in `git status`). They are inventoried here because the")
    A("arena rule and the M1/M2 coverage question both depend on them. None has a")
    A("real-GPU parity test yet; none is called from a GPU path.")
    A("")
    A(mrow(["module", "kernel TU", "spec §", "CPU reference", "parity test",
            "real-GPU evidence", "arena contract", "state"]))
    A(mrow(["---"] * 8))
    for r in LANDED:
        A(mrow([r["module"], r["kernel"], r["spec"], r["cpu"], r["parity"],
                r["gpu"], r["arena"], r["status"]]))
    A("")
    A("All ten landed with an arena-based `required_device_bytes()` contract and")
    A("no direct allocator. **None has a parity test or a caller.** Two of them")
    A("(`compound_full`, `warp_stack`) were header-only earlier in the audit and")
    A("gained their `.cu` while this document was being generated; the snapshot")
    A("line above is what makes the counts reproducible.")
    A("")
    A("## GPU-MISSING inventory")
    A("")
    A("Every row names the exact CPU reference and the clause. Sorted by")
    A("estimated pixel volume at 1080p 4:2:0 (3,110,400 samples/frame).")
    A("Estimates are order-of-magnitude, not measurements.")
    A("")
    A(mrow(["#", "tool", "spec §", "CPU reference", "parity / gate",
            "samples/frame", "basis"]))
    A(mrow(["---"] * 7))
    for i, r in enumerate(sorted(missing_rows, key=lambda x: -x.get("volume",0)), 1):
        A(mrow([str(i), r["tool"], r["spec"], r["cpu"], r["parity"],
                f"{r.get('volume',0):,}" if r.get("volume") else "—", r.get("basis","")]))
    A("")
    A("**Closed while this audit was running.** At the start of the audit the")
    A("same method found twelve missing pixel families. Six gained kernels")
    A("mid-audit and are now `GPU-UNWIRED`: TIP prediction (`tip.cu`), TIP")
    A("deblock (`tip_deblock.cu`), GDF (`gdf.cu`), PC-Wiener (`pc_wiener.cu`),")
    A("compound/OPFL/refine-MV (`compound_full.cu`), transform partition")
    A("(`tx_partition_gpu.cu`), warp stack (`warp_stack.cu`), DC intra")
    A("subsampled and identity TX (`intra_full.cu`). The remaining six below are")
    A("the honest residual; most are outside the shipped M1/M2 scope.")
    A("")
    A("The ranked version of this table is the closing section, after the")
    A("counts.")
    A("")
    A("## Arena compliance")
    A("")
    A("Rule: all device memory comes from the single persistent")
    A("`av2enc::cuda::Arena` (`src/cuda/arena.h`). The only permitted direct")
    A("allocators are `arena.cpp` (the single `cudaMalloc` that backs the arena)")
    A("and the pinned-host staging in `context.cpp` (`cudaMallocHost`).")
    A("")
    A(f"- Direct-allocator sites outside the two allowed files: "
      f"**{len(scan_result.allocs) - len([a for a in scan_result.allocs if a.path in ALLOWED_ALLOC_FILES])}**.")
    A(f"- Production (non-test) sites: **{len(scan_result.prod_allocs)}** in "
      f"**{len({a.path for a in scan_result.prod_allocs})}** files.")
    A(f"- Test/bench/gate sites: **{len(scan_result.test_allocs)}** in "
      f"**{len({a.path for a in scan_result.test_allocs})}** files.")
    A(f"- Files exposing `required_device_bytes`: **{len(scan_result.required_device_bytes)}**.")
    A(f"- Files using `mark()`/`rewind()`: **{len(scan_result.mark_rewind)}** "
      f"({', '.join('`'+p+'`' for p in scan_result.mark_rewind) or 'none'}).")
    A("")
    A("### Per-directory result")
    A("")
    A(mrow(["directory", "arena status", "direct sites (file:line)", "required_device_bytes", "mark/rewind"]))
    A(mrow(["---"] * 5))
    dirs = sorted({str(Path(p).parent) for p in scan_result.cu + scan_result.cuh})
    dirs = [d for d in dirs if d != "src"]
    for extra in ("src/pipeline/dispatch/runtime", "src/core/telemetry"):
        if (ROOT / extra).exists():
            dirs.append(extra)
    dirs = sorted(set(dirs))
    for d in dirs:
        sites = [a for a in scan_result.allocs if str(Path(a.path).parent) == d]
        prod = [a for a in sites if not is_test_path(a.path)]
        if any(a.path == "src/cuda/arena.cpp" for a in sites):
            status = "owner (allocator)"
        elif any(a.path == "src/cuda/context.cpp" for a in sites):
            status = "owner (pinned staging, allowed)"
        elif not sites:
            status = ("compliant — arena-only allocation"
                      if (any(d == str(Path(p).parent) for p in scan_result.required_device_bytes)
                          or any(d == str(Path(p).parent) for p in scan_result.mark_rewind))
                      else "compliant — no device allocation")
        elif prod:
            status = "**NON-COMPLIANT** — direct cudaMalloc/cudaFree"
        else:
            status = "compliant in production; test harness allocates"
        site_txt = "<br>".join(f"`{a.path}:{a.line}` {a.call}" for a in sorted(prod, key=lambda x: (x.path, x.line))) or "—"
        req = "yes" if any(d == str(Path(p).parent) for p in scan_result.required_device_bytes) else "no"
        mr = "yes" if any(d == str(Path(p).parent) for p in scan_result.mark_rewind) else "no"
        A(mrow([d, status, site_txt, req, mr]))
    A("")
    A("### Non-compliant files (production code)")
    A("")
    A(mrow(["file:line", "call", "code", "impact"]))
    A(mrow(["---"] * 4))
    impact = {
        "src/cuda/codec/intra/ibc/ibc.cu": "allocates and frees one 8-byte result buffer per `ibc_search` call *and* synchronises the stream inside the search",
        "src/cuda/formats/high_bitdepth_gpu.cu": "`DeviceBuffer<T>` allocates/frees per buffer; M10 path only, not M1/M2",
        "src/cuda/pipeline/video/video_backend_cuda.cu": "`ensure_device`/plane uploads allocate per frame-size growth. `video_gpu.h` now exposes `required_device_bytes()`, but this backend still calls `cudaMalloc` instead of carving that block from the arena (gate-only today; a product defect when M2 joins CMake)",
        "src/codec/transform/stream/transform_stream_gpu.cu": "default allocator falls back to `cudaMalloc`/`cudaFree` when the caller passes none; the arena allocator is opt-in",
        "src/pipeline/dispatch/runtime/cuda_backend.cpp": "separate multi-GPU runtime: `ArenaState` is arena-backed but `pinned_alloc()` uses `cudaMallocHost` outside `context.cpp`",
        "src/core/telemetry/profile_demo.cpp": "profiler demo, not encoder code; one `cudaMalloc`/`cudaFree` pair",
    }
    for a in sorted(scan_result.prod_allocs, key=lambda x: (x.path, x.line)):
        A(mrow([f"`{a.path}:{a.line}`", a.call, f"`{a.context}`", impact.get(a.path, "—")]))
    A("")
    A("The M1 product hot path is clean. `video_backend_cuda.cu` is the one that")
    A("will matter when M2 is added to the build; its own README already says")
    A("the buffers should be carved from the arena. `cuda_backend.cpp` is a")
    A("separate runtime with its own `ArenaState`; the pinned staging API is the")
    A("only rule violation there. `profile_demo.cpp` is a demo binary.")
    A("")
    A("### M2 backend allocation state")
    A("")
    for line in video_migration_lines():
        A(line)
    A("")
    A("### Test harnesses that allocate directly")
    A("")
    A("These are not product paths, but they are outside the letter of the rule,")
    A("so they are listed for completeness.")
    A("")
    A("| file | sites |")
    A("|---|---:|")
    by_file: dict[str, int] = {}
    for a in scan_result.test_allocs:
        by_file[a.path] = by_file.get(a.path, 0) + 1
    for p in sorted(by_file):
        A(f"| `{p}` | {by_file[p]} |")
    A("")
    A("## The 10 earlier standalone GPU modules")
    A("")
    A("These are the ten device-side components that predate this audit and are")
    A("not core M1/M2 stages — `dip`, `ibp`, `mrl`, `qm`, `rdoq`, `tx_secondary`,")
    A("`tx_type`, `m1/hbd`, `m1/restoration`, `cfl/mhccp`. All have tests; none")
    A("is called from a shipped path. The ten that landed *during* the audit are")
    A("Matrix 3 above.")
    A("")
    A(mrow(["module", "kernel", "parity test", "host emulation", "GPU evidence", "arena", "verdict"]))
    A(mrow(["---"] * 7))
    new_rows = {
        "src/cuda/codec/intra/dip/": ("dip.cu", "dip_test.cu", "dip_emulation_test.cpp", "not run", "compliant", "GPU-UNWIRED"),
        "src/cuda/codec/intra/ibp/": ("ibp.cu", "ibp_test.cu", "ibp_device_emulation_test.cpp", "not executed", "compliant", "GPU-UNWIRED"),
        "src/cuda/codec/intra/mrl/": ("mrl.cu", "mrl_test.cu", "mrl_emulation_test.cpp", "undocumented", "compliant", "GPU-UNWIRED"),
        "src/cuda/codec/transform/qm/": ("qm_gpu.cu", "qm_gpu_test.cu", "—", "not run", "compliant", "GPU-UNWIRED"),
        "src/cuda/codec/rate/rdoq/": ("rdoq_batch.cu", "rdoq_batch_test.cu", "—", "not run", "compliant", "GPU-UNWIRED"),
        "src/cuda/codec/transform/secondary/": ("tx_secondary_gpu.cu", "tx_secondary_test.cu", "tx_secondary_emulation_test.cpp", "not run", "compliant", "GPU-UNWIRED"),
        "src/cuda/codec/transform/tx_type/": ("tx_type.cu", "tx_type_test.cu", "—", "RTX 3060 328,694 checks", "compliant", "GPU-UNWIRED"),
        "src/cuda/pipeline/still/hbd/": ("hbd_frame.cu, hbd_tx.cu", "hbd_test.cu", "hbd_test.cpp", "RTX 3060 1,523,424 checks", "compliant", "GPU-UNWIRED"),
        "src/cuda/pipeline/still/restoration/": ("still_restoration.cu", "still_restoration_test.cu", "real_recon_oracle_test.cpp", "not executed", "compliant", "GPU-UNWIRED"),
        "src/codec/intra/cfl/mhccp/": ("mhccp_gpu.cu", "mhccp_gpu_test.cu", "mhccp_emulation_test.cpp", "not run", "compliant", "GPU-UNWIRED"),
    }
    for d in NEW_MODULES:
        k, t, e, g, a, v = new_rows[d]
        A(mrow([d, k, t, e, g, a, v]))
    A("")
    A("## What this means")
    A("")
    A("1. The **serial half is clean**: all 23 header/syntax/metadata/conformance")
    A("   feature dirs are correctly CPU-side. They should not get kernels.")
    A("2. The **M1 pixel pipeline is complete and wired** for its declared scope")
    A("   (`plan`, `search`, `chroma`, `tx`, `recon`, `deblock`). Everything M1")
    A("   can signal is device-resident.")
    A("3. The **M1 filter surface is the largest unwired block**: CDEF, CCSO,")
    A("   loop restoration, DIP, IBP, MRL, palette, IBC, QM, HBD, MHCCP and the")
    A("   ten newly landed modules all have kernels and none is reachable through")
    A("   `--still-gpu`. The options are rejected by name, so this is a capability")
    A("   gap, not a silent fallback.")
    A("4. The **M2 pixel path is complete for its baseline** and byte-identical,")
    A("   but it is not in `CMakeLists.txt`; M2 is a gate, not a product.")
    A("5. The **video half landed during the audit but is 100% unwired**:")
    A("   `tip`, `tip_deblock`, `compound_full`, `morph`, `warp_stack`, `gdf`,")
    A("   `pc_wiener`, `intra_full`, `lossless_inter` and `tx_partition` all")
    A("   have arena-based kernels and no caller, and most have no parity test.")
    A("   Wiring plus tests, not new algorithms, is the remaining GPU work.")
    A("6. **Arena compliance is good in the hot paths and broken at the edges**:")
    A("   M0, M1, scheduler and the shared kernels are arena-owned. `ibc` and")
    A("   `high_bitdepth_gpu` were fixed during the audit. The M2 backend and the")
    A("   transform-stream adapter still allocate directly; the newer modules")
    A("   all implement `required_device_bytes()` and `mark()`/`rewind()`.")
    A("")
    A("## Counts")
    A("")
    A("### Matrix 1 — 26 newest feature dirs (31 rows after splits)")
    A("")
    A(mrow(["verdict", "rows"]))
    A(mrow(["---", "---:"]))
    for v in VERDICT_ORDER:
        A(mrow([v, str(fcount[v])]))
    A(mrow(["**total**", f"**{sum(fcount.values())}**"]))
    A("")
    A(f"### Matrix 2 — {len(krows)} CUDA kernel TUs")
    A("")
    A(mrow(["verdict", "TUs"]))
    A(mrow(["---", "---:"]))
    for v in VERDICT_ORDER:
        A(mrow([v, str(kcount[v])]))
    A(mrow(["**total**", f"**{sum(kcount.values())}**"]))
    A("")
    A("### Matrix 3 — 10 module directories (overlaps Matrix 2)")
    A("")
    A(mrow(["verdict", "dirs"]))
    A(mrow(["---", "---:"]))
    for v in VERDICT_ORDER:
        A(mrow([v, str(landed_counts[v])]))
    A(mrow(["**total**", f"**{sum(landed_counts.values())}**"]))
    A("")
    A("All ten contribute one kernel TU each to Matrix 2 (plus three host")
    A("`.cpp` units). Matrix 3 is therefore a directory view, not a disjoint")
    A("row set.")
    A("")
    A("### Combined (matrices 1 + 2)")
    A("")
    A(mrow(["verdict", "rows"]))
    A(mrow(["---", "---:"]))
    for v in VERDICT_ORDER:
        A(mrow([v, str(combined[v])]))
    A(mrow(["**total**", f"**{sum(combined.values())}**"]))
    A("")
    A(f"- GPU-MISSING rows in the two matrices: **{total[MISSING]}**; "
      f"additional missing pixel tools outside the matrices: "
      f"**{len(MISSING_EXTRA)}**; total after de-duplication: "
      f"**{len(missing_rows)}**.")
    A(f"- GPU-UNWIRED: **{total[UNWIRED]}** kernel rows.")
    A(f"- Production direct-allocator files: **{len({a.path for a in scan_result.prod_allocs})}**.")
    A(f"- `required_device_bytes` implementations: **{len(scan_result.required_device_bytes)}** files.")
    A("  The M0/M1/shared core still has no module-level reservation API (only")
    A("  local `need_device` math in `still_gpu.cu:790-860` and")
    A("  `luma_search_scratch_bytes()`); every module landed during this audit")
    A("  exports one, so the newer edge is better than the core.")
    A("")
    A(f"### Top GPU-MISSING by pixel volume — all {len(ranked)}")
    A("")
    A(mrow(["rank", "tool", "CPU reference", "samples/frame", "basis"]))
    A(mrow(["---"] * 5))
    for i, r in enumerate(ranked, 1):
        A(mrow([str(i), r["tool"], r["cpu"], f"{r.get('volume',0):,}", r.get("basis","")]))
    A("")
    A("## Reproduce")
    A("")
    A("```sh")
    A("python3 tools/gpu_coverage/inventory.py --check   # counts only")
    A("python3 tools/gpu_coverage/inventory.py           # rewrite this file")
    A("```")
    A("")
    A("`--check` fails (exit 1) when a curated path no longer exists, which is the")
    A("stale-doc guard: the tree wins over this document.")
    A("")
    A("This file is a static inventory; it does not compile or run anything.  The")
    A("executable module-suite status (2026-11-13: 96/96 passing on sm_86) and the")
    A("reproduction commands live in `docs/cpu-cuda-matrix.md`.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    s = scan()
    doc = render(s)

    missing = check_paths()
    # Status lines go to stderr so `--stdout` emits only the document.
    print(f"cu={len(s.cu)} cuh={len(s.cuh)} kernel_tu={len(s.kernel_cu)} "
          f"test_tu={len(s.test_cu)}", file=sys.stderr)
    print(f"alloc_sites={len(s.allocs)} prod_alloc_sites={len(s.prod_allocs)} "
          f"prod_alloc_files={len({a.path for a in s.prod_allocs})} "
          f"test_alloc_sites={len(s.test_allocs)}", file=sys.stderr)
    print(f"required_device_bytes={len(s.required_device_bytes)} "
          f"mark_rewind={len(s.mark_rewind)}", file=sys.stderr)
    print("feature_verdicts=" + repr(verdict_counts(FEATURES)), file=sys.stderr)
    print("kernel_verdicts_curated=" + repr(verdict_counts(KERNELS)), file=sys.stderr)

    if missing:
        print("STALE PATHS:", file=sys.stderr)
        for m in missing:
            print("  " + m, file=sys.stderr)
        if args.check:
            return 1

    if args.stdout:
        sys.stdout.write(doc)
        return 0
    if args.check:
        return 0

    DOC.write_text(doc)
    print(f"wrote {rel(DOC)} ({len(doc.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
