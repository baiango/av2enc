#!/usr/bin/env python3
"""Create an experiment-only copy of still_encoder.cpp with the RDO intra hook.

This never edits src/encoder/m1/still_encoder.cpp. It reads the file, verifies the
two anchors below, and writes a patched copy for the experiment build.

The hook is active only when AV2ENC_RDO_INTRA is set:
  * it collects the same 61 luma intra candidates M1 already evaluates;
  * it computes the candidate's actual quantised levels with Encoder::compute_tx
    and estimates coefficient bits from them;
  * it calls av2enc::rdo::select_intra_mode() and encodes the winner.

With the env var unset, the patched copy must produce byte-identical output to
the unpatched encoder; run_still_intra_rdo.sh checks that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "src" / "encoder" / "m1" / "still_encoder.cpp"

INCLUDE_ANCHOR = '#include <vector>\n'
INCLUDE_INSERT = (
    '#include <vector>\n\n'
    '// Experiment-only RDO hook; the production M1 owner should call this API\n'
    '// from choose_luma_mode instead of patching a copy.\n'
    '#include "codec/rate/rdo/intra_decision.h"\n'
)

PRE_LOOP_ANCHOR = """    const bool no_directional = std::getenv("AV2ENC_STILL_NO_DIR") != nullptr;
    best.sse = std::numeric_limits<std::int64_t>::max();
"""

PRE_LOOP_INSERT = """    const bool no_directional = std::getenv("AV2ENC_STILL_NO_DIR") != nullptr;
    const bool intra_rdo_experiment =
        std::getenv("AV2ENC_RDO_INTRA") != nullptr;
    std::vector<av2enc::rdo::IntraCandidateCost> intra_rdo_candidates;
    double intra_rdo_scale = av2enc::rdo::kIntraDefaultSseScale;
    if (const char* scale_env = std::getenv("AV2ENC_RDO_SCALE")) {
      const double parsed_scale = std::atof(scale_env);
      if (parsed_scale > 0.0) intra_rdo_scale = parsed_scale;
    }
    const av2enc::rdo::IntraRdoConfig intra_rdo_config =
        av2enc::rdo::make_intra_rdo_config(
            static_cast<double>(txv::kQuantStep), intra_rdo_scale);
    if (intra_rdo_experiment) {
      intra_rdo_candidates.reserve(static_cast<std::size_t>(count));
    }
    best.sse = std::numeric_limits<std::int64_t>::max();
"""

LOOP_ANCHOR = """      const std::int64_t sse = sse_against_source(0, x0, y0, w, h, pred);
      if (sse < best.sse) {
        best.sse = sse;
        best.mode = params.mode;
        best.angle_delta = params.angle_delta;
        best.av2_mode = av2_y_mode(params.mode);
      }
    }
    return true;
  }
"""

LOOP_REPLACEMENT = """      const std::int64_t sse = sse_against_source(0, x0, y0, w, h, pred);
      if (sse < best.sse) {
        best.sse = sse;
        best.mode = params.mode;
        best.angle_delta = params.angle_delta;
        best.av2_mode = av2_y_mode(params.mode);
      }
      if (intra_rdo_experiment) {
        av2enc::rdo::IntraCandidateCost candidate;
        candidate.mode = av2_y_mode(params.mode);
        candidate.angle_delta = params.angle_delta;
        candidate.sse = static_cast<double>(sse);
        candidate.syntax_rate_bits =
            av2enc::rdo::default_intra_syntax_rate_bits(
                candidate.mode, candidate.angle_delta);
        // Real quantised levels for this candidate. compute_tx() is pure with
        // respect to the reconstruction planes, so probing does not perturb
        // the encoder state.
        TxPlan probe_plan;
        if (!compute_tx(0, x0, y0, w, h, pred, probe_plan)) {
          error_ = "RDO experiment compute_tx failed";
          return false;
        }
        std::vector<std::uint16_t> probe_scan(static_cast<std::size_t>(
            probe_plan.adj_w * probe_plan.adj_h));
        syntax::build_scan(probe_plan.adj_w, probe_plan.adj_h, probe_scan);
        std::vector<std::int32_t> probe_levels;
        probe_levels.reserve(static_cast<std::size_t>(probe_plan.eob));
        for (int scan_pos = 0; scan_pos < probe_plan.eob; ++scan_pos) {
          probe_levels.push_back(
              probe_plan.levels[probe_scan[static_cast<std::size_t>(scan_pos)]]);
        }
        candidate.coefficient_rate_bits =
            av2enc::rdo::intra_coefficient_rate_bits_from_levels(
                probe_levels, probe_plan.eob,
                probe_plan.adj_w * probe_plan.adj_h);
        intra_rdo_candidates.push_back(candidate);
      }
    }
    if (intra_rdo_experiment && !intra_rdo_candidates.empty()) {
      const av2enc::rdo::IntraModeChoice choice =
          av2enc::rdo::select_intra_mode(intra_rdo_candidates,
                                         intra_rdo_config);
      if (choice.found) {
        best.mode = static_cast<intra::Mode>(choice.mode);
        best.angle_delta = choice.angle_delta;
        best.av2_mode = choice.mode;
        best.sse = static_cast<std::int64_t>(choice.sse);
      }
    }
    return true;
  }
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"patch: {label}: expected exactly one anchor, found {count}. "
            "still_encoder.cpp changed; update the experiment patch."
        )
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()

    text = SOURCE.read_text()
    text = replace_once(text, INCLUDE_ANCHOR, INCLUDE_INSERT, "include")
    text = replace_once(text, PRE_LOOP_ANCHOR, PRE_LOOP_INSERT, "pre-loop")
    text = replace_once(text, LOOP_ANCHOR, LOOP_REPLACEMENT, "candidate loop")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"patched {SOURCE} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
