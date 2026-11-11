// Conformance check for the plans that the product actually documents.
//
// The matrix mirrors bench/multi_gpu_plan/README.md ("1/2/4 GPUs at 1080p and
// 4K, default layouts (4x2, 8x2, 16x2)") and src/encoder/pipeline/README.md.
// It also carries the two known violations called out in the audit:
//
//   * 16x2 (16 tile columns) at level 5.1 vs MaxTileCols = 8
//     (Table A.9, p1106; A.4, p1110).
//   * DCI-4K 4096x2304 = 9,437,184 > MaxPicSize = 8,912,896
//     at level 5.1 (Table A.8, p1105; A.4, p1110).
//
// Checking these plans does not change any encoder behaviour: the audit is a
// pure function of the plan plus a nominal output rate. Use it in a design
// review or a gate; the actual frame bytes are unaffected.
#pragma once

#include <string>
#include <vector>

#include "encoder/conformance/conformance.h"

namespace av2enc::conformance {

struct DocumentedPlan {
  const char* name = "";
  const char* source = "";
  int width = 0;
  int height = 0;
  int gpu_count = 0;
  int tile_cols = 1;
  int tile_rows = 1;
  int seq_level_idx = 7;  // 7 = level 5.1
  int seq_tier = 0;
};

struct PlanAudit {
  DocumentedPlan plan;
  Config config;
  Result result;
};

// 1080p and 4K at 1/2/4 GPUs (4x2, 8x2, 16x2), the 16x2 plan at level 6.0 as a
// remediation witness, and DCI-4K 4096x2304 at level 5.1.
std::vector<DocumentedPlan> documented_plans();

// Uses a nominal 60 fps display rate (120 fps for the throughput matrix is not
// documented) and 4 reference frames, the bench default.
PlanAudit audit_plan(const DocumentedPlan& plan, double fps = 60.0,
                     int num_ref_frames = 4);
std::vector<PlanAudit> audit_documented_plans(double fps = 60.0,
                                              int num_ref_frames = 4);

// Human-readable report. Empty string is never returned; a clean plan prints
// "PASS" plus any warnings/not-evaluated lines.
std::string render_plan_audit(const PlanAudit& audit);
std::string render_plan_matrix();

}  // namespace av2enc::conformance
