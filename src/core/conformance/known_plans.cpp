#include "core/conformance/known_plans.h"

#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::conformance {

std::vector<DocumentedPlan> documented_plans() {
  constexpr const char* kBench =
      "bench/multi_gpu_plan/README.md simulated matrix";
  constexpr const char* kPipeline = "src/encoder/pipeline/README.md";
  std::vector<DocumentedPlan> plans;
  plans.push_back({"1080p-1gpu-4x2", kBench, 1920, 1080, 1, 4, 2, 7, 0});
  plans.push_back({"1080p-2gpu-8x2", kBench, 1920, 1080, 2, 8, 2, 7, 0});
  plans.push_back({"1080p-4gpu-16x2", kBench, 1920, 1080, 4, 16, 2, 7, 0});
  plans.push_back({"4k-1gpu-4x2", kBench, 3840, 2160, 1, 4, 2, 7, 0});
  plans.push_back({"4k-2gpu-8x2", kBench, 3840, 2160, 2, 8, 2, 7, 0});
  plans.push_back({"4k-4gpu-16x2", kBench, 3840, 2160, 4, 16, 2, 7, 0});
  // Remediation witness: the same 16 columns at level 6.0 (idx 10), where
  // MaxTileCols is 16 (Table A.9, p1106).
  plans.push_back({"4k-4gpu-16x2-level6.0", kPipeline, 3840, 2160, 4, 16, 2,
                   10, 0});
  // DCI-4K, the known MaxPicSize violation. 4096x2304 is 9,437,184 samples.
  plans.push_back(
      {"dci4k-4096x2304-4x2", kPipeline, 4096, 2304, 1, 4, 2, 7, 0});
  return plans;
}

PlanAudit audit_plan(const DocumentedPlan& plan, double fps, int num_ref_frames) {
  PlanAudit audit;
  audit.plan = plan;

  Config& config = audit.config;
  config.width = plan.width;
  config.height = plan.height;
  config.sb_size = 64;
  config.seq_profile_idc = 0;  // Main_420_10_IP0
  config.seq_level_idx = plan.seq_level_idx;
  config.seq_tier = plan.seq_tier;
  config.chroma_format_idc = 0;
  config.bit_depth_idc = 1;  // 8-bit
  config.still_picture = false;
  config.single_picture_header_flag = false;
  config.explicit_num_ref_frames = true;
  config.num_ref_frames = num_ref_frames;
  config.tiles = TileGrid::uniform(plan.width, plan.height, plan.tile_cols,
                                   plan.tile_rows, config.sb_size);

  StreamStats stats;
  stats.fps = fps;
  stats.num_ref_frames = num_ref_frames;
  stats.output_frames_in_tu = 1;
  stats.tu_display_luma_samples =
      static_cast<int64_t>(plan.width) * static_cast<int64_t>(plan.height);
  stats.tu_duration_seconds = fps > 0.0 ? 1.0 / fps : 0.0;
  stats.max_frame_headers_per_second =
      fps > 0.0 ? static_cast<int>(std::ceil(fps)) : 0;

  audit.result = validate(config, stats);
  return audit;
}

std::vector<PlanAudit> audit_documented_plans(double fps, int num_ref_frames) {
  std::vector<PlanAudit> audits;
  for (const DocumentedPlan& plan : documented_plans()) {
    audits.push_back(audit_plan(plan, fps, num_ref_frames));
  }
  return audits;
}

std::string render_plan_audit(const PlanAudit& audit) {
  std::ostringstream out;
  out << audit.plan.name << " (" << audit.plan.width << "x"
      << audit.plan.height << ", " << audit.plan.gpu_count << " GPU, "
      << audit.plan.tile_cols << "x" << audit.plan.tile_rows << ", level "
      << level_name(audit.plan.seq_level_idx) << ", tier "
      << audit.plan.seq_tier << ")\n";
  out << "  " << (audit.result.conformant() ? "PASS" : "FAIL") << ": "
      << audit.result.summary() << "\n";
  for (const Violation& violation : audit.result.violations) {
    out << "  ["
        << (violation.severity == Severity::kError ? "error" : "warning")
        << "] " << violation.check << " (" << violation.section << " p"
        << violation.page << "): " << violation.detail << "\n";
  }
  for (const std::string& skipped : audit.result.not_evaluated) {
    out << "  [not evaluated] " << skipped << "\n";
  }
  return out.str();
}

std::string render_plan_matrix() {
  std::ostringstream out;
  for (const PlanAudit& audit : audit_documented_plans()) {
    out << render_plan_audit(audit);
  }
  return out.str();
}

}  // namespace av2enc::conformance
