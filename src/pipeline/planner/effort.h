// Effort/cpu-used feature ranking for the Still (CPU reference) encoder.
//
// The selector is offline and deterministic: it reads the measured feature
// table that was compiled in by tools/effort/gen_table.py and expands a budget
// into whole existing CLI flags.  It never encodes during selection and never
// invents a score.
//
// Runtime algorithm (docs/effort.md):
//   1. rank eligible features by measured score ascending (efficient first);
//   2. walk the ranking and take a prefix while the cumulative score fits the
//      budget; stop at the first overshoot and never exceed;
//   3. use the measured marginal score when the optional marginal pass covered
//      the feature, otherwise the solo score;
//   4. skip a feature whose flag is already explicit or conflicts with an
//      explicit/selected flag (the CLI's own rejection rules);
//   5. presets are just rank cutoffs: --cpu-used K maps to a geometric budget.
#ifndef AV2ENC_PIPELINE_PLANNER_EFFORT_H
#define AV2ENC_PIPELINE_PLANNER_EFFORT_H

#include <string>
#include <string_view>
#include <vector>

namespace av2enc::effort {

enum class Mode {
  kStill = 0,
  kVideo = 1,
};

// Encoder options that decide whether a feature is compatible with what the
// user already asked for.  Values mirror src/main.cpp's Options; the selector
// is host-side and must not include the CLI.
struct Constraints {
  bool cfl = true;
  bool sb128 = false;
  bool sb256 = false;
  bool qm = false;
  bool dc_ac_delta_q = false;
  bool still_single = false;
  bool mrl = false;
  bool wiener = false;
  bool chroma_lr = false;
  bool pc_wiener = false;
  bool gdf = false;
};

struct Selection {
  bool active = false;
  double budget = 0.0;
  double used = 0.0;
  bool near_budget = false;
  std::vector<std::string> flags;       // enabled flags, in rank order
  std::vector<std::string> explicit_features;  // caller-owned, not charged
  std::vector<std::string> excluded;    // "flag: reason" (measured ineligible)
  std::vector<std::string> conflicts;   // "flag: conflict" (skipped)
  std::vector<std::string> marginal;    // flags whose marginal cost was used
  std::string budget_source;            // "--effort 61", "--cpu-used 2", ...
};

// --cpu-used 0..8 -> geometric budget in [10, 10000]: 8 -> 10, 0 -> 10000.
double budget_for_cpu_used(int cpu_used);

// Expand a budget into a ranked feature prefix.  `explicit_flags` are the CLI
// option tokens the user typed: an explicitly requested feature is kept by the
// caller and not charged to the budget.  Returns false only for invalid input;
// an empty selection is a valid result ("nothing measured fits").
bool select(double budget, Mode mode,
            const std::vector<std::string_view>& explicit_flags,
            const Constraints& constraints, Selection& selection,
            std::string& error);

// Human-readable report (budget, selected features, skipped features, safety
// margin, prefix validation).  Printed by the CLI when a preset is active.
std::string format_selection(const Selection& selection);

}  // namespace av2enc::effort

#endif  // AV2ENC_PIPELINE_PLANNER_EFFORT_H
