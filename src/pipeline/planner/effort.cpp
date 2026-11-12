#include "pipeline/planner/effort.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace av2enc::effort {
namespace {

namespace table {
#include "pipeline/planner/effort_table.inc"
}  // namespace table

constexpr double kMinBudget = 10.0;
constexpr double kMaxBudget = 10000.0;
constexpr double kSafetyFraction = 0.90;

std::string fmt1(double value) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.1f", value);
  return buffer;
}

std::string fmt6(double value) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.6f", value);
  return buffer;
}

const table::Feature* find_feature(std::string_view flag) {
  for (const table::Feature& feature : table::kFeatures) {
    if (feature.flag != nullptr && flag == feature.flag) return &feature;
  }
  return nullptr;
}

bool in_list(const char* const (&list)[4], std::string_view value) {
  for (const char* item : list) {
    if (item != nullptr && value == item) return true;
  }
  return false;
}

bool requires_cfl_conflict(const table::Feature& feature,
                           const Constraints& constraints) {
  return feature.requires_cfl && !constraints.cfl;
}

// The explicit user flags that conflict with a feature.  This covers flags that
// are not themselves in the measured table (for example --still-wiener).
bool explicit_conflict(const table::Feature& feature,
                       const std::vector<std::string_view>& explicit_flags) {
  for (const std::string_view flag : explicit_flags) {
    if (in_list(feature.conflicts, flag)) return true;
  }
  return false;
}

bool conflicts_with_selected(const table::Feature& feature,
                             const std::vector<const table::Feature*>& selected) {
  for (const table::Feature* other : selected) {
    if (other == nullptr) continue;
    if (in_list(feature.conflicts, other->flag)) return true;
    if (in_list(other->conflicts, feature.flag)) return true;
  }
  return false;
}

double effective_score(const table::Feature& feature, bool& marginal_used) {
  if (feature.marginal_score >= 0.0) {
    marginal_used = true;
    return feature.marginal_score;
  }
  marginal_used = false;
  return feature.score;
}

}  // namespace

double budget_for_cpu_used(int cpu_used) {
  if (cpu_used < 0) cpu_used = 0;
  if (cpu_used > 8) cpu_used = 8;
  const double exponent = static_cast<double>(8 - cpu_used) / 8.0;
  const double raw = 10.0 * std::pow(1000.0, exponent);
  double budget = std::round(raw);
  budget = std::clamp(budget, kMinBudget, kMaxBudget);
  return budget;
}

bool select(double budget, Mode mode,
            const std::vector<std::string_view>& explicit_flags,
            const Constraints& constraints, Selection& selection,
            std::string& error) {
  selection = Selection{};
  if (!std::isfinite(budget) || budget < kMinBudget || budget > kMaxBudget) {
    error = "effort budget must be in [10, 10000]";
    return false;
  }
  selection.active = true;
  selection.budget = budget;

  const int mode_value = mode == Mode::kStill ? 0 : 1;
  std::vector<const table::Feature*> ranked;
  for (const table::Feature& feature : table::kFeatures) {
    if (feature.mode != mode_value) continue;
    if (!feature.eligible) {
      selection.excluded.emplace_back(
          std::string(feature.flag) + ": " +
          (feature.exclude_reason != nullptr ? feature.exclude_reason
                                             : "measured ineligible"));
      continue;
    }
    ranked.push_back(&feature);
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const table::Feature* a, const table::Feature* b) {
              if (a->score != b->score) return a->score < b->score;
              return std::string_view(a->flag) < std::string_view(b->flag);
            });

  std::vector<const table::Feature*> selected;
  for (const table::Feature* feature : ranked) {
    const std::string_view flag(feature->flag);
    if (std::find(explicit_flags.begin(), explicit_flags.end(), flag) !=
        explicit_flags.end()) {
      selection.explicit_features.emplace_back(feature->flag);
      continue;  // caller owns this feature; it is not charged
    }
    if (requires_cfl_conflict(*feature, constraints) ||
        explicit_conflict(*feature, explicit_flags) ||
        conflicts_with_selected(*feature, selected)) {
      selection.conflicts.emplace_back(
          std::string(feature->flag) + ": conflicts with an explicit or "
          "higher-ranked flag");
      continue;
    }
    bool marginal_used = false;
    const double cost = effective_score(*feature, marginal_used);
    if (cost < 0.0 || feature->marginal_score == -2.0) {
      selection.conflicts.emplace_back(
          std::string(feature->flag) +
          ": marginal pass measured no equal-rate gain");
      continue;
    }
    if (selection.used + cost > budget + 1.0e-9) {
      break;  // prefix: do not skip ahead to a cheaper feature
    }
    selection.used += cost;
    selection.flags.emplace_back(feature->flag);
    selected.push_back(feature);
    if (marginal_used) selection.marginal.emplace_back(feature->flag);
  }
  selection.near_budget = selection.used > kSafetyFraction * budget;
  return true;
}

std::string format_selection(const Selection& selection) {
  std::ostringstream out;
  out << "av2enc: effort: budget " << fmt1(selection.budget);
  if (!selection.budget_source.empty()) {
    out << " (" << selection.budget_source << ")";
  }
  out << "; selected " << selection.flags.size() << " feature(s); used "
      << fmt1(selection.used) << " of " << fmt1(selection.budget) << "\n";
  out << "  score unit: best measured feature = 10 (the --effort minimum; table "
         "scale " << fmt6(table::kScoreScale)
      << " of the literal formula; see docs/effort.md)\n";
  if (selection.flags.empty()) {
    out << "  enabled: none (no measured feature fits this budget)\n";
  }
  for (std::size_t i = 0; i < selection.flags.size(); ++i) {
    const table::Feature* feature = find_feature(selection.flags[i]);
    if (feature == nullptr) continue;
    const bool marginal = std::find(selection.marginal.begin(),
                                    selection.marginal.end(),
                                    selection.flags[i]) !=
                          selection.marginal.end();
    const double time_pct =
        marginal ? feature->marginal_d_time_pct : feature->d_time_pct;
    const double rate_pct =
        marginal ? feature->marginal_d_rate_pct : feature->d_rate_pct;
    out << "  #" << (i + 1) << " " << feature->flag << " ("
        << (marginal ? "marginal " : "score ") << fmt1(
               marginal ? feature->marginal_score : feature->score)
        << "; " << fmt1(time_pct) << "% time, "
        << fmt1(rate_pct) << "% rate at equal CVVDP; "
        << feature->confidence << " confidence)\n";
  }
  if (!selection.explicit_features.empty()) {
    out << "  explicit (caller, not charged):";
    for (const std::string& flag : selection.explicit_features) {
      out << " " << flag;
    }
    out << "\n";
  }
  if (!selection.excluded.empty()) {
    out << "  excluded:\n";
    for (const std::string& item : selection.excluded) {
      out << "    " << item << "\n";
    }
  }
  if (!selection.conflicts.empty()) {
    out << "  skipped:\n";
    for (const std::string& item : selection.conflicts) {
      out << "    " << item << "\n";
    }
  }
  if (selection.near_budget) {
    out << "  note: used " << fmt1(selection.used / selection.budget * 100.0)
        << "% of the budget (scores are upper-bound p75 A/B time)\n";
  }
  out << "  top-K prefix calibration: additive "
      << fmt1(table::kPrefixValidation.additive_score) << ", measured score "
      << fmt1(table::kPrefixValidation.score) << " ("
      << fmt1(table::kPrefixValidation.d_time_pct) << "% time / "
      << fmt1(table::kPrefixValidation.d_rate_pct)
      << "% rate at equal CVVDP)";
  if (!table::kPrefixValidation.measured) out << " (not measured)";
  out << "\n";
  if (table::kPsnrFallback) {
    out << "  note: calibration used PSNR fallback, not CVVDP\n";
  }
  if (table::kUncalibratedCount > 0) {
    out << "  uncalibrated flags excluded from selection: "
        << table::kUncalibratedCount << " (docs/effort.md)\n";
  }
  out << "  calibration: " << table::kCalibrationMetric << " on "
      << table::kCalibrationHost << "; " << table::kCalibrationFixture
      << "; scores are host-specific\n";
  return out.str();
}

}  // namespace av2enc::effort
