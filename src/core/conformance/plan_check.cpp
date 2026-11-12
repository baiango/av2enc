// Standalone printer for the documented 1080p/4K plan matrix.
//
// Build (no CUDA, no encoder):
//   clang++ -std=c++20 -I src -O2 \
//     src/encoder/conformance/conformance_tables.cpp \
//     src/encoder/conformance/conformance.cpp \
//     src/encoder/conformance/known_plans.cpp \
//     src/encoder/conformance/plan_check.cpp -o plan_check
//
// This is a design-review aid, not an encode path. Exit status is 0 even when
// a plan fails: the known violations are expected and documented. Use the API
// in a gate if you want a hard failure.

#include "core/conformance/known_plans.h"

#include <iostream>
#include <vector>

int main() {
  const std::vector<av2enc::conformance::PlanAudit> audits =
      av2enc::conformance::audit_documented_plans();
  int failing = 0;
  for (const auto& audit : audits) {
    if (!audit.result.conformant()) {
      ++failing;
    }
    std::cout << av2enc::conformance::render_plan_audit(audit);
  }
  std::cout << failing << " of " << audits.size()
            << " documented plan(s) fail a level constraint.\n";
  return 0;
}
