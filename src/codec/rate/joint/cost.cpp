#include "cost.h"

#include <cmath>
#include <limits>

namespace av2enc::still::rdo_loop {
namespace {

constexpr double kInfiniteCost = std::numeric_limits<double>::infinity();

bool finite_non_negative(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

double cost_from(const CostTerms& terms, double lambda, bool rate_aware,
                 int leaves) noexcept {
  if (!cost_terms_valid(terms) || !std::isfinite(lambda) || lambda < 0.0 ||
      leaves < 0) {
    return kInfiniteCost;
  }
  if (!rate_aware) {
    return terms.distortion_sse;
  }
  return terms.distortion_sse + lambda * total_bits(terms);
}

}  // namespace

double total_bits(const CostTerms& terms) noexcept {
  return terms.mode_bits + terms.coefficient_bits + terms.tx_type_bits +
         terms.signal_bits;
}

bool cost_terms_valid(const CostTerms& terms) noexcept {
  return finite_non_negative(terms.distortion_sse) &&
         finite_non_negative(terms.mode_bits) &&
         finite_non_negative(terms.coefficient_bits) &&
         finite_non_negative(terms.tx_type_bits) &&
         finite_non_negative(terms.signal_bits) &&
         std::isfinite(total_bits(terms));
}

bool node_cost_valid(const NodeCost& cost) noexcept {
  return std::isfinite(cost.lambda) && cost.lambda >= 0.0 &&
         std::isfinite(cost.cost) && cost.cost >= 0.0 &&
         cost.leaves >= 0 && cost_terms_valid(cost.terms);
}

NodeCost make_node_cost(const CostTerms& terms, double lambda, bool rate_aware,
                        int leaves) noexcept {
  NodeCost cost;
  cost.terms = terms;
  cost.lambda = lambda;
  cost.rate_aware = rate_aware;
  cost.leaves = leaves;
  cost.cost = cost_from(terms, lambda, rate_aware, leaves);
  return cost;
}

void accumulate_node_cost(NodeCost& target, const NodeCost& addend) noexcept {
  if (!node_cost_valid(addend) || !std::isfinite(target.cost)) {
    target.cost = kInfiniteCost;
    return;
  }
  target.terms.distortion_sse += addend.terms.distortion_sse;
  target.terms.mode_bits += addend.terms.mode_bits;
  target.terms.coefficient_bits += addend.terms.coefficient_bits;
  target.terms.tx_type_bits += addend.terms.tx_type_bits;
  target.terms.signal_bits += addend.terms.signal_bits;
  target.leaves += addend.leaves;
  target.cost = cost_from(target.terms, target.lambda, target.rate_aware,
                          target.leaves);
}

void add_signal_bits(NodeCost& target, double bits) noexcept {
  if (!cost_terms_valid(target.terms) || !finite_non_negative(bits)) {
    target.cost = kInfiniteCost;
    return;
  }
  target.terms.signal_bits += bits;
  target.cost = cost_from(target.terms, target.lambda, target.rate_aware,
                          target.leaves);
}

bool node_cost_better(const NodeCost& a, partition::PartitionType part_a,
                      int index_a, const NodeCost& b,
                      partition::PartitionType part_b, int index_b) noexcept {
  if (a.cost != b.cost) {
    return a.cost < b.cost;
  }
  if (a.terms.distortion_sse != b.terms.distortion_sse) {
    return a.terms.distortion_sse < b.terms.distortion_sse;
  }
  const double rate_a = total_bits(a.terms);
  const double rate_b = total_bits(b.terms);
  if (rate_a != rate_b) {
    return rate_a < rate_b;
  }
  if (a.leaves != b.leaves) {
    return a.leaves < b.leaves;
  }
  if (part_a != part_b) {
    return static_cast<int>(part_a) < static_cast<int>(part_b);
  }
  return index_a < index_b;
}

}  // namespace av2enc::still::rdo_loop
