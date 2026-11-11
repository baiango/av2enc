#include "batch.h"

#include <cmath>
#include <limits>

namespace av2enc::m1::rdo_loop {
namespace {

constexpr double kInfiniteCost = std::numeric_limits<double>::infinity();

bool record_terms_valid(const CandidateRecord& record) noexcept {
  return std::isfinite(record.distortion_sse) && record.distortion_sse >= 0.0 &&
         std::isfinite(record.mode_bits) && record.mode_bits >= 0.0 &&
         std::isfinite(record.coefficient_bits) &&
         record.coefficient_bits >= 0.0 &&
         std::isfinite(record.tx_type_bits) && record.tx_type_bits >= 0.0 &&
         std::isfinite(record.signal_bits) && record.signal_bits >= 0.0;
}

}  // namespace

bool append_candidate(CandidateBatch& batch, const CandidateRecord& record,
                      std::string* error) {
  if (batch.count < 0 || batch.count >= kMaxBatchCandidates) {
    ++batch.rejected;
    if (error != nullptr) *error = "candidate batch is full";
    return false;
  }
  if (record.domain != static_cast<std::uint8_t>(batch.domain)) {
    ++batch.rejected;
    if (error != nullptr) *error = "candidate domain does not match batch";
    return false;
  }
  batch.records[batch.count++] = record;
  return true;
}

bool finalize_batch(CandidateBatch& batch, std::string* error) {
  batch.best_index = -1;
  batch.best_cost = kInfiniteCost;
  batch.evaluated = 0;
  if (batch.count <= 0 || batch.count > kMaxBatchCandidates) {
    if (error != nullptr) *error = "candidate batch has no candidates";
    return false;
  }
  if (!std::isfinite(batch.lambda) || batch.lambda < 0.0) {
    if (error != nullptr) *error = "candidate batch lambda is invalid";
    return false;
  }
  const bool rate_aware = batch.rate_aware != 0;

  int best_position = -1;
  for (int i = 0; i < batch.count; ++i) {
    CandidateRecord& record = batch.records[i];
    record.valid = 1;
    if (!record_terms_valid(record)) {
      record.valid = 0;
      record.cost = kInfiniteCost;
      ++batch.rejected;
      continue;
    }
    const double bits = record.mode_bits + record.coefficient_bits +
                        record.tx_type_bits + record.signal_bits;
    record.cost = rate_aware ? record.distortion_sse + batch.lambda * bits
                             : record.distortion_sse;
    if (!std::isfinite(record.cost)) {
      record.valid = 0;
      record.cost = kInfiniteCost;
      ++batch.rejected;
      continue;
    }
    ++batch.evaluated;
    if (best_position < 0 ||
        batch_record_better(record, i, batch.records[best_position],
                            best_position, rate_aware)) {
      best_position = i;
    }
  }
  if (best_position < 0) {
    if (error != nullptr) *error = "candidate batch has no valid candidate";
    return false;
  }
  batch.best_index = best_position;
  batch.best_cost = batch.records[best_position].cost;
  return true;
}

CandidateRecord make_candidate_record(std::uint32_t group, std::uint16_t index,
                                      CandidateDomain domain,
                                      const CostTerms& terms) noexcept {
  CandidateRecord record;
  record.group = group;
  record.index = index;
  record.domain = static_cast<std::uint8_t>(domain);
  record.distortion_sse = terms.distortion_sse;
  record.mode_bits = terms.mode_bits;
  record.coefficient_bits = terms.coefficient_bits;
  record.tx_type_bits = terms.tx_type_bits;
  record.signal_bits = terms.signal_bits;
  return record;
}

}  // namespace av2enc::m1::rdo_loop
