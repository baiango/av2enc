// Gate-side state for the transform-partition dav2d gate.
//
// The production M2 writer has no TX-partition syntax.  The gate compiles
// build-directory patched copies of m2_syntax.cpp / m2_headers.cpp (see
// README.md and the two patch scripts).  The patched writer calls this hook to
// obtain (a) the per-frame tx_mode_select bit and (b) the split plan and
// per-transform-block coefficient layout for the single non-skip inter block.
//
// Nothing in this header is part of the production encoder API.
#pragma once

#include <cstdint>

#include "encoder/m2/tx_partition/tx_partition.h"

namespace av2enc::m2::txpartition::gate {

struct GateSubTx {
  std::uint16_t mi_row = 0;
  std::uint16_t mi_col = 0;
  std::uint8_t tx_size = 0;
  std::uint32_t eob = 0;
  std::uint32_t levels_offset = 0;
};

struct GateSplitPlan {
  bool active = false;
  std::uint16_t mi_row = 0;
  std::uint16_t mi_col = 0;
  std::uint8_t mi_size = 0;
  std::uint8_t partition = 0;  // txpartition::TxPartition value
  int count = 0;
  GateSubTx subs[4];
};

// Resets the hook for one frame.  `plan` may be null (no split); the
// tx_mode_select flag applies to every non-key frame encoded afterwards.
void reset(const GateSplitPlan* plan, bool tx_mode_select);

const GateSplitPlan* active_split_plan();
bool tx_mode_select_enabled();

// CDF state for the patched writer.  One tile per frame; the gate resets this
// before assembling a frame that signals TX partitions.
TxPartitionCdfs* tx_partition_cdfs();

}  // namespace av2enc::m2::txpartition::gate
