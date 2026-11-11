#include "encoder/m2/tx_partition/gate/tx_partition_gate_hook.h"

namespace av2enc::m2::txpartition::gate {
namespace {

GateSplitPlan g_plan;
TxPartitionCdfs g_cdfs;
bool g_tx_mode_select = false;

}  // namespace

void reset(const GateSplitPlan* plan, bool tx_mode_select) {
  g_plan = plan != nullptr ? *plan : GateSplitPlan{};
  g_plan.active = plan != nullptr && plan->active;
  g_tx_mode_select = tx_mode_select;
  g_cdfs.reset(false);
}

const GateSplitPlan* active_split_plan() { return &g_plan; }

bool tx_mode_select_enabled() { return g_tx_mode_select; }

TxPartitionCdfs* tx_partition_cdfs() { return &g_cdfs; }

}  // namespace av2enc::m2::txpartition::gate
