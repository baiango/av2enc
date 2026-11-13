#ifndef AV2ENC_CUDA_STILL_INTRA_EXTRA_H_
#define AV2ENC_CUDA_STILL_INTRA_EXTRA_H_

// Extra intra mode-decision stages for the M1 GPU still pipeline: MRL
// (§7.13.2.1/.7/.8, mrl index 1..3 + secondary line) and DIP (§7.13.2.3-.6 /
// §5.20.7.9).
//
// The base 61-candidate luma search lives in `pipeline/still/search`.  This
// module extends it with the candidates the CPU `choose_luma_mode()` evaluates
// after (MRL) and after the whole loop (DIP), preserving the CPU's exact
// candidate order and strict-less-than tie-break:
//
//   for each active base candidate in CPU order:
//       compare the base candidate SSE
//       if MRL and the candidate is directional:
//           for mrl_index 1..3, sec 0..1 (0 only for 4x4):
//               predict with `cuda/codec/intra/mrl` and compare
//   if DIP: evaluate §7.13.2.3 modes/transposes and compare once more
//
// Every prediction comes from the existing bit-exact device cores
// (`mrl_device.cuh`, `dip_device.cuh`); this file only adds the search loop,
// the SSE reduction and the device descriptors for the chosen prediction.

#include <cstdint>

#include <cuda_runtime.h>

#include "cuda/codec/intra/dip/dip.h"
#include "cuda/codec/intra/ibp/ibp.h"
#include "cuda/codec/intra/mrl/mrl.h"
#include "cuda/pipeline/still/search/luma_search.h"

namespace av2enc::cuda::still::intra_extra {

namespace srch = ::av2enc::cuda::still::search;

// Device-resident result of the extra stages.  The base LumaSearchResult keeps
// mode/angle/sse and is updated by DIP when it overrides the mode.
struct Choice {
  int mrl_index = 0;
  int mrl_sec_index = 0;
  int use_dip = 0;
  int dip_mode = 0;
  int dip_transpose = 0;
};

// Full 61-candidate luma search using the §7.13.2.7/.8/.9 IBP predictor.  The
// candidate list, gates and ordered tie-break are identical to
// `search::luma_search`; only the DC/directional prediction is IBP-aware.
cudaError_t ibp_luma_search(const srch::LumaSearchBlock* block, int sb_height,
                            unsigned flags, int enable_edge_filter,
                            srch::LumaSearchResult* result,
                            std::int64_t* debug_sse,
                            cudaStream_t stream = nullptr);

// MRL search for one block.  `base_sse` points at the block's 61-entry debug
// SSE array in CPU candidate order (inactive candidates are skipped through
// `flags`).  `result` is updated in place when an MRL candidate wins.
cudaError_t mrl_search(const srch::LumaSearchBlock* block, int sb_height,
                       int enable_edge_filter,
                       const std::int64_t* base_sse, unsigned flags,
                       srch::LumaSearchResult* result, Choice* choice,
                       cudaStream_t stream = nullptr);

// DIP §5.20.7.9 evaluation against the current best.  `base_only` mirrors
// AV2ENC_STILL_BASE_ONLY (the CPU skips DIP in that debug mode).
cudaError_t dip_search(const srch::LumaSearchBlock* block, int sb_height,
                       int enable_edge_filter, int enable_dip, int base_only,
                       srch::LumaSearchResult* result, Choice* choice,
                       cudaStream_t stream = nullptr);

// Copies the chosen mode/angle/MRL/DIP parameters into the prebuilt
// chosen-prediction descriptors for `block_index`.
cudaError_t prepare_jobs(mrl::MrlJob* mrl_jobs, dip::DipJob* dip_jobs,
                         ibp::IbpJob* ibp_jobs,
                         const srch::LumaSearchBlock* search_blocks,
                         const srch::LumaSearchResult* results,
                         const Choice* choices, int block_index,
                         int enable_edge_filter,
                         cudaStream_t stream = nullptr);

// Copies the winning alternative prediction into the packed destination.
// `dst` already holds the ordinary prediction, so only an MRL/DIP winner is
// copied.  Null scratch pointers are allowed when that tool is disabled.
cudaError_t select_prediction(const Choice* choice,
                              const std::uint8_t* mrl_pred,
                              const std::uint8_t* dip_pred, std::uint8_t* dst,
                              int area, cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::still::intra_extra

#endif  // AV2ENC_CUDA_STILL_INTRA_EXTRA_H_
