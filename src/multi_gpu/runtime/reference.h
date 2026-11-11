// Reference duplication and host-transfer scheduling.
//
// Each GPU owns a full reference pool. Inter prediction can read any region of
// a reference frame, so a GPU cannot use only the regions it reconstructed.
// Per frame the runtime:
//
//   1. gathers each GPU's reconstructed tile regions (D2H, one frame total),
//   2. assembles the full frame on the host (pinned staging),
//   3. copies each GPU's own regions device-to-device from recon into the next
//      reference slot,
//   4. fills every missing region H2D from the host assembly.
//
// Only the new reconstructed frame moves. The persistent refs * frame_bytes
// pool is never re-uploaded, and VRAM is never pooled across GPUs.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "../planner.h"
#include "device_backend.h"

namespace av2enc::multigpu::runtime {

enum class TransferPhase : std::uint8_t {
  kSourceUpload = 0,
  kPayloadDownload = 1,
  kReconGather = 2,
  kReferenceFill = 3,
};

enum class TransferDirection : std::uint8_t {
  kHostToDevice = 0,
  kDeviceToHost = 1,
  kDeviceToDevice = 2,
};

struct TransferOp {
  TransferPhase phase = TransferPhase::kSourceUpload;
  TransferDirection direction = TransferDirection::kHostToDevice;
  int gpu_slot = -1;
  int ordinal = -1;
  int tile_index = -1;  // -1 = whole frame (source upload / payload aggregate)
  std::uint64_t bytes = 0;
  int stream_role = 0;  // 0 = compute stream, 1 = per-GPU copy stream
  int depends_on = -1;  // index of the event-producing op, or -1
};

struct ReferenceSchedule {
  bool valid = false;
  std::string error;
  int gpu_count = 0;
  int tile_count = 0;
  int reference_frames = 0;
  std::uint64_t source_upload_bytes = 0;
  std::uint64_t payload_bytes = 0;
  std::uint64_t recon_gather_bytes = 0;
  std::uint64_t reference_fill_bytes = 0;   // H2D only
  std::uint64_t reference_d2d_bytes = 0;    // local recon -> ref slot
  std::uint64_t host_tx_bytes = 0;          // H2D + D2H, excludes D2D
  double estimated_ms = 0.0;                // analytic, planner PCIe model
  std::vector<TransferOp> ops;              // deterministic order
};

// payload_bytes_per_gpu is optional; empty means zero. `gpu_ordinals` maps
// planner slots to physical device ordinals; empty uses the planner slot index.
// Payload sizes are an accounting input (actual sizes are known only after
// encoding).
ReferenceSchedule plan_reference_schedule(
    const MultiGpuPlan& plan, std::span<const int> gpu_ordinals,
    std::span<const std::uint64_t> payload_bytes_per_gpu);

// Bytes of one tile rectangle across Y/U/V for 4:2:0 at `bit_depth` bits.
std::uint64_t tile_region_bytes(const tiles::TileLayout& layout,
                                const tiles::TileRect& rect, int bit_depth);

// Enqueue row-wise copies for one tile rectangle. The same full-frame offset is
// used in both buffers; pass the base of the plane (source, recon, or a ref
// slot) and offset 0 when the base is already the frame start.
Status enqueue_tile_region_copy(DeviceBackend& backend, CopyDirection direction,
                                StreamHandle stream, std::uint8_t* dst_base,
                                const std::uint8_t* src_base,
                                std::uint64_t frame_offset,
                                const tiles::TileLayout& layout,
                                const tiles::TileRect& rect, int bit_depth);

// Host-only equivalent used to assemble the full reconstruction frame from
// per-GPU staging buffers.
Status copy_tile_region_host(std::uint8_t* dst_frame,
                             const std::uint8_t* src_frame,
                             const tiles::TileLayout& layout,
                             const tiles::TileRect& rect, int bit_depth);

}  // namespace av2enc::multigpu::runtime
