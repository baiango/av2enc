// M1 GPU -> syntax bridge.
//
// The GPU pipeline (src/cuda/m1/) produces plan/decision/coefficient PODs; the
// CPU syntax writer (src/encoder/m1/m1_syntax.cpp) turns a *block-result*
// stream into the AV2 bitstream.  This bridge is the only place that knows the
// mapping between the two representations, so both the CPU reference and the
// GPU path emit the same bytes from the same decisions.
//
// What this file does:
//   * converts av2enc::cuda::m1gpu::BlockPlan / BlockDecision / coefficient
//     planes into syntax::BlockSyntax + a flat scan-order level array;
//   * re-orders coefficients from raster (forward-transform output) order into
//     the scan order the writer consumes;
//   * derives the trivial raster partition tree (one PARTITION_NONE block per
//     superblock) that m1gpu::FramePlan describes while is_placeholder == 1;
//   * validates the result before a single bit is written.
//
// What it deliberately does not do: decide modes, transform, reconstruct,
// write OBU headers, or own CUDA state.  It is host-only, allocation-free
// except for the output vectors, and safe to call on the orchestrator thread
// after the single per-frame download.
//
// Ordering and CDF-context requirements the orchestrator must satisfy are
// documented in src/encoder/m1/stream/README.md.
#pragma once

#include "cuda/m1/m1_gpu.h"
#include "cuda/m1/plan/plan.h"
#include "encoder/m1/m1_syntax.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::m1::stream {

// One partition-tree node as reported by the GPU plan.
//
// `cuda::m1gpu::BlockPlan` currently describes leaves only, so a plan that
// carries a real tree supplies this parallel array in preorder (superblocks in
// raster order, each node followed by its `child_count` children).  Leaves
// point at their entry in the BlockPlan/BlockDecision arrays with
// `leaf_index`.
//
// This is the one structural gap against m1_gpu.h: BlockPlan has no partition
// symbols.  Growing BlockPlan with `partition`, `allowed_mask`, `forced` and
// `child_count` would let the bridge read the tree from the block array and
// delete this struct; see src/encoder/m1/stream/README.md §7.
struct TreeNode {
  std::uint16_t mi_row = 0;
  std::uint16_t mi_col = 0;
  std::uint8_t bsize = 0;         // partition::BlockSize
  std::uint8_t partition = 0;     // partition::PartitionType
  std::uint8_t allowed_mask = 0;  // bit p => allowed[PartitionType p]
  std::uint8_t flags = 0;         // syntax::kFlagForced (kFlagSbRoot is added)
  std::uint8_t child_count = 0;   // 0 for a leaf, else 1..4 in decode order
  std::uint32_t leaf_index = 0;   // leaf_index into BlockPlan / BlockDecision
};

// Bridge output: everything syntax::assemble_m1_frame() needs.
//
// `blocks` and `levels` are *derived* from the GPU PODs; the coefficient layout
// conversion (per-plane GPU buffers -> one interleaved flat array) is described
// in "Coefficient layout conversion" below.
struct PackedFrame {
  syntax::FramePlan plan;
  std::vector<syntax::BlockSyntax> blocks;
  std::vector<std::int32_t> levels;
};

// ---------------------------------------------------------------------------
// Coefficient layout conversion
// ---------------------------------------------------------------------------
//
// The two representations do not have the same shape, and this bridge is the
// only place that reconciles them:
//
//   GPU (cuda::m1gpu)                        CPU syntax stream (m1_syntax.h)
//   -------------------------------------    -------------------------------
//   three independent per-plane buffers      one flat, block-interleaved array
//   coeff_y / coeff_u / coeff_v              levels[]
//   each indexed from BlockPlan's own        per *block*: luma block, then U,
//   coeff_offset_y / _u / _v                 then V, each exactly `eob` entries
//   raster order at the adjusted TX size     scan order at the adjusted TX size
//
// Per block and plane, the bridge performs
//
//     for c in [0, eob[plane]):
//         levels[levels_offset + sum(eob[0..plane-1]) + c]
//             = plane_buffer[coeff_offset_plane + scan[plane][c]]
//
// so a block occupies a contiguous `eob_y + eob_u + eob_v` slice of `levels`,
// starting at its `BlockSyntax::levels_offset` (assigned by the bridge, in block
// visit order), and only the non-zero prefix of the scan is stored.
//
// Worked example (one 64x64 block, luma coded 64x64 -> adjusted 32x32, chroma
// 32x32 -> adjusted 32x32):
//
//   BlockPlan{ coeff_offset_y = 0, coeff_offset_u = 0, coeff_offset_v = 64 }
//   BlockDecision{ eob_y = 3, eob_u = 2, eob_v = 0 }
//   coeff_y = [a0 a1 a2 ...]           (raster 32x32, plane-local)
//   coeff_u = [b0 b1 ...]              (raster 32x32, plane-local)
//   coeff_v = [ ... ]                  (never read: eob_v == 0)
//
//   levels  = [ y(scan 0), y(scan 1), y(scan 2), u(scan 0), u(scan 1) ]
//             \_______ eob_y = 3 (levels_offset +0..2) ______/
//                                                 \__ eob_u = 2 (+3..4) __/
//
// The per-plane offsets therefore never appear in the packed stream, and the
// packed stream never carries a per-plane stride.  Two further requirements are
// validated before any bit is written: a plane's declared ranges must not
// overlap another block's (that would mean two blocks share coefficients), and
// `coeff_offset + adjusted_area` must fit both `FramePlan::coeff_count_*` (when
// the plan fills it) and the span handed to the bridge.
//
// Coefficient planes as the GPU TX stage writes them: raster order over the
// *adjusted* transform size, one plane each, indexed from BlockPlan's
// coeff_offset_*.  eob == 0 planes never read the array, so an empty span is
// legal for an all-skip frame.
struct GpuCoefficients {
  std::span<const std::int32_t> y;
  std::span<const std::int32_t> u;
  std::span<const std::int32_t> v;
};

// Pack a GPU frame with an explicit partition tree.
bool pack_gpu_frame(const cuda::m1gpu::FramePlan& plan,
                    std::span<const TreeNode> tree,
                    std::span<const cuda::m1gpu::BlockPlan> blocks,
                    std::span<const cuda::m1gpu::BlockDecision> decisions,
                    const GpuCoefficients& coeffs, PackedFrame& out,
                    std::string& error);

// Pack a GPU frame whose plan is the fallback raster (one PARTITION_NONE block
// per superblock).  The spec's allowed/forced partition flags are derived here
// exactly as partition::solve_node() derives them, so the emitted do_split
// symbols match the CPU encoder in force_single_block mode.
bool pack_gpu_raster_frame(const cuda::m1gpu::FramePlan& plan,
                           std::span<const cuda::m1gpu::BlockPlan> blocks,
                           std::span<const cuda::m1gpu::BlockDecision> decisions,
                           const GpuCoefficients& coeffs, PackedFrame& out,
                           std::string& error);

// Pack + assemble in one call: GPU PODs -> finished temporal unit.
bool assemble_gpu_frame(const cuda::m1gpu::FramePlan& plan,
                        std::span<const TreeNode> tree,
                        std::span<const cuda::m1gpu::BlockPlan> blocks,
                        std::span<const cuda::m1gpu::BlockDecision> decisions,
                        const GpuCoefficients& coeffs,
                        const M1Config& config, std::vector<std::uint8_t>& out,
                        std::string& error, std::size_t* tile_bytes = nullptr);

bool assemble_gpu_raster_frame(
    const cuda::m1gpu::FramePlan& plan,
    std::span<const cuda::m1gpu::BlockPlan> blocks,
    std::span<const cuda::m1gpu::BlockDecision> decisions,
    const GpuCoefficients& coeffs, const M1Config& config,
    std::vector<std::uint8_t>& out, std::string& error,
    std::size_t* tile_bytes = nullptr);

// ---------------------------------------------------------------------------
// Host plan module (src/cuda/m1/plan/) -> syntax
// ---------------------------------------------------------------------------
//
// `m1::plan::FramePlan::tree` is already the preorder node list the writer
// needs, and `FramePlan::blocks` is the leaf-only view in the same order, so
// packing is a 1:1 conversion plus the coefficient interleave.  The bridge
// validates the plan's preorder bookkeeping (`index`, `first_child`,
// `child_count`, `subtree`, `leaf_index`) against a recomputation and against
// `plan.nodes` / `plan.leaves`.
//
// `plan_tree_to_nodes()` converts + validates the exported tree: preorder
// bookkeeping (`index`, `first_child`, `child_count`, `subtree`), leaf order
// (`leaf_index` must be 0,1,2,... into `plan.blocks`), and the spec-derived
// `allowed_mask` / `forced` flags with the plan's own has_chroma/chromaOffset
// recursion.  A tree that cannot be coded is rejected before any bit is
// written.  (Deriving the tree from the leaf list alone is ambiguous — a 32x32
// node made of four 16x16 leaves can be HORZ→VERT or VERT→HORZ — which is why
// the plan exports the tree.)
bool plan_tree_to_nodes(const plan::FramePlan& plan,
                        std::vector<TreeNode>& tree, std::string& error);

// Decisions are the same POD the GPU path uses (`BlockDecision`); one entry per
// plan leaf, in plan order.  Coefficients are indexed by the plan's per-plane
// `PlaneTxPlan::coeff_offset` and converted to the flat Y/U/V-interleaved
// layout exactly as described above.
bool pack_plan_frame(const plan::FramePlan& plan,
                     std::span<const plan::BlockPlan> blocks,
                     std::span<const cuda::m1gpu::BlockDecision> decisions,
                     const GpuCoefficients& coeffs, PackedFrame& out,
                     std::string& error);

bool assemble_plan_frame(const plan::FramePlan& plan,
                         std::span<const plan::BlockPlan> blocks,
                         std::span<const cuda::m1gpu::BlockDecision> decisions,
                         const GpuCoefficients& coeffs, const M1Config& config,
                         std::vector<std::uint8_t>& out, std::string& error,
                         std::size_t* tile_bytes = nullptr);

}  // namespace av2enc::m1::stream
