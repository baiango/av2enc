// Arena-backed intra workspace: tool dispatch implementations.
//
// Every device byte comes from the Arena captured by Workspace::init().  This
// translation unit contains no cudaMalloc / cudaFree / cudaMallocHost.

#include "cuda/codec/intra/workspace/intra_workspace.h"

namespace av2enc::cuda::intra_workspace {

// ---------------------------------------------------------------------------
// §7.13.2 intra modes
// ---------------------------------------------------------------------------

Status Workspace::modes_predict(const intra_modes::IntraJob* jobs, int count,
                                cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_jobs ||
      (count > 0 && jobs == nullptr)) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  Status status = stage_bytes(
      jobs, sizeof(intra_modes::IntraJob) * static_cast<std::size_t>(count),
      stream);
  if (!is_ok(status)) {
    return status;
  }
  const auto* d_jobs = static_cast<const intra_modes::IntraJob*>(staging_);
  return status_from_cuda(
      intra_modes::intra_predict_batch(d_jobs, count, stream));
}

// ---------------------------------------------------------------------------
// §7.13.2.1/.7/.8 multiple reference lines
// ---------------------------------------------------------------------------

Status Workspace::mrl_predict(const mrl::MrlJob* jobs, int count,
                              cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_jobs ||
      (count > 0 && jobs == nullptr)) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  Status status = stage_bytes(jobs,
                              sizeof(mrl::MrlJob) *
                                  static_cast<std::size_t>(count),
                              stream);
  if (!is_ok(status)) {
    return status;
  }
  const auto* d_jobs = static_cast<const mrl::MrlJob*>(staging_);
  return status_from_cuda(mrl::mrl_predict_batch(d_jobs, count, stream));
}

// ---------------------------------------------------------------------------
// §7.13.2.7/.9/.12 intra bi-prediction
// ---------------------------------------------------------------------------

Status Workspace::ibp_predict(const ibp::IbpJob* jobs, int count,
                              cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_jobs ||
      (count > 0 && jobs == nullptr)) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  Status status = stage_bytes(jobs,
                              sizeof(ibp::IbpJob) *
                                  static_cast<std::size_t>(count),
                              stream);
  if (!is_ok(status)) {
    return status;
  }
  const auto* d_jobs = static_cast<const ibp::IbpJob*>(staging_);
  return status_from_cuda(ibp::ibp_predict_batch(d_jobs, count, stream));
}

// ---------------------------------------------------------------------------
// §7.13.2.3-.6 data-driven intra prediction
// ---------------------------------------------------------------------------

Status Workspace::dip_predict(const dip::DipJob* jobs, int count,
                              cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_jobs ||
      (count > 0 && jobs == nullptr)) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  Status status = stage_bytes(jobs,
                              sizeof(dip::DipJob) *
                                  static_cast<std::size_t>(count),
                              stream);
  if (!is_ok(status)) {
    return status;
  }
  const auto* d_jobs = static_cast<const dip::DipJob*>(staging_);
  return status_from_cuda(dip::dip_predict_batch(d_jobs, count, stream));
}

// ---------------------------------------------------------------------------
// §7.13.5 chroma-from-luma
// ---------------------------------------------------------------------------

Status Workspace::cfl_predict(const CflJob* jobs, int count,
                              cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_jobs ||
      (count > 0 && jobs == nullptr)) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  Status status =
      stage_bytes(jobs, sizeof(CflJob) * static_cast<std::size_t>(count),
                  stream);
  if (!is_ok(status)) {
    return status;
  }
  const auto* d_jobs = static_cast<const CflJob*>(staging_);
  return status_from_cuda(cfl_predict_batch(d_jobs, count, stream));
}

Status Workspace::cfl_predict_derived(const cfl::Block* blocks,
                                      int* d_alpha_out, int count,
                                      cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_jobs ||
      (count > 0 && blocks == nullptr)) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  Status status = stage_bytes(
      blocks, sizeof(cfl::Block) * static_cast<std::size_t>(count), stream);
  if (!is_ok(status)) {
    return status;
  }
  const auto* d_blocks = static_cast<const cfl::Block*>(staging_);
  return status_from_cuda(
      cfl_predict_derived_batch(d_blocks, d_alpha_out, count, stream));
}

// ---------------------------------------------------------------------------
// §5.20.8 / §7.13.4 palette
// ---------------------------------------------------------------------------

Status Workspace::palette_apply(const PaletteDeviceBlock* blocks,
                                const PaletteContextDevice* contexts,
                                int count, cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  if (count < 0 || count > config_.max_jobs ||
      (count > 0 && blocks == nullptr)) {
    return Status::kInvalidArgument;
  }
  if (count == 0) {
    return Status::kOk;
  }
  const std::size_t block_bytes =
      sizeof(PaletteDeviceBlock) * static_cast<std::size_t>(count);
  if (block_bytes > detail::staging_bytes(config_)) {
    return Status::kInvalidArgument;
  }
  Status status = stage_bytes(blocks, block_bytes, stream);
  if (!is_ok(status)) {
    return status;
  }
  const auto* d_blocks = static_cast<const PaletteDeviceBlock*>(staging_);
  status = status_from_cuda(palette_build_map_batch(d_blocks, count, stream));
  if (!is_ok(status)) {
    return status;
  }
  status = status_from_cuda(palette_predict_batch(d_blocks, count, stream));
  if (!is_ok(status)) {
    return status;
  }
  if (contexts == nullptr) {
    return Status::kOk;
  }
  // The descriptor region is reused: this upload is ordered after the map and
  // prediction kernels on the same stream, so they have already read `blocks`.
  const std::size_t context_bytes =
      sizeof(PaletteContextDevice) * static_cast<std::size_t>(count);
  status = stage_bytes(contexts, context_bytes, stream);
  if (!is_ok(status)) {
    return status;
  }
  const auto* d_contexts =
      static_cast<const PaletteContextDevice*>(staging_);
  return status_from_cuda(
      palette_color_contexts_batch(d_contexts, count, stream));
}

// ---------------------------------------------------------------------------
// §7.13.4 / §6.19.7.12 IntraBC
// ---------------------------------------------------------------------------

Status Workspace::ibc_copy(const IbcDeviceFrame& frame, int mi_row, int mi_col,
                           int bsize, av2enc::ibc::BlockVector bv,
                           std::uint8_t* dst_y, int dst_stride_y,
                           std::uint8_t* dst_u, int dst_stride_u,
                           std::uint8_t* dst_v, int dst_stride_v,
                           int visible_w, int visible_h,
                           cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  return status_from_cuda(ibc_copy_block(frame, mi_row, mi_col, bsize, bv,
                                         dst_y, dst_stride_y, dst_u,
                                         dst_stride_u, dst_v, dst_stride_v,
                                         visible_w, visible_h, stream));
}

Status Workspace::ibc_sad(const IbcDeviceFrame& frame, int mi_row, int mi_col,
                          int bsize,
                          const av2enc::ibc::BlockVector* host_bvs, int n,
                          std::int64_t* d_sads, cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  if (n < 0 || n > config_.max_jobs || (n > 0 && host_bvs == nullptr) ||
      (n > 0 && d_sads == nullptr)) {
    return Status::kInvalidArgument;
  }
  if (n == 0) {
    return Status::kOk;
  }
  Status status = stage_bytes(
      host_bvs, sizeof(av2enc::ibc::BlockVector) * static_cast<std::size_t>(n),
      stream);
  if (!is_ok(status)) {
    return status;
  }
  const auto* d_bvs = static_cast<const av2enc::ibc::BlockVector*>(staging_);
  return status_from_cuda(
      ibc_sad_batch(frame, mi_row, mi_col, bsize, d_bvs, n, d_sads, stream));
}

Status Workspace::ibc_validate(const IbcDeviceGeometry& geometry, int mi_row,
                               int mi_col, int bsize,
                               const av2enc::ibc::BlockVector* host_bvs, int n,
                               std::uint8_t* d_valid, cudaStream_t stream) {
  if (!initialized()) {
    return Status::kNotInitialized;
  }
  if (n < 0 || n > config_.max_jobs || (n > 0 && host_bvs == nullptr) ||
      (n > 0 && d_valid == nullptr)) {
    return Status::kInvalidArgument;
  }
  if (n == 0) {
    return Status::kOk;
  }
  Status status = stage_bytes(
      host_bvs, sizeof(av2enc::ibc::BlockVector) * static_cast<std::size_t>(n),
      stream);
  if (!is_ok(status)) {
    return status;
  }
  const auto* d_bvs = static_cast<const av2enc::ibc::BlockVector*>(staging_);
  return status_from_cuda(ibc_validate_global(geometry, mi_row, mi_col, bsize,
                                              d_bvs, n, d_valid, stream));
}

}  // namespace av2enc::cuda::intra_workspace
