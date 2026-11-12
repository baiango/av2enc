// Host-emulation parity test for the CUDA TX_MODE_SELECT transform-partition
// components.  Needs no GPU and no nvcc.
//
// The module's shared device core (tx_partition_device.cuh) is compiled by the
// host compiler, and the real `av2enc::cuda::Arena` is linked against a tiny
// host CUDA runtime stub.  That makes the arena budget, the plan/job offsets,
// the per-part geometry, the 25x3 scan tables and the LumaTxSizes grid
// testable on macOS.  The transforms themselves are the CPU reference
// (`src/encoder/tx_variants/`), invoked through the exact `TxBlockJob`
// descriptors the GPU multi-kernel consumes.
//
// What still has to run on the sm_86 box: the real kernels (resolve, dequant,
// scan, grid) and the tx_variants `_multi` transforms over device buffers.  See
// build_gpu_test.sh / tx_partition_gpu_test.cu.
//
// Build/run:
//   ./build_and_test.sh
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/error.h"
#include "cuda/codec/transform/tx_partition/tx_partition_device.cuh"
#include "cuda/codec/transform/tx_partition/tx_partition_plan.h"
#include "codec/transform/tx_partition/tx_partition.h"
#include "codec/transform/tx_partition/tx_scan.h"
#include "codec/transform/variants/tx_variants.h"

namespace {

namespace cpu = av2enc::video::txpartition;
namespace txpart = av2enc::cuda::txpart;
namespace cuda = av2enc::cuda;
using av2enc::txv::TxParams;
using av2enc::txv::TxSize;
using av2enc::txv::TxType;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 40) {
      std::printf("FAIL [%d] %s\n", line, what);
    }
  }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

void check_status(cuda::Status actual, cuda::Status expected, int line) {
  ++g_checks;
  if (actual != expected) {
    ++g_failures;
    if (g_failures <= 40) {
      std::printf("FAIL [%d] status %s != %s\n", line,
                  cuda::status_name(actual), cuda::status_name(expected));
    }
  }
}

#define CHECK_STATUS(actual, expected) check_status((actual), (expected), __LINE__)

inline std::size_t flat(int y, int x, int width) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
         static_cast<std::size_t>(x);
}

// Deterministic value at an absolute pixel coordinate; catches a part that was
// sliced from the wrong origin.
std::int16_t coordinate_residual(int abs_y, int abs_x, int salt) {
  const int v = (abs_y * 131 + abs_x * 17 + salt * 29) % 511;
  return static_cast<std::int16_t>(v - 255);
}

template <typename T>
void check_vectors(const std::vector<T>& expected, const std::vector<T>& actual,
                   const char* what, int line, int max_reports = 3) {
  if (expected.size() != actual.size()) {
    ++g_checks;
    ++g_failures;
    if (g_failures <= 40) {
      std::printf("FAIL [%d] %s size %zu != %zu\n", line, what,
                  expected.size(), actual.size());
    }
    return;
  }
  int reported = 0;
  int mismatches = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    ++g_checks;
    if (expected[i] != actual[i]) {
      ++mismatches;
      ++g_failures;
      if (reported < max_reports && g_failures <= 40) {
        std::printf("FAIL [%d] %s[%zu] %g != %g\n", line, what, i,
                    static_cast<double>(expected[i]),
                    static_cast<double>(actual[i]));
        ++reported;
      }
    }
  }
  if (mismatches > max_reports && g_failures <= 40) {
    std::printf("FAIL [%d] %s: %d mismatches total\n", line, what, mismatches);
  }
}

#define CHECK_VECTORS(expected, actual, what) \
  check_vectors((expected), (actual), (what), __LINE__)

// FNV-1a over a byte range, used only for compact reporting.
std::uint64_t fnv1a(const void* data, std::size_t size) {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  std::uint64_t hash = 1469598103934665603ull;
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

// ---------------------------------------------------------------------------
// Geometry parity
// ---------------------------------------------------------------------------

void test_geometry() {
  const int origins[][2] = {{0, 0}, {4, 8}, {12, 3}};
  for (int mi = 0; mi < txpart::kBlockCount; ++mi) {
    for (int reduced = 0; reduced < 2; ++reduced) {
      const cpu::LegalSplits cpu_legal =
          cpu::legal_splits(mi, reduced != 0);
      const txpart::LegalSplits dev_legal =
          txpart::legal_splits(mi, reduced);
      CHECK(dev_legal.applicable == (cpu_legal.applicable ? 1 : 0));
      CHECK(dev_legal.chunked == (cpu_legal.chunked ? 1 : 0));
      CHECK(dev_legal.allow_horz == (cpu_legal.allow_horz ? 1 : 0));
      CHECK(dev_legal.allow_vert == (cpu_legal.allow_vert ? 1 : 0));
      CHECK(dev_legal.has_partition_type ==
            (cpu_legal.has_partition_type ? 1 : 0));
      CHECK(dev_legal.has_2or3 == (cpu_legal.has_2or3 ? 1 : 0));
      CHECK(dev_legal.option_count == cpu_legal.option_count);
      for (int i = 0; i < cpu_legal.option_count; ++i) {
        CHECK(dev_legal.options[i] == static_cast<int>(cpu_legal.options[i]));
      }
    }
    for (int part = 0; part < 8; ++part) {
      CHECK(txpart::partition_signallable(mi, part) ==
            (cpu::tx_partition_signallable(
                 mi, static_cast<cpu::TxPartition>(part))
                 ? 1
                 : 0));
      for (const auto& origin : origins) {
        cpu::SplitGeometry cpu_geom;
        const bool cpu_ok = cpu::tx_partition_geometry(
            mi, origin[0], origin[1], static_cast<cpu::TxPartition>(part),
            &cpu_geom, nullptr);
        txpart::PlanGeometry dev_geom;
        const bool dev_ok = txpart::plan_geometry(mi, part, origin[0],
                                                  origin[1], &dev_geom);
        CHECK(dev_ok == cpu_ok);
        if (dev_ok && cpu_ok) {
          CHECK(dev_geom.count == cpu_geom.count);
          for (int i = 0; i < cpu_geom.count; ++i) {
            CHECK(dev_geom.parts[i].mi_row == cpu_geom.parts[i].mi_row);
            CHECK(dev_geom.parts[i].mi_col == cpu_geom.parts[i].mi_col);
            CHECK(dev_geom.parts[i].tx_size == cpu_geom.parts[i].tx_size);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Scan parity: 25 sizes x 3 classes (rectangular included)
// ---------------------------------------------------------------------------

void test_scans() {
  std::vector<std::uint16_t> data(
      static_cast<std::size_t>(txpart::kScanTableCount) *
      txpart::kMaxScanArea);
  std::vector<std::uint16_t> length(
      static_cast<std::size_t>(txpart::kScanTableCount));
  txpart::ScanTableView view;
  view.data = data.data();
  view.length = length.data();

  for (int tx = 0; tx < 25; ++tx) {
    for (int cls = 0; cls < 3; ++cls) {
      CHECK(txpart::scan_table_fill_one(view, tx, cls));
      const int area = txpart::scan_area(tx);
      CHECK(area == cpu::tx_scan_area(tx));
      CHECK(length[static_cast<std::size_t>(txpart::scan_table_index(tx, cls))] ==
            static_cast<std::uint16_t>(area));
      std::vector<std::uint16_t> reference(static_cast<std::size_t>(area));
      CHECK(cpu::get_scan(tx, static_cast<cpu::TxScanClass>(cls),
                          std::span<std::uint16_t>(reference.data(),
                                                   reference.size())));
      std::vector<bool> seen(static_cast<std::size_t>(area), false);
      for (int pos = 0; pos < area; ++pos) {
        const std::uint16_t value =
            txpart::scan_table_at(view, tx, cls, pos);
        ++g_checks;
        if (value != reference[static_cast<std::size_t>(pos)]) {
          ++g_failures;
          if (g_failures <= 40) {
            std::printf("FAIL scan tx=%d cls=%d pos=%d %u != %u\n", tx, cls,
                        pos, static_cast<unsigned>(value),
                        static_cast<unsigned>(reference[static_cast<std::size_t>(pos)]));
          }
        }
        CHECK(value < static_cast<std::uint16_t>(area));
        if (value < static_cast<std::uint16_t>(area)) {
          CHECK(!seen[value]);
          seen[value] = true;
        }
      }
    }
  }
  CHECK(txpart::scan_table_at(view, 25, 0, 0) == txpart::kScanInvalid);
  CHECK(txpart::scan_table_at(view, 0, 3, 0) == txpart::kScanInvalid);
  std::printf("  scans: %zu table entries, fingerprint 0x%016llx\n",
              data.size(),
              static_cast<unsigned long long>(
                  fnv1a(data.data(), data.size() * sizeof(std::uint16_t))));
}

// ---------------------------------------------------------------------------
// Plan construction / grid parity
// ---------------------------------------------------------------------------

bool collect_requests(std::vector<txpart::PlanRequest>* requests,
                      std::vector<bool>* sizes_seen) {
  requests->clear();
  sizes_seen->assign(25, false);
  for (int mi = 1; mi < txpart::kBlockCount; ++mi) {
    for (int part = 0; part < 8; ++part) {
      txpart::PlanGeometry geom;
      if (!txpart::plan_geometry(mi, part, 0, 0, &geom)) continue;
      txpart::PlanRequest request;
      request.mi_size = mi;
      request.partition = part;
      request.mi_row = 0;
      request.mi_col = 0;
      request.tx_type = 0;
      requests->push_back(request);
      for (int i = 0; i < geom.count; ++i) {
        (*sizes_seen)[static_cast<std::size_t>(geom.parts[i].tx_size)] = true;
      }
    }
  }
  return true;
}

void test_plan_layout() {
  std::vector<txpart::PlanRequest> requests;
  std::vector<bool> sizes_seen;
  collect_requests(&requests, &sizes_seen);
  std::printf("  legal (block size, partition) pairs: %zu\n", requests.size());
  for (int s = 0; s < 25; ++s) {
    CHECK(sizes_seen[static_cast<std::size_t>(s)]);
  }
  CHECK(requests.size() >= 70);

  int check_steps = 0;
  for (const txpart::PlanRequest& request : requests) {
    txpart::PlanBatch batch;
    std::string error;
    CHECK_STATUS(txpart::make_batch(request, &batch, &error), cuda::Status::kOk);
    if (batch.plans.empty()) continue;

    cpu::SplitGeometry cpu_geom;
    CHECK(cpu::tx_partition_geometry(
        request.mi_size, request.mi_row, request.mi_col,
        static_cast<cpu::TxPartition>(request.partition), &cpu_geom, nullptr));
    CHECK(batch.total_parts == cpu_geom.count);
    CHECK(batch.plans[0].part_base == 0);
    CHECK(batch.plans[0].in_base == 0);
    CHECK(batch.plans[0].out_base == 0);

    std::int32_t expected_adj = 0;
    std::int32_t expected_coded = 0;
    for (int i = 0; i < cpu_geom.count; ++i) {
      expected_adj += txpart::scan_width(cpu_geom.parts[i].tx_size) *
                      txpart::scan_height(cpu_geom.parts[i].tx_size);
      expected_coded += txpart::tx_width(cpu_geom.parts[i].tx_size) *
                        txpart::tx_height(cpu_geom.parts[i].tx_size);
    }
    CHECK(batch.total_adj_elems == expected_adj);
    CHECK(batch.total_coded_elems == expected_coded);

    // The grid must match a direct §5.20.6.3 set_tx_size() walk.
    cpu::LumaTxGrid reference;
    reference.reset(64, 64);
    for (int i = 0; i < cpu_geom.count; ++i) {
      const int tx_size = cpu_geom.parts[i].tx_size;
      reference.set(cpu_geom.parts[i].mi_row, cpu_geom.parts[i].mi_col,
                    cpu::tx_height(tx_size) / 4, cpu::tx_width(tx_size) / 4,
                    tx_size);
    }
    const cpu::LumaTxGrid grid = txpart::build_luma_grid(batch, 64, 64);
    CHECK(grid.mi_rows == 64);
    CHECK(grid.mi_cols == 64);
    for (int row = 0; row < 64; ++row) {
      for (int col = 0; col < 64; ++col) {
        ++g_checks;
        if (grid.at(row, col) != reference.at(row, col)) {
          ++g_failures;
          if (g_failures <= 40) {
            std::printf("FAIL grid mi=%d part=%d cell(%d,%d) %u != %u\n",
                        request.mi_size, request.partition, row, col,
                        static_cast<unsigned>(grid.at(row, col)),
                        static_cast<unsigned>(reference.at(row, col)));
          }
        }
      }
    }
    ++check_steps;
  }
  std::printf("  plan layout cases: %d\n", check_steps);

  // Chunked blocks: one NONE plan per 64x64 chunk, mi_size == BLOCK_64X64.
  for (int mi = 1; mi < txpart::kBlockCount; ++mi) {
    const cpu::LegalSplits legal = cpu::legal_splits(mi, false);
    if (!legal.chunked) continue;
    std::vector<txpart::PlanRequest> chunk_requests;
    for (int row = 0; row < txpart::block_height_mi(mi); row += 16) {
      for (int col = 0; col < txpart::block_width_mi(mi); col += 16) {
        txpart::PlanRequest request;
        request.mi_size = 12;  // BLOCK_64X64
        request.partition = txpart::kPartNone;
        request.mi_row = row;
        request.mi_col = col;
        request.tx_type = 0;
        chunk_requests.push_back(request);
      }
    }
    txpart::PlanBatch batch;
    std::string error;
    CHECK_STATUS(txpart::make_batch(chunk_requests.data(),
                                    static_cast<int>(chunk_requests.size()),
                                    &batch, &error),
                 cuda::Status::kOk);
    const cpu::LumaTxGrid grid = txpart::build_luma_grid(
        batch, txpart::block_height_mi(mi), txpart::block_width_mi(mi));
    for (int row = 0; row < txpart::block_height_mi(mi); ++row) {
      for (int col = 0; col < txpart::block_width_mi(mi); ++col) {
        ++g_checks;
        if (grid.at(row, col) != 4) {  // TX_64X64
          ++g_failures;
          if (g_failures <= 40) {
            std::printf("FAIL chunk grid mi=%d cell(%d,%d) %u != 4\n", mi, row,
                        col, static_cast<unsigned>(grid.at(row, col)));
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Transform parity through the exact TxBlockJob descriptors
// ---------------------------------------------------------------------------

void resolve_scratch_plans(const txpart::Workspace& workspace,
                           txpart::LaunchScratch& scratch,
                           const txpart::PlanBatch& batch) {
  CHECK_STATUS(cuda::status_from_cuda(cudaMemcpyAsync(
                   scratch.d_plans, batch.plans.data(),
                   sizeof(txpart::PlanDesc) *
                       static_cast<std::size_t>(batch.plans.size()),
                   cudaMemcpyHostToDevice, nullptr)),
               cuda::Status::kOk);
  for (std::size_t i = 0; i < batch.plans.size(); ++i) {
    CHECK(txpart::plan_resolve(workspace.host_params.data(),
                               scratch.d_plans[i], scratch.d_jobs,
                               scratch.part_capacity, scratch.d_parts,
                               scratch.part_capacity));
  }
}

// Runs one plan batch through the emulated GPU path and compares it with an
// independent per-part CPU reference.  `arena` is used exactly like the GPU
// launcher: mark -> scratch -> rewind.
void run_plan_batch(cuda::Arena& arena, const txpart::Workspace& workspace,
                    const txpart::PlanRequest* requests, int request_count,
                    const char* label) {
  txpart::PlanBatch batch;
  std::string error;
  const cuda::Status batch_status =
      txpart::make_batch(requests, request_count, &batch, &error);
  CHECK_STATUS(batch_status, cuda::Status::kOk);
  if (batch_status != cuda::Status::kOk) return;

  std::vector<txpart::ResolvedPart> parts(
      static_cast<std::size_t>(batch.total_parts));
  CHECK_STATUS(txpart::build_host_parts(batch, parts.data(), batch.total_parts,
                                        &error),
               cuda::Status::kOk);

  std::vector<std::int16_t> adj(
      static_cast<std::size_t>(batch.total_adj_elems));
  std::vector<std::int16_t> coded;  // per-plan temporary
  std::vector<std::int32_t> expected_in(
      static_cast<std::size_t>(batch.total_parts));
  std::vector<std::int32_t> expected_out(
      static_cast<std::size_t>(batch.total_parts));

  // Build the adjusted residual for each part from a coded-size plan residual,
  // exactly like a real encoder: downsample the part's coded residual.
  for (const txpart::PlanDesc& plan : batch.plans) {
    const int tx_max = txpart::max_tx_size_rect(plan.mi_size);
    const int plan_w = txpart::tx_width(tx_max);
    const int plan_h = txpart::tx_height(tx_max);
    coded.resize(static_cast<std::size_t>(plan_w) *
                 static_cast<std::size_t>(plan_h));
    for (int y = 0; y < plan_h; ++y) {
      for (int x = 0; x < plan_w; ++x) {
        coded[flat(y, x, plan_w)] =
            coordinate_residual(plan.mi_row * 4 + y, plan.mi_col * 4 + x,
                                plan.tx_type);
      }
    }
    int adj_offset = plan.in_base;
    int out_offset = plan.out_base;
    txpart::PlanGeometry geom;
    CHECK(txpart::plan_geometry(plan.mi_size, plan.partition, plan.mi_row,
                                plan.mi_col, &geom));
    std::int32_t expected_adj_span = 0;
    for (int i = 0; i < geom.count; ++i) {
      const int index = plan.part_base + i;
      const int tx_size = geom.parts[i].tx_size;
      const TxParams params = workspace.host_params[static_cast<std::size_t>(
          plan.tx_type * txpart::kTxSizeCount + tx_size)];
      const int local_x = (geom.parts[i].mi_col - plan.mi_col) * 4;
      const int local_y = (geom.parts[i].mi_row - plan.mi_row) * 4;
      std::vector<std::int16_t> sub(static_cast<std::size_t>(params.coded_w) *
                                    static_cast<std::size_t>(params.coded_h));
      for (int y = 0; y < params.coded_h; ++y) {
        for (int x = 0; x < params.coded_w; ++x) {
          sub[flat(y, x, params.coded_w)] =
              coded[flat(local_y + y, local_x + x, plan_w)];
        }
      }
      av2enc::txv::downsample_residual(
          sub.data(), params, adj.data() + adj_offset);
      expected_in[static_cast<std::size_t>(index)] = adj_offset;
      expected_out[static_cast<std::size_t>(index)] = out_offset;
      adj_offset += params.w * params.h;
      out_offset += params.coded_w * params.coded_h;
      expected_adj_span += params.w * params.h;
    }
    CHECK(adj_offset == plan.in_base + expected_adj_span);
  }

  // Independent CPU reference: one forward/inverse per part.
  std::vector<double> coeff_ref(
      static_cast<std::size_t>(batch.total_adj_elems));
  std::vector<std::int32_t> levels_ref(
      static_cast<std::size_t>(batch.total_adj_elems));
  std::vector<std::int32_t> dequant_ref(
      static_cast<std::size_t>(batch.total_adj_elems));
  std::vector<std::int16_t> recon_ref(
      static_cast<std::size_t>(batch.total_coded_elems));
  for (const txpart::PlanDesc& plan : batch.plans) {
    txpart::PlanGeometry geom;
    CHECK(txpart::plan_geometry(plan.mi_size, plan.partition, plan.mi_row,
                                plan.mi_col, &geom));
    for (int i = 0; i < geom.count; ++i) {
      const int index = plan.part_base + i;
      const int tx_size = geom.parts[i].tx_size;
      const TxParams params = workspace.host_params[static_cast<std::size_t>(
          plan.tx_type * txpart::kTxSizeCount + tx_size)];
      const std::int32_t in = expected_in[static_cast<std::size_t>(index)];
      const std::int32_t out = expected_out[static_cast<std::size_t>(index)];
      av2enc::txv::forward_transform(adj.data() + in, params,
                                     coeff_ref.data() + in,
                                     levels_ref.data() + in);
      av2enc::txv::dequantize(levels_ref.data() + in, params,
                              dequant_ref.data() + in);
      av2enc::txv::inverse_block(dequant_ref.data() + in, params,
                                 recon_ref.data() + out);
    }
  }

  // Emulated GPU launch: acquire scratch, upload plans, resolve on the "device"
  // (shared plan_resolve), run the job list.
  const std::uint64_t mark = arena.mark();
  txpart::LaunchScratch scratch;
  CHECK_STATUS(txpart::acquire_scratch(
                   arena, workspace, static_cast<int>(batch.plans.size()),
                   &scratch, nullptr, &error),
               cuda::Status::kOk);
  resolve_scratch_plans(workspace, scratch, batch);

  std::vector<double> coeff(
      static_cast<std::size_t>(batch.total_adj_elems));
  std::vector<std::int32_t> levels(
      static_cast<std::size_t>(batch.total_adj_elems));
  std::vector<std::int16_t> recon(
      static_cast<std::size_t>(batch.total_coded_elems));
  for (int j = 0; j < batch.total_parts; ++j) {
    const av2enc::cuda::TxBlockJob& job = scratch.d_jobs[j];
    const int index = j;
    CHECK(job.in_offset == expected_in[static_cast<std::size_t>(index)]);
    CHECK(job.out_offset == expected_out[static_cast<std::size_t>(index)]);
    CHECK(job.params.tx_size == parts[static_cast<std::size_t>(index)].tx_size);
    av2enc::txv::forward_transform(adj.data() + job.in_offset, job.params,
                                   coeff.data() + job.in_offset,
                                   levels.data() + job.in_offset);
    // Fused inverse (matches txv_inverse_transform_multi).
    av2enc::txv::inverse_transform(levels.data() + job.in_offset, job.params,
                                   dequant_ref.data() + job.in_offset,
                                   recon.data() + job.out_offset);
  }

  CHECK_VECTORS(coeff_ref, coeff, label);
  CHECK_VECTORS(levels_ref, levels, label);

  // The fused inverse writes dequant_ref as scratch; rebuild it for a clean
  // comparison and then run the explicit dequant -> inverse_block path.
  for (const txpart::PlanDesc& plan : batch.plans) {
    txpart::PlanGeometry geom;
    (void)txpart::plan_geometry(plan.mi_size, plan.partition, plan.mi_row,
                                plan.mi_col, &geom);
    for (int i = 0; i < geom.count; ++i) {
      const int index = plan.part_base + i;
      const int tx_size = geom.parts[i].tx_size;
      const TxParams params = workspace.host_params[static_cast<std::size_t>(
          plan.tx_type * txpart::kTxSizeCount + tx_size)];
      av2enc::txv::dequantize(levels_ref.data() +
                                  expected_in[static_cast<std::size_t>(index)],
                              params,
                              dequant_ref.data() +
                                  expected_in[static_cast<std::size_t>(index)]);
    }
  }
  CHECK_VECTORS(recon_ref, recon, label);

  // In-place inverse: the coded-size output aliases the adjusted-residual
  // buffer.  The transform only reads levels, so this is legal scratch reuse.
  std::vector<std::int16_t> recon_inplace(
      std::max(adj.size(), recon_ref.size()), 0);
  std::memcpy(recon_inplace.data(), adj.data(),
              adj.size() * sizeof(std::int16_t));
  for (int j = 0; j < batch.total_parts; ++j) {
    const av2enc::cuda::TxBlockJob& job = scratch.d_jobs[j];
    av2enc::txv::inverse_transform(levels.data() + job.in_offset, job.params,
                                   dequant_ref.data() + job.in_offset,
                                   recon_inplace.data() + job.out_offset);
  }
  CHECK_VECTORS(recon_ref, recon_inplace, label);

  // In-place dequant: levels buffer used as the dequant destination.
  std::vector<std::int32_t> dequant_inplace = levels;
  for (int j = 0; j < batch.total_parts; ++j) {
    const av2enc::cuda::TxBlockJob& job = scratch.d_jobs[j];
    av2enc::txv::dequantize(dequant_inplace.data() + job.in_offset, job.params,
                            dequant_inplace.data() + job.in_offset);
  }
  CHECK_VECTORS(dequant_ref, dequant_inplace, label);

  txpart::release_scratch(arena, scratch);
  CHECK(arena.used() == mark);
}

void test_transforms(cuda::Arena& arena, const txpart::Workspace& workspace) {
  std::vector<txpart::PlanRequest> single_requests;
  std::vector<bool> sizes_seen;
  collect_requests(&single_requests, &sizes_seen);

  int cases = 0;
  for (const txpart::PlanRequest& base : single_requests) {
    for (int tx_type = 0; tx_type < 16; ++tx_type) {
      txpart::PlanRequest request = base;
      request.tx_type = tx_type;
      run_plan_batch(arena, workspace, &request, 1, "single-plan");
      ++cases;
    }
  }
  std::printf("  single-plan transform cases: %d\n", cases);

  // Every chunk of every >64x64 block.
  int chunk_cases = 0;
  for (int mi = 1; mi < txpart::kBlockCount; ++mi) {
    const cpu::LegalSplits legal = cpu::legal_splits(mi, false);
    if (!legal.chunked) continue;
    for (int row = 0; row < txpart::block_height_mi(mi); row += 16) {
      for (int col = 0; col < txpart::block_width_mi(mi); col += 16) {
        txpart::PlanRequest request;
        request.mi_size = 12;
        request.partition = txpart::kPartNone;
        request.mi_row = row;
        request.mi_col = col;
        for (int tx_type = 0; tx_type < 16; ++tx_type) {
          request.tx_type = tx_type;
          run_plan_batch(arena, workspace, &request, 1, "chunk-plan");
          ++chunk_cases;
        }
      }
    }
  }
  std::printf("  chunk-plan transform cases: %d\n", chunk_cases);
}

void test_batch_mode(cuda::Arena& arena, const txpart::Workspace& workspace) {
  std::vector<txpart::PlanRequest> requests;
  std::vector<bool> sizes_seen;
  collect_requests(&requests, &sizes_seen);
  // Mix both partition decisions and TX types in one arena scratch window.
  std::vector<txpart::PlanRequest> mixed;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    txpart::PlanRequest request = requests[i];
    request.tx_type = static_cast<int>((i * 5) % 16);
    mixed.push_back(request);
    if (i % 7 == 3) {
      // Add a chunked-block plan at a non-zero origin.
      txpart::PlanRequest chunk;
      chunk.mi_size = 12;
      chunk.partition = txpart::kPartNone;
      chunk.mi_row = 16;
      chunk.mi_col = 32;
      chunk.tx_type = static_cast<int>((i * 3) % 16);
      mixed.push_back(chunk);
    }
  }
  CHECK(static_cast<int>(mixed.size()) <= workspace.max_plans);
  run_plan_batch(arena, workspace, mixed.data(),
                 static_cast<int>(mixed.size()), "mixed batch");
  std::printf("  batch plans: %zu (parts: see workspace)\n", mixed.size());

  // A second, distinct batch through the same arena window.
  std::vector<txpart::PlanRequest> second;
  for (std::size_t i = 0; i < requests.size(); i += 2) {
    txpart::PlanRequest request = requests[i];
    request.tx_type = static_cast<int>((i * 11 + 1) % 16);
    second.push_back(request);
  }
  run_plan_batch(arena, workspace, second.data(),
                 static_cast<int>(second.size()), "batch 2");
}

// ---------------------------------------------------------------------------
// Arena sizing, exactness and leak behaviour
// ---------------------------------------------------------------------------

void test_arena(const txpart::LaunchConfig& config) {
  const std::uint64_t required = txpart::required_device_bytes(config);
  const std::uint64_t persistent =
      txpart::persistent_device_bytes(config);
  const std::uint64_t scratch = txpart::scratch_device_bytes(config);
  CHECK(required == persistent + scratch);
  CHECK(required % cuda::Arena::kAlignment == 0);

  std::printf("  required_device_bytes(%d plans) = %llu\n", config.max_plans,
              static_cast<unsigned long long>(required));
  std::printf("    persistent = %llu, scratch = %llu\n",
              static_cast<unsigned long long>(persistent),
              static_cast<unsigned long long>(scratch));

  // Exact fit: the workspace takes exactly `persistent` and leaves exactly the
  // configured scratch.
  {
    cuda::Arena arena;
    CHECK_STATUS(arena.init(required), cuda::Status::kOk);
    txpart::Workspace workspace;
    std::uint64_t required_out = 0;
    std::string error;
    CHECK_STATUS(workspace.init(arena, config, &required_out, &error),
                 cuda::Status::kOk);
    CHECK(required_out == required);
    CHECK(arena.used() == persistent);
    CHECK(arena.remaining() == scratch);

    // The uploaded tables must be the CPU tables (the emulation "device"
    // memory is host memory).
    for (int tx_type = 0; tx_type < 16; ++tx_type) {
      for (int tx_size = 0; tx_size < 25; ++tx_size) {
        const TxParams& params = workspace.d_params[static_cast<std::size_t>(
            tx_type * 25 + tx_size)];
        const TxParams& host = workspace.host_params[static_cast<std::size_t>(
            tx_type * 25 + tx_size)];
        CHECK(params.tx_size == host.tx_size);
        CHECK(params.tx_type == host.tx_type);
        CHECK(params.coded_w == host.coded_w);
        CHECK(params.coded_h == host.coded_h);
        CHECK(params.w == host.w);
        CHECK(params.h == host.h);
        CHECK(params.scale_w == host.scale_w);
        CHECK(params.scale_h == host.scale_h);
        CHECK(params.quant_factor == host.quant_factor);
      }
    }
    for (int tx_size = 0; tx_size < 25; ++tx_size) {
      for (int cls = 0; cls < 3; ++cls) {
        const int index = txpart::scan_table_index(tx_size, cls);
        CHECK(workspace.d_scan_length[static_cast<std::size_t>(index)] ==
              static_cast<std::uint16_t>(txpart::scan_area(tx_size)));
        for (int pos = 0; pos < txpart::scan_area(tx_size); ++pos) {
          ++g_checks;
          const std::uint16_t value =
              workspace.d_scan_data[static_cast<std::size_t>(index) *
                                        static_cast<std::size_t>(
                                            txpart::kMaxScanArea) +
                                    static_cast<std::size_t>(pos)];
          if (value >= static_cast<std::uint16_t>(txpart::scan_area(tx_size))) {
            ++g_failures;
          }
        }
      }
    }

    // Two launches back to back, each with the full scratch window, must return
    // the arena exactly to the mark (no leak).
    const std::uint64_t mark = arena.mark();
    for (int launch = 0; launch < 2; ++launch) {
      txpart::LaunchScratch window;
      CHECK_STATUS(txpart::acquire_scratch(arena, workspace, config.max_plans,
                                           &window, &required_out, &error),
                   cuda::Status::kOk);
      CHECK(arena.used() == required);
      txpart::release_scratch(arena, window);
      CHECK(arena.used() == mark);
      CHECK(arena.remaining() == scratch);
    }
    CHECK(arena.peak_used() == required);

    // A third window that is one plan over max_plans is rejected without
    // touching the arena.
    txpart::LaunchScratch too_big;
    CHECK_STATUS(txpart::acquire_scratch(arena, workspace,
                                         config.max_plans + 1, &too_big,
                                         &required_out, &error),
                 cuda::Status::kInvalidArgument);
    CHECK(too_big.mark == txpart::kNoArenaMark);
    CHECK(arena.used() == mark);
    arena.release();
  }

  // One byte short: the workspace init fails cleanly and reports the exact
  // requirement.  The caller can release and re-init a larger arena.
  {
    cuda::Arena arena;
    CHECK_STATUS(arena.init(required - 1), cuda::Status::kOk);
    txpart::Workspace workspace;
    std::uint64_t required_out = 0;
    std::string error;
    const cuda::Status status =
        workspace.init(arena, config, &required_out, &error);
    CHECK_STATUS(status, cuda::Status::kArenaExhausted);
    CHECK(required_out == required);
    CHECK(!workspace.valid());
    CHECK(arena.used() == 0);
    CHECK(!error.empty());
    arena.release();
  }
}

}  // namespace

int main() {
  const txpart::LaunchConfig config{1024};

  std::printf("av2enc CUDA transform-partition host-emulation test\n");
  std::printf("config: max_plans=%d (max parts=%d)\n", config.max_plans,
              config.max_plans * txpart::kMaxParts);

  test_geometry();
  test_scans();
  test_plan_layout();

  {
    cuda::Arena arena;
    std::uint64_t required = 0;
    std::string error;
    const std::uint64_t need = txpart::required_device_bytes(config);
    // The transform tests need their own host-side buffers only; the arena is
    // exactly the component budget, proving the workspace + scratch fit.
    CHECK_STATUS(arena.init(need), cuda::Status::kOk);
    txpart::Workspace workspace;
    CHECK_STATUS(workspace.init(arena, config, &required, &error),
                 cuda::Status::kOk);
    test_transforms(arena, workspace);
    test_batch_mode(arena, workspace);
    arena.release();
  }

  test_arena(config);

  std::printf("\n%lld checks, %d failures\n",
              static_cast<long long>(g_checks), g_failures);
  std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
