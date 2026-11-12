// Real-GPU parity test for the CUDA TX_MODE_SELECT transform partition.
//
// Build/run on the sm_86 box:
//   ./build_gpu_test.sh
//
// Everything is allocated from one persistent av2enc::cuda::Arena sized from
// required_device_bytes(); the module itself never allocates.  The test checks
//   * device geometry (through the device LumaTxSizes grid builder),
//   * the device 25x3 scan table against the CPU §5.20.7.30 generator,
//   * batched forward/inverse/dequant against the CPU tx_variants reference
//     through the same TxBlockJob descriptors,
//   * in-place inverse and in-place dequant,
//   * arena exact fit, two back-to-back launches with mark/rewind (no leak),
//     and a one-byte-short arena that fails cleanly with the required size.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/error.h"
#include "cuda/codec/transform/tx_partition/tx_partition_device.cuh"
#include "cuda/codec/transform/tx_partition/tx_partition_gpu.h"
#include "cuda/codec/transform/tx_partition/tx_partition_plan.h"
#include "codec/transform/tx_partition/tx_partition.h"
#include "codec/transform/tx_partition/tx_scan.h"
#include "codec/transform/variants/tx_variants.h"

namespace {

namespace cpu = av2enc::video::txpartition;
namespace txpart = av2enc::cuda::txpart;
namespace cuda = av2enc::cuda;
using av2enc::txv::TxParams;

constexpr int kGridSide = 64;
constexpr std::size_t kGridCells =
    static_cast<std::size_t>(kGridSide) * static_cast<std::size_t>(kGridSide);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 40) std::printf("FAIL [%d] %s\n", line, what);
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

#define CHECK_STATUS(actual, expected) \
  check_status((actual), (expected), __LINE__)

#define CUDA_TRY(expr, what)                                                \
  do {                                                                      \
    const cudaError_t err_ = (expr);                                        \
    if (err_ != cudaSuccess) {                                              \
      std::printf("CUDA error %s at %s: %s\n", cudaGetErrorString(err_),    \
                  what, #expr);                                             \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

std::uint32_t g_rng = 0x2545f491u;

std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

template <typename T>
void upload(T* device, const std::vector<T>& host) {
  if (host.empty()) return;
  CUDA_TRY(cudaMemcpy(device, host.data(), sizeof(T) * host.size(),
                      cudaMemcpyHostToDevice),
           "upload");
}

template <typename T>
void download(std::vector<T>* host, const T* device) {
  if (host->empty()) return;
  CUDA_TRY(cudaMemcpy(host->data(), device, sizeof(T) * host->size(),
                      cudaMemcpyDeviceToHost),
           "download");
}

template <typename T>
void check_vectors(const std::vector<T>& expected, const std::vector<T>& actual,
                   const char* what, int line) {
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
  for (std::size_t i = 0; i < expected.size(); ++i) {
    ++g_checks;
    if (expected[i] != actual[i]) {
      ++g_failures;
      if (reported < 3 && g_failures <= 40) {
        std::printf("FAIL [%d] %s[%zu] %g != %g\n", line, what, i,
                    static_cast<double>(expected[i]),
                    static_cast<double>(actual[i]));
        ++reported;
      }
    }
  }
}

#define CHECK_VECTORS(expected, actual, what) \
  check_vectors((expected), (actual), (what), __LINE__)

// ---------------------------------------------------------------------------
// Legal plan requests
// ---------------------------------------------------------------------------
std::vector<txpart::PlanRequest> collect_legal_requests() {
  std::vector<txpart::PlanRequest> requests;
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
      requests.push_back(request);
    }
  }
  return requests;
}

std::vector<txpart::PlanRequest> collect_chunk_requests() {
  std::vector<txpart::PlanRequest> requests;
  for (int mi = 1; mi < txpart::kBlockCount; ++mi) {
    const cpu::LegalSplits legal = cpu::legal_splits(mi, false);
    if (!legal.chunked) continue;
    for (int row = 0; row < txpart::block_height_mi(mi); row += 16) {
      for (int col = 0; col < txpart::block_width_mi(mi); col += 16) {
        txpart::PlanRequest request;
        request.mi_size = 12;  // BLOCK_64X64
        request.partition = txpart::kPartNone;
        request.mi_row = row;
        request.mi_col = col;
        request.tx_type = 0;
        requests.push_back(request);
      }
    }
  }
  return requests;
}

std::uint64_t align_up_bytes(std::uint64_t bytes, std::uint64_t alignment) {
  return (bytes + alignment - 1) & ~(alignment - 1);
}

struct DeviceBuffers {
  std::int16_t* samples = nullptr;   // max(max_adj, max_coded) int16
  double* coeff = nullptr;           // max_adj double
  std::int32_t* levels = nullptr;    // max_adj int32
  std::int32_t* dequant = nullptr;   // max_adj int32
  std::int16_t* recon = nullptr;     // max_coded int16
  std::int16_t* recon_second = nullptr;  // max_coded int16
  std::uint16_t* scan_data = nullptr;
  std::uint16_t* scan_length = nullptr;
  std::uint8_t* grid_tx = nullptr;
  std::uint8_t* grid_middle = nullptr;
  std::uint8_t* grid_scan = nullptr;
  std::uint64_t bytes = 0;
};

std::uint64_t device_buffer_bytes(const txpart::PlanBatch& batch) {
  const std::uint64_t adj = static_cast<std::uint64_t>(batch.total_adj_elems);
  const std::uint64_t coded =
      static_cast<std::uint64_t>(batch.total_coded_elems);
  const std::uint64_t samples = std::max(adj, coded);
  std::uint64_t total = 0;
  total += align_up_bytes(samples * sizeof(std::int16_t), cuda::Arena::kAlignment);
  total += align_up_bytes(adj * sizeof(double), cuda::Arena::kAlignment);
  total += align_up_bytes(adj * sizeof(std::int32_t), cuda::Arena::kAlignment);
  total += align_up_bytes(adj * sizeof(std::int32_t), cuda::Arena::kAlignment);
  total += align_up_bytes(coded * sizeof(std::int16_t), cuda::Arena::kAlignment);
  total += align_up_bytes(coded * sizeof(std::int16_t), cuda::Arena::kAlignment);
  total += align_up_bytes(static_cast<std::uint64_t>(
                              txpart::kScanTableCount) *
                              txpart::kMaxScanArea * sizeof(std::uint16_t),
                          cuda::Arena::kAlignment);
  total += align_up_bytes(static_cast<std::uint64_t>(txpart::kScanTableCount) *
                              sizeof(std::uint16_t),
                          cuda::Arena::kAlignment);
  total += align_up_bytes(static_cast<std::uint64_t>(3 * kGridSide * kGridSide),
                          cuda::Arena::kAlignment);
  return total;
}

bool allocate_device_buffers(cuda::Arena& arena,
                             const txpart::PlanBatch& batch,
                             DeviceBuffers* buffers) {
  const std::uint64_t adj = static_cast<std::uint64_t>(batch.total_adj_elems);
  const std::uint64_t coded =
      static_cast<std::uint64_t>(batch.total_coded_elems);
  const std::uint64_t samples = std::max(adj, coded);
  buffers->samples = arena.allocate_array<std::int16_t>(samples);
  buffers->coeff = arena.allocate_array<double>(adj);
  buffers->levels = arena.allocate_array<std::int32_t>(adj);
  buffers->dequant = arena.allocate_array<std::int32_t>(adj);
  buffers->recon = arena.allocate_array<std::int16_t>(coded);
  buffers->recon_second = arena.allocate_array<std::int16_t>(coded);
  buffers->scan_data = arena.allocate_array<std::uint16_t>(
      static_cast<std::size_t>(txpart::kScanTableCount) *
      txpart::kMaxScanArea);
  buffers->scan_length = arena.allocate_array<std::uint16_t>(
      static_cast<std::size_t>(txpart::kScanTableCount));
  buffers->grid_tx = arena.allocate_array<std::uint8_t>(kGridCells);
  buffers->grid_middle = arena.allocate_array<std::uint8_t>(kGridCells);
  buffers->grid_scan = arena.allocate_array<std::uint8_t>(kGridCells);
  buffers->bytes = device_buffer_bytes(batch);
  return buffers->samples != nullptr && buffers->coeff != nullptr &&
         buffers->levels != nullptr && buffers->dequant != nullptr &&
         buffers->recon != nullptr && buffers->recon_second != nullptr &&
         buffers->scan_data != nullptr && buffers->scan_length != nullptr &&
         buffers->grid_tx != nullptr && buffers->grid_middle != nullptr &&
         buffers->grid_scan != nullptr;
}

// ---------------------------------------------------------------------------
// Device scans
// ---------------------------------------------------------------------------

void test_device_scans(const DeviceBuffers& buffers) {
  txpart::ScanTableView view;
  view.data = buffers.scan_data;
  view.length = buffers.scan_length;
  CHECK_STATUS(txpart::build_scan_tables_device(view, nullptr),
               cuda::Status::kOk);
  CUDA_TRY(cudaDeviceSynchronize(), "scan sync");

  std::vector<std::uint16_t> host_data(
      static_cast<std::size_t>(txpart::kScanTableCount) * txpart::kMaxScanArea);
  std::vector<std::uint16_t> host_length(
      static_cast<std::size_t>(txpart::kScanTableCount));
  download(&host_data, buffers.scan_data);
  download(&host_length, buffers.scan_length);

  int table_checks = 0;
  for (int tx = 0; tx < 25; ++tx) {
    for (int cls = 0; cls < 3; ++cls) {
      const int index = txpart::scan_table_index(tx, cls);
      const int area = txpart::scan_area(tx);
      CHECK(host_length[static_cast<std::size_t>(index)] ==
            static_cast<std::uint16_t>(area));
      std::vector<std::uint16_t> reference(static_cast<std::size_t>(area));
      CHECK(cpu::get_scan(tx, static_cast<cpu::TxScanClass>(cls),
                          std::span<std::uint16_t>(reference.data(),
                                                   reference.size())));
      for (int pos = 0; pos < area; ++pos) {
        ++g_checks;
        const std::uint16_t value =
            host_data[static_cast<std::size_t>(index) *
                          static_cast<std::size_t>(txpart::kMaxScanArea) +
                      static_cast<std::size_t>(pos)];
        if (value != reference[static_cast<std::size_t>(pos)]) {
          ++g_failures;
          if (g_failures <= 40) {
            std::printf("FAIL device scan tx=%d cls=%d pos=%d %u != %u\n", tx,
                        cls, pos, static_cast<unsigned>(value),
                        static_cast<unsigned>(
                            reference[static_cast<std::size_t>(pos)]));
          }
        }
      }
      ++table_checks;
    }
  }
  std::printf("  device scans: %d (size,class) tables\n", table_checks);
}

// ---------------------------------------------------------------------------
// Device grid / geometry
// ---------------------------------------------------------------------------

void compare_grid(const cpu::LumaTxGrid& reference,
                  const DeviceBuffers& buffers, const char* label) {
  std::vector<std::uint8_t> tx(kGridCells);
  std::vector<std::uint8_t> middle(kGridCells);
  std::vector<std::uint8_t> scan(kGridCells);
  download(&tx, buffers.grid_tx);
  download(&middle, buffers.grid_middle);
  download(&scan, buffers.grid_scan);
  for (int row = 0; row < 64; ++row) {
    for (int col = 0; col < 64; ++col) {
      const std::size_t index = static_cast<std::size_t>(row) * 64 +
                                static_cast<std::size_t>(col);
      ++g_checks;
      if (tx[index] != reference.at(row, col) ||
          middle[index] != reference.middle_at(row, col) ||
          scan[index] != reference.scan_order_at(row, col)) {
        ++g_failures;
        if (g_failures <= 40) {
          std::printf("FAIL %s grid cell(%d,%d) %u/%u/%u != %u/%u/%u\n", label,
                      row, col, static_cast<unsigned>(tx[index]),
                      static_cast<unsigned>(middle[index]),
                      static_cast<unsigned>(scan[index]),
                      static_cast<unsigned>(reference.at(row, col)),
                      static_cast<unsigned>(reference.middle_at(row, col)),
                      static_cast<unsigned>(reference.scan_order_at(row, col)));
        }
      }
    }
  }
}

void test_device_grid(cuda::Arena& arena, txpart::Workspace& workspace,
                      const DeviceBuffers& buffers) {
  std::vector<std::uint8_t> zeros(kGridCells, 0);
  txpart::TxGridView grid;
  grid.tx_size = buffers.grid_tx;
  grid.middle = buffers.grid_middle;
  grid.scan_order = buffers.grid_scan;
  grid.mi_rows = kGridSide;
  grid.mi_cols = kGridSide;

  const std::vector<txpart::PlanRequest> legal = collect_legal_requests();
  const std::vector<txpart::PlanRequest> chunks = collect_chunk_requests();
  std::vector<txpart::PlanRequest> all = legal;
  all.insert(all.end(), chunks.begin(), chunks.end());

  int cases = 0;
  for (const txpart::PlanRequest& request : all) {
    txpart::PlanBatch batch;
    std::string error;
    CHECK_STATUS(txpart::make_batch(request, &batch, &error),
                 cuda::Status::kOk);
    if (batch.plans.empty()) continue;
    CHECK(static_cast<int>(batch.plans.size()) <= workspace.max_plans);
    for (int pass = 0; pass < 2; ++pass) {
      upload(buffers.grid_tx, zeros);
      upload(buffers.grid_middle, zeros);
      upload(buffers.grid_scan, zeros);
      CHECK_STATUS(txpart::build_grid_device(arena, workspace, batch, grid,
                                             nullptr, nullptr, &error),
                   cuda::Status::kOk);
      CUDA_TRY(cudaDeviceSynchronize(), "grid sync");
      if (pass == 0) {
        const cpu::LumaTxGrid reference = txpart::build_luma_grid(batch, 64, 64);
        compare_grid(reference, buffers, "device");
      } else {
        // Second pass through the same arena window (mark/rewind exercised by
        // the launcher) must produce identical bytes.
        std::vector<std::uint8_t> tx(kGridCells);
        download(&tx, buffers.grid_tx);
        const cpu::LumaTxGrid reference = txpart::build_luma_grid(batch, 64, 64);
        for (int row = 0; row < 64; ++row) {
          for (int col = 0; col < 64; ++col) {
            ++g_checks;
            if (tx[static_cast<std::size_t>(row) * 64 +
                   static_cast<std::size_t>(col)] != reference.at(row, col)) {
              ++g_failures;
            }
          }
        }
      }
    }
    ++cases;
  }
  std::printf("  device grid/geometry cases: %d (legal + chunked)\n", cases);
}

// ---------------------------------------------------------------------------
// Batched transforms
// ---------------------------------------------------------------------------

void run_transform_batch(cuda::Arena& arena, txpart::Workspace& workspace,
                         DeviceBuffers& buffers,
                         const std::vector<txpart::PlanRequest>& requests,
                         const char* label) {
  txpart::PlanBatch batch;
  std::string error;
  CHECK_STATUS(txpart::make_batch(requests.data(),
                                  static_cast<int>(requests.size()), &batch,
                                  &error),
               cuda::Status::kOk);
  CHECK(static_cast<int>(batch.plans.size()) <= workspace.max_plans);

  std::vector<av2enc::cuda::TxBlockJob> jobs(
      static_cast<std::size_t>(batch.total_parts));
  CHECK_STATUS(txpart::build_host_jobs(workspace, batch, jobs.data(),
                                       batch.total_parts, &error),
               cuda::Status::kOk);

  const int adj_n = batch.total_adj_elems;
  const int coded_n = batch.total_coded_elems;
  std::vector<std::int16_t> adj(static_cast<std::size_t>(adj_n));
  std::vector<double> coeff_ref(static_cast<std::size_t>(adj_n));
  std::vector<std::int32_t> levels_ref(static_cast<std::size_t>(adj_n));
  std::vector<std::int32_t> dequant_ref(static_cast<std::size_t>(adj_n));
  std::vector<std::int16_t> recon_ref(static_cast<std::size_t>(coded_n));

  for (int j = 0; j < batch.total_parts; ++j) {
    const av2enc::cuda::TxBlockJob& job = jobs[static_cast<std::size_t>(j)];
    const int elems = job.params.w * job.params.h;
    for (int i = 0; i < elems; ++i) {
      adj[static_cast<std::size_t>(job.in_offset + i)] =
          static_cast<std::int16_t>(static_cast<int>(next_random() % 511u) -
                                    255);
    }
    av2enc::txv::forward_transform(adj.data() + job.in_offset, job.params,
                                   coeff_ref.data() + job.in_offset,
                                   levels_ref.data() + job.in_offset);
    av2enc::txv::dequantize(levels_ref.data() + job.in_offset, job.params,
                            dequant_ref.data() + job.in_offset);
    av2enc::txv::inverse_block(dequant_ref.data() + job.in_offset, job.params,
                               recon_ref.data() + job.out_offset);
  }

  upload(buffers.samples, adj);
  CHECK_STATUS(txpart::forward_batch(arena, workspace, batch,
                                     buffers.samples, buffers.coeff,
                                     buffers.levels, nullptr, nullptr, &error),
               cuda::Status::kOk);
  CUDA_TRY(cudaDeviceSynchronize(), "forward sync");

  std::vector<double> coeff(static_cast<std::size_t>(adj_n));
  std::vector<std::int32_t> levels(static_cast<std::size_t>(adj_n));
  download(&coeff, buffers.coeff);
  download(&levels, buffers.levels);
  CHECK_VECTORS(coeff_ref, coeff, label);
  CHECK_VECTORS(levels_ref, levels, label);

  // Out-of-place inverse.
  upload(buffers.levels, levels_ref);
  CHECK_STATUS(txpart::inverse_batch(arena, workspace, batch, buffers.levels,
                                     buffers.recon, nullptr, nullptr, &error),
               cuda::Status::kOk);
  CUDA_TRY(cudaDeviceSynchronize(), "inverse sync");
  std::vector<std::int16_t> recon(static_cast<std::size_t>(coded_n));
  download(&recon, buffers.recon);
  CHECK_VECTORS(recon_ref, recon, label);

  // In-place inverse: the coded-size output aliases the adjusted residual
  // buffer (only levels are read).
  upload(buffers.samples, adj);
  CHECK_STATUS(txpart::inverse_batch(arena, workspace, batch, buffers.levels,
                                     buffers.samples, nullptr, nullptr, &error),
               cuda::Status::kOk);
  CUDA_TRY(cudaDeviceSynchronize(), "in-place inverse sync");
  std::vector<std::int16_t> recon_inplace(static_cast<std::size_t>(coded_n));
  download(&recon_inplace, buffers.samples);
  CHECK_VECTORS(recon_ref, recon_inplace, label);

  // Dequant, out of place.
  upload(buffers.levels, levels_ref);
  CHECK_STATUS(txpart::dequantize_batch(arena, workspace, batch, buffers.levels,
                                        buffers.dequant, nullptr, nullptr,
                                        &error),
               cuda::Status::kOk);
  CUDA_TRY(cudaDeviceSynchronize(), "dequant sync");
  std::vector<std::int32_t> dequant(static_cast<std::size_t>(adj_n));
  download(&dequant, buffers.dequant);
  CHECK_VECTORS(dequant_ref, dequant, label);

  // Dequant in place: levels used as the destination, then inverse_block.
  upload(buffers.levels, levels_ref);
  CHECK_STATUS(txpart::dequantize_batch(arena, workspace, batch, buffers.levels,
                                        buffers.levels, nullptr, nullptr,
                                        &error),
               cuda::Status::kOk);
  CUDA_TRY(cudaDeviceSynchronize(), "in-place dequant sync");
  std::vector<std::int32_t> dequant_inplace(static_cast<std::size_t>(adj_n));
  download(&dequant_inplace, buffers.levels);
  CHECK_VECTORS(dequant_ref, dequant_inplace, label);

  // Explicit dequant -> inverse_block path (fresh upload).
  upload(buffers.levels, levels_ref);
  CHECK_STATUS(txpart::inverse_dequant_batch(arena, workspace, batch,
                                             buffers.levels, buffers.dequant,
                                             buffers.recon, nullptr, nullptr,
                                             &error),
               cuda::Status::kOk);
  CUDA_TRY(cudaDeviceSynchronize(), "inverse_dequant sync");
  std::vector<std::int16_t> recon_path(static_cast<std::size_t>(coded_n));
  download(&recon_path, buffers.recon);
  CHECK_VECTORS(recon_ref, recon_path, label);

  std::printf("  %s: %zu plans, %d parts, %d adj elems, %d coded elems\n",
              label, batch.plans.size(), batch.total_parts, adj_n, coded_n);
}

void test_transforms(cuda::Arena& arena, txpart::Workspace& workspace,
                     DeviceBuffers& buffers) {
  std::vector<txpart::PlanRequest> all = collect_legal_requests();
  const std::vector<txpart::PlanRequest> chunks = collect_chunk_requests();
  all.insert(all.end(), chunks.begin(), chunks.end());

  // One large mixed batch covering every legal partition and all 25 TX sizes,
  // four TX_TYPE families.
  const int tx_types[4] = {0, 3, 9, 15};
  std::vector<txpart::PlanRequest> batch_requests;
  for (int tx_type : tx_types) {
    for (const txpart::PlanRequest& base : all) {
      txpart::PlanRequest request = base;
      request.tx_type = tx_type;
      batch_requests.push_back(request);
    }
  }
  CHECK(static_cast<int>(batch_requests.size()) <= workspace.max_plans);
  run_transform_batch(arena, workspace, buffers, batch_requests, "mixed batch");

  // A smaller second batch, back to back with the first, exercises the arena
  // reuse and a different plan count.
  std::vector<txpart::PlanRequest> second;
  for (std::size_t i = 0; i < all.size(); i += 3) {
    txpart::PlanRequest request = all[i];
    request.tx_type = static_cast<int>((i * 7) % 16);
    second.push_back(request);
  }
  run_transform_batch(arena, workspace, buffers, second, "batch 2");
}

}  // namespace

int main() {
  cudaDeviceProp prop{};
  CUDA_TRY(cudaGetDeviceProperties(&prop, 0), "device properties");
  std::printf("av2enc CUDA transform-partition test\n");
  std::printf("device: %s (sm_%d%d, %d SMs, %.1f GiB)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount,
              static_cast<double>(prop.totalGlobalMem) /
                  (1024.0 * 1024.0 * 1024.0));

  const txpart::LaunchConfig config{1024};
  const std::uint64_t required = txpart::required_device_bytes(config);
  const std::uint64_t persistent = txpart::persistent_device_bytes(config);
  const std::uint64_t scratch = txpart::scratch_device_bytes(config);
  CHECK(required == persistent + scratch);
  CHECK(required % cuda::Arena::kAlignment == 0);
  std::printf("config: max_plans=%d\n", config.max_plans);
  std::printf("required_device_bytes = %llu (persistent %llu, scratch %llu)\n",
              static_cast<unsigned long long>(required),
              static_cast<unsigned long long>(persistent),
              static_cast<unsigned long long>(scratch));

  // Build the largest batch first so the arena can include the test buffers.
  std::vector<txpart::PlanRequest> probe = collect_legal_requests();
  const std::vector<txpart::PlanRequest> probe_chunks = collect_chunk_requests();
  probe.insert(probe.end(), probe_chunks.begin(), probe_chunks.end());
  const int tx_types[4] = {0, 3, 9, 15};
  std::vector<txpart::PlanRequest> full_batch;
  for (int tx_type : tx_types) {
    for (const txpart::PlanRequest& base : probe) {
      txpart::PlanRequest request = base;
      request.tx_type = tx_type;
      full_batch.push_back(request);
    }
  }
  txpart::PlanBatch sizing_batch;
  std::string error;
  CHECK_STATUS(txpart::make_batch(full_batch.data(),
                                  static_cast<int>(full_batch.size()),
                                  &sizing_batch, &error),
               cuda::Status::kOk);
  const std::uint64_t test_bytes = device_buffer_bytes(sizing_batch);
  const std::uint64_t arena_bytes =
      required + test_bytes + cuda::Arena::kAlignment;
  std::printf("arena: %llu bytes (component %llu + test buffers %llu)\n",
              static_cast<unsigned long long>(arena_bytes),
              static_cast<unsigned long long>(required),
              static_cast<unsigned long long>(test_bytes));

  cuda::Arena arena;
  CHECK_STATUS(arena.init(arena_bytes), cuda::Status::kOk);
  txpart::Workspace workspace;
  std::uint64_t required_out = 0;
  CHECK_STATUS(workspace.init(arena, config, &required_out, &error),
               cuda::Status::kOk);
  CHECK(required_out == required);
  CHECK(arena.used() == persistent);
  std::printf("arena after workspace init: used=%llu remaining=%llu\n",
              static_cast<unsigned long long>(arena.used()),
              static_cast<unsigned long long>(arena.remaining()));

  DeviceBuffers buffers;
  CHECK(allocate_device_buffers(arena, sizing_batch, &buffers));
  const std::uint64_t test_mark = arena.mark();

  test_device_scans(buffers);
  test_device_grid(arena, workspace, buffers);
  test_transforms(arena, workspace, buffers);

  // Two full-batch launches back to back, then the arena must be exactly at the
  // post-allocation mark (no leak).
  const std::uint64_t mark = arena.mark();
  for (int launch = 0; launch < 2; ++launch) {
    run_transform_batch(arena, workspace, buffers, full_batch, "leak check");
    CHECK(arena.used() == mark);
  }
  CUDA_TRY(cudaDeviceSynchronize(), "leak sync");
  CHECK(arena.used() == mark);
  std::printf("arena peak_used = %llu (test mark %llu)\n",
              static_cast<unsigned long long>(arena.peak_used()),
              static_cast<unsigned long long>(test_mark));

  // One byte short: a separate arena must fail cleanly with the exact
  // requirement, and must not consume anything.
  {
    cuda::Arena short_arena;
    CHECK_STATUS(short_arena.init(required - 1), cuda::Status::kOk);
    txpart::Workspace short_workspace;
    std::uint64_t short_required = 0;
    std::string short_error;
    const cuda::Status status = short_workspace.init(
        short_arena, config, &short_required, &short_error);
    CHECK_STATUS(status, cuda::Status::kArenaExhausted);
    CHECK(short_required == required);
    CHECK(!short_workspace.valid());
    CHECK(short_arena.used() == 0);
    std::printf("short arena: status=%s required=%llu\n",
                cuda::status_name(status),
                static_cast<unsigned long long>(short_required));
    short_arena.release();
  }

  arena.release();
  std::printf("\n%lld checks, %d failures\n",
              static_cast<long long>(g_checks), g_failures);
  std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
