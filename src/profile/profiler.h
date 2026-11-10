// Low-overhead, always-available profiler for the av2enc encoder.
//
// Design goals:
//   * Works on the Vast.ai box where Nsight Compute hardware counters are
//     blocked (ERR_NVGPUCTRPERM) and `perf` is usually unavailable.
//   * Disabled cost is one predictable branch per scope; no file, no clock.
//   * Enabled cost is two steady_clock reads per CPU scope and one JSONL line.
//   * CUDA stages use cudaEventRecord only. Events are resolved in
//     Profiler::end_frame(), which performs at most one synchronization per
//     stream per frame -- never one sync per kernel.
//   * No heap allocation in the hot path after stage registration / CUDA event
//     pool creation.
//
// Usage:
//
//   av2enc::profile::Profiler profiler;
//   profiler.open_from_env();              // AV2ENC_PROFILE=<path>
//   profiler.begin_frame(frame_index);
//   {
//     av2enc::profile::CpuScope scope(profiler,
//                                     av2enc::profile::stages::kInputDecode);
//     ...
//   }
//   {
//     // `stream` is a cudaStream_t; nullptr means the default stream.
//     av2enc::profile::CudaScope scope(profiler,
//                                      av2enc::profile::stages::kIntraSearch,
//                                      stream);
//     launch_intra_search(...);
//   }
//   profiler.end_frame();                  // CUDA event resolution + one sync
//   ...
//   profiler.close();                      // writes <path>.summary.json
//
// Environment:
//   AV2ENC_PROFILE=<path>       Write per-event JSONL to <path>.
//   AV2ENC_PROFILE_NVTX=1       Emit NVTX ranges (requires AV2ENC_HAVE_NVTX).
//                               May be used without AV2ENC_PROFILE for a
//                               stats-only/Nsight Systems run.
//   AV2ENC_VRAM_LIMIT_MB=<MiB>  VRAM accounting budget.  Default 7500 (the
//                               8 GB compatibility target); 0 disables the
//                               over-budget warning.
//
// The summary always records precompute_profile (default "compat8g") and
// vram_limit_mb.  Compatibility profiles above 8 GB are frozen estimates.
//
// Everything is a no-op when the profiler is disabled.

#ifndef AV2ENC_PROFILE_PROFILER_H_
#define AV2ENC_PROFILE_PROFILER_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>

namespace av2enc::profile {

using StageId = std::uint16_t;
inline constexpr StageId kInvalidStage = 0xFFFFU;
inline constexpr std::uint32_t kMaxStages = 64;
inline constexpr std::size_t kMaxStageName = 48;
inline constexpr std::uint32_t kInvalidCudaToken = 0xFFFFFFFFU;

// Focus freeze: the primary target is 8 GB Ampere.  compat8g is the default
// and only supported configuration.  12/16/24 GB profiles are frozen estimates
// and are not implemented by the profiler.
inline constexpr std::string_view kDefaultPrecomputeProfile = "compat8g";
inline constexpr std::uint32_t kDefaultVramLimitMb = 7500;
inline constexpr std::size_t kMaxPrecomputeProfile = 32;

class Profiler;

// Canonical stage names.  Use these ids for zero-lookup scopes.  The JSON
// summary keeps CPU and CUDA samples separate because a stage such as
// intra_search usually has both a host launch interval and a device interval.
namespace stages {

inline constexpr StageId kInputDecode = 0;      // input/decode
inline constexpr StageId kTileSchedule = 1;     // tile_schedule
inline constexpr StageId kIntraSearch = 2;      // intra_search
inline constexpr StageId kInterSearch = 3;      // inter_search
inline constexpr StageId kRdo = 4;              // rdo
inline constexpr StageId kTransformQuant = 5;   // transform_quant
inline constexpr StageId kRecon = 6;            // recon
inline constexpr StageId kLoopFilters = 7;      // loop_filters
inline constexpr StageId kEntropy = 8;          // entropy
inline constexpr StageId kBitstream = 9;        // bitstream
inline constexpr StageId kHostSync = 10;        // host_sync
inline constexpr StageId kTotal = 11;           // total (frame wall time)
inline constexpr std::uint32_t kBuiltinCount = 12;

inline constexpr std::string_view kNames[kBuiltinCount] = {
    "input_decode", "tile_schedule", "intra_search", "inter_search",
    "rdo",          "transform_quant", "recon",     "loop_filters",
    "entropy",      "bitstream",      "host_sync", "total"};

}  // namespace stages

// RAII wall-clock scope for a CPU stage or function.
class CpuScope {
 public:
  CpuScope(Profiler& profiler, StageId stage);
  CpuScope(Profiler& profiler, std::string_view stage_name);
  ~CpuScope();

  CpuScope(const CpuScope&) = delete;
  CpuScope& operator=(const CpuScope&) = delete;
  CpuScope(CpuScope&&) = delete;
  CpuScope& operator=(CpuScope&&) = delete;

 private:
  Profiler* profiler_ = nullptr;
  StageId stage_ = kInvalidStage;
  bool active_ = false;
  bool nvtx_ = false;
  std::uint64_t start_ns_ = 0;
  std::uint64_t frame_ = 0;
};

// RAII CUDA stage scope.  `stream` is a cudaStream_t; pass nullptr for the
// default stream.  The constructor records a start event, the destructor
// records an end event, and Profiler::end_frame() resolves elapsed time and
// synchronizes each stream at most once per frame.
//
// Do not construct CudaScope while a CUDA graph is being captured.  Wrap the
// graph launch itself instead.
//
// The stream handle is typed as void* so this header has no CUDA dependency;
// C++ allows an implicit cudaStream_t -> void* conversion.  CUDA event support
// requires compiling profiler.cpp with AV2ENC_HAVE_CUDA=1.  Without CUDA the
// class is a no-op (NVTX annotation still works if NVTX support is compiled).
class CudaScope {
 public:
  CudaScope(Profiler& profiler, StageId stage, void* stream);
  CudaScope(Profiler& profiler, std::string_view stage_name, void* stream);
  ~CudaScope();

  CudaScope(const CudaScope&) = delete;
  CudaScope& operator=(const CudaScope&) = delete;
  CudaScope(CudaScope&&) = delete;
  CudaScope& operator=(CudaScope&&) = delete;

 private:
  Profiler* profiler_ = nullptr;
  StageId stage_ = kInvalidStage;
#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  std::uint32_t token_ = kInvalidCudaToken;
#endif
  bool nvtx_ = false;
};

class Profiler {
 public:
  Profiler();
  ~Profiler();

  Profiler(const Profiler&) = delete;
  Profiler& operator=(const Profiler&) = delete;

  // Open the JSONL log.  If jsonl_path is empty and enable_nvtx is true, the
  // profiler runs in NVTX/stats-only mode.  Returns true if any sink is active.
  // Re-opening resets all counters and stage statistics.
  bool open(const std::string& jsonl_path, bool enable_nvtx,
            std::string* error = nullptr);

  // Read AV2ENC_PROFILE and AV2ENC_PROFILE_NVTX.  Returns true when enabled.
  // No environment variables means "disabled", which is not an error.
  bool open_from_env(std::string* error = nullptr);

  // Resolve pending CUDA events, write the summary sidecar, flush and close
  // the JSONL file.  Idempotent.  Also called by the destructor.
  void close();

  bool enabled() const noexcept {
    return enabled_.load(std::memory_order_relaxed);
  }
  bool nvtx_enabled() const noexcept {
    return nvtx_enabled_.load(std::memory_order_relaxed);
  }
  const std::string& jsonl_path() const noexcept { return jsonl_path_; }
  const std::string& summary_path() const noexcept { return summary_path_; }

  // Frame protocol.  begin_frame/end_frame must be called from the launch
  // thread.  end_frame() is the only place where pending CUDA events are
  // resolved and streams are synchronized.
  void begin_frame(std::uint64_t frame_index);
  void end_frame();
  std::uint64_t current_frame() const noexcept {
    return current_frame_.load(std::memory_order_relaxed);
  }

  // Register a stage name once during warm-up.  Returns kInvalidStage when the
  // 64-entry table is full or the name is empty.  Lookup is lock-free after
  // registration; no heap allocation is performed.
  StageId stage(std::string_view name) noexcept;
  std::string_view stage_name(StageId id) const noexcept;

  // Record the precompute profile name in the summary.  The default is
  // "compat8g" (8 GB Ampere target, 7500 MiB cap); set this before encoding
  // when the CLI exposes --precompute.  Names are copied and truncated to 31
  // characters; an empty name restores "compat8g".
  void set_precompute_profile(std::string_view name) noexcept;
  std::string_view precompute_profile() const noexcept;

  void flush();
  void print_summary(std::ostream& out) const;
  std::uint64_t dropped_cuda_events() const noexcept;

  // Sample device-wide VRAM with cudaMemGetInfo() and update the peak used
  // value.  Called automatically at every end_frame() and close(); call it
  // manually after large persistent allocations to catch the true peak.
  // No-op when the profiler is disabled or CUDA is unavailable.
  void sample_vram() noexcept;

 private:
  friend class CpuScope;
  friend class CudaScope;

  struct TimingStats {
    std::uint64_t calls = 0;
    std::uint64_t total_ns = 0;
    std::uint64_t min_ns = 0;
    std::uint64_t max_ns = 0;
  };

  struct StageSlot {
    char name[kMaxStageName] = {};
    std::uint32_t name_len = 0;
    TimingStats cpu;
    TimingStats cuda;
  };

  struct ThreadSlot {
    std::uint64_t id = 0;
    std::uint64_t cpu_calls = 0;
    std::uint64_t cuda_calls = 0;
  };

  static constexpr std::uint32_t kMaxThreads = 32;

#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  struct CudaState;
  static constexpr std::uint32_t kMaxPendingCuda = 512;
  static constexpr std::uint32_t kMaxCudaStreams = 32;
#endif

  static std::uint64_t now_ns() noexcept;
  static std::uint64_t thread_id() noexcept;
  static void add_sample(TimingStats& stats, std::uint64_t ns) noexcept;

  void reset_stats_locked() noexcept;
  void record_thread_locked(std::uint64_t thread, bool cuda) noexcept;

  void record_cpu(StageId stage, std::uint64_t start_ns, std::uint64_t end_ns,
                  std::uint64_t thread, std::uint64_t frame) noexcept;
  void record_cuda_locked(StageId stage, std::uint64_t duration_ns,
                          std::uint64_t thread, std::uint64_t frame,
                          std::uint64_t at_ns) noexcept;

#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  bool cuda_pool_init_locked() noexcept;
  std::uint32_t cuda_begin(StageId stage, void* stream,
                           std::uint64_t at_ns) noexcept;
  void cuda_end(std::uint32_t token) noexcept;
  void cuda_flush() noexcept;
  void vram_init_locked() noexcept;
  void vram_sample_locked() noexcept;
  void vram_update_peak_locked(std::uint64_t free_bytes,
                               std::uint64_t total_bytes) noexcept;
#endif

  bool nvtx_push_cpu(StageId stage) noexcept;
  void nvtx_pop_cpu() noexcept;
  bool nvtx_push_cuda(StageId stage) noexcept;
  void nvtx_pop_cuda() noexcept;
  bool nvtx_init_locked() noexcept;
  void nvtx_shutdown_locked() noexcept;

  void write_event_locked(StageId stage, const char* kind,
                          std::uint64_t thread, std::uint64_t frame,
                          std::uint64_t at_ns, std::uint64_t dur_ns) noexcept;
  void write_summary_locked() noexcept;
  void flush_file_locked() noexcept;

  std::array<StageSlot, kMaxStages> stages_{};
  std::array<ThreadSlot, kMaxThreads> threads_{};
  std::atomic<std::uint32_t> stage_count_{0};

  char precompute_profile_[kMaxPrecomputeProfile] = {};
  std::uint32_t precompute_profile_len_ = 0;
  std::uint32_t vram_limit_mb_ = kDefaultVramLimitMb;

  mutable std::mutex stage_mutex_;  // registration only
  mutable std::mutex mutex_;        // file, statistics, CUDA event pool
  std::FILE* file_ = nullptr;
  std::string jsonl_path_;
  std::string summary_path_;

  std::atomic<bool> enabled_{false};
  std::atomic<bool> nvtx_enabled_{false};
  std::atomic<std::uint64_t> current_frame_{0};

  std::uint64_t origin_ns_ = 0;
  std::uint64_t frame_start_ns_ = 0;
  std::uint64_t wall_ns_ = 0;
  std::uint64_t sequence_ = 0;
  std::uint64_t event_count_ = 0;
  std::uint64_t frame_count_ = 0;
  std::uint32_t thread_count_ = 0;
  bool frame_active_ = false;
  bool write_error_ = false;
  bool warned_stage_full_ = false;
  bool warned_write_ = false;

  void* nvtx_cpu_domain_ = nullptr;
  void* nvtx_cuda_domain_ = nullptr;

#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  std::unique_ptr<CudaState> cuda_;
  std::uint64_t cuda_dropped_ = 0;
  std::uint64_t cuda_errors_ = 0;
  bool warned_cuda_overflow_ = false;
  bool warned_cuda_error_ = false;

  bool vram_available_ = false;
  std::uint64_t vram_capacity_bytes_ = 0;
  std::uint64_t vram_free_bytes_ = 0;
  std::uint64_t vram_peak_used_bytes_ = 0;
  bool warned_vram_limit_ = false;
  bool warned_vram_query_ = false;
#endif
};

}  // namespace av2enc::profile

#endif  // AV2ENC_PROFILE_PROFILER_H_
