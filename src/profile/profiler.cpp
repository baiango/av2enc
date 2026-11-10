#include "profile/profiler.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <new>
#include <string_view>
#include <thread>
#include <utility>

#if defined(AV2ENC_HAVE_NVTX) && AV2ENC_HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace av2enc::profile {
namespace {

constexpr std::size_t kEventLineCapacity = 1024;
#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
constexpr std::uint32_t kCudaTokenIndexBits = 9;  // 512 pending events
constexpr std::uint32_t kCudaTokenIndexMask = (1U << kCudaTokenIndexBits) - 1U;
constexpr std::uint32_t kCudaGenerationMask = 0x7FFFFFU;  // 23 bits
#endif

int format_ms(char* out, std::size_t capacity, std::uint64_t ns) noexcept {
  // Six fractional digits: 1e-6 ms == 1 ns.  Integer math keeps the JSON
  // locale-independent and avoids floating-point formatting in the hot path.
  const std::uint64_t whole = ns / 1'000'000ULL;
  const std::uint64_t fraction = ns % 1'000'000ULL;
  return std::snprintf(out, capacity, "%llu.%06llu",
                       static_cast<unsigned long long>(whole),
                       static_cast<unsigned long long>(fraction));
}

#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
int format_mb(char* out, std::size_t capacity, std::uint64_t bytes) noexcept {
  // MB means MiB throughout the profiler.  Two fractional digits.
  constexpr std::uint64_t kMiB = 1024ULL * 1024ULL;
  const std::uint64_t whole = bytes / kMiB;
  const std::uint64_t fraction = (bytes % kMiB) * 100ULL / kMiB;
  return std::snprintf(out, capacity, "%llu.%02llu",
                       static_cast<unsigned long long>(whole),
                       static_cast<unsigned long long>(fraction));
}
#endif

void write_json_string(std::FILE* file, const char* text,
                       std::size_t length) noexcept {
  std::fputc('"', file);
  for (std::size_t i = 0; i < length; ++i) {
    const unsigned char value = static_cast<unsigned char>(text[i]);
    switch (value) {
      case '"':
        std::fputs("\\\"", file);
        break;
      case '\\':
        std::fputs("\\\\", file);
        break;
      case '\b':
        std::fputs("\\b", file);
        break;
      case '\f':
        std::fputs("\\f", file);
        break;
      case '\n':
        std::fputs("\\n", file);
        break;
      case '\r':
        std::fputs("\\r", file);
        break;
      case '\t':
        std::fputs("\\t", file);
        break;
      default:
        if (value < 0x20U) {
          std::fprintf(file, "\\u%04x", static_cast<unsigned int>(value));
        } else {
          std::fputc(static_cast<int>(value), file);
        }
        break;
    }
  }
  std::fputc('"', file);
}

void write_json_string(std::FILE* file, std::string_view text) noexcept {
  write_json_string(file, text.data(), text.size());
}

// Small fixed-capacity JSON line builder.  No allocation, bounded stack use.
class LineBuilder {
 public:
  LineBuilder(char* data, std::size_t capacity) noexcept
      : data_(data), capacity_(capacity) {}

  bool ok() const noexcept { return ok_; }
  std::size_t size() const noexcept { return length_; }
  const char* data() const noexcept { return data_; }

  void append(const char* text, std::size_t length) noexcept {
    if (!ok_) {
      return;
    }
    if (length > capacity_ - length_) {
      ok_ = false;
      return;
    }
    std::memcpy(data_ + length_, text, length);
    length_ += length;
  }

  void append(const char* text) noexcept { append(text, std::strlen(text)); }

  void appendf(const char* format, ...) noexcept {
    if (!ok_) {
      return;
    }
    va_list args;
    va_start(args, format);
    const int written =
        std::vsnprintf(data_ + length_, capacity_ - length_, format, args);
    va_end(args);
    if (written < 0 ||
        static_cast<std::size_t>(written) >= capacity_ - length_) {
      ok_ = false;
      return;
    }
    length_ += static_cast<std::size_t>(written);
  }

  void append_json(const char* text, std::size_t length) noexcept {
    if (!ok_) {
      return;
    }
    append("\"");
    for (std::size_t i = 0; i < length; ++i) {
      const unsigned char value = static_cast<unsigned char>(text[i]);
      switch (value) {
        case '"':
          append("\\\"");
          break;
        case '\\':
          append("\\\\");
          break;
        case '\b':
          append("\\b");
          break;
        case '\f':
          append("\\f");
          break;
        case '\n':
          append("\\n");
          break;
        case '\r':
          append("\\r");
          break;
        case '\t':
          append("\\t");
          break;
        default:
          if (value < 0x20U) {
            char escape[7];
            const int written = std::snprintf(
                escape, sizeof(escape), "\\u%04x",
                static_cast<unsigned int>(value));
            if (written > 0) {
              append(escape, static_cast<std::size_t>(written));
            }
          } else {
            const char character = static_cast<char>(value);
            append(&character, 1);
          }
          break;
      }
    }
    append("\"");
  }

  void append_json(std::string_view text) noexcept {
    append_json(text.data(), text.size());
  }

  void append_ms(std::uint64_t ns) noexcept {
    char buffer[40];
    const int written = format_ms(buffer, sizeof(buffer), ns);
    if (written > 0) {
      append(buffer, static_cast<std::size_t>(written));
    } else {
      ok_ = false;
    }
  }

 private:
  char* data_;
  std::size_t capacity_;
  std::size_t length_ = 0;
  bool ok_ = true;
};

bool env_flag(const char* value) noexcept {
  if (value == nullptr || *value == '\0') {
    return false;
  }
  std::string_view text(value);
  char lowered[8] = {};
  const std::size_t length = std::min(text.size(), sizeof(lowered) - 1U);
  for (std::size_t i = 0; i < length; ++i) {
    const char character = text[i];
    lowered[i] = (character >= 'A' && character <= 'Z')
                     ? static_cast<char>(character - 'A' + 'a')
                     : character;
  }
  return std::strcmp(lowered, "1") == 0 || std::strcmp(lowered, "true") == 0 ||
         std::strcmp(lowered, "yes") == 0 || std::strcmp(lowered, "on") == 0;
}

std::uint32_t parse_vram_limit_mb() noexcept {
  const char* value = std::getenv("AV2ENC_VRAM_LIMIT_MB");
  if (value == nullptr || *value == '\0') {
    return kDefaultVramLimitMb;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed > 1000000UL) {
    std::fprintf(stderr,
                 "av2enc profile: invalid AV2ENC_VRAM_LIMIT_MB='%s'; using "
                 "%u\n",
                 value, kDefaultVramLimitMb);
    return kDefaultVramLimitMb;
  }
  return static_cast<std::uint32_t>(parsed);
}

#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
void warn_cuda_error(const char* what, cudaError_t error) noexcept {
  std::fprintf(stderr, "av2enc profile: %s: %s\n", what,
               cudaGetErrorString(error));
}
#endif

}  // namespace

#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
struct Profiler::CudaState {
  struct Pending {
    StageId stage = kInvalidStage;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    std::uint64_t thread = 0;
    std::uint64_t frame = 0;
    std::uint64_t at_ns = 0;
  };

  static constexpr std::uint32_t kCapacity = kMaxPendingCuda;
  static_assert((1U << kCudaTokenIndexBits) == kCapacity,
                "CUDA token index bits must match the pending pool size");

  cudaEvent_t events[2U * kCapacity] = {};
  Pending pending[kCapacity] = {};
  std::uint32_t count = 0;
  std::uint32_t generation = 0;
  bool init_attempted = false;
  bool available = false;

  ~CudaState() noexcept {
    for (cudaEvent_t event : events) {
      if (event != nullptr) {
        (void)cudaEventDestroy(event);
      }
    }
  }
};
#endif

CpuScope::CpuScope(Profiler& profiler, StageId stage)
    : profiler_(&profiler),
      stage_(stage),
      active_(profiler.enabled() && stage != kInvalidStage),
      frame_(active_ ? profiler.current_frame() : 0U) {
  if (!active_) {
    return;
  }
  if (profiler.nvtx_enabled()) {
    nvtx_ = profiler.nvtx_push_cpu(stage_);
  }
  start_ns_ = Profiler::now_ns();
}

CpuScope::CpuScope(Profiler& profiler, std::string_view stage_name)
    : CpuScope(profiler, profiler.stage(stage_name)) {}

CpuScope::~CpuScope() {
  if (!active_) {
    return;
  }
  const std::uint64_t end_ns = Profiler::now_ns();
  if (nvtx_) {
    profiler_->nvtx_pop_cpu();
  }
  profiler_->record_cpu(stage_, start_ns_, end_ns, Profiler::thread_id(),
                        frame_);
}

CudaScope::CudaScope(Profiler& profiler, StageId stage, void* stream)
    : profiler_(&profiler), stage_(stage) {
  if (!profiler.enabled() || stage == kInvalidStage) {
    return;
  }
  if (profiler.nvtx_enabled()) {
    nvtx_ = profiler.nvtx_push_cuda(stage_);
  }
#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  token_ = profiler.cuda_begin(stage_, stream, Profiler::now_ns());
#else
  (void)stream;
#endif
}

CudaScope::CudaScope(Profiler& profiler, std::string_view stage_name,
                     void* stream)
    : CudaScope(profiler, profiler.stage(stage_name), stream) {}

CudaScope::~CudaScope() {
#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  if (token_ != kInvalidCudaToken) {
    profiler_->cuda_end(token_);
  }
#endif
  if (nvtx_) {
    profiler_->nvtx_pop_cuda();
  }
}

Profiler::Profiler() {
  set_precompute_profile(kDefaultPrecomputeProfile);
  origin_ns_ = now_ns();
  for (const std::string_view name : stages::kNames) {
    (void)stage(name);
  }
}

Profiler::~Profiler() { close(); }

bool Profiler::open(const std::string& jsonl_path, bool enable_nvtx,
                    std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  close();

  std::lock_guard<std::mutex> lock(mutex_);
  reset_stats_locked();
  jsonl_path_.clear();
  summary_path_.clear();
  vram_limit_mb_ = parse_vram_limit_mb();

  if (jsonl_path.empty() && !enable_nvtx) {
    enabled_.store(false, std::memory_order_relaxed);
    return false;
  }

  if (!jsonl_path.empty()) {
    std::FILE* file = std::fopen(jsonl_path.c_str(), "wb");
    if (file == nullptr) {
      if (error != nullptr) {
        *error = "cannot open profile log '" + jsonl_path + "': " +
                 std::strerror(errno);
      }
      enabled_.store(false, std::memory_order_relaxed);
      return false;
    }
    file_ = file;
    jsonl_path_ = jsonl_path;
    summary_path_ = jsonl_path + ".summary.json";
    write_error_ = false;
    warned_write_ = false;
  }

#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  // Allocated at setup time so the first CUDA scope does not allocate.  Event
  // objects themselves are created lazily by cuda_pool_init_locked().
  cuda_.reset(new (std::nothrow) CudaState());
  if (cuda_ == nullptr) {
    if (file_ != nullptr) {
      std::fclose(file_);
      file_ = nullptr;
    }
    if (error != nullptr) {
      *error = "out of memory allocating CUDA profiler state";
    }
    enabled_.store(false, std::memory_order_relaxed);
    return false;
  }
  vram_init_locked();
#endif

  bool nvtx_ok = false;
  if (enable_nvtx) {
    nvtx_ok = nvtx_init_locked();
  }

  // Start the wall clock after CUDA/NVTX setup so timestamps and wall_ms
  // describe the encode, not profiler initialization.
  origin_ns_ = now_ns();
  enabled_.store(file_ != nullptr || nvtx_ok, std::memory_order_relaxed);
  if (enable_nvtx && !nvtx_ok && error != nullptr && file_ == nullptr) {
    *error = "NVTX requested but not available in this build";
  }
  return enabled();
}

bool Profiler::open_from_env(std::string* error) {
  const char* path_env = std::getenv("AV2ENC_PROFILE");
  const char* nvtx_env = std::getenv("AV2ENC_PROFILE_NVTX");
  const bool want_nvtx = env_flag(nvtx_env);

  std::string path;
  if (path_env != nullptr && *path_env != '\0' && !env_flag(path_env) &&
      std::strcmp(path_env, "0") != 0 && std::strcmp(path_env, "off") != 0 &&
      std::strcmp(path_env, "false") != 0) {
    path = path_env;
  }

  if (path.empty() && !want_nvtx) {
    close();
    if (error != nullptr) {
      error->clear();
    }
    return false;
  }
  return open(path, want_nvtx, error);
}

void Profiler::close() {
  const bool was_open =
      file_ != nullptr || nvtx_cpu_domain_ != nullptr ||
      nvtx_cuda_domain_ != nullptr || enabled_.load(std::memory_order_relaxed);
  if (!was_open) {
    return;
  }

#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  cuda_flush();
#endif
  sample_vram();

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (origin_ns_ != 0) {
      wall_ns_ = now_ns() - origin_ns_;
    }
    if (file_ != nullptr) {
      write_summary_locked();
      (void)std::fflush(file_);
      (void)std::fclose(file_);
      file_ = nullptr;
    }
#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
    cuda_.reset();
#endif
    nvtx_shutdown_locked();
  }
  nvtx_enabled_.store(false, std::memory_order_relaxed);
  enabled_.store(false, std::memory_order_relaxed);
}

void Profiler::begin_frame(std::uint64_t frame_index) {
  if (!enabled()) {
    return;
  }
#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  // Defensive: if the caller forgot end_frame(), resolve the previous frame
  // before starting another one.
  cuda_flush();
#endif
  current_frame_.store(frame_index, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(mutex_);
  frame_start_ns_ = now_ns();
  frame_active_ = true;
}

void Profiler::end_frame() {
  if (!enabled()) {
    return;
  }
#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  cuda_flush();
#endif
  sample_vram();

  const std::uint64_t end_ns = now_ns();
  std::lock_guard<std::mutex> lock(mutex_);
  if (!frame_active_) {
    return;
  }
  const std::uint64_t frame = current_frame_.load(std::memory_order_relaxed);
  const std::uint64_t wall_ns =
      end_ns >= frame_start_ns_ ? end_ns - frame_start_ns_ : 0;
  ++frame_count_;
  frame_active_ = false;

  if (file_ != nullptr && !write_error_) {
    char line[160];
    LineBuilder builder(line, sizeof(line));
    builder.append("{\"type\":\"frame\",\"frame\":");
    builder.appendf("%llu", static_cast<unsigned long long>(frame));
    builder.append(",\"wall_ms\":");
    builder.append_ms(wall_ns);
    builder.append("}\n");
    if (builder.ok()) {
      const std::size_t written =
          std::fwrite(builder.data(), 1, builder.size(), file_);
      if (written != builder.size()) {
        write_error_ = true;
        if (!warned_write_) {
          warned_write_ = true;
          std::fprintf(stderr,
                       "av2enc profile: short write to profile JSONL; "
                       "events will be dropped\n");
        }
      }
    }
  }
  flush_file_locked();
}

std::uint64_t Profiler::now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

std::uint64_t Profiler::thread_id() noexcept {
  return static_cast<std::uint64_t>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

void Profiler::add_sample(TimingStats& stats, std::uint64_t ns) noexcept {
  if (stats.calls == 0 || ns < stats.min_ns) {
    stats.min_ns = ns;
  }
  if (ns > stats.max_ns) {
    stats.max_ns = ns;
  }
  stats.total_ns += ns;
  ++stats.calls;
}

StageId Profiler::stage(std::string_view name) noexcept {
  if (name.empty()) {
    return kInvalidStage;
  }
  std::uint32_t count = stage_count_.load(std::memory_order_acquire);
  for (std::uint32_t i = 0; i < count; ++i) {
    if (stages_[i].name_len == name.size() &&
        std::memcmp(stages_[i].name, name.data(), name.size()) == 0) {
      return static_cast<StageId>(i);
    }
  }

  std::lock_guard<std::mutex> lock(stage_mutex_);
  count = stage_count_.load(std::memory_order_relaxed);
  for (std::uint32_t i = 0; i < count; ++i) {
    if (stages_[i].name_len == name.size() &&
        std::memcmp(stages_[i].name, name.data(), name.size()) == 0) {
      return static_cast<StageId>(i);
    }
  }
  if (count >= kMaxStages) {
    if (!warned_stage_full_) {
      warned_stage_full_ = true;
      std::fprintf(stderr,
                   "av2enc profile: stage table full (%u); stage '%s' dropped\n",
                   kMaxStages, std::string(name).c_str());
    }
    return kInvalidStage;
  }

  const std::size_t length = std::min(name.size(), kMaxStageName - 1U);
  StageSlot& slot = stages_[count];
  std::memcpy(slot.name, name.data(), length);
  slot.name[length] = '\0';
  slot.name_len = static_cast<std::uint32_t>(length);
  stage_count_.store(count + 1U, std::memory_order_release);
  return static_cast<StageId>(count);
}

std::string_view Profiler::stage_name(StageId id) const noexcept {
  const std::uint32_t count = stage_count_.load(std::memory_order_acquire);
  if (id >= count) {
    return {};
  }
  return std::string_view(stages_[id].name, stages_[id].name_len);
}

void Profiler::set_precompute_profile(std::string_view name) noexcept {
  if (name.empty()) {
    name = kDefaultPrecomputeProfile;
  }
  const std::size_t length = std::min(name.size(), kMaxPrecomputeProfile - 1U);
  std::memcpy(precompute_profile_, name.data(), length);
  precompute_profile_[length] = '\0';
  precompute_profile_len_ = static_cast<std::uint32_t>(length);
}

std::string_view Profiler::precompute_profile() const noexcept {
  return std::string_view(precompute_profile_, precompute_profile_len_);
}

void Profiler::flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  flush_file_locked();
}

void Profiler::sample_vram() noexcept {
#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  if (!enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!vram_available_) {
    return;
  }
  vram_sample_locked();
#endif
}

void Profiler::record_thread_locked(std::uint64_t thread,
                                    bool cuda) noexcept {
  for (std::uint32_t i = 0; i < thread_count_; ++i) {
    if (threads_[i].id == thread) {
      if (cuda) {
        ++threads_[i].cuda_calls;
      } else {
        ++threads_[i].cpu_calls;
      }
      return;
    }
  }
  if (thread_count_ < kMaxThreads) {
    ThreadSlot& slot = threads_[thread_count_++];
    slot.id = thread;
    slot.cpu_calls = cuda ? 0U : 1U;
    slot.cuda_calls = cuda ? 1U : 0U;
  }
}

void Profiler::record_cpu(StageId stage, std::uint64_t start_ns,
                          std::uint64_t end_ns, std::uint64_t thread,
                          std::uint64_t frame) noexcept {
  if (!enabled()) {
    return;
  }
  const std::uint64_t duration_ns =
      end_ns >= start_ns ? end_ns - start_ns : 0U;
  std::lock_guard<std::mutex> lock(mutex_);
  const std::uint32_t count = stage_count_.load(std::memory_order_relaxed);
  if (stage >= count) {
    return;
  }
  add_sample(stages_[stage].cpu, duration_ns);
  record_thread_locked(thread, false);
  write_event_locked(stage, "cpu", thread, frame, start_ns, duration_ns);
}

void Profiler::record_cuda_locked(StageId stage, std::uint64_t duration_ns,
                                  std::uint64_t thread, std::uint64_t frame,
                                  std::uint64_t at_ns) noexcept {
  const std::uint32_t count = stage_count_.load(std::memory_order_relaxed);
  if (stage >= count) {
    return;
  }
  add_sample(stages_[stage].cuda, duration_ns);
  record_thread_locked(thread, true);
  write_event_locked(stage, "cuda", thread, frame, at_ns, duration_ns);
}

void Profiler::write_event_locked(StageId stage, const char* kind,
                                  std::uint64_t thread, std::uint64_t frame,
                                  std::uint64_t at_ns,
                                  std::uint64_t dur_ns) noexcept {
  if (file_ == nullptr || write_error_ || stage >= stage_count_.load()) {
    return;
  }
  const std::string_view name = stage_name(stage);
  const std::uint64_t at = at_ns >= origin_ns_ ? at_ns - origin_ns_ : 0U;

  char line[kEventLineCapacity];
  LineBuilder builder(line, sizeof(line));
  builder.append("{\"type\":\"event\",\"seq\":");
  builder.appendf("%llu", static_cast<unsigned long long>(sequence_));
  builder.append(",\"frame\":");
  builder.appendf("%llu", static_cast<unsigned long long>(frame));
  builder.append(",\"stage\":");
  builder.append_json(name);
  builder.appendf(",\"kind\":\"%s\",\"thread\":", kind);
  builder.appendf("%llu", static_cast<unsigned long long>(thread));
  builder.append(",\"at_ms\":");
  builder.append_ms(at);
  builder.append(",\"dur_ms\":");
  builder.append_ms(dur_ns);
  builder.append("}\n");

  if (!builder.ok()) {
    return;
  }
  const std::size_t written =
      std::fwrite(builder.data(), 1, builder.size(), file_);
  if (written != builder.size()) {
    write_error_ = true;
    if (!warned_write_) {
      warned_write_ = true;
      std::fprintf(stderr,
                   "av2enc profile: short write to profile JSONL; events "
                   "will be dropped\n");
    }
    return;
  }
  ++sequence_;
  ++event_count_;
}

void Profiler::flush_file_locked() noexcept {
  if (file_ == nullptr || write_error_) {
    return;
  }
  if (std::fflush(file_) != 0) {
    write_error_ = true;
    if (!warned_write_) {
      warned_write_ = true;
      std::fprintf(stderr, "av2enc profile: flush failed; events will be "
                           "dropped\n");
    }
  }
}

void Profiler::reset_stats_locked() noexcept {
  const std::uint32_t count = stage_count_.load(std::memory_order_relaxed);
  for (std::uint32_t i = 0; i < count; ++i) {
    stages_[i].cpu = TimingStats{};
    stages_[i].cuda = TimingStats{};
  }
  for (ThreadSlot& thread : threads_) {
    thread = ThreadSlot{};
  }
  thread_count_ = 0;
  origin_ns_ = now_ns();
  frame_start_ns_ = 0;
  wall_ns_ = 0;
  sequence_ = 0;
  event_count_ = 0;
  frame_count_ = 0;
  frame_active_ = false;
  write_error_ = false;
  warned_write_ = false;
  vram_limit_mb_ = kDefaultVramLimitMb;
  current_frame_.store(0, std::memory_order_relaxed);
#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  cuda_dropped_ = 0;
  cuda_errors_ = 0;
  warned_cuda_overflow_ = false;
  warned_cuda_error_ = false;
  vram_available_ = false;
  vram_capacity_bytes_ = 0;
  vram_free_bytes_ = 0;
  vram_peak_used_bytes_ = 0;
  warned_vram_limit_ = false;
  warned_vram_query_ = false;
#endif
}

std::uint64_t Profiler::dropped_cuda_events() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  return cuda_dropped_;
#else
  return 0;
#endif
}

#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA

bool Profiler::cuda_pool_init_locked() noexcept {
  if (cuda_ == nullptr) {
    return false;
  }
  if (cuda_->init_attempted) {
    return cuda_->available;
  }
  cuda_->init_attempted = true;

  for (std::uint32_t i = 0; i < 2U * CudaState::kCapacity; ++i) {
    const cudaError_t result =
        cudaEventCreateWithFlags(&cuda_->events[i], cudaEventDefault);
    if (result != cudaSuccess) {
      if (!warned_cuda_error_) {
        warned_cuda_error_ = true;
        warn_cuda_error("cannot create CUDA timing events", result);
      }
      return false;
    }
  }
  cuda_->available = true;
  return true;
}

std::uint32_t Profiler::cuda_begin(StageId stage, void* stream,
                                   std::uint64_t at_ns) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!enabled() || !cuda_pool_init_locked()) {
    return kInvalidCudaToken;
  }
  if (cuda_->count >= CudaState::kCapacity) {
    ++cuda_dropped_;
    if (!warned_cuda_overflow_) {
      warned_cuda_overflow_ = true;
      std::fprintf(stderr,
                   "av2enc profile: CUDA event pool full (%u pending); "
                   "call end_frame() every frame\n",
                   CudaState::kCapacity);
    }
    return kInvalidCudaToken;
  }

  const std::uint32_t index = cuda_->count;
  const std::uint32_t generation = cuda_->generation & kCudaGenerationMask;
  const std::uint32_t token =
      (generation << kCudaTokenIndexBits) | index;
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  const cudaError_t result =
      cudaEventRecord(cuda_->events[2U * index], cuda_stream);
  if (result != cudaSuccess) {
    ++cuda_errors_;
    if (!warned_cuda_error_) {
      warned_cuda_error_ = true;
      warn_cuda_error("cudaEventRecord(start) failed", result);
    }
    return kInvalidCudaToken;
  }

  CudaState::Pending& pending = cuda_->pending[index];
  pending.stage = stage;
  pending.stream = cuda_stream;
  pending.start = cuda_->events[2U * index];
  pending.end = cuda_->events[2U * index + 1U];
  pending.thread = thread_id();
  pending.frame = current_frame_.load(std::memory_order_relaxed);
  pending.at_ns = at_ns;
  ++cuda_->count;
  return token;
}

void Profiler::cuda_end(std::uint32_t token) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (cuda_ == nullptr) {
    return;
  }
  const std::uint32_t index = token & kCudaTokenIndexMask;
  const std::uint32_t generation = token >> kCudaTokenIndexBits;
  if (generation != (cuda_->generation & kCudaGenerationMask) ||
      index >= cuda_->count) {
    return;
  }
  CudaState::Pending& pending = cuda_->pending[index];
  if (pending.stage == kInvalidStage) {
    return;
  }
  const cudaError_t result = cudaEventRecord(pending.end, pending.stream);
  if (result != cudaSuccess) {
    pending.stage = kInvalidStage;
    ++cuda_errors_;
    if (!warned_cuda_error_) {
      warned_cuda_error_ = true;
      warn_cuda_error("cudaEventRecord(end) failed", result);
    }
  }
}

void Profiler::cuda_flush() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (cuda_ == nullptr || cuda_->count == 0) {
    return;
  }

  cudaStream_t streams[kMaxCudaStreams] = {};
  std::uint32_t stream_count = 0;
  bool stream_overflow = false;
  for (std::uint32_t i = 0; i < cuda_->count; ++i) {
    const cudaStream_t stream = cuda_->pending[i].stream;
    bool seen = false;
    for (std::uint32_t j = 0; j < stream_count; ++j) {
      if (streams[j] == stream) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      if (stream_count < kMaxCudaStreams) {
        streams[stream_count++] = stream;
      } else {
        stream_overflow = true;
      }
    }
  }

  // One synchronization per distinct stream per frame.  In the normal encoder
  // that is exactly one cudaStreamSynchronize() per frame.
  if (stream_overflow) {
    const cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      ++cuda_errors_;
      if (!warned_cuda_error_) {
        warned_cuda_error_ = true;
        warn_cuda_error("cudaDeviceSynchronize failed", result);
      }
    }
  } else {
    for (std::uint32_t i = 0; i < stream_count; ++i) {
      const cudaError_t result = cudaStreamSynchronize(streams[i]);
      if (result != cudaSuccess) {
        ++cuda_errors_;
        if (!warned_cuda_error_) {
          warned_cuda_error_ = true;
          warn_cuda_error("cudaStreamSynchronize failed", result);
        }
      }
    }
  }

  for (std::uint32_t i = 0; i < cuda_->count; ++i) {
    CudaState::Pending& pending = cuda_->pending[i];
    if (pending.stage == kInvalidStage) {
      continue;
    }
    float elapsed_ms = 0.0F;
    const cudaError_t result = cudaEventElapsedTime(
        &elapsed_ms, pending.start, pending.end);
    if (result != cudaSuccess) {
      ++cuda_errors_;
      if (!warned_cuda_error_) {
        warned_cuda_error_ = true;
        warn_cuda_error("cudaEventElapsedTime failed", result);
      }
      continue;
    }
    std::uint64_t duration_ns = 0;
    if (elapsed_ms > 0.0F) {
      duration_ns = static_cast<std::uint64_t>(
          static_cast<double>(elapsed_ms) * 1.0e6 + 0.5);
    }
    record_cuda_locked(pending.stage, duration_ns, pending.thread,
                       pending.frame, pending.at_ns);
  }

  cuda_->count = 0;
  cuda_->generation =
      (cuda_->generation + 1U) & kCudaGenerationMask;
}

void Profiler::vram_update_peak_locked(std::uint64_t free_bytes,
                                       std::uint64_t total_bytes) noexcept {
  const std::uint64_t used_bytes =
      total_bytes >= free_bytes ? total_bytes - free_bytes : 0U;
  if (used_bytes > vram_peak_used_bytes_) {
    vram_peak_used_bytes_ = used_bytes;
  }
  if (vram_limit_mb_ != 0) {
    const std::uint64_t limit_bytes =
        static_cast<std::uint64_t>(vram_limit_mb_) * 1024ULL * 1024ULL;
    if (used_bytes > limit_bytes && !warned_vram_limit_) {
      warned_vram_limit_ = true;
      std::fprintf(
          stderr,
          "av2enc profile: VRAM use %llu MB exceeds "
          "AV2ENC_VRAM_LIMIT_MB=%u (device capacity %llu MB)\n",
          static_cast<unsigned long long>(used_bytes / (1024ULL * 1024ULL)),
          static_cast<unsigned int>(vram_limit_mb_),
          static_cast<unsigned long long>(total_bytes / (1024ULL * 1024ULL)));
    }
  }
}

void Profiler::vram_init_locked() noexcept {
  vram_available_ = false;
  vram_capacity_bytes_ = 0;
  vram_free_bytes_ = 0;
  vram_peak_used_bytes_ = 0;

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t result = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (result != cudaSuccess) {
    // A CUDA-enabled build running on a host without a visible GPU is not an
    // error; omit VRAM fields in that case.
    if (result != cudaErrorNoDevice &&
        result != cudaErrorInsufficientDriver && !warned_vram_query_) {
      warned_vram_query_ = true;
      warn_cuda_error("cudaMemGetInfo(init) failed", result);
    }
    return;
  }
  vram_available_ = true;
  vram_capacity_bytes_ = static_cast<std::uint64_t>(total_bytes);
  vram_free_bytes_ = static_cast<std::uint64_t>(free_bytes);
  vram_update_peak_locked(vram_free_bytes_, vram_capacity_bytes_);
}

void Profiler::vram_sample_locked() noexcept {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t result = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (result != cudaSuccess) {
    if (!warned_vram_query_) {
      warned_vram_query_ = true;
      warn_cuda_error("cudaMemGetInfo failed", result);
    }
    return;
  }
  vram_capacity_bytes_ = static_cast<std::uint64_t>(total_bytes);
  vram_free_bytes_ = static_cast<std::uint64_t>(free_bytes);
  vram_update_peak_locked(vram_free_bytes_, vram_capacity_bytes_);
}

#endif  // AV2ENC_HAVE_CUDA

bool Profiler::nvtx_init_locked() noexcept {
#if defined(AV2ENC_HAVE_NVTX) && AV2ENC_HAVE_NVTX
  if (nvtx_cpu_domain_ != nullptr && nvtx_cuda_domain_ != nullptr) {
    return true;
  }
  nvtxDomainHandle_t cpu_domain = nvtxDomainCreateA("av2enc.cpu");
  nvtxDomainHandle_t cuda_domain = nvtxDomainCreateA("av2enc.cuda");
  if (cpu_domain == nullptr || cuda_domain == nullptr) {
    if (cpu_domain != nullptr) {
      nvtxDomainDestroy(cpu_domain);
    }
    if (cuda_domain != nullptr) {
      nvtxDomainDestroy(cuda_domain);
    }
    std::fprintf(stderr, "av2enc profile: failed to create NVTX domains\n");
    return false;
  }
  nvtx_cpu_domain_ = static_cast<void*>(cpu_domain);
  nvtx_cuda_domain_ = static_cast<void*>(cuda_domain);
  nvtx_enabled_.store(true, std::memory_order_relaxed);
  return true;
#else
  std::fprintf(stderr,
               "av2enc profile: AV2ENC_PROFILE_NVTX=1 but this build has no "
               "NVTX support (define AV2ENC_HAVE_NVTX=1 and link nvToolsExt)\n");
  return false;
#endif
}

void Profiler::nvtx_shutdown_locked() noexcept {
#if defined(AV2ENC_HAVE_NVTX) && AV2ENC_HAVE_NVTX
  if (nvtx_cpu_domain_ != nullptr) {
    nvtxDomainDestroy(static_cast<nvtxDomainHandle_t>(nvtx_cpu_domain_));
    nvtx_cpu_domain_ = nullptr;
  }
  if (nvtx_cuda_domain_ != nullptr) {
    nvtxDomainDestroy(static_cast<nvtxDomainHandle_t>(nvtx_cuda_domain_));
    nvtx_cuda_domain_ = nullptr;
  }
#endif
  nvtx_enabled_.store(false, std::memory_order_relaxed);
}

bool Profiler::nvtx_push_cpu(StageId stage) noexcept {
#if defined(AV2ENC_HAVE_NVTX) && AV2ENC_HAVE_NVTX
  nvtxDomainHandle_t domain = static_cast<nvtxDomainHandle_t>(nvtx_cpu_domain_);
  if (domain == nullptr) {
    return false;
  }
  const std::string_view name = stage_name(stage);
  if (name.empty()) {
    return false;
  }
  nvtxEventAttributes_t attributes = {};
  attributes.version = NVTX_VERSION;
  attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attributes.colorType = NVTX_COLOR_ARGB;
  attributes.color = 0xFF3B82F6U;  // blue
  attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attributes.message.ascii = name.data();
  return nvtxDomainRangePushEx(domain, &attributes) >= 0;
#else
  (void)stage;
  return false;
#endif
}

void Profiler::nvtx_pop_cpu() noexcept {
#if defined(AV2ENC_HAVE_NVTX) && AV2ENC_HAVE_NVTX
  nvtxDomainHandle_t domain = static_cast<nvtxDomainHandle_t>(nvtx_cpu_domain_);
  if (domain != nullptr) {
    (void)nvtxDomainRangePop(domain);
  }
#endif
}

bool Profiler::nvtx_push_cuda(StageId stage) noexcept {
#if defined(AV2ENC_HAVE_NVTX) && AV2ENC_HAVE_NVTX
  nvtxDomainHandle_t domain =
      static_cast<nvtxDomainHandle_t>(nvtx_cuda_domain_);
  if (domain == nullptr) {
    return false;
  }
  const std::string_view name = stage_name(stage);
  if (name.empty()) {
    return false;
  }
  nvtxEventAttributes_t attributes = {};
  attributes.version = NVTX_VERSION;
  attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attributes.colorType = NVTX_COLOR_ARGB;
  attributes.color = 0xFF10B981U;  // green
  attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attributes.message.ascii = name.data();
  return nvtxDomainRangePushEx(domain, &attributes) >= 0;
#else
  (void)stage;
  return false;
#endif
}

void Profiler::nvtx_pop_cuda() noexcept {
#if defined(AV2ENC_HAVE_NVTX) && AV2ENC_HAVE_NVTX
  nvtxDomainHandle_t domain =
      static_cast<nvtxDomainHandle_t>(nvtx_cuda_domain_);
  if (domain != nullptr) {
    (void)nvtxDomainRangePop(domain);
  }
#endif
}

void Profiler::write_summary_locked() noexcept {
  if (summary_path_.empty()) {
    return;
  }
  std::FILE* file = std::fopen(summary_path_.c_str(), "wb");
  if (file == nullptr) {
    std::fprintf(stderr, "av2enc profile: cannot write summary '%s': %s\n",
                 summary_path_.c_str(), std::strerror(errno));
    return;
  }

  const std::uint64_t wall_ns =
      wall_ns_ != 0 ? wall_ns_
                    : (origin_ns_ != 0 ? now_ns() - origin_ns_ : 0U);
  const std::uint32_t count = stage_count_.load(std::memory_order_relaxed);
  std::uint64_t cuda_dropped = 0;
  std::uint64_t cuda_errors = 0;
#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  cuda_dropped = cuda_dropped_;
  cuda_errors = cuda_errors_;
#endif

  char wall_ms[40];
  (void)format_ms(wall_ms, sizeof(wall_ms), wall_ns);

  std::fputs("{\n  \"profiler\": \"av2enc\",\n  \"version\": 1,\n", file);
  std::fputs("  \"jsonl\": ", file);
  write_json_string(file, std::string_view(jsonl_path_));
  std::fputs(",\n  \"precompute_profile\": ", file);
  write_json_string(file, precompute_profile());
  std::fprintf(file,
               ",\n  \"vram_limit_mb\": %u,\n  \"wall_ms\": %s,\n  \"frames\": "
               "%llu,\n  \"events\": %llu,\n  \"cuda_events_dropped\": %llu,\n  "
               "\"cuda_errors\": %llu,\n  \"write_error\": %s,\n",
               static_cast<unsigned int>(vram_limit_mb_), wall_ms,
               static_cast<unsigned long long>(frame_count_),
               static_cast<unsigned long long>(event_count_),
               static_cast<unsigned long long>(cuda_dropped),
               static_cast<unsigned long long>(cuda_errors),
               write_error_ ? "true" : "false");

#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  if (vram_available_) {
    char peak_mb[40];
    char capacity_mb[40];
    char free_mb[40];
    (void)format_mb(peak_mb, sizeof(peak_mb), vram_peak_used_bytes_);
    (void)format_mb(capacity_mb, sizeof(capacity_mb), vram_capacity_bytes_);
    (void)format_mb(free_mb, sizeof(free_mb), vram_free_bytes_);
    std::fprintf(file,
                 "  \"peak_vram_mb\": %s,\n  \"vram_capacity_mb\": %s,\n  "
                 "\"vram_free_mb\": %s,\n",
                 peak_mb, capacity_mb, free_mb);
  }
#endif

  std::fputs("  \"threads\": [\n", file);
  for (std::uint32_t i = 0; i < thread_count_; ++i) {
    if (i != 0) {
      std::fputs(",\n", file);
    }
    std::fprintf(file,
                 "    {\"id\": %llu, \"cpu_calls\": %llu, \"cuda_calls\": "
                 "%llu}",
                 static_cast<unsigned long long>(threads_[i].id),
                 static_cast<unsigned long long>(threads_[i].cpu_calls),
                 static_cast<unsigned long long>(threads_[i].cuda_calls));
  }
  std::fputs("\n  ],\n  \"stages\": [\n", file);

  const auto write_stats = [file](const TimingStats& stats) {
    char total_ms[40];
    char min_ms[40];
    char max_ms[40];
    char mean_ms[40];
    const std::uint64_t mean_ns =
        stats.calls != 0 ? stats.total_ns / stats.calls : 0U;
    (void)format_ms(total_ms, sizeof(total_ms), stats.total_ns);
    (void)format_ms(min_ms, sizeof(min_ms), stats.calls != 0 ? stats.min_ns : 0U);
    (void)format_ms(max_ms, sizeof(max_ms), stats.max_ns);
    (void)format_ms(mean_ms, sizeof(mean_ms), mean_ns);
    std::fprintf(file,
                 "{\"calls\": %llu, \"total_ms\": %s, \"min_ms\": %s, "
                 "\"max_ms\": %s, \"mean_ms\": %s}",
                 static_cast<unsigned long long>(stats.calls), total_ms,
                 min_ms, max_ms, mean_ms);
  };

  for (std::uint32_t i = 0; i < count; ++i) {
    if (i != 0) {
      std::fputs(",\n", file);
    }
    const StageSlot& slot = stages_[i];
    std::fputs("    {\"stage\": ", file);
    write_json_string(file, std::string_view(slot.name, slot.name_len));
    std::fputs(", \"cpu\": ", file);
    write_stats(slot.cpu);
    std::fputs(", \"cuda\": ", file);
    write_stats(slot.cuda);
    std::fputc('}', file);
  }
  std::fputs("\n  ]\n}\n", file);

  if (std::fclose(file) != 0) {
    std::fprintf(stderr, "av2enc profile: failed to close summary '%s': %s\n",
                 summary_path_.c_str(), std::strerror(errno));
  }
}

void Profiler::print_summary(std::ostream& out) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::uint64_t wall_ns =
      wall_ns_ != 0 ? wall_ns_
                    : (origin_ns_ != 0 ? now_ns() - origin_ns_ : 0U);
  char wall_ms[40];
  (void)format_ms(wall_ms, sizeof(wall_ms), wall_ns);

  std::uint64_t cuda_dropped = 0;
  std::uint64_t cuda_errors = 0;
#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  cuda_dropped = cuda_dropped_;
  cuda_errors = cuda_errors_;
#endif

  out << "av2enc profiler summary\n";
  out << "  jsonl:  " << (jsonl_path_.empty() ? "(none)" : jsonl_path_)
      << '\n';
  out << "  profile: " << precompute_profile()
      << ", vram limit: " << vram_limit_mb_ << " MB\n";
  out << "  frames: " << frame_count_ << ", events: " << event_count_
      << ", wall: " << wall_ms << " ms, cuda dropped: " << cuda_dropped
      << ", cuda errors: " << cuda_errors << '\n';

#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
  if (vram_available_) {
    char peak_mb[40];
    char capacity_mb[40];
    char free_mb[40];
    (void)format_mb(peak_mb, sizeof(peak_mb), vram_peak_used_bytes_);
    (void)format_mb(capacity_mb, sizeof(capacity_mb), vram_capacity_bytes_);
    (void)format_mb(free_mb, sizeof(free_mb), vram_free_bytes_);
    out << "  vram:   " << peak_mb << " MB peak used / " << capacity_mb
        << " MB capacity, " << free_mb << " MB free\n";
  }
#endif

  char line[256];
  std::snprintf(line, sizeof(line),
                "  %-28s %-4s %10s %14s %14s %14s %14s\n", "stage", "kind",
                "calls", "total_ms", "min_ms", "max_ms", "mean_ms");
  out << line;

  const std::uint32_t count = stage_count_.load(std::memory_order_relaxed);
  bool printed = false;
  for (std::uint32_t i = 0; i < count; ++i) {
    const StageSlot& slot = stages_[i];
    const TimingStats* kinds[2] = {&slot.cpu, &slot.cuda};
    const char* kind_names[2] = {"cpu", "cuda"};
    for (int kind = 0; kind < 2; ++kind) {
      const TimingStats& stats = *kinds[kind];
      if (stats.calls == 0) {
        continue;
      }
      char total_ms[40];
      char min_ms[40];
      char max_ms[40];
      char mean_ms[40];
      const std::uint64_t mean_ns = stats.total_ns / stats.calls;
      (void)format_ms(total_ms, sizeof(total_ms), stats.total_ns);
      (void)format_ms(min_ms, sizeof(min_ms), stats.min_ns);
      (void)format_ms(max_ms, sizeof(max_ms), stats.max_ns);
      (void)format_ms(mean_ms, sizeof(mean_ms), mean_ns);
      std::snprintf(
          line, sizeof(line), "  %-28.*s %-4s %10llu %14s %14s %14s %14s\n",
          static_cast<int>(slot.name_len), slot.name, kind_names[kind],
          static_cast<unsigned long long>(stats.calls), total_ms, min_ms,
          max_ms, mean_ms);
      out << line;
      printed = true;
    }
  }
  if (!printed) {
    out << "  (no samples)\n";
  }
}

}  // namespace av2enc::profile
