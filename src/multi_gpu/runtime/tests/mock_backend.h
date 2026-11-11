// CPU-only mock device backend for the multi-GPU runtime tests.
//
// It models devices, contexts, streams, events, arenas, pinned host memory, and
// host-mediated copies. "Device" memory is ordinary host memory, so a test
// FrameWorker can fill device payload/recon buffers directly or through the
// backend. Failure injection covers the three startup cases the runtime must
// handle deterministically: a missing ordinal, an arena allocation failure
// (OOM), and a copy failure mid-frame.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../device_backend.h"

namespace av2enc::multigpu::runtime::testing {

enum class MockCopyKind : std::uint8_t {
  kHostToDevice = 0,
  kDeviceToHost = 1,
  kDeviceToDevice = 2,
};

struct MockCopy {
  MockCopyKind kind = MockCopyKind::kHostToDevice;
  int ordinal = -1;
  StreamHandle stream = kInvalidStream;
  std::size_t bytes = 0;
};

enum class MockEventOp : std::uint8_t {
  kCreate = 0,
  kRecord = 1,
  kWait = 2,
  kSynchronize = 3,
  kStreamSynchronize = 4,
  kDeviceSynchronize = 5,
  kDestroy = 6,
};

struct MockEventRecord {
  MockEventOp op = MockEventOp::kCreate;
  int ordinal = -1;
  EventHandle event = kInvalidEvent;
  StreamHandle stream = kInvalidStream;
};

struct MockOptions {
  bool peer_access = false;
  // If the requested arena capacity exceeds this many bytes, arena_init()
  // returns kOutOfMemory. This is how a cudaMalloc failure is simulated.
  std::map<int, std::uint64_t> arena_fail_above;
  // Fail arena_allocate() for this device when the request exceeds the value.
  std::map<int, std::uint64_t> arena_allocate_fail_above;
  // Fail the (0-based) copy after this many successful copies; -1 = never.
  long fail_copy_after = -1;
  bool fail_pinned_alloc = false;
  double fake_ms_per_event_id = 0.25;
};

inline DeviceProperties mock_device(int ordinal, std::uint64_t total_bytes,
                                    int cc_major = 8, int cc_minor = 6,
                                    int sm_count = 28,
                                    std::string name = "Mock RTX 3060") {
  DeviceProperties props;
  props.ordinal = ordinal;
  props.name = std::move(name);
  props.cc_major = cc_major;
  props.cc_minor = cc_minor;
  props.sm_count = sm_count;
  props.total_vram_bytes = total_bytes;
  props.free_vram_bytes = total_bytes - (128ull * 1024ull * 1024ull);
  return props;
}

class MockBackend final : public DeviceBackend {
 public:
  explicit MockBackend(std::vector<DeviceProperties> devices,
                       MockOptions options = {})
      : devices_(std::move(devices)), options_(options) {}

  const char* backend_name() const noexcept override { return "mock"; }

  Status device_count(int* out) override {
    if (out == nullptr) {
      return Status::kInvalidArgument;
    }
    *out = static_cast<int>(devices_.size());
    return Status::kOk;
  }

  Status query_device(int ordinal, DeviceProperties* out) override {
    if (out == nullptr || ordinal < 0 ||
        ordinal >= static_cast<int>(devices_.size())) {
      return Status::kDeviceUnavailable;
    }
    *out = devices_[static_cast<std::size_t>(ordinal)];
    return Status::kOk;
  }

  Status context_init(int ordinal, std::string* error) override {
    if (ordinal < 0 || ordinal >= static_cast<int>(devices_.size())) {
      if (error != nullptr) {
        *error = "mock: no such device";
      }
      return Status::kDeviceUnavailable;
    }
    contexts_.insert(ordinal);
    return Status::kOk;
  }

  void context_shutdown(int ordinal) override { contexts_.erase(ordinal); }

  bool context_open(int ordinal) const {
    return contexts_.find(ordinal) != contexts_.end();
  }

  Status stream_create(int ordinal, StreamHandle* out,
                       std::string* error) override {
    (void)error;
    if (out == nullptr) {
      return Status::kInvalidArgument;
    }
    if (ordinal < 0 || ordinal >= static_cast<int>(devices_.size())) {
      return Status::kDeviceUnavailable;
    }
    const StreamHandle handle = next_handle_++;
    streams_[handle] = ordinal;
    *out = handle;
    return Status::kOk;
  }

  void stream_destroy(StreamHandle stream) override { streams_.erase(stream); }

  Status stream_synchronize(StreamHandle stream) override {
    if (streams_.find(stream) == streams_.end()) {
      return Status::kInvalidArgument;
    }
    events_.push_back(MockEventRecord{MockEventOp::kStreamSynchronize,
                                      streams_[stream], kInvalidEvent, stream});
    return Status::kOk;
  }

  Status event_create(int ordinal, bool timing, EventHandle* out,
                      std::string* error) override {
    (void)timing;
    (void)error;
    if (out == nullptr) {
      return Status::kInvalidArgument;
    }
    if (ordinal < 0 || ordinal >= static_cast<int>(devices_.size())) {
      return Status::kDeviceUnavailable;
    }
    const EventHandle handle = next_handle_++;
    events_map_[handle] = EventState{ordinal, kInvalidStream, false};
    events_.push_back(
        MockEventRecord{MockEventOp::kCreate, ordinal, handle, kInvalidStream});
    *out = handle;
    return Status::kOk;
  }

  void event_destroy(EventHandle event) override {
    const auto it = events_map_.find(event);
    if (it == events_map_.end()) {
      return;
    }
    events_.push_back(MockEventRecord{MockEventOp::kDestroy, it->second.ordinal,
                                      event, kInvalidStream});
    events_map_.erase(it);
  }

  Status event_record(EventHandle event, StreamHandle stream) override {
    const auto it = events_map_.find(event);
    const auto stream_it = streams_.find(stream);
    if (it == events_map_.end() || stream_it == streams_.end()) {
      return Status::kInvalidArgument;
    }
    it->second.recorded = true;
    it->second.stream = stream;
    events_.push_back(MockEventRecord{MockEventOp::kRecord,
                                      it->second.ordinal, event, stream});
    return Status::kOk;
  }

  Status event_wait(EventHandle event, StreamHandle stream) override {
    const auto it = events_map_.find(event);
    const auto stream_it = streams_.find(stream);
    if (it == events_map_.end() || stream_it == streams_.end()) {
      return Status::kInvalidArgument;
    }
    events_.push_back(MockEventRecord{MockEventOp::kWait, it->second.ordinal,
                                      event, stream});
    return Status::kOk;
  }

  Status event_synchronize(EventHandle event) override {
    const auto it = events_map_.find(event);
    if (it == events_map_.end()) {
      return Status::kInvalidArgument;
    }
    events_.push_back(MockEventRecord{MockEventOp::kSynchronize,
                                      it->second.ordinal, event,
                                      kInvalidStream});
    return Status::kOk;
  }

  Status event_elapsed_ms(EventHandle start, EventHandle stop,
                          double* out) override {
    if (out == nullptr || events_map_.find(start) == events_map_.end() ||
        events_map_.find(stop) == events_map_.end()) {
      return Status::kInvalidArgument;
    }
    const double delta = static_cast<double>(
        static_cast<long>(stop) - static_cast<long>(start));
    *out = options_.fake_ms_per_event_id * (delta >= 0.0 ? delta : -delta);
    return Status::kOk;
  }

  Status arena_init(int ordinal, std::uint64_t capacity_bytes,
                    ArenaHandle* out, std::string* error) override {
    if (out == nullptr || ordinal < 0 ||
        ordinal >= static_cast<int>(devices_.size())) {
      return Status::kInvalidArgument;
    }
    const auto fail = options_.arena_fail_above.find(ordinal);
    if (fail != options_.arena_fail_above.end() &&
        capacity_bytes > fail->second) {
      if (error != nullptr) {
        *error = "mock: cudaMalloc refused (OOM injection)";
      }
      return Status::kOutOfMemory;
    }
    if (capacity_bytes == 0) {
      return Status::kInvalidArgument;
    }
    const ArenaHandle handle = next_handle_++;
    ArenaState state;
    state.ordinal = ordinal;
    state.capacity = capacity_bytes;
    arenas_[handle] = std::move(state);
    *out = handle;
    return Status::kOk;
  }

  void arena_release(ArenaHandle arena) override {
    const auto it = arenas_.find(arena);
    if (it == arenas_.end()) {
      return;
    }
    for (void* block : it->second.blocks) {
      std::free(block);
    }
    arenas_.erase(it);
  }

  Status arena_allocate(ArenaHandle arena, std::size_t bytes,
                        std::size_t alignment, void** out,
                        std::string* error) override {
    if (out == nullptr) {
      return Status::kInvalidArgument;
    }
    *out = nullptr;
    const auto it = arenas_.find(arena);
    if (it == arenas_.end()) {
      return Status::kNotInitialized;
    }
    if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
      return Status::kInvalidArgument;
    }
    const auto fail =
        options_.arena_allocate_fail_above.find(it->second.ordinal);
    if (fail != options_.arena_allocate_fail_above.end() &&
        bytes > fail->second) {
      if (error != nullptr) {
        *error = "mock: arena allocation refused (OOM injection)";
      }
      return Status::kArenaExhausted;
    }
    const std::uint64_t aligned =
        (it->second.used + static_cast<std::uint64_t>(alignment) - 1) &
        ~(static_cast<std::uint64_t>(alignment) - 1);
    if (aligned > it->second.capacity ||
        bytes > it->second.capacity - aligned) {
      if (error != nullptr) {
        *error = "mock: arena exhausted";
      }
      return Status::kArenaExhausted;
    }
    void* block = std::malloc(bytes);
    if (block == nullptr) {
      if (error != nullptr) {
        *error = "mock: host malloc failed";
      }
      return Status::kOutOfMemory;
    }
    it->second.blocks.push_back(block);
    it->second.used = aligned + static_cast<std::uint64_t>(bytes);
    *out = block;
    return Status::kOk;
  }

  std::uint64_t arena_capacity(ArenaHandle arena) const override {
    const auto it = arenas_.find(arena);
    return it == arenas_.end() ? 0 : it->second.capacity;
  }

  std::uint64_t arena_used(ArenaHandle arena) const override {
    const auto it = arenas_.find(arena);
    return it == arenas_.end() ? 0 : it->second.used;
  }

  Status pinned_alloc(std::size_t bytes, void** out,
                      std::string* error) override {
    if (out == nullptr || bytes == 0) {
      return Status::kInvalidArgument;
    }
    if (options_.fail_pinned_alloc) {
      if (error != nullptr) {
        *error = "mock: pinned allocation refused";
      }
      return Status::kOutOfMemory;
    }
    void* block = std::malloc(bytes);
    if (block == nullptr) {
      return Status::kOutOfMemory;
    }
    *out = block;
    pinned_[block] = bytes;
    return Status::kOk;
  }

  void pinned_free(void* ptr) override {
    if (ptr == nullptr) {
      return;
    }
    pinned_.erase(ptr);
    std::free(ptr);
  }

  Status copy_h2d_async(StreamHandle stream, void* device_dst,
                        const void* host_src, std::size_t bytes) override {
    return do_copy(MockCopyKind::kHostToDevice, stream, device_dst, host_src,
                   bytes);
  }

  Status copy_d2h_async(StreamHandle stream, void* host_dst,
                        const void* device_src, std::size_t bytes) override {
    return do_copy(MockCopyKind::kDeviceToHost, stream, host_dst, device_src,
                   bytes);
  }

  Status copy_d2d_async(StreamHandle stream, void* device_dst,
                        const void* device_src, std::size_t bytes) override {
    return do_copy(MockCopyKind::kDeviceToDevice, stream, device_dst,
                   device_src, bytes);
  }

  bool peer_access(int from_ordinal, int to_ordinal) override {
    (void)from_ordinal;
    (void)to_ordinal;
    return options_.peer_access;
  }

  Status device_synchronize(int ordinal) override {
    events_.push_back(MockEventRecord{MockEventOp::kDeviceSynchronize, ordinal,
                                      kInvalidEvent, kInvalidStream});
    return Status::kOk;
  }

  // Test inspection helpers.
  const std::vector<MockCopy>& copies() const { return copies_; }
  const std::vector<MockEventRecord>& event_log() const { return events_; }
  std::size_t copy_count(MockCopyKind kind) const {
    std::size_t count = 0;
    for (const MockCopy& copy : copies_) {
      if (copy.kind == kind) {
        ++count;
      }
    }
    return count;
  }
  std::uint64_t copied_bytes(MockCopyKind kind) const {
    std::uint64_t total = 0;
    for (const MockCopy& copy : copies_) {
      if (copy.kind == kind) {
        total += copy.bytes;
      }
    }
    return total;
  }
  std::size_t event_count(MockEventOp op) const {
    std::size_t count = 0;
    for (const MockEventRecord& record : events_) {
      if (record.op == op) {
        ++count;
      }
    }
    return count;
  }
  bool has_wait(EventHandle event, StreamHandle stream) const {
    for (const MockEventRecord& record : events_) {
      if (record.op == MockEventOp::kWait && record.event == event &&
          record.stream == stream) {
        return true;
      }
    }
    return false;
  }
  bool has_record(EventHandle event, StreamHandle stream) const {
    for (const MockEventRecord& record : events_) {
      if (record.op == MockEventOp::kRecord && record.event == event &&
          record.stream == stream) {
        return true;
      }
    }
    return false;
  }
  std::size_t device_sync_count() const {
    return event_count(MockEventOp::kDeviceSynchronize);
  }
  void clear_logs() {
    copies_.clear();
    events_.clear();
  }
  std::size_t live_arena_blocks() const {
    std::size_t count = 0;
    for (const auto& entry : arenas_) {
      count += entry.second.blocks.size();
    }
    return count;
  }
  std::size_t live_pinned_allocations() const { return pinned_.size(); }
  std::size_t live_arenas() const { return arenas_.size(); }
  std::size_t live_streams() const { return streams_.size(); }
  std::size_t live_contexts() const { return contexts_.size(); }

 private:
  struct EventState {
    int ordinal = -1;
    StreamHandle stream = kInvalidStream;
    bool recorded = false;
  };
  struct ArenaState {
    int ordinal = -1;
    std::uint64_t capacity = 0;
    std::uint64_t used = 0;
    std::vector<void*> blocks;
  };

  Status do_copy(MockCopyKind kind, StreamHandle stream, void* dst,
                 const void* src, std::size_t bytes) {
    const auto stream_it = streams_.find(stream);
    if (stream_it == streams_.end() || dst == nullptr || src == nullptr ||
        bytes == 0) {
      return Status::kInvalidArgument;
    }
    if (options_.fail_copy_after >= 0 &&
        static_cast<long>(copies_.size()) >= options_.fail_copy_after) {
      return Status::kBackendError;
    }
    std::memcpy(dst, src, bytes);
    copies_.push_back(
        MockCopy{kind, stream_it->second, stream, bytes});
    return Status::kOk;
  }

  std::vector<DeviceProperties> devices_;
  MockOptions options_;
  std::set<int> contexts_;
  std::unordered_map<StreamHandle, int> streams_;
  std::unordered_map<EventHandle, EventState> events_map_;
  std::unordered_map<ArenaHandle, ArenaState> arenas_;
  std::unordered_map<void*, std::size_t> pinned_;
  std::vector<MockCopy> copies_;
  std::vector<MockEventRecord> events_;
  StreamHandle next_handle_ = 1;
};

}  // namespace av2enc::multigpu::runtime::testing
