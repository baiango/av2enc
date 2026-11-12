// Deterministic pattern worker for the runtime tests and the CUDA gate.
//
// It does not encode video. It writes one variable-length payload per tile and
// a known reconstruction pattern, using only DeviceBackend copies, so the same
// worker exercises the mocked host build and a real CUDA backend. Direct device
// reads are enabled only for the mock backend, where arena memory is ordinary
// host memory.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../runtime.h"

namespace av2enc::multigpu::runtime::testing {

inline std::vector<std::uint8_t> expected_source_frame(int width, int height,
                                                       int frame) {
  std::vector<std::uint8_t> frame_bytes(
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u /
      2u);
  const int chroma_width = width / 2;
  const int chroma_height = height / 2;
  const std::size_t luma_size =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const std::size_t chroma_size =
      static_cast<std::size_t>(chroma_width) *
      static_cast<std::size_t>(chroma_height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      frame_bytes[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                  static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>((x * 7 + y * 11 + frame * 5) & 0xff);
    }
  }
  for (int plane = 1; plane < 3; ++plane) {
    const std::size_t plane_offset =
        luma_size + static_cast<std::size_t>(plane - 1) * chroma_size;
    for (int y = 0; y < chroma_height; ++y) {
      for (int x = 0; x < chroma_width; ++x) {
        frame_bytes[plane_offset +
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(chroma_width) +
                    static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>(
                (x * 13 + y * 3 + plane * 23 + frame * 5) & 0xff);
      }
    }
  }
  return frame_bytes;
}

inline std::vector<std::uint8_t> expected_recon_frame(int width, int height,
                                                      int frame) {
  std::vector<std::uint8_t> frame_bytes(
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u /
      2u);
  const int chroma_width = width / 2;
  const int chroma_height = height / 2;
  const std::size_t luma_size =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const std::size_t chroma_size =
      static_cast<std::size_t>(chroma_width) *
      static_cast<std::size_t>(chroma_height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      frame_bytes[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                  static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>((x * 3 + y * 5 + frame * 11) & 0xff);
    }
  }
  for (int plane = 1; plane < 3; ++plane) {
    const std::size_t plane_offset =
        luma_size + static_cast<std::size_t>(plane - 1) * chroma_size;
    for (int y = 0; y < chroma_height; ++y) {
      for (int x = 0; x < chroma_width; ++x) {
        frame_bytes[plane_offset +
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(chroma_width) +
                    static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>(
                (x * 9 + y * 19 + plane * 17 + frame * 11) & 0xff);
      }
    }
  }
  return frame_bytes;
}

inline std::uint32_t pattern_payload_size(int tile, std::uint32_t stride) {
  if (stride <= 2) {
    return 2;
  }
  return 2u + (static_cast<std::uint32_t>(tile) * 37u +
               static_cast<std::uint32_t>(tile + 1) * 11u) %
                  (stride - 1);
}

inline std::vector<std::uint8_t> expected_payload(int tile, int frame,
                                                  std::uint32_t stride) {
  const std::uint32_t size = pattern_payload_size(tile, stride);
  std::vector<std::uint8_t> bytes(size);
  for (std::uint32_t j = 0; j < size; ++j) {
    bytes[j] = static_cast<std::uint8_t>(
        (static_cast<std::uint32_t>(tile) * 31u +
         static_cast<std::uint32_t>(frame) * 7u + j * 13u) &
        0xffu);
  }
  return bytes;
}

class PatternWorker final : public FrameWorker {
 public:
  struct Visit {
    int frame = -1;
    int slot = -1;
    int ordinal = -1;
    int reference_index = -1;
    std::vector<int> tiles;
    bool source_ok = true;
    bool reference_ok = true;
  };

  explicit PatternWorker(std::uint32_t stride, bool direct_device_access)
      : stride_(stride), direct_device_access_(direct_device_access) {}

  ~PatternWorker() override {
    if (backend_ != nullptr) {
      if (payload_buffer_ != nullptr) {
        backend_->pinned_free(payload_buffer_);
      }
      if (recon_buffer_ != nullptr) {
        backend_->pinned_free(recon_buffer_);
      }
    }
  }

  Status encode(const GpuFrameContext& context, std::string* error) override {
    if (context.backend == nullptr || context.layout == nullptr ||
        context.tile_sizes_out == nullptr || context.device_payload == nullptr ||
        context.device_recon == nullptr || context.device_source == nullptr) {
      if (error != nullptr) {
        *error = "pattern worker got an incomplete frame context";
      }
      return Status::kInvalidArgument;
    }
    if (context.bit_depth != 8) {
      if (error != nullptr) {
        *error = "pattern worker models 8-bit 4:2:0 only";
      }
      return Status::kInvalidArgument;
    }
    if (context.payload_stride_bytes != stride_) {
      if (error != nullptr) {
        *error = "pattern worker payload stride does not match the runtime";
      }
      return Status::kInvalidArgument;
    }
    if (fail_frame_ >= 0 &&
        (context.frame_index == fail_frame_ || fail_frame_ == -2) &&
        (fail_slot_ < 0 || context.slot == fail_slot_)) {
      if (error != nullptr) {
        *error = "injected worker failure";
      }
      return Status::kDispatchFailed;
    }

    Visit visit;
    visit.frame = context.frame_index;
    visit.slot = context.slot;
    visit.ordinal = context.ordinal;
    visit.reference_index = context.reference_index;
    if (context.tile_count > 0 && context.tile_indices != nullptr) {
      visit.tiles.assign(context.tile_indices,
                         context.tile_indices + context.tile_count);
    }

    if (direct_device_access_) {
      const std::vector<std::uint8_t> source =
          expected_source_frame(context.frame_width, context.frame_height,
                                context.frame_index);
      visit.source_ok =
          source.size() == context.source_bytes &&
          std::equal(source.begin(), source.end(), context.device_source);
      if (context.frame_index > 0) {
        const std::vector<std::uint8_t> reference =
            expected_recon_frame(context.frame_width, context.frame_height,
                                 context.frame_index - 1);
        visit.reference_ok =
            reference.size() == context.source_bytes &&
            std::equal(reference.begin(), reference.end(),
                       context.device_reference);
      }
    }

    if (backend_ == nullptr) {
      backend_ = context.backend;
    }
    if (backend_ != context.backend) {
      if (error != nullptr) {
        *error = "pattern worker was used with two different backends";
      }
      return Status::kInvalidArgument;
    }
    if (!ensure_staging(context.source_bytes, error)) {
      return Status::kOutOfMemory;
    }
    const std::vector<std::uint8_t> recon =
        expected_recon_frame(context.frame_width, context.frame_height,
                             context.frame_index);
    std::copy(recon.begin(), recon.end(), recon_buffer_);
    for (int index = 0; index < context.tile_count; ++index) {
      const int tile = context.tile_indices[index];
      const std::vector<std::uint8_t> payload =
          expected_payload(tile, context.frame_index, context.payload_stride_bytes);
      std::fill(payload_buffer_, payload_buffer_ + stride_, 0u);
      std::copy(payload.begin(), payload.end(), payload_buffer_);
      std::uint32_t size = static_cast<std::uint32_t>(payload.size());
      if (bad_tile_size_slot_ == context.slot && index == 0) {
        size = bad_tile_size_value_;
      }
      context.tile_sizes_out[index] = size;
      if (size == 0) {
        continue;  // let the runtime reject the zero size
      }
      const std::size_t copy_bytes =
          std::min<std::size_t>(static_cast<std::size_t>(size), payload.size());
      const Status status = context.backend->copy_h2d_async(
          context.compute_stream,
          context.device_payload +
              static_cast<std::size_t>(index) * context.payload_stride_bytes,
          payload_buffer_, copy_bytes);
      if (!is_ok(status)) {
        if (error != nullptr) {
          *error = "pattern payload upload failed";
        }
        return status;
      }
      const tiles::TileRect& rect =
          context.layout->tiles[static_cast<std::size_t>(tile)];
      const Status recon_status = enqueue_tile_region_copy(
          *context.backend, CopyDirection::kHostToDevice, context.compute_stream,
          context.device_recon, recon_buffer_, 0, *context.layout, rect,
          context.bit_depth);
      if (!is_ok(recon_status)) {
        if (error != nullptr) {
          *error = "pattern reconstruction upload failed";
        }
        return recon_status;
      }
    }
    if (bad_tile_size_slot_ == context.slot) {
      // The first size is already set above; make sure it is visible.
      context.tile_sizes_out[0] = bad_tile_size_value_;
    }
    visits_.push_back(std::move(visit));
    return Status::kOk;
  }

  int fail_slot_ = -1;   // -1 = any slot
  int fail_frame_ = -1;  // -1 = no injected failure; -2 = any frame
  int bad_tile_size_slot_ = -1;
  std::uint32_t bad_tile_size_value_ = 0;

  const std::vector<Visit>& visits() const { return visits_; }
  void clear_visits() { visits_.clear(); }

 private:
  bool ensure_staging(std::size_t recon_bytes, std::string* error) {
    if (recon_bytes > recon_capacity_) {
      if (recon_buffer_ != nullptr) {
        backend_->pinned_free(recon_buffer_);
        recon_buffer_ = nullptr;
        recon_capacity_ = 0;
      }
      void* pointer = nullptr;
      if (!is_ok(backend_->pinned_alloc(recon_bytes, &pointer, error))) {
        return false;
      }
      recon_buffer_ = static_cast<std::uint8_t*>(pointer);
      recon_capacity_ = recon_bytes;
    }
    if (stride_ > payload_capacity_) {
      if (payload_buffer_ != nullptr) {
        backend_->pinned_free(payload_buffer_);
        payload_buffer_ = nullptr;
        payload_capacity_ = 0;
      }
      void* pointer = nullptr;
      if (!is_ok(backend_->pinned_alloc(stride_, &pointer, error))) {
        return false;
      }
      payload_buffer_ = static_cast<std::uint8_t*>(pointer);
      payload_capacity_ = stride_;
    }
    return true;
  }

  std::uint32_t stride_;
  bool direct_device_access_ = false;
  std::vector<Visit> visits_;
  // Persistent pinned worker staging: the CUDA backend may still be reading
  // these after encode() returns, until the compute-done event completes.
  DeviceBackend* backend_ = nullptr;
  std::uint8_t* payload_buffer_ = nullptr;
  std::uint8_t* recon_buffer_ = nullptr;
  std::size_t payload_capacity_ = 0;
  std::size_t recon_capacity_ = 0;
};

}  // namespace av2enc::multigpu::runtime::testing
