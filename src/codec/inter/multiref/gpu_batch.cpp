#include "codec/inter/multiref/gpu_batch.h"

#include <cstddef>

#include "codec/inter/multiref/compound_batch_core.h"

namespace av2enc::video::multiref {

void CompoundBatch::reset(int width, int height) {
  const int cw = (width + 1) / 2;
  const int ch = (height + 1) / 2;
  plane_width[0] = width;
  plane_height[0] = height;
  plane_stride[0] = width;
  plane_width[1] = cw;
  plane_height[1] = ch;
  plane_stride[1] = cw;
  plane_width[2] = cw;
  plane_height[2] = ch;
  plane_stride[2] = cw;
  const std::size_t luma =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const std::size_t chroma =
      static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch);
  plane_dst_offset[0] = 0;
  plane_dst_offset[1] = static_cast<int>(luma);
  plane_dst_offset[2] = static_cast<int>(luma + chroma);
  plane_src_offset[0] = 0;
  plane_src_offset[1] = static_cast<int>(luma);
  plane_src_offset[2] = static_cast<int>(luma + chroma);
  dst.assign(luma + 2 * chroma, 0);
  src.assign(luma + 2 * chroma, 0);
  pred0.clear();
  pred1.clear();
  masks.clear();
  items.clear();
}

int CompoundBatch::alloc_pred(std::size_t samples) {
  const int offset = static_cast<int>(pred0.size());
  pred0.resize(pred0.size() + samples);
  pred1.resize(pred1.size() + samples);
  return offset;
}

int CompoundBatch::alloc_mask(std::size_t samples) {
  const int offset = static_cast<int>(masks.size());
  masks.resize(masks.size() + samples);
  return offset;
}

bool validate_compound_batch(const CompoundBatch& batch, std::string& error) {
  error.clear();
  const auto fits = [](int offset, int stride, int x, int y, int w, int h,
                       std::size_t pool) {
    if (offset < 0 || stride <= 0 || w <= 0 || h <= 0 || x < 0 || y < 0) {
      return false;
    }
    const std::size_t last =
        static_cast<std::size_t>(offset) +
        static_cast<std::size_t>(y + h - 1) * static_cast<std::size_t>(stride) +
        static_cast<std::size_t>(x + w);
    return last <= pool;
  };
  for (std::size_t i = 0; i < batch.items.size(); ++i) {
    const CompoundBatchItem& item = batch.items[i];
    if (item.plane < 0 || item.plane > 2) {
      error = "item " + std::to_string(i) + ": invalid plane";
      return false;
    }
    if (item.rect_w <= 0 || item.rect_h <= 0 || item.rect_w > 64 ||
        item.rect_h > 64) {
      error = "item " + std::to_string(i) + ": invalid rectangle";
      return false;
    }
    if (!fits(item.p0_offset, item.p0_stride, 0, 0, item.rect_w,
              item.rect_h, batch.pred0.size())) {
      error = "item " + std::to_string(i) + ": p0 outside the pool";
      return false;
    }
    if (!fits(item.p1_offset, item.p1_stride, 0, 0, item.rect_w,
              item.rect_h, batch.pred1.size())) {
      error = "item " + std::to_string(i) + ": p1 outside the pool";
      return false;
    }
    if (!fits(item.dst_offset, item.dst_stride, item.rect_x, item.rect_y,
              item.rect_w, item.rect_h, batch.dst.size())) {
      error = "item " + std::to_string(i) + ": dst outside the pool";
      return false;
    }
    if (item.compute_sse &&
        !fits(item.src_offset, item.src_stride, item.rect_x, item.rect_y,
              item.rect_w, item.rect_h, batch.src.size())) {
      error = "item " + std::to_string(i) + ": src outside the pool";
      return false;
    }
    if (item.mask_stride > 0 &&
        !fits(item.mask_offset, item.mask_stride, 0, 0, item.rect_w,
              item.rect_h, batch.masks.size())) {
      error = "item " + std::to_string(i) + ": mask outside the pool";
      return false;
    }
    if (item.luma_mask_stride > 0 &&
        !fits(item.luma_mask_offset, item.luma_mask_stride, 0, 0, item.rect_w,
              item.rect_h, batch.masks.size())) {
      error = "item " + std::to_string(i) + ": luma mask outside the pool";
      return false;
    }
  }
  return true;
}

bool run_compound_batch_cpu(CompoundBatch& batch, std::string& error) {
  error.clear();
  if (!validate_compound_batch(batch, error)) return false;
  for (CompoundBatchItem& item : batch.items) {
    adv::CompoundJob job{};
    load_compound_job(item, batch.pred0.data(), batch.pred1.data(),
                      batch.dst.data(), batch.masks.data(), job);
    if (!adv::apply_compound(job)) {
      error = "compound batch item rejected";
      return false;
    }
    if (item.compute_sse) {
      item.sse_out = adv::block_sse(
          batch.dst.data() + item.dst_offset, item.dst_stride, item.rect_x,
          item.rect_y, batch.src.data() + item.src_offset, item.src_stride,
          item.rect_x, item.rect_y, item.rect_w, item.rect_h);
    }
  }
  return true;
}

}  // namespace av2enc::video::multiref
