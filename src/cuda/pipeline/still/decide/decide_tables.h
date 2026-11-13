// Host-side packing of the §8.3.1 coefficient CDFs and the §7.16 exact
// `-log2(p)` table for the device decision core.
//
// `decide_core.cuh` addresses every CDF by a byte offset from one contiguous
// blob, so the same POD (`DecTables`) works with host pointers in the parity
// test and with the uploaded arena buffer on the GPU.  The blob is built from
// the shipped `still::qctx` tables; nothing is hand-typed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "codec/entropy/qctx/qctx.h"
#include "cuda/pipeline/still/decide/decide_core.cuh"

namespace av2enc::cuda::still::decide {

// `kDecLog2Entries` doubles: index p in [1, 32768] is
// `-std::log2(p / 32768.0)`, the exact value the host models compute.
inline constexpr int kDecLog2Entries = 32769;

inline std::vector<double> build_log2_prob_table() {
  std::vector<double> table(kDecLog2Entries, 0.0);
  for (int p = 1; p < kDecLog2Entries; ++p) {
    table[static_cast<std::size_t>(p)] =
        -std::log2(static_cast<double>(p) / kDecCdfScale);
  }
  return table;
}

struct DecHostTables {
  std::vector<std::uint8_t> blob;
  DecTables view;
};

namespace detail {

template <typename T>
void append_table(std::vector<std::uint8_t>& blob, const T* src,
                  std::size_t count, std::uint32_t* offset) {
  const std::size_t aligned = (blob.size() + 15u) & ~static_cast<std::size_t>(15u);
  blob.resize(aligned);
  *offset = static_cast<std::uint32_t>(blob.size());
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(src);
  blob.insert(blob.end(), bytes, bytes + count * sizeof(T));
}

}  // namespace detail

// Packs all four qctx slices.  `view.base` is left null; call
// `set_base()` after the blob is uploaded (or point it at `blob.data()` for a
// host-side use).
inline bool build_dec_tables(DecHostTables* out) {
  if (out == nullptr) return false;
  out->blob.clear();
  out->blob.reserve(64 * 1024);
  out->view = DecTables{};
  for (int slice = 0; slice < kDecCdfSlices; ++slice) {
    const av2enc::still::qctx::CoeffCdfSlice& q =
        av2enc::still::qctx::tables_for_slice(slice);
    DecTableOffsets& o = out->view.off[slice];
    detail::append_table(out->blob, &(*q.txb_skip)[0][0][0][0],
                         sizeof(*q.txb_skip) / sizeof(std::uint16_t),
                         &o.txb_skip);
    detail::append_table(out->blob, &(*q.v_txb_skip)[0][0],
                         sizeof(*q.v_txb_skip) / sizeof(std::uint16_t),
                         &o.v_txb_skip);
    detail::append_table(out->blob, &(*q.coeff_base)[0][0][0][0],
                         sizeof(*q.coeff_base) / sizeof(std::uint16_t),
                         &o.coeff_base);
    detail::append_table(out->blob, &(*q.coeff_base_lf)[0][0][0][0],
                         sizeof(*q.coeff_base_lf) / sizeof(std::uint16_t),
                         &o.coeff_base_lf);
    detail::append_table(out->blob, &(*q.coeff_base_uv)[0][0],
                         sizeof(*q.coeff_base_uv) / sizeof(std::uint16_t),
                         &o.coeff_base_uv);
    detail::append_table(out->blob, &(*q.coeff_base_lf_uv)[0][0],
                         sizeof(*q.coeff_base_lf_uv) / sizeof(std::uint16_t),
                         &o.coeff_base_lf_uv);
    detail::append_table(out->blob, &(*q.coeff_base_eob)[0][0][0],
                         sizeof(*q.coeff_base_eob) / sizeof(std::uint16_t),
                         &o.coeff_base_eob);
    detail::append_table(out->blob, &(*q.coeff_base_lf_eob)[0][0][0],
                         sizeof(*q.coeff_base_lf_eob) / sizeof(std::uint16_t),
                         &o.coeff_base_lf_eob);
    detail::append_table(out->blob, &(*q.coeff_base_eob_uv)[0][0],
                         sizeof(*q.coeff_base_eob_uv) / sizeof(std::uint16_t),
                         &o.coeff_base_eob_uv);
    detail::append_table(
        out->blob, &(*q.coeff_base_lf_eob_uv)[0][0],
        sizeof(*q.coeff_base_lf_eob_uv) / sizeof(std::uint16_t),
        &o.coeff_base_lf_eob_uv);
    detail::append_table(out->blob, &(*q.coeff_br)[0][0],
                         sizeof(*q.coeff_br) / sizeof(std::uint16_t),
                         &o.coeff_br);
    detail::append_table(out->blob, &(*q.coeff_br_lf)[0][0],
                         sizeof(*q.coeff_br_lf) / sizeof(std::uint16_t),
                         &o.coeff_br_lf);
    detail::append_table(out->blob, &(*q.coeff_br_uv)[0][0],
                         sizeof(*q.coeff_br_uv) / sizeof(std::uint16_t),
                         &o.coeff_br_uv);
    detail::append_table(out->blob, &(*q.dc_sign)[0][0][0][0],
                         sizeof(*q.dc_sign) / sizeof(std::uint16_t),
                         &o.dc_sign);
    detail::append_table(out->blob, &(*q.eob_extra)[0],
                         sizeof(*q.eob_extra) / sizeof(std::uint16_t),
                         &o.eob_extra);
    detail::append_table(out->blob, &(*q.eob_pt_16)[0][0],
                         sizeof(*q.eob_pt_16) / sizeof(std::uint16_t),
                         &o.eob_pt_16);
    detail::append_table(out->blob, &(*q.eob_pt_32)[0][0],
                         sizeof(*q.eob_pt_32) / sizeof(std::uint16_t),
                         &o.eob_pt_32);
    detail::append_table(out->blob, &(*q.eob_pt_64)[0][0],
                         sizeof(*q.eob_pt_64) / sizeof(std::uint16_t),
                         &o.eob_pt_64);
    detail::append_table(out->blob, &(*q.eob_pt_128)[0][0],
                         sizeof(*q.eob_pt_128) / sizeof(std::uint16_t),
                         &o.eob_pt_128);
    detail::append_table(out->blob, &(*q.eob_pt_256)[0][0],
                         sizeof(*q.eob_pt_256) / sizeof(std::uint16_t),
                         &o.eob_pt_256);
    detail::append_table(out->blob, &(*q.eob_pt_512)[0][0],
                         sizeof(*q.eob_pt_512) / sizeof(std::uint16_t),
                         &o.eob_pt_512);
    detail::append_table(out->blob, &(*q.eob_pt_1024)[0][0],
                         sizeof(*q.eob_pt_1024) / sizeof(std::uint16_t),
                         &o.eob_pt_1024);
  }
  return true;
}

// Total device bytes for the blob plus the log2 table (both 256-aligned by the
// caller's arena allocation).
inline std::size_t dec_tables_device_bytes(std::size_t blob_bytes) {
  return ((blob_bytes + 255u) & ~static_cast<std::size_t>(255u)) +
         sizeof(double) * static_cast<std::size_t>(kDecLog2Entries);
}

}  // namespace av2enc::cuda::still::decide
