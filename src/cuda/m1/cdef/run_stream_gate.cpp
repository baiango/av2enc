// M1 GPU path — CDEF stream gate (dav2d).
//
// Emits two real AV2 streams from one M1 encode and decodes both with the
// pinned `dav2d`:
//
//   baseline : the current M1 stream (enable_cdef = 0)
//   cdef     : the same tile payload with enable_cdef = 1 and cdef_params()
//              added to the frame header, CdefStrengths = 1 (no cdef_idx
//              symbols, so the tile payload is byte-identical)
//
// It then checks, byte for byte on all three planes:
//
//   decoded(baseline) == M1Result.recon_*        (M1 still exact)
//   decoded(cdef)     == §7.18 CDEF(recon)       (this stage's filter)
//
// and reports the quality delta (PSNR) against the original picture.
//
// Why CdefStrengths == 1: §5.20.10.1 only codes `cdef_index0` when
// CdefStrengths > 1, so a single strength set costs zero tile bits.  Emitting
// per-64x64 indices needs `read_cdef()` in the M1 syntax writer (see README.md
// for the exact code); until that lands the multi-strength plan is measured as
// a component delta only.
//
// The gate also rebuilds the *baseline* headers with its own writer first and
// refuses to run unless that reproduces the M1 stream exactly.  That is what
// makes the header rewrite trustworthy.
//
// Build/run: ./build_and_test.sh --gate <src.y4m>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "cuda/m1/cdef/m1_cdef.h"
#include "encoder/filters/cdef/cdef.h"
#include "encoder/filters/cdef/cdef_search.h"
#include "encoder/m1/m1_headers.h"
#include "encoder/m1/m1_still.h"
#include "io/ivf.h"
#include "io/y4m.h"

namespace {

using av2enc::BitWriter;
using av2enc::cdef::CdefStatus;
using av2enc::cuda::m1::CdefMiGrids;
using av2enc::cuda::m1::CdefMiInputs;
using av2enc::cuda::m1::CdefStageFrame;
using av2enc::cuda::m1::CdefStageOptions;
using av2enc::cuda::m1::CdefStagePlan;

int g_failures = 0;

void fail(const char* what) {
  ++g_failures;
  std::fprintf(stderr, "FAIL: %s\n", what);
}

// --------------------------------------------------------------------------
// Header writer (mirrors src/encoder/m1/m1_headers.cpp plus CDEF)
// --------------------------------------------------------------------------

void put_uvlc(BitWriter& w, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
  }
  w.put_bits(0U, leading_zeros);
  w.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  w.put_bits(payload, leading_zeros);
}

void put_ns(BitWriter& w, std::uint32_t n, std::uint32_t value) {
  unsigned width = 0;
  while ((1U << width) < n) {
    ++width;
  }
  const std::uint32_t m = (1U << width) - n;
  if (value < m) {
    w.put_bits(value, width - 1U);
  } else {
    const std::uint32_t x = value + m;
    w.put_bits(x >> 1U, width - 1U);
    w.put_bit((x & 1U) != 0U);
  }
}

unsigned bit_width(std::uint32_t value) {
  unsigned bits = 0;
  while (value != 0) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

void put_trailing_bits(BitWriter& w) {
  w.put_bit(true);
  w.align_to_byte();
}

struct SeqConfig {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint8_t level = 7;
  bool enable_cdef = false;
};

void write_sequence_header(BitWriter& w, const SeqConfig& c) {
  put_uvlc(w, 0);                        // seq_header_id
  w.put_bits(0U, 5U);                    // seq_profile_idc
  w.put_bit(true);                       // single_picture_header_flag
  w.put_bits(c.level, 5U);               // seq_level_idx
  put_uvlc(w, 0);                        // chroma_format_idc = 4:2:0
  put_uvlc(w, 1);                        // bit_depth_idc = 1 => 8-bit
  unsigned width_bits = bit_width(c.width - 1U);
  if (width_bits == 0) width_bits = 1;
  unsigned height_bits = bit_width(c.height - 1U);
  if (height_bits == 0) height_bits = 1;
  w.put_bits(width_bits - 1U, 4U);
  w.put_bits(height_bits - 1U, 4U);
  w.put_bits(c.width - 1U, width_bits);
  w.put_bits(c.height - 1U, height_bits);
  w.put_bit(false);  // seq_cropping_window_present_flag
  w.put_bit(false);  // use_256x256_superblock
  w.put_bit(false);  // use_128x128_superblock
  w.put_bit(false);  // enable_sdp
  w.put_bit(false);  // enable_ext_partitions
  w.put_bit(false);  // reduce_pb_aspect_ratio
  w.put_bit(false);  // enable_ext_seg
  w.put_bit(false);  // seq_seg_info_present_flag
  w.put_bit(false);  // enable_dip
  w.put_bit(true);   // enable_intra_edge_filter
  w.put_bit(false);  // enable_mrls
  w.put_bit(true);   // enable_cfl_intra
  w.put_bits(0U, 2U);  // cfl_ds_filter_index
  w.put_bit(false);  // enable_mhccp
  w.put_bit(false);  // enable_ibp
  w.put_bit(false);  // enable_refmvbank
  w.put_bit(true);   // disable_drl_reorder
  put_ns(w, 3U, 0U); // seq_max_bvp_drl_bits_minus_1
  w.put_bit(false);  // allow_frame_max_bvp_drl_bits
  w.put_bit(false);  // enable_bawp
  w.put_bit(false);  // enable_fsc
  w.put_bit(false);  // enable_idtx_intra
  w.put_bit(false);  // enable_intra_ist
  w.put_bit(false);  // enable_inter_ist
  w.put_bit(true);   // enable_chroma_dctonly
  w.put_bit(false);  // reduced_tx_part_set
  w.put_bit(false);  // enable_cctx
  w.put_bit(false);  // enable_tcq
  w.put_bit(false);  // enable_parity_hiding
  w.put_bit(false);  // separate_uv_delta_q
  w.put_bit(true);   // equal_ac_dc_q
  w.put_bits(23U, 5U);  // base_uv_ac_delta_q
  w.put_bit(false);     // uv_ac_delta_q_enabled
  // §5.4.10 sequence_filter_config()
  w.put_bit(false);                  // disable_loopfilters_across_tiles
  w.put_bit(c.enable_cdef);          // enable_cdef
  w.put_bit(false);                  // enable_gdf
  w.put_bit(false);                  // enable_restoration
  w.put_bit(false);                  // enable_ccso
  w.put_bits(0U, 2U);                // df_par_bits_minus_2
  w.put_bit(false);                  // seq_tile_info_present_flag
  w.put_bit(false);                  // film_grain_params_present
  w.put_bit(false);                  // obu_extension_flag (m1_headers layout)
  put_trailing_bits(w);
}

struct FrameConfig {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint8_t base_q_idx = 90;
  bool deblock = true;
  bool cdef = false;
  // Written only when the pinned decoder expects them; see README.md.
  bool write_df_delta_q_present = true;
  const av2enc::cuda::m1::CdefHeaderParams* cdef_header = nullptr;
};

void write_frame_header(BitWriter& w, const FrameConfig& c) {
  const std::uint32_t mi_cols = 2U * ((c.width + 7U) >> 3U);
  const std::uint32_t mi_rows = 2U * ((c.height + 7U) >> 3U);
  const std::uint32_t sb_cols = (mi_cols + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows + 15U) >> 4U;
  put_uvlc(w, 0);       // cur_mfh_id
  put_uvlc(w, 0);       // seq_header_id_in_frame_header
  w.put_bit(false);     // allow_screen_content_tools
  w.put_bit(false);     // allow_intrabc
  w.put_bit(true);      // disable_cdf_update
  w.put_bit(true);      // uniform_tile_spacing_flag
  if (sb_cols > 1U) w.put_bit(false);
  if (sb_rows > 1U) w.put_bit(false);
  w.put_bits(c.base_q_idx, 8U);
  w.put_bit(false);     // segmentation_enabled
  w.put_bit(false);     // using_qmatrix
  w.put_bit(false);     // delta_q_present
  // §5.18.5.2 deblocking_filter_params()
  w.put_bit(c.deblock);
  w.put_bit(c.deblock);
  if (c.deblock) {
    w.put_bit(c.deblock);
    w.put_bit(c.deblock);
    if (c.write_df_delta_q_present) {
      for (int i = 0; i < 4; ++i) {
        w.put_bit(false);  // df_delta_q_present[i] = 0
      }
    }
  }
  // gdf_params() writes nothing (enable_gdf = 0).
  // §5.18.7.10 cdef_params()
  if (c.cdef && c.cdef_header != nullptr) {
    const auto& h = *c.cdef_header;
    // single_picture_header_flag => cdef_frame_enable = 1, no bit.
    w.put_bits(static_cast<std::uint64_t>(h.damping_minus_3), 2U);
    w.put_bits(static_cast<std::uint64_t>(h.strengths_minus_1), 3U);
    w.put_bit(h.on_skip_txfm_frame_enable);
    for (int i = 0; i < h.num_strengths; ++i) {
      const auto& s = h.set[i];
      w.put_bit(s.y_pri_zero);
      if (!s.y_pri_zero) {
        w.put_bits(static_cast<std::uint64_t>(s.y_pri_strength), 4U);
      }
      w.put_bits(static_cast<std::uint64_t>(s.y_sec_strength), 2U);
      w.put_bit(s.uv_pri_zero);
      if (!s.uv_pri_zero) {
        w.put_bits(static_cast<std::uint64_t>(s.uv_pri_strength), 4U);
      }
      w.put_bits(static_cast<std::uint64_t>(s.uv_sec_strength), 2U);
    }
  }
  // lr_params(), ccso_params() write nothing (both disabled).
  w.put_bit(false);      // tx_mode_select = TX_MODE_LARGEST
  w.put_bits(0U, 2U);    // reduced_tx_set
}

std::vector<std::uint8_t> make_obu(std::uint8_t type,
                                   std::span<const std::uint8_t> payload) {
  std::vector<std::uint8_t> out;
  std::uint32_t size = static_cast<std::uint32_t>(payload.size()) + 1U;
  do {
    std::uint8_t byte = static_cast<std::uint8_t>(size & 0x7FU);
    size >>= 7U;
    if (size != 0) byte |= 0x80U;
    out.push_back(byte);
  } while (size != 0);
  out.push_back(static_cast<std::uint8_t>(type << 2));
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

std::vector<std::uint8_t> assemble_frame(const SeqConfig& seq,
                                         const FrameConfig& frame,
                                         std::span<const std::uint8_t> tile) {
  BitWriter seq_writer;
  write_sequence_header(seq_writer, seq);
  BitWriter frame_writer;
  frame_writer.put_bit(true);  // is_first_tile_group
  write_frame_header(frame_writer, frame);
  frame_writer.align_to_byte();
  const std::vector<std::uint8_t> seq_bytes = seq_writer.take_bytes();
  const std::vector<std::uint8_t> frame_bytes = frame_writer.take_bytes();
  std::vector<std::uint8_t> payload = frame_bytes;
  payload.insert(payload.end(), tile.begin(), tile.end());

  std::vector<std::uint8_t> out;
  const std::vector<std::uint8_t> td = make_obu(2, {});
  const std::vector<std::uint8_t> seq_obu = make_obu(1, seq_bytes);
  const std::vector<std::uint8_t> clk_obu = make_obu(4, payload);
  out.insert(out.end(), td.begin(), td.end());
  out.insert(out.end(), seq_obu.begin(), seq_obu.end());
  out.insert(out.end(), clk_obu.begin(), clk_obu.end());
  return out;
}

// --------------------------------------------------------------------------
// OBU parsing / file helpers
// --------------------------------------------------------------------------

struct Obus {
  std::vector<std::uint8_t> seq;
  std::vector<std::uint8_t> clk;
};

bool parse_obus(std::span<const std::uint8_t> data, Obus* out,
                std::string* error) {
  std::size_t off = 0;
  while (off < data.size()) {
    std::uint32_t size = 0;
    unsigned shift = 0;
    while (true) {
      if (off >= data.size() || shift > 28) {
        *error = "truncated OBU size";
        return false;
      }
      const std::uint8_t byte = data[off++];
      size |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
      shift += 7;
      if ((byte & 0x80U) == 0) break;
    }
    if (off + size > data.size() || size < 1) {
      *error = "truncated OBU payload";
      return false;
    }
    const std::uint8_t header = data[off];
    const int type = (header >> 2) & 0x1f;
    const std::span<const std::uint8_t> payload = data.subspan(off + 1, size - 1);
    if (type == 1) {
      out->seq.assign(payload.begin(), payload.end());
    } else if (type == 4) {
      out->clk.assign(payload.begin(), payload.end());
    }
    off += size;
  }
  return true;
}

bool write_ivf(const std::string& path, int width, int height,
               std::span<const std::uint8_t> frame, std::uint32_t fps_num,
               std::uint32_t fps_den, std::string* error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = fps_num;
  header.fps_den = fps_den;
  if (!writer.open(path, header, *error)) return false;
  if (!writer.write_frame(frame, 0, *error)) return false;
  return writer.close(*error);
}

struct Planes {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

bool load_first_frame(const std::string& path, Planes* out, std::string* error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult result = reader.read_frame(frame, *error);
  if (result != av2enc::Y4mReadResult::kFrame) {
    if (error->empty()) *error = "no frame in " + path;
    return false;
  }
  out->width = static_cast<int>(frame.width());
  out->height = static_cast<int>(frame.height());
  out->y.assign(frame.y().begin(), frame.y().end());
  out->u.assign(frame.u().begin(), frame.u().end());
  out->v.assign(frame.v().begin(), frame.v().end());
  return true;
}

bool run_decoder(const std::string& dav2d, const std::string& ivf,
                 const std::string& y4m, std::string* error) {
  const std::string command = "'" + dav2d + "' -i '" + ivf + "' -o '" + y4m +
                              "' >/dev/null 2>&1";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    *error = "dav2d failed on " + ivf;
    return false;
  }
  return true;
}

double psnr(const std::vector<std::uint8_t>& a,
            const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size() || a.empty()) return 0.0;
  std::uint64_t se = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
    se += static_cast<std::uint64_t>(d * d);
  }
  if (se == 0) return 99.0;
  const double mse = static_cast<double>(se) / static_cast<double>(a.size());
  return 10.0 * std::log10(255.0 * 255.0 / mse);
}

// Sample-weighted PSNR over the 4:2:0 frame (Y has 4x the chroma samples).
double combined_psnr(const Planes& a, const Planes& b) {
  std::uint64_t se = 0;
  std::uint64_t n = 0;
  const auto add = [&se, &n](const std::vector<std::uint8_t>& x,
                             const std::vector<std::uint8_t>& y,
                             std::uint64_t weight) {
    for (std::size_t i = 0; i < x.size(); ++i) {
      const int d = static_cast<int>(x[i]) - static_cast<int>(y[i]);
      se += static_cast<std::uint64_t>(d * d) * weight;
      n += weight;
    }
  };
  add(a.y, b.y, 4);
  add(a.u, b.u, 1);
  add(a.v, b.v, 1);
  if (se == 0) return 99.0;
  const double mse = static_cast<double>(se) / static_cast<double>(n);
  return 10.0 * std::log10(255.0 * 255.0 / mse);
}

// Plane dump shared with the CUDA parity test (m1_cdef_gpu_test.cu):
//   "CDEFPL01", u32 width, u32 height,
//   recon Y, U, V (tightly packed), source Y, U, V.
bool write_plane_dump(const std::string& path, const Planes& recon,
                      const Planes& source, std::string* error) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    *error = "cannot open " + path;
    return false;
  }
  const char magic[8] = {'C', 'D', 'E', 'F', 'P', 'L', '0', '1'};
  out.write(magic, 8);
  const std::uint32_t width = static_cast<std::uint32_t>(source.width);
  const std::uint32_t height = static_cast<std::uint32_t>(source.height);
  out.write(reinterpret_cast<const char*>(&width), 4);
  out.write(reinterpret_cast<const char*>(&height), 4);
  out.write(reinterpret_cast<const char*>(recon.y.data()),
            static_cast<std::streamsize>(recon.y.size()));
  out.write(reinterpret_cast<const char*>(recon.u.data()),
            static_cast<std::streamsize>(recon.u.size()));
  out.write(reinterpret_cast<const char*>(recon.v.data()),
            static_cast<std::streamsize>(recon.v.size()));
  out.write(reinterpret_cast<const char*>(source.y.data()),
            static_cast<std::streamsize>(source.y.size()));
  out.write(reinterpret_cast<const char*>(source.u.data()),
            static_cast<std::streamsize>(source.u.size()));
  out.write(reinterpret_cast<const char*>(source.v.data()),
            static_cast<std::streamsize>(source.v.size()));
  if (!out) {
    *error = "write failed: " + path;
    return false;
  }
  return true;
}

// Writes one 4:2:0 frame as Y4M so tools/quality.py can score it.
bool write_y4m(const std::string& path, const Planes& planes,
               std::string* error) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    *error = "cannot open " + path;
    return false;
  }
  out << "YUV4MPEG2 W" << planes.width << " H" << planes.height
      << " F30:1 Ip A1:1 C420\nFRAME\n";
  out.write(reinterpret_cast<const char*>(planes.y.data()),
            static_cast<std::streamsize>(planes.y.size()));
  out.write(reinterpret_cast<const char*>(planes.u.data()),
            static_cast<std::streamsize>(planes.u.size()));
  out.write(reinterpret_cast<const char*>(planes.v.data()),
            static_cast<std::streamsize>(planes.v.size()));
  if (!out) {
    *error = "write failed: " + path;
    return false;
  }
  return true;
}

int count_mismatch(const std::vector<std::uint8_t>& a,
                   const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size()) return -1;
  int count = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) ++count;
  }
  return count;
}

// CDEF plane set on the MI-aligned grid.
struct CdefPlanes {
  int width = 0;
  int height = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  std::vector<std::uint8_t> src[3];
  std::vector<std::uint8_t> dst[3];
  int stride[3] = {0, 0, 0};
};

void init_cdef_planes(CdefPlanes* p, int width, int height) {
  p->width = width;
  p->height = height;
  p->mi_cols = 2 * ((width + 7) >> 3);
  p->mi_rows = 2 * ((height + 7) >> 3);
  for (int plane = 0; plane < 3; ++plane) {
    const int sub = (plane == 0) ? 0 : 1;
    const int w = (p->mi_cols * 4) >> sub;
    const int h = (p->mi_rows * 4) >> sub;
    p->stride[plane] = w;
    p->src[plane].assign(static_cast<std::size_t>(w) *
                             static_cast<std::size_t>(h),
                         0);
    p->dst[plane].assign(p->src[plane].size(), 0);
  }
}

void copy_visible_into(CdefPlanes* p, const Planes& visible) {
  for (int plane = 0; plane < 3; ++plane) {
    const int sub = (plane == 0) ? 0 : 1;
    const int w = (p->mi_cols * 4) >> sub;
    const int vw = (visible.width + (1 << sub) - 1) >> sub;
    const int vh = (visible.height + (1 << sub) - 1) >> sub;
    const std::vector<std::uint8_t>* source =
        plane == 0 ? &visible.y : (plane == 1 ? &visible.u : &visible.v);
    for (int y = 0; y < vh && y < ((p->mi_rows * 4) >> sub); ++y) {
      std::memcpy(p->src[plane].data() +
                      static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(w),
                  source->data() + static_cast<std::size_t>(y) *
                                       static_cast<std::size_t>(vw),
                  static_cast<std::size_t>(vw) * sizeof(std::uint8_t));
    }
  }
}

void visible_from_cdef(const CdefPlanes& p, Planes* out) {
  out->width = p.width;
  out->height = p.height;
  for (int plane = 0; plane < 3; ++plane) {
    const int sub = (plane == 0) ? 0 : 1;
    const int vw = (p.width + (1 << sub) - 1) >> sub;
    const int vh = (p.height + (1 << sub) - 1) >> sub;
    std::vector<std::uint8_t>* dest =
        plane == 0 ? &out->y : (plane == 1 ? &out->u : &out->v);
    dest->resize(static_cast<std::size_t>(vw) * static_cast<std::size_t>(vh));
    for (int y = 0; y < vh; ++y) {
      std::memcpy(dest->data() + static_cast<std::size_t>(y) *
                                     static_cast<std::size_t>(vw),
                  p.dst[plane].data() +
                      static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(p.stride[plane]),
                  static_cast<std::size_t>(vw) * sizeof(std::uint8_t));
    }
  }
}

int run_plan_and_filter(CdefPlanes* planes, const Planes& source,
                        const CdefMiGrids& grids, int num_strengths,
                        int damping, CdefStagePlan* plan,
                        double* plan_ms = nullptr,
                        double* filter_ms = nullptr) {
  CdefStageFrame view{};
  for (int plane = 0; plane < 3; ++plane) {
    view.plane[plane].src = planes->src[plane].data();
    view.plane[plane].dst = planes->dst[plane].data();
    view.plane[plane].stride = planes->stride[plane];
    view.ref[plane] = plane == 0 ? source.y.data()
                                 : (plane == 1 ? source.u.data()
                                               : source.v.data());
    const int sub = (plane == 0) ? 0 : 1;
    view.ref_stride[plane] =
        (source.width + (1 << sub) - 1) >> sub;
  }
  view.width = planes->width;
  view.height = planes->height;
  view.mi_cols = planes->mi_cols;
  view.mi_rows = planes->mi_rows;
  view.skip = grids.skip.data();
  view.lossless = grids.lossless.data();
  view.lossless_chroma = grids.lossless_chroma.data();

  CdefStageOptions options;
  options.damping = damping;
  options.num_strengths = num_strengths;
  options.include_off = true;
  std::string error;
  const auto plan_start = std::chrono::steady_clock::now();
  if (!av2enc::cuda::m1::m1_cdef_plan(view, options, plan, &error)) {
    std::fprintf(stderr, "plan failed: %s\n", error.c_str());
    return -1;
  }
  if (plan_ms != nullptr) {
    *plan_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - plan_start)
                   .count();
  }
  const auto filter_start = std::chrono::steady_clock::now();
  av2enc::cdef::CdefFrame frame{};
  for (int plane = 0; plane < 3; ++plane) {
    frame.planes[plane].src = planes->src[plane].data();
    frame.planes[plane].dst = planes->dst[plane].data();
    frame.planes[plane].stride = planes->stride[plane];
  }
  frame.mi_cols = planes->mi_cols;
  frame.mi_rows = planes->mi_rows;
  frame.subsampling_x = 1;
  frame.subsampling_y = 1;
  frame.skip = grids.skip.data();
  frame.lossless = grids.lossless.data();
  frame.lossless_chroma = grids.lossless_chroma.data();
  frame.cdef_idx = plan->cdef_idx.data();
  frame.n64_cols = plan->n64_cols;
  frame.n64_rows = plan->n64_rows;
  const CdefStatus status = av2enc::cdef::cdef_frame(frame, plan->search.options);
  if (status != CdefStatus::kOk) {
    std::fprintf(stderr, "cdef_frame failed: %s\n",
                 av2enc::cdef::cdef_status_string(status));
    return -1;
  }
  if (filter_ms != nullptr) {
    *filter_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - filter_start)
                     .count();
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <src.y4m> <out_dir> [--strengths N] [--damping D]\n"
                 "       [--dav2d PATH] [--no-df-check]\n",
                 argv[0]);
    return 2;
  }
  const std::string src_path = argv[1];
  const std::string out_dir = argv[2];
  int report_strengths = 4;
  int damping = 3;
  std::string dav2d = "tools/bin/dav2d";
  std::string dump_path;
  std::string y4m_path;
  bool run_df_check = true;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--strengths" && i + 1 < argc) {
      report_strengths = std::atoi(argv[++i]);
    } else if (arg == "--damping" && i + 1 < argc) {
      damping = std::atoi(argv[++i]);
    } else if (arg == "--dav2d" && i + 1 < argc) {
      dav2d = argv[++i];
    } else if (arg == "--dump-planes" && i + 1 < argc) {
      dump_path = argv[++i];
    } else if (arg == "--write-y4m" && i + 1 < argc) {
      y4m_path = argv[++i];
    } else if (arg == "--no-df-check") {
      run_df_check = false;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }
  if (!std::filesystem::exists(dav2d)) {
    const std::string macos = dav2d + "-macos";
    if (std::filesystem::exists(macos)) {
      dav2d = macos;
    } else {
      std::fprintf(stderr, "dav2d not found: %s\n", dav2d.c_str());
      return 127;
    }
  }
  std::filesystem::create_directories(out_dir);

  std::string error;
  Planes source;
  if (!load_first_frame(src_path, &source, &error)) {
    std::fprintf(stderr, "cannot read %s: %s\n", src_path.c_str(),
                 error.c_str());
    return 1;
  }
  std::printf("source: %s %dx%d\n", src_path.c_str(), source.width,
              source.height);

  av2enc::m1::StillPicture picture;
  picture.width = static_cast<std::uint32_t>(source.width);
  picture.height = static_cast<std::uint32_t>(source.height);
  picture.y = source.y;
  picture.u = source.u;
  picture.v = source.v;
  av2enc::m1::M1Options options;
  av2enc::m1::M1Result encoded;
  if (!av2enc::m1::encode_m1_still_picture(picture, options, encoded, error)) {
    std::fprintf(stderr, "M1 encode failed: %s\n", error.c_str());
    return 1;
  }
  std::printf("M1 baseline: %u bytes, tile %u bytes, %d leaves\n",
              encoded.frame_bytes, encoded.tile_bytes, encoded.leaves);

  // Baseline IVF + OBU split.
  const std::string base_ivf = out_dir + "/baseline.ivf";
  if (!write_ivf(base_ivf, source.width, source.height, encoded.frame, 30, 1,
                 &error)) {
    std::fprintf(stderr, "cannot write %s: %s\n", base_ivf.c_str(),
                 error.c_str());
    return 1;
  }
  Obus obus;
  if (!parse_obus(encoded.frame, &obus, &error)) {
    std::fprintf(stderr, "cannot parse the M1 stream: %s\n", error.c_str());
    return 1;
  }
  if (obus.clk.size() < encoded.tile_bytes) {
    std::fprintf(stderr, "tile payload larger than the CLK OBU\n");
    return 1;
  }
  const std::size_t tile_offset = obus.clk.size() - encoded.tile_bytes;
  const std::span<const std::uint8_t> tile(
      obus.clk.data() + tile_offset, encoded.tile_bytes);

  // Self-check: rebuild the *baseline* headers and require byte identity.
  SeqConfig seq;
  seq.width = picture.width;
  seq.height = picture.height;
  FrameConfig frame_config;
  frame_config.width = picture.width;
  frame_config.height = picture.height;
  frame_config.base_q_idx = options.base_q_idx;
  frame_config.deblock = options.enable_deblock;
  frame_config.write_df_delta_q_present = true;
  const std::vector<std::uint8_t> rebuilt =
      assemble_frame(seq, frame_config, tile);
  const bool header_identity =
      rebuilt.size() == encoded.frame.size() &&
      std::memcmp(rebuilt.data(), encoded.frame.data(), rebuilt.size()) == 0;
  std::printf("header rewrite self-check (CDEF off): %s\n",
              header_identity ? "PASS (byte-identical)" : "FAIL");
  if (!header_identity) {
    fail("CDEF-off header rewrite is not byte-identical to the M1 stream");
    return 1;
  }

  // CDEF planes from the encoder reconstruction.
  const int aligned_w = 4 * (2 * ((source.width + 7) >> 3));
  const int aligned_h = 4 * (2 * ((source.height + 7) >> 3));
  if (source.width != aligned_w || source.height != aligned_h) {
    std::printf(
        "note: %dx%d is not MI-aligned; the plane comparison covers the "
        "visible area only\n",
        source.width, source.height);
  }
  CdefPlanes planes;
  init_cdef_planes(&planes, source.width, source.height);
  Planes recon;
  recon.width = source.width;
  recon.height = source.height;
  recon.y = encoded.recon_y;
  recon.u = encoded.recon_u;
  recon.v = encoded.recon_v;
  copy_visible_into(&planes, recon);
  if (!dump_path.empty()) {
    if (!write_plane_dump(dump_path, recon, source, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    std::printf("wrote plane dump: %s\n", dump_path.c_str());
  }

  CdefMiInputs mi_inputs;
  mi_inputs.mi_cols = planes.mi_cols;
  mi_inputs.mi_rows = planes.mi_rows;
  CdefMiGrids grids;
  if (!av2enc::cuda::m1::m1_cdef_build_mi_grids(mi_inputs, &grids, &error)) {
    std::fprintf(stderr, "MI grids failed: %s\n", error.c_str());
    return 1;
  }

  // Stream plan: one strength set (zero tile bits).
  CdefStagePlan stream_plan;
  double plan_ms = 0.0;
  double filter_ms = 0.0;
  if (run_plan_and_filter(&planes, source, grids, /*num_strengths=*/1, damping,
                          &stream_plan, &plan_ms, &filter_ms) != 0) {
    return 1;
  }
  std::printf("timing (CPU reference, single thread): plan %.1f ms, filter %.1f ms\n",
              plan_ms, filter_ms);
  std::printf(
      "CDEF stream plan: %d strength set(s), y %d/%d uv %d/%d, damping %d, "
      "header %d bits\n",
      stream_plan.num_strengths, stream_plan.search.strengths[0].y_pri,
      stream_plan.search.strengths[0].y_sec,
      stream_plan.search.strengths[0].uv_pri,
      stream_plan.search.strengths[0].uv_sec, damping, stream_plan.header_bits);

  Planes filtered;
  visible_from_cdef(planes, &filtered);

  // Emit the CDEF stream.
  SeqConfig cdef_seq = seq;
  cdef_seq.enable_cdef = true;
  FrameConfig cdef_frame = frame_config;
  cdef_frame.cdef = true;
  cdef_frame.cdef_header = &stream_plan.header;
  const std::vector<std::uint8_t> cdef_frame_bytes =
      assemble_frame(cdef_seq, cdef_frame, tile);
  const std::string cdef_ivf = out_dir + "/cdef.ivf";
  if (!write_ivf(cdef_ivf, source.width, source.height, cdef_frame_bytes, 30, 1,
                 &error)) {
    std::fprintf(stderr, "cannot write %s: %s\n", cdef_ivf.c_str(),
                 error.c_str());
    return 1;
  }
  std::printf("CDEF stream: %zu bytes (+%zu), tile unchanged %u bytes\n",
              cdef_frame_bytes.size(),
              cdef_frame_bytes.size() - encoded.frame.size(),
              encoded.tile_bytes);

  // Decode both.
  const std::string base_y4m = out_dir + "/baseline.y4m";
  const std::string cdef_y4m = out_dir + "/cdef.y4m";
  if (!run_decoder(dav2d, base_ivf, base_y4m, &error) ||
      !run_decoder(dav2d, cdef_ivf, cdef_y4m, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    fail("dav2d did not decode a stream");
    return 1;
  }
  Planes decoded_base;
  Planes decoded_cdef;
  if (!load_first_frame(base_y4m, &decoded_base, &error) ||
      !load_first_frame(cdef_y4m, &decoded_cdef, &error)) {
    std::fprintf(stderr, "cannot read dav2d output: %s\n", error.c_str());
    return 1;
  }

  // (1) M1 still exact: decoded baseline == encoder reconstruction.
  const int base_y = count_mismatch(decoded_base.y, recon.y);
  const int base_u = count_mismatch(decoded_base.u, recon.u);
  const int base_v = count_mismatch(decoded_base.v, recon.v);
  std::printf("decoded(baseline) vs M1 recon: Y %d U %d V %d mismatches\n",
              base_y, base_u, base_v);
  if (base_y != 0 || base_u != 0 || base_v != 0) {
    fail("baseline decode is not bit-exact with the M1 reconstruction");
  }

  // (2) CDEF parity: decoded(cdef) == our §7.18 output.
  const int cdef_y = count_mismatch(decoded_cdef.y, filtered.y);
  const int cdef_u = count_mismatch(decoded_cdef.u, filtered.u);
  const int cdef_v = count_mismatch(decoded_cdef.v, filtered.v);
  std::printf("decoded(cdef) vs stage filter: Y %d U %d V %d mismatches\n",
              cdef_y, cdef_u, cdef_v);
  if (cdef_y != 0 || cdef_u != 0 || cdef_v != 0) {
    fail("dav2d's CDEF output differs from the stage filter");
  }

  // Quality delta against the original picture.
  const double base_y_psnr = psnr(source.y, decoded_base.y);
  const double base_u_psnr = psnr(source.u, decoded_base.u);
  const double base_v_psnr = psnr(source.v, decoded_base.v);
  const double cdef_y_psnr = psnr(source.y, decoded_cdef.y);
  const double cdef_u_psnr = psnr(source.u, decoded_cdef.u);
  const double cdef_v_psnr = psnr(source.v, decoded_cdef.v);
  std::printf(
      "PSNR            Y        U        V      combined\n"
      "baseline   %8.4f %8.4f %8.4f %8.4f\n"
      "cdef       %8.4f %8.4f %8.4f %8.4f\n"
      "delta      %+8.4f %+8.4f %+8.4f %+8.4f\n",
      base_y_psnr, base_u_psnr, base_v_psnr, combined_psnr(source, decoded_base),
      cdef_y_psnr, cdef_u_psnr, cdef_v_psnr, combined_psnr(source, decoded_cdef),
      cdef_y_psnr - base_y_psnr, cdef_u_psnr - base_u_psnr,
      cdef_v_psnr - base_v_psnr,
      combined_psnr(source, decoded_cdef) - combined_psnr(source, decoded_base));

  // Component-only: what the M1 owner unlocks by coding cdef_idx.
  if (report_strengths > 1) {
    CdefPlanes multi_planes = planes;
    CdefStagePlan multi_plan;
    if (run_plan_and_filter(&multi_planes, source, grids, report_strengths,
                            damping, &multi_plan) == 0) {
      Planes multi;
      visible_from_cdef(multi_planes, &multi);
      std::printf(
          "index-enabled plan (%d sets): sse %lld -> %lld, header %d bits, "
          "index ~%d bits; PSNR Y %.4f U %.4f V %.4f combined %.4f "
          "(delta vs baseline %+0.4f)\n",
          multi_plan.num_strengths,
          static_cast<long long>(multi_plan.sse_off_total),
          static_cast<long long>(multi_plan.sse_total),
          multi_plan.header_bits, multi_plan.index_bits_estimate,
          psnr(source.y, multi.y), psnr(source.u, multi.u),
          psnr(source.v, multi.v), combined_psnr(source, multi),
          combined_psnr(source, multi) - combined_psnr(source, decoded_base));
      int used = 0;
      std::printf("  strength sets:");
      for (int i = 0; i < multi_plan.num_strengths; ++i) {
        const auto& s = multi_plan.search.strengths[i];
        used += s.y_pri != 0 ? 1 : 0;
        std::printf(" [%d] y %d/%d uv %d/%d", i, s.y_pri, s.y_sec, s.uv_pri,
                    s.uv_sec);
      }
      std::printf("  (%d non-zero, %d blocks chose copy)\n", used,
                  multi_plan.search.blocks_off);
      if (!y4m_path.empty()) {
        std::string write_error;
        if (write_y4m(y4m_path, multi, &write_error)) {
          std::printf("  wrote index-enabled filtered frame: %s\n",
                      y4m_path.c_str());
        } else {
          std::fprintf(stderr, "%s\n", write_error.c_str());
        }
      }
    }
  }

  // Evidence for the df_delta_q_present note: without the four zero bits the
  // pinned decoder desyncs.  This is expected to fail.
  if (run_df_check) {
    FrameConfig no_df = cdef_frame;
    no_df.write_df_delta_q_present = false;
    const std::vector<std::uint8_t> bad =
        assemble_frame(cdef_seq, no_df, tile);
    const std::string bad_ivf = out_dir + "/cdef_no_df_bits.ivf";
    if (write_ivf(bad_ivf, source.width, source.height, bad, 30, 1, &error)) {
      std::string decode_error;
      const bool decoded = run_decoder(dav2d, bad_ivf, out_dir + "/bad.y4m",
                                       &decode_error);
      std::printf(
          "df_delta_q_present check: stream without the 4 zero bits -> %s "
          "(expected: decode failure)\n",
          decoded ? "decoded (unexpected)" : "rejected by dav2d (expected)");
      if (decoded) {
        fail("dav2d accepted a CDEF stream without df_delta_q_present bits");
      }
    }
  }

  std::printf("stream gate: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
