// Decoder-gated stream for non-DCT intra transform types, plus the all-off
// byte-identity control and a pinned identity-type stream.
//
// Three streams are produced from the same 64x64 source and the same M1 plan:
//
//   feature   enable_idtx_intra = 0, enable_chroma_dctonly = 0
//             luma   : the planner's legal set, restricted to TX_CLASS_2D
//                      (DCT_DCT, ADST_DCT, DCT_ADST, ADST_ADST, FLIPADST_*),
//                      chosen by coded-domain SSE.
//             chroma : the §5.20.7.29 derived type (UVMode = V_PRED ->
//                      ADST_DCT) with no signalled type.
//   all-off   enable_idtx_intra = 0, enable_chroma_dctonly = 1, DCT_DCT on
//             every plane.  Must be byte-identical to the pre-feature writer
//             path (the planner is bypassed for the baseline build).
//   identity  enable_idtx_intra = 1, enable_chroma_dctonly = 0
//             luma   : identity types only (IDTX where the set and EOB make it
//                      legal, DCT_DCT otherwise).  Needs the build-dir writer
//                      patch that adds the §5.20.5.5 fsc_mode symbol
//                      (patch_m1_syntax.py); without it the pinned decoder
//                      rejects every enable_idtx_intra = 1 tile.
//
// The feature and all-off streams are decoded with
// `dav2d --strict 1 --threads 1` and the decoder output must equal the
// encoder's own reconstruction sample for sample.
#include "encoder/transform/idtx_intra/idtx_intra.h"

#include "cuda/m1/plan/plan.h"
#include "encoder/intra_modes/intra_modes.h"
#include "encoder/m1/m1_headers.h"
#include "encoder/m1/m1_syntax.h"
#include "encoder/partition/partition.h"
#include "encoder/tx_variants/tx_variants.h"
#include "idtx_goldens.h"
#include "io/ivf.h"
#include "io/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

namespace idtx = av2enc::idtx;
namespace plan = av2enc::m1::plan;
using av2enc::txv::TxParams;
using av2enc::txv::TxType;

constexpr int kWidth = 64;
constexpr int kHeight = 64;
constexpr int kQp = 90;

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

struct Plane {
  int width = 0;
  int height = 0;
  int stride = 0;
  int rows = 0;
  std::vector<std::uint8_t> samples;

  std::uint8_t at(int x, int y) const {
    return samples[static_cast<std::size_t>(y * stride + x)];
  }
  void set(int x, int y, std::uint8_t value) {
    samples[static_cast<std::size_t>(y * stride + x)] = value;
  }
};

std::uint64_t fnv1a(std::uint64_t hash, const std::uint8_t* data,
                    std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

enum class StreamMode { kFeature, kAllOff, kIdentity };

struct Mode {
  StreamMode stream = StreamMode::kAllOff;
  bool baseline = false;  // bypass the planner (pre-feature writer path)
};

struct EncodeResult {
  std::vector<std::uint8_t> bytes;
  Plane y, u, v;
  int luma_types[16] = {};
  int chroma_types[16] = {};
  int type_symbols = 0;
  bool ok = false;
  std::string error;
};

struct Prediction {
  bool ok = false;
  std::vector<std::uint8_t> samples;
};

Prediction predict(const Plane& recon, int x, int y, int w, int h, int max_x,
                   int max_y, int color_plane, av2enc::intra::Mode mode) {
  Prediction out;
  av2enc::intra::NeighborRequest request;
  request.recon.data = recon.samples.data();
  request.recon.stride = recon.stride;
  request.x = x;
  request.y = y;
  request.w = w;
  request.h = h;
  request.max_x = max_x;
  request.max_y = max_y;
  request.have_above = y > 0;
  request.have_left = x > 0;
  request.color_plane = color_plane;
  av2enc::intra::PredictParams params;
  params.mode = mode;
  params.enable_edge_filter = false;
  out.samples.resize(static_cast<std::size_t>(w * h));
  out.ok = av2enc::intra::predict_block(request, 8, params,
                                        out.samples.data(),
                                        static_cast<std::size_t>(w)) ==
           av2enc::intra::IntraStatus::kOk;
  return out;
}

void write_recon(Plane* plane, int x, int y, int w, int h,
                 const std::vector<std::uint8_t>& pred,
                 const std::vector<std::int16_t>& residual) {
  for (int j = 0; j < h; ++j) {
    const int py = y + j;
    if (py < 0 || py >= plane->rows) continue;
    for (int i = 0; i < w; ++i) {
      const int px = x + i;
      if (px < 0 || px >= plane->stride) continue;
      const std::size_t local = static_cast<std::size_t>(j * w + i);
      plane->set(px, py, static_cast<std::uint8_t>(clip3(
                            0, 255, static_cast<int>(pred[local]) +
                                        static_cast<int>(residual[local]))));
    }
  }
}

// Forward transform + scalar quantisation.  `levels` is raster order; the
// caller reorders to the §5.20.7.30 diagonal scan for the writer.
void transform_plane(const std::vector<std::int16_t>& residual,
                     const TxParams& params, std::vector<std::int32_t>* levels,
                     std::vector<std::int16_t>* recon_coded) {
  const int area = params.w * params.h;
  std::vector<std::int16_t> adj(static_cast<std::size_t>(area));
  std::vector<double> coeff(static_cast<std::size_t>(area));
  levels->assign(static_cast<std::size_t>(area), 0);
  av2enc::txv::downsample_residual(residual.data(), params, adj.data());
  av2enc::txv::forward_transform(adj.data(), params, coeff.data(),
                                 levels->data());
  std::vector<std::int32_t> dequant(static_cast<std::size_t>(area));
  recon_coded->resize(
      static_cast<std::size_t>(params.coded_w * params.coded_h));
  av2enc::txv::inverse_transform(levels->data(), params, dequant.data(),
                                 recon_coded->data());
}

int eob_of(const std::vector<std::int32_t>& levels_raster, int w, int h) {
  // Diagonal scan, §5.20.7.30.
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(w * h));
  int x = 0;
  int y = 0;
  for (int c = 0; c < w * h; ++c) {
    scan[static_cast<std::size_t>(c)] =
        static_cast<std::uint16_t>(y * w + x);
    ++x;
    --y;
    if (y < 0 || x >= w) {
      ++x;
      const int s = std::min(x, h - 1 - y);
      x -= s;
      y += s;
    }
  }
  int eob = 0;
  for (int c = w * h - 1; c >= 0; --c) {
    if (levels_raster[static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])] !=
        0) {
      eob = c + 1;
      break;
    }
  }
  return eob;
}

void to_scan_order(const std::vector<std::int32_t>& levels_raster, int w, int h,
                   std::vector<std::int32_t>* out) {
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(w * h));
  int x = 0;
  int y = 0;
  for (int c = 0; c < w * h; ++c) {
    scan[static_cast<std::size_t>(c)] =
        static_cast<std::uint16_t>(y * w + x);
    ++x;
    --y;
    if (y < 0 || x >= w) {
      ++x;
      const int s = std::min(x, h - 1 - y);
      x -= s;
      y += s;
    }
  }
  out->resize(static_cast<std::size_t>(w * h));
  for (int c = 0; c < w * h; ++c) {
    (*out)[static_cast<std::size_t>(c)] =
        levels_raster[static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])];
  }
}

struct Choice {
  TxType type = TxType::kDctDct;
  std::vector<std::int32_t> levels_raster;
  std::vector<std::int16_t> recon;
  int eob = 0;
  int adj_w = 0;
  int adj_h = 0;
};

// Picks the lowest-SSE legal type for this block under the active mode.
Choice choose_luma(const std::vector<std::int16_t>& residual,
                   const idtx::Query& query, StreamMode stream) {
  Choice best;
  double best_sse = 0.0;
  bool have = false;

  idtx::TypeSet types;
  std::string error;
  if (!idtx::enumerate_types(query, &types, &error)) {
    return best;
  }
  // The identity stream uses an identity type where the set and EOB allow one
  // and falls back to DCT_DCT otherwise; the M1 writer's diagonal scan only
  // admits TX_CLASS_2D types, so IDTX is the only streamable identity type.
  bool identity_available = false;
  for (int i = 0; i < types.count; ++i) {
    if (idtx::is_identity_type(types.candidates[i].type) &&
        idtx::tx_class(types.candidates[i].type) == 0) {
      identity_available = true;
    }
  }
  for (int i = 0; i < types.count; ++i) {
    const TxType type = types.candidates[i].type;
    if (stream == StreamMode::kIdentity) {
      if (identity_available) {
        if (!idtx::is_identity_type(type)) continue;
      } else if (type != TxType::kDctDct) {
        continue;
      }
    }
    if (stream == StreamMode::kAllOff && type != TxType::kDctDct) {
      continue;
    }
    // The M1 writer codes every block with the §5.20.7.30 diagonal scan, so
    // only TX_CLASS_2D types are byte-compatible with it.
    if (idtx::tx_class(type) != 0) continue;

    const auto params = av2enc::txv::make_tx_params(query.coded_w, query.coded_h,
                                                   type);
    if (!params.has_value()) continue;
    Choice candidate;
    candidate.type = type;
    candidate.adj_w = params->w;
    candidate.adj_h = params->h;
    transform_plane(residual, *params, &candidate.levels_raster,
                    &candidate.recon);
    candidate.eob =
        eob_of(candidate.levels_raster, params->w, params->h);

    // §5.20.8.2: a non-DCT type with eob <= 1 is not signallable.
    idtx::Query check = query;
    check.eob = candidate.eob;
    idtx::TypePlan plan;
    if (!idtx::plan_type(check, type, &plan, &error)) continue;

    std::int64_t sse = 0;
    for (std::size_t k = 0; k < residual.size(); ++k) {
      const std::int64_t d =
          static_cast<std::int64_t>(candidate.recon[k]) - residual[k];
      sse += d * d;
    }
    const double cost = static_cast<double>(sse);
    if (!have || cost < best_sse ||
        (cost == best_sse && static_cast<int>(type) <
                                 static_cast<int>(best.type))) {
      have = true;
      best = std::move(candidate);
      best_sse = cost;
    }
  }
  return best;
}

EncodeResult encode(const plan::FramePlan& frame_plan, const Plane& src_y,
                    const Plane& src_u, const Plane& src_v, const Mode& mode) {
  EncodeResult result;
  result.y.width = kWidth;
  result.y.height = kHeight;
  result.y.stride = frame_plan.luma_stride;
  result.y.rows = frame_plan.luma_padded_height;
  result.y.samples.assign(
      static_cast<std::size_t>(result.y.stride * result.y.rows), 0);
  result.u.width = kWidth / 2;
  result.u.height = kHeight / 2;
  result.u.stride = frame_plan.chroma_stride;
  result.u.rows = frame_plan.chroma_padded_height;
  result.u.samples.assign(
      static_cast<std::size_t>(result.u.stride * result.u.rows), 0);
  result.v = result.u;

  idtx::SeqFlags flags;
  flags.enable_idtx_intra = mode.stream == StreamMode::kIdentity;
  flags.enable_chroma_dctonly = mode.stream != StreamMode::kFeature;

  const int uv_mode = mode.stream == StreamMode::kFeature ? 1 : 0;
  const av2enc::intra::Mode chroma_mode =
      uv_mode == 1 ? av2enc::intra::Mode::kV : av2enc::intra::Mode::kDc;

  std::vector<av2enc::m1::syntax::BlockSyntax> blocks;
  blocks.reserve(frame_plan.tree.size());
  std::vector<std::int32_t> stream_levels;

  for (const plan::TreeNode& node : frame_plan.tree) {
    av2enc::m1::syntax::BlockSyntax entry;
    entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
    entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
    entry.bsize = node.block_size;
    entry.partition = node.partition;
    entry.allowed_mask = node.allowed_mask;
    entry.subtree = static_cast<std::uint16_t>(node.subtree);
    if (node.sb_root) entry.flags |= av2enc::m1::syntax::kFlagSbRoot;
    if (node.forced) entry.flags |= av2enc::m1::syntax::kFlagForced;

    if (node.leaf_index != plan::kNoTreeNode) {
      const plan::BlockPlan& block = frame_plan.blocks[node.leaf_index];
      entry.levels_offset = static_cast<std::uint32_t>(stream_levels.size());
      entry.y_mode = 0;  // DC_PRED
      entry.y_angle_delta = 0;
      entry.uv_mode = static_cast<std::uint8_t>(uv_mode);

      // ----- luma -----
      const Prediction pred = predict(result.y, block.x, block.y, block.luma_w,
                                      block.luma_h, frame_plan.coded_luma_w - 1,
                                      frame_plan.coded_luma_h - 1, 0,
                                      av2enc::intra::Mode::kDc);
      if (!pred.ok) {
        result.error = "luma prediction failed";
        return result;
      }
      std::vector<std::int16_t> residual(
          static_cast<std::size_t>(block.luma_w * block.luma_h));
      for (int j = 0; j < block.luma_h; ++j) {
        for (int i = 0; i < block.luma_w; ++i) {
          const int src = src_y.at(block.x + i, block.y + j);
          residual[static_cast<std::size_t>(j * block.luma_w + i)] =
              static_cast<std::int16_t>(
                  src - static_cast<int>(pred.samples[static_cast<std::size_t>(
                            j * block.luma_w + i)]));
        }
      }
      idtx::Query query;
      query.coded_w = block.luma_w;
      query.coded_h = block.luma_h;
      query.plane = 0;
      query.mapped_mode = 0;
      query.y_mode = 0;
      query.eob = 2;  // set selection only; re-checked per candidate
      query.flags = flags;
      Choice luma;
      if (mode.baseline) {
        const auto params = av2enc::txv::make_tx_params(
            block.luma_w, block.luma_h, TxType::kDctDct);
        luma.type = TxType::kDctDct;
        luma.adj_w = params->w;
        luma.adj_h = params->h;
        transform_plane(residual, *params, &luma.levels_raster, &luma.recon);
        luma.eob = eob_of(luma.levels_raster, params->w, params->h);
      } else {
        luma = choose_luma(residual, query, mode.stream);
      }
      if (luma.levels_raster.empty()) {
        result.error = "no legal luma type";
        return result;
      }
      entry.tx_size[0] = static_cast<std::uint8_t>(
          av2enc::txv::tx_size_index(block.luma_w, block.luma_h));
      entry.tx_type[0] = static_cast<std::uint8_t>(luma.type);
      entry.eob[0] = static_cast<std::uint32_t>(luma.eob);
      if (!mode.baseline) {
        idtx::Query check = query;
        check.eob = luma.eob;
        idtx::TypePlan plan_out;
        if (!idtx::plan_type(check, luma.type, &plan_out, &result.error)) {
          return result;
        }
        if (plan_out.type_syntax) ++result.type_symbols;
      }
      ++result.luma_types[static_cast<int>(luma.type)];
      std::vector<std::int32_t> scan_levels;
      to_scan_order(luma.levels_raster, luma.adj_w, luma.adj_h, &scan_levels);
      stream_levels.insert(stream_levels.end(), scan_levels.begin(),
                           scan_levels.begin() + luma.eob);
      write_recon(&result.y, block.x, block.y, block.luma_w, block.luma_h,
                  pred.samples, luma.recon);

      // ----- chroma -----
      for (int plane = 1; plane <= 2; ++plane) {
        const Plane& src = plane == 1 ? src_u : src_v;
        Plane& recon_plane = plane == 1 ? result.u : result.v;
        const Prediction cpred =
            predict(recon_plane, block.chroma_x, block.chroma_y, block.chroma_w,
                    block.chroma_h, frame_plan.coded_chroma_w - 1,
                    frame_plan.coded_chroma_h - 1, plane, chroma_mode);
        if (!cpred.ok) {
          result.error = "chroma prediction failed";
          return result;
        }
        std::vector<std::int16_t> cres(
            static_cast<std::size_t>(block.chroma_w * block.chroma_h));
        for (int j = 0; j < block.chroma_h; ++j) {
          for (int i = 0; i < block.chroma_w; ++i) {
            const int srcv = src.at(block.chroma_x + i, block.chroma_y + j);
            cres[static_cast<std::size_t>(j * block.chroma_w + i)] =
                static_cast<std::int16_t>(
                    srcv - static_cast<int>(cpred.samples[static_cast<std::size_t>(
                               j * block.chroma_w + i)]));
          }
        }
        TxType ctype = TxType::kDctDct;
        if (!mode.baseline) {
          if (!idtx::derive_chroma_type(uv_mode, 0, block.chroma_w,
                                        block.chroma_h, flags, &ctype)) {
            result.error = "chroma type derivation failed";
            return result;
          }
        }
        const auto cparams = av2enc::txv::make_tx_params(block.chroma_w,
                                                         block.chroma_h, ctype);
        std::vector<std::int32_t> clevels;
        std::vector<std::int16_t> crecon;
        transform_plane(cres, *cparams, &clevels, &crecon);
        const int ceob = eob_of(clevels, cparams->w, cparams->h);
        if (!mode.baseline) {
          idtx::Query cquery;
          cquery.coded_w = block.chroma_w;
          cquery.coded_h = block.chroma_h;
          cquery.plane = plane;
          cquery.uv_mode = uv_mode;
          cquery.eob = ceob;
          cquery.flags = flags;
          idtx::TypePlan cplan;
          if (!idtx::plan_type(cquery, ctype, &cplan, &result.error)) {
            return result;
          }
        }
        ++result.chroma_types[static_cast<int>(ctype)];
        entry.tx_size[plane] = static_cast<std::uint8_t>(
            av2enc::txv::tx_size_index(block.chroma_w, block.chroma_h));
        entry.tx_type[plane] = static_cast<std::uint8_t>(ctype);
        entry.eob[plane] = static_cast<std::uint32_t>(ceob);
        std::vector<std::int32_t> cscan;
        to_scan_order(clevels, cparams->w, cparams->h, &cscan);
        stream_levels.insert(stream_levels.end(), cscan.begin(),
                             cscan.begin() + ceob);
        write_recon(&recon_plane, block.chroma_x, block.chroma_y,
                    block.chroma_w, block.chroma_h, cpred.samples, crecon);
      }
    }
    blocks.push_back(entry);
  }

  av2enc::m1::M1Config config;
  config.max_frame_width = kWidth;
  config.max_frame_height = kHeight;
  config.base_q_idx = kQp;
  config.disable_cdf_update = true;
  config.enable_intra_edge_filter = false;
  config.enable_cfl_intra = false;
  config.enable_chroma_dctonly = flags.enable_chroma_dctonly;
  config.enable_idtx_intra = flags.enable_idtx_intra;
  config.reduced_tx_set = 0;
  config.apply_deblock_y_vert = false;
  config.apply_deblock_y_horz = false;
  config.apply_deblock_u = false;
  config.apply_deblock_v = false;

  av2enc::m1::syntax::FramePlan syntax_plan;
  syntax_plan.width = kWidth;
  syntax_plan.height = kHeight;
  syntax_plan.sb_size = 64;
  std::size_t tile_bytes = 0;
  if (!av2enc::m1::syntax::assemble_m1_frame(syntax_plan, blocks, stream_levels,
                                             config, result.bytes,
                                             result.error, &tile_bytes)) {
    return result;
  }
  result.ok = true;
  return result;
}

bool read_source(const std::string& path, Plane* y, Plane* u, Plane* v,
                 std::string* error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame frame;
  if (reader.read_frame(frame, *error) != av2enc::Y4mReadResult::kFrame) {
    *error = "no frame in " + path;
    return false;
  }
  if (frame.width() != kWidth || frame.height() != kHeight) {
    *error = "gate needs a 64x64 source";
    return false;
  }
  y->width = kWidth;
  y->height = kHeight;
  y->stride = kWidth;
  y->rows = kHeight;
  y->samples.assign(frame.y().begin(), frame.y().end());
  u->width = kWidth / 2;
  u->height = kHeight / 2;
  u->stride = u->width;
  u->rows = u->height;
  u->samples.assign(frame.u().begin(), frame.u().end());
  *v = *u;
  v->samples.assign(frame.v().begin(), frame.v().end());
  return true;
}

bool read_decoded(const std::string& path, Plane* y, Plane* u, Plane* v,
                  std::string* error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame frame;
  if (reader.read_frame(frame, *error) != av2enc::Y4mReadResult::kFrame) {
    *error = "no decoded frame in " + path;
    return false;
  }
  if (frame.width() != kWidth || frame.height() != kHeight) {
    *error = "decoded frame has the wrong size";
    return false;
  }
  y->width = kWidth;
  y->height = kHeight;
  y->stride = kWidth;
  y->rows = kHeight;
  y->samples.assign(frame.y().begin(), frame.y().end());
  u->width = kWidth / 2;
  u->height = kHeight / 2;
  u->stride = u->width;
  u->rows = u->height;
  u->samples.assign(frame.u().begin(), frame.u().end());
  *v = *u;
  v->samples.assign(frame.v().begin(), frame.v().end());
  return true;
}

int max_abs_diff(const Plane& a, const Plane& b) {
  int worst = 0;
  for (int j = 0; j < a.height; ++j) {
    for (int i = 0; i < a.width; ++i) {
      const int d = std::abs(static_cast<int>(a.at(i, j)) -
                             static_cast<int>(b.at(i, j)));
      if (d > worst) worst = d;
    }
  }
  return worst;
}

bool write_ivf(const std::string& path, const std::vector<std::uint8_t>& frame,
               std::string* error) {
  av2enc::IvfHeader header;
  header.width = kWidth;
  header.height = kHeight;
  header.fps_num = 30;
  header.fps_den = 1;
  av2enc::IvfWriter writer;
  if (!writer.open(path, header, *error)) return false;
  if (!writer.write_frame(frame, 0, *error)) return false;
  return writer.close(*error);
}

std::string shell_quote(const std::string& value) {
  std::string out = "'";
  for (char c : value) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

int run_decoder(const std::string& dav2d, const std::string& stream,
                const std::string& decoded) {
  const std::string command = shell_quote(dav2d) + " --input " +
                              shell_quote(stream) + " --output " +
                              shell_quote(decoded) +
                              " --muxer yuv4mpeg2 --strict 1 --threads 1"
                              " --quiet";
  return std::system(command.c_str());
}

bool verify_stream(const std::string& dav2d, const std::string& ivf,
                   const EncodeResult& encoded, bool keep,
                   const char* label) {
  std::string error;
  if (!write_ivf(ivf, encoded.bytes, &error)) {
    std::fprintf(stderr, "%s ivf: %s\n", label, error.c_str());
    return false;
  }
  const std::string y4m = ivf + ".decoded.y4m";
  if (run_decoder(dav2d, ivf, y4m) != 0) {
    std::fprintf(stderr, "dav2d failed on the %s stream\n", label);
    return false;
  }
  Plane dy;
  Plane du;
  Plane dv;
  if (!read_decoded(y4m, &dy, &du, &dv, &error)) {
    std::fprintf(stderr, "%s decoded: %s\n", label, error.c_str());
    return false;
  }
  const int ymax = max_abs_diff(dy, encoded.y);
  const int umax = max_abs_diff(du, encoded.u);
  const int vmax = max_abs_diff(dv, encoded.v);
  std::printf("%-9s recon: max|diff| Y=%d U=%d V=%d\n", label, ymax, umax,
              vmax);
  if (!keep) {
    std::remove(ivf.c_str());
    std::remove(y4m.c_str());
  }
  return ymax == 0 && umax == 0 && vmax == 0;
}

double psnr(const Plane& decoded, const Plane& source) {
  std::int64_t sse = 0;
  long long count = 0;
  for (int y = 0; y < source.height; ++y) {
    for (int x = 0; x < source.width; ++x) {
      const std::int64_t d = static_cast<std::int64_t>(decoded.at(x, y)) -
                             static_cast<std::int64_t>(source.at(x, y));
      sse += d * d;
      ++count;
    }
  }
  if (sse == 0) return 99.0;
  return 10.0 * std::log10(255.0 * 255.0 * static_cast<double>(count) /
                           static_cast<double>(sse));
}

double frame_psnr(const EncodeResult& enc, const Plane& sy, const Plane& su,
                  const Plane& sv) {
  // 4:2:0 weighting: 4 luma samples per chroma sample per plane.
  const double py = psnr(enc.y, sy);
  const double pu = psnr(enc.u, su);
  const double pv = psnr(enc.v, sv);
  const double mse_y = std::pow(10.0, -py / 10.0) * 255.0 * 255.0;
  const double mse_u = std::pow(10.0, -pu / 10.0) * 255.0 * 255.0;
  const double mse_v = std::pow(10.0, -pv / 10.0) * 255.0 * 255.0;
  const double mse = (4.0 * mse_y + mse_u + mse_v) / 6.0;
  if (mse <= 0.0) return 99.0;
  return 10.0 * std::log10(255.0 * 255.0 / mse);
}

int count_type(const int (&types)[16], int first, int last) {
  int total = 0;
  for (int i = first; i <= last && i < 16; ++i) total += types[i];
  return total;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dav2d;
  std::string source = "tests/data/m0_64x64_1f.y4m";
  std::string prefix;
  bool keep = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--dav2d" && i + 1 < argc) {
      dav2d = argv[++i];
    } else if (arg == "--y4m" && i + 1 < argc) {
      source = argv[++i];
    } else if (arg == "--prefix" && i + 1 < argc) {
      prefix = argv[++i];
    } else if (arg == "--keep") {
      keep = true;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }
  if (dav2d.empty()) {
    std::fprintf(stderr,
                 "usage: %s --dav2d PATH [--y4m FILE] [--prefix DIR] [--keep]\n",
                 argv[0]);
    return 2;
  }

  Plane sy;
  Plane su;
  Plane sv;
  std::string error;
  if (!read_source(source, &sy, &su, &sv, &error)) {
    std::fprintf(stderr, "source: %s\n", error.c_str());
    return 1;
  }

  plan::PlanOptions options;
  options.base_q_idx = kQp;
  options.min_block_size = 8;
  options.max_partition_depth = 4;
  options.allow_rect = true;
  options.allow_split = false;
  plan::FramePlan frame_plan;
  if (!plan::build_frame_plan(kWidth, kHeight, sy.samples, options, frame_plan,
                              error)) {
    std::fprintf(stderr, "plan: %s\n", error.c_str());
    return 1;
  }
  if (!plan::validate_frame_plan(frame_plan, error)) {
    std::fprintf(stderr, "plan: %s\n", error.c_str());
    return 1;
  }

  if (std::getenv("IDTX_DUMP_PLAN") != nullptr) {
    std::fputs(plan::plan_to_tsv(frame_plan).c_str(), stdout);
  }

  Mode feature;
  feature.stream = StreamMode::kFeature;
  Mode off;
  off.stream = StreamMode::kAllOff;
  Mode off_baseline;
  off_baseline.stream = StreamMode::kAllOff;
  off_baseline.baseline = true;
  Mode identity;
  identity.stream = StreamMode::kIdentity;

  const EncodeResult enc_feature = encode(frame_plan, sy, su, sv, feature);
  if (!enc_feature.ok) {
    std::fprintf(stderr, "feature encode: %s\n", enc_feature.error.c_str());
    return 1;
  }
  const EncodeResult enc_off = encode(frame_plan, sy, su, sv, off);
  if (!enc_off.ok) {
    std::fprintf(stderr, "all-off encode: %s\n", enc_off.error.c_str());
    return 1;
  }
  const EncodeResult enc_off_baseline =
      encode(frame_plan, sy, su, sv, off_baseline);
  if (!enc_off_baseline.ok) {
    std::fprintf(stderr, "baseline encode: %s\n",
                 enc_off_baseline.error.c_str());
    return 1;
  }

  const std::uint64_t feature_hash =
      fnv1a(1469598103934665603ULL, enc_feature.bytes.data(),
            enc_feature.bytes.size());
  const std::uint64_t off_hash = fnv1a(1469598103934665603ULL,
                                       enc_off.bytes.data(),
                                       enc_off.bytes.size());

  std::printf("feature   stream: %zu bytes, fnv %016llx, type symbols %d\n",
              enc_feature.bytes.size(),
              static_cast<unsigned long long>(feature_hash),
              enc_feature.type_symbols);
  std::printf("all-off   stream: %zu bytes, fnv %016llx\n",
              enc_off.bytes.size(), static_cast<unsigned long long>(off_hash));

  bool ok = true;
  if (feature_hash != kFeatureStreamFnv) {
    std::fprintf(stderr, "feature stream FNV mismatch: pinned %016llx\n",
                 static_cast<unsigned long long>(kFeatureStreamFnv));
    ok = false;
  }
  if (off_hash != kAllOffStreamFnv) {
    std::fprintf(stderr, "all-off stream FNV mismatch: pinned %016llx\n",
                 static_cast<unsigned long long>(kAllOffStreamFnv));
    ok = false;
  }
  if (enc_off.bytes != enc_off_baseline.bytes) {
    std::fprintf(stderr,
                 "all-off byte-identity control failed: planner path differs "
                 "from the pre-feature writer path\n");
    ok = false;
  } else {
    std::printf("all-off   control: byte-identical to the DCT-only writer\n");
  }

  const int non_dct_luma =
      count_type(enc_feature.luma_types, 1, 15);
  std::printf("feature   types  : %d non-DCT luma blocks, %d chroma ADST_DCT\n",
              non_dct_luma,
              enc_feature.chroma_types[static_cast<int>(TxType::kAdstDct)]);
  if (non_dct_luma == 0) {
    std::fprintf(stderr, "feature stream signalled no non-DCT luma type\n");
    ok = false;
  }
  if (enc_feature.chroma_types[static_cast<int>(TxType::kAdstDct)] == 0) {
    std::fprintf(stderr, "feature stream derived no non-DCT chroma type\n");
    ok = false;
  }

  const std::string dir = prefix.empty() ? "." : prefix;
  ok = verify_stream(dav2d, dir + "/idtx_feature.ivf", enc_feature, keep,
                     "feature") &&
       ok;
  ok = verify_stream(dav2d, dir + "/idtx_all_off.ivf", enc_off, keep,
                     "all-off") &&
       ok;

  // Identity stream.  The build-dir writer patch adds the §5.20.5.5 fsc_mode
  // symbol that allow_fsc_intra() requires once enable_idtx_intra = 1; without
  // it the pinned dav2d rejects the tile (the docs/m1-features.md open item).
  const EncodeResult enc_identity = encode(frame_plan, sy, su, sv, identity);
  if (!enc_identity.ok) {
    std::fprintf(stderr, "identity encode: %s\n", enc_identity.error.c_str());
    return 1;
  }
  const std::uint64_t identity_hash =
      fnv1a(1469598103934665603ULL, enc_identity.bytes.data(),
            enc_identity.bytes.size());
  const int identity_blocks = count_type(enc_identity.luma_types, 9, 15);
  std::printf("identity  stream: %zu bytes, fnv %016llx, %d identity blocks\n",
              enc_identity.bytes.size(),
              static_cast<unsigned long long>(identity_hash), identity_blocks);
  if (identity_hash != kIdentityStreamFnv) {
    std::fprintf(stderr, "identity stream FNV mismatch: pinned %016llx\n",
                 static_cast<unsigned long long>(kIdentityStreamFnv));
    ok = false;
  }
  if (identity_blocks == 0) {
    std::fprintf(stderr, "identity stream signalled no identity luma type\n");
    ok = false;
  }
  ok = verify_stream(dav2d, dir + "/idtx_identity.ivf", enc_identity, keep,
                     "identity") &&
       ok;
  std::printf("quality         : feature %.3f dB (%zu B)  all-off %.3f dB "
              "(%zu B)  identity %.3f dB (%zu B)\n",
              frame_psnr(enc_feature, sy, su, sv), enc_feature.bytes.size(),
              frame_psnr(enc_off, sy, su, sv), enc_off.bytes.size(),
              frame_psnr(enc_identity, sy, su, sv), enc_identity.bytes.size());

  if (!ok) {
    std::fprintf(stderr, "IDTX GATE FAIL\n");
    return 1;
  }
  std::printf("IDTX GATE PASS\n");
  return 0;
}
