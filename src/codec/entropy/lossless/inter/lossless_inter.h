// AV2 v1.0.0 lossless inter coding (P frames) — CPU reference.
//
// This module is a separate workstream from `encoder/lossless` (the still
// picture reference).  It reuses the public reversible helpers and the
// committed CDF tables, but owns its own sequence/frame/tile syntax so the
// lossless inter paths the audit marked missing can be written and read:
//
//   * §5.20.6.1  `lossless_tx_size` for inter coding blocks
//   * §5.20.8.2  `lossless_inter_tx_type` (DCT_DCT WHT vs IDTX at TX_4X4)
//   * §5.18.2    `CodedLossless`, `LosslessArray` and per-segment `ALT_Q`
//   * §7.15.4    inter residual reconstruction (WHT / IDTX + DPCM)
//   * §5.20.7.27 FSC (`coeff_base_bob` / `coeff_base_idtx` / `coeff_br_idtx`
//                / `idtx_sign`) coefficient syntax at 4x4..32x32
//   * §6.4.2     lossless 4:2:0, 4:2:2 and 4:4:4 chroma geometries
//
// Scope: 8/10-bit, 64x64 superblocks, one tile, one key frame followed by P
// frames or a closed hierarchical-P/B GOP with up to two references per frame.
// Prediction uses zero-motion GLOBALMV (single) or GLOBAL_GLOBALMV
// COMPOUND_AVERAGE (B); there is no motion search.  The independent reader
// tracks reference slots and returns pictures in order-hint order.
//
// Normative references: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7).
#pragma once

#include "codec/entropy/lossless/lossless.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::lossless::inter {

// ---------------------------------------------------------------------------
// Lossless flag derivation (§5.18.2) for inter frames
// ---------------------------------------------------------------------------

// The §5.18.2 derivation is frame-type independent: `LosslessArray[]` and
// `CodedLossless` depend only on the quantization inputs.  This wrapper makes
// that explicit for callers that encode inter frames and exposes the derived
// per-segment qindex as well.
struct FrameQuantInputs {
  int base_q_idx = 0;
  bool delta_q_present = false;

  int delta_q_y_dc = 0;
  int base_y_dc_delta_q = 0;
  int delta_q_u_dc = 0;
  int delta_q_v_dc = 0;
  int delta_q_u_ac = 0;
  int delta_q_v_ac = 0;
  int base_uv_dc_delta_q = 0;
  int base_uv_ac_delta_q = 0;

  bool alt_q_enabled[kMaxSegments] = {};
  int alt_q[kMaxSegments] = {};
  int num_segments = kBaseSegments;
};

struct FrameLossless {
  bool coded_lossless = false;
  bool has_lossless_segment = false;
  bool lossless[kMaxSegments] = {};
  int qindex[kMaxSegments] = {};
  int num_segments = 0;
};

// `frame_is_inter` is accepted for API clarity and for tests that assert the
// derivation does not change between intra and inter frames.  The returned
// flags are exactly §5.18.2's LosslessArray/CodedLossless.
FrameLossless derive_lossless(const FrameQuantInputs& inputs,
                              bool frame_is_inter);

// ---------------------------------------------------------------------------
// Geometry / format
// ---------------------------------------------------------------------------

enum class ChromaFormat : std::uint8_t {
  k420 = 0,
  k400 = 1,  // monochrome (not encoded here)
  k444 = 2,
  k422 = 3,
};

struct Geometry {
  int width = 0;
  int height = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  int chroma_w = 0;
  int chroma_h = 0;
  int chroma_mi_cols = 0;
  int chroma_mi_rows = 0;
  int sb_cols = 0;
  int sb_rows = 0;
  int sub_x = 1;
  int sub_y = 1;
  // 0 for 4:0:0 (§6.4.2 Table 6.2: Monochrome).  Chroma planes are not coded
  // and chroma_w/chroma_h are zero.
  bool has_chroma = true;
  ChromaFormat format = ChromaFormat::k420;
};

bool make_geometry(int width, int height, ChromaFormat format, Geometry* out,
                   std::string* error);

// ---------------------------------------------------------------------------
// Reversible helpers (exposed for tests and for the FSC syntax tests)
// ---------------------------------------------------------------------------

// §7.15.2.2 inverse / exact forward WHT on four samples.
void inverse_wht4(const int in[4], int shift, int out[4]);
void forward_wht4(const int in[4], int out[4]);

// Full 2D lossless residual transform.  `dct_dct` selects the reversible 4x4
// WHT (§7.15.2.2 / §7.15.4); otherwise the IDTX path copies levels because
// Dequant >> (3 - shift) == level at q=0 (§7.14.4, §7.15.4).
void forward_lossless_block(const int* residual, int w, int h, bool dct_dct,
                            int* levels);
void inverse_lossless_block(const int* levels, int w, int h, bool dct_dct,
                            int* residual);

// DPCM first-difference / cumulative sum in the transform domain.
void dpcm_difference_block(const int* residual, int w, int h, bool vertical,
                           int* out);
void dpcm_cumsum_block(const int* in, int w, int h, bool vertical, int* out);

// ---------------------------------------------------------------------------
// §5.20.6.1 / §5.20.8.2 lossless transform-size signalling
// ---------------------------------------------------------------------------

// find_tx_size(Min(32,w), Min(32,h)) for square/rectangular power-of-two
// sizes.  Returns true and the (coded) transform dimensions.
bool find_tx_size(int coded_w, int coded_h, int* tx_w, int* tx_h);

// read_tx_size() lossless branch for one coding block.
struct TxSizeDecision {
  int coded_w = 4;
  int coded_h = 4;
  bool lossless_tx_size = false;
  bool signal_lossless_tx_size = false;
};

// `is_inter`, `fsc_mode`, `allow_select` and the coding block size are the
// inputs used by §5.20.6.1.  `want_large` chooses the value of the
// `lossless_tx_size` symbol when it is present.
TxSizeDecision select_tx_size(int block_w, int block_h, bool is_inter,
                              bool fsc_mode, bool allow_select,
                              bool want_large);

// ---------------------------------------------------------------------------
// Encoder / reader
// ---------------------------------------------------------------------------

struct Picture {
  int width = 0;
  int height = 0;
  ChromaFormat format = ChromaFormat::k420;
  // Samples are LSB-aligned at `bit_depth` (8, 10 or 12) and stored as
  // uint16_t so one reader/writer serves every depth.
  int bit_depth = 8;
  std::vector<std::uint16_t> y;
  std::vector<std::uint16_t> u;
  std::vector<std::uint16_t> v;
};

struct Options {
  // 64 only.
  int sb_size = 64;
  int seq_level_idx = 31;
  // When false (default), disable_cdf_update = 1 and both sides use the
  // default CDFs.
  bool adaptive_cdf = false;
  // Intra blocks: DPCM search (V/H DPCM vs DC).  Inter blocks have
  // use_dpcm_*=0 by syntax (§5.20.7.6).
  bool enable_dpcm = true;
  // enable_fsc in the sequence header.  When on, inter luma IDTX blocks use
  // the FSC coefficient path and intra blocks may signal fsc_mode.
  bool enable_fsc = true;
  // Coding block target size (8, 16 or 32).  Boundary blocks split down to 8.
  int block_size = 8;
  // Fraction of P-frame blocks forced to intra, in 1/N units.  0 means every
  // block is inter.  4 means every fourth block is intra so a P frame also
  // exercises intra DPCM / WHT / FSC syntax.
  int intra_period = 0;
  // For inter blocks: alternate lossless_tx_size / lossless_inter_tx_type so
  // both branches of §5.20.6.1 and §5.20.8.2 appear in one stream.
  bool alternate_inter_tx = true;
  // Debug/test coverage: force TX_4X4 inter transforms (lossless_tx_size = 0)
  // and choose the lossless_inter_tx_type value.
  bool inter_4x4_only = false;
  bool inter_4x4_idtx = true;
  // Force fsc_mode = 1 on every intra block (test coverage).
  bool force_intra_fsc = false;
  // Verify every reconstructed sample before returning success.
  bool verify_exact = true;
  // Sample depth of the pictures and the signalled bit_depth_idc
  // (8 = bit_depth_idc 1, 10 = bit_depth_idc 0).  The lossless reconstruction
  // clips to [0, (1 << bit_depth) - 1].
  int bit_depth = 8;
  // Hierarchical-P temporal-layer GOP size (4/8/16).  0 keeps the historical
  // flat key+P chain.  The schedule is the one --video-tlayers uses: the
  // tlayers plan's coding order, one single reference per frame, implicit
  // output pictures and per-OBU obu_tlayer_id.
  int temporal_layers = 0;
  // When true (with temporal_layers = 4 or 8) use the tlayers hierarchical-B
  // schedule instead of hierarchical-P.  Interior frames carry two references
  // (one past, one future) and every inter block codes a two-sided
  // COMPOUND_AVERAGE prediction (GLOBAL_GLOBALMV).  Frame 7 of an 8-frame GOP
  // is the B anchor with a single reference.
  bool b_schedule = false;
};

struct FrameStats {
  int inter_blocks = 0;
  int intra_blocks = 0;
  int lossless_tx_size_1 = 0;
  int lossless_tx_size_0 = 0;
  int idtx_4x4 = 0;
  int dct_dct_4x4 = 0;
  int fsc_luma_blocks = 0;
  int dpcm_y_blocks = 0;
  int dpcm_uv_blocks = 0;
  std::uint64_t max_abs_recon_error = 0;
  bool exact = false;
};

struct EncodeResult {
  // One Annex B temporal unit per frame.  Unit 0 has TD + sequence header +
  // CLK; later units have TD + regular tile group.
  std::vector<std::vector<std::uint8_t>> units;
  std::vector<Picture> recon;
  std::vector<FrameStats> stats;
  std::uint64_t max_abs_recon_error = 0;
  bool exact = false;
};

bool encode_video(std::span<const Picture> pictures, const Options& options,
                  EncodeResult* result, std::string* error);

struct DecodeResult {
  std::vector<Picture> frames;
  // Per-frame block statistics.  The reader has no source reference, so
  // FrameStats::exact / max_abs_recon_error are not meaningful here; callers
  // compare `frames` against the source or the encoder's recon.
  std::vector<FrameStats> stats;
};

bool decode_video(std::span<const std::vector<std::uint8_t>> units,
                  DecodeResult* result, std::string* error);

// ---------------------------------------------------------------------------
// FSC coefficient syntax (§5.20.7.27) at arbitrary lossless sizes
// ---------------------------------------------------------------------------

// A transform block's coefficient coding context.  The writer and reader both
// derive the same context from the neighbouring level/dc grids; tests use this
// struct directly to exercise sizes 4x4..32x32 without a full frame.
struct CoeffContext {
  int plane = 0;
  bool is_inter = false;
  bool fsc_mode = false;
  int coded_w = 4;
  int coded_h = 4;
  int x4 = 0;
  int y4 = 0;
  int bsize_w = 4;
  int bsize_h = 4;
  int mi_cols = 0;
  int mi_rows = 0;
  int eob_u = 0;
  // Neighbour grids, indexed by x4/y4.  A full frame owns one of each per
  // plane; the syntax test allocates zeroed buffers.
  const std::uint8_t* above_level = nullptr;
  const std::uint8_t* left_level = nullptr;
  const std::uint8_t* above_dc = nullptr;
  const std::uint8_t* left_dc = nullptr;
};

// Round-trip one coefficient block through the §5.20.7.27 FSC path.  The
// writer and reader are independent (separate loops / separate CDF state).
// `levels` is raster order in the coded transform block.  Returns true on a
// byte-exact round trip; `decoded` receives the reader's levels.
bool fsc_coeff_roundtrip(const int* levels, const CoeffContext& context,
                         int* decoded, std::string* error);

// ---------------------------------------------------------------------------
// Pinned goldens
// ---------------------------------------------------------------------------

struct GoldenCase {
  const char* name;
  int width;
  int height;
  ChromaFormat format;
  int block_size;
  int intra_period;
  bool enable_fsc;
  // MD5 of the concatenated Annex B temporal units (frames in order).
  const char* md5;
};

std::span<const GoldenCase> golden_cases();

}  // namespace av2enc::lossless::inter
