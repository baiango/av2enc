#pragma once

// §7.13.3.19/.20 warped prediction adapter for the M2 product paths.
//
// Both the CPU reference (`video_frame.cpp`) and the CPU-emulation backend
// (`video_backend_cpu.cpp`) call this, so the GLOBALMV warped predictor is the
// exact `src/codec/inter/advanced/warp.cpp` transcription on both sides.  The
// CUDA backend uses the shared `src/cuda/codec/inter/warp_stack/` device
// kernels, which compile the same header.

#include <cstdint>
#include <string>
#include <vector>

#include "codec/inter/gme/gme.h"
#include "codec/inter/mc/mc.h"

namespace av2enc::video {

// Predicts one plane into a tightly packed `w`x`h` buffer.  The block origin is
// in plane samples; `mi_col`/`mi_row`/`bw4`/`bh4` are the §7.12.2.2 luma MI
// geometry.  Chooses block warp vs extended warp exactly like
// `adv::warp_needs_extended()`.
bool warp_predict_plane(const inter::gme::GmParams& params,
                        const inter::mc::RefFrame& reference, int plane, int x,
                        int y, int w, int h, int mi_row, int mi_col, int bw4,
                        int bh4, std::vector<std::uint8_t>& out,
                        std::string& error);

// Luma + 4:2:0 chroma warped prediction at the block origin (x, y).
bool warp_predict_420(const inter::gme::GmParams& params,
                      const inter::mc::RefFrame& reference, int x, int y, int w,
                      int h, int mi_row, int mi_col, int bw4, int bh4,
                      std::vector<std::uint8_t>& y_out,
                      std::vector<std::uint8_t>& u_out,
                      std::vector<std::uint8_t>& v_out, std::string& error);

}  // namespace av2enc::video
