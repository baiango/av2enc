// POD layout shared by the CPU coefficient planner, the syntax writer and the
// device coefficient writer.
//
// Deliberately std-free so `inter_residual_device.cuh` can include it in device
// compilation without pulling <vector>/<string> in.
#pragma once

#include <cstdint>

namespace av2enc::m2::residual {

// One coded coefficient in scan order, with everything needed to emit it.
struct CoeffPlanEntry {
  std::uint8_t base_level = 0;  // coeff_base / coeff_base_eob symbol
  std::uint8_t br_level = 0;    // coeff_br symbol (0 when not coded)
  std::uint8_t sign = 0;        // 1 = negative
  std::uint8_t is_eob = 0;      // c == eob - 1
  std::uint8_t is_lf = 0;       // low-frequency (get_lf_limits)
  std::uint8_t codes_br = 0;
  std::uint8_t codes_dc_sign = 0;  // luma DC: CDF sign instead of a raw bit
  std::uint8_t codes_escape = 0;   // |level| >= maxLevel => read_quant escape
  std::uint8_t reserved = 0;
  std::int16_t base_ctx = 0;  // CDF row for the base symbol
  std::int16_t br_ctx = 0;    // CDF row for the br symbol
  std::int16_t eob_ctx = 0;   // CDF row for coeff_base_(lf_)eob
  std::int16_t dc_sign_ctx = 0;
  std::int32_t pos = 0;       // raster position in the adjusted transform
  std::int32_t abs_level = 0;
  std::int32_t pre_level = 0;  // base + br, capped at the coded maximum
  std::int32_t escape_delta = 0;  // |level| - pre_level when escaping
};

}  // namespace av2enc::m2::residual
