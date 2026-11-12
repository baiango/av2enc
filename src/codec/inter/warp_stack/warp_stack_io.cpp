#include "codec/inter/warp_stack/warp_stack_io.h"

#include <sstream>

namespace av2enc::inter::warp {

std::string write_snapshot(const WarpStack& stack) {
  std::ostringstream out;
  out << "WARPSTACK 1\n";
  out << "COUNT " << stack.count << " NEAREST " << stack.num_nearest << " WARP "
      << stack.num_warp_found << " SORTED " << (stack.sorted ? 1 : 0) << "\n";
  for (int i = 0; i < stack.count; ++i) {
    const WarpCandidate& c = stack.cand[i];
    out << "C " << c.mv[0].y << ' ' << c.mv[0].x << ' ' << c.mv[1].y << ' '
        << c.mv[1].x << ' ' << c.weight << ' ' << c.row_offset << ' '
        << c.col_offset << ' ' << static_cast<int>(c.cwp) << "\n";
  }
  for (int i = 0; i < stack.num_warp_found; ++i) {
    out << "W";
    for (int j = 0; j < 6; ++j) out << ' ' << stack.warp_params[i].p[j];
    out << "\n";
  }
  out << "END\n";
  return out.str();
}

}  // namespace av2enc::inter::warp
