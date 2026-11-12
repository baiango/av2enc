#ifndef AV2ENC_INTER_WARP_STACK_WARP_STACK_IO_H_
#define AV2ENC_INTER_WARP_STACK_WARP_STACK_IO_H_

// Canonical text snapshot of a WarpStack.
//
// The writer (`warp_stack_io.cpp`) and the reader (`warp_stack_reader.cpp`) are
// deliberately separate translation units that share only this declaration.
// The reader is a hand-written tokenizer: it does not reuse the writer, the
// builder or any `operator<<`, so writing a snapshot, parsing it and comparing
// it is an independent check that the stack is fully described by its public
// fields (a missing field would round-trip as a dropped value).
//
// Format (all integers decimal, whitespace separated):
//
//   WARPSTACK 1
//   COUNT <count> NEAREST <num_nearest> WARP <num_warp_found> SORTED <0|1>
//   C <mv0y> <mv0x> <mv1y> <mv1x> <weight> <row_off> <col_off> <cwp>
//   ... one C record per candidate, in stack order ...
//   W <p0> <p1> <p2> <p3> <p4> <p5>
//   ... one W record per warp candidate, in warp-stack order ...
//   END
//
// `warp_idx` addresses the W records, so their order is part of the bitstream
// contract; the parser preserves it exactly.

#include <string>

#include "encoder/inter/warp_stack/warp_stack.h"

namespace av2enc::inter::warp {

std::string write_snapshot(const WarpStack& stack);

bool read_snapshot(const std::string& text, WarpStack* stack, std::string* error);

}  // namespace av2enc::inter::warp

#endif  // AV2ENC_INTER_WARP_STACK_WARP_STACK_IO_H_
