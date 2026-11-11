#!/usr/bin/env python3
"""Independent bit-level reference for the mlayers goldens.

This script intentionally does not include or call the C++ writer. It builds the
§5.4.1 p55-58 and §5.2.2 p54 syntax from field lists and prints the pinned byte
strings embedded in mlayers_test.cpp.

Usage: ./golden_reference.py
"""

from __future__ import annotations


class BitWriter:
    def __init__(self) -> None:
        self.bits: list[int] = []

    def bit(self, value: int | bool) -> None:
        self.bits.append(1 if value else 0)

    def bits_n(self, value: int, count: int) -> None:
        for shift in range(count - 1, -1, -1):
            self.bit((value >> shift) & 1)

    def uvlc(self, value: int) -> None:
        leading = 0
        while value >= (1 << (leading + 1)) - 1:
            leading += 1
        for _ in range(leading):
            self.bit(0)
        self.bit(1)
        payload = value - ((1 << leading) - 1)
        self.bits_n(payload, leading)

    def ns(self, n: int, value: int) -> None:
        assert n >= 2 and value < n
        w = 0
        while (1 << w) < n:
            w += 1
        m = (1 << w) - n
        if value < m:
            self.bits_n(value, w - 1)
        else:
            x = value + m
            self.bits_n(x >> 1, w - 1)
            self.bit(x & 1)

    def align(self) -> None:
        while len(self.bits) % 8:
            self.bit(0)

    def bytes(self) -> bytes:
        self.align()
        out = bytearray()
        for index in range(0, len(self.bits), 8):
            value = 0
            for bit in self.bits[index : index + 8]:
                value = (value << 1) | bit
            out.append(value)
        return bytes(out)


def ceil_log2(value: int) -> int:
    if value <= 1:
        return 0
    bits = 0
    value -= 1
    while value:
        value >>= 1
        bits += 1
    return bits


def trailing(writer: BitWriter) -> None:
    writer.bit(1)
    writer.align()


def baseline_tail(writer: BitWriter, cfg: dict) -> None:
    for _ in range(5):
        writer.bit(0)
    for _ in range(2):
        writer.bit(0)
    writer.bit(0)  # dip
    writer.bit(1)  # intra edge
    writer.bit(0)  # mrls
    writer.bit(1)  # cfl
    writer.bits_n(0, 2)  # cfl ds
    writer.bit(0)  # mhccp
    writer.bit(0)  # ibp
    for _ in range(4):
        writer.bit(0)  # motion modes
    writer.bit(0)  # masked compound
    writer.bit(0)  # ref frame mvs
    writer.bits_n(cfg["order_hint_bits"] - 1, 4)
    writer.bit(0)  # refmvbank
    writer.bit(1)  # disable drl reorder
    writer.bit(1)  # explicit ref frame map
    writer.bit(1)  # explicit num ref frames
    writer.bits_n(cfg["num_ref_frames"] - 1, 4)
    writer.bits_n(0, 3)  # long term frame id bits
    writer.ns(5, cfg["max_drl_bits_minus_1"])
    writer.bit(0)  # allow frame max drl
    writer.ns(3, 0)  # seq max bvp drl
    writer.bit(0)  # allow frame max bvp drl
    writer.bits_n(0, 2)  # num same ref compound
    writer.bit(0)  # tip
    writer.bit(0)  # mv traj
    writer.bit(0)  # bawp
    writer.bit(0)  # cwp
    writer.bit(0)  # imp msk bld
    writer.bit(0)  # df sub pu
    writer.bits_n(0, 2)  # opfl refine
    writer.bit(0)  # refinemv
    writer.bit(0)  # bru
    writer.bit(0)  # adaptive mvd
    writer.bit(0)  # mvd sign derive
    writer.bit(0)  # flex mvres
    writer.bit(0)  # global motion
    writer.bit(0)  # short refresh flags
    writer.bit(0)  # choose screen content
    writer.bit(0)  # force screen content
    writer.bit(0)  # fsc
    writer.bit(0)  # idtx
    writer.bit(0)  # intra ist
    writer.bit(0)  # inter ist
    writer.bit(1)  # chroma dctonly
    writer.bit(0)  # inter ddt
    writer.bit(0)  # reduced tx part
    writer.bit(0)  # cctx
    writer.bit(0)  # tcq
    writer.bit(0)  # parity hiding
    writer.bit(0)  # avg cdf
    writer.bit(0)  # separate uv delta q
    writer.bit(1)  # equal ac dc q
    writer.bits_n(23, 5)  # base uv ac delta q
    writer.bit(0)  # uv ac delta q enabled
    writer.bit(0)  # disable loopfilters across tiles
    writer.bit(0)  # cdef
    writer.bit(0)  # gdf
    writer.bit(0)  # restoration
    writer.bit(0)  # ccso
    writer.bit(0)  # cdef on skip always
    writer.bit(1)  # cdef on skip disabled
    writer.bits_n(0, 2)  # df par bits
    writer.bit(0)  # seq tile info
    writer.bit(0)  # film grain


def spec_layer_blocks(writer: BitWriter, layers: dict) -> None:
    max_m = layers["max_mlayer_id"]
    max_t = layers["max_tlayer_id"]
    mlayer = layers["mlayer"]
    tlayer = layers["tlayer"]
    if max_m > 0:
        writer.bit(layers["mlayer_present"])
        if layers["mlayer_present"]:
            for curr in range(1, max_m + 1):
                for ref in range(curr, -1, -1):
                    writer.bit(mlayer[curr][ref])
    if max_t > 0:
        writer.bit(layers["tlayer_present"])
        if layers["tlayer_present"]:
            if max_m > 0:
                writer.bit(layers["multi"])
            for m in range(0, max_m + 1):
                for curr in range(1, max_t + 1):
                    for ref in range(curr, -1, -1):
                        if layers["multi"] or m == 0:
                            writer.bit(tlayer[m][curr][ref])


def sequence_header(cfg: dict, dialect: str = "spec") -> bytes:
    w = cfg["width"]
    h = cfg["height"]
    layers = cfg["layers"]
    writer = BitWriter()
    writer.uvlc(0)  # seq_header_id
    writer.bits_n(0, 5)  # profile
    writer.bit(0)  # single_picture_header_flag
    writer.bits_n(cfg["seq_level_idx"], 5)
    if cfg["seq_level_idx"] > 3:
        writer.bit(0)  # seq_tier
    writer.uvlc(0)  # chroma 4:2:0
    writer.uvlc(1)  # 8-bit
    writer.bits_n(0, 3)  # seq_lcr_id
    writer.bit(0)  # still_picture
    writer.bits_n(layers["max_tlayer_id"], 2)
    writer.bits_n(layers["max_mlayer_id"], 3)
    if layers["max_mlayer_id"] > 0 and dialect == "spec":
        writer.bits_n(layers["seq_max_mlayer_cnt"] - 1,
                      ceil_log2(layers["max_mlayer_id"] + 1))
    writer.bit(cfg["monotonic"])
    width_bits = max(1, (w - 1).bit_length())
    height_bits = max(1, (h - 1).bit_length())
    writer.bits_n(width_bits - 1, 4)
    writer.bits_n(height_bits - 1, 4)
    writer.bits_n(w - 1, width_bits)
    writer.bits_n(h - 1, height_bits)
    writer.bit(0)  # crop
    writer.bit(0)  # display delay
    writer.bit(0)  # decoder model
    if dialect == "spec":
        spec_layer_blocks(writer, layers)
    else:
        # dav2d f9dafbc: tlayer rows first, n = 1..max-1, n-bit masks; then the
        # same for mlayer. Numeral count field is absent.
        if layers["max_tlayer_id"] > 0:
            writer.bit(layers["tlayer_present"])
            if layers["tlayer_present"]:
                for n in range(1, layers["max_tlayer_id"]):
                    mask = 0
                    for r in range(n):
                        if layers["tlayer"][0][n][r]:
                            mask |= 1 << r
                    writer.bits_n(mask, n)
        if layers["max_mlayer_id"] > 0:
            writer.bit(layers["mlayer_present"])
            if layers["mlayer_present"]:
                for n in range(1, layers["max_mlayer_id"]):
                    mask = 0
                    for r in range(n):
                        if layers["mlayer"][n][r]:
                            mask |= 1 << r
                    writer.bits_n(mask, n)
    baseline_tail(writer, cfg)
    writer.bit(0)  # obu_extension_flag
    trailing(writer)
    payload = writer.bytes()
    obu_header = bytes([0x04])  # ext=0, type=1, tlayer=0
    size = len(obu_header) + len(payload)
    assert size < 128
    return bytes([size]) + obu_header + payload


def default_maps(max_m: int, max_t: int) -> tuple[list[list[bool]], list]:
    mlayer = [
        [r <= c and c <= max_m for r in range(8)] for c in range(8)
    ]
    tlayer = [
        [
            [r <= c and c <= max_t and m <= max_m for r in range(4)]
            for c in range(4)
        ]
        for m in range(8)
    ]
    return mlayer, tlayer


def layers_config(max_m: int, max_t: int, *, present_m=False, present_t=False,
                  multi=False, mlayer=None, tlayer=None,
                  seq_cnt=None) -> dict:
    dm, dt = default_maps(max_m, max_t)
    return {
        "max_mlayer_id": max_m,
        "max_tlayer_id": max_t,
        "seq_max_mlayer_cnt": max_m + 1 if seq_cnt is None else seq_cnt,
        "mlayer_present": present_m,
        "tlayer_present": present_t,
        "multi": multi,
        "mlayer": dm if mlayer is None else mlayer,
        "tlayer": dt if tlayer is None else tlayer,
    }


def base(width: int, height: int, layers: dict) -> dict:
    return {
        "width": width,
        "height": height,
        "seq_level_idx": 7,
        "order_hint_bits": 4,
        "num_ref_frames": 2,
        "max_drl_bits_minus_1": 4,
        "monotonic": 1,
        "layers": layers,
    }


def obu_header_bytes(obu_type: int, tlayer: int, mlayer: int, xlayer: int) -> bytes:
    inferred_global = obu_type in (2, 20)
    inferred_mlayer = 0
    inferred_xlayer = 31 if inferred_global else 0
    extension = 1 if (mlayer != inferred_mlayer or xlayer != inferred_xlayer) else 0
    first = (extension << 7) | (obu_type << 2) | tlayer
    out = bytes([first])
    if extension:
        out += bytes([(mlayer << 5) | xlayer])
    return out


def show(name: str, data: bytes) -> None:
    print(f"{name} = {data.hex()}")


def main() -> None:
    l0 = layers_config(0, 0)
    show("kGoldenBaseline64", sequence_header(base(64, 64, l0)))

    l1 = layers_config(1, 0)
    show("kGoldenM1Inferred64", sequence_header(base(64, 64, l1)))

    l2 = layers_config(1, 0, present_m=True)
    show("kGoldenM1Explicit64", sequence_header(base(64, 64, l2)))

    l3 = layers_config(2, 2, present_m=True, present_t=True, multi=True)
    show("kGoldenM2T2Multi320", sequence_header(base(320, 180, l3)))

    l4 = layers_config(2, 2, present_m=True, present_t=True, multi=False)
    show("kGoldenM2T2Single320", sequence_header(base(320, 180, l4)))

    l5 = layers_config(1, 0)
    show("kGoldenCompatM1_64", sequence_header(base(64, 64, l5), "compat"))

    l6 = layers_config(7, 0)
    show("kGoldenM7Inferred64", sequence_header(base(64, 64, l6)))

    # OBU headers (§5.2.2 p54).
    show("kGoldenObuSeq", obu_header_bytes(1, 0, 0, 0))
    show("kGoldenObuTd", obu_header_bytes(2, 0, 0, 31))
    show("kGoldenObuClkM1", obu_header_bytes(4, 0, 1, 0))
    show("kGoldenObuClkX5", obu_header_bytes(4, 0, 0, 5))
    show("kGoldenObuMsdo", obu_header_bytes(20, 0, 0, 31))


if __name__ == "__main__":
    main()
