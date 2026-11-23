#!/usr/bin/env bash
# Pin + Stage-0 dump oracle for the superblock wavefront
# (docs/wavefront-plan.md §9.3/§9.4).
#
#   tools/refactor/check_wavefront_pins.sh --bin build-t/av2enc [--threads LIST]
#
# Encodes every pinned recipe at each thread count in LIST (default "1 8"),
# checks the IVF bytes and md5 against the pin, checks the AV2ENC_BLOCKDUMP md5
# against the Stage-0 record, and `cmp`s both across thread counts.  The dump is
# the oracle that catches an availability-order bug even when the stream
# happens to match (docs/wavefront-plan.md §5 D2/D3).
#
# Exit: 0 pass, 1 mismatch, 2 usage/missing input.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BIN=""
THREADS="1 8"
while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        --threads) THREADS=${2:-}; shift 2 ;;
        # Accepted for tools/run_gates.sh compatibility; this check never decodes.
        --dav2d) shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 2 ;;
    esac
done
BIN=${BIN:-"$ROOT/build-t/av2enc"}
TD=${AV2ENC_TESTDATA:-"$ROOT/testdata"}
NASA="$TD/nasa_480x240_30fps_60f.y4m"
SPACE="$TD/space8k_1440x720_60f.y4m"
for fixture in nasa_480x240_30fps_60f.y4m space8k_1440x720_60f.y4m; do
    [ -f "$TD/$fixture" ] || {
        # Gates exit 77 when prerequisites are absent so the suite runs on a
        # tree without the gitignored testdata.
        echo "check_wavefront_pins: missing fixture $TD/$fixture (skipping)" >&2
        exit 77
    }
done

TMP=$(mktemp -d "${TMPDIR:-/tmp}/wf-pins.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

# name | fixture | frames | encode args | bytes | md5 | dump md5
CASES=(
"default|$NASA|60|--video --qp 189|25009|a1d0dba1259b20a4b628d4cd1e14c33c|ab38bfb606c46f109f2102303fafa9e0"
"quality|$NASA|60|--video --qp 190 --video-rd-preset quality|25141|3984677b9b665952984b07b7cf75c6ea|2558da60a0918b0205bf32f9da086e5a"
"nightly|$NASA|60|--video --qp 110 --nightly|23690|1b99127a4cd1d386fa5600fb63c57a83|d96b3a5368411d8a960d821271a49b52"
"explambda|$NASA|60|--video --qp 110 --nightly --video-qp-lambda --video-tx-lambda 136 --video-key-tx-lambda 136 --video-leaf-rd-lambda 125 --video-rdoq-scale 0.1 --video-partition-rd-lambda 2.75e-6|24113|7f03eb6e82d6ba4cf6666c78285f945b|3deaf2520bf2b137e8336437e073ea31"
"720p|$SPACE|60|--video --qp 217 --nightly --video-key-qp-offset -57 --video-leaf-rd-lambda 250 --video-ext-partitions|36572|c4ba28138d8f6569d5e08721a66d755c|-"
)

MD5=md5
command -v md5 >/dev/null || MD5="md5sum"
md5_of() { $MD5 -q "$1" 2>/dev/null || $MD5 "$1" | awk '{print $1}'; }

fail=0
for spec in "${CASES[@]}"; do
    IFS='|' read -r name fixture frames args want_bytes want_md5 want_dump <<< "$spec"
    dumps=(); ivfs=()
    for t in $THREADS; do
        out="$TMP/$name-$t.ivf"; dump="$TMP/$name-$t.dump"
        AV2ENC_BLOCKDUMP="$dump" "$BIN" "$fixture" -o "$out" --frames "$frames" \
            $args --cpu-threads "$t" >"$TMP/$name-$t.log" 2>&1 || {
            echo "FAIL $name t=$t encode"; tail -5 "$TMP/$name-$t.log"; fail=1; continue; }
        b=$(wc -c <"$out" | tr -d ' '); m=$(md5_of "$out")
        dm="-"; [ -f "$dump" ] && dm=$(md5_of "$dump")
        echo "  $name t=$t ${b} B $m dump=$dm"
        [ "$b" = "$want_bytes" ] || { echo "FAIL $name bytes $b != $want_bytes"; fail=1; }
        [ "$m" = "$want_md5" ] || { echo "FAIL $name md5 $m != $want_md5"; fail=1; }
        if [ "$want_dump" != "-" ] && [ "$dm" != "$want_dump" ]; then
            echo "FAIL $name dump $dm != $want_dump"; fail=1; fi
        dumps+=("$dump"); ivfs+=("$out")
    done
    for i in $(seq 1 $((${#ivfs[@]} - 1))); do
        cmp -s "${ivfs[0]}" "${ivfs[$i]}" || { echo "FAIL $name ivf differs t=$i"; fail=1; }
        cmp -s "${dumps[0]}" "${dumps[$i]}" || { echo "FAIL $name dump differs t=$i"; fail=1; }
    done
done
# Trace output is diagnostic but gates parse it (gate_video_edge_mask counts
# leaves from --video-trace).  Superblocks must not emit trace lines
# concurrently: the wavefront interleaved std::cerr insertions and the count
# flapped between 53 and 56 on the same binary.  Trace-enabled encodes therefore
# keep the serial superblock walk, and the text must match across thread counts.
if [ -f "$NASA" ]; then
    LAST_T=$(echo "$THREADS" | awk '{print $NF}')
    for t in 1 "$LAST_T"; do
        "$BIN" "$NASA" -o "$TMP/trace-$t.ivf" --video --frames 8 --qp 190 \
            --cpu-threads "$t" --video-rd-preset quality --video-edge-inter \
            --video-edge-mask --video-trace \
            >"$TMP/trace-$t.log" 2>"$TMP/trace-$t.txt" || {
            echo "FAIL trace encode t=$t"; fail=1; }
    done
    # Compare against the *last* thread count in the list, not a hard-coded 8:
    # with --threads "1 10" the file is trace-10.txt.
    if cmp -s "$TMP/trace-1.txt" "$TMP/trace-$LAST_T.txt"; then
        echo "  trace t=1 == t=$LAST_T ($(wc -l <"$TMP/trace-1.txt" | tr -d ' ') lines)"
    else
        echo "FAIL --video-trace output differs across thread counts"; fail=1
    fi
fi

[ $fail -eq 0 ] && echo "check_wavefront_pins: OK"
exit $fail
