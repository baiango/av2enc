// Usage/help text and the option table.  Moved verbatim from src/main.cpp so
// the printed help stays byte-identical.
#include "cli/options.h"

#include <initializer_list>
#include <ostream>
#include <string>
#include <string_view>

void print_usage(std::ostream& out) {
  constexpr std::size_t kColumn = 30;
  const auto item = [&out](std::string_view name,
                           std::initializer_list<std::string_view> lines) {
    out << "  " << name;
    if (name.size() < kColumn) {
      out << std::string(kColumn - name.size(), ' ');
    } else {
      out << "\n" << std::string(kColumn + 2, ' ');
    }
    bool first = true;
    for (const std::string_view text : lines) {
      if (!first) {
        out << std::string(kColumn + 2, ' ');
      }
      out << text << "\n";
      first = false;
    }
  };

  out << "Usage: av2enc <input.y4m> -o <output.ivf> [options]\n"
         "       av2enc <input.y4m> --gpu-plan <N> [--tiles <layout>]\n"
         "\n"
         "Modes (default: Basic):\n"
         "  Basic       One key frame: 8-bit 4:2:0, 64x64 superblocks, 32x32\n"
         "              DCT, DC intra only, no filters.\n"
         "  Still       --still: partition tree, full intra modes, variable\n"
         "              transform sizes, CfL and deblock. The CPU reference.\n"
         "  Still GPU   --still-gpu: the same still syntax with the block\n"
         "              pipeline on the GPU. Requires a CUDA build and implies\n"
         "              --still; Still tools without a GPU stage are rejected\n"
         "              by name instead of being ignored.\n"
         "  Video       --video: multi-frame P-frame sequence.\n"
         "  Lossless    --lossless: lossless still picture with its own writer\n"
         "              (DPCM intra, 8x8 blocks, Q=0).\n"
         "Still modes encode exactly one frame; Video encodes --frames frames.\n"
         "Every mode writes an AV2 IVF (fourcc AV02). Usage and option-\n"
         "combination errors exit 2; encoder and I/O failures exit 1.\n"
         "\n"
         "Basic options:\n";
  item("-o, --output <file>", {"Output IVF path (required unless --gpu-plan)"});
  item("--frames <N>", {"Frames to encode; still modes accept only 1"});
  item("--qp <Q>", {"base_q_idx (default 90). Basic pins 90; the CPU Still",
                    "path accepts 1..255 (1..303 for 10-bit); Q=0 requires",
                    "--lossless"});
  item("--adaptive-cdf", {"Enable normative 8.2.6 CDF adaptation"});
  item("--avg-cdf", {"Multi-tile still/video: seed each tile from the frame",
                     "CDFs and average at frame end (§8.2.4/§7.5)"});
  item("--cdf-continuity", {"Video: carry the adapted CDFs across frames by",
                            "loading the primary reference's saved context",
                            "(§5.18.2/§7.4.4/§7.5). Requires --avg-cdf",
                            "--adaptive-cdf; CPU-only, single tile. The first",
                            "inter frame of a key+P chain still resets",
                            "(§5.18.2 derived PRIMARY_REF_NONE)"});
  item("--video-cdf-per-frame-qp", {"Video: allow --avg-cdf --adaptive-cdf",
                                    "with a per-frame QP plan; every frame",
                                    "resets the §8.2.2 CDF image from its own",
                                    "coefficient qctx slice (§8.3.1), so all",
                                    "frames must share one slice. CPU only"});
  item("--vram-cap <MiB>", {"CUDA arena upper bound; 0 takes the largest",
                            "reservation the device can serve (default 0).",
                            "CPU-only builds accept it as a documented no-op"});
  item("-h, --help", {"Show this help"});

  out << "\nStill options (--still, --still-gpu):\n";
  item("--no-cfl", {"Disable chroma-from-luma"});
  item("--no-deblock", {"Disable the deblocking filter"});
  item("--still-deblock", {"Opt in to §7.17 deblocking on the format "
                           "(--bit-depth/--chroma) paths (off by default so "
                           "pinned streams are unchanged)"});
  item("--still-full-intra", {"Directional/smooth intra search (SSE-only mode",
                              "decision; DC is the default)"});
  item("--still-uv-full", {"Full chroma intra search (DC/CfL is the default;",
                           "CPU-only)"});
  item("--still-rdo-intra", {"Rate-aware luma intra mode decision (CPU-only)"});
  item("--still-rdoq", {"Full-trellis coefficient RDOQ (CPU-only)"});
  item("--still-tx-type", {"Intra luma TX_TYPE search (CPU-only)"});
  item("--still-cdef", {"CDEF, after deblock (CPU; runs on --still-gpu; the "
                        "8-bit chroma format path is CPU-only)"});
  item("--still-ccso", {"Band-offset-only CCSO, after CDEF (CPU; runs on",
                       "--still-gpu)"});
  item("--still-ccso-edge", {"Edge-classified CCSO + per-unit skip (CPU; runs",
                             "on --still-gpu)"});
  item("--still-wiener", {"Luma Wiener NS loop restoration (CPU; runs on",
                          "--still-gpu)"});
  item("--still-chroma-lr", {"Chroma Wiener NS loop restoration (CPU; runs on",
                             "--still-gpu)"});
  item("--still-pc-wiener", {"Luma PC-Wiener loop restoration (CPU; runs on",
                             "--still-gpu)"});
  item("--still-gdf", {"Guided detail filter after loop restoration (CPU;",
                       "runs on --still-gpu)"});
  item("--still-grain", {"Signal film grain for decoder synthesis (CPU; runs",
                         "on --still-gpu)"});
  item("--still-grain-adaptive", {"Estimate and signal a film-grain model from",
                                  "the reconstruction; requires --still-grain",
                                  "(CPU-only)"});
  item("--still-cfl-ds <N>", {"cfl_ds_filter_index 0..3 (default 0; runs on",
                              "--still-gpu)"});
  item("--still-cfl-explicit", {"Allow explicit CfL alpha, CFL_EXPLICIT",
                                "(CPU; runs on --still-gpu)"});
  item("--still-ibp", {"Intra bi-prediction, 5.4.5 (CPU; runs on",
                       "--still-gpu)"});
  item("--still-mrl", {"Multiple reference lines for directional luma;",
                       "implies the directional search (CPU; runs on",
                       "--still-gpu)"});
  item("--still-dip", {"DIP as a luma DC alternative (CPU; runs on",
                       "--still-gpu)"});
  item("--still-mhccp", {"MHCCP (CFL_MULTI) chroma prediction (CPU; runs",
                         "on --still-gpu)"});
  item("--still-segmentation", {"Signal a neutral 8-segment table (runs on",
                                "--still-gpu)"});
  item("--still-delta-q", {"Signal neutral per-superblock delta-Q (runs",
                            "on --still-gpu)"});
  item("--still-per-segment-q", {"Real per-SB segmentation with ALT_Q; low-",
                                 "variance SBs use base_q_idx + delta (CPU;",
                                 "runs on --still-gpu)"});
  item("--still-sb128", {"128x128 superblocks (default 64x64; CPU; runs",
                          "on --still-gpu)"});
  item("--still-sb256", {"Signal 256x256 sequence superblocks; intra frames",
                         "still code 128x128 (CPU; runs on --still-gpu)"});
  item("--still-ext-partitions", {"HORZ_3/VERT_3 and uneven 4-way partitions",
                                  "(CPU; runs on --still-gpu)"});
  item("--still-sub8", {"Allow sub-8x8 coding blocks (4xN/Nx4); needs a depth",
                        "of at least 8 (CPU; runs on --still-gpu)"});
  item("--still-sdp", {"Semi-decoupled partition: luma tree then chroma tree",
                       "(forced PARTITION_NONE trees; 64-SB only; CPU; runs on",
                       "--still-gpu)"});
  item("--still-qm", {"Spec-defined quantizer matrix (CPU; runs on",
                      "--still-gpu)"});
  item("--still-user-qm", {"Carry a real user QM OBU (identity matrix; the",
                           "pinned dav2d rejects QM OBUs; CPU-only)"});
  item("--still-ist", {"Intra secondary transform, luma (CPU-only)"});
  item("--still-tcq", {"Trellis-coded quantisation, luma 2-D transforms",
                       "(parity hiding is inferred off in a single picture)"});
  item("--still-cctx", {"Cross-component chroma transform (cctx_type",
                        "rotation)"});
  item("--still-tx-partition", {"TX_MODE_SELECT luma transform partitions",
                                "(NONE/SPLIT/HORZ/VERT)"});
  item("--still-idtx", {"Identity intra transform (enable_idtx_intra +",
                        "fsc_mode)"});
  item("--still-ddt", {"Rejected: DDT/DDTX/FDDT is inter-only"});
  item("--still-dc-ac-delta-q", {"DC/AC delta-Q frame search (CPU; runs on",
                                 "--still-gpu)"});
  item("--still-content-interpretation",
       {"Emit a default content interpretation OBU (CPU-only)"});
  item("--metadata-timecode", {"Emit a neutral metadata_timecode OBU (suffix;",
                               "ignored by the pinned decoder)"});
  item("--metadata-scan-type", {"Emit a neutral metadata_scan_type OBU (suffix;",
                                "ignored by the pinned decoder)"});
  item("--metadata-cll [max:fall]", {"Emit a §6.16.5 content light level OBU",
                                     "(suffix; default 0:0 = not signalled)"});
  item("--metadata-mdcv [preset|vals]", {"Emit a §6.16.6 mastering display",
                                         "colour volume OBU (suffix; default",
                                         "unknown, or bt2020 / 10 integers)"});
  item("--metadata-icc [file]", {"Emit a §6.16.9 ICC profile OBU (suffix;",
                                 "default 132-byte v4.3 header-only profile)"});
  item("--metadata-temporal-point [t]", {"Emit a §6.16.11 temporal point info",
                                         "short OBU (suffix; default 0 ticks)"});
  item("--metadata-frame-hash [mode]", {"Emit a §6.16.13 MD5 decoded frame hash",
                                        "of the reconstruction (suffix;",
                                        "'per-plane' or one frame digest)"});
  item("--metadata-banding [kind]", {"Emit a neutral §6.16.8 banding hints OBU",
                                     "(suffix; default neither coding nor",
                                     "source banding, or 'source')"});
  item("--metadata-user-data [uuid[:hex]]", {"Emit a §6.16.12 user data",
                                              "unregistered OBU (suffix; default",
                                              "zero UUID, empty payload)"});
  item("--video-decoder-model [dbd:ebd:low]", {"Signal the §5.4.13 nested",
                                               "seq_decoder_model_info block",
                                               "(implies --video-timing)"});
  item("--video-brt [tick]", {"Insert a §6.11 buffer removal timing OBU before",
                              "each frame (br_time 0..511, default 0)"});
  item("--still-palette", {"Luma palette intra (screen content; sets",
                           "allow_screen_content_tools; CPU-only)"});
  item("--still-ibc", {"Intra block copy (screen content; exact copies,",
                       "skip_flag=1, force_integer_mv; CPU-only; needs a frame",
                       "at least 129 luma samples wide to find a source)"});
  item("--still-ibc-search <N>", {"IBC local BV search step in luma samples,",
                                  "1..64 (default 8)"});
  item("--tiles <N|CxR>", {"Uniform tile grid, default 1x1: a count",
                           "2/4/8/16/32 or a grid 2x1/2x2/4x2/8x2/16x2.",
                           "Requires --no-deblock; 8-bit 4:2:0 only"});
  item("--cpu-threads <N>", {"CPU --video reference worker count: 1 (default,",
                             "serial), 0 = auto, 2..256. Independent tile",
                             "encoders and the batched NEWMV search run in",
                             "parallel; the bitstream is byte-identical"});
  item("--bit-depth <8|10>", {"Sample depth (default 8). 10 requires --still, a",
                              "C420p10/C422p10/C444p10/Cmono10 input and",
                              "base_q_idx 1..303; 12-bit is not signalable in",
                              "AV2 v1.0.0 (CPU-only)"});
  item("--chroma <fmt>", {"420 (default), 422, 444 or 400. The input must",
                          "already have that exact format; 8-bit pins --qp 90,",
                          "10-bit takes base_q_idx 1..303 (CPU-only)"});
  item("--lossless", {"Q=0 lossless still picture with its own writer;",
                      "cannot be combined with --still"});

  out << "\nEffort presets (--still only; opt-in, default off):\n";
  item("--effort <N>", {"Ranked-feature budget, 10..10000 (log scale); enables",
                        "measured features in score order and never exceeds",
                        "N. Implies --still. Default off"});
  item("--cpu-used <0..8>", {"x264-style sugar for --effort: 8 -> 10,",
                             "0 -> 10000, geometric in between"});
  item("--effort-plan", {"Print the effort selection and exit without",
                         "encoding"});

  out << "\nVideo options (--video):\n";
  item("--video", {"Encode a multi-frame P-frame sequence"});
  item("--video-gpu", {"Opt-in CUDA backend for --video (implies --video);",
                       "same decisions/bytes as the CPU reference, no silent",
                       "CPU fallback. CUDA build only"});
  item("--video-timing", {"Signal decoder_model_info and a CI OBU with",
                          "timing_info() from the Y4M frame rate"});
  item("--video-mfh", {"Emit a multi-frame header (§5.7) and reference it",
                       "from every frame header"});
  item("--video-tier high", {"Signal §5.4.1 seq_tier = 1 (high tier);",
                             "requires level 4.0+, i.e. seq_level_idx > 3"});
  item("--video-multi-sequence", {"Carry a second §5.4.1 sequence header",
                                  "(seq_header_id 1) and select it with",
                                  "seq_header_id_in_frame_header from the key",
                                  "frame onward"});
  item("--video-lcr-full", {"Activate and emit a full §5.8 global/local LCR",
                            "and §5.10/§5.11 global/local OPS before the key",
                            "frame (--video; default: no LCR/OPS)"});
  item("--p-frames <N>", {"Number of P frames (default: all frames after the",
                          "keyframe)"});
  item("--b-frames <N>", {"B frames per closed IPBB GOP, 0..6 (default 0)"});
  item("--video-gops <N>", {"Consecutive closed B-frame GOPs (default 1);",
                            "each GOP starts with a CLK slot reset"});
  item("--video-tlayers <N>", {"Hierarchical P GOP with N = 4 or 8 temporal",
                              "layers; coding order differs from display order"});
  item("--video-tlayer-qp-offsets <a,b,c,d>", {"Per-temporal-layer base_q_idx",
                              "offsets for layers 0..3 on the chained",
                              "--video-tlayers --video-gops form (default off)"});
  item("--video-tlayer-anchor-residual", {"Force the inter residual on the",
                              "chained layer-0 anchors (drift refresh probe)",
                              "on the chained --video-tlayers --video-gops form"});
  item("--video-tlayer-refresh-threshold <dB>", {"Selective anchor refresh:",
                              "force only when the anchor's primary reference",
                              "is that far below the key (0 = force all)"});
  item("--video-tlayer-refresh-budget <B>", {"Extra-payload budget for the",
                              "selective chained anchor refresh (default 3250)"});
  item("--video-rd-driver", {"Sequential key-frame RD driver: measures every",
                              "candidate key layout in the reconstruction it",
                              "builds. Key picture only; --nightly-b sets it"});
  item("--video-eff-perc", {"Exact-rate / BA-shaped perceptual pipe on the",
                              "inter leaves (AV2ENC_EFF_PERC); --nightly-b sets it"});
  item("--video-tail-tx", {"Exact-rate TX refinement on the display-order",
                              "tail pictures; needs --video-eff-perc"});
  item("--video-head-tx", {"Exact-rate TX refinement on the display-order",
                              "head pictures; needs --video-eff-perc"});
  item("--video-tlayer-drop <L>", {"Encode only temporal layers <= L (dropped-",
                                  "layer substream); implies --video-tlayers"});
  item("--video-compound", {"Compound GLOBAL_GLOBALMV/AVERAGE prediction;",
                            "requires --video-refs >= 2"});
  item("--video-compound-wedge", {"Search §7.13.3.27 wedge masks for compound;",
                                  "requires --video-compound"});
  item("--video-compound-diffwtd", {"Search §7.13.3.28 difference-weight masks;",
                                     "requires --video-compound"});
  item("--video-compound-interintra", {"Single-reference §7.13.3.29/.30",
                                        "inter-intra: codes the §5.20.7.14",
                                        "motion mode/interintra_mode/",
                                        "wedge_interintra and blends the inter",
                                        "prediction with an intra prediction",
                                        "(CPU, --video-refs 1)"});
  item("--video-refs <N>", {"References per P frame (NumTotalRefs), 1..7",
                            "(default 1); 2..7 selects per block. Ignored by",
                            "--video-implicit-ref-map, which derives it"});
  item("--video-num-ref-frames <N>", {"§5.4.6 NumRefFrames, 4..8 (default: 2",
                                      "for one reference, 8 for multi-ref)",
                                      "and 8 for --video-implicit-ref-map"});
  item("--video-subpel-eighth", {"Add the 1/8-pel ME refinement stage"});
  item("--video-mv-precision P", {"eighth (default), quarter or half frame MV",
                                  "precision"});
  item("--video-adaptive-mvd", {"Opt-in §5.4.6 enable_adaptive_mvd: NEWMV",
                                 "blocks may code the §5.20.7.20/.21",
                                 "mv_joint/amvd_index MVD form"});
  item("--video-mvd-sign-derive", {"Opt-in §5.4.6 enable_mvd_sign_derive:",
                                    "eligible NEWMV blocks derive the last",
                                    "MVD sign (§5.20.7.13); needs a frame",
                                    "precision of quarter or half pel"});
  item("--video-flex-mvres", {"Opt-in §5.4.6 enable_flex_mvres: each NEWMV",
                               "block may select a coarser MV precision than",
                               "the frame value (§5.20.7.6)"});
  item("--video-inter-residual", {"Code a quantised inter residual"});
  item("--video-edge-inter", {"Opt-in bottom-edge inter: a leaf whose coded",
                                "extent runs past MiRows*4 is reconstructed",
                                "into a padded plane and may use inter",
                                "prediction on its visible rows. CPU only"});
  item("--video-edge-mask", {"Opt-in bottom-edge residual mask: a leaf whose",
                              "coded height crosses the visible bottom gets",
                              "the prediction as its source view below the",
                              "edge (zero residual, no levels for samples",
                              "the decoder discards). Needs --video-edge-inter",
                              "and changes no syntax; CPU only"});
  item("--video-leaf-rd", {"Opt-in leaf RD cost: intra/inter decided by one",
                             "cost with a QP-derived lambda and real-ish",
                             "intra mode/MRL/CfL and inter mode/MV bits"});
  item("--fast-decode", {"Latest CPU-only algorithmic RD mode: the measured",
                       "quality preset plus 64-band/refit/priced-unit CCSO,",
                       "bottom-edge inter and mask, the measured partition",
                       "probe, the reference-intro key/first-P QP offset, luma",
                       "inter RDOQ, CDEF, the BAWP carrier over inter residual,",
                       "and the calibrated leaf RD cost (--video-leaf-rd).",
                       "The §8.3.2 TX context rate is not part of the aggregate.",
                       "--video-gpu is rejected; the default and GPU paths keep",
                       "the historical CPU decisions and byte-identical streams.",
                       "--nightly (the pre-2026-11-20 name) is an exact alias."});
  item("--nightly-b", {"Measured chained single-key hierarchical-B aggregate:",
                       "the reordering-legal tool set plus the chained-form",
                       "bottom-edge path, key-QP offset -50, BAWP/carrier,",
                       "AVERAGE compound on the interior B pictures, TIP",
                       "output on the dead-end leaves (mode 1, 0.30 dB",
                       "coded-leaf floor), the CWP NEAR_NEARMV depth arm",
                       "(first stack pair per list, field guard 3) and the",
                       "exact compound-rate probe (comp_mode/is_joint/",
                       "compound_mode/jmvd/DRLs/comp_group_idx/cwp_idx/"
                       "filter priced in the leaf RD), the partition probe,",
                       "the layer-0 anchor residual, the key RD driver, the",
                       "eff-perc pipe, the tail exact-rate TX, the key TX/UV",
                       "RD gates, the band-gated TX descent and the joint",
                       "plan solve (one objective over partition +",
                       "TX/levels; K 87.570, +0.286 over the exact-rate arm).",
                       "K 87.570 at 100 kbps (+1.338 over the flat 86.232).",
                       "Pass --video-gops M > 1 (M 8-frame blocks; a 4-frame",
                       "tail is allowed) and the matching --frames. CPU only.",
                       "Research opt-in (docs/tlayers-chain.md): decoder-exact",
                       "under tools/validate_avmdec.py while the pinned dav2d",
                       "has the §7.21.6 drain defect; not the default.",
                       "The joint solve is a research gate: ~12 min per",
                       "60-frame 240p encode.  AV2ENC_RD_JOINT_P=0 restores",
                       "the 87.284 exact-rate arm.",
                       "The compound NEWMV modes, wedge/diffwtd, OPFL and",
                       "refine-MV stay off (below the +0.15 gate).",
                       "AV2ENC_COMPOUND_EXACT_RATE=0 restores the 87.023",
                       "depth arm; --video-compound-depth-off the 86.805 TIP",
                       "arm; --video-compound-off the 85.472 arm;",
                       "AV2ENC_TIP_LEAF=0 with the depth off the 86.607",
                       "compound arm"});

  item("--video-rd-preset <name>", {"Expand a measured CPU RD preset:",
                                     "'quality' = --video-inter-residual",
                                     "--video-deblock --video-ccso",
                                     "--video-tx-search",
                                     "--video-intra-full",
                                     "(default off; the flags stay",
                                     "individually available).  The TX",
                                     "context rate is a separate opt-in;",
                                     "'best' = the flat --fast-decode",
                                     "aggregate plus the measured lead",
                                     "stack (eff-perc, RD_UVFULL/UVLEVEL,",
                                     "RD_KEY_TX, the tail band descent,",
                                     "RD_DRIVER, head/tail exact-rate TX),",
                                     "--video-key-qp-offset -39 and the",
                                     "joint P plan solve (K 87.19, +0.68",
                                     "bracket over the base).  The joint",
                                     "solve is a RESEARCH gate: its serial",
                                     "raster walk costs ~12 min per",
                                     "60-frame 240p encode, so --fast-decode",
                                     "stays the shipped fast path.  Set",
                                     "AV2ENC_RD_JOINT_P=0 to get the 86.33",
                                     "lead stack"});
  item("--video-tx-search", {"Opt-in §5.18.8 TX_MODE_SELECT luma TX size/type/",
                                "partition search (changes the bitstream)"});
  item("--video-key-intra-txpart", {"Opt-in keyframe intra luma TX",
                                     "partition search with per-part",
                                     "prediction and reconstruction feedback",
                                     "(needs --video-tx-search; CPU-only",
                                     "8-bit 4:2:0; changes the bitstream)"});
  item("--video-tx-context-rate", {"Price TX-search candidates with the",
                                    "§8.3.2 coefficient-context model (the",
                                    "measured preset behavior; CPU-only, needs",
                                    "--video-tx-search)"});
  item("--video-tx-lambda <S>", {"SSE-per-bit weight for --video-tx-search",
                                  "(default 256)"});
  item("--video-qp-lambda", {"Opt-in QP-scaled RD lambdas: the leaf, TX,",
                             "key-TX, RDOQ-scale and partition-fallback",
                             "weights follow per-site power laws measured at",
                             "both operating points instead of constants",
                             "(CPU; an explicit <S> flag stays exact)"});
  item("--video-partition-rd", {"Opt-in measured §5.20.3.1 partition tree:",
                                "price each P-frame leaf with one NEWMV",
                                "prediction plus a quantised residual",
                                "(CPU-only; changes the bitstream)"});
  item("--video-partition-rd-lambda <S>", {"SSE-per-bit scale for the",
                                "partition probe (lambda = S*step^2,",
                                "default 1.1e-5)"});
  item("--video-partition-rd-depth <N>", {"Probe recursion bound: 1 prices the",
                                "superblock against its first split level",
                                "(default), 2..3 probe deeper"});
  item("--video-sdp", {"Opt-in §5.4.3 enable_sdp: the intra keyframe codes",
                       "per-64x64 luma and chroma partition trees",
                       "(CPU-only, 8-bit 4:2:0, single tile)"});
  item("--video-ddt", {"Opt-in §5.4.8 enable_inter_ddt: inter ADST/FDST",
                       "transforms use the DDTX/FDDT kernels selected per",
                       "block (CPU-only, 8-bit 4:2:0)"});
  item("--video-deblock", {"Deblock every reconstructed frame (§7.17) before",
                           "it is emitted or stored as a reference; CPU and",
                           "CUDA paths signal §5.18.7.9 and filter pixels"});
  item("--video-segmentation", {"Opt-in §5.18.7.1 segmentation: code the",
                                "§5.20.5.7/.8 segment id per leaf from a",
                                "per-superblock luma-variance split (ALT_Q 0)"});
  item("--video-delta-q", {"Opt-in §5.18.7.8 delta-Q: per-superblock",
                           "CurrentQIndex updates move the block quantizer"});
  item("--video-delta-q-res <N>", {"§5.18.7.8 delta_q_res 0..3 (default 0)"});
  item("--video-aq", {"Opt-in per-superblock adaptive quantisation: an",
                      "activity-driven §5.18.7.8 delta-Q plan (textured",
                      "superblocks get a lower qindex, flat ones a higher",
                      "one); implies delta_q_present"});
  item("--video-aq-strength <S>", {"Adaptive-Q strength: multiplier on the",
                                   "normalised activity deviation in",
                                   "reduced-delta steps (default 1.0)"});
  item("--video-aq-clamp <N>", {"Maximum |adaptive-Q offset| in reduced-delta",
                                "steps, 1..8 (default 3)"});
  item("--video-aq-bias <N>", {"Constant offset added to every adaptive-Q",
                               "superblock in reduced-delta steps, -8..8",
                               "(default: centre the plan on its",
                               "activity-weighted mean, i.e. rate-neutral)"});
  item("--video-per-segment-q", {"Opt-in real SEG_LVL_ALT_Q: segment 1 (low",
                                 "variance) uses base + --video-seg-delta in",
                                 "the block quantizer and deblock metadata"});
  item("--video-seg-delta <D>", {"Per-segment qindex delta for",
                                 "--video-per-segment-q and --video-delta-q",
                                 "(default 8)"});
  item("--video-uv-delta-q <D>", {"Opt-in §5.4.8/§5.18.6.1 separate UV/DC",
                                  "delta-Q: U uses +D and V uses -D for both DC",
                                  "and AC (separate_uv_delta_q=1,",
                                  "equal_ac_dc_q=0, diff_uv_delta=1)"});
  item("--video-seg-skip", {"Opt-in SEG_LVL_SKIP on the lowest-variance",
                            "segment (implies --video-segmentation)"});
  item("--video-seg-globalmv", {"Opt-in SEG_LVL_GLOBALMV on a segment",
                                "(implies --video-segmentation)"});
  item("--video-seg-temporal", {"Opt-in segmentation_temporal_update on",
                                "P frames whose primary reference is a P frame"});
  item("--video-segments <N>", {"Number of segments, 1..16 (default 2 when",
                                "segmentation is enabled); N > 8 signals",
                                "enable_ext_seg = 1"});
  item("--video-gme", {"Estimate and signal §5.18.9 global motion per",
                       "reference; GLOBALMV uses the projected §7.12.2.2 MV.",
                       "Default is translation-only: a fitted zoom/rotation/",
                       "affine model is flattened to its translation",
                       "projection (the warp mirror is not conformance-",
                       "proven; AV2ENC_GME_UNPROVEN_WARP=1 opts in)"});
  item("--video-tip", {"Encode 3..8 frames as key + future anchor P +",
                       "TIP_FRAME_AS_OUTPUT pictures (§7.10, §7.13.3.2);",
                       "hole fill (§7.10.3/.4) on; opt-in, CPU reference"}),
      item("--video-tip-block-ref",
           {"§5.20.7.10 TipFrameMode =", "TIP_FRAME_AS_REF: the TIP-referencing tail",
            "P codes RefFrame[0] == TIP_FRAME blocks", "(needs --video-ref-mvs",
            "--video-force-single)"}),
  item("--video-tip-no-hole-fill", {"Gate control: signal",
                                    "enable_tip_hole_fill = 0 on the TIP",
                                    "sequence (pre-fix A/B stream)"}),
  item("--video-warp", {"Alias for --video-gme (the per-block warp flag is",
                        "derived from the model, not signalled).  The fitted",
                        "warp model is still flattened to translation-only",
                        "unless AV2ENC_GME_UNPROVEN_WARP=1 opts into the",
                        "conformance-unproven §7.13.3.19/.20 predictor"});
  item("--video-bawp", {"Opt-in §7.13.3.25 block adaptive weighted prediction:",
                        "eligible inter blocks signal use_bawp and adapt the",
                        "luma prediction from the current/previous template",
                        "(fitted form, no chroma BAWP)"});
  item("--video-bawp-carrier", {"Opt-in CPU-only BAWP carrier (implies",
                                "--video-bawp): a GLOBALMV winner is",
                                "re-expressed as NEARMV/NEWMV at the same MV so",
                                "§5.20.7.6 can code use_bawp, and",
                                "--video-inter-residual is allowed on the",
                                "fitted predictor. Single tile, plain P chain"}),
  item("--video-morph", {"Opt-in §7.13.3.26 morphological prediction: the",
                         "intra keyframe codes screen-content + IBC mode info",
                         "and morph_pred, and the IBC copy is adapted with the",
                         "§7.13.3.25 BAWP template fit (CPU, single tile)"});
  item("--video-palette", {"Opt-in §5.20.8 luma palette on the intra keyframe:",
                           "signals the per-frame screen-content form and codes",
                           "real palette_mode_info()/tokens for selected blocks",
                           "(CPU, single tile)"});
  item("--video-ibc", {"Opt-in §5.20.5.3/.4 general intra block copy on the",
                       "intra keyframe: use_intrabc + BV stack/DRL search",
                       "(CPU, single tile)"});
  item("--video-scc-force", {"Write the §5.4.7 forced screen-content form",
                             "(seq_force_screen_content_tools = 1) instead of",
                             "the per-frame SELECT form: palette/IBC are",
                             "inferred on for every frame. Requires one of",
                             "--video-palette/--video-ibc/--video-morph"});
  item("--video-force-integer-mv", {"With --video-scc-force: write",
                                    "seq_force_integer_mv = 1 so every frame",
                                    "derives FrameMvPrecision = one pel; the",
                                    "inter MVs are quantised to the integer",
                                    "grid and no frame codes the precision",
                                    "pair"});
  item("--video-reduce-pb-aspect", {"Write §5.4.3 reduce_pb_aspect_ratio = 1",
                                    "with MaxPbAspectRatio = 4 (use",
                                    "--video-pb-aspect-2 for 2:1) and run the",
                                    "partition search under that limit"});
  item("--video-pb-aspect-2", {"With --video-reduce-pb-aspect: select",
                               "max_pb_aspect_ratio_log2_minus_1 = 0",
                               "(MaxPbAspectRatio = 2) instead of 4"});
  item("--video-seq-seg-info", {"Write §5.4.4 seq_seg_info_present_flag = 1:",
                                "the sequence header carries the segment",
                                "table (seq_allow_seg_info_change = 1) and",
                                "each frame codes reuse_seg_info or its own",
                                "seg_info(). Requires --video-segmentation"});
  item("--video-seq-seg-info-frozen", {"With --video-seq-seg-info: write",
                                       "seq_allow_seg_info_change = 0 so every",
                                       "frame reuses the sequence table with",
                                       "no reuse_seg_info bit"});
  item("--video-tile-group-compact", {"Write §5.4.2",
                                      "seq_tile_info_present_flag = 1 with",
                                      "allow_tile_info_change = 0: the",
                                      "sequence header carries the one-tile",
                                      "tile_params() and every frame infers",
                                      "reuse_tile_info = 1 (§5.18.7.2), so the",
                                      "per-frame tile_info() costs 0 bits"});
  item("--video-ref-mvs-stress", {"Gate stress mode for --video-ref-mvs:",
                                  "always prefer a temporal NEARMV candidate",
                                  "when one exists (proves the coded stream",
                                  "selects the §7.12.2.7-.8 entries)"});
  item("--video-bru", {"Rejected: the pinned dav2d does not parse",
                       "enable_bru/use_bru (Error parsing frame header);",
                       "see src/pipeline/video/video_bru_reject_gate.sh"});
  item("--video-ref-mvs", {"Opt-in §5.4.6 enable_ref_frame_mvs: store each",
                           "reference's §7.22 motion field and add the §7.9 /",
                           "§7.12.2.7-.8 temporal MV candidates (P refs >= 2,",
                           "B frames); off keeps the historical bitstream"});
  item("--video-short-refresh", {"Opt-in §5.4.6 enable_short_refresh_frame_flags:",
                                  "every P frame codes has_refresh_frame_flags",
                                  "and one frame_to_refresh index instead of the",
                                  "full refresh mask"});
  item("--video-ref-signal-compact", {"Opt-in §5.18.2 compact reference signalling:",
                                      "NumRefFrames = 1 and",
                                      "explicit_ref_frame_map = 0, so a",
                                      "single-reference P chain codes no",
                                      "ref-map/primary-ref bits at all"});
  item("--video-refmvbank", {"Opt-in §5.4.6 enable_refmvbank: insert the",
                             "§7.12.2.21 reference-MV bank candidates at",
                             "step 19 (flat P chain, explicit ref map)"});
  item("--video-mv-traj", {"Opt-in §5.4.6 enable_mv_traj: run the §7.9.2",
                           "trajectory fill and use TrajMv in the derived",
                           "candidates; requires --video-ref-mvs"});
  item("--video-same-ref-compound", {"Opt-in §5.4.6 num_same_ref_compound = 1:",
                                     "compound blocks may use the same",
                                     "reference in both lists; requires",
                                     "--video-compound and two references"});
  item("--video-compound-near-near", {"Opt-in §5.20.7.6 NEAR_NEARMV compound",
                                      "on a two-reference B frame (shifted MVs",
                                      "from the per-list merge stacks); needs",
                                      "--video-compound and --b-frames (or the chained --video-tlayers B form)"});
  item("--video-cwp-weighted", {"Search non-equal §5.20.7.6 CWP weights",
                                 "(enable_cwp + cwp_idx) for NEAR_NEARMV",
                                 "compound blocks; needs",
                                 "--video-compound-near-near and --b-frames (or the chained --video-tlayers B form)"});
  item("--video-imp-mask", {"Enable §5.4.6 enable_imp_msk_bld and apply the",
                             "derived §7.13.3.14 get_mask() blend on eligible",
                             "NEAR_NEARMV compound blocks; needs",
                             "--video-compound-near-near and --b-frames (or the chained --video-tlayers B form)"});
  item("--video-df-sub-pu", {"Code the §5.18.2 allow_df_sub_pu bit and use",
                              "§7.13.3.15 sub-PU edges in §7.17 deblocking;",
                              "CPU --video only, off by default"});
  item("--video-opfl", {"CPU §7.13.3.8-.12 optical-flow refinement on a",
                        "NEAR_NEARMV compound block (codes use_optflow);",
                        "needs --video-compound-near-near and --b-frames",
                        "(or the chained --video-tlayers B form)"});
  item("--video-refine-mv", {"CPU §7.13.3.3-.6 refine-MV on a NEAR_NEARMV",
                             "compound block (derived, no bit: §5.20.7.17",
                             "makes it non-switchable); needs",
                             "--video-compound-near-near and --b-frames (or the chained --video-tlayers B form)"});
  item("--video-cdef", {"Enable §7.18 CDEF on the reconstructed frame before",
                        "it becomes the output/reference (single strength set)"});
  item("--video-cdef-rd", {"Price the §5.18.7.10 params row in the per-frame",
                           "CDEF decision: a frame whose filter cannot pay for",
                           "its own header (zero or negative measured gain)",
                           "codes the one-bit cdef_frame_enable = 0 form and",
                           "keeps the unfiltered reconstruction"});
  item("--video-cdef-rd-lambda <L>", {"SSE-per-bit price for --video-cdef-rd",
                                      "(default 0: only the frames whose gain",
                                      "is non-positive, which are bit-identical)"});
  item("--video-cdef-strength <N>", {"CdefStrengths 1..8 (default 1: no tile",
                                     "symbol); 2..8 codes a per-64x64 index"});
  item("--video-cdef-damping <N>", {"CdefDamping 3..6 (default 3)"});
  item("--video-cdef-search <mode>", {"CDEF strength-search budget: 'full'",
                                      "(default: 10 sets on every 8x8 block),",
                                      "'ladder' (4 sets), 'step' (10 sets on a",
                                      "2x2-decimated block grid), 'fast'",
                                      "(both).  'step'/'fast' need",
                                      "CdefStrengths 1; the filter and the",
                                      "syntax are identical in every mode"});
  item("--video-ccso", {"Enable §7.19 CCSO after CDEF (band-offset only)"});
  item("--video-ccso-bands <N>", {"CcsoMaxBandLog2 0..6 (default 4)"});
  item("--video-ccso-rd", {"Rate-aware CCSO: price the tu(7) offset table,",
                           "sweep the free scale_idx and prune ccso_blk units"});
  item("--video-ccso-lambda <S>", {"SSE-per-bit weight for --video-ccso-rd",
                                   "(default 64; implies --video-ccso-rd)"});
  item("--video-ccso-refit <N>",
       {"Extra exact table refit + unit re-prune passes (default 0)"});
  item("--video-ccso-unit-bits <S>",
       {"Price of one ccso_blk unit symbol in the prune decision (default 1)"});
  item("--video-ccso-header-rd",
       {"Price the whole ccso_params() row (flags + tu(7) table) and choose",
        "--video-ccso-bands with the fit (implies --video-ccso-rd)"});
  item("--video-wiener", {"Enable §7.20 luma non-separable Wiener restoration"});
  item("--video-pc-wiener", {"Enable §7.20.4 luma PC-Wiener restoration",
                             "(RESTORE_PC_WIENER per-unit on/off)"});
  item("--video-chroma-lr", {"Enable §7.20 chroma loop restoration (U/V)"});
  item("--video-gdf", {"Enable §7.20.5 GDF (luma, header-only mode)"});
  item("--video-cfl-ds-filter <N>", {"§5.4.5 cfl_ds_filter_index 0..3 for the",
                                     "chroma-LR CfL taps (default 0)"});
  item("--video-intra-full", {"Opt-in 61-candidate luma intra search (DC, V, H,",
                              "8 directional modes with angle deltas, SMOOTH,",
                              "SMOOTH_V/H, PAETH); chroma stays DC. Default off",
                              "so the shipped streams stay byte-identical"});
  item("--video-cfl", {"Opt-in §7.13.5 chroma-from-luma (CfL) decision on intra",
                       "leaves: {DC, CfL} with derived alpha. Default off so the",
                       "shipped streams stay byte-identical"});
  item("--video-mhccp", {"Opt-in §7.13.6-.8 multi-hypothesis CfL (MHCCP): signals",
                         "enable_mhccp and searches the three CFL_MULTI directions.",
                         "Implies --video-cfl; default off"});
  item("--video-qm", {"Opt-in §5.18.6.2 using_qmatrix with the default §9.4",
                      "matrices (level 0, no user QM OBU)"});
  item("--video-user-qm", {"Opt-in §5.13 user-defined QM: a type-22",
                           "OBU_QUANTIZATION_MATRIX with an identity all-32",
                           "matrix for level 0 before the key frame (implies",
                           "--video-qm).  Pinned dav2d rejects QM OBUs; gate",
                           "with avmdec"});
  item("--video-tcq", {"Opt-in §5.4.8/§5.20.7.27 trellis-coded quantisation:",
                       "enable_tcq = 1 with choose_tcq_per_frame = 1, a per-frame",
                       "allow_tcq bit, luma TX_CLASS_2D trellis levels and the",
                       "§7.14.4 extra dequant shift.  Default off"});
  item("--video-cctx", {"Opt-in §5.4.8/§7.14.6 cross-component transform:",
                        "enable_cctx = 1, per-block cctx_type on plane 1 and",
                        "the joint U/V dequant rotation.  Default off"});
  item("--video-rdoq", {"Opt-in coefficient-level RDOQ for luma inter",
                        "residuals: the full trellis replaces the scalar",
                        "quantiser and the TX plan search is priced by the",
                        "same static AV2 §8.3 inter CDF model.  Default off"});
  item("--video-rdoq-scale", {"RDOQ SSE-domain lambda scale: lambda =",
                              "scale * (1/quant_factor)^2.  Implies",
                              "--video-rdoq.  Default 0.1"});
  item("--video-grain", {"Opt-in §5.14 film grain signalling: sequence header",
                         "film_grain_params_present = 1, a film grain OBU per",
                         "changed model, and §5.18.10.1 per-frame config",
                         "(CPU, 8-bit 4:2:0)"});
  item("--video-lossless", {"Opt-in Q=0 lossless P-frame sequence (implies",
                            "--video): its own sequence/frame headers,",
                            "CodedLossless quantization, DPCM, reversible WHT",
                            "and FSC coefficient syntax (CPU, 8/10-bit",
                            "4:0:0/4:2:0/4:2:2/4:4:4)"});
  item("--video-lossless-b", {"--video-lossless with the hierarchical-B",
                              "schedule: two references and two-sided",
                              "COMPOUND_AVERAGE prediction on interior",
                              "frames (needs --video-tlayers 4 or 8)"});
  item("--video-lossless-intra-period <N>", {"Force one intra block per N in",
                                             "every P/B frame (0 = all inter)"});
  item("--video-idtx", {"Opt-in §5.4.8 enable_idtx_intra with the 2-D",
                        "identity TX type (9) in the --video-tx-search",
                        "candidate set"});
  item("--video-mrl", {"Opt-in §5.4.5 enable_mrls: directional luma intra",
                       "searches mrl_index 1..3 (and the secondary line) with",
                       "the §7.13.2.1 extra reference rows/columns.  Needs",
                       "--video-intra-full for a non-zero mrl_index; CPU only"});
  item("--video-ibp", {"Opt-in §5.4.5 enable_ibp: the intra keyframe and every",
                       "intra leaf use §7.13.2 intra bi-prediction (modified",
                       "edge filtering, the two-angle blend and the",
                       "§7.13.2.12 DC post-filter).  There is no per-block",
                       "IBP bit; the decoder derives the decision from mode",
                       "and angle_delta.  CPU only; default off"});
  item("--video-dip", {"Opt-in §5.4.5 enable_dip: luma DC_PRED intra",
                       "keyframe blocks code §5.20.7.9 use_dip /",
                       "dip_transpose / dip_mode and use §7.13.2.3 data-driven",
                       "prediction.  CPU only, 8-bit 4:2:0"});
  item("--video-ist", {"Opt-in §5.4.8 enable_intra_ist: luma intra residuals",
                       "code sec_tx_type and the §7.15.3 secondary transform",
                       "(inter IST stays unsignalled).  CPU only"});
  item("--video-idtx-1d", {"Rejected: the one-axis V/H_DCT..V/H_FLIPADST",
                           "types need dc_sign_horz_vert and the",
                           "class-specific scans, and pinned dav2d rejects",
                           "negative H_DCT/V_DCT under --strict 1"});
  item("--video-long-term", {"Signal long-term reference ids; tag the key frame",
                             "as LongTermId 0 (at most 16 coded frames)"});
  item("--video-olk", {"Open random access GOP (CLK, open-loop key, P2, P3;",
                       "needs exactly 4 input frames)"});
  item("--video-leading <N>", {"N open-GOP leading pictures, 1..7 (needs at",
                               "least N+2 input frames)"});
  item("--video-sef", {"Re-show the last picture with a show-existing-frame",
                       "OBU (the last picture is coded hidden)"});
  item("--video-switch", {"CLK + OBU_SWITCH (exactly 2 input frames)"});
  item("--video-ras", {"CLK + OBU_SWITCH + OBU_RAS_FRAME with a long-term",
                       "reference (exactly 3 input frames)"});
  item("--video-implicit-ref-map", {"§7.7/§7.8 implicit reference map:",
                                    "sequence explicit_ref_frame_map = 0,",
                                    "NumTotalRefs/ref_frame_idx derived from",
                                    "the slot state (flat P, B frames,",
                                    "--video-tlayers)"});
  item("--video-drl-reorder", {"§5.4.6 DRL_REORDER_CONSTRAINT: enable the",
                               "§7.12.2.19 MV-stack reorder (see",
                               "--video-drl-reorder-always)"});
  item("--video-drl-reorder-always", {"DRL_REORDER_ALWAYS; implies the reorder",
                                      "flag and sorts every block with at",
                                      "least two nearest candidates"});
  item("--rc <mode>", {"Per-frame rate control: cqp|crf|vbr (default cqp, the",
                       "fixed --qp stream)"});
  item("--crf <V>", {"--rc crf quality 0..63 (default 32); owns base_q_idx",
                     "(drop --qp, which only anchors the VBR pass)"});
  item("--bitrate <kbps>", {"--rc vbr target bitrate; VBR runs a statistics",
                            "pass at --qp, then plans one base_q_idx per frame"});
  item("--video-key-qp-offset <N>", {"Opt-in reference-intro QP offset on the",
                            "fixed --qp stream (--video, cqp only): the key",
                            "frame gets --qp+N, and a later frame identical to",
                            "its source predecessor inherits the previous QP",
                            "(so a duplicated opening pair shares one QP).",
                            "N = 0 keeps the fixed-QP stream"});
  item("--video-drift-refresh", {"Opt-in per-frame drift refresh on the flat",
                            "key+P chain (CPU, single tile, cqp): measure the",
                            "primary reference's reconstruction PSNR and code",
                            "a frame as a refresh (frame QP - N, residual on",
                            "every inter leaf) when it drops below the",
                            "threshold, within one per-sequence byte budget"});
  item("--video-drift-refresh-threshold <dB>", {"Allowed drift below the key",
                            "picture's own luma PSNR; an inter frame whose",
                            "reference is worse refreshes (default 2.5)"});
  item("--video-drift-refresh-qp <N>", {"base_q_idx reduction on a refresh frame",
                            "(default 0 = residual only)"});
  item("--video-drift-refresh-budget <bytes>", {"Per-sequence refresh payload",
                            "allowance (default 1200)"});
  item("--video-tile-flush-min", {"Opt-in §8.2.4 minimum tile termination",
                            "(CPU video writer): stop the tile payload right",
                            "after the trailing 1 instead of writing its",
                            "fourteen zeros and the byte padding. Saves",
                            "1-2 bytes per coded frame at the same",
                            "reconstruction"});
  // --video-tile-flush-min block end.
  item("--svc-layers <spec>", {"SVC: spatial layers as x0=WxH[,x1=WxH...]"});
  item("--svc-lcr-id <N>", {"SVC: seq_lcr_id (1..7)"});
  item("--svc-temporal <N>", {"SVC: temporal layers 1..4; N > 1 codes one",
                              "hierarchical GOP of 2^N frames"});
  item("--svc-embedded <N>", {"SVC: embedded quality layers 1..3; layer m",
                              "predicts from m-1 of the same spatial layer"});
  item("--svc-inter-layer", {"Rejected: §6.8.2 requires",
                             "lcr_dependent_xlayers_flag = 0"});
  item("--oppoint <x,y>", {"SVC: emit/select one operating point"});
  item("--alllayers", {"SVC: emit all operating points"});
  item("--spatial-layers <N>", {"SVC: limit the emitted spatial layers"});

  out << "\nDebug options (test-only; not encoding-quality options):\n";
  item("--still-single", {"Force one coding block per superblock"});
  item("--still-dc", {"Force DC-only intra"});
  item("--still-depth <D>", {"Partition depth cap [0, 8]"});
  item("--still-rdo-scale S", {"lambda sse_scale for --still-rdo-intra (0.125)"});
  item("--still-rdoq-scale S", {"lambda sse_scale for --still-rdoq (0.125)"});
  item("--still-tx-type-scale S", {"lambda sse_scale for --still-tx-type (0.125)"});
  item("--still-ist-scale S", {"lambda sse_scale for --still-ist (0.125)"});
  item("--still-tcq-scale S", {"lambda sse_scale for --still-tcq (0.125)"});
  item("--still-cdef-strengths N", {"CDEF strength sets 1..8 (default 4);",
                                    "requires --still-cdef"});
  item("--still-cdef-on-skip-txfm", {"Code the ADAPTIVE frame bit for CDEF on",
                                     "skip-txfm; requires --still-cdef"});
  item("--still-ccso-bands N", {"CCSO max_band_log2 0..6 (default 6 = 64 bands);",
                                "requires --still-ccso"});
  item("--still-delta-q-res N", {"delta_q_res 0..3 (default 0); requires",
                                 "--still-delta-q"});
  item("--still-seg-delta N", {"ALT_Q delta for --still-per-segment-q, 0..64",
                               "(default 8)"});
  item("--still-qm-level L", {"QM default matrix level 0..14 (default 3);",
                              "requires --still-qm"});
  item("--lossless-fsc", {"Force the forward-skip-coding coefficient path"});
  item("--gpu-plan <N>", {"Print the host-only multi-GPU tile plan for N GPUs",
                          "(1..8) and exit: assignment, transfer and VRAM",
                          "estimates; no device memory, no stream. Reads the",
                          "input and --tiles/--vram-cap only"});
  item("--recon <file>", {"Dump the encoder reconstruction as Y4M (format paths",
                          "and multi-tile --tiles stills; the decoder gates",
                          "compare against it)"});
  item("--video-recon <file>", {"Video: write the encoder reconstruction Y4M"});
  item("--video-force-single", {"Video: one 64x64 leaf per superblock"});
  item("--video-skip-collapse",
       {"Video: collapse an all-static superblock (every leaf a plain",
        "zero-motion inter skip) to one superblock-sized skip leaf; the",
        "reconstruction is unchanged, only the partition tree disappears"});
  item("--video-sb128", {"Video: 128x128 superblocks (§5.4.3",
                         "use_128x128_superblock); leaves stay <= 64x64"});
  item("--video-sb256", {"Video: 256x256 superblocks (§5.4.3",
                         "use_256x256_superblock); leaves stay <= 64x64"});
  item("--video-ext-partitions", {"Video: extended/uneven 4-way partitions",
                                  "(§5.4.3 enable_ext_partitions); leaves",
                                  "stay >= 8x8 and own their chroma;",
                                  "composes with --nightly (same leaf RD",
                                  "model, carried CDFs and loop filters)"});
  item("--video-min-block <N>",
       {"Video: minimum partition block size, 4..64 (default 16); 4 enables",
        "§5.20.3.1 sub-8 keyframe leaves with shared-chroma ownership"});
  item("--video-max-depth <D>", {"Video: partition depth cap [0, 8] (default 4)"});
  item("--video-me-range <R>", {"Video: integer ME search radius (default 16)"});
  item("--video-trace", {"Video: trace block decisions to stderr"});
}
