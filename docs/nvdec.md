# NVDEC input path — feasibility and integration plan

**Status: feasibility complete; encoder integration is not started.** This page
records what was measured, what is blocked, and how to wire NVDEC into the
GPU-resident input path. The runnable demo and raw numbers are in
[`bench/nvdec_pipeline/`](../bench/nvdec_pipeline/README.md).

## Verdict

| Question | Answer |
|---|---|
| Can the RTX 3060 NVDEC decode VP9 Profile 0, 8-bit 4:2:0 at 4K? | **Yes, on the hardware.** Ampere has a 5th-gen NVDEC; NVIDIA's matrix lists VP9 4:2:0 8-bit decode as supported. |
| Does NVDEC work in the current Vast container? | **No.** The CUVID caps query and decoder open fail with `CUDA_ERROR_NO_DEVICE` (100) on driver 570.181. This is a container/driver regression, not a 3060 limitation. |
| Are FFmpeg CUDA hwframes device memory? | **Yes, proven.** `AVHWFramesContext` frames report `cudaMemoryTypeDevice`; `cudaPointerGetAttributes` says `host_accessible=no`. |
| Does NVDEC output use the same contract? | **Yes in FFmpeg** (NVDEC hwaccel returns `AV_PIX_FMT_CUDA` frames from an `AVHWFramesContext`), but the decoder is blocked here, so this link is not demonstrated directly. |
| Can a CUDA kernel consume them without a host copy? | **Yes, proven** on a synthetic 4K NV12 pool. Only 8-byte checksums cross PCIe. |
| Is NVDEC measured at 4K/1080p here? | **No.** The decoder fails before the first frame, so no NVDEC fps exists to report. Software-decode numbers are in the bench README. |
| Should the encoder block on NVDEC? | **No.** Ship an `InputSource` split now; enable the NVDEC backend when the container is fixed. |

The correct conclusion for M0/M1 is: **NVDEC is the right long-term input path,
but it is not available on this box today.** The CUDA handoff contract is
already proven; the missing piece is the NVDEC decoder itself.

## Source and test clips

The user's source is VP9 Profile 0, 3840×2160, 8-bit 4:2:0, 60 fps, BT.709 TV
range, 121.802 s, ~20.9 Mbps. It is SDR despite the "HDR" filename.

The bench uses two short clips generated on the Mac:

| Clip | How | Frames / duration | Size |
|---|---|---|---:|
| `4k_vp9_5s.webm` | stream copy from 20 s (authentic bitstream) | 424 / 7.05 s | 22 MB |
| `1080p_vp9_5s.webm` | scale to 1080p, libvpx-vp9 8 Mbps | 300 / 5.0 s | 7.1 MB |

The 4K copy starts on the keyframe before 20 s, hence 424 frames. Commands are
in the bench README. All decodes used `-f null`; no raw Y4M was written to the
16 GiB overlay.

## Measured summary

### NVDEC attempt

```sh
ffmpeg -hwaccel cuda -hwaccel_output_format cuda \
  -i 4k_vp9_5s.webm -an -f null -
```

```text
[vp9 @ ...] decoder->cvdl->cuvidGetDecoderCaps(&caps) failed ->
  CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
[vp9 @ ...] Failed setup for format cuda:
  hwaccel initialisation returned error.
```

- 4K and 1080p fail identically.
- `-c:v vp9_cuvid` fails at decoder open with exit 187.
- `-hwaccel cuda` **exits 0** by silently falling back to software. This is a
  trap: a successful exit does not mean NVDEC ran.
- During the 4K attempt: GPU util 0%, NVDEC/decoder util 0%, 126 MiB used,
  CPU 568% (software decode). Decode did not stay on the GPU.

### Software decode (idle box, median of 3)

| Resolution | Wall | FPS | Realtime | CPU | Max RSS |
|---|---:|---:|---:|---:|---:|
| 4K | 3.76 s | 112.8 | 1.88× | ~680% | 368 MB |
| 1080p | 1.06 s | 283.0 | 4.72× | ~560% | 130 MB |

The box is shared; CPU contention produced 14 fps at 4K in early runs, and a
moderate-load repeat gave 98.4 fps at 4K / 263 fps at 1080p. Treat these as
idle-box numbers and ranges, not guarantees. At 4K, software decode uses
~6.5–6.8 of the 7.68 available cores at full speed; feeding 60 fps uses roughly
half that on average, but it is still a large CPU cost and all frames land in
host memory.

### Device-frame handoff (synthetic, no NVDEC)

`./nvdec_probe --synthetic --frames 6 --copy`:

- `AVHWFramesContext`: `format=cuda`, `sw_format=nv12`, 3840×2160, pool 4.
- `frame->data[0]/[1]`: `cudaMemoryTypeDevice`, `host_accessible=no`.
- `linesize=4096/4096` for a 3840-wide frame. **Pitch is not tight.**
- Direct checksum and GPU→GPU copy checksum match on every frame.
- Host traffic: 0 bytes up; only 8-byte plane checksums down (32 bytes/frame
  in this demo: two planes, direct + handoff).
- Naive kernel: ~0.5 ms checksum, ~0.65 ms copy per 4K frame (box-load
  dependent; not a peak-bandwidth number).

### Fallback upload

Pinned 4K NV12 (11.87 MiB): 1.05 ms/frame, 11.8 GB/s host→device. That is the
minimum extra cost of a software-decode + upload path.

## Why NVDEC is blocked

The host has seven RTX 3060s. This container is on host GPU 3
(`/dev/nvidia3`, PCI `08:00.0`). CUDA compute works and `cudaGetDeviceCount()`
returns 1, but the device cgroup allows only `195:3` and a few control nodes —
not the other GPUs, not `/dev/nvidia-caps/*` (238:1/238:2). The cgroup is
read-only inside the container.

On driver 570.x/575.x/580.x, `libnvcuvid` asks the kernel for **all** attached
GPUs (`NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS`) and peer-inits every one. A GPU
whose node is not accessible makes peer-init fail, and the decoder returns
`CUDA_ERROR_NO_DEVICE` even though the assigned GPU is healthy. See
[NVIDIA/nvidia-container-toolkit#1249](https://github.com/NVIDIA/nvidia-container-toolkit/issues/1249)
and [#1415](https://github.com/NVIDIA/nvidia-container-toolkit/issues/1415).

A direct probe rules out FFmpeg and the CUDA hwcontext: `cuInit`,
`cuDeviceGet`, `cuCtxCreate`, and a current context all succeed; `hwupload_cuda`
works; `cuvidGetDecoderCaps` then fails for VP9, H.264, HEVC, and AV1. Aliasing
every `/dev/nvidiaN` node to `195:3` does **not** fix it, because the kernel
Resource Manager knows the fd is GPU 3 while the host's last GPU is 6.
Runtime `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES` overrides (index or
UUID) do not help either.

Infrastructure fixes, in order of practicality:

1. Pass **all** host GPUs into the container; select one with
   `CUDA_VISIBLE_DEVICES`. This is the known workaround.
2. Pass the **last** host GPU (minor 6) through. The app can still use any GPU.
3. Use a driver release with the fix. 550.x/565.x are reported to work, but
   they may not support CUDA 12.8; prefer 1 or 2 for this project.

None is a code change. Until one is applied, the NVDEC backend cannot be
validated on this host.

## Integration design

### Proposed `InputSource`

Keep `src/io/y4m` for tests and add a device-frame source. This is a proposal
for the scaffold agent, not an edit to `src/` from this task.

```cpp
// Proposed: src/io/input_source.h. Coordinate before adding FFmpeg/CUDA
// headers to the core I/O layer.
namespace av2enc::io {

struct DeviceFrame {
    CUdeviceptr luma   = 0;
    CUdeviceptr chroma = 0;   // NV12: interleaved U/V
    int luma_pitch     = 0;
    int chroma_pitch   = 0;
    int width          = 0;
    int height         = 0;
    int64_t pts        = 0;
    AVBufferRef *owner = nullptr;  // keeps the decoder surface alive
};

class InputSource {
public:
    virtual ~InputSource() = default;
    virtual bool next(DeviceFrame &out) = 0;  // false at EOF
    virtual AVRational frame_rate() const = 0;
    virtual int width() const = 0;
    virtual int height() const = 0;
};

}  // namespace av2enc::io
```

Backends:

| Implementation | Path | Use |
|---|---|---|
| `NvdecInputSource` | libavformat demux + libavcodec CUDA hwaccel + `AV_PIX_FMT_CUDA` | preferred, zero host copy |
| `SoftwareUploadInputSource` | libavcodec software decode + pinned H2D | fallback for this box |
| `Y4mInputSource` | existing host Y4M reader | deterministic tests |

`main.cpp` chooses with `--input-backend nvdec|software|y4m`; default is
`auto`: try NVDEC, fall back to software, and log which was used. Add
`--require-nvdec` for validation runs.

### NVDEC path

1. `avformat_open_input` / `avformat_find_stream_info`.
2. `av_hwdevice_ctx_create(&device_ref, AV_HWDEVICE_TYPE_CUDA, ...)`.
3. `avcodec_find_decoder(AV_CODEC_ID_VP9)` (or H.264/HEVC/AV1 by codec id).
4. Set `dec->hw_device_ctx = av_buffer_ref(device_ref)` and a `get_format`
   callback that returns `AV_PIX_FMT_CUDA`.
5. `avcodec_open2`; **check `dec->pix_fmt == AV_PIX_FMT_CUDA`**. If not, FFmpeg
   fell back to software; close and report.
6. `avcodec_receive_frame` returns an `AVFrame` with `format=AV_PIX_FMT_CUDA`
   and `hw_frames_ctx`. The device pointers are `frame->data[0]` (luma) and
   `frame->data[1]` (NV12 chroma); strides are `frame->linesize[]`.
7. Run encoder kernels directly on those pointers, or copy GPU→GPU into the
   encoder frame pool. Never call `av_hwframe_transfer_data` for the encoder
   path.

The demo in `bench/nvdec_pipeline/nvdec_probe.cu` is the reference
implementation of steps 1–7. It uses `cudaMalloc` once at init for clarity;
production must allocate from the 7,500 MB arena (`src/cuda/arena.*`) and must
not call `cudaMalloc` in the frame loop.

### Zero-copy ownership: three options

**A. Consume decoder-owned surfaces (simplest).**
Keep an `AVBufferRef` to the `AVFrame`/hwframe while the encoder reads it.
The decoder recycles the surface when the last ref drops. The pool must be
large enough for the encoder's in-flight frames plus decoder delay.

**B. Decode directly into the encoder pool (preferred).**
Give the decoder an `AVHWFramesContext` backed by the encoder arena (or a
custom `get_buffer2`) so NVDEC writes into encoder-owned surfaces. This avoids
both the separate decoder pool and the GPU copy. It requires:

- the same `sw_format` (NV12), dimensions, and compatible pitch;
- the same CUDA context (see below);
- a pool sized for refs + lookahead + decoder delay.

This is the target design, but it is **not tested** yet.

**C. Decoder pool + GPU copy (safe fallback).**
The demo's `--copy` path: checksum the decoded frame, copy luma/chroma into
tight encoder-owned device planes with a kernel, checksum again. It preserves
NV12, so an NV12→I420 step is still needed. Still zero host copy, but costs one
full-frame GPU read+write (~0.64 ms for 4K in the naive demo). Use this if B is
awkward.

### CUDA context requirement

FFmpeg's CUDA hwcontext creates a CUDA context; the encoder arena and kernels
must live in **the same context** as the decoder surfaces. Device pointers are
not freely shareable across unrelated driver contexts on the same GPU.

Practical rules:

- create the `AVHWDeviceContext` first, then initialize the encoder arena
  while that context is current; or
- create the hwdevice with `AV_CUDA_USE_PRIMARY_CONTEXT` and make the encoder
  use the primary context; or
- explicitly `cuCtxSetCurrent(cudactx->cuda_ctx)` before arena allocation.

The demo makes the contract explicit: after `av_hwdevice_ctx_create` it calls
`cuCtxSetCurrent(cudactx->cuda_ctx)`, then allocates and launches. The
integration must do the same (or use the primary-context flag). Add a startup
assertion that a pointer from `av_hwframe_get_buffer` is queryable from the
encoder context.

Streams: FFmpeg 6.1 exposes `AVCUDADeviceContext.stream = NULL` (default
stream). If the encoder uses its own stream, insert an event/wait after
`avcodec_receive_frame`; do not assume the decoder and encoder streams are
ordered.

### NV12, pitch, and color

NVDEC outputs **NV12** (semi-planar chroma) with a padded pitch: 3840×2160
gives `linesize=4096`. The M0 encoder data model is planar 4:2:0 (I420) with
tight rows.

Pick one:

- **Convert on GPU:** add an `nv12_to_i420` kernel that writes into the
  encoder's planar frame pool. 4K chroma is 4.1 MB; the copy is sub-millisecond
  on the 3060. Keeps all existing kernels unchanged.
- **Teach kernels NV12:** pass `chroma` as `uchar2` and a chroma pitch. More
  invasive; only worth it if chroma bandwidth matters.

Either way:

- every kernel takes a pitch, never `width`;
- the encoder arena should use the same padded pitch/alignment if it will be
  the decoder's target (option B);
- the source is BT.709 **TV range**. Preserve range/primaries/transfer
  metadata; do not silently expand to full range. If the Y4M path assumes
  full range, add an explicit range flag/test so PSNR comparisons stay
  meaningful.

### VRAM and lifetime

4K NV12 is 12.4 MB/frame. Budget:

| Item | Frames | VRAM |
|---|---:|---:|
| NVDEC surface pool | 8–16 | 100–200 MB |
| Encoder reference pool | 4 | ~50 MB |
| Lookahead / scratch | — | profile-dependent |
| CUDA context + libraries | — | ~300 MB |

This fits the 7,500 MB `compat8g` cap with room to spare. Size the pool as
`refs + lookahead + decoder_delay + 2`, log `peak_vram_mb`, and fail with a
clear message rather than letting the decoder stall or OOM.

Frame lifetime: hold `AVBufferRef`s, not raw `AVFrame*`; release as soon as
the encoder's last kernel that reads the frame has completed (or use a CUDA
event). Holding frames for lookahead without resizing the pool stalls decode.

## Integration steps

1. **Coordinate with the scaffold agent** before touching `src/io/` or
   `CMakeLists.txt`. This task owns only `bench/nvdec_pipeline/` and
   `docs/nvdec.md`.
2. Add an FFmpeg dependency (`libavformat`, `libavcodec`, `libavutil`) or use
   the NVIDIA Video Codec SDK directly. FFmpeg is the faster path and gives
   `AVHWFramesContext` ownership; the SDK avoids the dependency but is more
   code. The hard constraint only forbids third-party **encoder** code;
   libavcodec as an input decoder is fine.
3. Add `src/io/input_source.h` and `NvdecInputSource`.
4. Add `SoftwareUploadInputSource` as the fallback (software decode + pinned
   H2D, 1.05 ms/frame at 4K).
5. Make the frame pool accept external device frames: pitch-aware accessors,
   same-context allocation, refcounted ownership.
6. Add NV12→I420 (or NV12-aware kernels).
7. Wire `--input`, `--input-backend`, `--require-nvdec`; log
   `input_backend`, `host_to_device_frame_bytes`, and `device_to_host_bytes`.
8. Add tests:
   - synthetic `AVHWFramesContext` + kernel (runs on any CUDA box, no NVDEC);
   - NVDEC smoke test that skips cleanly when `dec->pix_fmt != AV_PIX_FMT_CUDA`;
   - PSNR between NVDEC and software decode on the same clip, to quantify
     decoder differences.
9. Validate on the target 8192 MiB 3060 once the container exposes NVDEC.
   Confirm `nvidia-smi` decoder utilization > 0 and
   `host_to_device_frame_bytes == 0`.

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Container/driver currently blocks NVDEC | **High** | Fix infrastructure (all/last GPU, driver 550/565); keep software fallback. |
| CUDA context mismatch between decoder and arena | **High** | Create hwdevice first / use primary context; assert pointer queryability. |
| NVDEC vs software decode is not bit-identical | Medium | Measure PSNR; pin the backend for deterministic tests; don't mix backends in one A/B. |
| NV12/pitch bugs | Medium | Pitch-aware kernels; NV12→I420 conversion test at 4K and odd sizes. |
| Frame lifetime / pool exhaustion | Medium | Refcounts, pool sizing, event-based release; log stalls. |
| VRAM pressure at 4K | Low–Medium | 12.4 MB/frame; log peak; cap pool size. |
| Multi-GPU: each GPU needs its own NVDEC | Medium | Decode per GPU (duplicate compressed input) or accept a host copy; GeForce has no P2P. |
| Driver/GPU variability | Medium | Pin driver; validate on the 8192 MiB target, not only the 12 GB proxy. |
| Color range/metadata | Low | Preserve BT.709 TV range; add range test. |
| NVDEC session limits on GeForce | Low | One stream per GPU; irrelevant at the 1 fps target. |
| FFmpeg dependency / licensing | Low | libavcodec is an input decoder, not encoder code; or use the SDK. |

## Reproduce

On a Linux box with CUDA 12.8 and FFmpeg dev headers:

```sh
cd bench/nvdec_pipeline
make
./nvdec_probe --synthetic --frames 6 --copy
./nvdec_probe --input 4k_vp9_5s.webm --frames 4 --copy
```

On this Vast box the second command exits 2 with the software-fallback
diagnostic. On a host with working NVDEC it prints the first CUDA frame's
device pointers and decodes at NVDEC speed.

## References

- NVIDIA Video Encode and Decode GPU Support Matrix:
  <https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new>
- NVIDIA container toolkit multi-GPU NVENC/NVDEC regression:
  <https://github.com/NVIDIA/nvidia-container-toolkit/issues/1249>,
  <https://github.com/NVIDIA/nvidia-container-toolkit/issues/1415>
- FFmpeg CUDA hwaccel / hwcontext:
  <https://ffmpeg.org/ffmpeg.html#HWAccelIntro>
- Bench report and raw numbers:
  [`bench/nvdec_pipeline/README.md`](../bench/nvdec_pipeline/README.md)
