# NVDEC VP9 input feasibility — RTX 3060

**Status: feasibility/demo only. No encoder integration is claimed.**

This bench answers whether the RTX 3060 can decode the user's 4K VP9 source
with NVDEC and hand GPU-resident frames to a CUDA encoder without a host frame
copy.

## Verdict

| Question | Result |
|---|---|
| Can a 3060 NVDEC decode VP9 Profile 0, 8-bit 4:2:0, 4K? | **Yes, on the hardware.** Ampere has a 5th-gen NVDEC; NVIDIA's matrix lists VP9 4:2:0 8-bit decode as supported. |
| Does NVDEC work in this Vast container? | **No.** The CUVID caps query and decoder open fail with `CUDA_ERROR_NO_DEVICE` (100) on driver 570.181. This is the NVIDIA 570.x multi-GPU container regression, not a missing 3060 capability. |
| Are FFmpeg CUDA hwframes real device memory? | **Yes, proven** with `AVHWFramesContext`: `cudaPointerGetAttributes` reports `cudaMemoryTypeDevice`, `host_accessible=no`. |
| Does NVDEC output use the same contract? | **Yes in FFmpeg** (NVDEC hwaccel returns `AV_PIX_FMT_CUDA` frames from an `AVHWFramesContext`), but the decoder cannot run here to demonstrate it directly. |
| Can a CUDA kernel consume the frames with no host frame copy? | **Yes, proven** with a 4K NV12 synthetic pool. Only 8-byte checksums cross PCIe. |
| Measured NVDEC decode fps at 4K/1080p | **Not measurable here**: the decoder fails before the first frame. Software-decode numbers are below. |
| Is the code ready for a fixed host? | **Yes.** `nvdec_probe --input clip.webm` is the exact FFmpeg + CUDA interop path, and fails with a clear diagnostic when NVDEC is unavailable. |

The short version: **the 3060 is capable, but this container cannot expose its
video decoder.** Do not block the encoder on NVDEC; implement the `InputSource`
split described in [`docs/nvdec.md`](../../docs/nvdec.md), with software decode
+ pinned upload as the fallback.

## Source

| Item | Value |
|---|---|
| File | [LG QNED 8K MiniLED │Shine 8K HDR 60fps](https://youtu.be/WEwgbKr5R9I) (local 4K VP9 download) |
| Container / codec | WebM (Matroska) / VP9 Profile 0 |
| Resolution | 3840×2160 |
| Pixel format | `yuv420p` — 8-bit 4:2:0 |
| Frame rate | 60/1 |
| Duration | 121.802 s |
| Size / bitrate | 318,164,700 B / ~20.9 Mbps |
| Color | BT.709, TV range |

Despite the filename, this stream is **8-bit SDR BT.709**, not 10-bit PQ/HLG.
No 10-bit or HDR input path is needed for this file.

### Test clips

Media is not committed. Generate it from the source:

```sh
# 4K, stream copy (authentic VP9 bitstream), 424 frames / 7.05 s
ffmpeg -ss 20 -i "$SRC" -t 5 -map 0:v:0 -c copy -an 4k_vp9_5s.webm

# 1080p, re-encoded for a 1080p decode test, 300 frames / 5 s
ffmpeg -ss 20 -i "$SRC" -t 5 -map 0:v:0 \
  -vf scale=1920:1080:flags=bicubic \
  -c:v libvpx-vp9 -b:v 8M -deadline realtime -cpu-used 8 -row-mt 1 \
  -an 1080p_vp9_5s.webm
```

The 4K copy starts at the keyframe before 20 s, so it is 424 frames, not 300.
On the Vast box the generated clips live at `/workspace/4k_vp9_5s.webm` and
`/workspace/1080p_vp9_5s.webm` (the overlay is non-persistent). All decodes in
this bench use `-f null -`; no raw Y4M is written to the 16 GiB overlay.

## Environment

| Item | Value |
|---|---|
| Host | Vast.ai, unprivileged Docker container |
| CPU | 2× Intel Xeon E5-2690 v4, 28C/56T, 125 GiB |
| GPU | NVIDIA GeForce RTX 3060, 12,288 MiB, driver 570.181, sm_86 |
| CUDA | 12.8, `nvcc` 12.8.93 |
| OS / FFmpeg | Ubuntu 24.04, `ffmpeg 6.1.1-3ubuntu5` |
| Disk | 16 GiB overlay, non-persistent |
| Mac source host | Apple M4, macOS 26.6.2 |

`ffmpeg` on the box advertises the CUDA hwaccel and CUVID decoders:

```text
$ ffmpeg -hwaccels
vdpau cuda vaapi qsv drm opencl vulkan

$ ffmpeg -decoders | grep -E 'vp9'
VFS..D vp9          Google VP9
V..... vp9_v4l2m2m  V4L2 mem2mem VP9 decoder wrapper
V..... libvpx-vp9   libvpx VP9 decoder
V..... vp9_cuvid    Nvidia CUVID VP9 decoder
V..... vp9_qsv      VP9 video (Intel Quick Sync Video)
```

Userspace dev headers were installed (no driver changes):

```sh
apt-get install -y --no-install-recommends \
  libavcodec-dev libavutil-dev libavformat-dev pkg-config time
```

## NVDEC CLI test

Command:

```sh
ffmpeg -hwaccel cuda -hwaccel_output_format cuda \
  -i 4k_vp9_5s.webm -an -f null -
```

Result on this box:

```text
[vp9 @ 0x...] decoder->cvdl->cuvidGetDecoderCaps(&caps) failed ->
  CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
[vp9 @ 0x...] Failed setup for format cuda:
  hwaccel initialisation returned error.
```

The same happens at 1080p. `-c:v vp9_cuvid` fails at open:

```text
[vp9_cuvid @ 0x...] ctx->cvdl->cuvidGetDecoderCaps(&ctx->caps12) failed ->
  CUDA_ERROR_NO_DEVICE
[vist#0:0/vp9 @ 0x...] Error while opening decoder: Generic error in an external library
```

Exit codes and the trap:

| Command | Exit | What actually happened |
|---|---:|---|
| `ffmpeg -hwaccel cuda ...` | **0** | FFmpeg silently fell back to software decode. Do not trust exit 0. |
| `ffmpeg -c:v vp9_cuvid ...` | 187 | Decoder open failed; no fallback. |

During the 4K `-hwaccel cuda` run, this was sampled every 200 ms:

```sh
nvidia-smi --query-gpu=utilization.gpu,utilization.decoder,utilization.memory,memory.used \
  --format=csv,noheader
```

```text
GPU util 0 %, decoder util 0 %, memory util 0 %, 126 MiB used
CPU 568 % (software VP9 decode)
```

So the attempt did **not** stay on the GPU. No NVDEC fps is reported because
no NVDEC frame was decoded.

## Why NVDEC is blocked in this container

The host has **seven** RTX 3060s; this container is on host GPU 3
(`/dev/nvidia3`, PCI bus `08:00.0`). `nvidia-smi` shows one GPU, CUDA compute
works, and `cudaGetDeviceCount()` returns 1.

The device cgroup allows only:

```text
c 195:3 rwm      # our GPU
c 195:254 rwm
c 195:255 rwm
c 235:0 rwm
c 235:1 rwm
```

It does not allow the other GPUs (`195:0..2,4..6`) or `/dev/nvidia-caps/*`
(`238:1`, `238:2`). The device cgroup is mounted read-only, so this cannot be
changed from inside.

The NVIDIA 570.x/575.x/580.x driver has a container regression: `libnvcuvid`
asks the kernel for **all attached GPUs**
(`NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS`), then peer-inits every GPU in the
list. GPUs whose device nodes are not accessible make the peer-init fail, and
the decoder aborts with `CUDA_ERROR_NO_DEVICE` — even though the assigned GPU
is fine. See:

- <https://github.com/NVIDIA/nvidia-container-toolkit/issues/1249>
- <https://github.com/NVIDIA/nvidia-container-toolkit/issues/1415>

A direct probe confirms it is not FFmpeg:

```text
cuInit=0  deviceCount=1  dev0=NVIDIA GeForce RTX 3060
cuCtxCreate=0  (context current)
VP9:  r=100 supported=0  max=0x0
H264: r=100 supported=0
HEVC: r=100 supported=0
AV1:  r=100 supported=0
```

CUDA hwcontext itself works: `ffmpeg ... -vf format=yuv420p,hwupload_cuda`
succeeds. Only the CUVID/NVDEC library fails.

We tried the known node-numbering workaround: aliasing `/dev/nvidia0..6` to
`195:3` so every node opens. It still fails, because the kernel Resource
Manager knows the fd is GPU 3 while the host's last GPU is 6. Setting
`CUDA_VISIBLE_DEVICES` or `NVIDIA_VISIBLE_DEVICES` to `0` or to the GPU UUID at
runtime also does not help; the RM list is not filtered by those variables.
The real workarounds are infrastructure changes:

1. Pass **all** host GPUs into the container and select ours with
   `CUDA_VISIBLE_DEVICES`, or
2. pass the **last** host GPU (minor 6) through, or
3. use a driver release with the fix (550.x/565.x are reported to work, but
   they may not support the project's CUDA 12.8 toolkit; prefer 1 or 2).

None of these is a code change; all require the container/driver owner.

## Software-decode baseline (what NVDEC would replace)

Idle-box runs, three repeats, full 424-frame 4K clip / 300-frame 1080p clip.
`ffmpeg -benchmark -i clip -an -f null -`, measured with `/usr/bin/time -v`.

### 4K, 3840×2160

| Run | Wall | FPS | Realtime | CPU | User time | Max RSS |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3.76 s | 112.8 | 1.88× | 687% | 24.89 s | 368 MB |
| 2 | 3.98 s | 106.5 | 1.78× | 637% | 24.47 s | 369 MB |
| 3 | 3.54 s | 119.8 | 2.00× | 679% | 23.53 s | 368 MB |
| median | 3.76 s | **112.8** | **1.88×** | ~680% | 24.5 s | 368 MB |

### 1080p, 1920×1080

| Run | Wall | FPS | Realtime | CPU | User time | Max RSS |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.04 s | 288.5 | 4.81× | 566% | 5.68 s | 130 MB |
| 2 | 1.06 s | 283.0 | 4.72× | 560% | 5.71 s | 130 MB |
| 3 | 1.07 s | 280.4 | 4.67× | 553% | 5.73 s | 130 MB |
| median | 1.06 s | **283.0** | **4.72×** | ~560% | 5.71 s | 130 MB |

Variants at 4K:

| Decoder / threading | Wall | FPS | CPU |
|---|---:|---:|---:|
| `vp9` default | 3.76 s | 112.8 | 687% |
| `vp9 -thread_type frame -threads 16` | 3.57 s | 118.8 | 681% |
| `libvpx-vp9 -threads 16` | 4.29 s | 98.8 | 622% |

**Shared-box warning.** The Vast box is shared with other agents. Early runs
during heavy CPU contention measured 14 fps at 4K (101% CPU) and 34 fps at
1080p (88% CPU). A later repeat under moderate load measured 98.4 fps at 4K
and 263 fps at 1080p. The tables above are the idle-box runs; treat the
4K/1080p numbers as ranges (98–120 / 263–289 fps) and pin/isolate the box for
repeatable measurements. Do not compare these to the M4.

Interpretation: software decode is fast enough to feed a 1 fps encoder even at
4K, but at full speed it burns ~6.5–6.8 of the 7.68 available CPU cores, and
every frame lands in host memory. Feeding 60 fps uses roughly half that CPU on
average, but it is still a large cost; the 4K decode has only ~1.9× realtime
headroom. NVDEC would remove both costs.

## CUDA device-frame handoff (proven without NVDEC)

`nvdec_probe.cu` has two modes. `--synthetic` builds a real
`AVHWFramesContext`, allocates 4K NV12 CUDA frames, fills them with a kernel,
checksums them, and copies them into encoder-owned device planes. This is the
exact `AVHWFramesContext → CUdeviceptr → kernel` path the encoder will use.

Build:

```sh
make            # or:
nvcc -O2 -std=c++17 -arch=sm_86 nvdec_probe.cu -o nvdec_probe \
  $(pkg-config --cflags --libs libavformat libavcodec libavutil) \
  -lcuda -lcudart
```

Run:

```sh
./nvdec_probe --synthetic --frames 6 --copy
```

Key output:

```text
CUDA device 0: NVIDIA GeForce RTX 3060, cc 8.6, 28 SMs, 1 device(s) visible
FFmpeg CUDA hwdevice: ctx=0x... stream=(nil)
CUDA current context=0x..., ffmpeg=0x...: same context (arena/frames shareable)
synthetic: AVHWFramesContext format=cuda sw_format=nv12 3840x2160 pool=4
frame 0: format=cuda 3840x2160 linesize=4096/4096 data0=0x... data1=0x...
  luma: ptr=0x... type=device device=0 host_accessible=no
  chroma: ptr=0x... type=device device=0 host_accessible=no
frame  0: checksum luma=... chroma=... kernel=0.487 ms copy_to_encoder_pool=0.645 ms match=yes
...
synthetic: 6 frames in 0.050 s (119.88 fps), avg kernel 0.484 ms, avg copy 0.643 ms
synthetic: host->device frame bytes=0, device->host bytes=192 (checksums only)
```

What this proves:

- `AVHWFramesContext.format = AV_PIX_FMT_CUDA`, `sw_format = NV12`.
- `frame->data[0]` / `data[1]` are `cudaMemoryTypeDevice` pointers; a host
  dereference is impossible, and `cudaPointerGetAttributes` says
  `host_accessible=no`.
- A CUDA kernel reads the hwframe directly. This is the same frame object and
  pointer contract NVDEC's CUDA hwaccel returns; only the decoder that fills
  it is synthetic here.
- The optional copy into encoder-owned device planes is GPU→GPU and its
  checksum matches the direct read. It preserves NV12 (interleaved chroma); it
  is not an I420 conversion. The encoder still needs an NV12→I420 kernel or
  NV12-aware chroma access.
- Host traffic is **0 bytes up, 8 bytes per checksum down** (192 bytes for
  6 frames × 2 planes × 2 checksum reads the way this demo does it).
- NVDEC pitches are **not tight**: 3840-wide 4K has `linesize=4096`.
  Every encoder kernel that consumes these frames must take a pitch/stride.

The kernel timings are naive (one byte per thread; not optimized). They are
functional evidence, not a peak-bandwidth number. The synthetic fps also
includes pool allocation and host synchronization; it is not a pipeline
throughput.

Real NVDEC mode on this box:

```sh
./nvdec_probe --input 4k_vp9_5s.webm --frames 4 --copy
# exit 2:
# NVDEC unavailable: decoder opened with yuv420p, not cuda.
# FFmpeg silently fell back to software after the CUDA hwaccel failed.
```

On a host where NVDEC works, the same command prints the first CUDA frame's
device pointers, decodes at NVDEC speed, and runs the checksum/copy kernels
per frame. That part is untested here.

## Fallback upload cost

If NVDEC is unavailable, a software-decoded 4K NV12 frame must be uploaded.
Pinned-memory benchmark on this box:

```text
H2D pinned: 1.05 ms/frame, 11.8 GB/s for 11.87 MiB (12.44 MB) of 4K NV12
D2H pinned: 1.05 ms/frame, 11.9 GB/s
```

That is ~1 ms on top of ~8.9 ms/frame of software decode at 112.8 fps. For a
1 fps encoder the fallback is viable; it is not GPU-resident end to end.

## Files

| File | Purpose |
|---|---|
| `nvdec_probe.cu` | NVDEC + CUDA interop demo; `--synthetic` path runs without NVDEC |
| `Makefile` | Builds the demo with `nvcc` + pkg-config |
| `.gitignore` | Keeps local `.webm` clips and the demo binary out of git |
| `README.md` | This report |
| [`docs/nvdec.md`](../../docs/nvdec.md) | Integration plan, `InputSource` design, risks |

## Recommendation

1. **Do not make M0/M1 depend on NVDEC.** Add an `InputSource` abstraction with
   `NvdecInputSource` and `SoftwareUploadInputSource`; use the latter on this
   box, the former when the container is fixed.
2. **Keep the zero-copy contract:** `NvdecInputSource` returns CUDA device
   pointers plus pitch and an owning `AVBufferRef`, never an `AVFrame` copied
   with `av_hwframe_transfer_data`.
3. **Handle NV12 and pitch from day one.** A 4K NVDEC frame is 3840×2160 with
   `linesize=4096`; the encoder must not assume tight I420 rows.
4. **Fail loudly on software fallback.** FFmpeg exits 0 after falling back.
   Check `dec->pix_fmt == AV_PIX_FMT_CUDA` after `avcodec_open2`.
5. **Fix the infrastructure before claiming NVDEC.** Pass all host GPUs (or the
   last one) into the container, or use driver 550.x/565.x. Validate with
   `./nvdec_probe --input ...` and `nvidia-smi` decoder utilization > 0.
