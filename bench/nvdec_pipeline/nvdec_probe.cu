// nvdec_probe.cu — NVDEC -> CUDA device-frame feasibility probe.
//
// This is a feasibility demo, not encoder integration. It answers three
// questions:
//   1. Can libavcodec open a VP9 stream with the CUDA hwaccel and return
//      AV_PIX_FMT_CUDA frames?
//   2. Is frame->data[0] a real CUDA device pointer (not a host staging
//      buffer)?
//   3. Can a CUDA kernel read that frame and write an encoder-owned device
//      frame without a host round trip?
//
// Two modes:
//   ./nvdec_probe --input clip.webm [--frames N] [--copy]
//       Real NVDEC decode. Fails cleanly if NVDEC is unavailable.
//   ./nvdec_probe --synthetic [--frames N] [--copy]
//       Builds an AVHWFramesContext by hand, allocates CUDA NV12 frames,
//       fills them with a kernel, checksums them, and optionally copies them
//       into encoder-owned device buffers. This exercises the exact
//       AVHWFramesContext -> CUdeviceptr -> kernel path used by the encoder
//       without depending on NVDEC.
//
// Build (Ubuntu 24.04, CUDA 12.8, FFmpeg 6.1 dev headers):
//   nvcc -O2 -std=c++17 -arch=sm_86 nvdec_probe.cu -o nvdec_probe
//     $(pkg-config --cflags --libs libavformat libavcodec libavutil)
//     -lcuda -lcudart
//
// See README.md for measured results and docs/nvdec.md for the integration
// plan and the container-level NVDEC blocker found on the Vast box.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
}

// ---------------------------------------------------------------------------
// CUDA helpers
// ---------------------------------------------------------------------------

#define CU_CHECK(expr)                                                         \
    do {                                                                       \
        CUresult _r = (expr);                                                  \
        if (_r != CUDA_SUCCESS) {                                              \
            const char *_s = "?";                                              \
            cuGetErrorString(_r, &_s);                                         \
            std::fprintf(stderr, "CUDA driver error %d (%s) at %s:%d: %s\n",   \
                         (int)_r, _s, __FILE__, __LINE__, #expr);              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define RT_CHECK(expr)                                                         \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA runtime error %d (%s) at %s:%d: %s\n",  \
                         (int)_e, cudaGetErrorString(_e), __FILE__, __LINE__,  \
                         #expr);                                               \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static void print_device_info() {
    int count = 0;
    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGetCount(&count));
    int dev = 0;
    CU_CHECK(cuDeviceGet(&dev, 0));
    char name[256] = {0};
    CU_CHECK(cuDeviceGetName(name, sizeof(name), dev));
    int sm = 0, cc_major = 0, cc_minor = 0;
    CU_CHECK(cuDeviceGetAttribute(&sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    CU_CHECK(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU_CHECK(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    std::printf("CUDA device 0: %s, cc %d.%d, %d SMs, %d device(s) visible\n",
                name, cc_major, cc_minor, sm, count);
}

static void print_pointer_kind(const char *label, const void *p) {
    cudaPointerAttributes attr{};
    cudaError_t e = cudaPointerGetAttributes(&attr, p);
    if (e != cudaSuccess) {
        std::printf("  %s: pointer %p not queryable (%s)\n", label, p,
                    cudaGetErrorString(e));
        return;
    }
    const char *type = "unknown";
    switch (attr.type) {
    case cudaMemoryTypeUnregistered: type = "unregistered/host"; break;
    case cudaMemoryTypeHost:         type = "host"; break;
    case cudaMemoryTypeDevice:       type = "device"; break;
    case cudaMemoryTypeManaged:      type = "managed"; break;
    default: break;
    }
    std::printf("  %s: ptr=%p type=%s device=%d host_accessible=%s\n", label, p,
                type, attr.device,
                attr.type == cudaMemoryTypeDevice ? "no" : "yes");
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void fill_kernel(uint8_t *p, size_t n, uint32_t seed) {
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        p[i] = (uint8_t)((i * 2654435761u + seed * 2246822519u) >> 13);
    }
}

// Sum only visible pixels, so a padded decoded plane and a tight encoder
// plane produce the same checksum.
__global__ void checksum_plane_kernel(const uint8_t *p, int pitch, int width,
                                      int height, unsigned long long *out) {
    __shared__ unsigned long long s[256];
    unsigned long long acc = 0;
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height;
         y += gridDim.y * blockDim.y) {
        const uint8_t *row = p + (size_t)y * pitch;
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width;
             x += gridDim.x * blockDim.x) {
            acc += row[x];
        }
    }
    s[threadIdx.y * blockDim.x + threadIdx.x] = acc;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (threadIdx.x < step) {
            s[threadIdx.y * blockDim.x + threadIdx.x] +=
                s[threadIdx.y * blockDim.x + threadIdx.x + step];
        }
        __syncthreads();
    }
    // One partial per thread row; reduce those into the global counter.
    if (threadIdx.x == 0) {
        unsigned long long rowsum = 0;
        for (int ty = 0; ty < blockDim.y; ++ty)
            rowsum += s[ty * blockDim.x];
        atomicAdd(out, rowsum);
    }
}

__global__ void copy_plane_kernel(uint8_t *dst, const uint8_t *src, int width,
                                  int height, int dst_pitch, int src_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        dst[(size_t)y * dst_pitch + x] = src[(size_t)y * src_pitch + x];
    }
}

// ---------------------------------------------------------------------------
// Per-frame measurement
// ---------------------------------------------------------------------------

struct FrameStats {
    unsigned long long luma = 0;
    unsigned long long chroma = 0;
    unsigned long long handoff_luma = 0;
    unsigned long long handoff_chroma = 0;
    double kernel_ms = 0.0;
    double copy_ms = 0.0;
    bool handoff_match = false;
};

struct BenchState {
    cudaStream_t stream = nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    unsigned long long *d_sum = nullptr;       // [0]=luma, [1]=chroma
    unsigned long long h_sum[2] = {0, 0};
    uint8_t *d_luma = nullptr;                 // encoder-owned tight planes
    uint8_t *d_chroma = nullptr;
    size_t luma_bytes = 0, chroma_bytes = 0;
    int width = 0, height = 0;
    bool do_copy = false;
    unsigned long long host_to_device_bytes = 0;
    unsigned long long device_to_host_bytes = 0;
    double total_kernel_ms = 0.0;
    double total_copy_ms = 0.0;
};

static void bench_init(BenchState &st, int width, int height, bool do_copy) {
    st.width = width;
    st.height = height;
    st.luma_bytes = (size_t)width * height;
    st.chroma_bytes = (size_t)width * (height / 2);
    st.do_copy = do_copy;
    RT_CHECK(cudaMalloc(&st.d_sum, 2 * sizeof(unsigned long long)));
    if (do_copy) {
        RT_CHECK(cudaMalloc(&st.d_luma, st.luma_bytes));
        RT_CHECK(cudaMalloc(&st.d_chroma, st.chroma_bytes));
    }
    RT_CHECK(cudaEventCreate(&st.ev0));
    RT_CHECK(cudaEventCreate(&st.ev1));
}

static void bench_free(BenchState &st) {
    if (st.ev0) cudaEventDestroy(st.ev0);
    if (st.ev1) cudaEventDestroy(st.ev1);
    if (st.d_sum) cudaFree(st.d_sum);
    if (st.d_luma) cudaFree(st.d_luma);
    if (st.d_chroma) cudaFree(st.d_chroma);
}

static FrameStats bench_frame(BenchState &st, uint8_t *luma, int luma_pitch,
                              uint8_t *chroma, int chroma_pitch) {
    FrameStats fs;
    dim3 block(32, 8);
    dim3 grid((st.width + block.x - 1) / block.x,
              (st.height + block.y - 1) / block.y);
    dim3 cgrid((st.width + block.x - 1) / block.x,
               (st.height / 2 + block.y - 1) / block.y);

    RT_CHECK(cudaMemsetAsync(st.d_sum, 0, 2 * sizeof(unsigned long long), st.stream));

    RT_CHECK(cudaEventRecord(st.ev0, st.stream));
    checksum_plane_kernel<<<grid, block, 0, st.stream>>>(
        luma, luma_pitch, st.width, st.height, &st.d_sum[0]);
    checksum_plane_kernel<<<cgrid, block, 0, st.stream>>>(
        chroma, chroma_pitch, st.width, st.height / 2, &st.d_sum[1]);
    RT_CHECK(cudaEventRecord(st.ev1, st.stream));
    RT_CHECK(cudaEventSynchronize(st.ev1));
    float ms = 0.0f;
    RT_CHECK(cudaEventElapsedTime(&ms, st.ev0, st.ev1));
    fs.kernel_ms = ms;
    st.total_kernel_ms += ms;

    RT_CHECK(cudaMemcpyAsync(st.h_sum, st.d_sum, 2 * sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost, st.stream));
    RT_CHECK(cudaStreamSynchronize(st.stream));
    st.device_to_host_bytes += 2 * sizeof(unsigned long long);
    fs.luma = st.h_sum[0];
    fs.chroma = st.h_sum[1];

    if (st.do_copy) {
        RT_CHECK(cudaMemsetAsync(st.d_sum, 0, 2 * sizeof(unsigned long long),
                                 st.stream));
        RT_CHECK(cudaEventRecord(st.ev0, st.stream));
        copy_plane_kernel<<<grid, block, 0, st.stream>>>(
            st.d_luma, luma, st.width, st.height, st.width, luma_pitch);
        copy_plane_kernel<<<cgrid, block, 0, st.stream>>>(
            st.d_chroma, chroma, st.width, st.height / 2, st.width, chroma_pitch);
        checksum_plane_kernel<<<grid, block, 0, st.stream>>>(
            st.d_luma, st.width, st.width, st.height, &st.d_sum[0]);
        checksum_plane_kernel<<<cgrid, block, 0, st.stream>>>(
            st.d_chroma, st.width, st.width, st.height / 2, &st.d_sum[1]);
        RT_CHECK(cudaEventRecord(st.ev1, st.stream));
        RT_CHECK(cudaEventSynchronize(st.ev1));
        RT_CHECK(cudaEventElapsedTime(&ms, st.ev0, st.ev1));
        fs.copy_ms = ms;
        st.total_copy_ms += ms;

        RT_CHECK(cudaMemcpyAsync(st.h_sum, st.d_sum,
                                 2 * sizeof(unsigned long long),
                                 cudaMemcpyDeviceToHost, st.stream));
        RT_CHECK(cudaStreamSynchronize(st.stream));
        st.device_to_host_bytes += 2 * sizeof(unsigned long long);
        fs.handoff_luma = st.h_sum[0];
        fs.handoff_chroma = st.h_sum[1];
        fs.handoff_match =
            fs.luma == fs.handoff_luma && fs.chroma == fs.handoff_chroma;
    }
    return fs;
}

// ---------------------------------------------------------------------------
// Synthetic AVHWFramesContext path
// ---------------------------------------------------------------------------

static int run_synthetic(AVBufferRef *device_ref, int frames, bool do_copy) {
    const int width = 3840, height = 2160;

    AVBufferRef *frames_ref = av_hwframe_ctx_alloc(device_ref);
    if (!frames_ref) {
        std::fprintf(stderr, "av_hwframe_ctx_alloc failed\n");
        return 1;
    }
    AVHWFramesContext *fc = (AVHWFramesContext *)frames_ref->data;
    fc->format = AV_PIX_FMT_CUDA;
    fc->sw_format = AV_PIX_FMT_NV12;
    fc->width = width;
    fc->height = height;
    fc->initial_pool_size = 4;
    int ret = av_hwframe_ctx_init(frames_ref);
    if (ret < 0) {
        char e[AV_ERROR_MAX_STRING_SIZE] = {0};
        av_strerror(ret, e, sizeof(e));
        std::fprintf(stderr, "av_hwframe_ctx_init failed: %s\n", e);
        av_buffer_unref(&frames_ref);
        return 1;
    }
    std::printf("synthetic: AVHWFramesContext format=%s sw_format=%s %dx%d pool=%d\n",
                av_get_pix_fmt_name(fc->format),
                av_get_pix_fmt_name(fc->sw_format), fc->width, fc->height,
                fc->initial_pool_size);

    BenchState st;
    st.stream = nullptr;  // default stream for synthetic mode
    bench_init(st, width, height, do_copy);

    AVFrame *frame = av_frame_alloc();
    if (!frame) {
        std::fprintf(stderr, "av_frame_alloc failed\n");
        bench_free(st);
        av_buffer_unref(&frames_ref);
        return 1;
    }

    std::printf("synthetic: allocating %d frames from the AVHWFramesContext pool\n",
                frames);
    auto t0 = std::chrono::steady_clock::now();
    int ok = 0;
    for (int i = 0; i < frames; ++i) {
        ret = av_hwframe_get_buffer(frames_ref, frame, 0);
        if (ret < 0) {
            char e[AV_ERROR_MAX_STRING_SIZE] = {0};
            av_strerror(ret, e, sizeof(e));
            std::fprintf(stderr, "av_hwframe_get_buffer(%d) failed: %s\n", i, e);
            break;
        }
        if (frame->format != AV_PIX_FMT_CUDA || !frame->data[0]) {
            std::fprintf(stderr, "frame %d is not CUDA: format=%d data0=%p\n", i,
                         frame->format, (void *)frame->data[0]);
            break;
        }
        if (i == 0) {
            std::printf("frame 0: format=%s %dx%d linesize=%d/%d data0=%p data1=%p\n",
                        av_get_pix_fmt_name((AVPixelFormat)frame->format),
                        frame->width, frame->height, frame->linesize[0],
                        frame->linesize[1], (void *)frame->data[0],
                        (void *)frame->data[1]);
            print_pointer_kind("luma", frame->data[0]);
            print_pointer_kind("chroma", frame->data[1]);
            if (frame->hw_frames_ctx) {
                AVHWFramesContext *fc2 =
                    (AVHWFramesContext *)frame->hw_frames_ctx->data;
                std::printf("frame hw_frames_ctx: format=%s sw_format=%s %dx%d\n",
                            av_get_pix_fmt_name(fc2->format),
                            av_get_pix_fmt_name(fc2->sw_format), fc2->width,
                            fc2->height);
            }
        }

        uint8_t *luma = (uint8_t *)frame->data[0];
        uint8_t *chroma = (uint8_t *)frame->data[1];
        int lp = frame->linesize[0], cp = frame->linesize[1];

        // Fill the device frame without touching host memory.
        fill_kernel<<<1024, 256, 0, st.stream>>>(luma,
                                                 (size_t)lp * height, 1000 + i);
        fill_kernel<<<1024, 256, 0, st.stream>>>(chroma,
                                                 (size_t)cp * (height / 2),
                                                 2000 + i);
        RT_CHECK(cudaStreamSynchronize(st.stream));

        FrameStats fs = bench_frame(st, luma, lp, chroma, cp);
        std::printf("frame %2d: checksum luma=0x%016llx chroma=0x%016llx "
                    "kernel=%.3f ms",
                    i, fs.luma, fs.chroma, fs.kernel_ms);
        if (do_copy) {
            std::printf(" copy_to_encoder_pool=%.3f ms match=%s", fs.copy_ms,
                        fs.handoff_match ? "yes" : "NO");
        }
        std::printf("\n");
        ++ok;
        av_frame_unref(frame);
    }
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    std::printf("synthetic: %d frames in %.3f s (%.2f fps), "
                "avg kernel %.3f ms, avg copy %.3f ms\n",
                ok, sec, sec > 0 ? ok / sec : 0.0,
                ok ? st.total_kernel_ms / ok : 0.0,
                ok ? st.total_copy_ms / ok : 0.0);
    std::printf("synthetic: host->device frame bytes=%llu, "
                "device->host bytes=%llu (checksums only)\n",
                st.host_to_device_bytes, st.device_to_host_bytes);

    av_frame_free(&frame);
    bench_free(st);
    av_buffer_unref(&frames_ref);
    return ok == frames ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Real NVDEC path
// ---------------------------------------------------------------------------

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx,
                                        const enum AVPixelFormat *pix_fmts) {
    (void)ctx;
    for (const enum AVPixelFormat *p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == AV_PIX_FMT_CUDA) return *p;
    }
    return AV_PIX_FMT_NONE;
}

static int run_nvdec(AVBufferRef *device_ref, const char *input, int max_frames,
                     bool do_copy) {
    AVFormatContext *fmt = nullptr;
    int ret = avformat_open_input(&fmt, input, nullptr, nullptr);
    if (ret < 0) {
        char e[AV_ERROR_MAX_STRING_SIZE] = {0};
        av_strerror(ret, e, sizeof(e));
        std::fprintf(stderr, "avformat_open_input(%s) failed: %s\n", input, e);
        return 1;
    }
    ret = avformat_find_stream_info(fmt, nullptr);
    if (ret < 0) {
        std::fprintf(stderr, "avformat_find_stream_info failed\n");
        avformat_close_input(&fmt);
        return 1;
    }
    const AVCodec *decoder = nullptr;
    int vstream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (vstream < 0 || !decoder) {
        std::fprintf(stderr, "no video stream / decoder\n");
        avformat_close_input(&fmt);
        return 1;
    }
    AVStream *vs = fmt->streams[vstream];
    std::printf("input: %s, codec=%s, %dx%d, %s\n", fmt->iformat->name,
                decoder->name, vs->codecpar->width, vs->codecpar->height,
                av_get_pix_fmt_name((AVPixelFormat)vs->codecpar->format));

    AVCodecContext *dec = avcodec_alloc_context3(decoder);
    if (!dec) {
        avformat_close_input(&fmt);
        return 1;
    }
    ret = avcodec_parameters_to_context(dec, vs->codecpar);
    if (ret < 0) {
        std::fprintf(stderr, "avcodec_parameters_to_context failed\n");
        avcodec_free_context(&dec);
        avformat_close_input(&fmt);
        return 1;
    }
    dec->hw_device_ctx = av_buffer_ref(device_ref);
    dec->get_format = get_hw_format;
    ret = avcodec_open2(dec, decoder, nullptr);
    if (ret < 0) {
        char e[AV_ERROR_MAX_STRING_SIZE] = {0};
        av_strerror(ret, e, sizeof(e));
        std::fprintf(stderr, "NVDEC decoder open failed: %s (%d)\n", e, ret);
        std::fprintf(stderr,
                     "This is the failure signature of the NVIDIA 570.x/575.x/"
                     "580.x container regression when a container sees a subset "
                     "of a multi-GPU host. The 3060 NVDEC hardware itself is "
                     "capable of VP9 Profile 0 8-bit decode. See docs/nvdec.md.\n");
        avcodec_free_context(&dec);
        avformat_close_input(&fmt);
        return 2;
    }
    if (dec->pix_fmt != AV_PIX_FMT_CUDA) {
        std::fprintf(stderr,
                     "NVDEC unavailable: decoder opened with %s, not cuda. "
                     "FFmpeg silently fell back to software after the CUDA "
                     "hwaccel failed.\n",
                     av_get_pix_fmt_name(dec->pix_fmt));
        std::fprintf(stderr,
                     "This is the failure signature of the NVIDIA 570.x/575.x/"
                     "580.x container regression when a container sees a subset "
                     "of a multi-GPU host. The 3060 NVDEC hardware itself is "
                     "capable of VP9 Profile 0 8-bit decode. See docs/nvdec.md.\n");
        avcodec_free_context(&dec);
        avformat_close_input(&fmt);
        return 2;
    }
    std::printf("NVDEC decoder opened: %s, pix_fmt=%s, %dx%d\n", decoder->name,
                av_get_pix_fmt_name(dec->pix_fmt), dec->width, dec->height);

    BenchState st;
    AVHWDeviceContext *devctx = (AVHWDeviceContext *)device_ref->data;
    AVCUDADeviceContext *cudactx = (AVCUDADeviceContext *)devctx->hwctx;
    st.stream = (cudaStream_t)cudactx->stream;
    bench_init(st, dec->width, dec->height, do_copy);

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    if (!pkt || !frame) {
        std::fprintf(stderr, "alloc packet/frame failed\n");
        av_packet_free(&pkt);
        av_frame_free(&frame);
        bench_free(st);
        avcodec_free_context(&dec);
        avformat_close_input(&fmt);
        return 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    int decoded = 0;
    bool saw_cuda = false;
    bool printed = false;

    auto handle = [&](AVFrame *f) {
        if (f->format != AV_PIX_FMT_CUDA) {
            std::fprintf(stderr, "decoded frame is %s, not CUDA\n",
                         av_get_pix_fmt_name((AVPixelFormat)f->format));
            return false;
        }
        saw_cuda = true;
        if (!f->hw_frames_ctx || !f->data[0] || !f->data[1]) {
            std::fprintf(stderr,
                         "CUDA frame is missing hw_frames_ctx or plane pointers "
                         "(hw_frames_ctx=%p data0=%p data1=%p)\n",
                         (void *)f->hw_frames_ctx, (void *)f->data[0],
                         (void *)f->data[1]);
            return false;
        }
        if (!printed) {
            AVHWFramesContext *fc = (AVHWFramesContext *)f->hw_frames_ctx->data;
            std::printf("first CUDA frame: format=%s sw_format=%s %dx%d "
                        "linesize=%d/%d data0=%p data1=%p\n",
                        av_get_pix_fmt_name((AVPixelFormat)f->format),
                        av_get_pix_fmt_name(fc->sw_format), f->width, f->height,
                        f->linesize[0], f->linesize[1], (void *)f->data[0],
                        (void *)f->data[1]);
            print_pointer_kind("luma", f->data[0]);
            print_pointer_kind("chroma", f->data[1]);
            print_pointer_kind("decoder-owned surface", f->buf[0] ? f->buf[0]->data : nullptr);
            printed = true;
        }
        FrameStats fs = bench_frame(st, (uint8_t *)f->data[0], f->linesize[0],
                                    (uint8_t *)f->data[1], f->linesize[1]);
        if (decoded < 3 || decoded % 120 == 0) {
            std::printf("frame %4d: checksum luma=0x%016llx chroma=0x%016llx "
                        "kernel=%.3f ms%s\n",
                        decoded, fs.luma, fs.chroma, fs.kernel_ms,
                        do_copy ? (fs.handoff_match ? " copy=match"
                                                    : " copy=NO-MATCH")
                                : "");
        }
        ++decoded;
        return true;
    };

    while (true) {
        if (max_frames > 0 && decoded >= max_frames) break;
        ret = av_read_frame(fmt, pkt);
        if (ret < 0) break;
        if (pkt->stream_index == vstream) {
            ret = avcodec_send_packet(dec, pkt);
            if (ret < 0 && ret != AVERROR(EAGAIN)) {
                std::fprintf(stderr, "avcodec_send_packet failed\n");
                break;
            }
            while (true) {
                ret = avcodec_receive_frame(dec, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) {
                    std::fprintf(stderr, "avcodec_receive_frame failed\n");
                    break;
                }
                handle(frame);
                av_frame_unref(frame);
                if (max_frames > 0 && decoded >= max_frames) break;
            }
        }
        av_packet_unref(pkt);
    }
    if (max_frames <= 0 || decoded < max_frames) {
        avcodec_send_packet(dec, nullptr);
        while (avcodec_receive_frame(dec, frame) == 0) {
            handle(frame);
            av_frame_unref(frame);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    std::printf("NVDEC: decoded %d frames in %.3f s (%.2f fps), "
                "avg kernel %.3f ms, avg copy %.3f ms\n",
                decoded, sec, sec > 0 ? decoded / sec : 0.0,
                decoded ? st.total_kernel_ms / decoded : 0.0,
                decoded ? st.total_copy_ms / decoded : 0.0);
    std::printf("NVDEC: host->device frame bytes=%llu, "
                "device->host bytes=%llu (checksums only)\n",
                st.host_to_device_bytes, st.device_to_host_bytes);
    std::printf("NVDEC: all frames stayed on device=%s\n",
                saw_cuda ? "yes" : "no");

    av_packet_free(&pkt);
    av_frame_free(&frame);
    bench_free(st);
    avcodec_free_context(&dec);
    avformat_close_input(&fmt);
    return decoded > 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    const char *input = nullptr;
    bool synthetic = false;
    bool do_copy = false;
    int frames = 5;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--input") && i + 1 < argc) {
            input = argv[++i];
        } else if (!std::strcmp(argv[i], "--synthetic")) {
            synthetic = true;
        } else if (!std::strcmp(argv[i], "--copy")) {
            do_copy = true;
        } else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) {
            frames = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--help")) {
            std::printf("usage: %s --input clip.webm [--frames N] [--copy]\n"
                        "       %s --synthetic [--frames N] [--copy]\n",
                        argv[0], argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 1;
        }
    }
    if (!input && !synthetic) {
        std::fprintf(stderr, "need --input <clip> or --synthetic\n");
        return 1;
    }

    print_device_info();

    AVBufferRef *device_ref = nullptr;
    int ret = av_hwdevice_ctx_create(&device_ref, AV_HWDEVICE_TYPE_CUDA, nullptr,
                                     nullptr, 0);
    if (ret < 0) {
        char e[AV_ERROR_MAX_STRING_SIZE] = {0};
        av_strerror(ret, e, sizeof(e));
        std::fprintf(stderr, "av_hwdevice_ctx_create(CUDA) failed: %s\n", e);
        return 1;
    }
    AVHWDeviceContext *devctx = (AVHWDeviceContext *)device_ref->data;
    AVCUDADeviceContext *cudactx = (AVCUDADeviceContext *)devctx->hwctx;
    std::printf("FFmpeg CUDA hwdevice: ctx=%p stream=%p\n",
                (void *)cudactx->cuda_ctx, (void *)cudactx->stream);
    // Make FFmpeg's context current before allocating the encoder arena or
    // launching kernels. Do not rely on the runtime API's primary context:
    // the decoder surfaces belong to this context.
    CU_CHECK(cuCtxSetCurrent(cudactx->cuda_ctx));
    CUcontext current_ctx = nullptr;
    CU_CHECK(cuCtxGetCurrent(&current_ctx));
    std::printf("CUDA current context=%p, ffmpeg=%p: %s\n",
                (void *)current_ctx, (void *)cudactx->cuda_ctx,
                current_ctx == cudactx->cuda_ctx
                    ? "same context (arena/frames shareable)"
                    : "DIFFERENT contexts (fix before allocating an arena)");

    int rc;
    if (synthetic) {
        rc = run_synthetic(device_ref, frames, do_copy);
    } else {
        rc = run_nvdec(device_ref, input, frames, do_copy);
    }
    av_buffer_unref(&device_ref);
    return rc;
}
