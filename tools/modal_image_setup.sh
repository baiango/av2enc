#!/bin/sh
# Image setup for tools/modal_rd.py: the two metrics libjxl builds and the
# pinned dav2d decoder. Kept as a file because Modal parses run_commands
# strings as Dockerfile lines, where a multi-line shell block is invalid.
set -eu

# libjxl tools are only needed when the image scores in the cloud. The normal
# split encodes on Modal and scores on the Mac with the pinned local tools.
if [ "${WITH_METRICS:-0}" = "1" ] && \
   { ! command -v butteraugli_main >/dev/null 2>&1 || \
     ! command -v ssimulacra2 >/dev/null 2>&1; }; then
    git clone --depth 1 --branch v0.11.1 --recursive \
        https://github.com/libjxl/libjxl.git /opt/libjxl
    cmake -S /opt/libjxl -B /opt/libjxl/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DJPEGXL_ENABLE_BENCHMARK=OFF \
        -DJPEGXL_ENABLE_DEVTOOLS=ON \
        -DJPEGXL_ENABLE_EXAMPLES=OFF \
        -DJPEGXL_ENABLE_MANPAGES=OFF \
        -DJPEGXL_ENABLE_OPENEXR=OFF \
        -DJPEGXL_ENABLE_SJPEG=OFF
    cmake --build /opt/libjxl/build \
        --target butteraugli_main ssimulacra2 -j"$(nproc)"
    install -m0755 /opt/libjxl/build/tools/butteraugli_main /usr/local/bin/
    install -m0755 /opt/libjxl/build/tools/ssimulacra2 /usr/local/bin/
fi

# Pinned decoder, static, matching tools/build_dav2d_linux.sh.
git clone https://code.videolan.org/videolan/dav2d.git /opt/dav2d
git -C /opt/dav2d checkout f9dafbc747e82a9c51ca648bf2108d9fa0f8de7d
meson setup /opt/dav2d/build-linux-static /opt/dav2d --buildtype release \
    -Ddefault_library=static \
    -Denable_tools=true -Denable_tests=false \
    -Denable_examples=false -Denable_docs=false
ninja -C /opt/dav2d/build-linux-static
install -m0755 /opt/dav2d/build-linux-static/tools/dav2d /usr/local/bin/dav2d

if [ "${WITH_METRICS:-0}" = "1" ]; then command -v butteraugli_main ssimulacra2 ffmpeg; fi
command -v dav2d
dav2d --version | head -1
