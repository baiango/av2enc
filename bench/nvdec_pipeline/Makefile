# NVDEC feasibility demo. Linux + CUDA only; this is not built by the root
# CMakeLists (bench components are standalone).

NVCC      ?= nvcc
ARCH      ?= sm_86
CXXSTD    ?= c++17
FF_CFLAGS := $(shell pkg-config --cflags libavformat libavcodec libavutil)
FF_LIBS   := $(shell pkg-config --libs libavformat libavcodec libavutil)

nvdec_probe: nvdec_probe.cu
	$(NVCC) -O2 -std=$(CXXSTD) -arch=$(ARCH) $< -o $@ \
	  $(FF_CFLAGS) $(FF_LIBS) -lcuda -lcudart

clean:
	rm -f nvdec_probe

.PHONY: clean
