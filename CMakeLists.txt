cmake_minimum_required(VERSION 3.24)

project(av2enc
  VERSION 0.1.0
  DESCRIPTION "From-scratch AV2 encoder"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA is optional. Keep the option OFF by default; detect nvcc so the build
# still works on macOS and other CPU-only hosts.
option(AV2ENC_ENABLE_CUDA "Build the optional CUDA backend (requires nvcc)" OFF)

include(CheckLanguage)
check_language(CUDA)
set(AV2ENC_CUDA_AVAILABLE OFF)
if(CMAKE_CUDA_COMPILER)
  set(AV2ENC_CUDA_AVAILABLE ON)
endif()

set(AV2ENC_HAVE_CUDA OFF)
if(AV2ENC_ENABLE_CUDA)
  if(AV2ENC_CUDA_AVAILABLE)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
    enable_language(CUDA)
    set(AV2ENC_HAVE_CUDA ON)
    message(STATUS "av2enc: CUDA backend enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
  else()
    message(WARNING
      "av2enc: AV2ENC_ENABLE_CUDA=ON but no CUDA compiler was found; "
      "building the CPU-only encoder")
  endif()
else()
  message(STATUS "av2enc: CUDA backend disabled (AV2ENC_ENABLE_CUDA=OFF)")
endif()

function(av2enc_apply_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wshadow)
  endif()
endfunction()

# Host-side code shared by the CLI and the tests. GPU code will be added as a
# separate target once CUDA kernels exist.
add_library(av2enc_core STATIC
  src/io/y4m.cpp
  src/io/ivf.cpp
  src/bitstream/bitwriter.cpp
  src/bitstream/obu.cpp)
target_include_directories(av2enc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(av2enc_core PUBLIC cxx_std_20)
av2enc_apply_warnings(av2enc_core)

if(AV2ENC_HAVE_CUDA)
  target_compile_definitions(av2enc_core PUBLIC AV2ENC_HAVE_CUDA=1)
else()
  target_compile_definitions(av2enc_core PUBLIC AV2ENC_HAVE_CUDA=0)
endif()

add_executable(av2enc src/main.cpp)
target_link_libraries(av2enc PRIVATE av2enc_core)
av2enc_apply_warnings(av2enc)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
