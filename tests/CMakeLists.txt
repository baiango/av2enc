add_executable(bitwriter_test bitwriter_test.cpp)
target_link_libraries(bitwriter_test PRIVATE av2enc_core)
av2enc_apply_warnings(bitwriter_test)
add_test(NAME bitwriter_test COMMAND bitwriter_test)

add_executable(ivf_test ivf_test.cpp)
target_link_libraries(ivf_test PRIVATE av2enc_core)
av2enc_apply_warnings(ivf_test)
add_test(NAME ivf_test COMMAND ivf_test)

add_executable(y4m_test y4m_test.cpp)
target_link_libraries(y4m_test PRIVATE av2enc_core)
av2enc_apply_warnings(y4m_test)
add_test(NAME y4m_test COMMAND y4m_test)

add_executable(basic_headers_test basic_headers_test.cpp)
target_link_libraries(basic_headers_test PRIVATE av2enc_core)
av2enc_apply_warnings(basic_headers_test)
add_test(NAME basic_headers_test COMMAND basic_headers_test)

# Deterministic parallel-for helper.  The executable is registered as the
# gate_parallel_for product gate (tests/gates) rather than a core CTest test,
# so the pinned ctest inventory in refactor/baseline.json stays unchanged.
add_executable(parallel_for_test parallel_for_test.cpp)
target_link_libraries(parallel_for_test PRIVATE av2enc_core)
av2enc_apply_warnings(parallel_for_test)

add_executable(still_headers_test still_headers_test.cpp)
target_link_libraries(still_headers_test PRIVATE av2enc_core)
av2enc_apply_warnings(still_headers_test)
add_test(NAME still_headers_test COMMAND still_headers_test)

# End-to-end first frame: encode a 64x64 still picture, require dav2d to decode
# it and match the pinned MD5. Exit 77 (skip) when no decoder is available.
add_test(NAME basic_encode_smoke
  COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/basic_encode_smoke.sh"
          $<TARGET_FILE:av2enc> "${PROJECT_SOURCE_DIR}")
set_tests_properties(basic_encode_smoke PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 180)

# Pinned decoder smoke test. The wrapper exits 77 when no dav2d binary is
# available, which CTest reports as a clean skip.
add_test(NAME dav2d_smoke
  COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/dav2d_smoke.sh" "${PROJECT_SOURCE_DIR}")
set_tests_properties(dav2d_smoke PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 180)

# Deterministic Y4M generator for edge-geometry tests (no ffmpeg dependency).
add_executable(gen_y4m gen_y4m.cpp)
av2enc_apply_warnings(gen_y4m)

# --cpu-threads identity is tests/multithread_identity.sh, registered as the
# gate_multithread_identity product gate in tests/gates/ (see the note above:
# the pinned ctest inventory must not gain a core test).

# Edge superblock decode gate: the five reported failing sizes plus the pinned
# 64x64 / 640x360 / 1080p regressions. M0 must handle every size it claims to
# support; 136x88 and 518x322 must fall back to M1 with a clear note.
add_test(NAME basic_edge_smoke
  COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/basic_edge_smoke.sh"
          $<TARGET_FILE:av2enc> $<TARGET_FILE:gen_y4m>
          "${PROJECT_SOURCE_DIR}")
set_tests_properties(basic_edge_smoke PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 600)

# Product gates (wave 4): additive test surface, see tests/gates/CMakeLists.txt.
# Product gates encode real streams (1080p loop restoration, VBR video) and
# are not unit tests. Register them only when a build explicitly asks for the
# full gate inventory; the default `ctest` stays fast.
option(AV2ENC_PRODUCT_GATES "Register the product gates as CTest tests" OFF)
if(AV2ENC_PRODUCT_GATES)
  add_subdirectory(gates)
endif()
