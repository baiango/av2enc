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

add_executable(m0_headers_test m0_headers_test.cpp)
target_link_libraries(m0_headers_test PRIVATE av2enc_core)
av2enc_apply_warnings(m0_headers_test)
add_test(NAME m0_headers_test COMMAND m0_headers_test)

add_executable(m1_headers_test m1_headers_test.cpp)
target_link_libraries(m1_headers_test PRIVATE av2enc_core)
av2enc_apply_warnings(m1_headers_test)
add_test(NAME m1_headers_test COMMAND m1_headers_test)

# End-to-end first frame: encode a 64x64 still picture, require dav2d to decode
# it and match the pinned MD5. Exit 77 (skip) when no decoder is available.
add_test(NAME m0_encode_smoke
  COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/m0_encode_smoke.sh"
          $<TARGET_FILE:av2enc> "${PROJECT_SOURCE_DIR}")
set_tests_properties(m0_encode_smoke PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 180)

# Pinned decoder smoke test. The wrapper exits 77 when no dav2d binary is
# available, which CTest reports as a clean skip.
add_test(NAME dav2d_smoke
  COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/dav2d_smoke.sh" "${PROJECT_SOURCE_DIR}")
set_tests_properties(dav2d_smoke PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 180)
