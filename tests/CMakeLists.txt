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

# Pinned decoder smoke test. The wrapper exits 77 when no dav2d binary is
# available, which CTest reports as a clean skip.
add_test(NAME dav2d_smoke
  COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/dav2d_smoke.sh" "${PROJECT_SOURCE_DIR}")
set_tests_properties(dav2d_smoke PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 180)
