cmake_minimum_required(VERSION 3.16)
project(tq3-ascend-port VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Options ──────────────────────────────────────────────────────────
option(BUILD_TESTS "Build test executables" ON)

# ─── TQ3 dequant library (CPU dequant functions, no CANN dependency) ──
add_library(tq3_dequant STATIC
    ggml-cann-integration/tq3_cann_kernels.h
)

target_include_directories(tq3_dequant PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml-cann-integration
)

# ─── Tests ────────────────────────────────────────────────────────────
if(BUILD_TESTS)
    find_package(Threads REQUIRED)

    # Test 1: CPU dequant standalone (no CANN needed)
    add_executable(test_tq3_dequant
        tests/test_tq3_dequant.cpp
    )
    target_link_libraries(test_tq3_dequant PRIVATE tq3_dequant m)

    # Test 2: CANN integration (requires llama.cpp-tq3 + CANN)
    # This target is only built if llama.cpp-tq3 paths are provided.
    if(DEFINED LLAMA_CPP_TQ3_DIR AND DEFINED CANN_INSTALL_DIR)
        add_executable(test_tq3_cann_integration
            tests/test_tq3_cann_integration.cpp
        )
        target_include_directories(test_tq3_cann_integration PRIVATE
            ${LLAMA_CPP_TQ3_DIR}/ggml/include
            ${LLAMA_CPP_TQ3_DIR}/ggml/src
            ${LLAMA_CPP_TQ3_DIR}/ggml/src/ggml-cann
            ${CMAKE_CURRENT_SOURCE_DIR}/ggml-cann-integration
            ${LLAMA_CPP_TQ3_DIR}/build-cann/ggml/src
            ${LLAMA_CPP_TQ3_DIR}/build-cann/ggml/src/ggml-cann
            ${CANN_INSTALL_DIR}/include
            ${CANN_INSTALL_DIR}/include/aclnn
        )
        target_link_directories(test_tq3_cann_integration PRIVATE
            ${LLAMA_CPP_TQ3_DIR}/build-cann/ggml/src/ggml-cann
            ${LLAMA_CPP_TQ3_DIR}/build-cann/ggml/src
            ${CANN_INSTALL_DIR}/lib64
        )
        target_link_libraries(test_tq3_cann_integration PRIVATE
            tq3_dequant
            ggml-cann ggml ggml-base ggml-cpu
            ascendcl nnopbase opapi acl_op_compiler
            Threads::Threads dl rt m
        )
    endif()
endif()
