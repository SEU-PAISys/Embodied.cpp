cmake_minimum_required(VERSION 3.22)
project(vla_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp)

add_library(vla_core
    runtime/model.cpp
    models/smolvla.cpp
    models/pi0.cpp
    models/pi05.cpp
    models/evo1.cpp
    models/gr00tn1d5.cpp
    models/gr00tn1d6.cpp
    models/gr00tn1d7.cpp
    models/bitvla.cpp
    models/lingbot_va.cpp
    models/hy_vla.cpp
)
target_include_directories(vla_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor
)
target_link_libraries(vla_core PUBLIC llama ggml mtmd)

if(GGML_CUDA)
    target_compile_definitions(vla_core PUBLIC GGML_USE_CUDA)

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90 100 120 CACHE STRING "" FORCE)
    endif()
    add_library(bitvla_cuda_kernels STATIC
        kernels/bitvla/bitnet_kernels.cu
        kernels/bitvla/bitvla_lm_cuda.cu
        kernels/bitvla/bitvla_vit_cuda.cu
        kernels/bitvla/bitvla_fp32head_cuda.cu
        kernels/lingbot/lingbot_flex_attn_cuda.cu
    )
    set_target_properties(bitvla_cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_features(bitvla_cuda_kernels PRIVATE cxx_std_17)
    target_compile_options(bitvla_cuda_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xptxas=-O3>
    )
    target_link_libraries(bitvla_cuda_kernels PUBLIC CUDA::cublas CUDA::cudart)
    target_link_libraries(vla_core PRIVATE bitvla_cuda_kernels)
    target_compile_definitions(vla_core PUBLIC VLA_BITVLA_CUDA_KERNELS)
    target_compile_definitions(vla_core PUBLIC VLA_LINGBOT_FLEX_CUDA_KERNELS)
    target_include_directories(vla_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

add_library(vlm_core
    vlm/engine.cpp
)
target_include_directories(vlm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/common
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor
)
target_link_libraries(vlm_core PUBLIC llama ggml mtmd llama-common)

find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZeroMQ REQUIRED IMPORTED_TARGET libzmq)

set(VLA_PROTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/serving/vla.proto)
set(VLA_PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-gen/serving)
file(MAKE_DIRECTORY ${VLA_PROTO_GEN_DIR})

add_custom_command(
    OUTPUT  ${VLA_PROTO_GEN_DIR}/vla.pb.h
            ${VLA_PROTO_GEN_DIR}/vla.pb.cc
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/serving
            --cpp_out=${VLA_PROTO_GEN_DIR}
            ${VLA_PROTO_SRC}
    DEPENDS ${VLA_PROTO_SRC}
    VERBATIM
)

add_executable(vla-server
    serving/server.cpp
    ${VLA_PROTO_GEN_DIR}/vla.pb.cc
)
target_include_directories(vla-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_BINARY_DIR}/proto-gen
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor/stb
    ${Protobuf_INCLUDE_DIR}
)
target_link_libraries(vla-server PRIVATE
    vla_core
    ${Protobuf_LIBRARIES}
    PkgConfig::ZeroMQ
)

add_executable(hy-vla-direct-debug
    tools/hy_vla_direct_debug.cpp
)
target_include_directories(hy-vla-direct-debug PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
)
target_link_libraries(hy-vla-direct-debug PRIVATE
    vla_core
)

set(LINGBOT_PROTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/serving/lingbot.proto)
set(LINGBOT_PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-gen/serving)
file(MAKE_DIRECTORY ${LINGBOT_PROTO_GEN_DIR})

add_custom_command(
    OUTPUT  ${LINGBOT_PROTO_GEN_DIR}/lingbot.pb.h
            ${LINGBOT_PROTO_GEN_DIR}/lingbot.pb.cc
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/serving
            --cpp_out=${LINGBOT_PROTO_GEN_DIR}
            ${LINGBOT_PROTO_SRC}
    DEPENDS ${LINGBOT_PROTO_SRC}
    VERBATIM
)

add_executable(lingbot-world-server
    serving/lingbot-world-server.cpp
    ${LINGBOT_PROTO_GEN_DIR}/lingbot.pb.cc
)
target_include_directories(lingbot-world-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_BINARY_DIR}/proto-gen
    ${Protobuf_INCLUDE_DIR}
)
target_link_libraries(lingbot-world-server PRIVATE
    vla_core
    ${Protobuf_LIBRARIES}
    PkgConfig::ZeroMQ
)

set(VLM_PROTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/serving/vlm.proto)
set(VLM_PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-gen/serving)
file(MAKE_DIRECTORY ${VLM_PROTO_GEN_DIR})

add_custom_command(
    OUTPUT  ${VLM_PROTO_GEN_DIR}/vlm.pb.h
            ${VLM_PROTO_GEN_DIR}/vlm.pb.cc
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/serving
            --cpp_out=${VLM_PROTO_GEN_DIR}
            ${VLM_PROTO_SRC}
    DEPENDS ${VLM_PROTO_SRC}
    VERBATIM
)

add_executable(vlm-server
    serving/vlm-server.cpp
    ${VLM_PROTO_GEN_DIR}/vlm.pb.cc
)
target_include_directories(vlm-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_BINARY_DIR}/proto-gen
    ${Protobuf_INCLUDE_DIR}
)
target_link_libraries(vlm-server PRIVATE
    vlm_core
    ${Protobuf_LIBRARIES}
    PkgConfig::ZeroMQ
)
