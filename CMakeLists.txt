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
    models/pi05.cpp
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

add_library(embodied_adapter
    adapter/adapter.cpp
)
target_include_directories(embodied_adapter
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime
)
target_link_libraries(embodied_adapter PUBLIC vla_core)

if(GGML_CUDA)
    target_compile_definitions(vla_core PUBLIC GGML_USE_CUDA)

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90 100 120 CACHE STRING "" FORCE)
    endif()
    add_library(lingbot_cuda_kernels STATIC
        kernels/lingbot/lingbot_flex_attn_cuda.cu
    )
    set_target_properties(lingbot_cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_features(lingbot_cuda_kernels PRIVATE cxx_std_17)
    target_compile_options(lingbot_cuda_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xptxas=-O3>
    )
    target_link_libraries(lingbot_cuda_kernels PUBLIC CUDA::cublas CUDA::cudart)
    target_link_libraries(vla_core PRIVATE lingbot_cuda_kernels)
    target_compile_definitions(vla_core PUBLIC VLA_LINGBOT_FLEX_CUDA_KERNELS)
    target_include_directories(vla_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

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
    embodied_adapter
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
