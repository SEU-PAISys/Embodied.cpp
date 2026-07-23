cmake_minimum_required(VERSION 3.22)
project(vla_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MODEL_BUILD_VLA_PI05 "Build pi0.5 runtime support" OFF)
option(MODEL_BUILD_VLA_HY_VLA "Build HY-VLA runtime support" OFF)
option(MODEL_BUILD_VLA_GROOT_N1 "Build NVIDIA GR00T N1 runtime support" OFF)
option(MODEL_BUILD_WAM_LINGBOT_VA "Build LingBot-VA runtime support" OFF)
option(MODEL_BUILD_WAM_COSMOS3 "Build Cosmos3 WAM runtime support" OFF)

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
)
if(MODEL_BUILD_VLA_PI05)
    target_sources(vla_core PRIVATE models/pi05.cpp)
endif()
if(MODEL_BUILD_VLA_HY_VLA)
    target_sources(vla_core PRIVATE models/hy_vla.cpp)
endif()
if(MODEL_BUILD_VLA_GROOT_N1)
    target_sources(vla_core PRIVATE models/groot_n1.cpp)
endif()
if(MODEL_BUILD_WAM_LINGBOT_VA)
    target_sources(vla_core PRIVATE models/lingbot_va.cpp)
endif()

if(MODEL_BUILD_WAM_COSMOS3)
    target_sources(vla_core PRIVATE models/cosmos3.cpp)
endif()
target_include_directories(vla_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor
)
target_compile_definitions(vla_core PUBLIC
    MODEL_BUILD_VLA_PI05=$<BOOL:${MODEL_BUILD_VLA_PI05}>
    MODEL_BUILD_VLA_HY_VLA=$<BOOL:${MODEL_BUILD_VLA_HY_VLA}>
    MODEL_BUILD_VLA_GROOT_N1=$<BOOL:${MODEL_BUILD_VLA_GROOT_N1}>
    MODEL_BUILD_WAM_LINGBOT_VA=$<BOOL:${MODEL_BUILD_WAM_LINGBOT_VA}>
    MODEL_BUILD_WAM_COSMOS3=$<BOOL:${MODEL_BUILD_WAM_COSMOS3}>
)
target_link_libraries(vla_core PUBLIC llama ggml mtmd)

set(EMBODIED_ENABLED_MODELS "")
if(MODEL_BUILD_VLA_PI05)
    list(APPEND EMBODIED_ENABLED_MODELS "VLA/pi0.5")
endif()
if(MODEL_BUILD_VLA_HY_VLA)
    list(APPEND EMBODIED_ENABLED_MODELS "VLA/HY-VLA")
endif()
if(MODEL_BUILD_VLA_GROOT_N1)
    list(APPEND EMBODIED_ENABLED_MODELS "VLA/GR00T-N1")
endif()
if(MODEL_BUILD_WAM_LINGBOT_VA)
    list(APPEND EMBODIED_ENABLED_MODELS "WAM/LingBot-VA")
endif()
if(MODEL_BUILD_WAM_COSMOS3)
    list(APPEND EMBODIED_ENABLED_MODELS "WAM/Cosmos3")
endif()
if(EMBODIED_ENABLED_MODELS)
    string(REPLACE ";" ", " EMBODIED_ENABLED_MODELS_STR "${EMBODIED_ENABLED_MODELS}")
    message(STATUS "Embodied.cpp model targets enabled: ${EMBODIED_ENABLED_MODELS_STR}")
else()
    message(WARNING
        "No model runtimes are enabled. Set one or more MODEL_BUILD_* options "
        "to build usable inference binaries.")
endif()

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
    if(MODEL_BUILD_WAM_LINGBOT_VA)
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
    if(MODEL_BUILD_WAM_COSMOS3)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90 CACHE STRING "" FORCE)
        endif()
        add_library(cosmos3_cuda_kernels STATIC
            kernels/cosmos3/cosmos3_action_bridge_cuda.cu
            kernels/cosmos3/cosmos3_mot_action_cuda.cu
            kernels/cosmos3/cosmos3_w8_marlin_cuda.cu
            kernels/cosmos3/third_party/vllm_marlin/marlin_raw.cu
            kernels/cosmos3/third_party/vllm_marlin/sm80_kernel_bfloat16_u8b128_bfloat16.cu
            kernels/cosmos3/cosmos3_qwen_language_cuda.cu
            kernels/cosmos3/cosmos3_qwen_visual_cuda.cu
            kernels/cosmos3/cosmos3_wan_vae_cuda.cu
        )
        set_target_properties(cosmos3_cuda_kernels PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
        )
        target_compile_features(cosmos3_cuda_kernels PRIVATE cxx_std_17)
        target_compile_options(cosmos3_cuda_kernels PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr -Xptxas=-O3>
        )
        target_include_directories(cosmos3_cuda_kernels PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cosmos3
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cosmos3/third_party/vllm_marlin
        )
        find_library(CUDNN_LIBRARY cudnn
            PATHS
                ${CUDAToolkit_LIBRARY_DIR}
                /usr/local/cuda/lib64
                /usr/local/cuda-11.6/targets/x86_64-linux/lib
            NO_DEFAULT_PATH)
        if(CUDNN_LIBRARY)
            target_compile_definitions(cosmos3_cuda_kernels PUBLIC VLA_COSMOS3_USE_CUDNN)
            target_link_libraries(cosmos3_cuda_kernels PUBLIC ${CUDNN_LIBRARY})
        endif()
        target_link_libraries(cosmos3_cuda_kernels PUBLIC CUDA::cublas CUDA::cudart)
        target_link_libraries(vla_core PRIVATE cosmos3_cuda_kernels)
        target_compile_definitions(vla_core PUBLIC VLA_COSMOS3_CUDA_KERNELS)
        target_include_directories(vla_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    endif()
    if(NOT MODEL_BUILD_WAM_LINGBOT_VA AND NOT MODEL_BUILD_WAM_COSMOS3)
        message(STATUS "GGML_CUDA is ON but no WAM CUDA kernels are enabled")
    endif()
endif()

find_program(EMBODIED_SYSTEM_PROTOC protoc
    PATHS /usr/bin /bin
    NO_DEFAULT_PATH
)
if(EMBODIED_SYSTEM_PROTOC)
    set(Protobuf_PROTOC_EXECUTABLE "${EMBODIED_SYSTEM_PROTOC}" CACHE FILEPATH
        "Protocol Buffers compiler used by Embodied.cpp" FORCE)
endif()
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZeroMQ REQUIRED IMPORTED_TARGET libzmq)
find_package(Python3 COMPONENTS Interpreter)

if(MODEL_BUILD_WAM_COSMOS3)
    set(WAM_PROTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/serving/wam.proto)
    set(WAM_PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-gen/serving)
    file(MAKE_DIRECTORY ${WAM_PROTO_GEN_DIR})

    add_custom_command(
        OUTPUT  ${WAM_PROTO_GEN_DIR}/wam.pb.h
                ${WAM_PROTO_GEN_DIR}/wam.pb.cc
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
                --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/serving
                --cpp_out=${WAM_PROTO_GEN_DIR}
                ${WAM_PROTO_SRC}
        DEPENDS ${WAM_PROTO_SRC}
        VERBATIM
    )

    add_executable(wam-server
        serving/wam-server.cpp
        ${WAM_PROTO_GEN_DIR}/wam.pb.cc
    )
    target_include_directories(wam-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime
        ${CMAKE_CURRENT_BINARY_DIR}/proto-gen
        ${Protobuf_INCLUDE_DIR}
    )
    target_link_libraries(wam-server PRIVATE
        vla_core
        ${Protobuf_LIBRARIES}
        PkgConfig::ZeroMQ
    )
endif()

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

function(add_vla_server_target target_name)
    add_executable(${target_name}
        serving/server.cpp
        ${VLA_PROTO_GEN_DIR}/vla.pb.cc
    )
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime
        ${CMAKE_CURRENT_BINARY_DIR}/proto-gen
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor/stb
        ${Protobuf_INCLUDE_DIR}
    )
    target_link_libraries(${target_name} PRIVATE
        embodied_adapter
        vla_core
        ${Protobuf_LIBRARIES}
        PkgConfig::ZeroMQ
    )
endfunction()

if(MODEL_BUILD_VLA_PI05)
    add_vla_server_target(vla-pi05-server)
endif()

if(MODEL_BUILD_VLA_HY_VLA)
    add_vla_server_target(vla-hy-vla-server)
endif()

if(MODEL_BUILD_VLA_GROOT_N1)
    add_vla_server_target(vla-groot-n1-server)
endif()

if(MODEL_BUILD_WAM_LINGBOT_VA)
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

    add_executable(wam-lingbot-server
        serving/lingbot-world-server.cpp
        ${LINGBOT_PROTO_GEN_DIR}/lingbot.pb.cc
    )
    target_include_directories(wam-lingbot-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime
        ${CMAKE_CURRENT_BINARY_DIR}/proto-gen
        ${Protobuf_INCLUDE_DIR}
    )
    target_link_libraries(wam-lingbot-server PRIVATE
        vla_core
        ${Protobuf_LIBRARIES}
        PkgConfig::ZeroMQ
    )
endif()
