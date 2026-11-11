# Builds the four probes against one llama.cpp source tree (build.sh does this for all four versions).
#
#   cmake -S . -B work/build-B -DLLAMA_CPP_DIR=<llama.cpp tree> -DPROBE_LLAMA_TAG=8fc0376 \
#         -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
#   cmake --build work/build-B --target matmul_paths kv_paths ppl_paths quantize_pure llama-bench
#
# The CUDA flags are llama.cpp's own; nothing here overrides them, so the kernels under test are compiled
# exactly as upstream compiles them. Same settings as the runs in results/; only the list of tools is shorter.
cmake_minimum_required(VERSION 3.18)
project(llamacpp_probes LANGUAGES C CXX CUDA)

set(LLAMA_CPP_DIR "" CACHE PATH "llama.cpp source tree")
if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "set -DLLAMA_CPP_DIR to a llama.cpp source tree")
endif()
set(PROBE_LLAMA_TAG "unknown" CACHE STRING "llama.cpp revision recorded in every output line")

set(GGML_CUDA             ON  CACHE BOOL "" FORCE)
set(GGML_NATIVE           OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI        OFF CACHE BOOL "" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS     OFF CACHE BOOL "")   # the runs used ON
set(LLAMA_BUILD_COMMON    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     ON  CACHE BOOL "" FORCE)   # llama-bench
set(LLAMA_BUILD_SERVER    ON  CACHE BOOL "" FORCE)   # configured, never built (EXCLUDE_FROM_ALL)
set(LLAMA_OPENSSL         OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)
find_package(CUDAToolkit REQUIRED)

add_executable(matmul_paths probes/matmul_paths.cpp)
target_link_libraries(matmul_paths PRIVATE ggml CUDA::cudart)

foreach(tool kv_paths quantize_pure ppl_paths)
    add_executable(${tool} probes/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE llama ggml CUDA::cudart)
endforeach()

foreach(tool matmul_paths kv_paths quantize_pure ppl_paths)
    target_compile_features(${tool} PRIVATE cxx_std_17)
    target_compile_definitions(${tool} PRIVATE PROBE_LLAMA_TAG="${PROBE_LLAMA_TAG}")
endforeach()
