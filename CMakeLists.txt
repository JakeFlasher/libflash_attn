include(CMakePrintHelpers)
cmake_minimum_required(VERSION 3.12)
project(flash-attention LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

# GPU detection
find_program(GPU_dev_info nvidia-smi)
if(NOT GPU_dev_info)
    message(FATAL_ERROR "GPU driver not found.")
endif()

find_program(NVCC nvcc)
if(NOT NVCC)
    message(FATAL_ERROR "NVCC not found. Please make sure CUDA Toolkit is installed.")
endif()

execute_process(COMMAND ${GPU_dev_info} OUTPUT_VARIABLE GPU_dev_version)
string(REGEX MATCH "H100" GPU_H100 ${GPU_dev_version})

execute_process(COMMAND ${NVCC} --version OUTPUT_VARIABLE NVCC_VERSION_OUTPUT)
string(REGEX MATCH "([0-9]+\\.[0-9]+)" NVCC_VERSION ${NVCC_VERSION_OUTPUT})

if(NOT NVCC_VERSION)
    message(FATAL_ERROR "Failed to determine NVCC version.")
endif()

if(GPU_H100 AND NVCC_VERSION GREATER_EQUAL 12)
    set(compute_capability "90a")
    add_definitions(-DHOPPER)
else()
    set(compute_capability "80")
endif()

message("compute_capability: ${compute_capability}")

# CUDA configuration
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math -t 8 -std=c++17")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${compute_capability},code=\\\"sm_${compute_capability},compute_${compute_capability}\\\"")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo -O0")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")
endif()

# Find required packages
find_package(CUDAToolkit REQUIRED)
cmake_print_variables(CUDAToolkit_INCLUDE_DIRS)
cmake_print_variables(CUDA_TOOLKIT_ROOT_DIR)

# Set paths
set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass)
set(SYS_CUDART_PATH "/usr/local/cuda")
set(SYS_CUDA_DRIVER_PATH "/usr/lib/x86_64-linux-gnu")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUTLASS_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${SYS_CUDART_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR} # For argparse.hpp
)

# Link directories
link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${SYS_CUDA_DRIVER_PATH}
    ${SYS_CUDART_PATH}/lib64
)

# Download argparse if not present
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/argparse/argparse.hpp")
    message(STATUS "Downloading argparse.hpp...")
    file(DOWNLOAD
         "https://raw.githubusercontent.com/p-ranav/argparse/master/include/argparse/argparse.hpp"
         "${CMAKE_CURRENT_SOURCE_DIR}/argparse/argparse.hpp"
         SHOW_PROGRESS
    )
endif()

# Source files for the library
set(COMMON_SRCS
    src/flash.cu
    src/flash_fwd_hdim128_fp16_sm80.cu
    src/flash_fwd_hdim160_fp16_sm80.cu
    src/flash_fwd_hdim192_fp16_sm80.cu
    src/flash_fwd_hdim224_fp16_sm80.cu
    src/flash_fwd_hdim256_fp16_sm80.cu
    src/flash_fwd_hdim32_fp16_sm80.cu
    src/flash_fwd_hdim64_fp16_sm80.cu
    src/flash_fwd_hdim96_fp16_sm80.cu
    src/flash_fwd_split_hdim128_fp16_sm80.cu
    src/flash_fwd_split_hdim160_fp16_sm80.cu
    src/flash_fwd_split_hdim192_fp16_sm80.cu
    src/flash_fwd_split_hdim224_fp16_sm80.cu
    src/flash_fwd_split_hdim256_fp16_sm80.cu
    src/flash_fwd_split_hdim32_fp16_sm80.cu
    src/flash_fwd_split_hdim64_fp16_sm80.cu
    src/flash_fwd_split_hdim96_fp16_sm80.cu
)

# Add Hopper-specific files if needed
if(compute_capability STREQUAL "90a")
    set(HOPPER_SRCS
        hopper/flash_fwd_hdim64_fp16_sm90.cu
        hopper/flash_fwd_hdim128_fp16_sm90.cu
        hopper/flash_fwd_hdim256_fp16_sm90.cu
        hopper/flash_fwd_hdim64_e4m3_sm90.cu
        hopper/flash_fwd_hdim128_e4m3_sm90.cu
        hopper/flash_fwd_hdim256_e4m3_sm90.cu
    )
    list(APPEND COMMON_SRCS ${HOPPER_SRCS})
endif()

# Create the shared library
add_library(flash_attn SHARED ${COMMON_SRCS})
set_target_properties(flash_attn PROPERTIES CUDA_ARCHITECTURES "${compute_capability}")

# Create the test executable
add_executable(bench src/bench.cc)
set_target_properties(bench PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(bench
    flash_attn
    CUDA::cudart
    CUDA::cublas
    CUDA::cuda_driver
)
