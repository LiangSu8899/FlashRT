cmake_minimum_required(VERSION 3.24)
project(flash_wm_kernels CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# ── GPU architecture ──
if(NOT DEFINED GPU_ARCH)
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
    OUTPUT_VARIABLE _detected_cc
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _detect_result
  )
  if(_detect_result EQUAL 0 AND _detected_cc)
    string(REGEX MATCH "[0-9]+\\.[0-9]+" _first_cc "${_detected_cc}")
    string(REPLACE "." "" GPU_ARCH "${_first_cc}")
    message(STATUS "Auto-detected GPU arch: sm_${GPU_ARCH}")
  else()
    set(GPU_ARCH "110")
    message(STATUS "GPU detection failed, defaulting to sm_${GPU_ARCH}")
  endif()
endif()

set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})

# ── Find Python & pybind11 ──
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# ── Find cuBLAS ──
find_package(CUDAToolkit REQUIRED)

# ── CUTLASS include (for bf16-out FP4 GEMM) ──
# Prefer a local checkout, fall back to the pytorch NGC container bundle,
# or accept an env override (CUTLASS_INCLUDE_DIR=/path/to/cutlass/include).
set(CUTLASS_INCLUDE "")
foreach(cand
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass/include"
    "/opt/pytorch/pytorch/third_party/cutlass/include"
    "$ENV{CUTLASS_INCLUDE_DIR}"
)
  if(cand AND EXISTS "${cand}/cutlass/cutlass.h")
    set(CUTLASS_INCLUDE "${cand}")
    break()
  endif()
endforeach()
if(CUTLASS_INCLUDE STREQUAL "")
  message(FATAL_ERROR "CUTLASS headers not found — set CUTLASS_INCLUDE_DIR or place CUTLASS at <repo>/third_party/cutlass")
endif()
message(STATUS "CUTLASS include: ${CUTLASS_INCLUDE}")

# ── CUTLASS Blackwell FMHA (example 77) headers ──
# flash_wm's cutlass_fp8_fmha.cu pulls {kernel, device, most of collective}
# from examples/77, but uses flash_rt's LOCAL modified mainloop which has
# kPRescale softmax denorm protection. So include both dirs.
set(CUTLASS_FMHA77_INC "${CUTLASS_INCLUDE}/../examples/77_blackwell_fmha")
set(FLASHVLA_ATTN_INC "${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/attention")
if(NOT EXISTS "${CUTLASS_FMHA77_INC}/device/fmha.hpp")
  message(WARNING "CUTLASS example 77 headers not found at ${CUTLASS_FMHA77_INC} — FP8 FMHA will not build")
endif()

# ── Build flash_wm_kernels.so ──
pybind11_add_module(flash_wm_kernels
  bindings.cpp
  kernels_bf16.cu
  cutlass_fp4_gemm_bf16out.cu
  cutlass_fp8_fmha.cu
  cutlass_fp8_sq_bias_bf16out.cu
  bagel_silu_mul_fp4_sfa.cu
  bagel_res_rms_fp4_sfa_bf16.cu
)

target_include_directories(flash_wm_kernels PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CUTLASS_INCLUDE}
  ${CUTLASS_INCLUDE}/../tools/util/include
  ${FLASHVLA_ATTN_INC}    # flash_rt's modified mainloop (kPRescale FP8 fix)
  ${CUTLASS_FMHA77_INC}   # CUTLASS example 77 headers (kernel/, device/, rest of collective/)
)

target_link_libraries(flash_wm_kernels PRIVATE
  CUDA::cublas
  CUDA::cudart
)

# CUDA flags.
# IMPORTANT: CUTLASS FP4 GEMM for SM100/SM110 uses TMA instructions that
# require the 'a' suffix arch flag (e.g. sm_110a) to be enabled. Without
# it kernels compile but hit a runtime assertion inside cute prefetch_tma.
# We override CMake's default -gencode to add the 'a' suffix.
set_target_properties(flash_wm_kernels PROPERTIES
  CUDA_ARCHITECTURES OFF
)
target_compile_options(flash_wm_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    -O3
    -gencode arch=compute_${GPU_ARCH}a,code=sm_${GPU_ARCH}a
  >
)
# CUTLASS SM100 MMA path (Blackwell tcgen05 tensor cores) is gated by this
# preprocessor define. Without it, the SM100 FMHA kernel compiles but hits
# "Arch conditional MMA instruction used without targeting appropriate
# compute capability" at launch. SM110 (Thor) supports the SM100 MMA ISA,
# so enabling this is correct.
target_compile_definitions(flash_wm_kernels PRIVATE
  CUTLASS_ARCH_MMA_SM100_SUPPORTED=1
)

set_target_properties(flash_wm_kernels PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
)
