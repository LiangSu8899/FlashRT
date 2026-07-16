# PI0.5 owns its model sources, backend variants, public C face, and tests.
# Generic native mechanisms remain in the parent cpp targets.

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(_pi05_public_include "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(_pi05_internal_include "${CMAKE_CURRENT_SOURCE_DIR}/internal/include")

set(_pi05_sources
  src/spec.cpp
  src/native_weights.cpp
  src/native_weight_ops.cpp
  src/native_device_weights.cpp
  src/native_weight_packer.cpp
  src/native_weight_materializer.cpp
  src/native_calibration.cpp
  src/native_calibration_session.cpp
  src/native_workspace.cpp
  src/native_rtx_attention.cpp
  src/prompt_format.cpp
  src/prompt_embed.cpp
  src/io.cpp
  src/runtime.cpp)
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  list(APPEND _pi05_sources
    src/native_quantization.cu
    src/native_rope.cu)
else()
  list(APPEND _pi05_sources
    src/native_quantization_unavailable.cpp
    src/native_rope_unavailable.cpp)
endif()

add_library(flashrt_cpp_pi05 STATIC ${_pi05_sources})
target_include_directories(flashrt_cpp_pi05
  PUBLIC ${_pi05_public_include}
  PRIVATE ${_pi05_internal_include})
target_link_libraries(flashrt_cpp_pi05
  PUBLIC flashrt_cpp_modalities flashrt_cpp_vla flashrt_cpp_loader)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Preserve producer-side add/cast/multiply operation boundaries.
  set_property(SOURCE src/native_weight_ops.cpp APPEND PROPERTY
    COMPILE_OPTIONS -ffp-contract=off)
endif()

if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  add_library(flashrt_cpp_pi05_kernels STATIC
    src/native_bf16_forward.cpp
    src/native_graph_runtime.cpp
    src/native_kernel_driver.cu
    src/native_rtx_linear.cpp
    src/native_rtx_weight_packer.cpp
    src/native_style_precompute.cu
    ${FLASHRT_CPP_SOURCE_DIR}/../csrc/gemm/gemm_runner.cu
    ${FLASHRT_CPP_SOURCE_DIR}/../csrc/kernels/activation.cu
    ${FLASHRT_CPP_SOURCE_DIR}/../csrc/kernels/dit_bf16.cu
    ${FLASHRT_CPP_SOURCE_DIR}/../csrc/kernels/elementwise.cu
    ${FLASHRT_CPP_SOURCE_DIR}/../csrc/kernels/fusion.cu
    ${FLASHRT_CPP_SOURCE_DIR}/../csrc/kernels/norm.cu
    ${FLASHRT_CPP_SOURCE_DIR}/../csrc/kernels/patch_embed.cu
    ${FLASHRT_CPP_SOURCE_DIR}/../csrc/kernels/quantize.cu
    ${FLASHRT_CPP_SOURCE_DIR}/../csrc/kernels/rope.cu)
  target_include_directories(flashrt_cpp_pi05_kernels
    PRIVATE ${_pi05_public_include}
            ${_pi05_internal_include}
            ${FLASHRT_CPP_SOURCE_DIR}/../csrc/gemm
            ${FLASHRT_CPP_SOURCE_DIR}/../csrc/kernels)
  target_link_libraries(flashrt_cpp_pi05_kernels
    PUBLIC flashrt_cpp_pi05 CUDA::cublas CUDA::cublasLt CUDA::cudart)
  set_target_properties(flashrt_cpp_pi05_kernels PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
  target_compile_options(flashrt_cpp_pi05_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3
      --ftz=true --prec-div=false --prec-sqrt=false>)

  if(FLASHRT_CPP_WITH_THOR_FP8)
    add_library(flashrt_cpp_pi05_thor_precise OBJECT
      src/native_thor_setup_ops.cu
      ${FLASHRT_CPP_SOURCE_DIR}/../csrc/attention/fmha_fp16_strided.cu)
    target_include_directories(flashrt_cpp_pi05_thor_precise PRIVATE
      ${FLASHRT_CPP_SOURCE_DIR}/../csrc/attention
      ${FLASHRT_CPP_CUTLASS_DIR}/examples/77_blackwell_fmha
      ${FLASHRT_CPP_CUTLASS_DIR}/include
      ${FLASHRT_CPP_CUTLASS_DIR}/tools/util/include)
    target_compile_options(flashrt_cpp_pi05_thor_precise PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr --expt-extended-lambda -O3>)
    set_target_properties(flashrt_cpp_pi05_thor_precise PROPERTIES
      CUDA_ARCHITECTURES 110a
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      POSITION_INDEPENDENT_CODE ON)

    target_sources(flashrt_cpp_pi05_kernels PRIVATE
      $<TARGET_OBJECTS:flashrt_cpp_pi05_thor_precise>
      src/native_thor_kernel_driver.cu
      src/native_thor_weight_materializer.cpp
      src/native_thor_fp8_forward.cpp
      src/native_thor_graph_owner.cpp
      src/native_thor_calibration_session.cpp
      src/native_thor_style_precompute.cpp
      ${FLASHRT_CPP_SOURCE_DIR}/../csrc/gemm/cutlass_sm100.cu
      ${FLASHRT_CPP_SOURCE_DIR}/../csrc/kernels/attention_cublas.cu
      ${FLASHRT_CPP_SOURCE_DIR}/../csrc/kernels/decoder_fused.cu
      ${FLASHRT_CPP_SOURCE_DIR}/../csrc/kernels/softmax.cu)
    target_include_directories(flashrt_cpp_pi05_kernels PRIVATE
      ${FLASHRT_CPP_SOURCE_DIR}/../csrc/attention
      ${FLASHRT_CPP_CUTLASS_DIR}/examples/77_blackwell_fmha
      ${FLASHRT_CPP_CUTLASS_DIR}/include
      ${FLASHRT_CPP_CUTLASS_DIR}/tools/util/include)
    target_compile_definitions(flashrt_cpp_pi05_kernels
      PUBLIC FLASHRT_CPP_WITH_THOR_FP8=1)
    target_compile_options(flashrt_cpp_pi05_kernels PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda --use_fast_math
        -gencode=arch=compute_110a,code=sm_110a>)
    set_target_properties(flashrt_cpp_pi05_kernels PROPERTIES
      CUDA_ARCHITECTURES 110a
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  endif()

  if(FLASHRT_CPP_WITH_FA2)
    target_sources(flashrt_cpp_pi05_kernels PRIVATE
      src/native_graph_owner.cpp
      src/native_rtx_autotune.cpp
      src/native_rtx_attention_driver.cu)
    if(FLASHRT_CPP_WITH_SENTENCEPIECE)
      target_sources(flashrt_cpp_pi05_kernels PRIVATE
        src/native_rtx_calibration_session.cpp)
    endif()
    target_include_directories(flashrt_cpp_pi05_kernels PRIVATE
      ${FLASHRT_CPP_SOURCE_DIR}/../csrc)
    target_link_libraries(flashrt_cpp_pi05_kernels
      PUBLIC flashrt_cpp_fa2_external)
    target_compile_definitions(flashrt_cpp_pi05_kernels
      PUBLIC FLASHRT_CPP_WITH_FA2=1)
  endif()
endif()

add_library(flashrt_cpp_pi05_c SHARED
  src/c_api.cpp
  src/model_runtime.cpp
  src/native_calibration_c_api.cpp
  src/native_model_runtime.cpp
  src/native_open.cpp)
target_link_libraries(flashrt_cpp_pi05_c
  PUBLIC flashrt_cpp_pi05 flashrt_runtime)
target_include_directories(flashrt_cpp_pi05_c
  PUBLIC ${_pi05_public_include}
  PRIVATE ${_pi05_internal_include})
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  target_link_libraries(flashrt_cpp_pi05_c
    PRIVATE flashrt_cpp_pi05_kernels)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
