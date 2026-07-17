# PI0.5 owns its model sources, backend variants, public C face, and tests.
# Generic native mechanisms remain in the parent cpp targets.

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(_pi05_public_include "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(_pi05_internal_include "${CMAKE_CURRENT_SOURCE_DIR}/internal/include")

set(_pi05_sources
  src/model/spec.cpp
  src/model/prompt_format.cpp
  src/model/prompt_embed.cpp
  src/model/io.cpp
  src/model/runtime.cpp
  src/support/native_weights.cpp
  src/support/native_weight_ops.cpp
  src/support/native_device_weights.cpp
  src/support/native_weight_materializer.cpp
  src/support/native_calibration.cpp
  src/support/native_workspace.cpp
  src/backend/native_calibration_session.cpp)
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  list(APPEND _pi05_sources
    src/support/native_quantization.cu)
  if(FLASHRT_CPP_WITH_FA2 OR FLASHRT_CPP_WITH_THOR_FP8)
    list(APPEND _pi05_sources src/support/native_rope.cu)
  else()
    list(APPEND _pi05_sources src/support/native_rope_unavailable.cpp)
  endif()
else()
  list(APPEND _pi05_sources
    src/support/native_quantization_unavailable.cpp
    src/support/native_rope_unavailable.cpp)
endif()

add_library(flashrt_cpp_pi05 STATIC ${_pi05_sources})
target_include_directories(flashrt_cpp_pi05
  PUBLIC ${_pi05_public_include}
  PRIVATE ${_pi05_internal_include})
target_link_libraries(flashrt_cpp_pi05
  PUBLIC flashrt_cpp_modalities flashrt_cpp_vla flashrt_cpp_loader)
if(FLASHRT_CPP_WITH_CUDA_KERNELS AND
   (FLASHRT_CPP_WITH_FA2 OR FLASHRT_CPP_WITH_THOR_FP8))
  target_include_directories(flashrt_cpp_pi05 PRIVATE
    ${FLASHRT_CPP_SOURCE_DIR}/../csrc/kernels)
  target_link_libraries(flashrt_cpp_pi05 PRIVATE flashrt_cpp_csrc_core)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Preserve producer-side add/cast/multiply operation boundaries.
  set_property(SOURCE src/support/native_weight_ops.cpp APPEND PROPERTY
    COMPILE_OPTIONS -ffp-contract=off)
endif()

if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  function(_flashrt_configure_pi05_cuda_target target)
    target_include_directories(${target} PRIVATE
      ${_pi05_public_include}
      ${_pi05_internal_include}
      ${FLASHRT_CPP_SOURCE_DIR}/../csrc/gemm
      ${FLASHRT_CPP_SOURCE_DIR}/../csrc/kernels)
    target_link_libraries(${target} PUBLIC
      flashrt_cpp_pi05 flashrt_cpp_native_cuda
      CUDA::cublas CUDA::cublasLt CUDA::cudart)
    set_target_properties(${target} PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr -O3
        --ftz=true --prec-div=false --prec-sqrt=false>)
  endfunction()

  add_library(flashrt_cpp_pi05_plans INTERFACE)
  set(_pi05_plan_targets)

  if(FLASHRT_CPP_WITH_FA2 OR FLASHRT_CPP_WITH_THOR_FP8)
    add_library(flashrt_cpp_pi05_pipeline_cuda STATIC
      src/model/pipeline.cpp)
    _flashrt_configure_pi05_cuda_target(flashrt_cpp_pi05_pipeline_cuda)
  endif()

  if(FLASHRT_CPP_WITH_FA2)
    add_library(flashrt_cpp_pi05_plan_sm120 STATIC
      src/plans/sm120/native_bf16_forward.cpp
      src/plans/sm120/lowered_plan.cpp
      src/plans/sm120/native_kernel_driver.cu
      src/plans/sm120/native_rtx_attention.cpp
      src/plans/sm120/native_rtx_attention_driver.cu
      src/plans/sm120/native_rtx_autotune.cpp
      src/plans/sm120/native_rtx_linear.cpp
      src/plans/sm120/native_rtx_weight_packer.cpp
      src/plans/sm120/native_style_precompute.cu)
    if(FLASHRT_CPP_WITH_SENTENCEPIECE)
      target_sources(flashrt_cpp_pi05_plan_sm120 PRIVATE
        src/plans/sm120/native_rtx_calibration_session.cpp)
    endif()
    _flashrt_configure_pi05_cuda_target(flashrt_cpp_pi05_plan_sm120)
    target_include_directories(flashrt_cpp_pi05_plan_sm120 PRIVATE
      ${FLASHRT_CPP_SOURCE_DIR}/../csrc)
    target_link_libraries(flashrt_cpp_pi05_plan_sm120 PUBLIC
      flashrt_cpp_pi05_pipeline_cuda flashrt_cpp_csrc_sm120
      flashrt_cpp_fa2_external)
    target_compile_definitions(flashrt_cpp_pi05_plan_sm120
      PUBLIC FLASHRT_CPP_WITH_FA2=1)
    list(APPEND _pi05_plan_targets flashrt_cpp_pi05_plan_sm120)
  endif()

  if(FLASHRT_CPP_WITH_THOR_FP8)
    add_library(flashrt_cpp_pi05_plan_sm110 STATIC
      src/plans/sm110/native_thor_kernel_driver.cu
      src/plans/sm110/native_thor_weight_materializer.cpp
      src/plans/sm110/native_thor_fp8_forward.cpp
      src/plans/sm110/lowered_plan.cpp
      src/plans/sm110/native_thor_calibration_session.cpp
      src/plans/sm110/native_thor_style_precompute.cpp)
    _flashrt_configure_pi05_cuda_target(flashrt_cpp_pi05_plan_sm110)
    target_include_directories(flashrt_cpp_pi05_plan_sm110 PRIVATE
      ${FLASHRT_CPP_SOURCE_DIR}/../csrc/attention
      ${FLASHRT_CPP_CUTLASS_DIR}/examples/77_blackwell_fmha
      ${FLASHRT_CPP_CUTLASS_DIR}/include
      ${FLASHRT_CPP_CUTLASS_DIR}/tools/util/include)
    target_link_libraries(flashrt_cpp_pi05_plan_sm110 PUBLIC
      flashrt_cpp_pi05_pipeline_cuda
      flashrt_cpp_csrc_sm110)
    target_compile_definitions(flashrt_cpp_pi05_plan_sm110
      PUBLIC FLASHRT_CPP_WITH_THOR_FP8=1)
    target_compile_options(flashrt_cpp_pi05_plan_sm110 PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda --use_fast_math
        -gencode=arch=compute_110a,code=sm_110a>)
    set_target_properties(flashrt_cpp_pi05_plan_sm110 PROPERTIES
      CUDA_ARCHITECTURES 110a
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    list(APPEND _pi05_plan_targets flashrt_cpp_pi05_plan_sm110)
  endif()

  target_link_libraries(flashrt_cpp_pi05_plans INTERFACE
    ${_pi05_plan_targets})
endif()

add_library(flashrt_cpp_pi05_c SHARED
  src/service/c_api.cpp
  src/service/model_runtime.cpp
  src/service/native_calibration_c_api.cpp
  src/service/native_model_runtime.cpp
  src/service/native_open.cpp)
target_link_libraries(flashrt_cpp_pi05_c
  PUBLIC flashrt_cpp_pi05 flashrt_runtime
  PRIVATE flashrt_cpp_native)
target_include_directories(flashrt_cpp_pi05_c
  PUBLIC ${_pi05_public_include}
  PRIVATE ${_pi05_internal_include})
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  target_link_libraries(flashrt_cpp_pi05_c
    PRIVATE flashrt_cpp_pi05_plans)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
