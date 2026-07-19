# FlashRT native C++ runtime path.
#
# This tree is intentionally separate from runtime/:
#   runtime/ = stable C ABI hand-off (frt_runtime_export_v1)
#   cpp/     = native model runtime implementation layers
#
# Build:
#   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build cpp/build -j

cmake_minimum_required(VERSION 3.18)
project(flashrt_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
include(CTest)

option(FLASHRT_CPP_WITH_EXEC "Build/link the exec layer for native replay" ON)
option(FLASHRT_CPP_WITH_CUDA_STAGING "Enable conservative CUDA H2D/D2H modality staging" ON)
option(FLASHRT_CPP_WITH_CUDA_KERNELS "Enable CUDA modality kernels" ON)
option(FLASHRT_CPP_WITH_SENTENCEPIECE
       "Enable native SentencePiece text tokenization" OFF)
option(FLASHRT_CPP_WITH_FA2
       "Enable the Python-free vendored FA2 operation library" OFF)
option(FLASHRT_CPP_WITH_PI05_SM120_TARGET
       "Build the PI0.5 SM120 frontend target" OFF)
set(FLASHRT_CPP_FA2_LIBRARY "" CACHE FILEPATH
    "Path to the Python-free libflashrt_fa2_raw shared library")
if(FLASHRT_CPP_WITH_CUDA_STAGING)
  find_package(CUDAToolkit REQUIRED)
endif()
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_library(flashrt_cpp_csrc_precise_math STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/kernels/rope_table.cu)
  target_include_directories(flashrt_cpp_csrc_precise_math PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/kernels)
  target_link_libraries(flashrt_cpp_csrc_precise_math PUBLIC CUDA::cudart)
  target_compile_options(flashrt_cpp_csrc_precise_math PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  set_target_properties(flashrt_cpp_csrc_precise_math PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
endif()
if(FLASHRT_CPP_WITH_FA2)
  if(NOT FLASHRT_CPP_WITH_CUDA_KERNELS)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_FA2 requires FLASHRT_CPP_WITH_CUDA_KERNELS")
  endif()
  if(NOT FLASHRT_CPP_FA2_LIBRARY)
    find_library(FLASHRT_CPP_FA2_LIBRARY
      NAMES flashrt_fa2_raw
      PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../flash_rt
      NO_DEFAULT_PATH)
  endif()
  if(NOT FLASHRT_CPP_FA2_LIBRARY OR
     NOT EXISTS "${FLASHRT_CPP_FA2_LIBRARY}")
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_FA2 requires libflashrt_fa2_raw; build the root "
      "flashrt_fa2_raw target or set FLASHRT_CPP_FA2_LIBRARY")
  endif()
  add_library(flashrt_cpp_fa2_external SHARED IMPORTED GLOBAL)
  set_target_properties(flashrt_cpp_fa2_external PROPERTIES
    IMPORTED_LOCATION ${FLASHRT_CPP_FA2_LIBRARY})
endif()
if(FLASHRT_CPP_WITH_PI05_SM120_TARGET)
  if(NOT FLASHRT_CPP_WITH_FA2)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_PI05_SM120_TARGET requires FLASHRT_CPP_WITH_FA2")
  endif()
  if(NOT FLASHRT_CPP_WITH_CUDA_STAGING)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_PI05_SM120_TARGET requires CUDA staging")
  endif()
  if(NOT FLASHRT_CPP_WITH_EXEC)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_PI05_SM120_TARGET requires the exec layer")
  endif()
endif()
if(FLASHRT_CPP_WITH_SENTENCEPIECE)
  find_path(FLASHRT_SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h)
  find_library(FLASHRT_SENTENCEPIECE_LIBRARY sentencepiece)
  if(FLASHRT_SENTENCEPIECE_INCLUDE_DIR AND FLASHRT_SENTENCEPIECE_LIBRARY)
    add_library(flashrt_sentencepiece_external INTERFACE)
    target_include_directories(flashrt_sentencepiece_external
      INTERFACE ${FLASHRT_SENTENCEPIECE_INCLUDE_DIR})
    target_link_libraries(flashrt_sentencepiece_external
      INTERFACE ${FLASHRT_SENTENCEPIECE_LIBRARY})
    set(FLASHRT_CPP_SENTENCEPIECE_TARGET flashrt_sentencepiece_external)
  else()
    include(FetchContent)
    set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
    set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
    set(SPM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      sentencepiece
      GIT_REPOSITORY https://github.com/google/sentencepiece.git
      GIT_TAG v0.2.1)
    FetchContent_MakeAvailable(sentencepiece)
    set(FLASHRT_SENTENCEPIECE_INCLUDE_DIR ${sentencepiece_SOURCE_DIR}/src)
    set(FLASHRT_CPP_SENTENCEPIECE_TARGET sentencepiece-static)
  endif()
endif()
if(FLASHRT_CPP_WITH_EXEC AND NOT TARGET flashrt_exec)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../exec
                   ${CMAKE_CURRENT_BINARY_DIR}/exec)
endif()
if(NOT TARGET flashrt_runtime)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../runtime
                   ${CMAKE_CURRENT_BINARY_DIR}/runtime)
endif()

add_library(flashrt_cpp INTERFACE)
target_include_directories(flashrt_cpp
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../exec/include)
if(FLASHRT_CPP_WITH_EXEC)
  target_link_libraries(flashrt_cpp INTERFACE flashrt_exec)
endif()

set(FLASHRT_CPP_MODALITY_SRCS
  modalities/src/types.cpp
  modalities/src/vision_cpu.cpp
  modalities/src/text_cpu.cpp
  modalities/src/action_cpu.cpp)
if(FLASHRT_CPP_WITH_SENTENCEPIECE)
  list(APPEND FLASHRT_CPP_MODALITY_SRCS
    modalities/src/sentencepiece_tokenizer.cpp)
else()
  list(APPEND FLASHRT_CPP_MODALITY_SRCS
    modalities/src/tokenizer_unavailable.cpp)
endif()
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  list(APPEND FLASHRT_CPP_MODALITY_SRCS
    modalities/src/vision_cuda.cu
    modalities/src/text_cuda.cu)
endif()

add_library(flashrt_cpp_modalities STATIC ${FLASHRT_CPP_MODALITY_SRCS})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_property(SOURCE modalities/src/vision_cpu.cpp APPEND PROPERTY
    COMPILE_OPTIONS -ffp-contract=off)
endif()
target_include_directories(flashrt_cpp_modalities
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/modalities/include)
if(FLASHRT_CPP_WITH_CUDA_STAGING)
  target_compile_definitions(flashrt_cpp_modalities
    PUBLIC FLASHRT_CPP_WITH_CUDA_STAGING=1)
  target_link_libraries(flashrt_cpp_modalities PUBLIC CUDA::cudart)
endif()
if(FLASHRT_CPP_WITH_SENTENCEPIECE)
  target_compile_definitions(flashrt_cpp_modalities
    PUBLIC FLASHRT_CPP_HAS_SENTENCEPIECE=1)
  target_include_directories(flashrt_cpp_modalities
    PUBLIC ${FLASHRT_SENTENCEPIECE_INCLUDE_DIR})
  target_link_libraries(flashrt_cpp_modalities
    PUBLIC ${FLASHRT_CPP_SENTENCEPIECE_TARGET})
endif()
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  target_compile_definitions(flashrt_cpp_modalities
    PRIVATE FLASHRT_CPP_WITH_CUDA_KERNELS=1)
  target_link_libraries(flashrt_cpp_modalities PUBLIC CUDA::cudart)
  set_target_properties(flashrt_cpp_modalities PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
endif()

add_library(flashrt_cpp_vla INTERFACE)
target_include_directories(flashrt_cpp_vla
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/families/vla/include)
target_link_libraries(flashrt_cpp_vla
  INTERFACE flashrt_cpp_modalities flashrt_cpp)

add_library(flashrt_cpp_native STATIC
  native/src/config_object.cpp)
target_include_directories(flashrt_cpp_native
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/native/include)

if(FLASHRT_CPP_WITH_CUDA_KERNELS AND FLASHRT_CPP_WITH_EXEC)
  add_library(flashrt_cpp_native_cuda STATIC
    native/src/cuda_graph_set.cpp)
  target_include_directories(flashrt_cpp_native_cuda
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/native/include)
  target_link_libraries(flashrt_cpp_native_cuda
    PUBLIC flashrt_cpp_modalities flashrt_exec CUDA::cudart)
endif()

if(FLASHRT_CPP_WITH_FA2)
  set(_flashrt_csrc_dir "${CMAKE_CURRENT_SOURCE_DIR}/../csrc")

  function(_flashrt_configure_csrc_target target)
    target_include_directories(${target} PUBLIC
      ${_flashrt_csrc_dir}
      ${_flashrt_csrc_dir}/attention
      ${_flashrt_csrc_dir}/gemm
      ${_flashrt_csrc_dir}/kernels)
    target_link_libraries(${target} PUBLIC
      CUDA::cublas CUDA::cublasLt CUDA::cudart)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr -O3
        --ftz=true --prec-div=false --prec-sqrt=false>)
    set_target_properties(${target} PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
  endfunction()

  add_library(flashrt_cpp_csrc_core STATIC
    ${_flashrt_csrc_dir}/gemm/gemm_runner.cu
    ${_flashrt_csrc_dir}/kernels/activation.cu
    ${_flashrt_csrc_dir}/kernels/elementwise.cu
    ${_flashrt_csrc_dir}/kernels/norm.cu
    ${_flashrt_csrc_dir}/kernels/patch_embed.cu
    ${_flashrt_csrc_dir}/kernels/quantize.cu
    ${_flashrt_csrc_dir}/kernels/rope.cu)
  _flashrt_configure_csrc_target(flashrt_cpp_csrc_core)
  target_link_libraries(flashrt_cpp_csrc_core PUBLIC
    flashrt_cpp_csrc_precise_math)

  add_library(flashrt_cpp_csrc_sm120 STATIC
    ${_flashrt_csrc_dir}/kernels/dit_bf16.cu
    ${_flashrt_csrc_dir}/kernels/fusion.cu)
  _flashrt_configure_csrc_target(flashrt_cpp_csrc_sm120)
  target_link_libraries(flashrt_cpp_csrc_sm120 PUBLIC
    flashrt_cpp_csrc_core)
endif()

add_library(flashrt_cpp_loader STATIC
  loader/src/safetensors.cpp
  loader/src/sha256.cpp)
set_property(SOURCE loader/src/sha256.cpp APPEND PROPERTY COMPILE_OPTIONS -O2)
target_include_directories(flashrt_cpp_loader
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/loader/include)
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
  target_compile_definitions(flashrt_cpp_loader
    PRIVATE FLASHRT_CPP_HAS_OPENSSL=1)
  target_link_libraries(flashrt_cpp_loader PRIVATE OpenSSL::Crypto)
endif()

set(FLASHRT_CPP_PI05_SUPPORT_SRCS
  models/pi05/src/support/native_rope_unavailable.cpp)
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  set(FLASHRT_CPP_PI05_SUPPORT_SRCS
    models/pi05/src/support/native_rope.cu)
endif()

add_library(flashrt_cpp_pi05 STATIC
  models/pi05/src/model/dims.cpp
  models/pi05/src/model/execution_plan.cpp
  models/pi05/src/model/linear_weight_groups.cpp
  models/pi05/src/model/prompt_embed.cpp
  models/pi05/src/model/prompt_format.cpp
  models/pi05/src/model/resolved_resources.cpp
  models/pi05/src/model/semantic_pipeline.cpp
  models/pi05/src/support/native_calibration.cpp
  models/pi05/src/support/native_device_weights.cpp
  models/pi05/src/support/native_weight_materializer.cpp
  models/pi05/src/support/native_resource_resolver.cpp
  models/pi05/src/support/native_weight_ops.cpp
  models/pi05/src/support/native_weights.cpp
  models/pi05/src/support/native_workspace.cpp
  ${FLASHRT_CPP_PI05_SUPPORT_SRCS}
  models/pi05/src/spec.cpp
  models/pi05/src/io.cpp
  models/pi05/src/runtime.cpp)
target_include_directories(flashrt_cpp_pi05
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
target_link_libraries(flashrt_cpp_pi05
  PRIVATE flashrt_cpp_loader
  PUBLIC flashrt_cpp_modalities flashrt_cpp_vla)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_property(SOURCE models/pi05/src/support/native_weight_ops.cpp
    APPEND PROPERTY COMPILE_OPTIONS -ffp-contract=off)
endif()
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  target_compile_definitions(flashrt_cpp_pi05
    PRIVATE FLASHRT_CPP_WITH_CUDA_KERNELS=1)
  target_link_libraries(flashrt_cpp_pi05 PRIVATE
    flashrt_cpp_csrc_precise_math CUDA::cudart)
  set_target_properties(flashrt_cpp_pi05 PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
endif()

if(TARGET flashrt_cpp_native_cuda)
  add_library(flashrt_cpp_pi05_capture STATIC
    models/pi05/src/model/captured_program.cpp
    models/pi05/src/model/native_session.cpp)
  target_include_directories(flashrt_cpp_pi05_capture
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(flashrt_cpp_pi05_capture
    PUBLIC flashrt_cpp_pi05 flashrt_cpp_native_cuda CUDA::cudart)
endif()

if(FLASHRT_CPP_WITH_PI05_SM120_TARGET)
  add_library(flashrt_cpp_pi05_target_sm120 STATIC
    models/pi05/src/targets/sm120/attention.cpp
    models/pi05/src/targets/sm120/attention_driver.cpp
    models/pi05/src/targets/sm120/bf16_linear.cpp
    models/pi05/src/targets/sm120/operations.cpp
    models/pi05/src/targets/sm120/bf16_scratch.cpp
    models/pi05/src/targets/sm120/fp8_linear.cpp
    models/pi05/src/targets/sm120/fp8_weight_packer.cpp
    models/pi05/src/targets/sm120/target.cpp)
  target_include_directories(flashrt_cpp_pi05_target_sm120
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(flashrt_cpp_pi05_target_sm120
    PUBLIC flashrt_cpp_pi05
    PRIVATE flashrt_cpp_loader flashrt_cpp_csrc_core flashrt_cpp_csrc_sm120
            flashrt_cpp_fa2_external CUDA::cudart)
endif()

add_library(flashrt_cpp_pi05_c SHARED
  models/pi05/src/c_api.cpp
  models/pi05/src/model_runtime.cpp)
target_link_libraries(flashrt_cpp_pi05_c
  PUBLIC flashrt_cpp_pi05 flashrt_runtime)
target_include_directories(flashrt_cpp_pi05_c
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include)

if(BUILD_TESTING)
  if(TARGET flashrt_cpp_csrc_precise_math)
    add_executable(test_csrc_rope_table tests/test_csrc_rope_table.cu)
    target_link_libraries(test_csrc_rope_table PRIVATE
      flashrt_cpp_csrc_precise_math CUDA::cudart)
    set_target_properties(test_csrc_rope_table PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    add_test(NAME csrc_rope_table COMMAND test_csrc_rope_table)

    add_executable(test_csrc_attention_seqused
      tests/test_csrc_attention_seqused.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/kernels/attention_cublas.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/kernels/softmax.cu)
    target_include_directories(test_csrc_attention_seqused PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/kernels)
    target_link_libraries(test_csrc_attention_seqused PRIVATE
      CUDA::cublas CUDA::cudart)
    target_compile_options(test_csrc_attention_seqused PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr -O3
        --ftz=true --prec-div=false --prec-sqrt=false>)
    set_target_properties(test_csrc_attention_seqused PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    add_test(NAME csrc_attention_seqused
      COMMAND test_csrc_attention_seqused)
  endif()

  if(TARGET flashrt_cpp_csrc_core)
    add_executable(test_csrc_bf16_operations
      tests/test_csrc_bf16_operations.cu)
    target_link_libraries(test_csrc_bf16_operations PRIVATE
      flashrt_cpp_csrc_core CUDA::cudart)
    set_target_properties(test_csrc_bf16_operations PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    add_test(NAME csrc_bf16_operations COMMAND test_csrc_bf16_operations)

    add_executable(test_csrc_fp8_weight_quantize
      tests/test_csrc_fp8_weight_quantize.cu)
    target_link_libraries(test_csrc_fp8_weight_quantize PRIVATE
      flashrt_cpp_csrc_core flashrt_runtime CUDA::cudart)
    set_target_properties(test_csrc_fp8_weight_quantize PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    add_test(NAME csrc_fp8_weight_quantize
      COMMAND test_csrc_fp8_weight_quantize)
  endif()

  add_executable(test_safetensors_loader tests/test_safetensors_loader.cpp)
  target_link_libraries(test_safetensors_loader PRIVATE flashrt_cpp_loader)
  add_test(NAME safetensors_loader COMMAND test_safetensors_loader)

  add_executable(test_sha256 tests/test_sha256.cpp)
  target_link_libraries(test_sha256 PRIVATE flashrt_cpp_loader)
  add_test(NAME sha256 COMMAND test_sha256)

  add_executable(test_native_config tests/test_native_config.cpp)
  target_link_libraries(test_native_config PRIVATE flashrt_cpp_native)
  add_test(NAME native_config COMMAND test_native_config)

  add_executable(test_text_modalities tests/test_text_modalities.cpp)
  target_link_libraries(test_text_modalities PRIVATE flashrt_cpp_modalities)
  add_test(NAME text_modalities COMMAND test_text_modalities)

  add_executable(test_text_tokenizer tests/test_text_tokenizer.cpp)
  target_link_libraries(test_text_tokenizer PRIVATE flashrt_cpp_modalities)
  add_test(NAME text_tokenizer COMMAND test_text_tokenizer)

  add_executable(test_cpp_modalities tests/test_modalities.cpp)
  target_link_libraries(test_cpp_modalities
    PRIVATE flashrt_cpp_pi05 flashrt_cpp_modalities)
  add_test(NAME cpp_modalities COMMAND test_cpp_modalities)

  add_executable(test_pi05_runtime tests/test_pi05_runtime.cpp)
  target_link_libraries(test_pi05_runtime
    PRIVATE flashrt_cpp_pi05 flashrt_cpp_modalities)
  add_test(NAME pi05_runtime COMMAND test_pi05_runtime)

  add_executable(test_pi05_prompt_format tests/test_pi05_prompt_format.cpp)
  target_include_directories(test_pi05_prompt_format PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(test_pi05_prompt_format PRIVATE flashrt_cpp_pi05)
  add_test(NAME pi05_prompt_format COMMAND test_pi05_prompt_format)

  add_executable(test_pi05_prompt_embed tests/test_pi05_prompt_embed.cpp)
  target_include_directories(test_pi05_prompt_embed PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  if(FLASHRT_CPP_WITH_CUDA_KERNELS)
    target_compile_definitions(test_pi05_prompt_embed PRIVATE
      FLASHRT_CPP_TEST_WITH_CUDA_KERNELS=1)
  endif()
  target_link_libraries(test_pi05_prompt_embed PRIVATE flashrt_cpp_pi05)
  add_test(NAME pi05_prompt_embed COMMAND test_pi05_prompt_embed)

  add_executable(test_pi05_native_calibration
    tests/test_pi05_native_calibration.cpp)
  target_include_directories(test_pi05_native_calibration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(test_pi05_native_calibration PRIVATE flashrt_cpp_pi05)
  add_test(NAME pi05_native_calibration COMMAND test_pi05_native_calibration)

  add_executable(test_pi05_semantic_pipeline
    tests/test_pi05_semantic_pipeline.cpp)
  target_include_directories(test_pi05_semantic_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(test_pi05_semantic_pipeline
    PRIVATE flashrt_cpp_pi05)
  add_test(NAME pi05_semantic_pipeline COMMAND test_pi05_semantic_pipeline)

  add_executable(test_pi05_linear_weight_groups
    tests/test_pi05_linear_weight_groups.cpp)
  target_include_directories(test_pi05_linear_weight_groups PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(test_pi05_linear_weight_groups
    PRIVATE flashrt_cpp_pi05)
  add_test(NAME pi05_linear_weight_groups
    COMMAND test_pi05_linear_weight_groups)

  add_executable(test_pi05_execution_plan
    tests/test_pi05_execution_plan.cpp)
  target_include_directories(test_pi05_execution_plan PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(test_pi05_execution_plan
    PRIVATE flashrt_cpp_pi05)
  add_test(NAME pi05_execution_plan COMMAND test_pi05_execution_plan)

  add_executable(test_pi05_resolved_resources
    tests/test_pi05_resolved_resources.cpp)
  target_include_directories(test_pi05_resolved_resources PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(test_pi05_resolved_resources
    PRIVATE flashrt_cpp_pi05)
  add_test(NAME pi05_resolved_resources COMMAND test_pi05_resolved_resources)

  add_executable(test_pi05_native_resource_resolver
    tests/test_pi05_native_resource_resolver.cpp)
  target_include_directories(test_pi05_native_resource_resolver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(test_pi05_native_resource_resolver PRIVATE
    flashrt_cpp_pi05)
  add_test(NAME pi05_native_resource_resolver
    COMMAND test_pi05_native_resource_resolver)

  if(TARGET flashrt_cpp_pi05_target_sm120)
    add_executable(test_pi05_sm120_attention
      tests/test_pi05_sm120_attention.cpp)
    target_link_libraries(test_pi05_sm120_attention PRIVATE
      flashrt_cpp_pi05_target_sm120 flashrt_exec CUDA::cudart)
    add_test(NAME pi05_sm120_attention COMMAND test_pi05_sm120_attention)

    add_executable(test_pi05_sm120_bf16_target
      tests/test_pi05_sm120_bf16_target.cpp)
    target_link_libraries(test_pi05_sm120_bf16_target PRIVATE
      flashrt_cpp_pi05_target_sm120 flashrt_exec CUDA::cudart)
    add_test(NAME pi05_sm120_bf16_target
      COMMAND test_pi05_sm120_bf16_target)

    add_executable(test_pi05_sm120_fp8_weight_packer
      tests/test_pi05_sm120_fp8_weight_packer.cpp)
    target_include_directories(test_pi05_sm120_fp8_weight_packer PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
    target_link_libraries(test_pi05_sm120_fp8_weight_packer PRIVATE
      flashrt_cpp_pi05_target_sm120 flashrt_exec CUDA::cudart)
    add_test(NAME pi05_sm120_fp8_weight_packer
      COMMAND test_pi05_sm120_fp8_weight_packer)

    add_executable(test_pi05_sm120_fp8_linear
      tests/test_pi05_sm120_fp8_linear.cpp)
    target_include_directories(test_pi05_sm120_fp8_linear PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
    target_link_libraries(test_pi05_sm120_fp8_linear PRIVATE
      flashrt_cpp_pi05_target_sm120 flashrt_exec CUDA::cudart)
    add_test(NAME pi05_sm120_fp8_linear
      COMMAND test_pi05_sm120_fp8_linear)

    add_executable(test_pi05_sm120_target_bundle
      tests/test_pi05_sm120_target_bundle.cpp)
    target_link_libraries(test_pi05_sm120_target_bundle PRIVATE
      flashrt_cpp_pi05_target_sm120 flashrt_exec CUDA::cudart)
    add_test(NAME pi05_sm120_target_bundle
      COMMAND test_pi05_sm120_target_bundle)

    if(TARGET flashrt_cpp_pi05_capture)
      add_executable(test_pi05_sm120_native_session
        tests/test_pi05_sm120_native_session.cpp)
      target_link_libraries(test_pi05_sm120_native_session PRIVATE
        flashrt_cpp_pi05_target_sm120 flashrt_cpp_pi05_capture
        flashrt_exec CUDA::cudart)
      add_test(NAME pi05_sm120_native_session
        COMMAND test_pi05_sm120_native_session)
    endif()
  endif()

  if(TARGET flashrt_cpp_native_cuda)
    add_executable(test_native_cuda_graph_set
      tests/test_native_cuda_graph_set.cpp)
    target_link_libraries(test_native_cuda_graph_set
      PRIVATE flashrt_cpp_native_cuda)
    add_test(NAME native_cuda_graph_set COMMAND test_native_cuda_graph_set)

    add_executable(test_pi05_captured_program
      tests/test_pi05_captured_program.cpp)
    target_link_libraries(test_pi05_captured_program
      PRIVATE flashrt_cpp_pi05_capture)
    add_test(NAME pi05_captured_program COMMAND test_pi05_captured_program)

    add_executable(test_pi05_native_session
      tests/test_pi05_native_session.cpp)
    target_link_libraries(test_pi05_native_session
      PRIVATE flashrt_cpp_pi05_capture)
    add_test(NAME pi05_native_session COMMAND test_pi05_native_session)
  endif()

  if(FLASHRT_CPP_WITH_CUDA_STAGING)
    add_executable(test_device_staging tests/test_device_staging.cpp)
    if(FLASHRT_CPP_WITH_CUDA_KERNELS)
      target_compile_definitions(test_device_staging PRIVATE
        FLASHRT_CPP_TEST_WITH_CUDA_KERNELS=1)
    endif()
    target_link_libraries(test_device_staging
      PRIVATE flashrt_cpp_pi05 flashrt_cpp_modalities CUDA::cudart)
    add_test(NAME device_staging COMMAND test_device_staging)

    add_executable(test_pi05_c_api tests/test_pi05_c_api.cpp)
    target_link_libraries(test_pi05_c_api
      PRIVATE flashrt_cpp_pi05_c flashrt_exec CUDA::cudart)
    add_test(NAME pi05_c_api COMMAND test_pi05_c_api)

    add_executable(test_pi05_model_runtime tests/test_pi05_model_runtime.cpp)
    target_link_libraries(test_pi05_model_runtime
      PRIVATE flashrt_cpp_pi05_c flashrt_exec flashrt_runtime CUDA::cudart)
    add_test(NAME pi05_model_runtime COMMAND test_pi05_model_runtime)
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    get_property(_flashrt_cpp_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target IN LISTS _flashrt_cpp_targets)
      if(_target MATCHES "^test_")
        target_compile_options(${_target} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG>
          $<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>)
      endif()
    endforeach()
  endif()
endif()
