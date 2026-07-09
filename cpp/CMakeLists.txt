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
option(FLASHRT_CPP_WITH_SENTENCEPIECE "Enable native SentencePiece text tokenization" OFF)
if(FLASHRT_CPP_WITH_CUDA_STAGING)
  find_package(CUDAToolkit REQUIRED)
endif()
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
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

add_library(flashrt_cpp_loader STATIC
  loader/src/safetensors.cpp)
target_include_directories(flashrt_cpp_loader
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/loader/include)

set(FLASHRT_CPP_PI05_SRCS
  models/pi05/src/spec.cpp
  models/pi05/src/native_weights.cpp
  models/pi05/src/native_weight_ops.cpp
  models/pi05/src/native_device_weights.cpp
  models/pi05/src/native_weight_packer.cpp
  models/pi05/src/native_weight_materializer.cpp
  models/pi05/src/prompt_format.cpp
  models/pi05/src/prompt_embed.cpp
  models/pi05/src/io.cpp
  models/pi05/src/runtime.cpp)
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  list(APPEND FLASHRT_CPP_PI05_SRCS
    models/pi05/src/native_quantization.cu)
else()
  list(APPEND FLASHRT_CPP_PI05_SRCS
    models/pi05/src/native_quantization_unavailable.cpp)
endif()

add_library(flashrt_cpp_pi05 STATIC ${FLASHRT_CPP_PI05_SRCS})
target_include_directories(flashrt_cpp_pi05
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include)
target_link_libraries(flashrt_cpp_pi05
  PUBLIC flashrt_cpp_modalities flashrt_cpp_vla flashrt_cpp_loader)

if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  add_library(flashrt_cpp_pi05_kernels STATIC
    models/pi05/src/native_kernel_driver.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/gemm/gemm_runner.cu)
  target_include_directories(flashrt_cpp_pi05_kernels
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/gemm)
  target_link_libraries(flashrt_cpp_pi05_kernels
    PUBLIC flashrt_cpp_modalities CUDA::cublas CUDA::cublasLt CUDA::cudart)
  set_target_properties(flashrt_cpp_pi05_kernels PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
  target_compile_options(flashrt_cpp_pi05_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr -O3
      --ftz=true --prec-div=false --prec-sqrt=false>)
endif()

add_library(flashrt_cpp_pi05_c SHARED
  models/pi05/src/c_api.cpp
  models/pi05/src/model_runtime.cpp
  models/pi05/src/native_open.cpp)
target_link_libraries(flashrt_cpp_pi05_c
  PUBLIC flashrt_cpp_pi05 flashrt_runtime)
target_include_directories(flashrt_cpp_pi05_c
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include)
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  target_link_libraries(flashrt_cpp_pi05_c
    PRIVATE flashrt_cpp_pi05_kernels)
endif()

if(BUILD_TESTING)
  add_executable(test_safetensors_loader tests/test_safetensors_loader.cpp)
  target_link_libraries(test_safetensors_loader PRIVATE flashrt_cpp_loader)
  add_test(NAME safetensors_loader COMMAND test_safetensors_loader)

  add_executable(test_cpp_modalities tests/test_modalities.cpp)
  target_link_libraries(test_cpp_modalities
    PRIVATE flashrt_cpp_pi05 flashrt_cpp_modalities)
  add_test(NAME cpp_modalities COMMAND test_cpp_modalities)

  add_executable(test_text_modalities tests/test_text_modalities.cpp)
  target_link_libraries(test_text_modalities PRIVATE flashrt_cpp_modalities)
  add_test(NAME text_modalities COMMAND test_text_modalities)

  add_executable(test_text_tokenizer tests/test_text_tokenizer.cpp)
  target_link_libraries(test_text_tokenizer PRIVATE flashrt_cpp_modalities)
  add_test(NAME text_tokenizer COMMAND test_text_tokenizer)

  add_executable(test_pi05_runtime tests/test_pi05_runtime.cpp)
  target_link_libraries(test_pi05_runtime
    PRIVATE flashrt_cpp_pi05 flashrt_cpp_modalities)
  add_test(NAME pi05_runtime COMMAND test_pi05_runtime)

  add_executable(test_pi05_native_weight_ops
    tests/test_pi05_native_weight_ops.cpp)
  target_link_libraries(test_pi05_native_weight_ops PRIVATE flashrt_cpp_pi05)
  add_test(NAME pi05_native_weight_ops COMMAND test_pi05_native_weight_ops)

  add_executable(pi05_native_weight_probe
    tests/pi05_native_weight_probe.cpp)
  target_link_libraries(pi05_native_weight_probe PRIVATE flashrt_cpp_pi05)

  if(FLASHRT_CPP_WITH_CUDA_KERNELS)
    add_executable(test_pi05_native_quantization
      tests/test_pi05_native_quantization.cpp)
    target_link_libraries(test_pi05_native_quantization PRIVATE flashrt_cpp_pi05)
    add_test(NAME pi05_native_quantization
      COMMAND test_pi05_native_quantization)

    add_executable(pi05_native_quant_probe
      tests/pi05_native_quant_probe.cpp)
    target_link_libraries(pi05_native_quant_probe PRIVATE flashrt_cpp_pi05)

    add_executable(test_pi05_native_weight_packer
      tests/test_pi05_native_weight_packer.cpp)
    target_link_libraries(test_pi05_native_weight_packer
      PRIVATE flashrt_cpp_pi05 flashrt_exec CUDA::cudart)
    add_test(NAME pi05_native_weight_packer
      COMMAND test_pi05_native_weight_packer)

    add_executable(test_pi05_native_kernel_driver
      tests/test_pi05_native_kernel_driver.cpp)
    target_link_libraries(test_pi05_native_kernel_driver
      PRIVATE flashrt_cpp_pi05_kernels flashrt_exec CUDA::cudart)
    add_test(NAME pi05_native_kernel_driver
      COMMAND test_pi05_native_kernel_driver)
  endif()

  add_executable(test_pi05_prompt_format tests/test_pi05_prompt_format.cpp)
  target_link_libraries(test_pi05_prompt_format PRIVATE flashrt_cpp_pi05)
  add_test(NAME pi05_prompt_format COMMAND test_pi05_prompt_format)

  add_executable(test_pi05_prompt_embed tests/test_pi05_prompt_embed.cpp)
  target_link_libraries(test_pi05_prompt_embed PRIVATE flashrt_cpp_pi05)
  add_test(NAME pi05_prompt_embed COMMAND test_pi05_prompt_embed)

  if(FLASHRT_CPP_WITH_CUDA_STAGING)
    add_executable(test_device_staging tests/test_device_staging.cpp)
    target_link_libraries(test_device_staging
      PRIVATE flashrt_cpp_pi05 flashrt_cpp_modalities CUDA::cudart)
    add_test(NAME device_staging COMMAND test_device_staging)

    add_executable(test_pi05_native_device_weights
      tests/test_pi05_native_device_weights.cpp)
    target_link_libraries(test_pi05_native_device_weights
      PRIVATE flashrt_cpp_pi05 flashrt_exec CUDA::cudart)
    add_test(NAME pi05_native_device_weights
      COMMAND test_pi05_native_device_weights)

    add_executable(test_pi05_native_weight_materializer
      tests/test_pi05_native_weight_materializer.cpp)
    target_link_libraries(test_pi05_native_weight_materializer
      PRIVATE flashrt_cpp_pi05 flashrt_exec CUDA::cudart)
    add_test(NAME pi05_native_weight_materializer
      COMMAND test_pi05_native_weight_materializer)

    add_executable(test_pi05_c_api tests/test_pi05_c_api.cpp)
    target_link_libraries(test_pi05_c_api
      PRIVATE flashrt_cpp_pi05_c flashrt_exec CUDA::cudart)
    add_test(NAME pi05_c_api COMMAND test_pi05_c_api)

    add_executable(test_pi05_model_runtime tests/test_pi05_model_runtime.cpp)
    target_link_libraries(test_pi05_model_runtime
      PRIVATE flashrt_cpp_pi05_c flashrt_exec flashrt_runtime CUDA::cudart)
    add_test(NAME pi05_model_runtime COMMAND test_pi05_model_runtime)

    add_executable(test_pi05_native_open tests/test_pi05_native_open.cpp)
    target_link_libraries(test_pi05_native_open
      PRIVATE flashrt_cpp_pi05_c flashrt_runtime)
    add_test(NAME pi05_native_open COMMAND test_pi05_native_open)
  endif()
endif()
