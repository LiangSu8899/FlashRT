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
include(GNUInstallDirs)

option(FLASHRT_CPP_WITH_EXEC "Build/link the exec layer for native replay" ON)
option(FLASHRT_CPP_WITH_CUDA_STAGING "Enable conservative CUDA H2D/D2H modality staging" ON)
option(FLASHRT_CPP_WITH_CUDA_KERNELS "Enable CUDA modality kernels" ON)
option(FLASHRT_ENABLE_NATIVE_CPP
       "Build opt-in native C++ operation and model extensions" OFF)
option(FLASHRT_CPP_WITH_PI05
       "Build the PI0.5 native C++ model frontend" OFF)
option(FLASHRT_CPP_WITH_SENTENCEPIECE
       "Enable native SentencePiece text tokenization" OFF)
option(FLASHRT_CPP_WITH_PI05_SM120_TARGET
       "Build the PI0.5 SM120 frontend target" OFF)
option(FLASHRT_CPP_WITH_PI05_SM110_TARGET
       "Build the PI0.5 SM110 frontend target" OFF)
option(FLASHRT_CPP_WITH_LLAMA_CPP_PROVIDER
       "Build the optional llama.cpp provider frontend" OFF)
option(FLASHRT_CPP_WITH_JETSON_PI
       "Back the llama.cpp provider with Jetson-PI" OFF)
set(JETSON_PI_ROOT "" CACHE PATH
    "Jetson-PI source root for an in-tree provider build")
set(JetsonPI_ROOT "" CACHE PATH
    "Jetson-PI install prefix for a packaged provider build")
if((FLASHRT_CPP_WITH_PI05 OR FLASHRT_CPP_WITH_SENTENCEPIECE) AND
   NOT FLASHRT_ENABLE_NATIVE_CPP)
  message(FATAL_ERROR
    "Native tokenizer and model frontends require "
    "FLASHRT_ENABLE_NATIVE_CPP=ON")
endif()
if((FLASHRT_CPP_WITH_PI05_SM120_TARGET OR
    FLASHRT_CPP_WITH_PI05_SM110_TARGET) AND
   NOT FLASHRT_CPP_WITH_PI05)
  message(FATAL_ERROR
    "PI0.5 hardware targets require FLASHRT_CPP_WITH_PI05=ON")
endif()
if(FLASHRT_CPP_WITH_JETSON_PI AND
   NOT FLASHRT_CPP_WITH_LLAMA_CPP_PROVIDER)
  message(FATAL_ERROR
    "FLASHRT_CPP_WITH_JETSON_PI requires "
    "FLASHRT_CPP_WITH_LLAMA_CPP_PROVIDER=ON")
endif()
if(FLASHRT_ENABLE_NATIVE_CPP AND NOT FLASHRT_CPP_WITH_EXEC)
  message(FATAL_ERROR
    "FLASHRT_ENABLE_NATIVE_CPP requires FLASHRT_CPP_WITH_EXEC=ON")
endif()
if(FLASHRT_CPP_WITH_PI05_SM120_TARGET AND
   FLASHRT_CPP_WITH_PI05_SM110_TARGET)
  message(FATAL_ERROR
    "PI0.5 SM110 and SM120 targets use different CUDA compilation profiles; "
    "build one deployment target per build tree")
endif()
if(FLASHRT_CPP_WITH_PI05_SM110_TARGET)
  set(FLASHRT_CPP_PI05_CUDA_ARCH 110)
elseif(FLASHRT_CPP_WITH_PI05_SM120_TARGET)
  set(FLASHRT_CPP_PI05_CUDA_ARCH 120)
endif()
set(FLASHRT_CPP_FA2_LIBRARY "" CACHE FILEPATH
    "Path to the Python-free libflashrt_fa2_raw shared library")
set(FLASHRT_CPP_CUTLASS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass" CACHE PATH
    "Path to CUTLASS for the PI0.5 SM110 frontend target")
if(FLASHRT_CPP_WITH_CUDA_STAGING)
  find_package(CUDAToolkit REQUIRED)
endif()
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
if(FLASHRT_CPP_WITH_JETSON_PI)
  if(JETSON_PI_ROOT AND JetsonPI_ROOT)
    message(FATAL_ERROR
      "Set only one Jetson-PI integration root: JETSON_PI_ROOT for a "
      "source build or JetsonPI_ROOT for an installed package")
  endif()
  if(NOT JETSON_PI_ROOT AND NOT JetsonPI_ROOT)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_JETSON_PI requires JETSON_PI_ROOT or JetsonPI_ROOT")
  endif()

  if(JETSON_PI_ROOT)
    if(NOT EXISTS "${JETSON_PI_ROOT}/CMakeLists.txt")
      message(FATAL_ERROR
        "JETSON_PI_ROOT does not identify a Jetson-PI source tree")
    endif()
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
    set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
    set(LLAMA_HTTPLIB OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(JETSON_PI_BUILD_MTMD ON CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_TOOLS_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory("${JETSON_PI_ROOT}"
                     "${CMAKE_CURRENT_BINARY_DIR}/jetson-pi"
                     EXCLUDE_FROM_ALL)
    set(_flashrt_jetson_pi_pi0 jetson_pi_pi0)
    set(_flashrt_jetson_pi_llm jetson_pi_llm)
    set(_flashrt_jetson_pi_mllm jetson_pi_mllm)
  else()
    list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
    find_package(JetsonPI REQUIRED)
    list(POP_FRONT CMAKE_MODULE_PATH)
    set(_flashrt_jetson_pi_pi0 JetsonPI::jetson_pi_pi0)
    set(_flashrt_jetson_pi_llm JetsonPI::jetson_pi_llm)
    set(_flashrt_jetson_pi_mllm JetsonPI::jetson_pi_mllm)
  endif()
endif()
if(FLASHRT_CPP_WITH_PI05_SM120_TARGET)
  if(NOT FLASHRT_CPP_WITH_CUDA_KERNELS)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_PI05_SM120_TARGET requires CUDA kernels")
  endif()
  if(NOT FLASHRT_CPP_WITH_CUDA_STAGING)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_PI05_SM120_TARGET requires CUDA staging")
  endif()
  if(NOT FLASHRT_CPP_WITH_EXEC)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_PI05_SM120_TARGET requires the exec layer")
  endif()
  if(TARGET flashrt_fa2_raw)
    add_library(flashrt_cpp_fa2_external ALIAS flashrt_fa2_raw)
  else()
    if(NOT FLASHRT_CPP_FA2_LIBRARY)
      find_library(FLASHRT_CPP_FA2_LIBRARY
        NAMES flashrt_fa2_raw
        PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../flash_rt
        NO_DEFAULT_PATH)
    endif()
    if(NOT FLASHRT_CPP_FA2_LIBRARY OR
       NOT EXISTS "${FLASHRT_CPP_FA2_LIBRARY}")
      message(FATAL_ERROR
        "The PI0.5 SM120 target requires the opt-in FA2 native C library; "
        "build flashrt_fa2_raw or set FLASHRT_CPP_FA2_LIBRARY")
    endif()
    add_library(flashrt_cpp_fa2_external SHARED IMPORTED GLOBAL)
    set_target_properties(flashrt_cpp_fa2_external PROPERTIES
      IMPORTED_LOCATION ${FLASHRT_CPP_FA2_LIBRARY})
  endif()
endif()
if(FLASHRT_CPP_WITH_PI05_SM110_TARGET)
  if(NOT FLASHRT_CPP_WITH_CUDA_KERNELS)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_PI05_SM110_TARGET requires CUDA kernels")
  endif()
  if(NOT FLASHRT_CPP_WITH_CUDA_STAGING)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_PI05_SM110_TARGET requires CUDA staging")
  endif()
  if(NOT FLASHRT_CPP_WITH_EXEC)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_PI05_SM110_TARGET requires the exec layer")
  endif()
  if(NOT EXISTS
     "${FLASHRT_CPP_CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_PI05_SM110_TARGET requires CUTLASS; set "
      "FLASHRT_CPP_CUTLASS_DIR to a compatible checkout")
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
  modalities/src/action_cpu.cpp)
if(FLASHRT_ENABLE_NATIVE_CPP)
  list(APPEND FLASHRT_CPP_MODALITY_SRCS modalities/src/text_cpu.cpp)
  if(FLASHRT_CPP_WITH_SENTENCEPIECE)
    list(APPEND FLASHRT_CPP_MODALITY_SRCS
      modalities/src/sentencepiece_tokenizer.cpp)
  else()
    list(APPEND FLASHRT_CPP_MODALITY_SRCS
      modalities/src/tokenizer_unavailable.cpp)
  endif()
endif()
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  list(APPEND FLASHRT_CPP_MODALITY_SRCS modalities/src/vision_cuda.cu)
  if(FLASHRT_ENABLE_NATIVE_CPP)
    list(APPEND FLASHRT_CPP_MODALITY_SRCS modalities/src/text_cuda.cu)
  endif()
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
  if(FLASHRT_CPP_PI05_CUDA_ARCH)
    set_target_properties(flashrt_cpp_modalities PROPERTIES
      CUDA_ARCHITECTURES ${FLASHRT_CPP_PI05_CUDA_ARCH})
  endif()
endif()

add_library(flashrt_cpp_vla INTERFACE)
target_include_directories(flashrt_cpp_vla
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/families/vla/include)
target_link_libraries(flashrt_cpp_vla
  INTERFACE flashrt_cpp_modalities flashrt_cpp)

if(FLASHRT_CPP_WITH_LLAMA_CPP_PROVIDER)
  add_library(flashrt_cpp_llama_cpp_provider STATIC
    providers/llama_cpp/src/checkpoint_identity.cpp
    providers/llama_cpp/src/pi0_runtime.cpp
    providers/llama_cpp/src/llm_runtime.cpp
    providers/llama_cpp/src/mllm_runtime.cpp)
  target_include_directories(flashrt_cpp_llama_cpp_provider
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/providers/llama_cpp/include)
  target_link_libraries(flashrt_cpp_llama_cpp_provider
    PUBLIC flashrt_runtime)

  if(FLASHRT_CPP_WITH_JETSON_PI)
    target_sources(flashrt_cpp_llama_cpp_provider PRIVATE
      providers/llama_cpp/src/jetson_pi_engine.cpp
      providers/llama_cpp/src/jetson_pi_link_check.cpp)
    target_compile_definitions(flashrt_cpp_llama_cpp_provider PRIVATE
      FLASHRT_CPP_WITH_JETSON_PI=1)
    target_link_libraries(flashrt_cpp_llama_cpp_provider PRIVATE
      ${_flashrt_jetson_pi_pi0}
      ${_flashrt_jetson_pi_llm}
      ${_flashrt_jetson_pi_mllm})

    add_library(flashrt_cpp_llama_cpp_provider_c SHARED
      providers/llama_cpp/src/checkpoint_identity.cpp
      providers/llama_cpp/src/pi0_runtime.cpp
      providers/llama_cpp/src/llm_runtime.cpp
      providers/llama_cpp/src/mllm_runtime.cpp
      providers/llama_cpp/src/jetson_pi_engine.cpp
      providers/llama_cpp/src/provider_open.cpp)
    target_include_directories(flashrt_cpp_llama_cpp_provider_c PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/providers/llama_cpp/include)
    target_compile_definitions(flashrt_cpp_llama_cpp_provider_c PRIVATE
      FLASHRT_CPP_WITH_JETSON_PI=1)
    target_compile_options(flashrt_cpp_llama_cpp_provider_c PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)
    target_link_libraries(flashrt_cpp_llama_cpp_provider_c PRIVATE
      flashrt_runtime
      ${_flashrt_jetson_pi_pi0}
      ${_flashrt_jetson_pi_llm}
      ${_flashrt_jetson_pi_mllm})
    if(UNIX AND NOT APPLE)
      target_link_options(flashrt_cpp_llama_cpp_provider_c PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/providers/llama_cpp/provider_exports.map"
        "LINKER:--disable-new-dtags")
    endif()
    set_target_properties(flashrt_cpp_llama_cpp_provider_c PROPERTIES
      INSTALL_RPATH "$ORIGIN")
    install(TARGETS flashrt_cpp_llama_cpp_provider_c flashrt_runtime
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include/flashrt/model_runtime.h
      ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include/flashrt/runtime.h
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flashrt)
  endif()
endif()

if(FLASHRT_ENABLE_NATIVE_CPP)
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
endif()

if(FLASHRT_CPP_WITH_PI05_SM120_TARGET OR
   FLASHRT_CPP_WITH_PI05_SM110_TARGET)
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
    if(FLASHRT_CPP_PI05_CUDA_ARCH)
      set_target_properties(${target} PROPERTIES
        CUDA_ARCHITECTURES ${FLASHRT_CPP_PI05_CUDA_ARCH})
    endif()
  endfunction()

  add_library(flashrt_native_cpp_cuda_support STATIC
    ${_flashrt_csrc_dir}/native_cpp/operations.cu
    ${_flashrt_csrc_dir}/native_cpp/fp8_quantize.cu
    ${_flashrt_csrc_dir}/native_cpp/attention_seqused.cu)
  _flashrt_configure_csrc_target(flashrt_native_cpp_cuda_support)
  target_include_directories(flashrt_native_cpp_cuda_support
    PUBLIC ${_flashrt_csrc_dir}/native_cpp/include
    PRIVATE ${_flashrt_csrc_dir}/native_cpp)
  set_target_properties(flashrt_native_cpp_cuda_support PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)

  add_library(flashrt_cpp_csrc_core STATIC
    ${_flashrt_csrc_dir}/gemm/gemm_runner.cu
    ${_flashrt_csrc_dir}/kernels/activation.cu
    ${_flashrt_csrc_dir}/kernels/elementwise.cu
    ${_flashrt_csrc_dir}/kernels/norm.cu
    ${_flashrt_csrc_dir}/kernels/patch_embed.cu
    ${_flashrt_csrc_dir}/kernels/quantize.cu
    ${_flashrt_csrc_dir}/kernels/rope.cu)
  _flashrt_configure_csrc_target(flashrt_cpp_csrc_core)

  if(FLASHRT_CPP_WITH_PI05_SM120_TARGET)
    add_library(flashrt_cpp_csrc_sm120 STATIC
      ${_flashrt_csrc_dir}/kernels/dit_bf16.cu
      ${_flashrt_csrc_dir}/kernels/fusion.cu)
    _flashrt_configure_csrc_target(flashrt_cpp_csrc_sm120)
    target_link_libraries(flashrt_cpp_csrc_sm120 PUBLIC
      flashrt_cpp_csrc_core)
  endif()

  if(FLASHRT_CPP_WITH_PI05_SM110_TARGET)
    add_library(flashrt_cpp_csrc_sm110_precise OBJECT
      ${_flashrt_csrc_dir}/attention/fmha_fp16_strided.cu)
    target_include_directories(flashrt_cpp_csrc_sm110_precise PRIVATE
      ${_flashrt_csrc_dir}/attention
      ${FLASHRT_CPP_CUTLASS_DIR}/examples/77_blackwell_fmha
      ${FLASHRT_CPP_CUTLASS_DIR}/include
      ${FLASHRT_CPP_CUTLASS_DIR}/tools/util/include)
    target_compile_options(flashrt_cpp_csrc_sm110_precise PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr --expt-extended-lambda -O3>)
    set_target_properties(flashrt_cpp_csrc_sm110_precise PROPERTIES
      CUDA_ARCHITECTURES 110a
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)

    add_library(flashrt_cpp_csrc_sm110 STATIC
      $<TARGET_OBJECTS:flashrt_cpp_csrc_sm110_precise>
      ${_flashrt_csrc_dir}/gemm/cutlass_sm100.cu
      ${_flashrt_csrc_dir}/kernels/attention_cublas.cu
      ${_flashrt_csrc_dir}/kernels/decoder_fused.cu
      ${_flashrt_csrc_dir}/kernels/softmax.cu)
    _flashrt_configure_csrc_target(flashrt_cpp_csrc_sm110)
    target_include_directories(flashrt_cpp_csrc_sm110 PRIVATE
      ${FLASHRT_CPP_CUTLASS_DIR}/examples/77_blackwell_fmha
      ${FLASHRT_CPP_CUTLASS_DIR}/include
      ${FLASHRT_CPP_CUTLASS_DIR}/tools/util/include)
    target_compile_options(flashrt_cpp_csrc_sm110 PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda --use_fast_math>)
    set_target_properties(flashrt_cpp_csrc_sm110 PROPERTIES
      CUDA_ARCHITECTURES 110a
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(flashrt_cpp_csrc_sm110 PUBLIC
      flashrt_cpp_csrc_core)
  endif()
endif()

if(FLASHRT_ENABLE_NATIVE_CPP)
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
endif()

if(FLASHRT_CPP_WITH_PI05)
  set(FLASHRT_CPP_PI05_SUPPORT_SRCS
    models/pi05/src/support/native_rope_unavailable.cpp)
  if(FLASHRT_CPP_WITH_PI05_SM120_TARGET OR
     FLASHRT_CPP_WITH_PI05_SM110_TARGET)
    set(FLASHRT_CPP_PI05_SUPPORT_SRCS
      models/pi05/src/support/native_rope.cu)
  endif()

  add_library(flashrt_cpp_pi05 STATIC
    models/pi05/src/spec.cpp
    models/pi05/src/io.cpp
    models/pi05/src/runtime.cpp
    models/pi05/src/model/dims.cpp
    models/pi05/src/model/execution_plan.cpp
    models/pi05/src/model/linear_weight_groups.cpp
    models/pi05/src/model/prompt_embed.cpp
    models/pi05/src/model/prompt_format.cpp
    models/pi05/src/model/resolved_resources.cpp
    models/pi05/src/model/semantic_pipeline.cpp
    models/pi05/src/support/native_calibration.cpp
    models/pi05/src/support/native_device_weights.cpp
    models/pi05/src/support/native_float16.cpp
    models/pi05/src/support/native_weight_materializer.cpp
    models/pi05/src/support/native_resource_resolver.cpp
    models/pi05/src/support/native_weight_ops.cpp
    models/pi05/src/support/native_weights.cpp
    models/pi05/src/support/native_workspace.cpp
    ${FLASHRT_CPP_PI05_SUPPORT_SRCS})
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
    target_link_libraries(flashrt_cpp_pi05 PRIVATE CUDA::cudart)
    if(TARGET flashrt_native_cpp_cuda_support)
      target_link_libraries(flashrt_cpp_pi05 PRIVATE
        flashrt_native_cpp_cuda_support)
    endif()
    set_target_properties(flashrt_cpp_pi05 PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    if(FLASHRT_CPP_PI05_CUDA_ARCH)
      set_target_properties(flashrt_cpp_pi05 PROPERTIES
        CUDA_ARCHITECTURES ${FLASHRT_CPP_PI05_CUDA_ARCH})
    endif()
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
    models/pi05/src/targets/sm120/fp8_autotune.cpp
    models/pi05/src/targets/sm120/bf16_scratch.cpp
    models/pi05/src/targets/sm120/fp8_linear.cpp
    models/pi05/src/targets/sm120/fp8_weight_packer.cpp
    models/pi05/src/targets/sm120/target.cpp)
  target_include_directories(flashrt_cpp_pi05_target_sm120
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(flashrt_cpp_pi05_target_sm120
    PUBLIC flashrt_cpp_pi05
    PRIVATE flashrt_cpp_loader flashrt_cpp_csrc_core flashrt_cpp_csrc_sm120
            flashrt_native_cpp_cuda_support flashrt_cpp_fa2_external
            CUDA::cudart)
  endif()

  if(FLASHRT_CPP_WITH_PI05_SM110_TARGET)
  add_library(flashrt_cpp_pi05_target_sm110 STATIC
    models/pi05/src/targets/sm110/frontend_bindings.cpp
    models/pi05/src/targets/sm110/fp8_weight_packer.cpp
    models/pi05/src/targets/sm110/operation_driver.cu
    models/pi05/src/targets/sm110/physical_resources.cpp
    models/pi05/src/targets/sm110/target.cpp)
  target_include_directories(flashrt_cpp_pi05_target_sm110
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(flashrt_cpp_pi05_target_sm110
    PUBLIC flashrt_cpp_pi05
    PRIVATE flashrt_cpp_loader flashrt_cpp_csrc_core
            flashrt_cpp_csrc_sm110 flashrt_native_cpp_cuda_support
            CUDA::cublas CUDA::cudart)
  set_target_properties(flashrt_cpp_pi05_target_sm110 PROPERTIES
    CUDA_ARCHITECTURES 110a
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
  endif()

  if(FLASHRT_CPP_WITH_PI05_SM120_TARGET OR
     FLASHRT_CPP_WITH_PI05_SM110_TARGET)
    add_library(flashrt_cpp_pi05_c SHARED
      models/pi05/src/c_api.cpp
      models/pi05/src/model_runtime.cpp)
    target_link_libraries(flashrt_cpp_pi05_c
      PUBLIC flashrt_cpp_pi05 flashrt_runtime)
    target_include_directories(flashrt_cpp_pi05_c
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
    target_compile_definitions(flashrt_cpp_pi05_c
      PRIVATE FLASHRT_PI05_C_BUILDING_LIBRARY=1)
    set_target_properties(flashrt_cpp_pi05_c PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      CUDA_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN YES)
    if(UNIX AND NOT APPLE)
      target_link_options(flashrt_cpp_pi05_c PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/pi05_c_exports.map"
        "LINKER:--exclude-libs,ALL")
      add_custom_command(TARGET flashrt_cpp_pi05_c POST_BUILD
        COMMAND ${CMAKE_COMMAND}
          -DNM_TOOL=${CMAKE_NM}
          -DINPUT_FILE=$<TARGET_FILE:flashrt_cpp_pi05_c>
          -DALLOWED_SYMBOLS_FILE=${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/pi05_c_exports.txt
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/verify_dynamic_exports.cmake
        VERBATIM)
    endif()
    if(TARGET flashrt_cpp_pi05_capture)
      target_link_libraries(flashrt_cpp_pi05_c PRIVATE
        flashrt_cpp_pi05_capture)
    endif()
    if(TARGET flashrt_cpp_pi05_capture AND
       (TARGET flashrt_cpp_pi05_target_sm120 OR
        TARGET flashrt_cpp_pi05_target_sm110))
      target_sources(flashrt_cpp_pi05_c PRIVATE
        models/pi05/src/native_calibration_c_api.cpp
        models/pi05/src/native_model_runtime.cpp
        models/pi05/src/native_open.cpp
        models/pi05/src/native_calibration_session.cpp
        models/pi05/src/native_session_factory.cpp)
      target_link_libraries(flashrt_cpp_pi05_c PRIVATE
        flashrt_cpp_loader flashrt_cpp_native)
    endif()
    if(TARGET flashrt_cpp_pi05_target_sm120)
      target_compile_definitions(flashrt_cpp_pi05_c
        PRIVATE FLASHRT_CPP_WITH_PI05_SM120_TARGET=1)
      target_link_libraries(flashrt_cpp_pi05_c
        PRIVATE flashrt_cpp_pi05_target_sm120)
    endif()
    if(TARGET flashrt_cpp_pi05_target_sm110)
      target_compile_definitions(flashrt_cpp_pi05_c
        PRIVATE FLASHRT_CPP_WITH_PI05_SM110_TARGET=1)
      target_link_libraries(flashrt_cpp_pi05_c
        PRIVATE flashrt_cpp_pi05_target_sm110)
    endif()
    set_target_properties(flashrt_cpp_pi05_c PROPERTIES
      INSTALL_RPATH "$ORIGIN")
    install(TARGETS flashrt_cpp_pi05_c flashrt_runtime flashrt_exec
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/../exec/include/flashrt/exec.h
      ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include/flashrt/runtime.h
      ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include/flashrt/model_runtime.h
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flashrt)
    install(FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include/flashrt/cpp/models/pi05/export.h
      ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include/flashrt/cpp/models/pi05/c_api.h
      ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include/flashrt/cpp/models/pi05/model_runtime.h
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flashrt/cpp/models/pi05)
  endif()
endif()

if(BUILD_TESTING)
  if(FLASHRT_CPP_WITH_LLAMA_CPP_PROVIDER)
    add_executable(test_llama_cpp_provider
      tests/test_llama_cpp_provider.cpp)
    target_link_libraries(test_llama_cpp_provider
      PRIVATE flashrt_cpp_llama_cpp_provider flashrt_runtime)
    add_test(NAME llama_cpp_provider COMMAND test_llama_cpp_provider)

    if(FLASHRT_CPP_WITH_JETSON_PI)
      add_executable(test_llama_cpp_jetson_pi_link
        tests/test_llama_cpp_jetson_pi_link.cpp)
      target_link_libraries(test_llama_cpp_jetson_pi_link
        PRIVATE flashrt_cpp_llama_cpp_provider)
      add_test(NAME llama_cpp_jetson_pi_link
        COMMAND test_llama_cpp_jetson_pi_link)

      if(UNIX AND NOT APPLE)
        add_test(NAME llama_cpp_provider_exports
          COMMAND ${CMAKE_COMMAND}
            -DPROVIDER_DSO=$<TARGET_FILE:flashrt_cpp_llama_cpp_provider_c>
            -DCMAKE_NM_TOOL=${CMAKE_NM}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_llama_cpp_provider_exports.cmake)
      endif()

      add_executable(test_llama_cpp_jetson_pi_llm
        tests/test_llama_cpp_jetson_pi_llm.cpp)
      target_link_libraries(test_llama_cpp_jetson_pi_llm
        PRIVATE flashrt_cpp_llama_cpp_provider ${_flashrt_jetson_pi_llm})
      add_test(NAME llama_cpp_jetson_pi_llm
        COMMAND test_llama_cpp_jetson_pi_llm)

      add_executable(test_llama_cpp_jetson_pi_mllm
        tests/test_llama_cpp_jetson_pi_mllm.cpp)
      target_link_libraries(test_llama_cpp_jetson_pi_mllm
        PRIVATE flashrt_cpp_llama_cpp_provider ${_flashrt_jetson_pi_mllm})
      add_test(NAME llama_cpp_jetson_pi_mllm
        COMMAND test_llama_cpp_jetson_pi_mllm)

      add_executable(test_llama_cpp_jetson_pi_llm_parity
        tests/test_llama_cpp_jetson_pi_llm_parity.cpp)
      target_link_libraries(test_llama_cpp_jetson_pi_llm_parity
        PRIVATE flashrt_cpp_llama_cpp_provider ${_flashrt_jetson_pi_llm})
      add_test(NAME llama_cpp_jetson_pi_llm_parity
        COMMAND test_llama_cpp_jetson_pi_llm_parity)

      add_executable(test_llama_cpp_jetson_pi_mllm_parity
        tests/test_llama_cpp_jetson_pi_mllm_parity.cpp)
      target_link_libraries(test_llama_cpp_jetson_pi_mllm_parity
        PRIVATE flashrt_cpp_llama_cpp_provider ${_flashrt_jetson_pi_mllm})
      add_test(NAME llama_cpp_jetson_pi_mllm_parity
        COMMAND test_llama_cpp_jetson_pi_mllm_parity)

      if(JETSON_PI_ROOT AND EXISTS "${JETSON_PI_ROOT}/vendor")
        add_executable(test_llama_cpp_jetson_pi_engine
          tests/test_llama_cpp_jetson_pi_engine.cpp)
        target_include_directories(test_llama_cpp_jetson_pi_engine PRIVATE
          ${JETSON_PI_ROOT}/vendor)
        target_link_libraries(test_llama_cpp_jetson_pi_engine
          PRIVATE flashrt_cpp_llama_cpp_provider ${_flashrt_jetson_pi_pi0})
        add_test(NAME llama_cpp_jetson_pi_engine
          COMMAND test_llama_cpp_jetson_pi_engine)

        add_executable(test_llama_cpp_jetson_pi_parity
          tests/test_llama_cpp_jetson_pi_parity.cpp)
        target_include_directories(test_llama_cpp_jetson_pi_parity PRIVATE
          ${JETSON_PI_ROOT}/vendor)
        target_link_libraries(test_llama_cpp_jetson_pi_parity
          PRIVATE flashrt_cpp_llama_cpp_provider ${_flashrt_jetson_pi_pi0})
        add_test(NAME llama_cpp_jetson_pi_parity
          COMMAND test_llama_cpp_jetson_pi_parity)

        # Opt-in PI0.5 reference-policy state-parity test. CI skips it when the
        # external checkpoint and fixture environment are unavailable.
        add_executable(test_llama_cpp_jetson_pi_state_parity
          tests/test_llama_cpp_jetson_pi_state_parity.cpp)
        target_include_directories(test_llama_cpp_jetson_pi_state_parity PRIVATE
          ${JETSON_PI_ROOT}/vendor)
        target_link_libraries(test_llama_cpp_jetson_pi_state_parity
          PRIVATE flashrt_cpp_llama_cpp_provider ${_flashrt_jetson_pi_pi0})
        add_test(NAME llama_cpp_jetson_pi_state_parity
          COMMAND test_llama_cpp_jetson_pi_state_parity)
      endif()
    endif()
  endif()

  if(TARGET flashrt_native_cpp_cuda_support)
    add_executable(test_csrc_rope_table tests/test_csrc_rope_table.cu)
    target_link_libraries(test_csrc_rope_table PRIVATE
      flashrt_native_cpp_cuda_support CUDA::cudart)
    set_target_properties(test_csrc_rope_table PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    add_test(NAME csrc_rope_table COMMAND test_csrc_rope_table)

    if(TARGET flashrt_cpp_csrc_sm110)
      add_executable(test_csrc_attention_seqused
        tests/test_csrc_attention_seqused.cu)
      target_link_libraries(test_csrc_attention_seqused PRIVATE
        flashrt_native_cpp_cuda_support flashrt_cpp_csrc_sm110
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
  endif()

  if(TARGET flashrt_cpp_csrc_core)
    add_executable(test_csrc_bf16_operations
      tests/test_csrc_bf16_operations.cu)
    target_link_libraries(test_csrc_bf16_operations PRIVATE
      flashrt_native_cpp_cuda_support CUDA::cudart)
    set_target_properties(test_csrc_bf16_operations PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    add_test(NAME csrc_bf16_operations COMMAND test_csrc_bf16_operations)

    add_executable(test_csrc_fp8_weight_quantize
      tests/test_csrc_fp8_weight_quantize.cu)
    target_include_directories(test_csrc_fp8_weight_quantize PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/native_cpp)
    target_link_libraries(test_csrc_fp8_weight_quantize PRIVATE
      flashrt_native_cpp_cuda_support flashrt_cpp_csrc_core
      flashrt_runtime CUDA::cudart)
    set_target_properties(test_csrc_fp8_weight_quantize PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    if(FLASHRT_CPP_PI05_CUDA_ARCH)
      set_target_properties(test_csrc_fp8_weight_quantize PROPERTIES
        CUDA_ARCHITECTURES ${FLASHRT_CPP_PI05_CUDA_ARCH})
    endif()
    add_test(NAME csrc_fp8_weight_quantize
      COMMAND test_csrc_fp8_weight_quantize)
  endif()

  if(FLASHRT_ENABLE_NATIVE_CPP)
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
  endif()

  if(FLASHRT_CPP_WITH_PI05)
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

    if(FLASHRT_CPP_WITH_PI05_SM120_TARGET OR
       FLASHRT_CPP_WITH_PI05_SM110_TARGET)
      add_executable(test_pi05_c_api tests/test_pi05_c_api.cpp)
      target_link_libraries(test_pi05_c_api
        PRIVATE flashrt_cpp_pi05_c flashrt_exec CUDA::cudart)
      add_test(NAME pi05_c_api COMMAND test_pi05_c_api)

      add_executable(test_pi05_model_runtime tests/test_pi05_model_runtime.cpp)
      target_link_libraries(test_pi05_model_runtime
        PRIVATE flashrt_cpp_pi05_c flashrt_exec flashrt_runtime CUDA::cudart)
      add_test(NAME pi05_model_runtime COMMAND test_pi05_model_runtime)

      add_executable(test_pi05_native_open tests/test_pi05_native_open.cpp)
      target_include_directories(test_pi05_native_open PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
      target_link_libraries(test_pi05_native_open
        PRIVATE flashrt_cpp_pi05_c flashrt_runtime)
      if(FLASHRT_CPP_WITH_SENTENCEPIECE)
        target_compile_definitions(test_pi05_native_open
          PRIVATE FLASHRT_CPP_PI05_NATIVE_OPEN_ENABLED=1)
      endif()
      add_test(NAME pi05_native_open COMMAND test_pi05_native_open)
    endif()
  endif()
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
