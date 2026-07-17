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
option(FLASHRT_CPP_WITH_FA2 "Enable the Python-free vendored FA2 driver" OFF)
option(FLASHRT_CPP_WITH_THOR_FP8
       "Enable the native Thor SM110 FP8 kernel backend" OFF)
option(FLASHRT_CPP_BUILD_PI05_DIAGNOSTICS
       "Build detailed PI0.5 real-checkpoint diagnostic probes" OFF)
set(FLASHRT_CPP_FA2_LIBRARY "" CACHE FILEPATH
    "Path to the Python-free libflashrt_fa2_raw shared library")
set(FLASHRT_CPP_CUTLASS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass" CACHE PATH
    "Path to CUTLASS for the native Thor SM110 FP8 backend")

if(FLASHRT_CPP_WITH_CUDA_STAGING)
  find_package(CUDAToolkit REQUIRED)
endif()
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
if(FLASHRT_CPP_WITH_FA2)
  if(NOT FLASHRT_CPP_WITH_CUDA_KERNELS)
    message(FATAL_ERROR "FLASHRT_CPP_WITH_FA2 requires CUDA kernels")
  endif()
  if(NOT FLASHRT_CPP_FA2_LIBRARY)
    unset(FLASHRT_CPP_FA2_LIBRARY CACHE)
    find_library(FLASHRT_CPP_FA2_LIBRARY
      NAMES flashrt_fa2_raw
      PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../flash_rt
      NO_DEFAULT_PATH)
  endif()
  if(NOT FLASHRT_CPP_FA2_LIBRARY)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_FA2 requires libflashrt_fa2_raw; build the root "
      "flashrt_fa2_raw target or set FLASHRT_CPP_FA2_LIBRARY")
  endif()
  add_library(flashrt_cpp_fa2_external SHARED IMPORTED GLOBAL)
  set_target_properties(flashrt_cpp_fa2_external PROPERTIES
    IMPORTED_LOCATION ${FLASHRT_CPP_FA2_LIBRARY})
endif()
if(FLASHRT_CPP_WITH_THOR_FP8)
  if(NOT FLASHRT_CPP_WITH_CUDA_KERNELS)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_THOR_FP8 requires FLASHRT_CPP_WITH_CUDA_KERNELS")
  endif()
  if(NOT FLASHRT_CPP_WITH_CUDA_STAGING)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_THOR_FP8 requires FLASHRT_CPP_WITH_CUDA_STAGING")
  endif()
  if(NOT FLASHRT_CPP_WITH_EXEC)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_THOR_FP8 requires FLASHRT_CPP_WITH_EXEC")
  endif()
  if(NOT EXISTS
     "${FLASHRT_CPP_CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_THOR_FP8 requires CUTLASS; set "
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
  # Match the CUDA preprocess kernels' explicit round-to-nearest operations in
  # optimized builds; otherwise the host compiler may contract lerps to FMAs.
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
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  add_library(flashrt_cpp_native_cuda STATIC
    native/src/cuda_graph_set.cpp)
  target_include_directories(flashrt_cpp_native_cuda
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/native/include)
  target_link_libraries(flashrt_cpp_native_cuda
    PUBLIC flashrt_cpp_modalities flashrt_exec CUDA::cudart)
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

set(FLASHRT_CPP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
add_subdirectory(models/pi05)
