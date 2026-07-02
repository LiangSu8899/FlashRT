# FlashRT runtime-export layer — standalone build.
#
# Builds:
#   libflashrt_runtime.so  — builder + export lifetime (no CUDA, no exec link:
#                            the builder only records borrowed handles)
#   _flashrt_runtime*.so   — the pybind setup/dev module (if pybind11 is found)
#
# Standalone on purpose, mirroring exec/: it does NOT touch the root build.
#   cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build runtime/build -j
#
# Consumers (C++ hosts) need only runtime/include + exec/include headers; the
# export struct is plain data — nothing to link.

cmake_minimum_required(VERSION 3.18)
project(flashrt_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
include(CTest)

add_library(flashrt_runtime SHARED
  src/runtime_export.cpp)
target_include_directories(flashrt_runtime
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
         ${CMAKE_CURRENT_SOURCE_DIR}/../exec/include)

add_library(flashrt_runtime_modalities STATIC
  ../runtime_modalities/src/types.cpp
  ../runtime_modalities/src/vision_cpu.cpp
  ../runtime_modalities/src/action_cpu.cpp)
target_include_directories(flashrt_runtime_modalities
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../runtime_modalities/include)

add_library(flashrt_pi05_runtime_spec STATIC
  ../runtime_models/pi05/src/cpp_runtime_spec.cpp
  ../runtime_models/pi05/src/cpp_runtime_io.cpp)
target_include_directories(flashrt_pi05_runtime_spec
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../runtime_models/pi05/include)
target_link_libraries(flashrt_pi05_runtime_spec
  PUBLIC flashrt_runtime_modalities)

target_include_directories(flashrt_runtime
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../runtime_modalities/include)

if(BUILD_TESTING)
  add_executable(test_runtime_modalities tests/test_modalities.cpp)
  target_link_libraries(test_runtime_modalities
    PRIVATE flashrt_pi05_runtime_spec flashrt_runtime_modalities)
  add_test(NAME runtime_modalities COMMAND test_runtime_modalities)
endif()

# --- pybind setup/dev module (optional) ---
find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND AND Python3_Interpreter_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pb_dir OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _pb_rc)
  if(_pb_rc EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH "${_pb_dir}")
    find_package(pybind11 CONFIG QUIET)
  endif()
endif()

if(pybind11_FOUND)
  pybind11_add_module(_flashrt_runtime bindings/runtime_pybind.cpp)
  target_link_libraries(_flashrt_runtime PRIVATE flashrt_runtime)
else()
  message(STATUS "pybind11 not found — building the C ABI library only (no Python module).")
endif()
