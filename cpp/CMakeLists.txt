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
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
include(CTest)

add_library(flashrt_cpp INTERFACE)
target_include_directories(flashrt_cpp
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../exec/include)

add_library(flashrt_cpp_modalities STATIC
  modalities/src/types.cpp
  modalities/src/vision_cpu.cpp
  modalities/src/action_cpu.cpp)
target_include_directories(flashrt_cpp_modalities
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/modalities/include)

add_library(flashrt_cpp_vla INTERFACE)
target_include_directories(flashrt_cpp_vla
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/families/vla/include)
target_link_libraries(flashrt_cpp_vla
  INTERFACE flashrt_cpp_modalities flashrt_cpp)

add_library(flashrt_cpp_pi05 STATIC
  models/pi05/src/spec.cpp
  models/pi05/src/io.cpp)
target_include_directories(flashrt_cpp_pi05
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include)
target_link_libraries(flashrt_cpp_pi05
  PUBLIC flashrt_cpp_modalities flashrt_cpp_vla)

if(BUILD_TESTING)
  add_executable(test_cpp_modalities tests/test_modalities.cpp)
  target_link_libraries(test_cpp_modalities
    PRIVATE flashrt_cpp_pi05 flashrt_cpp_modalities)
  add_test(NAME cpp_modalities COMMAND test_cpp_modalities)
endif()
