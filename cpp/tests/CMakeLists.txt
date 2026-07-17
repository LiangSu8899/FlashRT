# Tests for model-independent native loader and modality mechanisms.

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/UNDEBUG>)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG>)
endif()

add_executable(test_safetensors_loader test_safetensors_loader.cpp)
target_link_libraries(test_safetensors_loader PRIVATE flashrt_cpp_loader)
add_test(NAME safetensors_loader COMMAND test_safetensors_loader)

add_executable(test_sha256 test_sha256.cpp)
target_link_libraries(test_sha256 PRIVATE flashrt_cpp_loader)
add_test(NAME sha256 COMMAND test_sha256)

add_executable(test_native_config test_native_config.cpp)
target_link_libraries(test_native_config PRIVATE flashrt_cpp_native)
add_test(NAME native_config COMMAND test_native_config)

if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  add_executable(test_native_cuda_graph_set test_native_cuda_graph_set.cpp)
  target_link_libraries(test_native_cuda_graph_set
    PRIVATE flashrt_cpp_native_cuda)
  add_test(NAME native_cuda_graph_set COMMAND test_native_cuda_graph_set)
endif()

if(FLASHRT_CPP_WITH_CUDA_KERNELS AND
   (FLASHRT_CPP_WITH_FA2 OR FLASHRT_CPP_WITH_THOR_FP8))
  add_executable(test_csrc_rope_table test_csrc_rope_table.cu)
  target_include_directories(test_csrc_rope_table PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/kernels)
  target_link_libraries(test_csrc_rope_table PRIVATE
    flashrt_cpp_csrc_core CUDA::cudart)
  set_target_properties(test_csrc_rope_table PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
  add_test(NAME csrc_rope_table COMMAND test_csrc_rope_table)
endif()

add_executable(test_text_modalities test_text_modalities.cpp)
target_link_libraries(test_text_modalities PRIVATE flashrt_cpp_modalities)
add_test(NAME text_modalities COMMAND test_text_modalities)

add_executable(test_text_tokenizer test_text_tokenizer.cpp)
target_link_libraries(test_text_tokenizer PRIVATE flashrt_cpp_modalities)
add_test(NAME text_tokenizer COMMAND test_text_tokenizer)
