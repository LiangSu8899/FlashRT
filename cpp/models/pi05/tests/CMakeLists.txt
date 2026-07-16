# PI0.5 contract, backend, lifecycle, and opt-in diagnostic targets.

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(_pi05_test_source_dir "${FLASHRT_CPP_SOURCE_DIR}/tests")
include_directories("${_pi05_internal_include}")
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/UNDEBUG>)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG>)
endif()

add_executable(test_cpp_modalities
  ${_pi05_test_source_dir}/test_modalities.cpp)
target_link_libraries(test_cpp_modalities
  PRIVATE flashrt_cpp_pi05 flashrt_cpp_modalities)
add_test(NAME cpp_modalities COMMAND test_cpp_modalities)

add_executable(test_pi05_runtime
  ${_pi05_test_source_dir}/test_pi05_runtime.cpp)
target_link_libraries(test_pi05_runtime
  PRIVATE flashrt_cpp_pi05 flashrt_cpp_modalities)
add_test(NAME pi05_runtime COMMAND test_pi05_runtime)

add_executable(test_pi05_native_weight_ops
  ${_pi05_test_source_dir}/test_pi05_native_weight_ops.cpp)
target_link_libraries(test_pi05_native_weight_ops PRIVATE flashrt_cpp_pi05)
add_test(NAME pi05_native_weight_ops COMMAND test_pi05_native_weight_ops)

add_executable(test_pi05_native_calibration
  ${_pi05_test_source_dir}/test_pi05_native_calibration.cpp)
target_link_libraries(test_pi05_native_calibration PRIVATE flashrt_cpp_pi05)
add_test(NAME pi05_native_calibration COMMAND test_pi05_native_calibration)

add_executable(pi05_native_weight_probe
  ${_pi05_test_source_dir}/pi05_native_weight_probe.cpp)
target_link_libraries(pi05_native_weight_probe PRIVATE flashrt_cpp_pi05)

if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  add_executable(test_pi05_native_quantization
    ${_pi05_test_source_dir}/test_pi05_native_quantization.cpp)
  target_link_libraries(test_pi05_native_quantization PRIVATE flashrt_cpp_pi05)
  add_test(NAME pi05_native_quantization
    COMMAND test_pi05_native_quantization)

  add_executable(pi05_native_quant_probe
    ${_pi05_test_source_dir}/pi05_native_quant_probe.cpp)
  target_link_libraries(pi05_native_quant_probe
    PRIVATE flashrt_cpp_pi05_kernels flashrt_exec CUDA::cudart)

  add_executable(test_pi05_native_weight_packer
    ${_pi05_test_source_dir}/test_pi05_native_weight_packer.cpp)
  target_link_libraries(test_pi05_native_weight_packer
    PRIVATE flashrt_cpp_pi05 flashrt_exec CUDA::cudart)
  add_test(NAME pi05_native_weight_packer
    COMMAND test_pi05_native_weight_packer)

  add_executable(test_pi05_native_kernel_driver
    ${_pi05_test_source_dir}/test_pi05_native_kernel_driver.cpp)
  target_link_libraries(test_pi05_native_kernel_driver
    PRIVATE flashrt_cpp_pi05_kernels flashrt_exec CUDA::cudart)
  add_test(NAME pi05_native_kernel_driver
    COMMAND test_pi05_native_kernel_driver)

  add_executable(test_pi05_native_forward_primitives
    ${_pi05_test_source_dir}/test_pi05_native_forward_primitives.cpp)
  target_link_libraries(test_pi05_native_forward_primitives
    PRIVATE flashrt_cpp_pi05_kernels flashrt_exec CUDA::cudart)
  add_test(NAME pi05_native_forward_primitives
    COMMAND test_pi05_native_forward_primitives)

  add_executable(test_pi05_native_bf16_forward
    ${_pi05_test_source_dir}/test_pi05_native_bf16_forward.cpp)
  target_link_libraries(test_pi05_native_bf16_forward
    PRIVATE flashrt_cpp_pi05_kernels flashrt_exec CUDA::cudart)
  add_test(NAME pi05_native_bf16_forward
    COMMAND test_pi05_native_bf16_forward)

  add_executable(pi05_native_encoder_qkv_probe
    ${_pi05_test_source_dir}/pi05_native_encoder_qkv_probe.cpp)
  target_link_libraries(pi05_native_encoder_qkv_probe
    PRIVATE flashrt_cpp_pi05_kernels flashrt_exec CUDA::cudart)

  add_executable(test_pi05_native_style_precompute
    ${_pi05_test_source_dir}/test_pi05_native_style_precompute.cpp)
  target_link_libraries(test_pi05_native_style_precompute
    PRIVATE flashrt_cpp_pi05_kernels flashrt_exec CUDA::cudart)
  add_test(NAME pi05_native_style_precompute
    COMMAND test_pi05_native_style_precompute)

  add_executable(pi05_native_style_probe
    ${_pi05_test_source_dir}/pi05_native_style_probe.cpp)
  target_link_libraries(pi05_native_style_probe
    PRIVATE flashrt_cpp_pi05_kernels flashrt_exec CUDA::cudart)

  add_executable(test_pi05_native_workspace
    ${_pi05_test_source_dir}/test_pi05_native_workspace.cpp)
  target_link_libraries(test_pi05_native_workspace
    PRIVATE flashrt_cpp_pi05 flashrt_exec CUDA::cudart)
  add_test(NAME pi05_native_workspace
    COMMAND test_pi05_native_workspace)

  add_executable(test_pi05_native_rtx_attention
    ${_pi05_test_source_dir}/test_pi05_native_rtx_attention.cpp)
  target_link_libraries(test_pi05_native_rtx_attention
    PRIVATE flashrt_cpp_pi05 flashrt_exec CUDA::cudart)
  add_test(NAME pi05_native_rtx_attention
    COMMAND test_pi05_native_rtx_attention)

  if(FLASHRT_CPP_WITH_FA2)
    add_executable(pi05_native_encoder_layer_probe
      ${_pi05_test_source_dir}/pi05_native_encoder_layer_probe.cpp)
    target_link_libraries(pi05_native_encoder_layer_probe
      PRIVATE flashrt_cpp_pi05_kernels flashrt_exec CUDA::cudart)

    add_executable(pi05_native_encoder_probe
      ${_pi05_test_source_dir}/pi05_native_encoder_probe.cpp)
    target_link_libraries(pi05_native_encoder_probe
      PRIVATE flashrt_cpp_pi05_kernels flashrt_exec CUDA::cudart)

    add_executable(pi05_native_vision_probe
      ${_pi05_test_source_dir}/pi05_native_vision_probe.cpp)
    target_link_libraries(pi05_native_vision_probe
      PRIVATE flashrt_cpp_pi05_kernels flashrt_exec CUDA::cudart)

    add_executable(pi05_native_diffusion_probe
      ${_pi05_test_source_dir}/pi05_native_diffusion_probe.cpp)
    target_link_libraries(pi05_native_diffusion_probe
      PRIVATE flashrt_cpp_pi05_kernels flashrt_exec CUDA::cudart)

    add_executable(pi05_native_graph_probe
      ${_pi05_test_source_dir}/pi05_native_graph_probe.cpp)
    target_link_libraries(pi05_native_graph_probe
      PRIVATE flashrt_cpp_pi05_kernels flashrt_exec CUDA::cudart)

    if(FLASHRT_CPP_WITH_SENTENCEPIECE)
      add_executable(pi05_native_e2e_probe
        ${_pi05_test_source_dir}/pi05_native_e2e_probe.cpp)
      target_link_libraries(pi05_native_e2e_probe
        PRIVATE flashrt_cpp_pi05_c flashrt_exec CUDA::cudart)

      add_executable(pi05_native_dlopen_probe
        ${_pi05_test_source_dir}/pi05_native_dlopen_probe.cpp)
      target_include_directories(pi05_native_dlopen_probe
        PRIVATE ${FLASHRT_CPP_SOURCE_DIR}/../runtime/include
                ${FLASHRT_CPP_SOURCE_DIR}/../exec/include)
      target_link_libraries(pi05_native_dlopen_probe
        PRIVATE ${CMAKE_DL_LIBS})
    endif()

    add_executable(test_pi05_native_rtx_attention_driver
      ${_pi05_test_source_dir}/test_pi05_native_rtx_attention_driver.cpp)
    target_link_libraries(test_pi05_native_rtx_attention_driver
      PRIVATE flashrt_cpp_pi05_kernels flashrt_exec CUDA::cudart)
    add_test(NAME pi05_native_rtx_attention_driver
      COMMAND test_pi05_native_rtx_attention_driver)
  endif()

  if(FLASHRT_CPP_WITH_SENTENCEPIECE AND
     (FLASHRT_CPP_WITH_FA2 OR FLASHRT_CPP_WITH_THOR_FP8))
    add_executable(pi05_native_open_probe
      ${_pi05_test_source_dir}/pi05_native_open_probe.cpp)
    target_link_libraries(pi05_native_open_probe
      PRIVATE flashrt_cpp_pi05_c flashrt_exec CUDA::cudart)
  endif()

  if(FLASHRT_CPP_WITH_SENTENCEPIECE)
    add_executable(pi05_tokenizer_corpus_probe
      ${_pi05_test_source_dir}/pi05_tokenizer_corpus_probe.cpp)
    target_link_libraries(pi05_tokenizer_corpus_probe
      PRIVATE flashrt_cpp_pi05)
  endif()

  if(FLASHRT_CPP_WITH_SENTENCEPIECE AND
     (FLASHRT_CPP_WITH_FA2 OR FLASHRT_CPP_WITH_THOR_FP8))
    add_executable(pi05_native_fp8_calibration_probe
      ${_pi05_test_source_dir}/pi05_native_fp8_calibration_probe.cpp)
    target_link_libraries(pi05_native_fp8_calibration_probe
      PRIVATE flashrt_cpp_pi05_c flashrt_exec CUDA::cudart)
  endif()

  add_executable(pi05_native_rope_probe
    ${_pi05_test_source_dir}/pi05_native_rope_probe.cpp)
  target_link_libraries(pi05_native_rope_probe
    PRIVATE flashrt_cpp_pi05 flashrt_exec CUDA::cudart)
endif()

add_executable(test_pi05_prompt_format
  ${_pi05_test_source_dir}/test_pi05_prompt_format.cpp)
target_link_libraries(test_pi05_prompt_format PRIVATE flashrt_cpp_pi05)
add_test(NAME pi05_prompt_format COMMAND test_pi05_prompt_format)

add_executable(test_pi05_prompt_embed
  ${_pi05_test_source_dir}/test_pi05_prompt_embed.cpp)
target_link_libraries(test_pi05_prompt_embed PRIVATE flashrt_cpp_pi05)
add_test(NAME pi05_prompt_embed COMMAND test_pi05_prompt_embed)

if(FLASHRT_CPP_WITH_CUDA_STAGING)
  add_executable(test_device_staging
    ${_pi05_test_source_dir}/test_device_staging.cpp)
  target_link_libraries(test_device_staging
    PRIVATE flashrt_cpp_pi05 flashrt_cpp_modalities CUDA::cudart)
  add_test(NAME device_staging COMMAND test_device_staging)

  add_executable(test_pi05_native_device_weights
    ${_pi05_test_source_dir}/test_pi05_native_device_weights.cpp)
  target_link_libraries(test_pi05_native_device_weights
    PRIVATE flashrt_cpp_pi05 flashrt_exec CUDA::cudart)
  add_test(NAME pi05_native_device_weights
    COMMAND test_pi05_native_device_weights)

  add_executable(test_pi05_native_weight_materializer
    ${_pi05_test_source_dir}/test_pi05_native_weight_materializer.cpp)
  target_link_libraries(test_pi05_native_weight_materializer
    PRIVATE flashrt_cpp_pi05 flashrt_exec CUDA::cudart)
  add_test(NAME pi05_native_weight_materializer
    COMMAND test_pi05_native_weight_materializer)

  add_executable(test_pi05_c_api
    ${_pi05_test_source_dir}/test_pi05_c_api.cpp)
  target_link_libraries(test_pi05_c_api
    PRIVATE flashrt_cpp_pi05_c flashrt_exec CUDA::cudart)
  add_test(NAME pi05_c_api COMMAND test_pi05_c_api)

  add_executable(test_pi05_model_runtime
    ${_pi05_test_source_dir}/test_pi05_model_runtime.cpp)
  target_link_libraries(test_pi05_model_runtime
    PRIVATE flashrt_cpp_pi05_c flashrt_exec flashrt_runtime CUDA::cudart)
  add_test(NAME pi05_model_runtime COMMAND test_pi05_model_runtime)

  add_executable(test_pi05_native_open
    ${_pi05_test_source_dir}/test_pi05_native_open.cpp)
  target_link_libraries(test_pi05_native_open
    PRIVATE flashrt_cpp_pi05_c flashrt_runtime)
  if((FLASHRT_CPP_WITH_FA2 OR FLASHRT_CPP_WITH_THOR_FP8) AND
     FLASHRT_CPP_WITH_SENTENCEPIECE)
    target_compile_definitions(test_pi05_native_open
      PRIVATE FLASHRT_CPP_PI05_NATIVE_OPEN_ENABLED=1)
  endif()
  add_test(NAME pi05_native_open COMMAND test_pi05_native_open)
endif()
