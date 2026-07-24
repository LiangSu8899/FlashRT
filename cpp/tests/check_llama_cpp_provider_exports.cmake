if(NOT DEFINED PROVIDER_DSO OR NOT DEFINED CMAKE_NM_TOOL)
  message(FATAL_ERROR "PROVIDER_DSO and CMAKE_NM_TOOL are required")
endif()

execute_process(
  COMMAND "${CMAKE_NM_TOOL}" -D --defined-only "${PROVIDER_DSO}"
  RESULT_VARIABLE nm_result
  OUTPUT_VARIABLE nm_output
  ERROR_VARIABLE nm_error)
if(NOT nm_result EQUAL 0)
  message(FATAL_ERROR "dynamic symbol audit failed: ${nm_error}")
endif()

string(REGEX MATCHALL "[^\n]+" symbol_lines "${nm_output}")
set(found_open FALSE)
foreach(line IN LISTS symbol_lines)
  if(line MATCHES "[ 	]frt_model_runtime_open_v1(@@[^ 	]+)?$")
    set(found_open TRUE)
  elseif(NOT line MATCHES "[ 	]FLASHRT_LLAMA_CPP_PROVIDER_[^ 	]+$")
    message(FATAL_ERROR "unexpected provider export: ${line}")
  endif()
endforeach()
if(NOT found_open)
  message(FATAL_ERROR "frt_model_runtime_open_v1 is not exported")
endif()
