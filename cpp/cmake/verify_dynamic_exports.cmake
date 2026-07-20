if(NOT DEFINED NM_TOOL OR NOT DEFINED INPUT_FILE OR
   NOT DEFINED ALLOWED_SYMBOLS_FILE)
  message(FATAL_ERROR
    "NM_TOOL, INPUT_FILE, and ALLOWED_SYMBOLS_FILE are required")
endif()

execute_process(
  COMMAND "${NM_TOOL}" -D --defined-only "${INPUT_FILE}"
  RESULT_VARIABLE nm_result
  OUTPUT_VARIABLE nm_output
  ERROR_VARIABLE nm_error)
if(NOT nm_result EQUAL 0)
  message(FATAL_ERROR "nm failed for ${INPUT_FILE}: ${nm_error}")
endif()

file(STRINGS "${ALLOWED_SYMBOLS_FILE}" allowed_symbols
     REGEX "^[A-Za-z_][A-Za-z0-9_]*$")
string(REPLACE "\n" ";" nm_lines "${nm_output}")
set(actual_symbols)
foreach(line IN LISTS nm_lines)
  if(line MATCHES "[ \t]([A-Za-z_][A-Za-z0-9_]*)(@@[^ \t]+)?$")
    list(APPEND actual_symbols "${CMAKE_MATCH_1}")
  endif()
endforeach()
list(REMOVE_DUPLICATES actual_symbols)
list(SORT actual_symbols)
list(SORT allowed_symbols)

if(NOT actual_symbols STREQUAL allowed_symbols)
  message(FATAL_ERROR
    "Unexpected dynamic export surface for ${INPUT_FILE}.\n"
    "Expected: ${allowed_symbols}\nActual: ${actual_symbols}")
endif()
