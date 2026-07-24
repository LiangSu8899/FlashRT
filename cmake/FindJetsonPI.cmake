# FindJetsonPI.cmake — locate an INSTALLED Jetson-PI (the llama.cpp/GGML fork) for
# FlashRT's Jetson-PI provider, as the production alternative to the dev
# add_subdirectory(JETSON_PI_ROOT) path (jetsonpi迁移.txt §5.2).
#
# The narrow C API libs (jetson_pi_pi0 / jetson_pi_llm / jetson_pi_mllm) are the
# FlashRT provider's real link surface; they in turn need mtmd / llama / ggml
# (+ either directly linked ggml-* backend libs or runtime backend modules).
# This module finds them all under one install prefix and exposes the linked
# libraries as imported targets so the provider links JetsonPI::jetson_pi_pi0
# etc. exactly as it links the in-tree targets in the dev path.
#
# Usage:
#   find_package(JetsonPI REQUIRED)
#   target_link_libraries(my_provider PRIVATE JetsonPI::jetson_pi_pi0)
#
# Required hint variable:
#   JetsonPI_ROOT : install prefix (lib or lib64, include)
#
# Defines imported targets:
#   JetsonPI::jetson_pi_pi0  JetsonPI::jetson_pi_llm  JetsonPI::jetson_pi_mllm
#   JetsonPI::mtmd  JetsonPI::llama  JetsonPI::ggml  JetsonPI::ggml-base
#   JetsonPI::ggml-cpu  JetsonPI::ggml-cuda  JetsonPI::ggml-vulkan
#   JetsonPI::ggml-opencl  JetsonPI::ggml-sycl
# Backend imported targets exist only for a direct-link GGML package. For a
# GGML_BACKEND_DL package, JetsonPI_BACKEND_DIR and JetsonPI_BACKEND_MODULES
# identify the validated runtime modules and none are linked into the provider.
#
# Module-mode search: Jetson-PI does not ship a config-file package for the
# narrow C API libs (installing an install(EXPORT) graph would require
# mtmd/llama/ggml to be in an export set too). Instead this module locates the
# installed libs via find_library and builds JetsonPI::* imported targets with
# INTERFACE_LINK_LIBRARIES to the sibling installed deps — no export graph
# needed, just a coherent install prefix.
#
# --- module-mode search -------------------------------------------------------

if(NOT JetsonPI_ROOT)
  set(JetsonPI_FOUND FALSE)
  set(JetsonPI_NOT_FOUND_MESSAGE
      "JetsonPI_ROOT must name the installed Jetson-PI prefix")
  return()
endif()

set(_JetsonPI_HINT_PATHS ${JetsonPI_ROOT})

unset(ggml_DIR CACHE)
find_package(ggml CONFIG QUIET
  PATHS ${_JetsonPI_HINT_PATHS}
  NO_DEFAULT_PATH)
if(NOT ggml_FOUND)
  message(FATAL_ERROR
    "JetsonPI_ROOT='${JetsonPI_ROOT}' has no matching ggml-config.cmake. "
    "Install Jetson-PI and its bundled GGML into the same prefix.")
endif()
set(JetsonPI_BACKEND_DL ${GGML_BACKEND_DL})

# NO_DEFAULT_PATH: same no-system-mix rule as the find_library path below.
# Without it, a stray system header could paper over a bad JetsonPI_ROOT.
unset(JetsonPI_INCLUDE_DIR CACHE)
find_path(JetsonPI_INCLUDE_DIR
  NAMES jetson_pi_pi0.h
  HINTS ${_JetsonPI_HINT_PATHS}
  PATH_SUFFIXES include
  DOC "Jetson-PI narrow C API headers"
  NO_DEFAULT_PATH)

# Each lib we need. find_library per-target so a missing backend lib (e.g.
# ggml-vulkan when GGML_VULKAN=OFF) doesn't fail the whole find. We search ONLY
# the hint paths (NO_DEFAULT_PATH): intentionally NOT falling back to system
# default paths. A system /usr/lib libllama.so / libggml.so on a deploy host
# with a distro llama.cpp package would silently mix a different GGML version
# into the narrow libs — exactly the §15.1 "do not mix another llama.cpp/GGML
# version" hazard. A failed find should fail loudly via the FAIL_MESSAGE below,
# not paper over a bad/missing prefix with a stray system lib.
function(_jetsonpi_find_lib var name)
  unset(${var} CACHE)
  find_library(${var}
    NAMES ${name}
    HINTS ${_JetsonPI_HINT_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "Jetson-PI ${name} library"
    NO_DEFAULT_PATH)
endfunction()

_jetsonpi_find_lib(JetsonPI_pi0_LIBRARY      jetson_pi_pi0)
_jetsonpi_find_lib(JetsonPI_llm_LIBRARY      jetson_pi_llm)
_jetsonpi_find_lib(JetsonPI_mllm_LIBRARY     jetson_pi_mllm)
_jetsonpi_find_lib(JetsonPI_mtmd_LIBRARY     mtmd)
_jetsonpi_find_lib(JetsonPI_llama_LIBRARY    llama)
_jetsonpi_find_lib(JetsonPI_ggml_LIBRARY     ggml)
_jetsonpi_find_lib(JetsonPI_ggml_base_LIBRARY    ggml-base)
_jetsonpi_find_lib(JetsonPI_onemath_cublas_LIBRARY onemath_blas_cublas)

set(JetsonPI_BACKEND_MODULES "")
if(JetsonPI_BACKEND_DL)
  if(NOT GGML_BACKEND_DIR OR NOT IS_ABSOLUTE "${GGML_BACKEND_DIR}")
    message(FATAL_ERROR
      "The Jetson-PI GGML package uses GGML_BACKEND_DL, but its installed "
      "ggml-config.cmake does not contain an absolute GGML_BACKEND_DIR. "
      "Rebuild Jetson-PI with -DGGML_BACKEND_DIR=<install-prefix>/bin so "
      "runtime backend discovery is explicit and independent of cwd.")
  endif()
  get_filename_component(_JetsonPI_root_real "${JetsonPI_ROOT}" REALPATH)
  get_filename_component(JetsonPI_BACKEND_DIR "${GGML_BACKEND_DIR}" REALPATH)
  string(FIND "${JetsonPI_BACKEND_DIR}/" "${_JetsonPI_root_real}/"
    _JetsonPI_backend_prefix_index)
  if(NOT _JetsonPI_backend_prefix_index EQUAL 0)
    message(FATAL_ERROR
      "GGML_BACKEND_DIR='${GGML_BACKEND_DIR}' is outside "
      "JetsonPI_ROOT='${JetsonPI_ROOT}'. Install core libraries and runtime "
      "backend modules as one coherent deployment package.")
  endif()

  foreach(_JetsonPI_backend IN LISTS GGML_AVAILABLE_BACKENDS)
    string(REPLACE "-" "_" _JetsonPI_backend_id "${_JetsonPI_backend}")
    set(_JetsonPI_backend_var "JetsonPI_${_JetsonPI_backend_id}_MODULE")
    unset(${_JetsonPI_backend_var} CACHE)
    find_file(${_JetsonPI_backend_var}
      NAMES "${CMAKE_SHARED_MODULE_PREFIX}${_JetsonPI_backend}${CMAKE_SHARED_MODULE_SUFFIX}"
      HINTS "${JetsonPI_BACKEND_DIR}"
      DOC "Jetson-PI ${_JetsonPI_backend} runtime module"
      NO_DEFAULT_PATH)
    if(NOT ${_JetsonPI_backend_var})
      message(FATAL_ERROR
        "Jetson-PI's ggml-config.cmake declares backend "
        "'${_JetsonPI_backend}', but its runtime module is missing from "
        "GGML_BACKEND_DIR='${JetsonPI_BACKEND_DIR}'.")
    endif()
    list(APPEND JetsonPI_BACKEND_MODULES "${${_JetsonPI_backend_var}}")
    if(_JetsonPI_backend STREQUAL "ggml-cpu")
      set(JetsonPI_ggml_cpu_LIBRARY "${${_JetsonPI_backend_var}}")
    elseif(_JetsonPI_backend STREQUAL "ggml-cuda")
      set(JetsonPI_ggml_cuda_LIBRARY "${${_JetsonPI_backend_var}}")
    elseif(_JetsonPI_backend STREQUAL "ggml-vulkan")
      set(JetsonPI_ggml_vulkan_LIBRARY "${${_JetsonPI_backend_var}}")
    elseif(_JetsonPI_backend STREQUAL "ggml-opencl")
      set(JetsonPI_ggml_opencl_LIBRARY "${${_JetsonPI_backend_var}}")
    elseif(_JetsonPI_backend STREQUAL "ggml-sycl")
      set(JetsonPI_ggml_sycl_LIBRARY "${${_JetsonPI_backend_var}}")
    endif()
  endforeach()
else()
  set(JetsonPI_BACKEND_DIR "")
  _jetsonpi_find_lib(JetsonPI_ggml_cpu_LIBRARY     ggml-cpu)
  _jetsonpi_find_lib(JetsonPI_ggml_cuda_LIBRARY    ggml-cuda)
  _jetsonpi_find_lib(JetsonPI_ggml_vulkan_LIBRARY  ggml-vulkan)
  _jetsonpi_find_lib(JetsonPI_ggml_opencl_LIBRARY  ggml-opencl)
  _jetsonpi_find_lib(JetsonPI_ggml_sycl_LIBRARY    ggml-sycl)
endif()

if (JetsonPI_ggml_opencl_LIBRARY AND NOT JetsonPI_BACKEND_DL)
  find_package(OpenCL REQUIRED)
endif()
if (JetsonPI_ggml_sycl_LIBRARY AND NOT JetsonPI_BACKEND_DL)
  find_library(JetsonPI_sycl_LIBRARY NAMES sycl HINTS ENV ONEAPI_ROOT PATH_SUFFIXES lib lib64)
  find_library(JetsonPI_imf_LIBRARY NAMES imf HINTS ENV ONEAPI_ROOT PATH_SUFFIXES lib lib64)
  find_library(JetsonPI_svml_LIBRARY NAMES svml HINTS ENV ONEAPI_ROOT PATH_SUFFIXES lib lib64)
  find_library(JetsonPI_intlc_LIBRARY NAMES intlc intlc.so.5 HINTS ENV ONEAPI_ROOT PATH_SUFFIXES lib lib64)
  if (NOT JetsonPI_sycl_LIBRARY OR NOT JetsonPI_imf_LIBRARY OR
      NOT JetsonPI_svml_LIBRARY OR NOT JetsonPI_intlc_LIBRARY OR
      NOT JetsonPI_onemath_cublas_LIBRARY)
    message(FATAL_ERROR "JetsonPI ggml-sycl was found, but its DPC++/oneMath runtime libraries were not. Set ONEAPI_ROOT and install libsycl, libimf, libsvml, libintlc, and libonemath_blas_cublas.")
  endif()
  find_package(CUDAToolkit REQUIRED)
endif()

include(FindPackageHandleStandardArgs)
# The provider hard-needs the narrow libs + their core transitive deps. Backend
# ggml-* libs are optional (link whichever were built).
find_package_handle_standard_args(JetsonPI
  REQUIRED_VARS
    JetsonPI_INCLUDE_DIR
    JetsonPI_pi0_LIBRARY
    JetsonPI_llm_LIBRARY
    JetsonPI_mllm_LIBRARY
    JetsonPI_mtmd_LIBRARY
    JetsonPI_llama_LIBRARY
    JetsonPI_ggml_LIBRARY
    JetsonPI_ggml_base_LIBRARY
    JetsonPI_ggml_cpu_LIBRARY
  FAIL_MESSAGE "Jetson-PI not found. Set -DJetsonPI_ROOT=<install-prefix> (a prefix containing include/jetson_pi_pi0.h and lib{,64}/libjetson_pi_pi0.so etc.), or use the dev path -DJETSON_PI_ROOT=<source-tree> with add_subdirectory.")

if (JetsonPI_FOUND)
  # Aggregate the ggml backend libs that actually resolved (cuda/vulkan optional).
  # Target names use hyphens to match the soname (libggml-cuda.so) and the
  # header comment above (JetsonPI::ggml-cuda / ::ggml-vulkan).
  set(_JetsonPI_ggml_backends "")
  if (JetsonPI_ggml_cuda_LIBRARY AND NOT JetsonPI_BACKEND_DL)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND _JetsonPI_ggml_backends JetsonPI::ggml-cuda)
  endif()
  if (JetsonPI_ggml_vulkan_LIBRARY AND NOT JetsonPI_BACKEND_DL)
    list(APPEND _JetsonPI_ggml_backends JetsonPI::ggml-vulkan)
  endif()
  if (JetsonPI_ggml_opencl_LIBRARY AND NOT JetsonPI_BACKEND_DL)
    list(APPEND _JetsonPI_ggml_backends JetsonPI::ggml-opencl)
  endif()
  if (JetsonPI_ggml_sycl_LIBRARY AND NOT JetsonPI_BACKEND_DL)
    list(APPEND _JetsonPI_ggml_backends JetsonPI::ggml-sycl)
  endif()

  # Helper: declare one imported shared lib target with its transitive deps.
  function(_jetsonpi_import_target tgt lib dep)
    if (NOT ${lib})
      return()
    endif()
    if (TARGET ${tgt})
      return()
    endif()
    add_library(${tgt} SHARED IMPORTED)
    set_target_properties(${tgt} PROPERTIES
      IMPORTED_LOCATION "${${lib}}"
      INTERFACE_INCLUDE_DIRECTORIES "${JetsonPI_INCLUDE_DIR}")
    if (dep)
      set_target_properties(${tgt} PROPERTIES INTERFACE_LINK_LIBRARIES "${dep}")
    endif()
  endfunction()

  set(_ggml_deps "JetsonPI::ggml-base")
  _jetsonpi_import_target(JetsonPI::ggml-base   JetsonPI_ggml_base_LIBRARY   "")
  if(NOT JetsonPI_BACKEND_DL)
    list(APPEND _ggml_deps JetsonPI::ggml-cpu ${_JetsonPI_ggml_backends})
    _jetsonpi_import_target(JetsonPI::ggml-cpu    JetsonPI_ggml_cpu_LIBRARY    "JetsonPI::ggml-base")
    _jetsonpi_import_target(JetsonPI::ggml-cuda   JetsonPI_ggml_cuda_LIBRARY   "JetsonPI::ggml-base;CUDA::cudart;CUDA::cublas;CUDA::cuda_driver")
    _jetsonpi_import_target(JetsonPI::ggml-vulkan JetsonPI_ggml_vulkan_LIBRARY "JetsonPI::ggml-base")
    _jetsonpi_import_target(JetsonPI::ggml-opencl JetsonPI_ggml_opencl_LIBRARY "JetsonPI::ggml-base;OpenCL::OpenCL")
    _jetsonpi_import_target(JetsonPI::ggml-sycl   JetsonPI_ggml_sycl_LIBRARY   "JetsonPI::ggml-base;${JetsonPI_onemath_cublas_LIBRARY};${JetsonPI_sycl_LIBRARY};${JetsonPI_imf_LIBRARY};${JetsonPI_svml_LIBRARY};${JetsonPI_intlc_LIBRARY};CUDA::cublas;CUDA::cuda_driver")
  endif()
  _jetsonpi_import_target(JetsonPI::ggml        JetsonPI_ggml_LIBRARY        "${_ggml_deps}")
  _jetsonpi_import_target(JetsonPI::llama       JetsonPI_llama_LIBRARY       "JetsonPI::ggml")
  _jetsonpi_import_target(JetsonPI::mtmd        JetsonPI_mtmd_LIBRARY        "JetsonPI::llama;JetsonPI::ggml")
  _jetsonpi_import_target(JetsonPI::jetson_pi_pi0  JetsonPI_pi0_LIBRARY  "JetsonPI::mtmd;JetsonPI::llama;JetsonPI::ggml")
  _jetsonpi_import_target(JetsonPI::jetson_pi_llm  JetsonPI_llm_LIBRARY  "JetsonPI::llama;JetsonPI::ggml")
  _jetsonpi_import_target(JetsonPI::jetson_pi_mllm JetsonPI_mllm_LIBRARY "JetsonPI::mtmd;JetsonPI::llama;JetsonPI::ggml")

  set(JetsonPI_LIBRARY_DIRS "")
  foreach(_JetsonPI_library
      JetsonPI_pi0_LIBRARY JetsonPI_llm_LIBRARY JetsonPI_mllm_LIBRARY
      JetsonPI_mtmd_LIBRARY JetsonPI_llama_LIBRARY JetsonPI_ggml_LIBRARY
      JetsonPI_ggml_base_LIBRARY JetsonPI_ggml_cpu_LIBRARY
      JetsonPI_ggml_cuda_LIBRARY JetsonPI_ggml_vulkan_LIBRARY
      JetsonPI_ggml_opencl_LIBRARY JetsonPI_ggml_sycl_LIBRARY
      JetsonPI_onemath_cublas_LIBRARY JetsonPI_sycl_LIBRARY
      JetsonPI_imf_LIBRARY JetsonPI_svml_LIBRARY JetsonPI_intlc_LIBRARY)
    if(${_JetsonPI_library})
      get_filename_component(_JetsonPI_library_dir "${${_JetsonPI_library}}" DIRECTORY)
      list(APPEND JetsonPI_LIBRARY_DIRS "${_JetsonPI_library_dir}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES JetsonPI_LIBRARY_DIRS)

  if(JetsonPI_BACKEND_DL)
    message(STATUS
      "Jetson-PI found (module mode, dynamic GGML backends in ${JetsonPI_BACKEND_DIR}): ${JetsonPI_INCLUDE_DIR}")
  else()
    message(STATUS "Jetson-PI found (module mode): ${JetsonPI_INCLUDE_DIR}")
  endif()
endif()

mark_as_advanced(JetsonPI_INCLUDE_DIR JetsonPI_BACKEND_DIR
  JetsonPI_pi0_LIBRARY JetsonPI_llm_LIBRARY JetsonPI_mllm_LIBRARY
  JetsonPI_mtmd_LIBRARY JetsonPI_llama_LIBRARY
  JetsonPI_ggml_LIBRARY JetsonPI_ggml_base_LIBRARY
  JetsonPI_ggml_cpu_LIBRARY JetsonPI_ggml_cuda_LIBRARY JetsonPI_ggml_vulkan_LIBRARY
  JetsonPI_ggml_opencl_LIBRARY JetsonPI_ggml_sycl_LIBRARY
  JetsonPI_onemath_cublas_LIBRARY JetsonPI_sycl_LIBRARY
  JetsonPI_imf_LIBRARY JetsonPI_svml_LIBRARY JetsonPI_intlc_LIBRARY)
