# Script mode (cmake -P) worker for the check-kernel-headers / kernel-headers targets.
# Inputs: MODE (check|generate), SOURCE_DIR, WORK_DIR, MINILZOC, PYTHON,
#         and for generate: NVCC, NVCC_VERSION, CUDA_ARCH, NVCC_CCBIN, NVCC_EXTRA_FLAGS.

if(NOT PYTHON)
  message(FATAL_ERROR "Python 3 is required to format the kernel headers")
endif()
file(MAKE_DIRECTORY "${WORK_DIR}")
set(CL_SRC "${SOURCE_DIR}/clguetzli/clguetzli.cl")

# embed(<input> <header> <array name>): LZO-compress <input> and write it as a C byte array.
function(embed input header array)
  get_filename_component(name "${input}" NAME)
  set(lzo "${WORK_DIR}/${name}.lzo")
  execute_process(COMMAND "${MINILZOC}" "${input}" "${lzo}" OUTPUT_QUIET RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "minilzoc failed on ${input}")
  endif()
  execute_process(COMMAND "${PYTHON}" "${SOURCE_DIR}/format_header.py" "${lzo}" "${header}" "${array}"
                  RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "format_header.py failed on ${lzo}")
  endif()
endfunction()

if(MODE STREQUAL "check")
  embed("${CL_SRC}" "${WORK_DIR}/clguetzli_cl_src.h" clguetzli_cl_src_lzo)
  # Compare ignoring line endings: the generated file and the checkout may differ only in CRLF.
  file(READ "${WORK_DIR}/clguetzli_cl_src.h" generated)
  file(READ "${SOURCE_DIR}/clguetzli/clguetzli_cl_src.h" committed)
  string(REPLACE "\r\n" "\n" generated "${generated}")
  string(REPLACE "\r\n" "\n" committed "${committed}")
  if(NOT generated STREQUAL committed)
    message(FATAL_ERROR "clguetzli/clguetzli_cl_src.h is out of date with clguetzli.cl -- "
                        "build the kernel-headers target and commit both headers.")
  endif()
  message(STATUS "Kernel headers: OpenCL source header matches clguetzli.cl.")
  return()
endif()

if(NOT MODE STREQUAL "generate")
  message(FATAL_ERROR "MODE must be check or generate")
endif()

# --- PTX: pinned toolkit only (issue #15) ---------------------------------------------------
if(NOT NVCC)
  message(FATAL_ERROR "nvcc not found (set CUDA_PATH or GUETZLI_NVCC)")
endif()
execute_process(COMMAND "${NVCC}" --version OUTPUT_VARIABLE ver RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "failed to run ${NVCC}")
endif()
if(NOT ver MATCHES "release ${NVCC_VERSION},")
  string(REGEX MATCH "release [0-9.]+" found "${ver}")
  message(FATAL_ERROR "The embedded PTX must be built with CUDA ${NVCC_VERSION} (found: ${found}). "
                      "Newer toolkits emit PTX that older drivers reject (issue #15). "
                      "Change GUETZLI_KERNEL_NVCC_VERSION only deliberately.")
endif()

set(ptx "${SOURCE_DIR}/clguetzli/clguetzli.cu.ptx64")
set(cmd "${NVCC}" -O3 -DNDEBUG -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false)
if(NVCC_EXTRA_FLAGS)
  separate_arguments(extra NATIVE_COMMAND "${NVCC_EXTRA_FLAGS}")
  list(APPEND cmd ${extra})
endif()
if(NVCC_CCBIN)
  list(APPEND cmd -ccbin "${NVCC_CCBIN}")
endif()
if(CMAKE_HOST_WIN32)
  list(APPEND cmd -Xcompiler -wd4819)
endif()
list(APPEND cmd "-I${SOURCE_DIR}" -arch=${CUDA_ARCH} -ptx -o "${ptx}" "${SOURCE_DIR}/clguetzli/clguetzli.cu")
execute_process(COMMAND ${cmd} RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "nvcc failed")
endif()
embed("${ptx}" "${SOURCE_DIR}/clguetzli/clguetzli_cu_ptx.h" clguetzli_cu64_lzo)
file(STRINGS "${ptx}" ptx_version REGEX "^\\.version" LIMIT_COUNT 1)
file(STRINGS "${ptx}" ptx_target REGEX "^\\.target" LIMIT_COUNT 1)
message(STATUS "PTX: ${ptx_version}, ${ptx_target}")

# --- OpenCL source ------------------------------------------------------------------------------
embed("${CL_SRC}" "${SOURCE_DIR}/clguetzli/clguetzli_cl_src.h" clguetzli_cl_src_lzo)
message(STATUS "Kernel headers regenerated; commit clguetzli_cu_ptx.h and clguetzli_cl_src.h.")
