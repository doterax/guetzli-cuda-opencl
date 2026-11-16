# Script mode (cmake -P) worker for the check-kernel-headers / kernel-headers targets.
# Inputs: MODE (check|generate), SOURCE_DIR, WORK_DIR, EMBED (the embed_kernel tool),
#         and for generate: NVCC, NVCC_VERSION, CUDA_ARCH, NVCC_CCBIN, NVCC_EXTRA_FLAGS.

file(MAKE_DIRECTORY "${WORK_DIR}")
set(CL_SRC "${SOURCE_DIR}/kernels/guetzli.cl")
set(CL_HEADER "${SOURCE_DIR}/kernels/guetzli_cl_src.h")
set(PTX_HEADER "${SOURCE_DIR}/kernels/guetzli_cu_ptx.h")

# embed(<input> <header> <name>): LZO-compress <input> into a header defining k<name>Size / k<name>Lzo.
function(embed input header name)
  execute_process(COMMAND "${EMBED}" "${input}" "${header}" "${name}" RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "embed_kernel failed on ${input}")
  endif()
endfunction()

if(MODE STREQUAL "check")
  embed("${CL_SRC}" "${WORK_DIR}/guetzli_cl_src.h" OpenClSource)
  # Compare ignoring line endings: the checkout may have CRLF.
  file(READ "${WORK_DIR}/guetzli_cl_src.h" generated)
  file(READ "${CL_HEADER}" committed)
  string(REPLACE "\r\n" "\n" generated "${generated}")
  string(REPLACE "\r\n" "\n" committed "${committed}")
  if(NOT generated STREQUAL committed)
    message(FATAL_ERROR "kernels/guetzli_cl_src.h is out of date with kernels/guetzli.cl -- "
                        "build the kernel-headers target and commit both headers.")
  endif()
  message(STATUS "Kernel headers: OpenCL source header matches kernels/guetzli.cl.")
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

set(ptx "${WORK_DIR}/guetzli.ptx")
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
list(APPEND cmd "-I${SOURCE_DIR}" -arch=${CUDA_ARCH} -ptx -o "${ptx}" "${SOURCE_DIR}/kernels/guetzli.cu")
execute_process(COMMAND ${cmd} RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "nvcc failed")
endif()
embed("${ptx}" "${PTX_HEADER}" CudaPtx)
file(STRINGS "${ptx}" ptx_version REGEX "^\\.version" LIMIT_COUNT 1)
file(STRINGS "${ptx}" ptx_target REGEX "^\\.target" LIMIT_COUNT 1)
message(STATUS "PTX: ${ptx_version}, ${ptx_target}")

# --- OpenCL source ------------------------------------------------------------------------------
embed("${CL_SRC}" "${CL_HEADER}" OpenClSource)
message(STATUS "Kernel headers regenerated; commit kernels/guetzli_cu_ptx.h and kernels/guetzli_cl_src.h.")
