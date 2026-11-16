# Layering check (docs/design/modernization.md §5.1, §9): the core never includes anything from
# backend/ or the GPU layer, nor OpenCL/CUDA headers.
#   cmake -DSOURCE_DIR=<repo> -P cmake/CheckLayering.cmake
if(NOT SOURCE_DIR)
  message(FATAL_ERROR "SOURCE_DIR not set")
endif()

set(_core_dirs config logging core guetzli third_party/butteraugli/butteraugli)
set(_forbidden
    "#[ \t]*include[ \t]*[\"<](backend|clguetzli|kernels)/"
    "#[ \t]*include[ \t]*[\"<](ocl|ocu|clguetzli|cuguetzli|cuda_dynload|cumem_pool|gpu_error|gpu_options)\\.h[\">]"
    "#[ \t]*include[ \t]*[\"<](CL/|cuda\\.h|cuda_runtime|opencl\\.hpp|cl\\.hpp)")

set(_violations "")
foreach(_dir IN LISTS _core_dirs)
  file(GLOB_RECURSE _files "${SOURCE_DIR}/${_dir}/*.h" "${SOURCE_DIR}/${_dir}/*.cc" "${SOURCE_DIR}/${_dir}/*.cpp"
       "${SOURCE_DIR}/${_dir}/*.inc")
  foreach(_file IN LISTS _files)
    foreach(_pattern IN LISTS _forbidden)
      file(STRINGS "${_file}" _hits REGEX "${_pattern}")
      foreach(_hit IN LISTS _hits)
        file(RELATIVE_PATH _rel "${SOURCE_DIR}" "${_file}")
        string(APPEND _violations "  ${_rel}: ${_hit}\n")
      endforeach()
    endforeach()
  endforeach()
endforeach()

if(_violations)
  message(FATAL_ERROR "Core code includes backend/GPU headers:\n${_violations}")
endif()
message(STATUS "layering OK")
