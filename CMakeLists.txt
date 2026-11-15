# Guetzli CUDA/OpenCL -- CMake build (docs/design/modernization.md §6).
#
#   cmake --preset release            # configure (external deps from external/install)
#   cmake --build --preset release    # -> build/release/bin/guetzli[.exe]
#   ctest --preset release            # unit tests + smoke tests
#
# External dependencies come from the superbuild in external/ (`cmake -S external -B external/build`
# then `cmake --build external/build`), installed to external/install. GUETZLI_EXTERNAL_PREFIX
# points elsewhere if needed; libraries not found there are taken from the system.
#
# The embedded kernel headers (clguetzli_cl_src.h, clguetzli_cu_ptx.h) are committed and only
# consumed. Regenerate them explicitly with the `kernel-headers` target (pinned CUDA toolkit, §7);
# `check-kernel-headers` verifies the OpenCL one (CI).
cmake_minimum_required(VERSION 3.24)
project(guetzli LANGUAGES C CXX)

# ---------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------
option(GUETZLI_WITH_OPENCL "OpenCL backend" ON)
option(GUETZLI_WITH_CUDA "CUDA backend (driver API, loaded at runtime; needs only cuda.h)" ON)
option(GUETZLI_WITH_FULL_JPEG "JPEG input via libjpeg-turbo" ON)
if(WIN32)
  set(_guetzli_static_default ON)
else()
  set(_guetzli_static_default OFF)
endif()
option(GUETZLI_STATIC "Self-contained binary (static libgcc/libstdc++/winpthread and libraries)"
       ${_guetzli_static_default})
option(GUETZLI_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(GUETZLI_BUILD_TESTS "Build the doctest unit tests" ON)
set(GUETZLI_EXTERNAL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/external/install"
    CACHE PATH "Install prefix of the external dependency superbuild")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Same optimization flags as the Makefile build. -ffast-math is part of the output contract:
# changing it changes the generated JPEGs (tests/regress).
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")

include(cmake/Dependencies.cmake)

# ---------------------------------------------------------------------------------------------
# Library objects (everything except main), shared by the executable and the unit tests
# ---------------------------------------------------------------------------------------------
file(GLOB GUETZLI_SOURCES CONFIGURE_DEPENDS guetzli/*.cc)
list(REMOVE_ITEM GUETZLI_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/guetzli/guetzli.cc")
file(GLOB CLGUETZLI_SOURCES CONFIGURE_DEPENDS clguetzli/*.cpp)

add_library(guetzli_objs OBJECT
  ${GUETZLI_SOURCES}
  ${CLGUETZLI_SOURCES}
  third_party/butteraugli/butteraugli/butteraugli.cc
  third_party/minilzo/minilzo.c)
# The Makefile compiles minilzo.c with the C++ compiler; keep that until parity is no longer needed.
set_source_files_properties(third_party/minilzo/minilzo.c PROPERTIES LANGUAGE CXX)

# Include order matches the Makefile (some headers rely on it).
target_include_directories(guetzli_objs PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/butteraugli"
  "${CMAKE_CURRENT_SOURCE_DIR}/clguetzli"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenCL/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenCL-Wrapper"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/minilzo"
  ${GUETZLI_DEP_INCLUDE_DIRS})

if(MINGW)
  target_compile_options(guetzli_objs PUBLIC -Wno-unknown-pragmas)
endif()
if(GUETZLI_WARNINGS_AS_ERRORS)
  target_compile_options(guetzli_objs PUBLIC -Werror)
endif()

if(GUETZLI_WITH_CUDA)
  target_compile_definitions(guetzli_objs PUBLIC __USE_CUDA__)
endif()
if(GUETZLI_WITH_OPENCL)
  target_compile_definitions(guetzli_objs PUBLIC __USE_OPENCL__ CL_TARGET_OPENCL_VERSION=300
                                                 CL_HPP_MINIMUM_OPENCL_VERSION=200)
endif()
if(GUETZLI_WITH_FULL_JPEG)
  target_compile_definitions(guetzli_objs PUBLIC __SUPPORT_FULL_JPEG__)
endif()

# Link interface (libraries, options) for executables built from guetzli_objs; link guetzli_objs
# directly as well: object files do not propagate through an INTERFACE library.
add_library(guetzli_link INTERFACE)
target_link_libraries(guetzli_link INTERFACE guetzli_objs ${GUETZLI_DEP_LIBRARIES})  # objs: usage requirements
if(GUETZLI_STATIC)
  target_link_options(guetzli_link INTERFACE -static)
endif()

# ---------------------------------------------------------------------------------------------
# Executable and tests
# ---------------------------------------------------------------------------------------------
add_executable(guetzli guetzli/guetzli.cc)
target_link_libraries(guetzli PRIVATE guetzli_objs guetzli_link)

enable_testing()
if(GUETZLI_BUILD_TESTS)
  file(GLOB UNIT_TEST_SOURCES CONFIGURE_DEPENDS tests/unit/*.cpp)
  add_executable(guetzli_tests ${UNIT_TEST_SOURCES})
  target_include_directories(guetzli_tests PRIVATE third_party/doctest)
  target_link_libraries(guetzli_tests PRIVATE guetzli_objs guetzli_link)
  add_test(NAME unit COMMAND guetzli_tests)
endif()

# Extra guetzli flags for the smoke test. Default --c (CPU) so it does not depend on GPU drivers;
# set it empty to let --auto pick a device (e.g. pocl on Linux CI).
set(GUETZLI_SMOKE_FLAGS "--c" CACHE STRING "Extra guetzli flags for the smoke test")
find_program(GUETZLI_BASH bash)
if(GUETZLI_BASH)
  add_test(NAME smoke COMMAND "${GUETZLI_BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.sh"
                              "$<TARGET_FILE:guetzli>" "${GUETZLI_SMOKE_FLAGS}")
endif()

include(cmake/KernelHeaders.cmake)
