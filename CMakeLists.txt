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
# The embedded kernel headers (kernels/guetzli_cl_src.h, kernels/guetzli_cu_ptx.h) are committed and
# only consumed. Regenerate them explicitly with the `kernel-headers` target (pinned CUDA toolkit, §7);
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
set(GUETZLI_SANITIZE "" CACHE STRING "Build with -fsanitize=<value>, e.g. address (GCC/Clang on Linux or macOS)")
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

# -ffast-math is part of the output contract:
# changing it changes the generated JPEGs (tests/regress).
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")

include(cmake/Dependencies.cmake)

# ---------------------------------------------------------------------------------------------
# Library objects (everything except main), shared by the executable and the unit tests.
#   guetzli_core     config/, logging/, core/ (encoder, image, JPEG, butteraugli, I/O) -- no GPU code
#   guetzli_backend  backend/ (CPU, OpenCL, CUDA, --check) and the embedded kernels (kernels/)
# The dependency rule (core never includes backend/ or GPU headers) is checked by the `layering` test.
# ---------------------------------------------------------------------------------------------
file(GLOB CORE_SOURCES CONFIGURE_DEPENDS config/*.cpp logging/*.cpp core/*/*.cpp)
file(GLOB BACKEND_SOURCES CONFIGURE_DEPENDS backend/*.cpp backend/*/*.cpp kernels/*.cpp)

# Warning flags shared by all host code.
add_library(guetzli_flags INTERFACE)
if(MINGW)
  target_compile_options(guetzli_flags INTERFACE -Wno-unknown-pragmas)
endif()
if(WIN32)
  # fopen, getenv, strerror, tmpfile are standard C; the Microsoft CRT headers mark them deprecated.
  target_compile_definitions(guetzli_flags INTERFACE _CRT_SECURE_NO_WARNINGS)
endif()
if(GUETZLI_WARNINGS_AS_ERRORS)
  target_compile_options(guetzli_flags INTERFACE -Werror)
endif()
if(GUETZLI_SANITIZE)
  target_compile_options(guetzli_flags INTERFACE -fsanitize=${GUETZLI_SANITIZE} -fno-omit-frame-pointer)
  target_link_options(guetzli_flags INTERFACE -fsanitize=${GUETZLI_SANITIZE})
endif()

add_library(guetzli_core OBJECT ${CORE_SOURCES})
target_include_directories(guetzli_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" ${GUETZLI_DEP_INCLUDE_DIRS})
target_link_libraries(guetzli_core PUBLIC guetzli_flags)
if(GUETZLI_WITH_FULL_JPEG)
  target_compile_definitions(guetzli_core PUBLIC __SUPPORT_FULL_JPEG__)
endif()

add_library(guetzli_backend OBJECT
  ${BACKEND_SOURCES}
  third_party/minilzo/minilzo.c)
# Keep this order: the bundled OpenCL headers must come before the CUDA toolkit's older CL/ headers.
target_include_directories(guetzli_backend PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenCL/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenCL-Wrapper"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/minilzo"
  ${GUETZLI_DEP_INCLUDE_DIRS})
target_link_libraries(guetzli_backend PUBLIC guetzli_core)
if(GUETZLI_WITH_CUDA)
  target_compile_definitions(guetzli_backend PUBLIC __USE_CUDA__)
endif()
if(GUETZLI_WITH_OPENCL)
  # CL_HPP_MINIMUM_OPENCL_VERSION / CL_HPP_TARGET_OPENCL_VERSION are set by the OpenCL wrapper. The
  # CL_USE_DEPRECATED_* switches keep the Khronos C++ header's use of legacy APIs from warning.
  target_compile_definitions(guetzli_backend PUBLIC __USE_OPENCL__ CL_TARGET_OPENCL_VERSION=300
                             CL_USE_DEPRECATED_OPENCL_1_0_APIS CL_USE_DEPRECATED_OPENCL_1_1_APIS
                             CL_USE_DEPRECATED_OPENCL_1_2_APIS CL_USE_DEPRECATED_OPENCL_2_0_APIS
                             CL_USE_DEPRECATED_OPENCL_2_1_APIS CL_USE_DEPRECATED_OPENCL_2_2_APIS)
endif()

# Link interface (libraries, options) for executables built from the object libraries; link the
# object libraries directly as well: object files do not propagate through an INTERFACE library.
add_library(guetzli_link INTERFACE)
target_link_libraries(guetzli_link INTERFACE guetzli_backend ${GUETZLI_DEP_LIBRARIES})  # usage requirements
if(GUETZLI_STATIC)
  target_link_options(guetzli_link INTERFACE -static)
endif()

# ---------------------------------------------------------------------------------------------
# Executable and tests
# ---------------------------------------------------------------------------------------------
# Application layer (CLI, composition root); main.cpp only in the executable so tests can link the rest.
add_library(guetzli_app OBJECT app/cli.cpp app/application.cpp)
target_include_directories(guetzli_app PUBLIC third_party/CLI11)
target_link_libraries(guetzli_app PUBLIC guetzli_backend)

add_executable(guetzli app/main.cpp)
target_link_libraries(guetzli PRIVATE guetzli_app guetzli_backend guetzli_core guetzli_link)

enable_testing()
if(GUETZLI_BUILD_TESTS)
  file(GLOB UNIT_TEST_SOURCES CONFIGURE_DEPENDS tests/unit/*.cpp)
  add_executable(guetzli_tests ${UNIT_TEST_SOURCES})
  target_include_directories(guetzli_tests PRIVATE third_party/doctest)
  target_compile_definitions(guetzli_tests PRIVATE GUETZLI_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(guetzli_tests PRIVATE guetzli_app guetzli_backend guetzli_core guetzli_link)
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

# core/, config/ and logging/ must not include backend/, kernels/ or GPU headers.
add_test(NAME layering COMMAND "${CMAKE_COMMAND}" -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
                               -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckLayering.cmake")

include(cmake/KernelHeaders.cmake)

# `cmake --install build/release [--prefix /usr/local]` installs the executable to <prefix>/bin.
include(GNUInstallDirs)
install(TARGETS guetzli RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
