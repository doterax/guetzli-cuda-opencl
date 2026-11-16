# Guetzli CUDA/OpenCL — Build Guide

This document explains how to build the project from source and how the
GitHub Actions CI/CD pipeline works.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Building External Dependencies](#building-external-dependencies)
4. [Building Guetzli](#building-guetzli)
5. [Running Tests](#running-tests)
6. [Embedded GPU Kernels](#embedded-gpu-kernels)
7. [GitHub Actions CI/CD](#github-actions-cicd)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Build third-party libraries (one-time; installs to external/install)
cmake -S external -B external/build -DCMAKE_BUILD_TYPE=Release
cmake --build external/build

# 2. Configure, build, test
cmake --preset release            # on Windows/MinGW add: -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build --preset release    # -> build/release/bin/guetzli[.exe], guetzli_tests[.exe]
ctest --preset release            # unit tests + smoke test
```

---

## Prerequisites

### All Platforms

| Tool | Version | Purpose |
|------|---------|---------|
| C++20 compiler | GCC ≥ 11 / Clang ≥ 14 / Apple Clang ≥ 14 | Compile the sources |
| CMake | ≥ 3.24 | Build system (and the external-dependency superbuild) |
| Ninja | any recent | Generator used by the presets |
| Bash | any | Smoke test (`tests/run_tests.sh`); Git Bash or MSYS2 on Windows |

### Windows

Install **MSYS2** (<https://www.msys2.org/>) and from the **UCRT64** shell run:

```bash
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-pkg-config
```

Add `C:\msys64\ucrt64\bin` to your `PATH`. The release binary is built this way (UCRT64: MinGW-w64 GCC
linking the Universal C Runtime, see [README](README.md#windows-runtime)). A standalone MinGW-w64 GCC with CMake and Ninja works
as well. Pass `-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++` when configuring if CMake would
otherwise pick another compiler (see [Troubleshooting](#troubleshooting)).

### Linux (Ubuntu / Debian)

```bash
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build pkg-config
```

### macOS

```bash
brew install cmake ninja pkg-config
```

### Optional: CUDA

Building with CUDA support only needs the CUDA Toolkit's `cuda.h` header (any version ≥ 11.0;
set `CUDA_PATH` to the installation directory). The CUDA driver library (`nvcuda.dll` /
`libcuda.so`) is loaded at **runtime** — the binary will start and fall back gracefully on
machines without an NVIDIA GPU. `nvcc` is only needed to regenerate the embedded kernels
([Embedded GPU Kernels](#embedded-gpu-kernels)).

### Optional: OpenCL

- **Linux**: `sudo apt-get install opencl-headers ocl-icd-opencl-dev`
- **Windows**: Built from source as part of the external dependencies.
- **macOS**: OpenCL is available natively via Apple's framework — no extra
  packages needed.

---

## Building External Dependencies

All third-party libraries (zlib, libpng, libjpeg-turbo, libtiff, OpenCL
headers + ICD loader) are built from source into `external/install/` via a
CMake superbuild:

```bash
cmake -S external -B external/build -DCMAKE_BUILD_TYPE=Release
cmake --build external/build
```

This downloads, configures, builds, and installs every dependency. You only need to run it once
(or after editing `external/CMakeLists.txt`). The main build looks there by default
(`GUETZLI_EXTERNAL_PREFIX`); libraries not found there are taken from the system.

See [external/README.md](external/README.md) for details on library versions
and selective builds.

---

## Building Guetzli

```bash
cmake --preset release                          # or: debug
cmake --build --preset release
cmake --install build/release --prefix /usr/local   # optional: installs bin/guetzli
```

Presets: `release`, `debug`, and the CI presets `ci-linux`, `ci-linux-asan`, `ci-windows-mingw`,
`ci-macos`. Binaries are written to `build/<preset>/bin/`.

### Options

| Option | Default | Effect |
|--------|---------|--------|
| `GUETZLI_WITH_CUDA` | `ON` | CUDA backend (`__USE_CUDA__`; driver loaded at runtime, needs only `cuda.h`) |
| `GUETZLI_WITH_OPENCL` | `ON` | OpenCL backend (`__USE_OPENCL__`) |
| `GUETZLI_WITH_FULL_JPEG` | `ON` | JPEG input via libjpeg-turbo (`__SUPPORT_FULL_JPEG__`) |
| `GUETZLI_STATIC` | `ON` on Windows | Self-contained binary (static runtime and libraries) |
| `GUETZLI_WARNINGS_AS_ERRORS` | `OFF` | `-Werror` |
| `GUETZLI_SANITIZE` | (empty) | `-fsanitize=<value>`, e.g. `address` (GCC/Clang on Linux/macOS) |
| `GUETZLI_BUILD_TESTS` | `ON` | Build the doctest unit tests (`guetzli_tests`) |
| `GUETZLI_EXTERNAL_PREFIX` | `external/install` | Install prefix of the dependency superbuild |
| `GUETZLI_SMOKE_FLAGS` | `--c` | Extra guetzli flags for the smoke test (empty = auto-detect backend) |

For example, an OpenCL-only build: `cmake --preset release -DGUETZLI_WITH_CUDA=OFF`.

Release builds use `-O3 -ffast-math`. The flag is part of the output contract: changing compiler
flags changes the generated JPEGs (see `tests/regress`).

---

## Running Tests

```bash
ctest --preset release
```

runs:

1. `unit`: the doctest unit tests (`tests/unit/`, binary `guetzli_tests`).
2. `smoke`: `tests/run_tests.sh` checks that the binary runs, `--version` works and a tiny PNG
   (`tests/input/fro_small.png`, 6 KB) becomes a valid JPEG of sane size.
3. `layering`: core code (`config/`, `logging/`, `core/`) must not include `backend/`, `kernels/` or
   OpenCL/CUDA headers.

The byte-exact output regression harness is run separately (it needs a GPU for the GPU modes, so
CI does not run it):

```bash
tests/regress/regress.sh build/release/bin/guetzli.exe cpu opencl cuda checkcl checkcuda fallback-opencl fallback-cuda
```

---

## Embedded GPU Kernels

The kernels live in `kernels/`: `guetzli.cl` is the single kernel source, compiled as OpenCL C at
runtime and `#include`d by `guetzli.cu`, which `nvcc` compiles to PTX. Both are embedded in the
executable, LZO-compressed, as **committed** headers: `kernels/guetzli_cl_src.h` (OpenCL source)
and `kernels/guetzli_cu_ptx.h` (PTX). Normal builds and CI never regenerate them.

| Target | Description |
|--------|-------------|
| `check-kernel-headers` | Fail if the committed OpenCL header does not match `kernels/guetzli.cl` (run in CI) |
| `kernel-headers` | Regenerate both headers (explicit only; commit the result) |

If you change `kernels/guetzli.cl`, `kernels/guetzli.cu` or `backend/gpu/kernel_types.h`, run
`cmake --build --preset release --target kernel-headers` and commit both headers with your change.
The PTX must be built with the pinned toolkit (**CUDA 11.8**, `GUETZLI_KERNEL_NVCC_VERSION`): drivers
only JIT-compile PTX versions they know, and PTX from a newer toolkit is rejected by older drivers
([issue #15](https://github.com/doterax/guetzli-cuda-opencl/issues/15)). The target refuses other
`nvcc` versions.

| Option | Default | Description |
|--------|---------|-------------|
| `GUETZLI_KERNEL_NVCC_VERSION` | `11.8` | Required `nvcc` release |
| `GUETZLI_CUDA_ARCH` | `compute_75` | PTX target architecture |
| `GUETZLI_NVCC_CCBIN` | (empty) | Host compiler directory for `nvcc -ccbin` (e.g. an MSVC 14.29 `bin/Hostx64/x64` on Windows) |
| `GUETZLI_NVCC_EXTRA_FLAGS` | (empty) | Extra `nvcc` flags for local toolchain quirks (e.g. `-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`) |

---

## GitHub Actions CI/CD

The CI pipeline lives in `.github/workflows/ci.yml` and runs automatically.

### Workflow Triggers

| Event | Condition | What Happens |
|-------|-----------|--------------|
| **Pull Request** | Targeting `main`, `master`, `develop` or `refactor/modern-cpp` | Build + test on all platforms. **Must pass before merge.** |
| **Push** | To `main`, `master` or `refactor/modern-cpp` | Build + test + upload artefacts |
| **Tag** | `v*` (e.g. `v1.3.0`) | Build + test + **create GitHub Release** with binaries |

### Jobs

| Name | Runner | Preset | Features |
|------|--------|--------|----------|
| `clang-format` | `ubuntu-latest` | — | Formatting check (clang-format 21.1.5) |
| `linux-opencl` | `ubuntu-24.04` | `ci-linux` | OpenCL + Full JPEG |
| `linux-full` | `ubuntu-24.04` | `ci-linux` | CUDA (compile only) + OpenCL + Full JPEG |
| `linux-asan` | `ubuntu-24.04` | `ci-linux-asan` | AddressSanitizer/LeakSanitizer on the CPU path; no artefact |
| `win-full` | `windows-2022` (MSYS2 UCRT64) | `ci-windows-mingw` | CUDA (compile only) + OpenCL + Full JPEG |
| `macos-opencl` | `macos-26` (Apple Silicon) | `ci-macos` | OpenCL + Full JPEG |

> **Note:** GitHub-hosted runners have no GPU. The CUDA and OpenCL code is compiled and linked
> but cannot execute GPU kernels (Linux runs the smoke test on pocl, a CPU OpenCL). GPU changes
> must be verified locally with `tests/regress`.

### Pipeline Steps (per build job)

1. **Checkout** source code.
2. **MSYS2** setup (Windows only — UCRT64: MinGW-w64 GCC, CMake, Ninja).
3. **Cache** `external/install/` keyed on `external/CMakeLists.txt` hash.
4. **Install system packages** (apt / brew).
5. **Install CUDA Toolkit** (conditional, via `Jimver/cuda-toolkit`) — only for the `cuda.h`
   header; `nvcc` is not used in CI.
6. **Build external dependencies** (if cache miss).
7. **Configure and build** with the job's preset.
8. **Unit and smoke tests** (`ctest`); the ASan job also runs injected GPU failures (CPU fallback).
9. **Verify kernel headers**: `check-kernel-headers`, then `git diff --exit-code` on both
   committed headers (the build must not have regenerated them).
10. **Binary diagnostics** (`ldd` / `otool -L` / `objdump -p`, `--version`).
11. **Upload** build artefact.
12. *(On `v*` tag)* **Create GitHub Release** via `softprops/action-gh-release`.

### Release Automation

When you push a tag like `v1.3.0`:

1. All matrix builds run.
2. On success, the **release** job downloads all artefacts, packages them into
   per-platform archives, and creates a GitHub Release with auto-generated
   release notes.

| Platform | Archive |
|----------|---------|
| Linux x64 | `guetzli-cuda-opencl-linux-x64.tar.gz` |
| Windows x64 | `guetzli-cuda-opencl-win-x64.zip` |
| macOS arm64 | `guetzli-cuda-opencl-macos-arm64.tar.gz` |

### Setting Up GitHub Actions for Your Fork / Mirror

1. **Enable Actions** — Go to your GitHub repo → **Settings → Actions →
   General** → select *Allow all actions and reusable workflows*.
2. **No secrets needed** — the workflow uses only public actions and does not
   require any repository secrets for building and testing.
3. **Branch protection (recommended)** — Go to **Settings → Branches → Add
   rule** for `main`:
   - ✅ Require a pull request before merging
   - ✅ Require status checks to pass before merging
     - Add required checks: `Build (linux-opencl)`, `Build (win-full)`,
       `Build (macos-opencl)`
   - ✅ Require branches to be up-to-date before merging
4. **Creating a release** — push a version tag:
   ```bash
   git tag v1.3.0
   git push origin v1.3.0
   ```
   The workflow will build all platforms and create a GitHub Release
   automatically.
5. **Caching** — external dependencies are cached per-OS.  The cache key
   includes the hash of `external/CMakeLists.txt`, so bumping a dependency
   version automatically invalidates the cache.

---

## Troubleshooting

### "cmake: command not found" / "ninja: command not found"

- **Windows**: `pacman -S mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja` (MSYS2 UCRT64), or install
  from <https://cmake.org/download/> and <https://ninja-build.org/>.
- **Linux**: `sudo apt-get install cmake ninja-build`

### "nvcc: command not found"

- Only the `kernel-headers` target needs `nvcc` (CUDA 11.8). Normal builds use the committed headers.
- CUDA is **optional** — you can build without it: `-DGUETZLI_WITH_CUDA=OFF`.

### "OpenCL headers not found"

- **Linux**: `sudo apt-get install opencl-headers ocl-icd-opencl-dev`
- **Windows / macOS**: build the external dependencies first — the superbuild builds the
  OpenCL ICD loader from source.

### Build errors with Clang in MSVC mode on Windows

If CMake picks up LLVM Clang targeting MSVC, force MinGW GCC, both for the main build and the
externals:

```bash
cmake --preset release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake -S external -B external/build -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release
```

### Permission denied when linking on Windows

Another process may have the binary open.  Close any running `guetzli.exe`
instance and retry.
