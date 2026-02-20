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
6. [Makefile Reference](#makefile-reference)
7. [GitHub Actions CI/CD](#github-actions-cicd)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Build third-party libraries (one-time)
make external

# 2. Build guetzli
make CONFIG=release

# 3. Run smoke tests
make test
```

The binary is written to `bin/release/guetzli` (or `guetzli.exe` on Windows).

---

## Prerequisites

### All Platforms

| Tool | Version | Purpose |
|------|---------|---------|
| C++11 compiler | GCC ≥ 5 / Clang ≥ 3.5 | Compile C++ sources |
| CMake | ≥ 3.18 | Build external dependencies |
| Python 3 | ≥ 3.6 | Generate embedded GPU-kernel headers |
| GNU Make | ≥ 3.81 | Build system |

### Windows

Install **MSYS2** (<https://www.msys2.org/>) and from the MINGW64 shell run:

```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake \
          mingw-w64-x86_64-make make python3
```

Add `C:\msys64\mingw64\bin` and `C:\msys64\usr\bin` to your `PATH`.

Alternatively install standalone **MinGW-w64** + CMake + Python and use Git
Bash or any Unix-like shell.

### Linux (Ubuntu / Debian)

```bash
sudo apt-get update
sudo apt-get install build-essential cmake python3 pkg-config
```

### macOS

```bash
brew install cmake python3 pkg-config
```

### Optional: CUDA

Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
(version ≥ 11.0) and ensure `CUDA_PATH` points to the installation directory.
The CUDA driver library (`nvcuda.dll` / `libcuda.so`) is loaded at **runtime**
— the binary will start and fall back gracefully on machines without an
NVIDIA GPU.

### Optional: OpenCL

- **Linux**: `sudo apt-get install opencl-headers ocl-icd-opencl-dev`
- **Windows**: Built from source as part of the external dependencies.
- **macOS**: OpenCL is available natively via Apple's framework — no extra
  packages needed.

---

## Building External Dependencies

All third-party libraries (zlib, libpng, libjpeg-turbo, libtiff, OpenCL
headers + ICD loader) are built from source into `external/install/` via a
CMake superbuild.

```bash
make external
```

This downloads, configures, builds, and installs every dependency.  The result
is cached — you only need to run it once (or after editing
`external/CMakeLists.txt`).

Under the hood this runs:

```bash
cd external
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build --config Release
```

See [external/README.md](external/README.md) for details on library versions
and selective builds.

---

## Building Guetzli

After the external dependencies are in place:

```bash
# Release build with CUDA + OpenCL + full JPEG support (default)
make CONFIG=release

# Debug build
make CONFIG=debug

# OpenCL only (no CUDA)
make FEATURES="OPENCL FULL_JPEG"

# Specify compiler explicitly
make CXX=g++

# Enable link-time optimisation (slower compile, faster binary)
make LTO=1
```

### Output

| Artifact | Path |
|----------|------|
| Executable | `bin/release/guetzli[.exe]` |
| Static library | `bin/release/libguetzli_static.a` (via `make static`) |

---

## Running Tests

```bash
make test
```

This runs a small smoke-test suite (`tests/run_tests.sh` on Unix,
`tests/run_tests.bat` on Windows) that verifies:

1. The binary exists and is runnable.
2. `--version` exits without crashing.
3. A tiny PNG is converted to JPEG successfully.
4. The output has valid JPEG magic bytes (`FF D8`).
5. The output size is within a sane range.

Only the smallest test image (`tests/input/fro_small.png`, 6 KB) is used to
keep CI time low.

---

## Makefile Reference

### Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG` | `release` | `debug` or `release` |
| `FEATURES` | `CUDA OPENCL FULL_JPEG` | Space-separated feature list |
| `CXX` | auto-detected | C++ compiler |
| `CUDA_ARCH` | `compute_75` | CUDA virtual architecture for PTX |
| `LTO` | (unset) | Set to `1` to enable link-time optimisation |

### Targets

| Target | Description |
|--------|-------------|
| `all` | Build executable (default) |
| `static` | Build static library |
| `external` | Build external dependencies via CMake |
| `test` | Run smoke tests |
| `cuda-headers` | (Re)generate embedded CUDA/OpenCL kernel headers |
| `install` | Install to `PREFIX` (Linux/macOS) |
| `dist` | Create release archive |
| `clean` | Remove all build artefacts |
| `help` | Show available targets and options |

### Feature Flags

| Flag | Preprocessor Define | Effect |
|------|---------------------|--------|
| `CUDA` | `__USE_CUDA__` | CUDA acceleration (driver loaded at runtime) |
| `OPENCL` | `__USE_OPENCL__` | OpenCL acceleration |
| `FULL_JPEG` | `__SUPPORT_FULL_JPEG__` | JPEG → JPEG re-optimisation via libjpeg-turbo |

---

## GitHub Actions CI/CD

The CI pipeline lives in `.github/workflows/ci.yml` and runs automatically.

### Workflow Triggers

| Event | Condition | What Happens |
|-------|-----------|--------------|
| **Pull Request** | Targeting `main`, `master`, or `develop` | Build + test on all platforms. **Must pass before merge.** |
| **Push** | To `main` or `master` | Build + test + upload artefacts |
| **Tag** | `v*` (e.g. `v1.3.0`) | Build + test + **create GitHub Release** with binaries |

### Build Matrix

| Name | Runner | Features | CUDA? |
|------|--------|----------|-------|
| `linux-opencl` | `ubuntu-24.04` | OpenCL + Full JPEG | No |
| `linux-full` | `ubuntu-24.04` | CUDA + OpenCL + Full JPEG | Yes (compile only) |
| `win-full` | `windows-2022` (MSYS2) | CUDA + OpenCL + Full JPEG | Yes (compile only) |
| `macos-opencl` | `macos-14` (Apple Silicon) | OpenCL + Full JPEG | No |

> **Note:** GitHub-hosted runners have no GPU.  The CUDA and OpenCL code is
> compiled and linked but cannot execute GPU kernels.  The smoke tests run the
> CPU-fallback path.  Full GPU testing requires self-hosted runners.

### Pipeline Steps (per matrix entry)

1. **Checkout** source code.
2. **MSYS2** setup (Windows only — provides MinGW-w64 GCC + make + cmake).
3. **Cache** `external/install/` keyed on `external/CMakeLists.txt` hash.
4. **Install system packages** (apt / brew / MSYS2 pacman).
5. **Install CUDA Toolkit** (conditional, via `Jimver/cuda-toolkit` action).
6. **Build external dependencies** (if cache miss).
7. **Build guetzli** (`make CONFIG=release FEATURES="..."`)
8. **Run smoke tests** (`make test` — uses only `fro_small.png`, 6 KB).
9. **Upload** build artefact.
10. *(On `v*` tag)* **Create GitHub Release** via `softprops/action-gh-release`.

### Release Automation

When you push a tag like `v1.3.0`:

1. All four matrix builds run.
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

### "make: command not found"

- **Windows**: Install MSYS2 and add MinGW-w64 to `PATH`.
- **Linux**: `sudo apt-get install build-essential`

### "cmake: command not found"

- **Windows**: `pacman -S mingw-w64-x86_64-cmake` (MSYS2) or install from
  <https://cmake.org/download/>.
- **Linux**: `sudo apt-get install cmake`

### "nvcc: command not found"

- Install the CUDA Toolkit and set the `CUDA_PATH` environment variable.
- CUDA is **optional** — you can build without it: `make FEATURES="OPENCL FULL_JPEG"`

### "OpenCL headers not found"

- **Linux**: `sudo apt-get install opencl-headers ocl-icd-opencl-dev`
- **Windows / macOS**: Run `make external` first — it builds the OpenCL ICD
  loader from source.

### Build errors with Clang in MSVC mode on Windows

If CMake picks up LLVM Clang targeting MSVC, force MinGW GCC:

```bash
make CXX=g++
```

Or pass the compiler to CMake when building externals:

```bash
cd external
cmake -B build -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Permission denied when linking on Windows

Another process may have the binary open.  Close any running `guetzli.exe`
instance and retry.

