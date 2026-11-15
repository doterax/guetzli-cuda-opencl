# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GPU-accelerated fork of Google Guetzli (perceptual JPEG encoder). Adds CUDA and OpenCL backends (originally from Tencent) to the Butteraugli comparison / block-zeroing hot paths. C++20 host code, built with CMake + Ninja (MinGW-w64 GCC on Windows, GCC on Linux, Clang on macOS); the legacy Makefile still works until phase 6 of the modernization. See `BUILD.md` for prerequisites and CI details.

## Commands

```bash
cmake -S external -B external/build -DCMAKE_BUILD_TYPE=Release && cmake --build external/build   # one-time deps -> external/install
cmake --preset release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++   # force MinGW GCC on Windows (Clang-in-MSVC-mode breaks)
cmake --build --preset release        # -> build/release/bin/guetzli[.exe] and guetzli_tests[.exe]
ctest --preset release                # doctest unit tests (tests/unit) + smoke test (tests/run_tests.sh, --c)
cmake --preset release -DGUETZLI_WITH_CUDA=OFF                           # build without CUDA
cmake --build --preset release --target check-kernel-headers             # verify committed OpenCL header (CI runs this)
cmake --build --preset release --target kernel-headers                   # regenerate embedded kernel headers -- explicit only, CUDA 11.8 (see below)
```

On this Windows machine use `/c/mingw64/bin` first in PATH (CMake 4.1, Ninja, GCC 15); for `kernel-headers` configure with `-DGUETZLI_NVCC_CCBIN="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64" -DGUETZLI_NVCC_EXTRA_FLAGS=-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`. Legacy: `make CONFIG=release` (-> `bin/release/`), `make unit-test`, `make cuda-headers`.

To exercise a specific backend directly:

```bash
build/release/bin/guetzli --verbose --opencl tests/input/fro_small.png out.jpg
tests/run_tests.sh build/release/bin/guetzli --opencl     # 2nd arg = extra flags
build/release/bin/guetzli --version                       # also enumerates CUDA/OpenCL devices
```

Output-regression harness (tiny inputs, safe to run on a busy GPU): `tests/regress/regress.sh build/release/bin/guetzli.exe cpu opencl cuda checkcl checkcuda fallback-opencl fallback-cuda` compares output md5s to `tests/regress/expected/<mode>.md5` (`UPDATE=1` rewrites baselines; about a minute for all modes). `fallback-*` modes set `GUETZLI_TEST_GPU_FAIL=1` to inject a GPU init failure and expect the CPU result; `check*` modes also must reproduce the CPU baseline.

GPU failures throw `GpuError` (`backend/gpu_error.h`, via `CL_CHECK`/`CU_CHECK`); `Application::run()` (`app/application.cpp`) catches it, rebuilds the backend as CPU and reruns the job. Don't reintroduce `abort()`/`exit()` in backend code.

CI runners have no GPU, so CI only exercises the CPU fallback path; GPU changes must be verified locally.

## Architecture

**Layers (modernization in progress, see `docs/design/modernization.md`).** `app/` (CLI11 -> `Config`, composition root `Application`, `main.cpp`), `config/` (value types), `logging/` (`ILogger`; `ConsoleLogger` sends info to stdout, everything else to stderr; `ProcessStats::logger` carries the `--verbose` trace), `core/io/` (file I/O and PNG/TIFF decoding). `backend/` (factory + CPU/OpenCL/CUDA/check implementations of the encoder's interfaces, including the OpenCL/CUDA host code). `clguetzli/` keeps the kernel sources, the embedded kernel headers, `lzodec` and the optimized CPU butteraugli (`clbutter_comparator`). The encoder (`guetzli/`) is otherwise still the old structure. CMake builds `guetzli_core` (config, logging, core, guetzli, butteraugli; no GPU code) and `guetzli_backend`; the ctest `layering` fails if core code includes `backend/`, `clguetzli/` or GPU headers.

**Backend selection through injected interfaces (no global mode).** The encoder receives a `Backend` (`guetzli/backend.h`): `IComparatorFactory` (`guetzli/comparator.h`), `IBlockZeroingSearch` (`guetzli/block_zeroing.h`), `IDctOps` (`guetzli/dct_ops.h`, passed to every `OutputImage`). The interfaces live in core next to their consumer; implementations are in `backend/cpu`, `backend/gpu` (shared OpenCL/CUDA code over the typed launcher interfaces in `backend/gpu/gpu_kernels.h`), `backend/opencl`, `backend/cuda`. `backend/factory.h` is the only place that knows them: `GpuDevices` (owned by `Application::run()`; holds the OpenCL and CUDA devices of the run, `--device` as `GpuOptions`), `resolveBackendKind()` (Auto probes CUDA, then OpenCL, then CPU; a probe initializes the device and later backends reuse it), `makeBackend()`. `Application` resolves `Config::backend` (CLI11 in `app/cli.cpp`: `--backend auto|cpu|opencl|cuda`, `--check`, legacy aliases `--c`, `--opencl`, `--cuda`, `--checkcl`, `--checkcuda`, `--auto`; last one wins). `guetzli::ButteraugliComparator` runs the backend-dependent butteraugli operations through an injected `ButteraugliOps` (CPU: the `*Opt` functions in `clguetzli/clbutter_comparator.cpp`; GPU: kernels for images > 100×100, the reference `butteraugli.cc` code otherwise); whole-image `Compare()` is per backend. Never add backend checks to the encoder: add or extend an interface.

**`--checkcl` / `--checkcuda`** wrap the GPU backend in decorators (`backend/check/`) that run GPU and CPU per stage, log `CHK ...` lines for differences and return the CPU result, so the output equals `--c`. Use them to validate kernel changes for numerical parity.

**Single kernel source for two targets.** `clguetzli/clguetzli.cl` is the one kernel source:
- compiled as OpenCL at runtime;
- `#include`d by `clguetzli.cu` (with `__USE_CUDA__`) and compiled by `nvcc` to PTX.
Changes to `clguetzli.cl` must stay valid in both dialects. It is no longer compiled as C++ (the `clguetzli.cl.cpp` emulation was dropped). `clguetzli.cl.h` is host-only (`ocl_channels`, `ocu_channels`, and a `channel_info` that must match the kernel's layout). The two kernel helpers host code needs (`CalcOpsinDynamicsImage`, the float sRGB LUT) are ported in `guetzli/opsin_block.cc`. Its coefficient tables are deliberately non-const: with `-ffast-math`, const tables change the results in the last bits.

**Embedded, LZO-compressed kernels.** Kernels are not loaded from disk. `make cuda-headers` builds `minilzoc/` (LZO compressor tool), compresses the PTX / `.cl` source, and `format_header.py` turns them into byte arrays in `clguetzli/clguetzli_cu_ptx.h` and `clguetzli/clguetzli_cl_src.h`. These generated headers are **committed** and normal builds/CI never regenerate them — if you edit `clguetzli.cl`/`.cl.h`/`.cu`, run `make cuda-headers` and commit both headers with the change. The PTX must come from the pinned toolkit (CUDA 11.8, `KERNEL_NVCC_VERSION`; `make cuda-headers` refuses others): newer toolkits emit PTX that older drivers reject (issue #15). `CUDA_ARCH` defaults to `compute_75`. At runtime `lzodec.cpp` decompresses them.

**CUDA is loaded dynamically.** The `.cu` file is never linked; PTX is loaded via the driver API (`cuModuleLoadDataEx`) and all driver entry points go through the `CudaDriver` value (`backend/cuda/cuda_driver.{h,cpp}`, function pointers loaded from `nvcuda.dll` / `libcuda.so`), so the binary starts on machines without NVIDIA drivers. Add any new `cu*` driver call to `CudaDriver` rather than calling it directly.

**Device memory:** buffers are RAII handles (`ClBuffer`/`CuBuffer`, three-plane `ClChannels`/`CuChannels`) from the context: `zeroed(n)` zero-fills, `upload(n, ptr)` uploads, `uninitialized(n)` leaves contents undefined — use the latter only when a kernel writes every element before it is read (many kernels skip borders and rely on zero-fill). Destroying a handle returns the buffer to the context's pool (ordering relies on the single in-order queue/stream); limit lifetimes with scopes. Constant tables (blur weights, DoMask LUTs, sRGB LUT, convolution scales) live in the context's `ConstantCache` (`backend/gpu/constant_cache.h`). The GPU comparator keeps the reference image resident in a `GpuCompareSession` and never round-trips the output image through the host.

**GPU host code (`backend/opencl`, `backend/cuda`):** `opencl_context` / `cuda_context` = context, device, kernels, buffer pool and constant tables (`OpenClContext` / `CudaContext`, created on first use, no singletons); `opencl_kernels` / `cuda_kernels` = the launchers for each kernel (`OpenClKernels` / `CudaKernels`, implementing `GpuKernels`, owning the context); `cuda_memory_pool` = CUDA allocation pool. The two launcher files mirror each other but keep their own float constants and launch shapes; the output depends on them. The OpenCL program binary is cached on disk after first compile (user cache dir, or an `opencl_cache/` folder next to the executable if present).

**Feature defines:** `CUDA` → `__USE_CUDA__`, `OPENCL` → `__USE_OPENCL__`, `FULL_JPEG` → `__SUPPORT_FULL_JPEG__` (JPEG input via libjpeg-turbo). Guard new backend code accordingly.

## Conventions

- Version string lives in `guetzli/version.h` (`kVersion`); releases are bumped there and tagged `v*`, which triggers the GitHub Release job in `.github/workflows/ci.yml`.
- Changes that alter generated JPEG output should be avoided/discussed first (per `CONTRIBUTING.md`); speedups must preserve output.
- The Windows release binary is statically linked for self-contained distribution — keep new dependencies linkable statically.
- A full modernization (C++20, CMake, CLI11, doctest, no globals) is planned in `docs/design/modernization.md`; its "Start here" section is the handoff for starting that work.
