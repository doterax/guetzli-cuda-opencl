# Modernization: architecture & refactoring plan

Status: **draft, accepted direction** (2026-11-15). Work happens on the long-lived branch `refactor/modern-cpp`.

This document describes where the codebase is going and how we get there without breaking anyone. Read it before
opening a PR against the refactor branch.

### Start here (handoff, 2026-11-15)

All decisions in §3 are made; there are no open questions. **Phases 0 to 5 are done** (phase 0: `4b8d3f5`..`ff9f07f`,
phase 1: `7e1a9a4` + C++20 switch, phase 2: `7eba7f4`..`661a6ad`, phase 3: `cd2d241`..`74f75a0`, phase 4:
`4979de3`..`fe63f82`, phase 5: `3747847`..`925d84d`, phase 6: `4b7fd95`..`2e8adda`, on `refactor/modern-cpp`).
After phase 6 and before the release, the GPU items D1, D3, D4 and G2 and two exact CPU speedups were done
(`c3d2d5c`..`30bce6d`). **Phase 7** (§10) ships it as v3.0.0: version in `app/version.h`, release notes in
`docs/release-notes/v3.0.0.md` (the release job uses `docs/release-notes/<tag>.md` as the body), branch merged into
`master`, tag `v3.0.0`. What is still open is listed under "Open after v3.0.0" below.

### Open after v3.0.0

Carried over from the working notes (`todo.md`, not committed). File names there predate phase 6; the items are not
re-verified against the current code.
- **GPU memory is not accounted for** (todo 9): `--memlimit` is host-only; a compare holds ~18 float planes on the
  device, block zeroing ~24 B/px more. Query free VRAM / `CL_DEVICE_MAX_MEM_ALLOC_SIZE` before choosing the backend.
  A 30 MP image runs on a 6 GB card.
- **Small-image hazards in kernels** (14): unsigned `ysize - 8` in `clBlockDiffMapEx`, size 1 in
  `clDiffPrecomputeEx`, divide by zero in `Butteraugli8x8CornerEdgeDetectorDiff` for xsize < 7. Masked today because
  images below 32 px skip the comparator.
- **32-100 px images mix CPU and GPU math** (20): the `> 100` thresholds gate only the reference-side setup.
- **CUDA vs OpenCL numerics** (22): nvcc `-use_fast_math -ftz=true`, OpenCL `-cl-finite-math-only
  -cl-no-signed-zeros -cl-mad-enable`; threshold comparisons can flip, so the backends may write different JPEGs
  (they differ by a few hundred bytes on photos). Decide the policy deliberately.
- **Speed:** overlap serialization with the GPU compare (S6, `std::async`); separable `clMinSquareValEx` (S11);
  divergent `__constant__` lookups on CUDA (S12); device-to-device copy-backs and `clAverage5x5Ex` read-modify-writes
  (S13); pinned host memory, size-class CUDA pool, a second queue/stream (S17).
- **GPU migration:** G1 (device-resident `OutputImage`: coefficients and pixels stay on the GPU, ~1-1.5 s at 12 MP,
  changes `OutputImage` ownership), G3 (block-zeroing kernel: 1.27 s of ~21.5 s at 12 MP; warp efficiency of the
  8x4 tiling is 0.56-0.70, but no cheap predictor of per-block work exists (candidate count reaches 0.67-0.79) and
  launch shapes are within 2 %; the remaining route is several threads per candidate evaluation), G4 (Huffman
  coding on the GPU; must reproduce the bitstream exactly, probably not worth it).

Phase 6 notes. Order and scope decided 2026-11-16, all in phase 6:
1. Push the branch and fix CI. Done: everything passed on the first run (including `linux-asan`) except macOS,
   whose `macos-14` runner image is retired; the job now runs on `macos-15`.
2. Makefile removed; kernel tooling in CMake. Done. The kernel source moved to `kernels/` (`guetzli.cl`,
   `guetzli.cu`, generated `guetzli_cl_src.h` / `guetzli_cu_ptx.h`). `kernels/tools/embed_kernel.cpp` replaces
   `minilzoc` + `format_header.py` (no Python needed any more); the headers define `guetzli::kernels::k<Name>Lzo` and
   `k<Name>Size` (the uncompressed size). `kernels/embedded_kernels.{h,cpp}` replaces `lzodec` and decompresses to the
   exact size (no more guessing the ratio). minilzo is compiled as C. `cmake --install` replaces `make install`; the
   Makefile-era test scripts (`tests/run_tests.bat`, `tests/test_guetzli.bat`, `tests/output.md5`) are gone. The PTX
   rebuilt with nvcc 11.8 from the moved source is byte-identical to the committed one.
3. File moves. Done (pure renames + include paths; `.cc` became `.cpp`; `guetzli/`, `clguetzli/` and
   `third_party/butteraugli/` are gone). `core/jpeg/`: `JpegData`, reader, writer, Huffman, bit writer, entropy
   codes, quantization. `core/image/`: `OutputImage`, `IDctOps`, DCT/IDCT, colour/gamma, downsampling, the
   `JpegData` <-> pixels encoder/decoder (they need `OutputImage`, so they sit above `core/jpeg`).
   `core/quality/`: butteraugli (the former third-party copy, minus its Bazel files and `butteraugli_main.cc`),
   `ButteraugliComparator`, `Comparator`/`IComparatorFactory`, opsin helpers, quality and score.
   `core/encoder/`: also `Backend`, `IBlockZeroingSearch` (+ `order.inc`), `ProcessStats`, `debug_print`.
   `core/util/fast_log.h`, `app/version.h`. `clbutter_comparator` became `backend/cpu/optimized_butteraugli`,
   `clguetzli.cl.h` became `backend/gpu/kernel_types.h`. The layering check now covers `config/`, `logging/`,
   `core/`.
4. Naming sweep. Done, one mechanical commit. Convention in `.clang-tidy` (`readability-identifier-naming`):
   `PascalCase` types, `camelCase` functions, `snake_case_` private/protected members, `snake_case` public fields,
   parameters and locals, `kPascalCase` constants, `PascalCase` `enum class` values, `UPPER_CASE` plain-enum values
   and macros; acronyms are words (`JpegData`, `toSrgb`, `kDctBlockSize`). Done with clang-tidy's rename fixes (they
   find every reference), but with our own word splitting (clang-tidy turns `YUV444` into `YuV444`) and two passes
   (the second also normalizes acronyms in names clang-tidy's `CamelCase` accepts, like `JPEGData`). Pre-edits for
   names that would collide: `BitReaderState`'s state became private (`readSymbol` is a friend), the fdct table
   constants `C1`..`C7` became `m1`..`m7`, `PreProcessChannel`'s `blur`/`sharpen` flags became `do_*`, the unused
   union field `b_` went. **Verified with machine code:** every production object's code sections are
   byte-identical to the previous commit (compared COFF code sections + relocations, ignoring names), except one
   intended change, `kRangeLimit` becoming a `const` pointer (integer code in the YCbCr->RGB inline). Builds without
   GPU backends and without full JPEG (the `#else` branches clang-tidy did not see) compile and pass their tests.
5. clang-tidy clean. Done (commits `807590e`, `2803b28`, `990c03f`, `b195b58` + enforcement). The tree is clean
   with the release, no-GPU and no-full-JPEG configurations; CI runs clang-tidy 21.1.6 (PyPI, pinned like
   clang-format) in the `linux-full` job with every finding an error, and the CI presets build with
   `GUETZLI_WARNINGS_AS_ERRORS` (the only warnings were the Khronos C++ header's legacy-API deprecations, now
   switched off with `CL_USE_DEPRECATED_OPENCL_*_APIS`). Turned off on purpose (reasons in `.clang-tidy`):
   `readability-math-missing-parentheses`, `performance-enum-size`, `bugprone-implicit-widening-of-multiplication-result`.
   Auto-fixes were applied check by check (fixes deduplicated per diagnostic, braces in rounds: nested statements
   insert `}` at the same offset, which clang-apply-replacements merges) and gated on machine code (production
   objects byte-identical where expected), regress, and `--c --verbose` traces on larger images. Lessons: the
   implicit-bool fix wrote `while (n >>= 1 != 0u)` (same value by luck; check precedence after it); a mid-line
   `NOLINT` comment once commented out the rest of a coefficient table line (regress caught it; use
   `NOLINTNEXTLINE`); `bugprone-reserved-identifier`'s fix renamed `__restrict__` and half a header guard;
   `modernize-use-std-numbers` would replace a tuned coefficient that only resembles 1/sqrt(3).
6. `GUETZLI_LOG` / `ProcessStats` trace → `ILogger` (§5.7). Done. `OutputSelector`, `QuantMatrixSearch` and
   `FrequencyMasking` take an `ILogger&` and write the trace with `LogLevel::Debug` (same format strings; the
   quantization-matrix dump is `logQuantMatrix()`); `GUETZLI_LOG`, `GUETZLI_LOG_QUANT` and `debug_print.*` are gone.
   `ProcessStats` is three counters (`iterations`, `iterations_up`, `iterations_down`) instead of a string-keyed
   map, a logger pointer and an unused `filename`. The comparators' unused `ProcessStats*` parameter is gone from
   `IComparatorFactory::create()`. Verified: `--verbose` traces byte-identical (CPU on the larger images, OpenCL /
   CUDA / check on a regress input), regress unchanged.
7. OpenCL/CUDA launcher deduplication. Done. `backend/gpu/gpu_kernels_base.h` (`GpuKernelsBase<Derived, Types>`,
   CRTP) holds the butteraugli pipeline once; `OpenClKernels` / `CudaKernels` keep only the context, one `*Kernel()`
   launcher per kernel (argument packing, launch shape) and the DCT / block-zeroing round trips (1699 -> 376 + 385 +
   453 shared lines). The contexts gained `copy`, `readAsync`, `uploadAsync`, `finish` and a common
   `allocateMemory`. Numerics unified on the OpenCL variant (decided by the user): blur weights' `scaler` in double,
   the DoMask LUTs and `kGlobalScale` in double; CUDA used float for these. Verified: OpenCL traces byte-identical;
   CUDA distances change in the last digits on some rejected iterations, no regress output changes in any of the 7
   modes; CUDA-only, no-GPU and no-full-JPEG builds compile; clang-tidy clean.

Phase 5 notes:
- **Where things are.** `guetzli/processor.{h,cc}` is gone; the encoder is in `core/encoder/`: `Encoder` (one image:
  `encodeJpeg`/`encodeRgb`/`encodeJpegData`; owns the `ProcessStats`, takes `EncoderParams`, `Backend&`, `ILogger&`),
  `QuantMatrixSearch` + `QuantMatrixGenerator` (`quant_matrix_search`), `FrequencyMasking` +
  `CollectZeroingCandidates()` (`frequency_masking`), `OutputSelector` (serializes candidates, keeps the best by
  score), `JpegSizeEstimator` (`size_estimator`, also `EntropyCodedDataSize`/`ComputeEntropyCodes`/`EstimateDCSize`),
  `EncoderParams` (was `Params`). `narrow_cast` moved to `core/util/narrow_cast.h`. The libjpeg-turbo fallback decoder
  for JPEGs the encoder cannot take as coefficients moved to `core/io/jpeg_decode` (`decodeJpegPixels`).
  `OutputImage` stays in `guetzli/` (moving the rest of `guetzli/` into `core/jpeg`, `core/image` is left for phase 6).
- **Size estimator (todo S7).** Per histogram it keeps the code bits before the escape-byte estimate, so a changed
  symbol updates the total in constant time; the code lengths are still refreshed every 10th change (changing that
  would change the output). Exact: a unit test checks it against a full recomputation after every change.
- **OutputImage cleanup.** The requantization that a 2025 commit (`693dc7a`) added to `SetCoeffBlock()`,
  `SaveToJpegData()` and `SetDownsampledCoefficients()` is gone: every caller passes quantized blocks (now asserted;
  the debug build runs clean on a 4:4:4, a 4:2:0 and a grayscale regress input). `max_v_samp_factor` typo fixed (same result for 4:4:4 / 4:2:0).
  `exit()` replaced by an exception. `DownsampleConfig` with default member initializers.
- **Two encoder bugs fixed** (output changes, own commits, baselines updated; see the release list): the 4:2:0
  chroma pass changed the wrong blocks (todo 21, first half), and blocks without zeroing candidates could get a
  bogus one.
- Verified: the pure refactoring commits (`3747847`, `1af8289`) keep regress byte-identical in all 7 modes, and the
  old vs new `--c --verbose` traces (which print the size estimates) are identical on the regress inputs and two larger
  images. After the fixes all 7 modes pass with the new baselines (check modes still equal `--c`). CMake release,
  debug and a no-GPU/no-full-JPEG build, the Makefile build and `make unit-test` pass. New unit tests:
  `tests/unit/encoder_test.cpp`, `tests/unit/output_image_test.cpp` (68 test cases in total).
- Left for later: `GUETZLI_LOG` / `ProcessStats` counters are still used for the `--verbose` trace (§5.7); quantization
  matrices are still `int[3][64]`; `FrequencyMasking::adjust()` is still one long function (its float expressions are
  output-sensitive under `-ffast-math`, so it was moved verbatim).

Phase 4 notes:
- **Where things are.** The host GPU code moved from `clguetzli/` to `backend/` (pure renames in `7f20984`):
  `backend/gpu/` holds what OpenCL and CUDA share: `gpu_kernels.h` (the typed launcher interfaces
  `GpuButteraugliKernels`, `GpuDctKernels`, `GpuBlockZeroingKernel`, `GpuCompareSession`; `GpuKernels` = all three),
  `gpu_backend.h` (`makeGpuBackend(kind, device)`), `constant_cache.h` (`ConstantCache<Memory>`: blur weights, DoMask
  LUTs, sRGB LUT, convolution scales), `kernel_names.h` (`KernelName` + entry point names), `gpu_options.h`, and the
  comparator / DCT ops / block-zeroing search from phase 3, which now take the interfaces as `shared_ptr`.
  `backend/opencl/`: `opencl_context` (`OpenClContext`, `ClBuffer`, `ClChannels`, `ClMemory`, `CL_CHECK`),
  `opencl_kernels` (`OpenClKernels`: owns the context, implements `GpuKernels`), `kernel_args.h` (`setKernelArgs`).
  `backend/cuda/`: `cuda_driver` (`CudaDriver`), `cuda_context` (`CudaContext`, `CuBuffer`, `CuChannels`,
  `CuMemory`, `CU_CHECK`), `cuda_memory_pool`, `cuda_kernels` (`CudaKernels`). `backend/gpu_error.h`.
  `clguetzli/` keeps the kernel sources, the embedded kernel headers, `lzodec` and `clbutter_comparator` (CPU).
- **Ownership.** `GpuDevices` (`backend/factory.h`) is created by `Application::run()` and holds one
  `OpenClKernels` and one `CudaKernels` (`shared_ptr`, shared by every part of a backend made from it). Each creates
  its context on first use, or in `probe()` (the Auto probe), and keeps it until the run ends; a failed creation
  throws and is retried on the next use. `configureGpuBackends()`, `getOcl()`/`getOcu()`, `supports*()` and every
  function-local static are gone; `--version` uses `printGpuDevices()`. Buffers are RAII handles that return to the
  pool when destroyed (`zeroed()`, `uninitialized()`, `upload()` on the context; same rules as before); constant
  tables are unpooled `ClMemory`/`CuMemory` in the context's `ConstantCache`. `CudaDriver::load()` returns the entry
  points as a value (no `#define cuXxx` macros); the driver library is deliberately never unloaded.
- **Logging and errors.** The GPU layer logs through the injected `ILogger` (`LogError`/`LogInfo`/`utils.*` gone).
  `CL_CHECK`/`CU_CHECK` throw `GpuError` without logging; `Application` logs the message once. `--verbose` device
  details are `LogLevel::Debug`; the always-printed "CUDA Adapter: ..." line is `Warning` so it stays on stderr.
- **Deviations from §5.6.** The launchers are one class per API implementing three per-stage interfaces, not one
  class per stage (`OpsinLauncher`, ...). Kernel arguments still go through a variadic `setKernelArgs()` (typed,
  throws on error) or a `const void* args[]` array, not functions with named parameters. The launcher bodies are the
  old `cl*`/`cu*` functions moved verbatim (the per-backend float constants and launch shapes differ and the output
  depends on them).
- **Exit criteria.** No `allocMem`/`releaseMem` left. ASan/LeakSanitizer: there is no sanitizer runtime for MinGW on
  this machine, so it runs in CI instead: `GUETZLI_SANITIZE` (CMake) and preset `ci-linux-asan`, job `linux-asan`
  (unit tests, `--c` smoke test, injected GPU failures). Not yet run: it needs a push.
- Also fixed: building with neither GPU backend failed to compile since phase 3 (`channel_info` was only defined
  with a GPU define); verified with `-DGUETZLI_WITH_CUDA=OFF -DGUETZLI_WITH_OPENCL=OFF` (unit tests, CPU regress). A
  CUDA-only build (`-DGUETZLI_WITH_OPENCL=OFF`) also works (unit tests, regress `cpu cuda fallback-cuda`).
- New unit tests: `tests/unit/gpu_backend_test.cpp` (constant cache; `makeGpuBackend` and the GPU comparator with
  fake launchers). 50 test cases.
- Verified: regress byte-identical in all 7 modes; old vs new binary identical in stdout, stderr and exit codes for
  `--c --verbose`, `--cuda --verbose`, `--opencl --verbose`, `--auto`, `--checkcl`, `--checkcuda`, a small image with
  `--cuda`, injected failures, `--version` and an invalid `--device`, except the intended pool-statistics change below.
  Makefile build and `make unit-test` pass.
- Left for later: the OpenCL and CUDA launchers still duplicate the butteraugli pipeline (blur, diffmap, mask); they
  could share one implementation over a small device API once the per-backend constants are made explicit.
  `ocl_channels`/`ocu_channels` (unions in `clguetzli.cl.h`) are still the non-owning views.

Phase 3 notes:
- **Where things are.** The interfaces live next to their consumer, in `guetzli/` (core), so the core never includes
  `backend/` (this refines §5.3, which put them in `backend/interfaces.h`): `Comparator` + `IComparatorFactory` +
  `BlockReferenceData` (`comparator.h`), `IDctOps` + `CpuDctOps` (`dct_ops.h`), `IBlockZeroingSearch` +
  `CpuBlockZeroingSearch` + `CoeffData` (`block_zeroing.h`), `struct Backend` (`backend.h`). `backend/factory.h`:
  `resolveBackendKind()` (Auto, injectable probes), `configureGpuBackends()`, `makeBackend()`. Implementations:
  `backend/cpu`, `backend/gpu` (shared OpenCL/CUDA code, parameterized by the `cl*`/`cu*` launchers),
  `backend/opencl` and `backend/cuda` (leaf comparators holding the device compare context), `backend/check`
  (decorators). `Process()`/`ProcessJpegData()` take `Backend&`; `OutputImage` takes `IDctOps&`.
- **Butteraugli.** `ButteraugliComparatorEx` is merged into `guetzli::ButteraugliComparator`, which runs the
  backend-dependent image operations through an injected `ButteraugliOps` (CPU: the `*Opt` functions; GPU: kernels
  for images larger than 100 px in both dimensions, the reference code otherwise). The whole-image `Compare()` is per
  backend. The reference functions in `butteraugli.cc` lost their `_` prefix and the check-mode wrappers are gone.
  **`Blur()` passes `border_ratio` to `Convolution()` as `float` on purpose**: the GPU-port wrapper it used to call
  had a float parameter, and the output depends on it.
- **Check mode** = three decorators (§5.4). The reference is the CPU backend, so `--check` output equals `--c`
  (regress now asserts this for `checkcl`/`checkcuda`). Granularity is per stage, not per kernel: comparator
  distance map (tolerance 0.005, as in the old `tcl*` checks), block-zeroing orders (0.001), DCT operations (exact).
  sRGB conversions of a single 8x8 block are not checked (that is what made check mode take minutes); it now runs
  about as fast as `--c`. Expected CHK lines on the regress inputs: a few block-zeroing entries per image, and
  distance-map differences up to ~0.013 on JPEG inputs. They are identical for OpenCL and CUDA, so they come from the
  kernels vs the optimized CPU code rather than from a device; pre-existing, not investigated.
- **CMake:** `guetzli_objs` is split into `guetzli_core` (no GPU defines or include paths) and `guetzli_backend`. The
  ctest `layering` (`cmake/CheckLayering.cmake`) fails if `config/`, `logging/`, `core/`, `guetzli/` or butteraugli
  include `backend/`, `clguetzli/` or OpenCL/CUDA headers. `opsin_block` moved to `guetzli/` (it is CPU code). The
  Makefile still builds and passes regress.
- **Regress inputs:** `rgb_small_90x60.png` and `rgb_wide_140x72.png` cover the GPU modes' reference path (images not
  larger than 100 px in both dimensions); their baselines were taken with the pre-phase-3 binary.
- Verified: regress byte-identical in all 7 modes; old vs new binary identical in output files, stdout, stderr and
  exit codes for `--c --verbose`, `--cuda --verbose`, `--opencl`, `--auto` and injected fallbacks (only the reported
  OpenCL compile time differs).
- Left for later: `clguetzli/clbutter_comparator.*` (the optimized CPU butteraugli, used only by `backend/cpu`) keeps
  its historical name and place until butteraugli moves to `core/quality/` (§3). `clDiffmapOpsinDynamicsImage` /
  `cuDiffmapOpsinDynamicsImage` were already unreachable and are now unused; delete them in phase 4. The GPU
  backend objects still call the `cl*`/`cu*` free functions, which use the context singletons (phase 4).

Phase 2 notes:
- New tree: `app/` (cli, application = composition root, main), `config/`, `logging/`, `core/io/`. `guetzli/guetzli.cc`
  is gone. CLI11 is vendored in `third_party/CLI11/`. Console output, streams and exit codes were verified identical
  to the old binary for the common paths; keep it that way (scripts parse it).
- Deliberately left for later phases: `GpuOptions` (device, verbose) is stored inside the OpenCL/CUDA context singletons
  via `configureOpenCl()`/`configureCuda()` because initialization stays lazy (a small image with `--opencl` never
  touches the GPU); phase 4 makes contexts owned objects and the options a constructor argument. The GPU layer still
  logs through `LogError`/`LogInfo`; it gets the injected `ILogger` in phase 4 as well. (`g_mathMode` went in phase 3.)
- `Config` covers only what the CLI exposes (no `try_420`/`clear_metadata`/`CpuReference` yet; add them with a flag).
- After phase 2 (`3538db6`): TIFF alpha compositing fixed and distinct exit codes introduced (see the list below).

**Release: the refactor ships as the next major version (v3.0.0, decided 2026-11-15).** Correctness fixes are wanted
even when they change output or CLI behaviour; keep each one in its own commit with tests and add it here for the
release notes:
- TIFF with alpha: alpha was read from the wrong byte and blended twice; now composited correctly on black/white
  (also gray + alpha TIFFs). Opaque TIFFs unchanged. (`3538db6`)
- Exit codes: 0 success, 1 usage/I/O error, 2 input is not PNG/TIFF/JPEG, 3 input could not be processed (corrupt
  data, memory limit, encoder failure). Before, every failure printed "Unknown file format" and exited 2. (`3538db6`)
- Messages: "succeeded" spelling; memory-limit message names the limit and `--memlimit`/`--nomemlimit`; a damaged
  JPEG is reported as such. (`3538db6`)
- CLI: CLI11 help/usage text; new `--backend NAME`, `--check`, `--help`; legacy flags kept; `--memlimit` below -1 is
  rejected; non-numeric values are rejected instead of being read as 0 (`atoi`). (`661a6ad`)
- `--verbose` also prints the OpenCL device report (stderr). (`661a6ad`)
- Build: CMake + Ninja (`cmake --preset release`), binaries in `build/<preset>/bin`; C++20. (`7e1a9a4`, C++20 commit)
- `--check` / `--checkcl` / `--checkcuda`: compares whole stages (distance map, block-zeroing orders, DCT operations)
  instead of single kernels; the output is now exactly the `--c` output, and the mode is about as fast as `--c`
  instead of taking minutes. `CHK` lines have a new format (`CHK <stage> <part> <count>:<size> (max diff <d>)`). (phase 3)
- Asking for a GPU backend that is not compiled into the binary now prints the CPU-fallback message instead of
  silently running CPU code. (phase 3)
- Builds without OpenCL and CUDA use the same (optimized) comparator as `--c` in a GPU build; before, they used the
  reference comparator and gave different output. Release binaries are not affected. (phase 3)
- The CUDA memory-pool statistics line ("mem_pool has N blocks, ...") is printed only with `--verbose`; before, every
  CUDA run printed it to stderr at exit. (phase 4)
- A failing GPU call no longer prints a separate "Error: <function>:<line> returned <error>." line before the
  "GPU processing failed (...); falling back to CPU" message, which already contains the error. (phase 4)
- An OpenCL kernel argument rejected by the driver now fails the GPU run (CPU fallback); before, it was logged and the
  kernel was launched anyway. (phase 4)
- 4:2:0 output: the frequency-masking chroma pass applied each change meant for chroma block (x, y) to block
  (x/2, y/2) (bug since 2025, `50a6558`). Fixed; 4:2:0 outputs get smaller (regress: -2 and -50 bytes; two larger
  photos -1.8% and -0.4%). (`a5f1923`)
- Frequency masking: trailing blocks without zeroing candidates got a bogus one (the last candidate of an earlier
  block), so a coefficient could be zeroed that was never scored for that block (bug since 2017). Fixed; outputs change
  by a few bytes either way. (`925d84d`)
- JPEG inputs decoded through libjpeg-turbo (unsupported subsampling, CMYK, ...): a decoding error is reported and the
  run fails with exit code 3; before, the process exited with code 1 (or libjpeg terminated it). (phase 5)
- Windows binary: built with MSYS2 UCRT64 and linked against the Universal C Runtime instead of msvcrt.dll (MINGW64 is
  deprecated). Windows 10/11 include the UCRT; Windows 7 SP1 / 8.1 need KB2999226 (README, "Windows runtime"). Output
  unchanged (regress, 7 modes, and CPU traces identical to the msvcrt build). (phase 6)
- The process no longer calls std::terminate for an exception escaping main; the message and exit code are the same.
  (phase 6)
- CUDA backend: the host-side butteraugli constants (blur weights, DoMask LUTs, global scale) are computed like the
  OpenCL backend (in double); CUDA distances can differ in the last digits from v2. No output change on the
  regression inputs. (phase 6)
- CUDA: the binary embeds a fatbin (native code for sm_61/75/86/89 + compute_52 PTX) instead of compute_75 PTX:
  Pascal and Maxwell GPUs can use the CUDA backend (before: silent fallback to OpenCL), and Pascal..Ada GPUs no
  longer JIT-compile on the first run (~7 s on an RTX 3060, ~30 s reported in issue #15). A PTX that does not load
  is reported with the driver version, device and JIT log. The exe grows by ~1.6 MB. Output unchanged (native
  and forced-PTX-JIT paths). (D1, D3, D4)
- Linux CPU path: math calls on floats use the float overloads (std::exp etc.), like Windows; the Linux CPU output may
  differ slightly from v2 and now matches Windows. (phase 6)

Phase 1 notes:
- CMake (`CMakeLists.txt`, `CMakePresets.json`, `cmake/`) builds `guetzli` and `guetzli_tests` from one OBJECT library
  (`guetzli_objs`). Parity with the Makefile was verified on MinGW GCC 15.2: identical disassembly for all 34
  objects, same DLL imports, same regress md5s. CI builds/tests with the presets only.
- Host code is C++20 now (Makefile too). Codegen changed in ~20 objects, output did not (regress + 4 larger CPU encodes).
- Not done in phase 1, still open for later phases (the GPU/core library split was done in phase 3):
  `embed_kernel.cmake` with a new header format (§6.2; today's targets reuse minilzoc + `format_header.py`),
  `GUETZLI_WARNINGS_AS_ERRORS` in CI (the tree has warnings), and invoking the superbuild from the top-level configure.
- The Linux binary now links static zlib from the superbuild (the Makefile picked up `libz.so` there). CI on
  Linux/macOS runs the CMake build for the first time on the next push; that has not been observed yet.

Phase 0 notes for whoever continues:
- `tests/regress` has 8 inputs (odd sizes, odd 4:2:0, grayscale JPEG/PNG added). No ≥ 1 MP input yet (GPU is shared).
- The kernel C++ emulation is gone. Host code needed two kernel helpers; they are ported in `guetzli/opsin_block.cc`
  (`clguetzli/opsin_block.cpp` until phase 3).
  **Its coefficient tables are deliberately non-const:** with `-ffast-math`, `const` tables let GCC fold and reassociate
  `Gamma()`, which changed 90% of the per-block opsin values in the last bits (and so GPU-mode output). The port is
  bit-identical to the old object on ~147k blocks. Keep this in mind for any CMake flag or refactor touching it.
- `make unit-test` (doctest, `tests/unit/`) runs in CI. `.clang-format` (clang-format **21.1.5**, pinned in CI;
  `ReflowComments: Never`, `SortIncludes: Never`) is enforced; `.clang-tidy` is checked in but not enforced yet.
- `.gitattributes` keeps `*.cl`/`*.cu` LF: the OpenCL header used to be generated from a CRLF checkout and failed
  `check-kernel-headers` on Linux/macOS CI.
- CI now also runs on pushes/PRs for `refactor/modern-cpp`. CI does not run `tests/regress` (CPU md5s may differ
  between GCC/Clang with `-ffast-math`; per-platform baselines would be needed).
- Known pre-existing build failures, not fixed: `FEATURES="CUDA FULL_JPEG"` (`getOcl` unguarded in `clguetzli.cpp`) and
  `FEATURES="FULL_JPEG"` (`ushort` in `guetzli.cc` without CL headers). See `todo.md`. (Phase 4: both configurations
  build with CMake, `-DGUETZLI_WITH_OPENCL=OFF` and both GPU options OFF; verified with unit tests and regress.)

- **Branch.** `refactor/modern-cpp` was created from `fix/gpu-correctness-and-sync` (HEAD `7331f9e`), not from
  `master`: that branch holds the GPU fixes, speedups and the kernel-header policy, and has not been merged
  to `master` yet. Merge `master` into the refactor branch later as §10 says.
- **Uncommitted files on that branch** (kept out of commits so far, by the maintainer's choice): `BUILD.md` (modified;
  documents the kernel-header policy), and untracked `CLAUDE.md`, `docs/` (this file), `todo.md`, `todo-profiling.patch`.
  Ask before committing any `.md` file.
- **Gate for every step:** `tests/regress/regress.sh build/release/bin/guetzli.exe cpu opencl cuda checkcl checkcuda
  fallback-opencl fallback-cuda` must report `failures: 0` (byte-identical output; takes about a minute). The GPU is shared and busy: test GPU modes only on the tiny
  `tests/regress` inputs, never on large `tests/input` files.
- **Kernel headers** (`clguetzli_cu_ptx.h`, `clguetzli_cl_src.h`) are built locally and committed; CI never rebuilds them
  (§7). If phase 0 touches `clguetzli.cl`/`.cl.h`/`.cu`, regenerate with `make cuda-headers` (CUDA 11.8 only).
  On this Windows machine: add MSVC 14.29 `bin/Hostx64/x64` to `PATH`, then
  `make cuda-headers NVCC_EXTRA_FLAGS=-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`.
  The PTX must stay byte-identical unless the kernel source really changed. Removing the `__checkcl` macros from
  `clguetzli.cl.h` should not change it.
- **Build:** `make CONFIG=release` gives `bin/release/guetzli.exe` (the Makefile stays until phase 1 reaches parity).
- **Deferred, not part of the refactor:** GPU migration G1–G4 and driver items D1/D3/D4 (`todo.md`). They need
  testing on huge images and are planned against the new interfaces after phase 3.
- **CLI11 reference copy:** `d:\Work\OpenSource\Leanify\lib\CLI\CLI11.hpp` (v2.6.1), to vendor into `third_party/CLI11/`.

---

## 1. Why

The project has many authors and three layers of history (Google's Guetzli, Tencent's GPU port, this fork).
The result works, but:

- **Behaviour is driven by globals.** `g_mathMode` is read in 61 places across 7 files; `g_deviceIndex` and
  `verbose_cl` are `extern`s; `getOcl()` / `getOcu()` are hidden singletons. Any module can change any other
  module's behaviour, and nothing can be tested in isolation.
- **Backend selection is scattered.** `processor.cc`, `output_image.cc` and the comparator all contain
  `if (MODE_OPENCL == g_mathMode) ... else if (MODE_CUDA == ...)`. Adding a backend means editing the encoder.
- **Check modes** (`--checkcl`, `--checkcuda`) are more `if` branches threaded through the same code.
- **`Processor` does everything**: quant-matrix search, frequency masking, entropy estimation, output selection.
- **`main()`** is a 763-line file: hand-written `strcmp` CLI loop, file I/O, PNG decoding, memory limits, CPU fallback.
- **GPU resources are managed by hand** (`allocMem` / `releaseMem` pairs, manual teardown order).
- **No unit tests.** Only an md5 regression harness (`tests/regress/`).

## 2. Goals and non-goals

Goals
- Modern C++20, SOLID, no mutable global state. Every module gets its configuration and collaborators through
  its constructor.
- One place parses the command line (CLI11) into a validated `Config`; that object is passed down.
- Backends (CPU, OpenCL, CUDA) are interchangeable implementations of narrow interfaces.
- Unit-testable core (doctest), plus the existing byte-exact regression harness.
- CMake build; GitHub Actions keeps producing the release binaries for Windows, Linux and macOS.

Non-goals
- **No change to generated JPEG output.** Every phase must keep `tests/regress` byte-identical for all modes.
  Output-changing improvements are separate, explicitly discussed PRs (per `CONTRIBUTING.md`).
- No upstream compatibility with google/guetzli (dropped; upstream has been inactive for years).
- No CPU multithreading. Performance work goes to the GPU (see `todo.md`, items G1–G4).

## 3. Decisions

| Topic | Decision |
|---|---|
| Language | **C++20** for all host code. |
| Kernels | Kernel source stays C-like (OpenCL C / CUDA C subset); see §7 for driver compatibility. |
| Build | **CMake** (≥ 3.24) with presets. The GNU Makefile is removed at the end of the migration. |
| CLI | **CLI11** (single header, vendored in `third_party/CLI11/`). Existing flags kept as aliases. |
| Tests | **doctest** (single header, vendored in `third_party/doctest/`). |
| Upstream | Compatibility with google/guetzli dropped. |
| Delivery | One long-lived branch `refactor/modern-cpp`; `master` keeps shipping. |
| Release builds | Stay on GitHub Actions (`.github/workflows/ci.yml`); same three release archives. |
| Butteraugli (`third_party/butteraugli`) | **Move into `core/quality/` and refactor like the rest.** It is not a pristine vendored copy: it was renamed/edited for the GPU port and includes `clguetzli` headers, and the mode dispatch lives inside it, so it cannot meet the "core has no GPU includes, no globals" rule without being ours. The CPU reference and optimized CPU variants become ordinary classes behind `IComparator`. |
| C++ emulation of kernels (`clguetzli.cl.cpp`) | **Drop it** (decided 2026-11-15). It was written to debug a kernel that misbehaved on AMD only; after that fix it was never needed again. Today nothing executes it: `set_global_id()` is never called and no host code calls a C++-compiled kernel (verified 2026-11-15). Check mode compares the GPU against the real CPU implementation (decorator, §5.4). Dropping it removes the `__checkcl` path, the `get_global_id` stubs and the `__kernel`/`__global` keyword macros that leak through `clguetzli.cl.h` into common headers (they have broken macOS system headers before). Kernels become one source compiled two ways (OpenCL, CUDA). The `ButteraugliComparatorEx` code that lives in the same file is kept and moved (§12). |

## 4. Constraints we must keep

- **Byte-identical output** per backend (CPU vs GPU may differ from each other, as today; each must match its own baseline).
- **Static, self-contained Windows binary** (MinGW, `-static`). New dependencies must link statically.
- **CUDA is loaded at runtime** (`nvcuda.dll` / `libcuda.so` through function pointers). The binary must start on machines
  without NVIDIA drivers. Never link the CUDA runtime.
- **Kernels are embedded** (LZO-compressed) in the binary; no files next to the executable.
- **CI has no GPU.** CI verifies CPU output and the injected-failure fallback; GPU changes are verified locally
  with `tests/regress` (and on large images before performance claims).
- **GPU failure falls back to CPU** (`GpuError` → rerun on CPU). Kept, but owned by the composition root, not `main()`'s catch block.

## 5. Target architecture

### 5.1 Layout

```
app/
  main.cpp                 CLI11 → Config → composition root → Application::run()
  cli.{h,cpp}              CLI definition, legacy-flag aliases, --version / device listing
config/
  config.h                 Config value types (no logic besides validation)
core/                      pure C++, no GPU headers
  jpeg/                    JpegData, reader, writer, Huffman, bit writer
  image/                   OutputImage, OutputImageComponent, color transforms, DCT/IDCT
  encoder/
    encoder.{h,cpp}        orchestrates one image (today: ProcessJpegData)
    quant_matrix_search    today: SelectQuantMatrix / QuantMatrixGenerator / TryQuantMatrix
    frequency_masking      today: SelectFrequencyMasking / SelectFrequencyBackEnd
    size_estimator         today: histograms + EntropyCodedDataSize (incremental, see todo S7)
    output_selector        today: MaybeOutput / GuetzliOutput
  quality/                 Butteraugli, IComparator
  io/                      PNG / JPEG decode, file read/write
backend/
  interfaces.h             IComparator, IBlockZeroingSearch, IDctOps, (later) IImageStore
  backend.h                struct Backend { unique_ptr<...> per interface }; BackendKind
  factory.{h,cpp}          makeBackend(const BackendOptions&, ILogger&) → Backend (auto-detect, device, fallback)
  cpu/                     CPU implementations (wrap core)
  opencl/                  OpenClContext, Buffer, KernelSet, per-stage launchers
  cuda/                    CudaContext, DeviceBuffer, Module, per-stage launchers, dynload
  check/                   CheckingComparator, CheckingBlockZeroing (decorators)
kernels/
  guetzli.cl               the single kernel source (today clguetzli.cl)
  guetzli.cu               CUDA wrapper that #includes it
  cmake/embed_kernel.cmake generates the compressed headers at build time (§6.2)
logging/
  logger.h                 ILogger (levels, sinks); StderrLogger; NullLogger
third_party/               CLI11, doctest, minilzo, OpenCL headers + wrapper, libpng (as today)
tests/
  unit/                    doctest suites, one per module
  regress/                 existing byte-exact harness (+ more inputs, §8)
```

Dependency rule: `app → backend → core`, `app → config`, everything may use `logging`. **`core` never includes a GPU
header and never reads a global.** CI enforces this with a simple include check (§9).

### 5.2 Configuration and CLI

```cpp
// config/config.h
namespace guetzli {

enum class BackendKind { Auto, Cpu, CpuReference, OpenCl, Cuda };

struct BackendOptions {
  BackendKind kind = BackendKind::Auto;
  std::optional<int> device;            // index within the chosen backend
  bool check_against_cpu = false;       // --checkcl / --checkcuda
  bool fallback_to_cpu = true;          // on GpuError
};

struct EncoderOptions {
  int quality = 95;                     // converted to butteraugli target via quality.cc
  bool clear_metadata = true;
  bool try_420 = false, force_420 = false;
  bool blend_on_white = false;
};

struct IoOptions {
  std::filesystem::path input, output;
  std::optional<std::size_t> memlimit_mb;   // nullopt = --nomemlimit
};

struct LogOptions { LogLevel level = LogLevel::Info; };

struct Config {
  EncoderOptions encoder;
  BackendOptions backend;
  IoOptions io;
  LogOptions log;
  [[nodiscard]] std::vector<std::string> validate() const;   // empty = OK
};

}  // namespace guetzli
```

`app/cli.cpp` builds the CLI11 app and fills a `Config`. Existing flags remain as aliases so scripts keep working:

| Today | New canonical | Alias kept |
|---|---|---|
| `--c` | `--backend cpu` | yes |
| `--opencl` / `--cuda` / `--auto` | `--backend opencl` / `cuda` / `auto` | yes |
| `--checkcl` / `--checkcuda` | `--backend opencl --check` / `--backend cuda --check` | yes |
| `--device N` | `--device N` | — |
| `--quality Q`, `--memlimit M`, `--nomemlimit`, `--blend-on-white`, `--verbose`, `--version` | same | — |

CLI11's config-file support (`--config file.toml`) comes for free and maps 1:1 onto `Config`.
`--version` lists devices by asking each backend's `enumerateDevices()`; no backend initialization side effects.

### 5.3 Backend interfaces

Narrow interfaces, one per capability the encoder needs (interface segregation). Sketch:

```cpp
// backend/interfaces.h
namespace guetzli {

class IComparator {                         // replaces ButteraugliComparatorEx's GPU branches
 public:
  virtual ~IComparator() = default;
  virtual void compare(const OutputImage& img) = 0;
  [[nodiscard]] virtual double distance() const = 0;
  [[nodiscard]] virtual std::span<const float> distmap() const = 0;
  virtual void startBlockComparisons() = 0;
  virtual void finishBlockComparisons() = 0;
  // ... remaining Comparator API (ScoreOutputSize, BlockErrorLimit, ...)
};

class IBlockZeroingSearch {                 // today: cl/cuComputeBlockZeroingOrder + CPU loop
 public:
  virtual ~IBlockZeroingSearch() = default;
  virtual std::vector<CoeffData> run(const BlockZeroingInput& in) = 0;
};

class IDctOps {                             // today: cl/cuCopyFromJpegComponent, ApplyGlobalQuantization
 public:
  virtual ~IDctOps() = default;
  virtual void copyFromJpeg(OutputImageComponent& dst, const JPEGComponent& src, const QuantTable& q) = 0;
  virtual void applyGlobalQuantization(OutputImageComponent& c, const QuantTable& q) = 0;
};

struct Backend {
  BackendKind kind;
  std::unique_ptr<IComparatorFactory> comparators;   // per-image comparators
  std::unique_ptr<IBlockZeroingSearch> blockZeroing;
  std::unique_ptr<IDctOps> dct;
};

}  // namespace guetzli
```

The encoder receives `Backend&` (dependency inversion) and contains **no** mode checks. `OutputImage` stops calling GPU
functions itself; the operations that today branch inside it move into `IDctOps`.

Later (todo G1), `IImageStore` lets the image live on the device: the CPU implementation is today's `OutputImage`
storage; the GPU implementation keeps coefficients and pixels resident and hands the comparator device buffers.

### 5.4 Check mode = decorator

`--check` wraps the GPU implementation and the CPU implementation in a decorator with the same interface:

```cpp
class CheckingComparator final : public IComparator {
 public:
  CheckingComparator(std::unique_ptr<IComparator> tested, std::unique_ptr<IComparator> reference, ILogger&);
  void compare(const OutputImage& img) override;   // runs both, logs mismatches, returns the reference result
  ...
};
```

This replaces every `MODE_CHECKCL` / `MODE_CHECKCUDA` branch and the `tcl*` helpers in `clguetzli_test.cpp`.

### 5.5 Backend selection and fallback

`makeBackend(const BackendOptions&, ILogger&)` is the only place that knows about concrete backends:

- `Auto`: try CUDA, then OpenCL, then CPU, as today. Each probe fully initializes the device (as `supportsCuda()` /
  `supportsOpenCl()` do now) and returns a ready context; no second initialization.
- `--device N` is interpreted by the chosen backend; device listing labels indices per backend to end the current
  CUDA-vs-OpenCL index ambiguity.
- On `GpuError` during encoding, `Application` asks the factory for a CPU backend and reruns the image if
  `fallback_to_cpu` is set. `GUETZLI_TEST_GPU_FAIL` stays as the test hook.

### 5.6 GPU resource management

- `OpenClContext` / `CudaContext` own the device, queue/stream, program/module, kernel handles and the buffer pool.
  They are created by the factory and owned by the backend: **no singletons, no function-local statics**.
- `Buffer<T>` / `DeviceBuffer<T>`: move-only RAII handles that return memory to the pool in the destructor.
  `allocMem` / `releaseMem` pairs disappear. Zero-fill stays explicit (`Buffer::zeroed(n)` vs `Buffer::uninitialized(n)`),
  following the rules already in `CLAUDE.md`.
- Constant tables (blur weights, convolution scales, DoMask and sRGB LUTs) live in a `ConstantCache` owned by the context.
- Kernel launches go through typed launchers per pipeline stage (`OpsinLauncher`, `DiffmapLauncher`, ...), each owning
  its kernel handles and argument packing; the 30-argument `clSetKernelArgEx` calls become functions with named parameters.
- The CUDA driver function-pointer table (`cuda_dynload`) becomes a `CudaDriver` object held by the context, not
  `#define cuMemAlloc dynCuMemAlloc` macros.

### 5.7 Logging and errors

- `ILogger` is injected; `LogError`, `LogInfo`, `GUETZLI_LOG(stats, ...)` and `verbose_cl` go away. `ProcessStats`
  keeps only counters.
- Errors: exceptions for exceptional conditions (`GpuError`, `IoError`, `JpegError`), `std::expected`-style results
  (or `[[nodiscard]] bool` + message) for validation. No `exit()` / `abort()` outside `main`.

## 6. Build (CMake)

### 6.1 Targets and options

```
guetzli_core      STATIC   core/, logging/          (no GPU)
guetzli_backend   STATIC   backend/ (+ cpu/, opencl/, cuda/ as enabled)
guetzli           EXE      app/
guetzli_tests     EXE      tests/unit/ (doctest)
```

Options: `GUETZLI_WITH_OPENCL` (ON), `GUETZLI_WITH_CUDA` (ON), `GUETZLI_WITH_FULL_JPEG` (ON),
`GUETZLI_STATIC` (ON on Windows), `GUETZLI_WARNINGS_AS_ERRORS` (ON in CI). `CMakePresets.json` provides
`release`, `debug`, `ci-linux`, `ci-windows-mingw`, `ci-macos`.

Third-party libraries keep coming from the existing superbuild (`external/CMakeLists.txt`); the main project finds them
via `CMAKE_PREFIX_PATH=external/install`. Later, the superbuild can be invoked automatically from the top-level
configure step if the prefix is missing.

### 6.2 Kernels

- `kernels/tools/embed_kernel.cpp` (driven by `cmake/KernelHeaders.cmake`) replaced `minilzoc` + `format_header.py` +
  the Makefile rules in phase 6: a small host tool (built as part of the project) LZO-compresses the input and writes a
  header with the **uncompressed size** included.
- Both kernel headers are committed and consumed as-is. Regeneration is an explicit developer target
  (`cmake --build <dir> --target kernel-headers`), never part of the normal or CI build; the PTX recipe checks for the
  pinned `nvcc` (§7). CI verifies the OpenCL header against the source and that the build left both headers untouched.

## 7. CUDA toolchain and driver compatibility

Host code moves to C++20, but **`nvcc` only ever compiles `kernels/guetzli.cu` to PTX**: the CUDA host side is ordinary
C++ calling the driver API through function pointers. So C++20 does not require a newer `nvcc`.

What matters for users is the driver. The driver JIT-compiles the embedded PTX and has two hard limits:

1. **PTX ISA version.** A driver only accepts ISA versions up to the CUDA version it ships with. A newer toolkit
   emits a newer ISA. This is exactly [issue #15](https://github.com/doterax/guetzli-cuda-opencl/issues/15): a PTX
   built with CUDA 13.0 (`.version 9.0`) was rejected by driver 576.28 / CUDA 12.9 (`ptxas fatal: Unsupported .version
   9.0; current version is '8.8'`). Every kernel lookup then failed with `CUDA_ERROR_INVALID_HANDLE` and the run fell
   back to OpenCL; CUDA "worked in every other app", so users see it as guetzli dropping CUDA support.
2. **Target architecture.** PTX compiled for `compute_XX` can only be JIT-compiled for GPUs of architecture XX or newer.

Current state (2026-11-15): the committed PTX header (now `kernels/guetzli_cu_ptx.h`) is ISA 7.8 (CUDA 11.8) → OK for drivers ≥ R520, but
**`.target sm_75`** → Pascal (GTX 10xx, sm_61) and Maxwell GPUs cannot use the CUDA backend at all and silently fall
back to OpenCL. The same issue also reports **~30 s JIT compilation on first run** (cached afterwards).

Policy:
- **Pinned kernel toolkit:** the committed PTX is generated with **CUDA 11.8** (ISA 7.8, drivers ≥ R520). Raise it only
  deliberately and document the minimum driver in `README.md`. Never with whatever toolkit happens to be installed.
- **Low PTX target:** generate for `compute_52` (Maxwell+). 11.8 still supports it; 12.x deprecates < 5.0, 13.0
  removes Maxwell/Pascal/Volta offline targets, which is another reason not to build the PTX with 13.x.
  Lowering the target must be verified byte-exact against the current CUDA baselines on a local GPU before it ships.
- **Embedded cubins (SASS)** for common architectures (sm_61, sm_75, sm_86, sm_89, and newer as needed), packed as a
  fatbinary next to the PTX. The driver loads matching SASS directly (no JIT, no 30 s first run, no ISA check) and uses
  the PTX only for GPUs without a matching cubin. Cubins built with toolkit 12.x run on any 12.x driver; build them with
  the oldest toolkit that supports each target. Each cubin must give the same output as the PTX path (regression check per arch where hardware is available).
- **Diagnostics:** when module loading fails, report the JIT log (driver version, PTX ISA, target) in the
  `--auto` output, instead of a bare "CUDA is not supported". Issue #15 needed a special build to see the cause.
- The kernel source stays within the subset that both the pinned toolkit and OpenCL C accept (no C++20 in kernels).
- **Kernels are built locally and committed with the code; CI never rebuilds them** (decided 2026-11-15, implemented on
  `master` before the refactor). The `kernel-headers` CMake target (`make cuda-headers` before phase 6) is the only way
  to regenerate the headers and refuses any `nvcc` other than the pinned release; normal builds do not depend on it.
  CI only installs the toolkit for `cuda.h`, then runs the `check-kernel-headers` target (the OpenCL header must
  match `kernels/guetzli.cl`) and `git diff --exit-code` on both headers (the build must not have touched them).

## 8. Testing

- **doctest unit tests** (`tests/unit/`): JPEG read/write round-trips, Huffman, DCT/IDCT vs reference, quantization,
  size estimator (incremental vs full recompute: must be exact), CLI parsing to `Config` (including the legacy aliases),
  backend factory selection logic with fake probes, decorator behaviour with fake comparators.
- **Regression harness** (`tests/regress/`), the gate for every phase: add inputs for odd widths/heights, 4:2:0 with
  widths not a multiple of 16, grayscale JPEG, RGBA with partial alpha, and one ≥ 1 MP image run locally on GPU.
  Baselines per mode stay in `tests/regress/expected/`.
- **GPU tests** run locally only (CI has no GPU); PR descriptions must state which GPU modes were run.
- CI runs: unit tests on all three OSes, CPU regression, injected-failure fallback, PTX-header freshness check.

## 9. Code standards

- C++20. Prefer: `std::span`, `std::unique_ptr`, value types, `enum class`, `constexpr` tables, `[[nodiscard]]`,
  designated initializers for option structs, concepts where they replace SFINAE. Avoid `std::format` until the macOS
  CI toolchain supports it fully.
- No mutable globals, no singletons, no function-local statics holding state. Constants are `constexpr`.
- Ownership is explicit: raw pointers and references never own.
- Names: `PascalCase` types, `camelCase` functions, `snake_case_` members (one convention for the whole tree; the
  mechanical rename is its own commit).
- `.clang-format` (formatting) and `.clang-tidy` (`modernize-*`, `cppcoreguidelines-*` subset, `readability-*`,
  `bugprone-*`) checked in CI. Warnings are errors in CI.
- Layering check in CI: `core/` must not include anything from `backend/`, OpenCL or CUDA headers.

## 10. Migration plan

All work lands on `refactor/modern-cpp`. `master` keeps receiving fixes; they are merged into the branch regularly
(at least after every phase) to keep the gap small. Every phase must end with: build green on all CI targets,
`tests/regress` byte-identical for every mode run locally (CPU, OpenCL, CUDA, fallbacks), unit tests green.

| Phase | Content | Exit criteria |
|---|---|---|
| 0. Safety net | Extra regression inputs and baselines; doctest + first unit tests; `.clang-format`/`.clang-tidy`; one mechanical formatting commit; drop the kernel C++ emulation (§3): move `ButteraugliComparatorEx` out of `clguetzli.cl.cpp`, delete the `__checkcl` path and keyword macros from `clguetzli.cl.h` | Baselines committed; CI runs unit tests; no `__checkcl` in the tree |
| 1. CMake | CMake build reproducing today's binaries (static Windows, dynamic CUDA, embedded kernels); CI switched to CMake; Makefile kept until parity | Same md5s as the Makefile build on all regression inputs |
| 2. Config + CLI11 | `Config`, CLI11, legacy aliases; `Params`/`g_deviceIndex`/`verbose_cl`/memlimit globals removed; logger injected | No `extern` config globals; CLI unit tests |
| 3. Backend interfaces (done) | `IComparator`, `IBlockZeroingSearch`, `IDctOps`; factory; check-mode decorators; `g_mathMode` removed | `grep g_mathMode` empty; `core/` has no GPU includes |
| 4. GPU RAII (done) | Contexts, buffers, constant cache, typed launchers, `CudaDriver` object; singletons removed | No manual `releaseMem`; ASan/leak check clean on CPU path |
| 5. Encoder split (done) | `Processor` → encoder classes; `OutputImage` cleanup; incremental size estimator (todo S7) | Unit tests per class |
| 6. C++20 sweep + cleanup | Remaining modernization, naming, dead code (`clguetzli_test.cpp`), Makefile removed | clang-tidy clean |
| 7. Merge | Branch merged to `master`, version bump, release notes | Release artifacts from CI identical in behaviour |

After phase 3 the GPU performance items (todo G1–G4) can be developed against the new interfaces.

## 11. Risks

- **Silent output changes** during mechanical refactoring → byte-exact regression gate per mode per phase; never
  "fix" a numeric difference without a separate, discussed PR.
- **Long-lived branch drift** → merge `master` in after every phase; keep phases small enough to review.
- **Contributor churn** (large diffs, renamed files) → this document, phase-by-phase PRs into the branch, and a
  file-mapping table in each phase's PR description.
- **Static linking and toolchain differences under CMake** (MinGW static libstdc++/winpthread, macOS framework OpenCL)
  → phase 1 must reproduce the current binaries' dependency lists (`objdump -p`, `otool -L`, `ldd` as in CI).
- **macOS toolchain C++20 gaps** → stay within the features supported by the CI's Apple Clang.

## 12. Where today's code goes

| Today | Target |
|---|---|
| `guetzli/guetzli.cc` | `app/main.cpp`, `app/cli.cpp`, `core/io/` |
| `guetzli/processor.{h,cc}` | `core/encoder/*` |
| `guetzli/output_image.{h,cc}` | `core/image/` (GPU branches → `backend/*/dct_ops`) (done, phase 6) |
| `guetzli/jpeg_*`, `entropy_encode`, `fdct`, `idct`, `dct_double` | `core/jpeg/`, `core/image/` (done, phase 6) |
| `guetzli/butteraugli_comparator.*`, `third_party/butteraugli` | `core/quality/` (§3) (done, phase 6) |
| `clguetzli/clguetzli.h` (`MATH_MODE`, globals) | removed → `config/`, `backend/factory` |
| `clguetzli/ocl.*`, `clguetzli.cpp` | `backend/opencl/` |
| `clguetzli/ocu.*`, `cuguetzli.cpp`, `cumem_pool.*`, `cuda_dynload.*` | `backend/cuda/` |
| `clguetzli/clbutter_comparator.*`, `clguetzli.cl.cpp` (comparator part) | `backend/cpu/optimized_butteraugli.*` (done, phase 6), `backend/{cpu,opencl,cuda}` comparators (the kernel-emulation part is deleted, §3) |
| `clguetzli/clguetzli_test.*` | replaced by `backend/check/` decorators (deleted in phase 3) |
| `clguetzli/clguetzli.cl`, `.cu` | `kernels/guetzli.cl`, `kernels/guetzli.cu` (done, phase 6) |
| `clguetzli/clguetzli.cl.h` (host-side kernel types) | `backend/gpu/kernel_types.h` (done, phase 6) |
| `clguetzli/lzodec.*`, `minilzoc/`, `format_header.py` | `kernels/embedded_kernels.*`, `kernels/tools/embed_kernel.cpp` (done, phase 6) |
| `clguetzli/gpu_error.h` | `backend/gpu_error.h` |
| `Makefile` | `CMakeLists.txt` + `CMakePresets.json` (removed in phase 6) (done) |
