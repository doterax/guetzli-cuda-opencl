/*
 * Dynamic loading of CUDA driver API (nvcuda.dll / libcuda.so)
 *
 * Resolves all needed CUDA Driver API functions at runtime so the
 * binary does not have a hard link-time dependency on nvcuda.dll.
 */

#include "cuda_dynload.h"

#ifdef __USE_CUDA__

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#include <cstdio>

/* --- Function pointer definitions (initially NULL) --- */

PFN_cuInit                 dynCuInit              = nullptr;
PFN_cuCtxCreate            dynCuCtxCreate         = nullptr;
PFN_cuCtxDestroy           dynCuCtxDestroy        = nullptr;
PFN_cuCtxSetCacheConfig    dynCuCtxSetCacheConfig = nullptr;
PFN_cuDeviceGetAttribute   dynCuDeviceGetAttribute= nullptr;
PFN_cuDeviceGetCount       dynCuDeviceGetCount    = nullptr;
PFN_cuDeviceGet            dynCuDeviceGet         = nullptr;
PFN_cuDeviceGetName        dynCuDeviceGetName     = nullptr;
PFN_cuMemAlloc             dynCuMemAlloc          = nullptr;
PFN_cuMemFree              dynCuMemFree           = nullptr;
PFN_cuMemcpyDtoH          dynCuMemcpyDtoH        = nullptr;
PFN_cuMemcpyDtoHAsync     dynCuMemcpyDtoHAsync   = nullptr;
PFN_cuMemcpyHtoDAsync     dynCuMemcpyHtoDAsync   = nullptr;
PFN_cuMemcpyDtoD          dynCuMemcpyDtoD        = nullptr;
PFN_cuMemcpyDtoDAsync     dynCuMemcpyDtoDAsync   = nullptr;
PFN_cuMemsetD8Async       dynCuMemsetD8Async     = nullptr;
PFN_cuModuleLoadDataEx    dynCuModuleLoadDataEx  = nullptr;
PFN_cuModuleUnload        dynCuModuleUnload      = nullptr;
PFN_cuModuleGetFunction   dynCuModuleGetFunction = nullptr;
PFN_cuStreamCreate        dynCuStreamCreate      = nullptr;
PFN_cuStreamSynchronize   dynCuStreamSynchronize = nullptr;
PFN_cuLaunchKernel        dynCuLaunchKernel      = nullptr;

/* --- Platform-specific helpers --- */

#ifdef _WIN32
static HMODULE g_cudaLib = nullptr;

static void* loadSym(const char* name)
{
    return (void*)GetProcAddress(g_cudaLib, name);
}
#else
static void* g_cudaLib = nullptr;

static void* loadSym(const char* name)
{
    return dlsym(g_cudaLib, name);
}
#endif

static bool g_cudaLoaded = false;
static bool g_cudaInitDone = false;

/*
 * Some CUDA Driver API functions are versioned with a _v2 suffix in the
 * actual shared library (e.g. cuMemAlloc_v2, cuCtxDestroy_v2).  The
 * cuda.h header maps the unversioned name to the versioned one via
 * a #define, but since we #undef those in cuda_dynload.h, we must
 * resolve the versioned symbol names directly.
 *
 * The CUDA documentation states: "cuMemAlloc" in the header is actually
 * cuMemAlloc_v2 in the library.  We try the _v2 name first, then fall back.
 */

#define LOAD_SYM(ptr, type, name)                                     \
    do {                                                               \
        ptr = (type)loadSym(name "_v2");                               \
        if (!ptr) ptr = (type)loadSym(name);                           \
        if (!ptr) {                                                    \
            fprintf(stderr, "CUDA dynload: failed to resolve %s\n",    \
                    name);                                             \
            return false;                                              \
        }                                                              \
    } while (0)

/* For functions that are NOT versioned (no _v2 variant) */
#define LOAD_SYM_EXACT(ptr, type, name)                                \
    do {                                                               \
        ptr = (type)loadSym(name);                                     \
        if (!ptr) {                                                    \
            fprintf(stderr, "CUDA dynload: failed to resolve %s\n",    \
                    name);                                             \
            return false;                                              \
        }                                                              \
    } while (0)

bool cudaDynloadInit()
{
    if (g_cudaInitDone) return g_cudaLoaded;
    g_cudaInitDone = true;

#ifdef _WIN32
    g_cudaLib = LoadLibraryA("nvcuda.dll");
#elif defined(__APPLE__)
    /* macOS: CUDA driver is rarely available; try anyway */
    g_cudaLib = dlopen("libcuda.dylib", RTLD_LAZY);
#else
    g_cudaLib = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!g_cudaLib)
        g_cudaLib = dlopen("libcuda.so", RTLD_LAZY);
#endif

    if (!g_cudaLib) {
        g_cudaLoaded = false;
        return false;
    }

    /* Resolve all function pointers.
     * Functions with _v2 variants in the driver library: cuCtxCreate,
     * cuCtxDestroy, cuMemAlloc, cuMemFree, cuMemcpyDtoH, cuMemcpyDtoHAsync,
     * cuMemcpyHtoDAsync, cuMemcpyDtoD, cuMemcpyDtoDAsync, cuStreamCreate.
     * Functions without _v2: cuInit, cuCtxSetCacheConfig, cuDeviceGetAttribute,
     * cuDeviceGetName, cuMemsetD8Async, cuModuleLoadDataEx, cuModuleUnload,
     * cuModuleGetFunction, cuStreamSynchronize, cuLaunchKernel.
     */

    LOAD_SYM_EXACT (dynCuInit,               PFN_cuInit,               "cuInit");
    LOAD_SYM       (dynCuCtxCreate,          PFN_cuCtxCreate,          "cuCtxCreate");
    LOAD_SYM       (dynCuCtxDestroy,         PFN_cuCtxDestroy,         "cuCtxDestroy");
    LOAD_SYM_EXACT (dynCuCtxSetCacheConfig,  PFN_cuCtxSetCacheConfig,  "cuCtxSetCacheConfig");
    LOAD_SYM_EXACT (dynCuDeviceGetAttribute, PFN_cuDeviceGetAttribute, "cuDeviceGetAttribute");
    LOAD_SYM_EXACT (dynCuDeviceGetCount,     PFN_cuDeviceGetCount,     "cuDeviceGetCount");
    LOAD_SYM       (dynCuDeviceGet,          PFN_cuDeviceGet,          "cuDeviceGet");
    LOAD_SYM_EXACT (dynCuDeviceGetName,      PFN_cuDeviceGetName,      "cuDeviceGetName");
    LOAD_SYM       (dynCuMemAlloc,           PFN_cuMemAlloc,           "cuMemAlloc");
    LOAD_SYM       (dynCuMemFree,            PFN_cuMemFree,            "cuMemFree");
    LOAD_SYM       (dynCuMemcpyDtoH,        PFN_cuMemcpyDtoH,        "cuMemcpyDtoH");
    LOAD_SYM       (dynCuMemcpyDtoHAsync,   PFN_cuMemcpyDtoHAsync,   "cuMemcpyDtoHAsync");
    LOAD_SYM       (dynCuMemcpyHtoDAsync,   PFN_cuMemcpyHtoDAsync,   "cuMemcpyHtoDAsync");
    LOAD_SYM       (dynCuMemcpyDtoD,        PFN_cuMemcpyDtoD,        "cuMemcpyDtoD");
    LOAD_SYM       (dynCuMemcpyDtoDAsync,   PFN_cuMemcpyDtoDAsync,   "cuMemcpyDtoDAsync");
    LOAD_SYM_EXACT (dynCuMemsetD8Async,     PFN_cuMemsetD8Async,     "cuMemsetD8Async");
    LOAD_SYM_EXACT (dynCuModuleLoadDataEx,  PFN_cuModuleLoadDataEx,  "cuModuleLoadDataEx");
    LOAD_SYM_EXACT (dynCuModuleUnload,      PFN_cuModuleUnload,      "cuModuleUnload");
    LOAD_SYM_EXACT (dynCuModuleGetFunction, PFN_cuModuleGetFunction, "cuModuleGetFunction");
    LOAD_SYM       (dynCuStreamCreate,      PFN_cuStreamCreate,      "cuStreamCreate");
    LOAD_SYM_EXACT (dynCuStreamSynchronize, PFN_cuStreamSynchronize, "cuStreamSynchronize");
    LOAD_SYM_EXACT (dynCuLaunchKernel,      PFN_cuLaunchKernel,      "cuLaunchKernel");

    g_cudaLoaded = true;
    return true;
}

bool cudaDynloadAvailable()
{
    return g_cudaLoaded;
}

#endif /* __USE_CUDA__ */
