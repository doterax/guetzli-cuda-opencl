#pragma once

#ifdef __USE_OPENCL__

template<typename T>
inline cl_int clSetKernelArgK(cl_kernel k, int idx, T* t)
{
    return clSetKernelArg(k, idx, sizeof(T), t);
}

template<>
inline cl_int clSetKernelArgK(cl_kernel k, int idx, int* t)
{
    cl_int c = *t;
    return clSetKernelArg(k, idx, sizeof(cl_int), &c);
}

template<>
inline cl_int clSetKernelArgK(cl_kernel k, int idx, const int* t)
{
    cl_int c = *t;
    return clSetKernelArg(k, idx, sizeof(cl_int), &c);
}

template<>
inline cl_int clSetKernelArgK(cl_kernel k, int idx, size_t* t)
{
    cl_int c = *t;
    return clSetKernelArg(k, idx, sizeof(cl_int), &c);
}

template<>
inline cl_int clSetKernelArgK(cl_kernel k, int idx, const size_t* t)
{
    cl_int c = *t;
    return clSetKernelArg(k, idx, sizeof(cl_int), &c);
}

// Sets kernel arguments 0..N-1 from a list of pointers to host values.
// Returns the first error encountered (CL_SUCCESS if none).
inline cl_int clSetKernelArgFrom(cl_kernel, int)
{
    return CL_SUCCESS;
}

template<typename T, typename... Rest>
inline cl_int clSetKernelArgFrom(cl_kernel k, int idx, T* t, Rest*... rest)
{
    cl_int err = clSetKernelArgK(k, idx, t);
    if (err != CL_SUCCESS) LogError("clSetKernelArg(%d) failed: %d (size %d)\n", idx, (int)err, (int)sizeof(T));
    cl_int err_rest = clSetKernelArgFrom(k, idx + 1, rest...);
    return err != CL_SUCCESS ? err : err_rest;
}

template<typename... Args>
inline cl_int clSetKernelArgEx(cl_kernel k, Args*... args)
{
    return clSetKernelArgFrom(k, 0, args...);
}

#endif // __USE_OPENCL__
