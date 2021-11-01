#pragma once

#if !defined(NDEBUG)

#include <stdio.h>
#include <cuda_runtime.h>

// Here are some macros to check cuda error codes in debug builds.
// Just wrap a function that returns a cuda error with CUDA_CHECK():
//
//     CUDA_CHECK(cudaMalloc(...));
//
// You can also use CUDA_CHECK_LAST_ERROR() to check the last error
// as returned by cudaGetLastError(). The reason this is a separate macro
// is that CUDA_CHECK(cudaGetLastError()) would still call cudaGetLastError()
// in release mode, which resets the last error, but the returned code
// is not checked in that mode. Therefore CUDA_CHECK_LAST_ERROR()
// is preferred.

#define CUDA_CHECK(code) { cudaError_t _cudaCheckError = (code); cudaCheck(_cudaCheckError , __FILE__, __LINE__); }
#define CUDA_CHECK_LAST_ERROR() CUDA_CHECK(cudaGetLastError())

inline void cudaCheck(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		fprintf(stderr, "%s:%d: %s (%d)\n", file, line, cudaGetErrorString(err), err);
	}
}

#else

#define CUDA_CHECK(code) code
#define CUDA_CHECK_LAST_ERROR()

#endif
