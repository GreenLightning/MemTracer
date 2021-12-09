#pragma once

#include <cmath>

#ifdef __CUDACC__
	#include <cuda_runtime.h>
#endif

#ifndef __CUDACC__
	template <typename T> T min(T a, T b) {
		return a < b ? a : b;
	}

	template <typename T> T max(T a, T b) {
		return a > b ? a : b;
	}

	struct float4 { float x, y, z, w; };

	#define __host__
	#define __device__
	#define __global__

	#define __syncthreads()
#endif

template <typename T>
__host__ __device__ T clamp(T v, T a, T b) {
	return (v < a) ? a : (v > b) ? b : v;
}
