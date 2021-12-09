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
