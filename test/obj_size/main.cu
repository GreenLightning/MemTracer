#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "check.h"

// Example program used in the thesis to demonstrate the object size heuristics.

struct vec3 {
	float x, y, z;
};

__global__ void kernel(uint32_t count, uint32_t* infos, uint32_t* indices, vec3* positions, float* results) {
	uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= count) return;

	uint32_t begin = infos[2*x + 0];
	uint32_t end = infos[2*x + 1];

	float v = 0.0f;
	#pragma unroll 1
	for (uint32_t i = begin; i != end; i++) {
		uint32_t index = indices[i];
		vec3 p = positions[index];
		v += p.x * p.y * p.z;
	}

	results[x] = v;
}

float frac(float x) {
	return x - floor(x);
}

int main() {
	uint32_t positions_count = 100;
	std::vector<vec3> positions(positions_count);
	for (int i = 0; i < positions.size(); i++) {
		float w = static_cast<float>(i);
		positions[i].x = 100.0f * frac(sin(1.0f * w));
		positions[i].y = 100.0f * frac(sin(1.1f * w));
		positions[i].z = 100.0f * frac(sin(1.2f * w));
	}

	uint32_t index_count = 200;
	std::vector<uint32_t> indices(index_count);
	for (int i = 0; i < indices.size(); i++) {
		indices[i] = 138547 * i % positions_count;
	}

	uint32_t count = 4;
	std::vector<uint32_t> infos(2 * count);
	for (uint32_t i = 0; i < count; i++) {
		uint32_t begin = 138559 * i % index_count;
		uint32_t end = begin + 138563 * i % 30;
		if (end > index_count) end = index_count;
		infos[2*i + 0] = begin;
		infos[2*i + 1] = end;
	}

	std::vector<float> cpu_results(count);
	for (uint32_t x = 0; x < count; x++) {
		uint32_t begin = infos[2*x + 0];
		uint32_t end   = infos[2*x + 1];
		
		float v = 0.0f;
		for (uint32_t i = begin; i != end; i++) {
			uint32_t index = indices[i];
			vec3 p = positions[index];
			v += p.x * p.y * p.z;
		}

		cpu_results[x] = v;
	}

	std::vector<float> gpu_results(count);

	vec3* device_positions = nullptr;
	CUDA_CHECK(cudaMalloc(&device_positions, positions.size() * sizeof(positions[0])));
	CUDA_CHECK(cudaMemcpy(device_positions, positions.data(), positions.size() * sizeof(positions[0]), cudaMemcpyHostToDevice));

	uint32_t* device_indices = nullptr;
	CUDA_CHECK(cudaMalloc(&device_indices, indices.size() * sizeof(indices[0])));
	CUDA_CHECK(cudaMemcpy(device_indices, indices.data(), indices.size() * sizeof(indices[0]), cudaMemcpyHostToDevice));

	uint32_t* device_infos = nullptr;
	CUDA_CHECK(cudaMalloc(&device_infos, infos.size() * sizeof(infos[0])));
	CUDA_CHECK(cudaMemcpy(device_infos, infos.data(), infos.size() * sizeof(infos[0]), cudaMemcpyHostToDevice));

	float* device_results = nullptr;
	CUDA_CHECK(cudaMalloc(&device_results, gpu_results.size() * sizeof(gpu_results[0])));

	dim3 blockDim(32);
	dim3 gridDim((count + blockDim.x - 1) / blockDim.x);
	kernel<<<gridDim, blockDim>>>(count, device_infos, device_indices, device_positions, device_results);
	CUDA_CHECK_LAST_ERROR();

	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(gpu_results.data(), device_results, gpu_results.size() * sizeof(gpu_results[0]), cudaMemcpyDeviceToHost));

	float cpu = 0;
	for (uint32_t i = 0; i < count; i++) cpu += cpu_results[i];

	float gpu = 0;
	for (uint32_t i = 0; i < count; i++) gpu += gpu_results[i];

	printf("cpu: %f\n", cpu);
	printf("gpu: %f\n", gpu);
	return 0;
}
