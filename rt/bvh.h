#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vec.h"
#include "cuda.h"

enum Heuristic { SAH, MEDIAN };

bool validateHeuristic(const std::string& value);
Heuristic parseHeuristic(const std::string& value);

struct AABB {
	vec3 min, max;

	__host__ __device__ AABB() : min{ INFINITY, INFINITY, INFINITY }, max{ -INFINITY, -INFINITY, -INFINITY } {}

	// Size is negative if the AABB is invalid.
	__host__ __device__ vec3 size() {
		return max - min;
	}

	__host__ __device__ void feed(const vec3& pos) {
		feed_min(pos);
		feed_max(pos);
	}

	__host__ __device__ void feed(const AABB& aabb) {
		feed_min(aabb.min);
		feed_max(aabb.max);
	}
	
	__host__ __device__ void feed_min(const vec3& pos) {
		this->min = ::min(this->min, pos);
	}
	
	__host__ __device__ void feed_max(const vec3& pos) {
		this->max = ::max(this->max, pos);
	}
};

struct BVHBuilder {
	// maxPrimitives is the maximum number of primitives a leaf can contain.
	const int32_t maxPrimitives;

	// depth is the maximum depth of the hierarchy.
	int32_t depth = 0;

	std::vector<uint32_t> subtrees;
	std::vector<AABB> bounds;
	std::vector<uint32_t> leaf_nodes;

	BVHBuilder(int32_t maxPrimitives) : maxPrimitives(maxPrimitives) {}

	void construct(const std::vector<AABB>& aabbs, const std::vector<vec3>& centers, Heuristic heuristic);
};
