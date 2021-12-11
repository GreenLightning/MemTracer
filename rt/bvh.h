#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "bvh_types.h"

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
