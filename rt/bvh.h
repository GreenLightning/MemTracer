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

	std::vector<AABB> bounds;
	std::vector<uint32_t> subtrees;
	std::vector<uint32_t> leaf_nodes;

	BVHBuilder(int32_t maxPrimitives) : maxPrimitives(maxPrimitives) {}

	uint32_t num_nodes() {
		return subtrees.size();
	}

	void set_axis(uint32_t nodeIndex, uint32_t axis) {
		subtrees[nodeIndex] |= axis << 30;
	}

	void set_leaf(uint32_t nodeIndex, uint32_t *sat, uint32_t n, uint32_t nmax) {
		subtrees[nodeIndex] = n;
		set_axis(nodeIndex, 3); // mark as leaf

		uint32_t off = leaf_nodes.size();
		leaf_nodes.resize(off + nmax, -1u);
		std::copy(sat, sat + n, leaf_nodes.begin() + off);
	}

	void construct(const std::vector<AABB>& aabbs, const std::vector<vec3>& centers, Heuristic heuristic);
};
