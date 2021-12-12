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

struct BVH {
	using Node = uint32_t;

	// maxPrimitives is the maximum number of primitives a leaf can contain.
	const uint32_t maxPrimitives;

	// depth is the maximum depth of the tree.
	uint32_t depth = 0;

	// nodes contains a binary tree of inner nodes that have exactly two child
	// nodes and leaf nodes that contain a variable number of primitives.
	//
	// Nodes are distinguished by their most significant bit. Leaf nodes have
	// the MSB set and inner nodes do not.
	//
	// The tree is encoded implicitly in the structure of the array. The left
	// child of an inner node is its immediate successor in the array. To get
	// to the right child the inner node contains an offset, which has to be
	// added to the inner nodes index.
	std::vector<Node> nodes;

	// bounds contains two AABBs for each inner node describing the bounds of
	// its child nodes.
	std::vector<AABB> bounds;

	// primitives contains maxPrimitives primitive indices for each leaf node.	
	// The leaf node contains the number of valid indices (if the MSB is
	// masked off). Invalid indices are set to -1.
	std::vector<uint32_t> primitives;

	// NOTE: nodes contains both inner and leaf nodes, while bounds is only
	// for inner nodes and primitives is only for leaf nodes. Therefore each
	// vector has to be indexed differently. See trace.cu for an explanation
	// of the index computations.

	BVH(uint32_t maxPrimitives) : maxPrimitives(maxPrimitives) {}
};

BVH constructBVH(const std::vector<AABB>& aabbs, const std::vector<vec3>& centers, uint32_t maxPrimitives, Heuristic heuristic);
