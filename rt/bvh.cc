// Copyright (c) 2021, Max von Buelow, GRIS, Technical University of Darmstadt

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "vec.h"

#include "bvh.h"

bool validateHeuristic(const std::string& value) {
	return (value == "sah") || (value == "median");
}

Heuristic parseHeuristic(const std::string& value) {
	// NOTE: SAH is default value (for empty string).
	return (value == "median") ? MEDIAN : SAH;
}

enum SplitDescent { NODE_LEFT, NODE_RIGHT };

struct Split {
	// Range of primitives to be considered for this node.
	int32_t offset, count;
	uint32_t parentIndex;
	uint32_t level;
	SplitDescent desc;
};

BVH constructBVH(const std::vector<AABB>& aabbs, const std::vector<vec3>& centers, int32_t maxPrimitives, Heuristic heuristic) {
	BVH bvh = BVH(maxPrimitives);

	constexpr int32_t numBins = 256;

	const uint32_t minSplitCount = maxPrimitives + (maxPrimitives & 1) + 2; // used for the cost calculation

	// perm stores a permutation of aabbs and centers
	// as the primitives get sorted into the tree.
	std::vector<uint32_t> perm(aabbs.size());
	std::iota(perm.begin(), perm.end(), 0);

	// localPerms stores permutations for each axis.
	// Declared here to reuse the memory.
	std::vector<uint32_t> localPerms;

	std::vector<Split> splitStack;
	splitStack.push_back(Split{ 0, static_cast<int32_t>(aabbs.size()), UINT32_MAX, 0, NODE_LEFT });
	while (!splitStack.empty()) {
		Split s = splitStack.back(); splitStack.pop_back();
		
		bvh.depth = max(s.level, bvh.depth);

		uint32_t currentNodeIndex = static_cast<uint32_t>(bvh.nodes.size());
		bvh.nodes.emplace_back(0);

		if (s.desc == NODE_RIGHT) {
			// If we are a right child, we can compute our offset now and
			// store it in our parent.
			bvh.nodes[s.parentIndex] = currentNodeIndex - s.parentIndex;
		}

		if (s.count <= maxPrimitives) {
			// Create a leaf node.
			bvh.nodes[currentNodeIndex] = (1u << 31) | s.count;

			// Copy primitive indices.
			auto offset = bvh.primitives.size();
			bvh.primitives.resize(offset + maxPrimitives, UINT32_MAX);
			std::copy(perm.begin() + s.offset, perm.begin() + s.offset + s.count, bvh.primitives.begin() + offset);
			continue;
		}

		int32_t binSize = (s.count + numBins - 1) / numBins;

		float costs[3 * numBins];
		localPerms.resize(3 * s.count);
	
		float maxDim = 0;
		int maxDimAxis = 0;
	
		for (int axis = 0; axis < 3; axis++) {
			uint32_t* p = localPerms.data() + axis * s.count;
			for (int i = 0; i < s.count; i++) {
				p[i] = perm[s.offset + i];
			}

			std::sort(p, p + s.count, [&centers, axis] (uint32_t a, uint32_t b) {
				return centers[a][axis] < centers[b][axis];
			});

			AABB aabbLeft, aabbRight;
			float surfaceLeft[numBins] = {}, surfaceRight[numBins] = {};
			for (int left = 0, right = numBins * binSize - 1; left < numBins * binSize; left++, right--) {
				if (right % binSize == binSize - 1) {
					vec3 d = aabbRight.size();
					surfaceRight[right / binSize] = d.x * d.y + d.x * d.z + d.y * d.z;
				}

				if (left < s.count)  aabbLeft.feed(aabbs[p[left]]);
				if (right < s.count) aabbRight.feed(aabbs[p[right]]);

				if (left % binSize == binSize - 1) {
					vec3 d = aabbLeft.size();
					float surface = d.x * d.y + d.x * d.z + d.y * d.z;
					if (std::isnan(surface) || std::isinf(surface)) surface = std::numeric_limits<float>::infinity();
					surfaceLeft[left / binSize] = surface;
				}
			}

			// Both AABBs now contain all primitives.
			vec3 d = aabbLeft.size();
			float surface = d.x * d.y + d.x * d.z + d.y * d.z;

			if (axis == 0) {
				if (d.x > maxDim) {
					maxDim = d.x;
					maxDimAxis = 0;
				}
				if (d.y > maxDim) {
					maxDim = d.y;
					maxDimAxis = 1;
				}
				if (d.z > maxDim) {
					maxDim = d.z;
					maxDimAxis = 2;
				}
			}
		
			for (int k = 0; k < numBins; k++) {
				float lhs = surfaceLeft[k] / surface;
				float rhs = surfaceRight[k] / surface;
				uint32_t numLeft = (k + 1) * binSize;
				uint32_t numRight = s.count - numLeft;
				float cost = lhs * numLeft + rhs * numRight
					+ (1 - std::min(numLeft, minSplitCount) / (float)minSplitCount)
					+ (1 - std::min(numRight, minSplitCount) / (float)minSplitCount);
				costs[axis * numBins + k] = cost;
			}
		}

		int bestIndex = 0;
		float min_cost = std::numeric_limits<float>::infinity();
		for (int i = 0; i < 3 * numBins; i++) {
			if (costs[i] < min_cost) {
				bestIndex = i;
				min_cost = costs[i];
			}
		}

		int bestSplit = ((bestIndex % numBins) + 1) * binSize;
		int bestAxis = bestIndex / numBins;

		if (heuristic == MEDIAN) {
			bestAxis = maxDimAxis;
			bestSplit = s.count / 2;
		}

		uint32_t* p = localPerms.data() + bestAxis * s.count;
		for (int i = 0; i < s.count; i++) {
			perm[s.offset + i] = p[i];
		}

		int32_t countLeft = bestSplit;
		int32_t countRight = s.count - countLeft;

		// Calculate complete AABBs.
		AABB aabbLeft, aabbRight;
		for (int i = 0; i < countLeft; i++) {
			aabbLeft.feed(aabbs[perm[s.offset + i]]);
		}
		for (int i = 0; i < countRight; i++) {
			aabbRight.feed(aabbs[perm[s.offset + countLeft + i]]);
		}

		bvh.bounds.push_back(aabbLeft);
		bvh.bounds.push_back(aabbRight);

		// We have to handle the left node next, so we push it last on the stack.
		splitStack.push_back(Split{ s.offset + countLeft, countRight, currentNodeIndex, s.level + 1, NODE_RIGHT });
		splitStack.push_back(Split{ s.offset, countLeft, currentNodeIndex, s.level + 1, NODE_LEFT });
	}

	return bvh;
}
