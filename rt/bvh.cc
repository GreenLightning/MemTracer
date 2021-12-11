// Copyright (c) 2021, Max von Buelow, GRIS, Technical University of Darmstadt

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "vec.h"

#include "bvh.h"

enum SplitDescent { NODE_LEFT, NODE_RIGHT };

struct Split {
	uint32_t offset, count;
	uint32_t parent;
	int32_t level;
	SplitDescent desc;
};

void BVHBuilder::construct(const std::vector<AABB>& aabbs, const std::vector<vec3>& centers, Heuristic heuristic) {
	constexpr uint32_t numBins = 256;

	const uint32_t minSplitCount = maxPrimitives + (maxPrimitives & 1) + 2; // used for the cost calculation

	std::vector<uint32_t> perm(aabbs.size());
	std::iota(perm.begin(), perm.end(), 0);

	std::vector<Split> splitStack;
	splitStack.push_back(Split{ 0, static_cast<uint32_t>(aabbs.size()), -1u, 0, NODE_LEFT });

	std::vector<uint32_t> dimperms;

	while (!splitStack.empty()) {
		Split s = splitStack.back(); splitStack.pop_back();
		

		// uint32_t currentNode = this->emit_node(s.level, s.parent, s.desc);
		depth = max(s.level, depth);
		subtrees.emplace_back(0);
		uint32_t currentNode = num_nodes() - 1;
		if (s.desc == NODE_RIGHT) {
			subtrees[s.parent] = currentNode - s.parent - 1;
		}

		if (s.count <= maxPrimitives) {
			this->set_leaf(currentNode, perm.data() + s.offset, s.count, maxPrimitives);
			continue;
		}

		uint32_t binSize = (s.count + numBins - 1) / numBins;
		dimperms.resize(3 * s.count);
		float costs[3 * numBins];
	
		float maxDim = 0;
		int maxDimAxis;
		for (int axis = 0; axis < 3; axis++) {
			for (int i = 0; i < s.count; ++i) {
				dimperms[i + axis * s.count] = perm[s.offset + i];
			}

			uint32_t *dp = dimperms.data() + axis * s.count;
			std::sort(dp, dp + s.count, [&centers, axis] (uint32_t a, uint32_t b) {
				return centers[a][axis] < centers[b][axis];
			});

			AABB aabbLeft, aabbRight;
			float surface_l[numBins] = { 0 }, surface_r[numBins] = { 0 };
			for (int j = 0; j < binSize * numBins; ++j) {
				int jj = binSize * numBins - 1 - j;

				if (jj % binSize == binSize - 1) {
					vec3 d = aabbRight.size();
					surface_r[jj / binSize] = d.x * d.y + d.x * d.z + d.y * d.z;
				}

				if (j < s.count) {
					aabbLeft.feed(aabbs[dp[j]]);
				}

				if (jj < s.count) {
					aabbRight.feed(aabbs[dp[jj]]);
				}

				if (j % binSize == binSize - 1) {
					vec3 d = aabbLeft.size();
					surface_l[j / binSize] = d.x * d.y + d.x * d.z + d.y * d.z;
					if (std::isnan(surface_l[j / binSize]) || std::isinf(surface_l[j / binSize])) surface_l[j / binSize] = std::numeric_limits<float>::infinity();
				}
			}
			vec3 d = aabbLeft.size();
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
			float surface = d.x * d.y + d.x * d.z + d.y * d.z;
			for (int k = 0; k < numBins; ++k) {
				float lhs = surface_l[k] / surface;
				float rhs = surface_r[k] / surface;
				uint32_t num_l = (k + 1) * binSize;
				uint32_t num_r = s.count - num_l;
				costs[axis * numBins + k] = lhs * num_l + rhs * num_r;
				costs[axis * numBins + k] += (1 - std::min(num_l, minSplitCount) / (float)minSplitCount) + (1 - std::min(num_r, minSplitCount) / (float)minSplitCount);
			}
		}

		int bestIndex = 0;
		float min_cost = std::numeric_limits<float>::infinity();
		for (int i = 0; i < numBins * 3; ++i) {
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

		for (int i = 0; i < s.count; i++) {
			perm[s.offset + i] = dimperms[i + bestAxis * s.count];
		}

		uint32_t countLeft = bestSplit;
		uint32_t countRight = s.count - countLeft;

		// Calculate complete AABBs.
		AABB aabbLeft, aabbRight;
		for (int i = 0; i < countLeft; i++) {
			aabbLeft.feed(aabbs[perm[s.offset + i]]);
		}
		for (int i = 0; i < countRight; i++) {
			aabbRight.feed(aabbs[perm[s.offset + countLeft + i]]);
		}

		this->set_axis(currentNode, bestAxis);
		bounds.push_back(aabbLeft);
		bounds.push_back(aabbRight);

		// We have to handle the left node next, so we push it last on the stack.
		splitStack.push_back(Split{ s.offset + countLeft, countRight, currentNode, s.level + 1, NODE_RIGHT });
		splitStack.push_back(Split{ s.offset, countLeft, currentNode, s.level + 1, NODE_LEFT });
	}
}
