// Copyright (c) 2021, Max von Buelow, GRIS, Technical University of Darmstadt

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "vec.h"

#include "bvh.h"

#define NBINS 256

struct Split {
	uint32_t offset, count;
	uint32_t node;
	uint32_t split_axis;
	bool leaf;
	int32_t level;
	SplitDescent desc;
};

void BVHBuilder::construct(const std::vector<AABB>& aabbs, const std::vector<vec3>& centers, Heuristic heuristic) {
	// Used for the cost calculation.
	const uint32_t minSplitCount = maxPrimitives + (maxPrimitives & 1) + 2;

	std::vector<uint32_t> perm(aabbs.size());
	std::iota(perm.begin(), perm.end(), 0);

	std::vector<Split> splitStack;
	
	bool dosplit = aabbs.size() > maxPrimitives;

	splitStack.push_back(Split{ 0, aabbs.size(), -1u, 0, !dosplit, 0, NODE_LEFT });
	int  nidx, split_axis, parent_node;
	int32_t level;

	while (!splitStack.empty()) {
		Split s = splitStack.back(); splitStack.pop_back();
		parent_node = s.node; split_axis = s.split_axis; level = s.level;
		uint32_t cur_node = this->emit_node(level, parent_node, s.desc);

		// get bounding triangles and swap them to the first two positions of perm
		float val = std::numeric_limits<float>::infinity();
		uint32_t idx;

		if (s.leaf) {
			this->set_leaf(cur_node, perm.data() + s.offset, s.count, maxPrimitives);
			continue;
		}

		uint32_t binsize = (s.count + NBINS - 1) / NBINS;
		float min_cost = std::numeric_limits<float>::infinity();
		std::vector<uint32_t> dimperms(s.count * 3);
		float costs[NBINS * 3];
		vec3 mins_l[NBINS * 3], maxs_l[NBINS * 3], mins_r[NBINS * 3], maxs_r[NBINS * 3];
		float max_d = 0, max_d_axis;
		for (int axis = 0; axis < 3; ++axis) {
			for (int i = 0; i < s.count; ++i) {
				dimperms[i + axis * s.count] = perm[s.offset + i];
			}

			uint32_t *dp = dimperms.data() + axis * s.count;
			std::sort(dp, dp + s.count, [&](uint32_t a, uint32_t b) {
				return centers[a][axis] < centers[b][axis];
			});

			vec3 min_l(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()), max_l(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
			vec3 min_r(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()), max_r(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
			float surface_l[NBINS] = { 0 }, surface_r[NBINS] = { 0 };
			for (int j = 0; j < binsize * NBINS; ++j) {
				int jj = binsize * NBINS - 1 - j;

				if (jj % binsize == binsize - 1) {
					vec3 d = max_r - min_r;
					surface_r[jj / binsize] = d.x * d.y + d.x * d.z + d.y * d.z;

					mins_r[axis * NBINS + jj / binsize] = min_r;
					maxs_r[axis * NBINS + jj / binsize] = max_r;
				}
				if (j < s.count) {
					min_l = min(min_l, aabbs[dp[j]].min);
					max_l = max(max_l, aabbs[dp[j]].max);
				}

				if (jj < s.count) {
					min_r = min(min_r, aabbs[dp[jj]].min);
					max_r = max(max_r, aabbs[dp[jj]].max);
				}

				if (j % binsize == binsize - 1) {
					vec3 d = max_l - min_l;
					surface_l[j / binsize] = d.x * d.y + d.x * d.z + d.y * d.z;
					if (std::isnan(surface_l[j / binsize]) || std::isinf(surface_l[j / binsize])) surface_l[j / binsize] = std::numeric_limits<float>::infinity();

					mins_l[axis * NBINS + j / binsize] = min_l;
					maxs_l[axis * NBINS + j / binsize] = max_l;
				}
			}
			vec3 d = max_l - min_l;
			if (axis == 0) {
				if (d.x > max_d) {
					max_d = d.x;
					max_d_axis = 0;
				}
				if (d.y > max_d) {
					max_d = d.y;
					max_d_axis = 1;
				}
				if (d.z > max_d) {
					max_d = d.z;
					max_d_axis = 2;
				}
			}
			float surface = d.x * d.y + d.x * d.z + d.y * d.z;
			for (int k = 0; k < NBINS; ++k) {
				float lhs = surface_l[k] / surface;
				float rhs = surface_r[k] / surface;
				uint32_t num_l = (k + 1) * binsize;
				uint32_t num_r = s.count - num_l;
				costs[axis * NBINS + k] = lhs * num_l + rhs * num_r;
				costs[axis * NBINS + k] += (1 - std::min(num_l, minSplitCount) / (float)minSplitCount) + (1 - std::min(num_r, minSplitCount) / (float)minSplitCount);
			}
		}

		int best_idx = 0;
		for (int i = 0; i < NBINS * 3; ++i) {
			if (costs[i] < min_cost) {
				best_idx = i;
				min_cost = costs[i];
			}
		}
		int best_split = ((best_idx % NBINS) + 1) * binsize;
		int best_axis = best_idx / NBINS;

		if (heuristic == MEDIAN) {
			best_axis = max_d_axis/*(split_axis + 1) % 3*/;
			best_split = s.count / 2;
		}



		for (int i = 0; i < s.count; ++i) {
			perm[s.offset + i] = dimperms[i + best_axis * s.count];
		}

		// new split
		int nl = best_split;
		int nr = s.count - nl;

		// calculate complete AABBs
		AABB aabb_l, aabb_r;
		for (int i = 0; i < nl; ++i) {
			aabb_l.feed_min(aabbs[perm[s.offset + i]].min);
			aabb_l.feed_max(aabbs[perm[s.offset + i]].max);
		}
		for (int i = 0; i < nr; ++i) {
			aabb_r.feed_min(aabbs[perm[s.offset + nl + i]].min);
			aabb_r.feed_max(aabbs[perm[s.offset + nl + i]].max);
		}


		uint32_t lhs_min, lhs_max, rhs_min, rhs_max;

		if (nl < 0 || nr < 0) throw std::runtime_error("Negative split count");
		this->set_axis(cur_node, best_axis);
		this->set_bounds(cur_node, aabb_l, aabb_r);

		bool r = nr > maxPrimitives;
		bool l = nl > maxPrimitives;

		splitStack.push_back(Split{ s.offset + nl, nr, cur_node, best_axis, !r, level + 1, NODE_RIGHT });
		splitStack.push_back(Split{ s.offset, nl, cur_node, best_axis, !l, level + 1, NODE_LEFT });
	}
}
