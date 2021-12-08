// Copyright (c) 2021, Max von Buelow, GRIS, Technical University of Darmstadt

#include "bvh.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <stack>
#include <unordered_set>
#include <vector>

#include "vec.h"

#define NBINS 256


struct SplitX {
	uint32_t o, l;
	uint32_t node;
	uint32_t split_axis;
	bool leaf;
	uint32_t level;
	SplitDescent desc;
	SplitX()
	{}
	SplitX(uint32_t o, uint32_t l, uint32_t node, uint32_t split_axis, bool leaf, uint32_t level = 0, SplitDescent desc = NODE_LEFT) : o(o), l(l), node(node), split_axis(split_axis), leaf(leaf), level(level), desc(desc)
	{}
};

void BVHBuilder::construct(float *cens, float *aabbs, uint32_t n, uint32_t nleaf, Heuristic heuristic)
{
	int max_axis = 3; // TODO
	uint32_t leafminsplitcount = nleaf + (nleaf & 1) + 2;

// 	std::cout << "Overall surface: " << surface_all << std::endl;

	std::vector<uint32_t> perm(n);
	std::vector<uint32_t> tree;
	std::iota(perm.begin(), perm.begin() + n, 0);

// 	std::cout << "Initial compontents: " << compontents << std::endl;

	std::deque<SplitX> S;
	
	bool dosplit = n > nleaf;

	S.emplace_back(0, n, -1u, 0, !dosplit, 0);
	int o, l, nidx, split_axis, level, parent_node;

	double total = 0.;
	uint32_t finalized = 0;
	int maxlevel = 0;
	int RR = 0;
	while (!S.empty()) {
		SplitX s = S.back(); S.pop_back();
		o = s.o; l = s.l; parent_node = s.node; split_axis = s.split_axis; level = s.level;
		maxlevel = std::max(maxlevel, level);
		uint32_t cur_node = this->emit_node(level, parent_node, s.desc);
// 		std::cout << "len: " << l << std::endl;
// 		std::cout << "\rCur len: " << l << "                         ";
		std::cout.flush();

// 		// get bounding triangles and swap them to the first two positions of perm
		float val = std::numeric_limits<float>::infinity();
		uint32_t idx;
		// min
// 		std::cout << "SPLIT AXIS: " << split_axis << std::endl;
		if (s.leaf) {
			this->set_leaf(cur_node, perm.data() + o, l, nleaf);
			continue;
		}

		uint32_t binsize = (l + NBINS - 1) / NBINS;
		float min_cost = std::numeric_limits<float>::infinity();
		std::vector<uint32_t> dimperms(l * 3);
		float costs[NBINS * 3];
		vec3 mins_l[NBINS * 3], maxs_l[NBINS * 3], mins_r[NBINS * 3], maxs_r[NBINS * 3];
		float max_d = 0, max_d_axis;
		for (int axis = 0; axis < max_axis; ++axis) {
			for (int i = 0; i < l; ++i) {
				dimperms[i + axis * l] = perm[o + i];
			}

			uint32_t *dp = dimperms.data() + axis * l;
			std::sort(dp, dp + l, [&](uint32_t a, uint32_t b) {
				return cens[a * 3 + axis] < cens[b * 3 + axis];
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
				if (j < l) {
					min_l = min(min_l, vec3(aabbs + dp[j] * 6));
					max_l = max(max_l, vec3(aabbs + dp[j] * 6 + 3));
				}

				if (jj < l) {
					min_r = min(min_r, vec3(aabbs + dp[jj] * 6));
					max_r = max(max_r, vec3(aabbs + dp[jj] * 6 + 3));
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
				uint32_t num_r = l - num_l;
				costs[axis * NBINS + k] = lhs * num_l + rhs * num_r;
				costs[axis * NBINS + k] += (1 - std::min(num_l, leafminsplitcount) / (float)leafminsplitcount) + (1 - std::min(num_r, leafminsplitcount) / (float)leafminsplitcount);
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
			best_split = l / 2;
		}



		for (int i = 0; i < l; ++i) {
			perm[o + i] = dimperms[i + best_axis * l];
		}

		// new split
		int nl = best_split;
		int nr = l - nl;

		// calculate complete AABBs
		AABB aabb_l, aabb_r;
		for (int i = 0; i < nl; ++i) {
			aabb_l.feed_min(aabbs + perm[o + i] * 6);
			aabb_l.feed_max(aabbs + perm[o + i] * 6 + 3);
		}
		for (int i = 0; i < nr; ++i) {
			aabb_r.feed_min(aabbs + perm[o + nl + i] * 6);
			aabb_r.feed_max(aabbs + perm[o + nl + i] * 6 + 3);
		}


		uint32_t lhs_min, lhs_max, rhs_min, rhs_max;

		if (nl < 0 || nr < 0) throw std::runtime_error("Negative split count");
		this->set_axis(cur_node, best_axis);
		this->set_bounds(cur_node, aabb_l, aabb_r);

		bool r = nr > nleaf;
		bool l = nl > nleaf;

		S.emplace_back(o + nl, nr, cur_node, best_axis, !r, level + 1, NODE_RIGHT);
		S.emplace_back(o, nl, cur_node, best_axis, !l, level + 1, NODE_LEFT);
	}
	this->setmaxlvl(maxlevel);
}
