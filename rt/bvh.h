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

// Der BVH tree ist als uint32 array implementiert, bei dem zwei bits markieren (aus historischen Gründen 2 und nicht 1), obs ein leaf ist und die anderen bits den offset zum rechten child angeben (der linke ist das nächste Element)
//
struct BVHBuilder {
	struct Parent {
		uint32_t idx, subtree;
		SplitDescent desc;
		Parent() : subtree(0) {}
	};

	const int32_t maxLeaves;
	int32_t depth = 0;

	std::vector<AABB> bounds;
	std::vector<uint32_t> subtrees;
	std::vector<uint32_t> leaf_nodes;
	std::vector<Parent> parents;

	BVHBuilder(int32_t maxLeaves) : maxLeaves(maxLeaves), parents(128) {}

	uint32_t emit_node(int32_t level = 0, uint32_t parent = -1u, SplitDescent desc = NODE_LEFT) {
		depth = max(level, depth);
		subtrees.emplace_back(0);
		uint32_t s = parents.size();
		while (level >= s) s <<= 1;
		parents.resize(s);
		parents[level].idx = parent;
		parents[level].subtree = 0;
		parents[level].desc = desc;

		while (level > 0) {
			++parents[level].subtree;
			if (parents[level].desc == NODE_LEFT) {
				subtrees[parents[level].idx] &= 0xc0000000u;
				subtrees[parents[level].idx] |= parents[level].subtree;
			}
			--level;
		}
		++parents[level].subtree;

		return num_nodes() - 1;
	}

	uint32_t num_nodes() {
		return subtrees.size();
	}

	void set_axis(uint32_t nidx, uint32_t axis) {
		subtrees[nidx] |= axis << 30;
	}

	void set_bounds(uint32_t nidx, const AABB &l, const AABB &r) {
		bounds.push_back(l);
		bounds.push_back(r);
	}

	void set_leaf(uint32_t nidx, uint32_t *sat, uint32_t n, uint32_t nmax) {
		subtrees[nidx] = n;
		set_axis(nidx, 3); // mark as leaf

		uint32_t off = leaf_nodes.size();
		leaf_nodes.resize(off + nmax, -1u);
		std::copy(sat, sat + n, leaf_nodes.begin() + off);
	}

	void construct(const std::vector<AABB>& aabbs, const std::vector<vec3>& centers, Heuristic heuristic);
};
