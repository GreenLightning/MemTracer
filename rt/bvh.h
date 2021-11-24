#pragma once
#include <cstdint>
#include <iostream>
#include <fstream>
#include "aabb.h"
#include <unordered_set>
#include <queue>

enum Heuristic { SAH, MEDIAN };

enum SplitDescent { NODE_LEFT, NODE_RIGHT };
struct BVHBuilder
{
	uint32_t maxlvl;
	void setmaxlvl(uint32_t ml)
	{
		maxlvl = ml;
		std::cout << "SETTING ML: " << maxlvl << std::endl;
	}
	uint32_t num_nodes()
	{
		return subtrees.size();
	}
	std::vector<AABB> bounds;
	std::vector<uint32_t> subtrees;
	struct Parent {
		uint32_t idx, subtree;
		SplitDescent desc;
		Parent() : subtree(0)
		{}
	};
	std::vector<Parent> parents;
	BVHBuilder() : parents(128) {}
	std::vector<uint32_t> leaf_nodes;
	uint32_t depth = 0;
	template <typename T>
	void serialize(std::ostream &os, const std::vector<T> &data)
	{
		uint64_t size = data.size();
		os.write((const char*)&size, 8);
		os.write((const char*)data.data(), data.size() * sizeof(T));
	}
	template <typename T>
	void unserialize(std::istream &is, std::vector<T> &data)
	{
		uint64_t size;
		is.read((char*)&size, 8);
		data.resize(size);
		is.read((char*)data.data(), data.size() * sizeof(T));
	}
	void backup(const char *name, int tag)
	{
		std::ofstream os((std::string("bvhbuilder") + name + "." + std::to_string(tag) + ".bin").c_str());
		std::vector<uint32_t> dummy;
		dummy.push_back(maxlvl);
		serialize<uint32_t>(os, dummy);
		serialize<AABB>(os, bounds);
		serialize<uint32_t>(os, subtrees);
		serialize<uint32_t>(os, leaf_nodes);
	}
	bool restore(const char *name, int tag)
	{
		std::ifstream is((std::string("bvhbuilder") + name + "." + std::to_string(tag) + ".bin").c_str());
		if (!is) return false;
		std::vector<uint32_t> dummy;
		unserialize<uint32_t>(is, dummy);
		maxlvl = dummy.back();
		unserialize<AABB>(is, bounds);
		unserialize<uint32_t>(is, subtrees);
		unserialize<uint32_t>(is, leaf_nodes);
		return true;
	}
	uint32_t emit_node(uint32_t level = 0, uint32_t parent = -1u, SplitDescent desc = NODE_LEFT)
	{
		depth = std::max(level, depth);
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
	void set_axis(uint32_t nidx, uint32_t axis)
	{
		subtrees[nidx] |= axis << 30;
	}

	void set_bounds(uint32_t nidx, const AABB &l, const AABB &r)
	{
		bounds.push_back(l);
		bounds.push_back(r);
	}
	void set_leaf(uint32_t nidx, uint32_t *sat, uint32_t n, uint32_t nmax)
	{
		subtrees[nidx] = n;
		set_axis(nidx, 3); // mark as leaf


		uint32_t off = leaf_nodes.size();
		leaf_nodes.resize(off + nmax, -1u);
		std::copy(sat, sat + n, leaf_nodes.begin() + off);
	}

	void construct(float *cens, float *aabbs, uint32_t n, uint32_t nleaf, Heuristic heuristic);
};
