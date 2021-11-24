#pragma once

#include <unordered_map>
#include "meminf.h"

struct Meminf {
	int desc;
	std::size_t size; // filled by data from the driver API
};

extern std::unordered_map<uint64_t, Meminf> meminfs;
