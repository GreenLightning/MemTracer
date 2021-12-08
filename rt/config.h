#pragma once

#include <string>

#include "types.h"
#include "bvh_types.h"

struct Configuration {
	std::string input;
	std::string output;
	int32_t width = 0, height = 0;
	Heuristic heuristic = SAH;
	Camera camera;
};

void loadConfiguration(Configuration& config, const std::string& path);
