#pragma once

#include <string>
#include <vector>

enum Shading {
	SMOOTH, FLAT
};

struct Camera {
	float x = 0, y = 0, z = 0;
	float mat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
	float vertical_fov = 60;
};

struct Light {
	float x = 0, y = 0, z = 0;
};

struct Configuration {
	std::string input;
	std::string output;
	int32_t width = 0, height = 0;
	Shading shading = SMOOTH;
	bool shadows = false;
	std::string heuristic;
	std::vector<Camera> cameras;
	Light light;
};

void loadConfiguration(Configuration& config, const std::string& path);

// This function is used by the configuration code to detect errors early.
// It is forward declared here to achieve a very loose coupling.
bool validateHeuristic(const std::string& value);
