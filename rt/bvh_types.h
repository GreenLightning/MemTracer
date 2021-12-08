#pragma once

enum Heuristic { SAH, MEDIAN };

enum SplitDescent { NODE_LEFT, NODE_RIGHT };

struct AABB {
	float min[3], max[3];

	AABB() : min{ std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity() }, max{ -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() }
	{}

	void feed(const float *pos) {
		feed_min(pos);
		feed_max(pos);
	}
	
	void feed_min(const float *pos) {
		for (int i = 0; i < 3; ++i) {
			min[i] = std::min(min[i], pos[i]);
		}
	}
	
	void feed_max(const float *pos) {
		for (int i = 0; i < 3; ++i) {
			max[i] = std::max(max[i], pos[i]);
		}
	}
};
