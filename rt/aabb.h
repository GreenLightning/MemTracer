#pragma once

#include <limits>
#include <algorithm>

struct AABB {
	float min[3], max[3];

// 	AABB(const AABB &other)
// 	{
// 		for (int i = 0; i < 3; ++i) {
// 			min[i] = other.min[i];
// 			max[i] = other.max[i];
// 		}
// 	}
	AABB() : min{ std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity() }, max{ -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() }
	{}
	AABB(float *mi, float *ma)
	{
		for (int i = 0; i < 3; ++i) {
			min[i] = mi[i];
			max[i] = ma[i];
		}
	}

	void feed(const float *pos)
	{
		feed_min(pos);
		feed_max(pos);
	}
	void feed_min(const float *pos)
	{
		for (int i = 0; i < 3; ++i) {
			min[i] = std::min(min[i], pos[i]);
		}
	}
	void feed_max(const float *pos)
	{
		for (int i = 0; i < 3; ++i) {
			max[i] = std::max(max[i], pos[i]);
		}
	}
};
