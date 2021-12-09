#pragma once

#include "vec.h"
#include "cuda.h"

enum Heuristic { SAH, MEDIAN };

enum SplitDescent { NODE_LEFT, NODE_RIGHT };

struct AABB {
	vec3 min, max;

	AABB() : min{ INFINITY, INFINITY, INFINITY }, max{ -INFINITY, -INFINITY, -INFINITY } {}

	void feed(const vec3& pos) {
		feed_min(pos);
		feed_max(pos);
	}
	
	void feed_min(const vec3& pos) {
		this->min = ::min(this->min, pos);
	}
	
	void feed_max(const vec3& pos) {
		this->max = ::max(this->max, pos);
	}
};
