#pragma once

#include "vec.h"
#include "cuda.h"

enum Heuristic { SAH, MEDIAN };

enum SplitDescent { NODE_LEFT, NODE_RIGHT };

struct AABB {
	vec3 min, max;

	__host__ __device__ AABB() : min{ INFINITY, INFINITY, INFINITY }, max{ -INFINITY, -INFINITY, -INFINITY } {}

	__host__ __device__ void feed(const vec3& pos) {
		feed_min(pos);
		feed_max(pos);
	}
	
	__host__ __device__ void feed_min(const vec3& pos) {
		this->min = ::min(this->min, pos);
	}
	
	__host__ __device__ void feed_max(const vec3& pos) {
		this->max = ::max(this->max, pos);
	}
};