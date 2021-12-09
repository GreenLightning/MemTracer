#pragma once

#include <algorithm>
#include <ostream>

#include "cuda.h"

struct vec3 {
	float x, y, z;

	__device__ __host__ inline vec3() : x(0.0f), y(0.0f), z(0.0f) {}
	__device__ __host__ inline vec3(float a, float b, float c) : x(a), y(b), z(c) {}
	__device__ __host__ inline vec3(const float *a) : x(a[0]), y(a[1]), z(a[2]) {}

	__device__ __host__ inline float length() const {
		return sqrt(x * x + y * y + z * z);
	}
	
	__device__ __host__ inline vec3 normalizedOrZero() const {
		float len = length();
		return (len != 0.0f) ? (*this / len) : vec3();
	}

	// unary minus
	__device__ __host__ inline friend vec3 operator-(const vec3& v) { return vec3(-v.x, -v.y, -v.z); }

	__device__ __host__ inline vec3 operator-(const vec3 &other) const { return vec3(x - other.x, y - other.y, z - other.z); }
	__device__ __host__ inline vec3 operator+(const vec3 &other) const { return vec3(x + other.x, y + other.y, z + other.z); }
	__device__ __host__ inline void operator-=(const vec3 &other) { this->x -= other.x; this->y -= other.y; this->z -= other.z; }
	__device__ __host__ inline void operator+=(const vec3 &other) { this->x += other.x; this->y += other.y; this->z += other.z; }

	__device__ __host__ inline vec3 operator/(float a) const { return vec3(x / a, y / a, z / a); }
	__device__ __host__ inline vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
	__device__ __host__ inline friend vec3 operator/(float a, const vec3& other) { return vec3(other.x / a, other.y / a, other.z / a); }
	__device__ __host__ inline friend vec3 operator*(float a, const vec3& other) { return vec3(other.x * a, other.y * a, other.z * a); }
	
	__device__ __host__ inline float operator[](int a) const { return (&x)[a]; }
	__device__ __host__ inline float &operator[](int a) { return (&x)[a]; }
};

__device__ __host__ inline vec3 min(const vec3 &a, const vec3 &b) { return vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
__device__ __host__ inline vec3 max(const vec3 &a, const vec3 &b) { return vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }

__device__ __host__ inline float dot(const vec3 &a, const vec3 &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ inline vec3 cross(const vec3 &a, const vec3 &b) {
	return vec3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

inline std::ostream &operator<<(std::ostream &os, const vec3 &v) {
	return os << '[' << v.x << ", " << v.y << ", " << v.z << ']';
}
