#pragma once

#include <algorithm>
#include <ostream>

struct vec3 {
	float x, y, z;
	inline vec3() {}
	inline vec3(float a, float b, float c) : x(a), y(b), z(c) {}
	inline vec3(const float *a) : x(a[0]), y(a[1]), z(a[2]) {}
	inline vec3 operator-(const vec3 &other) { return vec3(x - other.x, y - other.y, z - other.z); }
	inline vec3 operator+(const vec3 &other) { return vec3(x + other.x, y + other.y, z + other.z); }
	inline vec3 operator/(float a) { return vec3(x / a, y / a, z / a); }
	inline vec3 operator*(float a) { return vec3(x * a, y * a, z * a); }
	inline float operator[](int a) const { return (&x)[a]; }
	inline float &operator[](int a) { return (&x)[a]; }
};

inline vec3 min(const vec3 &a, const vec3 &b) { return vec3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)); }
inline vec3 max(const vec3 &a, const vec3 &b) { return vec3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)); }

inline float dot(const vec3 &a, const vec3 &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline vec3 cross(const vec3 &a, const vec3 &b) {
	return vec3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

std::ostream &operator<<(std::ostream &os, const vec3 &v) {
	return os << '[' << v.x << ", " << v.y << ", " << v.z << ']';
}
