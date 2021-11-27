#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

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

struct Ray {
	float org[3];
	float dir[3];
	inline void normalize()
	{
		float n = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
		for (int i = 0; i < 3; ++i) dir[i] /= n;
	}
};

inline bool tri_intersect(float &t, float &uu, float &vv, const Ray &ray, const float *v0, const float *v1, const float *v2)
{
	// from wikipedia moeller trumbore
	const double EPSILON = std::numeric_limits<double>::epsilon();

	double e1x = v1[0] - v0[0], e1y = v1[1] - v0[1], e1z = v1[2] - v0[2];
	double e2x = v2[0] - v0[0], e2y = v2[1] - v0[1], e2z = v2[2] - v0[2];

	double hx = ray.dir[1] * e2z - ray.dir[2] * e2y;
	double hy = ray.dir[2] * e2x - ray.dir[0] * e2z;
	double hz = ray.dir[0] * e2y - ray.dir[1] * e2x;

	double a = e1x * hx + e1y * hy + e1z * hz;
	if (a > -EPSILON && a < EPSILON) return false; // This ray is parallel to this triangle.

	double f = 1.0 / a;
	double sx = ray.org[0] - v0[0], sy = ray.org[1] - v0[1], sz = ray.org[2] - v0[2];
	double u = f * (sx * hx + sy * hy + sz * hz);
	if (u < 0.0 || u > 1.0) return false;

	double qx = sy * e1z - sz * e1y;
	double qy = sz * e1x - sx * e1z;
	double qz = sx * e1y - sy * e1x;

	double v = f * (ray.dir[0] * qx + ray.dir[1] * qy + ray.dir[2] * qz);
	if (v < 0.0 || u + v > 1.0) return false;
	uu = u;
	vv = v;

	// At this stage we can compute t to find out where the intersection point is on the line.
	t = f * (e2x * qx + e2y * qy + e2z * qz);
	if (t > EPSILON) return true; // ray intersection
	else return false; // This means that there is a line intersection but not a ray intersection.
}

struct Vertex {
	float pos[6];
// 	float nrm[3];

	Vertex() {}
	Vertex(float x, float y, float z) : pos{ x, y, z }
	{}
	Vertex(const Vertex &vtx) : pos{ vtx.pos[0], vtx.pos[1], vtx.pos[2], vtx.pos[3], vtx.pos[4], vtx.pos[5] }
	{}
	Vertex(const volatile Vertex &vtx) : pos{ vtx.pos[0], vtx.pos[1], vtx.pos[2], vtx.pos[3], vtx.pos[4], vtx.pos[5] }
	{}
};

struct Face {
	uint32_t idx[3];

	Face()
	{}
	Face(uint32_t a, uint32_t b, uint32_t c) : idx{ a, b, c }
	{}
	Face(const Face &face) : Face(face.idx[0], face.idx[1], face.idx[2])
	{}
	Face(const volatile Face &face) : idx{ face.idx[0], face.idx[1], face.idx[2] }
	{}
// 	Face plusoff(uint32_t off) const
// 	{
// 		return Face{ idx[0] + off, idx[1] + off, idx[2] + off };
// 	}
};

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<Face> faces;
	std::vector<AABB> aabbs;

	void precompute_aabbs()
	{
		aabbs.resize(faces.size());
		for (int i = 0; i < aabbs.size(); ++i) {
			for (int v = 0; v < 3; ++v) {
				aabbs[i].feed(vertices[faces[i].idx[v]].pos);
			}
		}
	}
	void compute_normals()
	{
		for (int i = 0; i < faces.size(); ++i) {
			float ux = get_coord(i, 1)[0] - get_coord(i, 0)[0], uy = get_coord(i, 1)[1] - get_coord(i, 0)[1], uz = get_coord(i, 1)[2] - get_coord(i, 0)[2];
			float vx = get_coord(i, 2)[0] - get_coord(i, 0)[0], vy = get_coord(i, 2)[1] - get_coord(i, 0)[1], vz = get_coord(i, 2)[2] - get_coord(i, 0)[2];

			float nx = -uy * vz + uz * vy;
			float ny = -uz * vx + ux * vz;
			float nz = -ux * vy + uy * vx;
			float nl = std::sqrt(nx * nx + ny * ny + nz * nz);
			if (nl != 0.f) { nx /= nl; ny /= nl; nz /= nl; }

			for (int j = 0; j < 3; ++j) {
				vertices[faces[i].idx[j]].pos[3] = nx;
				vertices[faces[i].idx[j]].pos[4] = ny;
				vertices[faces[i].idx[j]].pos[5] = nz;
			}
		}
	}

	AABB compute_aabb() const
	{
		AABB aabb;
		for (int i = 0; i < vertices.size(); ++i) {
			aabb.feed(vertices[i].pos);
		}
		return aabb;
	}

	std::size_t size_vertices() const
	{
		return vertices.size();
	}
	std::size_t size() const
	{
		return faces.size();
	}
	float get_bounding_min(int axis, uint32_t idx) const
	{
		return aabbs[idx].min[axis];
	}
	float get_bounding_max(int axis, uint32_t idx) const
	{
		return aabbs[idx].max[axis];
	}

	const float *get_coord(int face, int edge) const
	{
		return vertices[faces[face].idx[edge]].pos;
	}

	bool intersect(float &t, void *data, uint32_t idx, const Ray &ray) const
	{
		float u, v;
		if (!tri_intersect(t, u, v, ray, get_coord(idx, 0), get_coord(idx, 1), get_coord(idx, 2))) return false;

		float *d = (float*)data;
		for (int i = 0; i < 6; ++i) { // NOTE: n comes after v in memory
			d[i] = vertices[faces[idx].idx[0]].pos[i] * (1.f - u - v) + vertices[faces[idx].idx[1]].pos[i] * u + vertices[faces[idx].idx[2]].pos[i] * v;
		}

		return true;
	}
};

