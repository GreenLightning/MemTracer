#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

struct Camera {
	float x = 0, y = 0, z = 0;
	float mat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
	float fov = 60;
};

struct AABB {
	float min[3], max[3];

	AABB() : min{ std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity() }, max{ -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() }
	{}

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

struct Vertex {
	float pos[6];
// 	float nrm[3];

	Vertex() {}
	Vertex(float x, float y, float z) : pos{ x, y, z }
	{}
	Vertex(const Vertex &vtx) : pos{ vtx.pos[0], vtx.pos[1], vtx.pos[2], vtx.pos[3], vtx.pos[4], vtx.pos[5] }
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
};

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<Face> faces;

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

	const float *get_coord(int face, int vertex) const
	{
		return vertices[faces[face].idx[vertex]].pos;
	}
};

Mesh loadMesh(const std::string& name);
