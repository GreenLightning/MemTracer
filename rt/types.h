#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include "vec.h"
#include "cuda.h"

struct Camera {
	float x = 0, y = 0, z = 0;
	float mat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
	float vertical_fov = 60;
};

struct Ray {
	vec3 origin;
	vec3 dir;
};

struct Vertex {
	float pos[6];
// 	float nrm[3];

	Vertex() {}
	Vertex(float x, float y, float z) : pos{ x, y, z } {}
	Vertex(const Vertex &vtx) : pos{ vtx.pos[0], vtx.pos[1], vtx.pos[2], vtx.pos[3], vtx.pos[4], vtx.pos[5] } {}
};

struct Face {
	uint32_t idx[3];

	__device__ __host__ Face() {}
	__device__ __host__ Face(uint32_t a, uint32_t b, uint32_t c) : idx{ a, b, c } {}
	__device__ __host__ Face(const Face &face) : Face(face.idx[0], face.idx[1], face.idx[2]) {}
};

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<Face> faces;

	void compute_normals() {
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

	const float *get_coord(int face, int vertex) const {
		return vertices[faces[face].idx[vertex]].pos;
	}
};

Mesh loadMesh(const std::string& name);
