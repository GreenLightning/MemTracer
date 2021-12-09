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
	vec3 position;
	vec3 normal;

	Vertex() {}
	Vertex(float x, float y, float z) : position{ x, y, z } {}
	Vertex(const Vertex &vtx) : position{ vtx.position }, normal{ vtx.normal } {}
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
			Face face = faces[i];

			const vec3& a = vertices[face.idx[0]].position;
			const vec3& b = vertices[face.idx[1]].position;
			const vec3& c = vertices[face.idx[2]].position;

			vec3 e1 = b - a;
			vec3 e2 = c - a;

			vec3 normal = -cross(e1, e2);
			float length = std::sqrt(dot(normal, normal));
			if (length != 0.0f) normal = normal / length;

			vertices[faces[i].idx[0]].normal = normal;
			vertices[faces[i].idx[1]].normal = normal;
			vertices[faces[i].idx[2]].normal = normal;
		}
	}

	const vec3& get_coord(int face, int vertex) const {
		return vertices[faces[face].idx[vertex]].position;
	}
};

Mesh loadMesh(const std::string& name);
