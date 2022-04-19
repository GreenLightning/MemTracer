#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include "vec.h"
#include "cuda.h"

struct Vertex {
	vec3 position;
	vec3 normal;

	__device__ __host__ Vertex() {}
	__device__ __host__ Vertex(float x, float y, float z) : position{ x, y, z } {}
	__device__ __host__ Vertex(const Vertex &vtx) : position{ vtx.position }, normal{ vtx.normal } {}
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

	void compute_normals();
};

Mesh loadMesh(const std::string& name);
Mesh loadMeshPly(const std::string& name);
Mesh loadMeshObj(const std::string& name);
