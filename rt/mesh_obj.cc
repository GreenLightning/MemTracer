#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "mesh.h"

Mesh loadMeshObj(const std::string& name) {
	Mesh mesh;

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warning, error;

	bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warning, &error, name.c_str(), nullptr, false);
	if (!ok) throw std::runtime_error(warning + error);

	mesh.vertices.reserve(attrib.vertices.size() / 3);
	for (size_t i = 0; i + 3 <= attrib.vertices.size(); i += 3) {
		auto x = attrib.vertices[i + 0];
		auto y = attrib.vertices[i + 1];
		auto z = attrib.vertices[i + 2];
		mesh.vertices.emplace_back(x, y, z);
	}

	for (const auto& shape : shapes) { // for each shape
		size_t indexBase = 0;
		for (const auto vertexCountForCurrentFace : shape.mesh.num_face_vertices) { // for each face
			// Perform basic triangulation for faces with more than 3 vertices.
			for (size_t i = 1; i + 1 < vertexCountForCurrentFace; i++) { // for each vertex
				auto a = static_cast<uint32_t>(shape.mesh.indices[indexBase + 0  ].vertex_index);
				auto b = static_cast<uint32_t>(shape.mesh.indices[indexBase + i  ].vertex_index);
				auto c = static_cast<uint32_t>(shape.mesh.indices[indexBase + i+1].vertex_index);
				mesh.faces.emplace_back(a, b, c);
			}
			indexBase += vertexCountForCurrentFace;
		}
	}

	return mesh;
}
