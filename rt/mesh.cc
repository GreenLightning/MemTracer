#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "mesh.h"

void Mesh::compute_normals() {
	for (int i = 0; i < vertices.size(); i++) {
		vertices[i].normal = vec3();
	}

	for (int i = 0; i < faces.size(); i++) {
		Face face = faces[i];

		const vec3& a = vertices[face.idx[0]].position;
		const vec3& b = vertices[face.idx[1]].position;
		const vec3& c = vertices[face.idx[2]].position;

		vec3 e1 = b - a;
		vec3 e2 = c - a;

		vec3 normal = cross(e1, e2).normalizedOrZero();

		vertices[faces[i].idx[0]].normal += normal;
		vertices[faces[i].idx[1]].normal += normal;
		vertices[faces[i].idx[2]].normal += normal;
	}

	for (int i = 0; i < vertices.size(); i++) {
		vertices[i].normal = vertices[i].normal.normalizedOrZero();
	}
}

Mesh loadMesh(const std::string& name) {
	try {
		auto index = name.rfind(".");
		if (index == std::string::npos) throw std::runtime_error("missing file extension");
		auto extension = name.substr(index+1);

		if (extension == "ply") {
			return loadMeshPly(name);
		}
		if (extension == "obj") {
			return loadMeshObj(name);
		}

		throw std::runtime_error("unrecognized file extension: " + extension);
	} catch (const std::runtime_error& e) {
		throw std::runtime_error("failed to load mesh: " + name + ": " + e.what());
	}
}
