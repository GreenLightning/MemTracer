#include <cstdint>
#include <string>
#include <vector>

#include <happly.h>

#include "types.h"

Mesh loadMesh(const std::string& name) {
	Mesh mesh;

	happly::PLYData data(name);

	auto& vertex = data.getElement("vertex");
	std::vector<float> x = vertex.getProperty<float>("x");
	std::vector<float> y = vertex.getProperty<float>("y");
	std::vector<float> z = vertex.getProperty<float>("z");

	mesh.vertices.reserve(x.size());
	for (int i = 0; i < x.size(); i++) {
		mesh.vertices.emplace_back(x[i], y[i], z[i]);
	}

	std::vector<std::vector<size_t>> indicesList = data.getFaceIndices<size_t>();
	for (auto& indices : indicesList) {
		// Perform basic triangulation for faces with more than 3 vertices.
		for (int i = 1; i + 1 < indices.size(); i++) {
			mesh.faces.emplace_back(indices[0], indices[i], indices[i+1]);
		}
	}

	return mesh;
}
