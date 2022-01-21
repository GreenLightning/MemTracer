#include "toml.hpp"

#include "config.h"

float as_float(const toml::value& value) {
	if (value.is_integer()) return static_cast<float>(value.as_integer(std::nothrow));
	return static_cast<float>(value.as_floating());
}

Camera loadCamera(const toml::value& camera) {
	Camera cam;

	if (camera.contains("position")) {
		auto position = toml::find(camera, "position");
		if (!position.is_array()) throw std::runtime_error(toml::format_error("[error] expected array", position, "here"));
		if (position.size() != 3) throw std::runtime_error(toml::format_error("[error] position should have three elements", position, "here"));
		cam.x = as_float(toml::find(position, 0));
		cam.y = as_float(toml::find(position, 1));
		cam.z = as_float(toml::find(position, 2));
	}

	if (camera.contains("matrix")) {
		auto matrix = toml::find(camera, "matrix");
		if (!matrix.is_array()) throw std::runtime_error(toml::format_error("[error] expected array", matrix, "here"));
		if (matrix.size() != 9) throw std::runtime_error(toml::format_error("[error] matrix should have 9 elements", matrix, "here"));
		for (int i = 0; i < 9; i++) {
			cam.mat[i] = as_float(toml::find(matrix, i));
		}
	}

	if (camera.contains("vertical_fov")) {
		cam.vertical_fov = as_float(toml::find(camera, "vertical_fov"));
	}

	return cam;
}

void loadConfiguration(Configuration& config, const std::string& path) {
	auto data = toml::parse(path);

	std::string basePath;
	auto index = path.rfind("/");
	if (index != std::string::npos) {
		basePath = path.substr(0, index+1);
	}
	#ifdef _WIN32
	{
		auto index2 = path.rfind("\\");
		if (index2 != std::string::npos && (index == std::string::npos || index2 > index)) {
			basePath = path.substr(0, index2+1);
		}
	}
	#endif

	if (data.contains("input")) {
		std::string value = toml::find<std::string>(data, "input");
		if (!value.empty()) {
			if (value[0] != '/') value = basePath + value;
			config.input = value;
		}
	}

	if (data.contains("output")) {
		std::string value = toml::find<std::string>(data, "output");
		if (!value.empty()) {
			if (value[0] != '/') value = basePath + value;
			config.output = value;
		}
	}

	if (data.contains("width")) config.width = toml::find<int32_t>(data, "width");
	if (data.contains("height")) config.height = toml::find<int32_t>(data, "height");

	if (data.contains("size")) {
		auto size = toml::find(data, "size");
		if (!size.is_array()) throw std::runtime_error(toml::format_error("[error] expected array", size, "here"));
		if (size.size() != 2) throw std::runtime_error(toml::format_error("[error] size should have two elements", size, "here"));
		config.width = toml::find<int32_t>(size, 0);
		config.height = toml::find<int32_t>(size, 1);
	}

	if (data.contains("shading")) {
		auto shading = toml::find(data, "shading");
		std::string value = shading.as_string();
		if (value == "smooth") {
			config.shading = SMOOTH;
		} else if (value == "flat") {
			config.shading = FLAT;
		} else {
			throw std::runtime_error(toml::format_error("[error] unknown value for shading", shading, "here"));
		}
	}

	if (data.contains("shadows")) {
		config.shadows = toml::find<bool>(data, "shadows");
	}

	if (data.contains("heuristic")) {
		auto heuristic = toml::find(data, "heuristic");
		std::string value = heuristic.as_string();
		if (!validateHeuristic(value)) {
			throw std::runtime_error(toml::format_error("[error] unknown value for heuristic", heuristic, "here"));
		}
		config.heuristic = value;
	}

	if (data.contains("camera")) {
		auto camera = toml::find(data, "camera");
		if (camera.is_table()) {
			Camera cam = loadCamera(camera);
			config.cameras.push_back(cam);
		} else {
			const auto cameras = toml::get<std::vector<toml::table>>(camera);
			for (const auto& camera : cameras) {
				Camera cam = loadCamera(camera);
				config.cameras.push_back(cam);
			}
		}
	}

	if (data.contains("light")) {
		auto light = toml::find(data, "light");

		if (light.contains("position")) {
			auto position = toml::find(light, "position");
			if (!position.is_array()) throw std::runtime_error(toml::format_error("[error] expected array", position, "here"));
			if (position.size() != 3) throw std::runtime_error(toml::format_error("[error] position should have three elements", position, "here"));
			config.light.x = toml::find<float>(position, 0);
			config.light.y = toml::find<float>(position, 1);
			config.light.z = toml::find<float>(position, 2);
		}
	}
}
