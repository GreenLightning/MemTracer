#include "toml.hpp"

#include "config.h"

void loadConfiguration(Configuration& config, const std::string& path) {
	auto data = toml::parse(path);

	std::string basePath;
	auto index = path.rfind("/");
	if (index != std::string::npos) {
		basePath = path.substr(0, index+1);
	}

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

	if (data.contains("heuristic")) {
		auto heuristic = toml::find(data, "heuristic");
		std::string value = heuristic.as_string();
		if (value == "sah") {
			config.heuristic = SAH;
		} else if (value == "median") {
			config.heuristic = MEDIAN;
		} else {
			throw std::runtime_error(toml::format_error("[error] unknown value for heuristic", heuristic, "here"));
		}
	}

	if (data.contains("camera")) {
		auto camera = toml::find(data, "camera");

		if (camera.contains("position")) {
			auto position = toml::find(camera, "position");
			if (!position.is_array()) throw std::runtime_error(toml::format_error("[error] expected array", position, "here"));
			if (position.size() != 3) throw std::runtime_error(toml::format_error("[error] position should have three elements", position, "here"));
			config.camera.x = toml::find<float>(position, 0);
			config.camera.y = toml::find<float>(position, 1);
			config.camera.z = toml::find<float>(position, 2);
		}

		if (camera.contains("vertical_fov")) {
			auto value = toml::find(camera, "vertical_fov");
			config.camera.fov = value.is_floating() ? value.as_floating(std::nothrow) : static_cast<double>(value.as_integer());
		}
	}
}
