#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <imgui.h>

#include <defer.hpp>
#include <mio.hpp>

#include "meow_hash_x64_aesni.h"

#include "common.h"
#include "gl_utils.hpp"

#include "application.hpp"

struct Application {
	int width, height;

	std::string filename;
	std::string error;
	mio::mmap_source mmap;

	bool load(std::string filename) {
		this->filename = filename;
		this->error = "";
		this->mmap.unmap();

		defer {
			if (!this->error.empty()) this->mmap.unmap();
		};

		std::error_code err_code;
		this->mmap.map(filename, err_code);
		if (err_code) {
			this->error = err_code.message(); return false;
		}

		if (this->mmap.size() == 0) {
			this->error = "empty file"; return false;
		}

		if (this->mmap.size() < 32) {
			this->error = "file too short"; return false;
		}

		header_t header;
		memcpy(&header, this->mmap.data(), 32);

		if (header.magic != (('T' << 0) | ('R' << 8) | ('A' << 16) | ('C' << 24))) {
			this->error = "invalid file (magic)"; return false;
		}

		if (header.version != 1) {
			this->error = "file version is not supported"; return false;
		}

		if (header.header_size < 32 || header.header_size > sizeof(header_t)) {
			this->error = "invalid file (header size)"; return false;
		}

		memcpy(&header, this->mmap.data(), header.header_size);

		meow_u128 expectedHash;
		memcpy(&expectedHash, &header.hash, 128 / 8);
		memset(&header.hash, 0, 128 / 8);

		meow_state hashState;
		MeowBegin(&hashState, MeowDefaultSeed);
		MeowAbsorb(&hashState, this->mmap.size() - header.header_size, &this->mmap[header.header_size]);
		MeowAbsorb(&hashState, header.header_size, &header);
		meow_u128 actualHash = MeowEnd(&hashState, nullptr);

		int match = MeowHashesAreEqual(actualHash, expectedHash);
		if (!match) {
			this->error = "invalid file (hash)"; return false;
		}

		return true;
	}

};

namespace {
	Application app;
}

void appInitialize(GLFWwindow* window, int argc, char* argv[]) {
	if (argc > 2) {
		fprintf(stderr, "too many arguments\n");
		fprintf(stderr, "usage: <trace-filename>\n");
		exit(1);
	}

	if (argc == 2) {
		bool ok = app.load(argv[1]);
		if (!ok) {
			fprintf(stderr, "failed to load %s: %s\n", app.filename.c_str(), app.error.c_str());
			exit(1);
		}
	}
}

void appTerminate(GLFWwindow* window) {
}

void appKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

void appMouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {}

void appCursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {}

void appSetSize(GLFWwindow* window, int width, int height) {
	app.width = width;
	app.height = height;
	glViewport(0, 0, width, height);
}

void appRender(GLFWwindow* window, float delta) {
	// UPDATE

	// RENDERING

	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	GL_CHECK();
}

void appRenderGui(GLFWwindow* window, float delta) {
	ImGui::Text(app.filename.c_str());
}
