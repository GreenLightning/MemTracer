#include "imgui.h"

#include "gl_utils.hpp"

#include "application.hpp"

namespace {
	int width, height;
}

void appInitialize(GLFWwindow* window, int argc, char* argv[]) {
	GL_CHECK();
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

void appSetSize(GLFWwindow* window, int newWidth, int newHeight) {
	width = newWidth;
	height = newHeight;

	float fWidth = (float) width;
	float fHeight = (float) height;

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
	ImGui::Text("Hello");
}
