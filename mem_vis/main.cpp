#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "application.hpp"

namespace {
	int framebufferWidth, framebufferHeight;
}

void setSize(GLFWwindow* window) {
	int newFramebufferWidth, newFramebufferHeight;
	glfwGetFramebufferSize(window, &newFramebufferWidth, &newFramebufferHeight);
	if (newFramebufferWidth != framebufferWidth || newFramebufferHeight != framebufferHeight) {
		framebufferWidth = newFramebufferWidth;
		framebufferHeight = newFramebufferHeight;
		appSetSize(window, framebufferWidth, framebufferHeight);
	}
}

void updateAndRender(GLFWwindow* window) {
	static double lastTime = glfwGetTime();
	double currentTime = glfwGetTime();
	float delta = (float) (currentTime - lastTime);
	lastTime = currentTime;

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	appRender(window, delta);
	appRenderGui(window, delta);

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glfwSwapBuffers(window);
}

void windowPosCallback(GLFWwindow* window, int xpos, int ypos) {
	updateAndRender(window);
}

void windowRefreshCallback(GLFWwindow* window) {
	setSize(window);
	updateAndRender(window);
}

void windowKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (ImGui::GetIO().WantCaptureKeyboard) return;
	appKeyCallback(window, key, scancode, action, mods);
}

void windowMouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (ImGui::GetIO().WantCaptureMouse) return;
	appMouseButtonCallback(window, button, action, mods);
}

void windowCursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (ImGui::GetIO().WantCaptureMouse) return;
	appCursorPositionCallback(window, xpos, ypos);
}

int main(int argc, char* argv[]) {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
	// glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

	GLFWwindow* window = glfwCreateWindow(1600, 900, "Memory Trace Visualizer", nullptr, nullptr);
	if (window == nullptr) {
		std::cout << "ERROR: failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		std::cout << "ERROR: failed to initialize GLEW" << std::endl;
		glfwTerminate();
		return -1;
	}

	// Ignore GL_INVALID_ENUM because GLEW still uses a deprecated function.
	// See: https://github.com/nigels-com/glew/issues/3
	GLenum error = glGetError();
	if (error != GL_NO_ERROR && error != GL_INVALID_ENUM) {
		std::cout << "ERROR: OpenGL error after GLEW initialization" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwSetWindowPosCallback(window, windowPosCallback);
	glfwSetWindowRefreshCallback(window, windowRefreshCallback);
	glfwSetKeyCallback(window, windowKeyCallback);
	glfwSetMouseButtonCallback(window, windowMouseButtonCallback);
	glfwSetCursorPosCallback(window, windowCursorPositionCallback);
	glfwSetDropCallback(window, appDropCallback);

	glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

	// glEnable(GL_FRAMEBUFFER_SRGB);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 120");

	appInitialize(window, argc, argv);
	appSetSize(window, framebufferWidth, framebufferHeight);

	glfwFocusWindow(window);

	while(!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		setSize(window);
		updateAndRender(window);
	}

	appTerminate(window);

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwTerminate();

	return 0;
}
