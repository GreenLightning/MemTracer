#pragma once

#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

void appInitialize(GLFWwindow* window, int argc, char* argv[]);
void appTerminate(GLFWwindow* window);

// event callbacks
void appKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void appMouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void appCursorPositionCallback(GLFWwindow* window, double xpos, double ypos);

// width and height in pixels
void appSetSize(GLFWwindow* window, int width, int height);
void appRender(GLFWwindow* window, float delta);
void appRenderGui(GLFWwindow* window, float delta);
