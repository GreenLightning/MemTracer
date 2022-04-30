#pragma once

#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

void appParseArguments(int argc, char* argv[]);
void appInitialize(GLFWwindow* window);
void appTerminate(GLFWwindow* window);

// event callbacks
void appKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void appMouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void appCursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
void appDropCallback(GLFWwindow* window, int count, const char** paths);

// width and height in pixels
void appSetSize(GLFWwindow* window, int width, int height);
void appRender(GLFWwindow* window, float delta);
void appRenderGui(GLFWwindow* window, float delta);
