#pragma once

#include <string>

#include <GL/glew.h>

#ifdef _DEBUG
	void glCheckError(const char* filename, int line);
	#define GL_CHECK() glCheckError(__FILE__, __LINE__)
#else
	#define GL_CHECK() ((void)0)
#endif

GLuint createProgram(std::string name);

void ImGuiFullWidthImage(GLuint texture, float aspect /* width / height */);
