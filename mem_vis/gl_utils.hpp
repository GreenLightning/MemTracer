#pragma once

#include <glad/glad.h>

#ifdef _DEBUG
	void glCheckError(const char* filename, int line);
	#define GL_CHECK() glCheckError(__FILE__, __LINE__)
#else
	#define GL_CHECK() ((void)0)
#endif

void compile_shader(GLuint shader, const GLchar* source, const char* name);
void link_program(GLuint program, const char* name);
