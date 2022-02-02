#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <GL/glew.h>

#include "imgui.h"

#include "gl_utils.hpp"

#ifdef _DEBUG

void glCheckError(const char* filename, int line) {
	using namespace std;

	GLenum error = glGetError();
	if (error == GL_NO_ERROR) return;

	const char* name = "unknown error";
	switch (error) {
		case GL_INVALID_ENUM:                  name = "invalid enum";                  break;
		case GL_INVALID_VALUE:                 name = "invalid value";                 break;
		case GL_INVALID_OPERATION:             name = "invalid operation";             break;
		case GL_INVALID_FRAMEBUFFER_OPERATION: name = "invalid framebuffer operation"; break;
		case GL_OUT_OF_MEMORY:                 name = "out of memory";                 break;
		#if defined(ENGINE_WINDOWS) || defined(ENGINE_OSX)
			case GL_STACK_UNDERFLOW:           name = "stack underflow";               break;
			case GL_STACK_OVERFLOW:            name = "stack overflow";                break;
		#endif
	}

	ios::fmtflags formatFlags = cout.flags();
	cout << "ERROR: OpenGL error 0x" << hex << setfill('0') << setw(8) << error << ": " << name << endl;
	cout.flags(formatFlags);

	cout << filename << "(" << line << "): " << "detected here" << endl;
}

#endif // _DEBUG

void compile_shader(GLuint shader, const GLchar* source, const char* name) {
	GLint success;
	glShaderSource(shader, 1, &source, nullptr);
	glCompileShader(shader);
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		GLchar infoLog[1024];
		glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
		std::cout << "ERROR: shader compilation failed: " << name << std::endl;
		std::cout << infoLog << std::endl;
	}
}

void link_program(GLuint program, const char* name) {
	GLint success;
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if(!success) {
		GLchar infoLog[1024];
		glGetProgramInfoLog(program, 1024, nullptr, infoLog);
		std::cout << "ERROR: program linking failed: " << name << std::endl;
		std::cout << infoLog << std::endl;
	}
}
