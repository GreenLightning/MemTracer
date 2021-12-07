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

GLuint createShader(GLenum type, std::string filename) {
	GLuint shader = glCreateShader(type);

	std::ifstream file(filename);
	if (!file) {
		std::cout << "ERROR: failed to open shader file: " << filename << std::endl;
		return shader;
	}

	file.seekg(0, std::ios::end);
	size_t size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> buffer(size + 1);
	file.read(buffer.data(), size);

	buffer.resize(file.gcount() + 1);
	buffer[buffer.size() - 1] = '\0';

	if (file.bad()) {
		std::cout << "ERROR: failed to load shader file: " << filename << std::endl;
		return shader;	
	}

	GLchar* source = buffer.data();
	glShaderSource(shader, 1, &source, nullptr);
	glCompileShader(shader);

	GLint success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		GLchar infoLog[1024];
		glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
		std::cout << "ERROR: shader compilation failed: " << filename << std::endl;
		std::cout << infoLog << std::endl;
	}

	return shader;
}

GLuint createProgram(std::string name) {
	GLuint program = glCreateProgram();

	GLuint vertexShader = createShader(GL_VERTEX_SHADER, "shaders/" + name + ".vert");
	GLuint fragmentShader = createShader(GL_FRAGMENT_SHADER, "shaders/" + name + ".frag");

	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
	glLinkProgram(program);

	GLint success;
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if(!success) {
		GLchar infoLog[1024];
		glGetProgramInfoLog(program, 1024, nullptr, infoLog);
		std::cout << "ERROR: program linking failed: " << name << std::endl;
		std::cout << infoLog << std::endl;
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	return program;
}

void ImGuiFullWidthImage(GLuint texture, float aspect /* width / height */) {
	float width = ImGui::GetWindowWidth() - 25.0f;
	float height = width / aspect;
	ImGui::Image((void*)(intptr_t)texture, ImVec2(width, height));
}
