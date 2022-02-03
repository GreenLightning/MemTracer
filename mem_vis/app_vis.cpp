const char* mesh_vertex_shader_source = R"DONE(
#version 330

uniform mat4 projection;
uniform mat4 model;
uniform mat4 view;

layout (location = 0) in vec4 position;

out VDATA {
	vec3 position;
} vdata;

void main() {
	gl_Position = projection * view * model * position;
	vdata.position = vec3(view * model * position);
}
)DONE";

const char* mesh_geometry_shader_source = R"DONE(
#version 330

layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

in VDATA {
	vec3 position;
} vdata[];

out vec3 normal;

void main() {
	vec3 a = vdata[1].position - vdata[0].position;
	vec3 b = vdata[2].position - vdata[0].position;
	vec3 n = normalize(cross(a, b));

	for (int i = 0; i < gl_in.length(); i++) {
		gl_Position = gl_in[i].gl_Position;
		normal = n;
		EmitVertex();
	}
	EndPrimitive();
}
)DONE";

const char* mesh_fragment_shader_source = R"DONE(
#version 330

layout (location = 0) out vec3 color;

in vec3 normal;

void main() {
	color = vec3(0.5, 0.5, 0.5) + 0.5 * normal;
	// color = vec3(gl_FragCoord.z);
}
)DONE";

struct Face {
	uint32_t idx[3] = {0, 0, 0};

	friend bool operator==(const Face& lhs, const Face& rhs) {
		return lhs.idx[0] == rhs.idx[0] && lhs.idx[1] == rhs.idx[1] && lhs.idx[2] == rhs.idx[2];
	}
};

template<>
struct std::hash<Face> {
	std::size_t operator()(Face const& key) const noexcept {
		uint64_t a = 7ull * key.idx[0];
		uint64_t b = 65867ull * key.idx[1];
		uint64_t c = 4294969111ull * key.idx[2];
		return a ^ b ^ c;
	}
};

void matrix_multiply(float* result, float* a, float* b) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			float v = 0;
			for (int k = 0; k < 4; k++) {
				v += a[4*i + k] * b[4*k + j];
			}
			result[4*i + j] = v;
		}
	}
}

struct Visualizer {
	bool culling = true;

	bool dragging = false;
	float horizontalAngle = 0;
	float verticalAngle = 0;

	GLuint program = 0;

	GLuint vao = 0;
	GLuint vbo = 0;
	GLuint ebo = 0;
	GLuint indices = 0;

	float minX = 0.0f, minY = 0.0f, minZ = 0.0f;
	float maxX = 0.0f, maxY = 0.0f, maxZ = 0.0f;
	float centerX = 0.0f, centerY = 0.0f, centerZ = 0.0f;

	bool framebufferInit = false, framebufferSuccess = false;
	int width = 1024, height = 768;
	GLuint framebuffer = 0;
	GLuint depthbuffer = 0;
	GLuint texture = 0;

	void initShader() {
		GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
		compile_shader(vertex_shader, mesh_vertex_shader_source, "vertex shader");
		
		GLuint geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);
		compile_shader(geometry_shader, mesh_geometry_shader_source, "geometry shader");
		
		GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
		compile_shader(fragment_shader, mesh_fragment_shader_source, "fragment shader");

		program = glCreateProgram();
		glAttachShader(program, vertex_shader);
		glAttachShader(program, geometry_shader);
		glAttachShader(program, fragment_shader);
		link_program(program, "mesh program");

		glDeleteShader(vertex_shader);
		glDeleteShader(geometry_shader);
		glDeleteShader(fragment_shader);

		GL_CHECK();
	}

	void extractMesh(Trace* trace) {
		// Check if trace contains memory contents.
		if (!trace->header.mem_contents_offset) return;
	
		TraceRegion* vertex_region = trace->find_region(0, 4);
		TraceRegion* face_region = trace->find_region(0, 3);

		Face* faces = (Face*) &trace->mmap[trace->header.mem_contents_offset + face_region->contents_offset];
		uint64_t face_count = face_region->size / sizeof(Face);

		std::vector<Face> filteredFaces;
		std::unordered_set<Face> seenFaces;

		for (uint64_t i = 0; i < face_count; i++) {
			Face f = faces[i];
			if (f.idx[0] == 0 && f.idx[1] == 0 && f.idx[2] == 0) continue;
			if (seenFaces.count(f)) continue;
			filteredFaces.push_back(f);
			seenFaces.insert(f);
		}

		this->indices = (GLuint) (3 * filteredFaces.size());

		float* vertex_data = (float*) &trace->mmap[trace->header.mem_contents_offset + vertex_region->contents_offset];
		uint64_t vertex_count = vertex_region->size / (3 * sizeof(float));
		minX = maxX = vertex_data[0];
		minY = maxY = vertex_data[1];
		minZ = maxZ = vertex_data[2];
		for (uint64_t i = 0; i < vertex_count; i++) {
			float x = vertex_data[3*i + 0];
			float y = vertex_data[3*i + 1];
			float z = vertex_data[3*i + 2];
			if (x < minX) minX = x;
			if (x > maxX) maxX = x;
			if (y < minY) minY = y;
			if (y > maxY) maxY = y;
			if (z < minZ) minZ = z;
			if (z > maxZ) maxZ = z;
			centerX += x;
			centerY += y;
			centerZ += z;
		}

		centerX /= vertex_count;
		centerY /= vertex_count;
		centerZ /= vertex_count;

		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
		glGenBuffers(1, &ebo);

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertex_region->size, vertex_data, GL_STATIC_DRAW);  

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, filteredFaces.size() * sizeof(Face), filteredFaces.data(), GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, (void*)0);

		glBindVertexArray(0);
	}

	void initFramebuffer() {
		framebufferInit = true;

		glGenFramebuffers(1, &framebuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

		glGenRenderbuffers(1, &depthbuffer);
		glBindRenderbuffer(GL_RENDERBUFFER, depthbuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthbuffer);

		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);

		GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
		glDrawBuffers(sizeof(drawBuffers)/sizeof(drawBuffers[0]), drawBuffers);

		framebufferSuccess = (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void renderToTexture() {
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

		glViewport(0, 0, width, height);
		if (dragging) {
			glClearColor(0.3f, 0.8f, 0.2f, 1.0f);
		} else {
			glClearColor(0.0f, 0.7f, 0.3f, 1.0f);
		}
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (culling) {
			glEnable(GL_CULL_FACE);
			glCullFace(GL_BACK);
		} else {
			glDisable(GL_CULL_FACE);
		}

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);

		if (vao) {
			glUseProgram(program);

			float vfov = 60.0f / 180.0f * IM_PI;
			float pos[3] = {0.0f, 0.0f, 0.2f};

			float v = std::tan(0.5f * vfov);
			float h = v * static_cast<float>(width) / static_cast<float>(height);
			float n = 0.05f, f = 1.0f;

			float projection[16] = {
				1/h,   0,  0, 0,
				  0, 1/v,  0, 0,
				  0,   0, -(f + n) / (f - n), -2.0f * f * n / (f - n),
				  0,   0, -1, 0,
			};

			float view[16] = {
				1, 0, 0, -pos[0],
				0, 1, 0, -pos[1],
				0, 0, 1, -pos[2],
				0, 0, 0, 1,
			};

			float s = std::sin(horizontalAngle);
			float c = std::cos(horizontalAngle);
			float horizontal[16] = {
				 c, 0, s, 0,
				 0, 1, 0, 0,
				-s, 0, c, 0,
				 0, 0, 0, 1,
			};

			s = std::sin(verticalAngle);
			c = std::cos(verticalAngle);
			float vertical[16] = {
				1, 0,  0, 0,
				0, c, -s, 0,
				0, s,  c, 0,
				0, 0,  0, 1,
			};

			float rotation[16];
			matrix_multiply(rotation, horizontal, vertical);

			float translation[16] = {
				1, 0, 0, -centerX,
				0, 1, 0, -centerY,
				0, 0, 1, -centerZ,
				0, 0, 0, 1,
			};

			float model[16];
			matrix_multiply(model, rotation, translation);

			GLint location;
			location = glGetUniformLocation(program, "projection");
			glUniformMatrix4fv(location, 1, GL_TRUE, projection);
			location = glGetUniformLocation(program, "view");
			glUniformMatrix4fv(location, 1, GL_TRUE, view);
			location = glGetUniformLocation(program, "model");
			glUniformMatrix4fv(location, 1, GL_TRUE, model);

			glBindVertexArray(vao);
			glDrawElements(GL_TRIANGLES, indices, GL_UNSIGNED_INT, 0);
			glBindVertexArray(0);

			glUseProgram(0);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		GL_CHECK();
	}

	void renderGui(Trace* trace) {
		if (!program) {
			initShader();
		}

		if (trace && !vao) {
			extractMesh(trace);
		}

		if (!framebufferInit) {
			initFramebuffer();
		}

		ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		if (ImGui::Begin("Visualizer", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
			ImGui::Checkbox("Culling", &culling);

			ImVec2 avail = ImGui::GetContentRegionAvail();
			ImVec2 size(avail.x, avail.x * static_cast<float>(height) / static_cast<float>(width));

			if (dragging) {
				if (!ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
					dragging = false;
				}
			} else {
				ImVec2 cursorPos = ImGui::GetCursorScreenPos();
				ImRect bb(cursorPos, ImVec2(cursorPos.x + size.x, cursorPos.y + size.y));
				ImGuiID id = ImGui::GetID("Texture");
				if (ImGui::ButtonBehavior(bb, id, nullptr, nullptr, ImGuiButtonFlags_PressedOnClick)) {
					dragging = true;
				}
			}

			if (dragging) {
				ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0f);
				ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
				horizontalAngle += delta.x / 200.0f;
				verticalAngle += delta.y / 200.0f;
				if (verticalAngle < -IM_PI) {
					verticalAngle = -IM_PI;
				} else if (verticalAngle > IM_PI) {
					verticalAngle = IM_PI;
				}
			}

			if (framebufferSuccess) {
				renderToTexture();
			}

			// Flip uv coordinates to convert from OpenGL (y up) to ImGui (y down) convention.
			ImGui::Image((void*)(intptr_t)this->texture, size, ImVec2(0, 1), ImVec2(1, 0));
		}
		ImGui::PopStyleVar();
		ImGui::End();
	}
};
