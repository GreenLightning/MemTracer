// MESH SHADER

const char* mesh_vertex_shader_source = R"DONE(
#version 330

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

layout (location = 0) in vec3 position;

out VDATA {
	vec3 position;
} vdata;

void main() {
	gl_Position = projection * view * model * vec4(position, 1);
	vdata.position = vec3(view * model * vec4(position, 1));
}
)DONE";

const char* mesh_geometry_shader_source = R"DONE(
#version 330

layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

in VDATA {
	vec3 position;
} vdata[];

out vec3 position;
out vec3 normal;

void main() {
	vec3 a = vdata[1].position - vdata[0].position;
	vec3 b = vdata[2].position - vdata[0].position;
	vec3 n = normalize(cross(a, b));

	for (int i = 0; i < gl_in.length(); i++) {
		gl_Position = gl_in[i].gl_Position;
		position = vdata[i].position;
		normal = n;
		EmitVertex();
	}
	EndPrimitive();
}
)DONE";

const char* mesh_fragment_shader_source = R"DONE(
#version 330

uniform int modelColorMode;

layout (location = 0) out vec3 color;

in vec3 position;
in vec3 normal;

void main() {
	if (modelColorMode == 1) {
		vec3 l = normalize(-position);
		vec3 n = normalize(normal);
		float d = max(dot(l, n), 0.0);
		float v = 0.05 + 0.95 * d;
		v = clamp(v, 0.0, 1.0);
		color = vec3(v, v, v);
	} else {
		color = vec3(0.5, 0.5, 0.5) + 0.5 * normal;
	}
}
)DONE";

// BOX SHADER

const char* box_vertex_shader_source = R"DONE(
#version 330

layout (location = 0) in vec3 min;
layout (location = 1) in vec3 max;
layout (location = 2) in vec3 color;

out VDATA {
	vec3 min;
	vec3 max;
	vec3 color;
} vdata;

void main() {
	vdata.min = min;
	vdata.max = max;
	vdata.color = color;
}
)DONE";

const char* box_geometry_shader_source = R"DONE(
#version 330

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

layout(points) in;
layout(line_strip, max_vertices=24) out;

in VDATA {
	vec3 min;
	vec3 max;
	vec3 color;
} vdata[];

out vec3 color;

void main() {
	mat4 pvm = projection * view * model;

	vec4 p0 = pvm * vec4(vdata[0].min.x, vdata[0].min.y, vdata[0].min.z, 1);
	vec4 p1 = pvm * vec4(vdata[0].min.x, vdata[0].min.y, vdata[0].max.z, 1);
	vec4 p2 = pvm * vec4(vdata[0].min.x, vdata[0].max.y, vdata[0].min.z, 1);
	vec4 p3 = pvm * vec4(vdata[0].min.x, vdata[0].max.y, vdata[0].max.z, 1);
	vec4 p4 = pvm * vec4(vdata[0].max.x, vdata[0].min.y, vdata[0].min.z, 1);
	vec4 p5 = pvm * vec4(vdata[0].max.x, vdata[0].min.y, vdata[0].max.z, 1);
	vec4 p6 = pvm * vec4(vdata[0].max.x, vdata[0].max.y, vdata[0].min.z, 1);
	vec4 p7 = pvm * vec4(vdata[0].max.x, vdata[0].max.y, vdata[0].max.z, 1);

	vec3 vc = vdata[0].color;

	gl_Position = p0; color = vc; EmitVertex();
	gl_Position = p1; color = vc; EmitVertex();
	gl_Position = p3; color = vc; EmitVertex();
	gl_Position = p2; color = vc; EmitVertex();
	gl_Position = p0; color = vc; EmitVertex();
	EndPrimitive();

	gl_Position = p4; color = vc; EmitVertex();
	gl_Position = p5; color = vc; EmitVertex();
	gl_Position = p7; color = vc; EmitVertex();
	gl_Position = p6; color = vc; EmitVertex();
	gl_Position = p4; color = vc; EmitVertex();
	EndPrimitive();

	gl_Position = p0; color = vc; EmitVertex();
	gl_Position = p4; color = vc; EmitVertex();
	EndPrimitive();

	gl_Position = p1; color = vc; EmitVertex();
	gl_Position = p5; color = vc; EmitVertex();
	EndPrimitive();

	gl_Position = p2; color = vc; EmitVertex();
	gl_Position = p6; color = vc; EmitVertex();
	EndPrimitive();

	gl_Position = p3; color = vc; EmitVertex();
	gl_Position = p7; color = vc; EmitVertex();
	EndPrimitive();
}
)DONE";

const char* box_fragment_shader_source = R"DONE(
#version 330

layout (location = 0) out vec3 fragment;

in vec3 color;

void main() {
	fragment = color;
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

struct AABB {
	float minX, minY, minZ;
	float maxX, maxY, maxZ;
};

struct Box {
	AABB aabb;
	float r, g, b;
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
	bool culling = false;
	bool showModel = true;
	bool showBoxes = true;
	int modelColorMode = 0;
	int boxColorMode = 0;
	float background[3] = {0.9f, 0.9f, 0.9f};
	bool fixedSize = false;
	bool doScreenshot = false;

	int mode = 0;
	bool full = false;
	bool pruned = false;
	bool fullyCorrect = false;
	int count = 0;

	bool dragging = false;
	float horizontalAngle = 0;
	float verticalAngle = 0;

	GLuint meshProgram = 0;
	GLuint boxProgram = 0;

	Trace* cachedTrace = nullptr;
	GLuint meshVAO = 0;
	GLuint meshVBO = 0;
	GLuint meshEBO = 0;
	GLuint meshIndexCount = 0;
	GLuint boxVAO = 0;
	GLuint boxVBO = 0;
	float centerX = 0.0f, centerY = 0.0f, centerZ = 0.0f;
	std::vector<AABB> aabbs;
	std::vector<Box> boxes;

	bool framebufferInit = false, framebufferSuccess = false;
	int targetWidth = 1024, targetHeight = 768;
	int width = 0, height = 0;
	GLuint framebuffer = 0;
	GLuint depthbuffer = 0;
	GLuint texture = 0;

	void initShaders() {
		{
			GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
			compile_shader(vertex_shader, mesh_vertex_shader_source, "vertex shader");
			
			GLuint geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);
			compile_shader(geometry_shader, mesh_geometry_shader_source, "geometry shader");
			
			GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
			compile_shader(fragment_shader, mesh_fragment_shader_source, "fragment shader");

			meshProgram = glCreateProgram();
			glAttachShader(meshProgram, vertex_shader);
			glAttachShader(meshProgram, geometry_shader);
			glAttachShader(meshProgram, fragment_shader);
			link_program(meshProgram, "mesh program");

			glDeleteShader(vertex_shader);
			glDeleteShader(geometry_shader);
			glDeleteShader(fragment_shader);
		}

		{
			GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
			compile_shader(vertex_shader, box_vertex_shader_source, "vertex shader");
			
			GLuint geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);
			compile_shader(geometry_shader, box_geometry_shader_source, "geometry shader");
			
			GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
			compile_shader(fragment_shader, box_fragment_shader_source, "fragment shader");

			boxProgram = glCreateProgram();
			glAttachShader(boxProgram, vertex_shader);
			glAttachShader(boxProgram, geometry_shader);
			glAttachShader(boxProgram, fragment_shader);
			link_program(boxProgram, "box program");

			glDeleteShader(vertex_shader);
			glDeleteShader(geometry_shader);
			glDeleteShader(fragment_shader);
		}

		GL_CHECK();
	}

	void extractMeshAndBounds(Trace* trace) {
		if (trace != cachedTrace) {
			glDeleteVertexArrays(1, &meshVAO);
			glDeleteBuffers(1, &meshVBO);
			glDeleteBuffers(1, &meshEBO);
			glDeleteVertexArrays(1, &boxVAO);
			glDeleteBuffers(1, &boxVBO);
			meshVAO = 0;
			boxVAO = 0;
			cachedTrace = nullptr;
		}

		// Check if trace contains memory contents.
		if (!trace || !trace->header.mem_contents_offset) return;
		if (trace == cachedTrace) return;

		cachedTrace = trace;

		TraceRegion* node_region = trace->find_region(0, 1);
		TraceRegion* bounds_region = trace->find_region(0, 2);
		TraceRegion* face_region = trace->find_region(0, 3);
		TraceRegion* vertex_region = trace->find_region(0, 4);

		// Extract mesh data.
		Face* faces = (Face*) trace->find_mem_contents(face_region);
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

		this->meshIndexCount = (GLuint) (3 * filteredFaces.size());

		float* vertex_data = (float*) trace->find_mem_contents(vertex_region);
		uint64_t vertex_count = vertex_region->size / (3 * sizeof(float));

		AABB root;
		root.minX = root.maxX = vertex_data[0];
		root.minY = root.maxY = vertex_data[1];
		root.minZ = root.maxZ = vertex_data[2];
		for (uint64_t i = 0; i < vertex_count; i++) {
			float x = vertex_data[3*i + 0];
			float y = vertex_data[3*i + 1];
			float z = vertex_data[3*i + 2];
			if (x < root.minX) root.minX = x;
			if (x > root.maxX) root.maxX = x;
			if (y < root.minY) root.minY = y;
			if (y > root.maxY) root.maxY = y;
			if (z < root.minZ) root.minZ = z;
			if (z > root.maxZ) root.maxZ = z;
			centerX += x;
			centerY += y;
			centerZ += z;
		}

		centerX /= vertex_count;
		centerY /= vertex_count;
		centerZ /= vertex_count;

		// Create mesh resources.
		glGenVertexArrays(1, &meshVAO);
		glGenBuffers(1, &meshVBO);
		glGenBuffers(1, &meshEBO);

		glBindVertexArray(meshVAO);
		glBindBuffer(GL_ARRAY_BUFFER, meshVBO);
		glBufferData(GL_ARRAY_BUFFER, vertex_region->size, vertex_data, GL_STATIC_DRAW);  

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshEBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, filteredFaces.size() * sizeof(Face), filteredFaces.data(), GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, (void*)0);

		glBindVertexArray(0);

		// Extract bounds.
		// This is a little involved, because the bounds are stored on the parent.

		AABB* bounds = (AABB*) trace->find_mem_contents(bounds_region);
		uint32_t* nodes = (uint32_t*) trace->find_mem_contents(node_region);
		uint64_t node_count = node_region->size / sizeof(uint32_t);

		this->count = (int) node_count;
		aabbs.resize(node_count);
		aabbs[0] = root;
		uint64_t bounds_index = 0;
		for (uint64_t i = 0; i < node_count; i++) {
			uint32_t node = nodes[i];
			uint32_t payload = node & 0x7fffffffu;
			bool isLeaf = node >> 31;
			if (isLeaf) continue;

			aabbs[i + 1] = bounds[bounds_index++];
			aabbs[i + payload] = bounds[bounds_index++];
		}

		// Create box resources.

		glGenVertexArrays(1, &boxVAO);
		glGenBuffers(1, &boxVBO);

		glBindVertexArray(boxVAO);
		glBindBuffer(GL_ARRAY_BUFFER, boxVBO);

		GLsizei stride = 3 * 3 * 4; // (min, max, color) * 3 floats * 4 bytes per float;
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 3 * 4, (void*)(0*3*4));
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * 3 * 4, (void*)(1*3*4));
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * 3 * 4, (void*)(2*3*4));

		glBindVertexArray(0);
	}

	void updateFramebuffer() {
		if (!framebufferInit) {
			framebufferInit = true;
			width = targetWidth;
			height = targetHeight;

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
		} else if (width != targetWidth || height != targetHeight) {
			width = targetWidth;
			height = targetHeight;

			glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			glBindRenderbuffer(GL_RENDERBUFFER, depthbuffer);
			glBindTexture(GL_TEXTURE_2D, texture);

			glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			glBindRenderbuffer(GL_RENDERBUFFER, 0);
			glBindTexture(GL_TEXTURE_2D, 0);
		}
	}

	void renderToTexture(Workspace* workspace, Trace* trace, uint64_t launch_id) {
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

		glViewport(0, 0, width, height);
		if (dragging) {
			float r = std::max(background[0] - 0.1f, 0.0f);
			float g = std::max(background[1] - 0.1f, 0.0f);
			float b = std::max(background[2] - 0.1f, 0.0f);
			glClearColor(r, g, b, 1.0f);
		} else {
			glClearColor(background[0], background[1], background[2], 1.0f);
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

		float vfov = 60.0f / 180.0f * IM_PI;
		float pos[3] = {0.0f, 0.0f, 0.2f};

		float v = std::tan(0.5f * vfov);
		float h = v * static_cast<float>(width) / static_cast<float>(height);
		float n = 0.05f, f = 100.0f;

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
		matrix_multiply(rotation, vertical, horizontal);

		float translation[16] = {
			1, 0, 0, -centerX,
			0, 1, 0, -centerY,
			0, 0, 1, -centerZ,
			0, 0, 0, 1,
		};

		float model[16];
		matrix_multiply(model, rotation, translation);

		if (showModel && meshVAO) {
			glUseProgram(meshProgram);

			GLint location;
			location = glGetUniformLocation(meshProgram, "projection");
			glUniformMatrix4fv(location, 1, GL_TRUE, projection);
			location = glGetUniformLocation(meshProgram, "view");
			glUniformMatrix4fv(location, 1, GL_TRUE, view);
			location = glGetUniformLocation(meshProgram, "model");
			glUniformMatrix4fv(location, 1, GL_TRUE, model);

			location = glGetUniformLocation(meshProgram, "modelColorMode");
			glUniform1i(location, modelColorMode);

			glBindVertexArray(meshVAO);
			glDrawElements(GL_TRIANGLES, meshIndexCount, GL_UNSIGNED_INT, 0);
			glBindVertexArray(0);

			glUseProgram(0);
		}

		if (showBoxes && boxVAO) {
			glUseProgram(boxProgram);

			GLint location;
			location = glGetUniformLocation(boxProgram, "projection");
			glUniformMatrix4fv(location, 1, GL_TRUE, projection);
			location = glGetUniformLocation(boxProgram, "view");
			glUniformMatrix4fv(location, 1, GL_TRUE, view);
			location = glGetUniformLocation(boxProgram, "model");
			glUniformMatrix4fv(location, 1, GL_TRUE, model);

			boxes.clear();
			boxes.reserve(aabbs.size());

			float baseColor[3];
			float markColor[3];

			if (boxColorMode == 1) {
				baseColor[0] = 0.0f;
				baseColor[1] = 0.0f;
				baseColor[2] = 0.0f;
				markColor[0] = 1.0f;
				markColor[1] = 1.0f;
				markColor[2] = 1.0f;
			} else {
				baseColor[0] = 1.0f;
				baseColor[1] = 0.0f;
				baseColor[2] = 0.0f;
				markColor[0] = 0.0f;
				markColor[1] = 0.0f;
				markColor[2] = 0.0f;
			}

			for (int i = 0; i < aabbs.size(); i++) {
				Box box;
				box.aabb = aabbs[i];
				box.r = baseColor[0]; box.g = baseColor[1]; box.b = baseColor[2];
				boxes.push_back(box);
			}

			auto markBox = [this,markColor] (size_t index) {
				Box& box = boxes.at(index);
				box.r = markColor[0]; box.g = markColor[1]; box.b = markColor[2];
			};

			auto isFullyCorrect = [&workspace] (Node* node) {
				auto it = workspace->reference_node_by_address.find(node->address);
				if (it == workspace->reference_node_by_address.end()) return false;

				Node* ref = it->second;
				switch (node->type) {
					case Node::PARENT: {
						if (ref->type != Node::PARENT) return false;

						bool connections_correct = false;
						bool bounds_correct = false;

						uint64_t nl = node->parent_data.edges[0].address;
						uint64_t nr = node->parent_data.edges[1].address;
						uint64_t rl = ref->parent_data.edges[0].address;
						uint64_t rr = ref->parent_data.edges[1].address;

						if ((nl == rl && nr == rr) || (nl == rr && nr == rl)) {
							connections_correct = true;
						}

						if (node->parent_data.bounds_address == ref->parent_data.bounds_address) {
							bounds_correct = true;
						}

						return connections_correct && bounds_correct;
					} break;

					case Node::LEAF: {
						if (ref->type != Node::LEAF) return false;
						return node->leaf_data.face_address == ref->leaf_data.face_address && node->leaf_data.face_count == ref->leaf_data.face_count;
					}
					break;

					default:
						return false;
				}
			};

			switch (mode) {
				// Accessed.
				case 0: {
					if (workspace) {
						if (full) {
							for (uint64_t j = 0; j < workspace->trace->header.launch_info_count; j++) {
								LinearAccessAnalysis* laa = &workspace->analysis[j].nodes_laa;
								for (uint64_t i = 0; i < boxes.size(); i++) {
									if (laa->flags[i]) {
										markBox(i);
									}
								}
							}
						} else {
							LinearAccessAnalysis* laa = &workspace->analysis[launch_id].nodes_laa;
							for (uint64_t i = 0; i < boxes.size(); i++) {
								if (laa->flags[i]) {
									markBox(i);
								}
							}
						}
					}
				} break;

				// Normal.
				case 1:
				// Precise.
				case 2:
				{
					if (workspace && trace) {
						TreeSet* set = (mode == 1) ? &workspace->normalTrees : &workspace->preciseTrees;
						Tree* tree = nullptr;
						if (full) {
							tree = pruned ? &set->fullReconstruction : &set->fullNodes;
						} else {
							tree = pruned ? &set->reconstructions[launch_id] : &set->nodes[launch_id];
						}

						TraceRegion* node_region = trace->find_region(0, 1);
						for (auto& node : tree->nodes) {
							int64_t index = calculate_index(node_region, node.address);
							if (fullyCorrect && !isFullyCorrect(&node)) continue;
							markBox(index);
						}
					}
				} break;
			}

			glDepthMask(false);

			glBindVertexArray(boxVAO);
			glBufferData(GL_ARRAY_BUFFER, boxes.size() * sizeof(Box), boxes.data(), GL_STREAM_DRAW);
			glDrawArrays(GL_POINTS, 0, (GLsizei) count);
			glBindVertexArray(0);

			glDepthMask(true);

			glUseProgram(0);
		}

		if (doScreenshot) {
			doScreenshot = false;
			takeScreenshot();
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		GL_CHECK();
	}

	void takeScreenshot() {
		char time[256];
		{ // format current local time
			std::time_t t = std::time(nullptr);
			if (!std::strftime(time, sizeof(time), "%Y-%m-%d_%H.%M.%S", std::localtime(&t)))
				return;
		}

		char filename[256];
		snprintf(filename, sizeof(filename), "screenshot_%s.png", time);

		try {
			GLubyte* buffer = new GLubyte[3 * width * height];

			glPixelStorei(GL_PACK_ALIGNMENT, 1);
			glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer);

			// flip image by starting at the last row and providing a negative stride
			GLsizei stride = 3 * width;
			GLubyte* last_row = buffer + (height - 1) * stride;
			stbi_write_png(filename, width, height, 3, last_row, -stride);

			delete[] buffer;
		} catch (const std::bad_alloc&) {}
	}

	void renderGui(Workspace* workspace, Selection& selected) {
		Trace* trace = workspace ? workspace->trace.get() : nullptr;

		if (!meshProgram) {
			initShaders();
		}

		extractMeshAndBounds(trace);

		updateFramebuffer();

		ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		if (ImGui::Begin("Visualizer", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
			ImGui::Checkbox("Culling", &culling);
			ImGui::Checkbox("Model", &showModel);
			ImGui::Checkbox("Boxes", &showBoxes);

			ImGui::Combo("Model Color", &modelColorMode, "UV\0Basic\0\0");
			ImGui::Combo("Box Color", &boxColorMode, "Black/Red\0White/Black\0\0");
			ImGui::ColorEdit3("Background Color", background);
			if (ImGui::Button("Default")) {
				background[0] = 0.9f;
				background[1] = 0.9f;
				background[2] = 0.9f;
			}
			ImGui::SameLine();
			if (ImGui::Button("Green")) {
				background[0] = 0.0f;
				background[1] = 0.7f;
				background[2] = 0.3f;
			}
			ImGui::SameLine();
			if (ImGui::Button("Vodka")) {
				background[0] = 197.0f / 255.0f;
				background[1] = 187.0f / 255.0f;
				background[2] = 252.0f / 255.0f;
			}

			ImGui::Checkbox("Fixed Size", &fixedSize);
			if (ImGui::Button("Screenshot")) doScreenshot = true;

			ImGui::Combo("Mode", &mode, "Accessed\0Normal\0Precise\0\0");
			ImGui::Checkbox("Full Trace", &full);
			ImGui::Checkbox("Prune Tree", &pruned);
			ImGui::Checkbox("Fully Correct", &fullyCorrect);
			ImGui::SliderInt("Count", &count, 0, (int) aabbs.size());

			ImVec2 avail = ImGui::GetContentRegionAvail();
			if (fixedSize) {
				targetWidth = 640;
				targetHeight = 640;
			} else {
				targetWidth = (int) avail.x;
				targetHeight = (int) avail.y;
			}
			ImVec2 size((float) width, (float) height);

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
				renderToTexture(workspace, trace, selected.launch_id);
			}

			// Flip uv coordinates to convert from OpenGL (y up) to ImGui (y down) convention.
			ImGui::Image((void*)(intptr_t)this->texture, size, ImVec2(0, 1), ImVec2(1, 0));
		}
		ImGui::PopStyleVar();
		ImGui::End();
	}
};
