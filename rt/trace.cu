// Copyright (c) 2021, Max von Buelow, GRIS, Technical University of Darmstadt

#include <cfloat>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

#include "image.h"

#include "cuda.h"
#include "check.h"
#include "bvh.h"
#include "config.h"
#include "types.h"

#ifdef MEMINF_ENABLED
	#include "meminf.h"
#endif

struct VertexData {
	vec3 normal;
};

struct HitPoint {
	uint32_t idx;
	float u, v;
};

template <typename T>
__device__ void swap(T &a, T &b) {
	T t;
	t = a;
	a = b;
	b = t;
}

__device__ Ray makeViewRay(float x, float y, float w, float h, const Camera& cam) {
	Ray ray;

	// t is half of the height of the image plane at unit distance from the camera center.
	float t = tan(cam.vertical_fov / 2.0f * M_PI / 180.0f);
	float px = +(2.0f * (x / w) - 1.0f) * t * w / h;
	float py = -(2.0f * (y / h) - 1.0f) * t;

	ray.origin = vec3(cam.x, cam.y, cam.z);

	// Compute the matrix-vector product between cam.mat and dir and store the result in ray.dir.
	vec3 dir = vec3(px, py, -1);
	for (int i = 0; i < 3; i++) {
		ray.dir[i] = dot(vec3(&cam.mat[3*i]), dir);
	}

	return ray;
}

__device__ void intersectRayAABB(const Ray& ray, const AABB& aabb, float& tmin, float& tmax) {
	tmin = FLT_MIN;
	tmax = FLT_MAX;
	for (int axis = 0; axis < 3; axis++) {
		float dirInverted = 1.0f / ray.dir[axis];
		float t1 = (aabb.min[axis] - ray.origin[axis]) * dirInverted;
		float t2 = (aabb.max[axis] - ray.origin[axis]) * dirInverted;
		float axisMin = min(t1, t2);
		float axisMax = max(t1, t2);
		tmin = max(tmin, axisMin);
		tmax = min(tmax, axisMax);
		if (tmin > tmax) break;
	}
}

__device__ bool intersectRayTriangle(const Ray& ray, const vec3& v0, const vec3& v1, const vec3& v2, float& t, float& uu, float& vv) {
	// from wikipedia moeller trumbore
	const float EPSILON = FLT_EPSILON;

	float e1x = v1[0] - v0[0], e1y = v1[1] - v0[1], e1z = v1[2] - v0[2];
	float e2x = v2[0] - v0[0], e2y = v2[1] - v0[1], e2z = v2[2] - v0[2];

	float hx = ray.dir[1] * e2z - ray.dir[2] * e2y;
	float hy = ray.dir[2] * e2x - ray.dir[0] * e2z;
	float hz = ray.dir[0] * e2y - ray.dir[1] * e2x;

	float a = e1x * hx + e1y * hy + e1z * hz;
	if (a > -EPSILON && a < EPSILON) return false; // This ray is parallel to this triangle.

	float f = 1.0 / a;
	float sx = ray.origin[0] - v0[0], sy = ray.origin[1] - v0[1], sz = ray.origin[2] - v0[2];
	float u = f * (sx * hx + sy * hy + sz * hz);
	if (u < 0.0 || u > 1.0) return false;

	float qx = sy * e1z - sz * e1y;
	float qy = sz * e1x - sx * e1z;
	float qz = sx * e1y - sy * e1x;

	float v = f * (ray.dir[0] * qx + ray.dir[1] * qy + ray.dir[2] * qz);
	if (v < 0.0 || u + v > 1.0) return false;
	uu = u;
	vv = v;

	// At this stage we can compute t to find out where the intersection point is on the line.
	t = f * (e2x * qx + e2y * qy + e2z * qz);
	if (t > EPSILON) return true; // ray intersection
	else return false; // This means that there is a line intersection but not a ray intersection.
}

__device__ bool intersectRayTriangle2(const Ray& ray, const Face* faces, const vec3* vertices, uint32_t idx, float &t, HitPoint* hitpoint) {
	Face f = faces[idx];
	vec3 a = vertices[f.idx[0]];
	vec3 b = vertices[f.idx[1]];
	vec3 c = vertices[f.idx[2]];

	float u, v;
	if (!intersectRayTriangle(ray, a, b, c, t, u, v)) return false;

	hitpoint->idx = idx;
	hitpoint->u = u;
	hitpoint->v = v;

	return true;
}

struct StackEntry {
	float t0, t1;
	uint32_t idx, leaves;
};

__device__ inline bool traverseBVH(float& t, HitPoint* hitpoint, const uint32_t* subtrees, const AABB* bounds, const Face* faces, const vec3* vertices, int maxLeaves, const Ray& ray) {
	bool hit = false;
	float t0 = 0, t1 = FLT_MAX;

	StackEntry stack[128];

	uint32_t ni = 0, li = 0, top = 0;
	while (true) {
		uint32_t st = subtrees[ni];
		uint32_t axis = st >> 30, left_subtree = st & 0x3fffffffu;
		__syncthreads();

		if (axis == 3) {

			uint32_t nn = left_subtree;

			uint32_t off = li * maxLeaves;
			for (int i = 0; i < nn; ++i) {
				float tt = FLT_MAX;
				HitPoint hitpoint_tmp;
				bool bhit = intersectRayTriangle2(ray, faces, vertices, off + i, tt, &hitpoint_tmp);
				bhit &= tt < t;
				if (bhit) {
					t = tt;
					*hitpoint = hitpoint_tmp;
					hit = true;
				}
				t1 = min(t1, t);
			}
		} else {
			uint32_t bi = ni - li;

			uint32_t cl = ni + 1, cr = ni + 1 + left_subtree;
			uint32_t ll = li, lr = li + (left_subtree + 1) / 2;


			// TODO check t0 and t
			float t0l = FLT_MAX, t1l = FLT_MIN, t0r = FLT_MAX, t1r = FLT_MIN;
			intersectRayAABB(ray, bounds[2 * bi + 0], t0l, t1l);
			intersectRayAABB(ray, bounds[2 * bi + 1], t0r, t1r);
			t0l = max(t0l, t0);
			t1l = min(t1l, t1);
			t0r = max(t0r, t0);
			t1r = min(t1r, t1);

			if (t0l > t1l || t0l > t0r) {
				swap(t0l, t0r);
				swap(t1l, t1r);
				swap(cl, cr);
				swap(ll, lr);
			}
			if (!(t0r > t1r)) {
				StackEntry e = StackEntry{t0r, t1r, cr, lr};
				stack[top] = e;
				++top;
			}

			if (!(t0l > t1l)) {
				t0 = t0l;
				t1 = t1l;
				ni = cl;
				li = ll;
				continue; // don't ascent
			}
		}

		do {
			if (top == 0) {
				return hit;
			}
			--top;
			StackEntry e = stack[top];
			ni = e.idx;
			li = e.leaves;
			t0 = e.t0;
			t1 = e.t1;
		} while (t0 > t);
	}
}

__device__ vec3 computeColor(const Vertex& vertex, const vec3& light, bool hit_shadow) {
	vec3 l = (light - vertex.position).normalizedOrZero();
	float d = max(dot(l, vertex.normal), 0.0f);
	float v = clamp(d - (hit_shadow ? 0.5f : 0.0f), 0.0f, 1.0f);
	return vec3(v, v, v);
}

__global__ void traceKernel(int x, int y, uint8_t* framebuffer, const uint32_t* subtrees, const AABB* bounds, const Face* faces, const vec3* vertices, const VertexData* vertexData, Camera cam, uint32_t w, uint32_t h, int maxLeaves) {
#ifdef __CUDACC__
	x = blockDim.x * (0+blockIdx.x) + threadIdx.x;
	y = blockDim.y * (0+blockIdx.y) + threadIdx.y;
	if (x >= w || y >= h) return;
#endif

	static const float light[] = { 50, 220, 1140 };

	Ray ray = makeViewRay(x + 0.5f, y + 0.5f, w, h, cam);

	float t = FLT_MAX;
	HitPoint hitpoint;
	bool hit = traverseBVH(t, &hitpoint, subtrees, bounds, faces, vertices, maxLeaves, ray);

	vec3 color;
	if (hit) {
		float u = hitpoint.u;
		float v = hitpoint.v;
		float w = 1.0f - u - v;

		Face f = faces[hitpoint.idx];

		// Load hit triangle completely.
		vec3 v0 = vertices[f.idx[0]];
		vec3 v1 = vertices[f.idx[1]];
		vec3 v2 = vertices[f.idx[2]];
		VertexData d0 = vertexData[f.idx[0]];
		VertexData d1 = vertexData[f.idx[1]];
		VertexData d2 = vertexData[f.idx[2]];

		// Interpolate vertices.
		Vertex vertex;
		vertex.position = v0 * w + v1 * u + v2 * v;
		vertex.normal = d0.normal * w + d1.normal * u + d2.normal * v;
	
		bool hit_shadow = false;
		color = computeColor(vertex, light, hit_shadow);
	} else {
		// Compute background color gradient.
		float tc = (y + 0.5) / h;
		color = (1.0f - tc) * vec3(115.0f, 193.0f, 245.0f) / 255.0f + tc * vec3(75.0f, 151.0f, 201.0f) / 255.0f;
	}

	framebuffer[3 * (y * w + x) + 0] = color.x * 255.0f;
	framebuffer[3 * (y * w + x) + 1] = color.y * 255.0f;
	framebuffer[3 * (y * w + x) + 2] = color.z * 255.0f;
}

void trace(uint8_t* framebuffer, const uint32_t* subtrees, const AABB* bounds, const Face* faces, const vec3* vertices, const VertexData* vertexData, Camera cam, uint32_t w, uint32_t h, int maxLeaves) {
#ifdef __CUDACC__
	dim3 blockDim(8, 8);
	dim3 gridDim((w + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);
	traceKernel<<<gridDim, blockDim>>>(0, 0, framebuffer, subtrees, bounds, faces, vertices, vertexData, cam, w, h, maxLeaves);
	CUDA_CHECK_LAST_ERROR();
#else
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			traceKernel(x, y, framebuffer, subtrees, bounds, faces, vertices, vertexData, cam, w, h, maxLeaves);
		}
	}
#endif
}

void *my_malloc(std::size_t size, int description) {
	void* result = nullptr;
	#ifdef __CUDACC__
		CUDA_CHECK(cudaMalloc(&result, size));
	#else
		result = malloc(size);
	#endif
	#ifdef MEMINF_ENABLED
		meminf_describe(result, description);
	#endif
	return result;
}

void my_upload(void* dst, const void* src, std::size_t size) {
	#ifdef __CUDACC__
		CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
	#else
		std::memcpy(dst, src, size);
	#endif
}

void my_download(void* dst, const void* src, std::size_t size) {
	#ifdef __CUDACC__
		CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
	#else
		std::memcpy(dst, src, size);
	#endif
}

void my_synchronize() {
	#ifdef __CUDACC__
		CUDA_CHECK(cudaDeviceSynchronize());
	#endif
}

void run(Configuration& config) {
	auto t0 = std::chrono::high_resolution_clock::now();

	std::cout << "Loading mesh..." << std::endl;
	Mesh mesh = loadMesh(config.input);
	mesh.compute_normals();
	std::cout << "Mesh: " << mesh.faces.size() << " faces; " << mesh.vertices.size() << " vertices" << std::endl;

	auto t1 = std::chrono::high_resolution_clock::now();

	std::cout << "Building BVH..." << std::endl;
	int maxLeaves = 32;
	BVHBuilder bvhb;
	std::vector<float> aabbs(mesh.faces.size() * 6);
	std::vector<float> centers(mesh.faces.size() * 3);
	for (uint32_t i = 0; i < mesh.faces.size(); ++i) {
		const Face& face = mesh.faces[i];
		const Vertex& v0 = mesh.vertices[face.idx[0]];
		const Vertex& v1 = mesh.vertices[face.idx[1]];
		const Vertex& v2 = mesh.vertices[face.idx[2]];

		AABB aabb;
		aabb.feed(v0.position);
		aabb.feed(v1.position);
		aabb.feed(v2.position);

		for (int j = 0; j < 3; ++j) {
			aabbs[i * 6 + j] = aabb.min[j];
			aabbs[i * 6 + j + 3] = aabb.max[j];
			centers[i * 3 + j] = (v0.position[j] + v1.position[j] + v2.position[j]) / 3;
		}
	}

	bvhb.construct(centers.data(), aabbs.data(), mesh.faces.size(), maxLeaves, config.heuristic);
	std::cout << "BVH: " << bvhb.num_nodes() << " nodes; " << bvhb.bounds.size() << " aabbs; " << bvhb.leaf_nodes.size() << " leaves" << std::endl;

	auto t2 = std::chrono::high_resolution_clock::now();

	std::cout << "Rearranging data..." << std::endl;
	std::vector<Face> bvhFaces(bvhb.leaf_nodes.size());
	std::vector<vec3> bvhVertices(mesh.vertices.size());
	std::vector<VertexData> bvhVertexData(mesh.vertices.size());

	for (int i = 0; i < bvhb.leaf_nodes.size(); i++) {
		uint32_t f = bvhb.leaf_nodes[i];
		if (f == -1u) bvhFaces[i] = Face(0, 0, 0);
		else bvhFaces[i] = mesh.faces[bvhb.leaf_nodes[i]];
	}

	for (int i = 0; i < mesh.vertices.size(); i++) {
		bvhVertices[i] = mesh.vertices[i].position;
		bvhVertexData[i] = VertexData{ mesh.vertices[i].normal };
	}

	auto t3 = std::chrono::high_resolution_clock::now();

	std::cout << "Uploading..." << std::endl;
	image_b output(config.width, config.height, 3);
	uint8_t* d_framebuffer = (uint8_t*) my_malloc(output.size(), 0);

	uint32_t *d_subtrees = (uint32_t*) my_malloc(bvhb.subtrees.size() * sizeof(uint32_t), 1);
	my_upload(d_subtrees, bvhb.subtrees.data(), bvhb.subtrees.size() * sizeof(uint32_t));

	AABB* d_bounds = (AABB*) my_malloc(bvhb.bounds.size() * sizeof(AABB), 2);
	my_upload(d_bounds, bvhb.bounds.data(), bvhb.bounds.size() * sizeof(AABB));

	Face* d_faces = (Face*) my_malloc(bvhFaces.size() * sizeof(Face), 3);
	my_upload(d_faces, bvhFaces.data(), bvhFaces.size() * sizeof(Face));

	vec3* d_vertices = (vec3*) my_malloc(bvhVertices.size() * sizeof(vec3), 4);
	my_upload(d_vertices, bvhVertices.data(), bvhVertices.size() * sizeof(vec3));

	VertexData *d_vertexData = (VertexData*) my_malloc(bvhVertexData.size() * sizeof(VertexData), 5);
	my_upload(d_vertexData, bvhVertexData.data(), bvhVertexData.size() * sizeof(VertexData));

	auto t4 = std::chrono::high_resolution_clock::now();

	std::cout << "Rendering..." << std::endl;
	trace(d_framebuffer, d_subtrees, d_bounds, d_faces, d_vertices, d_vertexData, config.camera, output.width(), output.height(), maxLeaves);
	my_synchronize();

	auto t5 = std::chrono::high_resolution_clock::now();

	std::cout << "Downloading..." << std::endl;
	my_download(output.data(), d_framebuffer, output.size());

	auto t6 = std::chrono::high_resolution_clock::now();

	if (!config.output.empty()) {
		std::cout << "Saving..." << std::endl;
		image_io::save(output, config.output.c_str());
	} else {
		std::cout << "No output file specified" << std::endl;
	}

	auto t7 = std::chrono::high_resolution_clock::now();

	printf("Mesh:      %0.9fs\n", std::chrono::duration<double>(t1 - t0).count());
	printf("BVH:       %0.9fs\n", std::chrono::duration<double>(t2 - t1).count());
	printf("Rearrange: %0.9fs\n", std::chrono::duration<double>(t3 - t2).count());
	printf("Upload:    %0.9fs\n", std::chrono::duration<double>(t4 - t3).count());
	printf("Render:    %0.9fs\n", std::chrono::duration<double>(t5 - t4).count());
	printf("Download:  %0.9fs\n", std::chrono::duration<double>(t6 - t5).count());
	if (!config.output.empty()) {
	printf("Save:      %0.9fs\n", std::chrono::duration<double>(t7 - t6).count());
	}

	fflush(stdout);
}

int main(int argc, const char** argv) {
	Configuration config;

	for (int i = 1; i < argc; i++) {
		std::string argument = argv[i];

		if (argument == "-config") {
			if (i + 1 >= argc) {
				std::cerr << "missing value for " << argument << std::endl;
				return 1;
			}
			try {
				loadConfiguration(config, argv[++i]);
			} catch (const std::exception& e) {
				std::cerr << e.what() << std::endl;
				return 1;
			}
			continue;
		}

		if (argument == "-input") {
			if (i + 1 >= argc) {
				std::cerr << "missing value for " << argument << std::endl;
				return 1;
			}
			config.input = argv[++i];
			continue;
		}

		if (argument == "-output") {
			if (i + 1 >= argc) {
				std::cerr << "missing value for " << argument << std::endl;
				return 1;
			}
			config.output = argv[++i];
			continue;
		}

		if (argument == "-size") {
			if (i + 1 >= argc) {
				std::cerr << "missing value for " << argument << std::endl;
				return 1;
			}
			int value = std::atoi(argv[++i]);
			if (value == 0) {
				std::cerr << "bad value for " << argument << std::endl;
				return 1;
			}
			config.width = value;
			config.height = value;
			continue;
		}

		if (argument == "-width") {
			if (i + 1 >= argc) {
				std::cerr << "missing value for " << argument << std::endl;
				return 1;
			}
			int value = std::atoi(argv[++i]);
			if (value == 0) {
				std::cerr << "bad value for " << argument << std::endl;
				return 1;
			}
			config.width = value;
			continue;
		}

		if (argument == "-height") {
			if (i + 1 >= argc) {
				std::cerr << "missing value for " << argument << std::endl;
				return 1;
			}
			int value = std::atoi(argv[++i]);
			if (value == 0) {
				std::cerr << "bad value for " << argument << std::endl;
				return 1;
			}
			config.height = value;
			continue;
		}

		if (argument == "-heuristic") {
			if (i + 1 >= argc) {
				std::cerr << "missing value for " << argument << std::endl;
				return 1;
			}
			std::string value = argv[++i];
			if (value == "sah") {
				config.heuristic = SAH;
			} else if (value == "median") {
				config.heuristic = MEDIAN;
			} else {
				std::cerr << "warning: ignoring unknown value for " << argument << ": " << value << std::endl;
			}
			continue;
		}

		std::cerr << "unknown argument " << argument << std::endl;
		return 1;
	}

	if (config.input.empty()) {
		std::cerr << "no input file specified, use -input or -config" << std::endl;
		return 1;
	}
	if (config.width == 0 || config.height == 0) {
		std::cerr << "no output size specified, use -width and -height or -config" << std::endl;
		return 1;
	}

	try {
		run(config);
		return 0;
	} catch (const std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}
}
