// Copyright (c) 2021, Max von Buelow, GRIS, Technical University of Darmstadt

#include <cassert>
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
#include "mesh.h"

struct Ray {
	vec3 origin;
	vec3 dir;
};

struct HitPoint {
	float t;
	float u, v;
	uint32_t idx;
};

struct VertexData {
	vec3 normal;
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

__device__ bool intersectRayTriangle(const Ray& ray, const Face* faces, const vec3* vertices, uint32_t idx, HitPoint& hitpoint) {
	// From Wikipedia: Möller-Trumbore intersection algorithm

	Face face = faces[idx];
	vec3 v0 = vertices[face.idx[0]];
	vec3 v1 = vertices[face.idx[1]];
	vec3 v2 = vertices[face.idx[2]];

	vec3 e1 = v1 - v0;
	vec3 e2 = v2 - v0;

	vec3 h = cross(ray.dir, e2);
	float a = dot(e1, h);
	float f = 1.0f / a;
	if (!isfinite(f)) return false; // The ray is parallel to the triangle.

	vec3 s = ray.origin - v0;
	float u = f * dot(s, h);
	if (u < 0.0 || u > 1.0) return false; // The ray intersects the plane outside of the triangle.

	vec3 q = cross(s, e1);
	float v = f * dot(ray.dir, q);
	if (v < 0.0 || u + v > 1.0) return false; // The ray intersects the plane outside of the triangle.

	// At this stage we can compute t to find out where the intersection point is on the ray.
	float t = f * dot(e2, q);
	if (t < FLT_EPSILON) return false; // The intersection is on the wrong side of the ray origin.

	hitpoint.t = t;
	hitpoint.u = u;
	hitpoint.v = v;
	hitpoint.idx = idx;
	return true;
}

struct StackEntry {
	float tmin, tmax;
	uint32_t nodeIndex, leafIndex;
};

__device__ bool traverseBVH(HitPoint& hitpoint, const BVH::Node* nodes, const AABB* bounds, const Face* faces, const vec3* vertices, uint32_t maxPrimitives, const Ray& ray) {
	StackEntry stack[128];
	uint32_t top = 0; // index of first free stack entry

	StackEntry entry = {};
	entry.tmin = 0; entry.tmax = FLT_MAX;
	bool hit = false;

	// hitpoint.t stores the t value of the closest hit so far,
	// so we need to initialize it here.
	hitpoint.t = FLT_MAX;

	while (true) {
		BVH::Node node = nodes[entry.nodeIndex];
		bool isLeaf = node >> 31;
		uint32_t payload = node & 0x7fffffffu;
		__syncthreads();

		if (isLeaf) {

			// This node is a leaf node.
			// Iterate over contained triangles.
			// Payload contains the number of triangles.

			uint32_t offset = entry.leafIndex * maxPrimitives;
			for (int i = 0; i < payload; i++) {
				HitPoint currentHitpoint;
				bool currentHit = intersectRayTriangle(ray, faces, vertices, offset + i, currentHitpoint);
				if (currentHit & (currentHitpoint.t < hitpoint.t)) {
					hitpoint = currentHitpoint;
					hit = true;
				}
			}

		} else {

			// This node is an inner node.
			// Recursively traverse both children.
			// Payload contains the offset to the right child.

			// Compute node indices of our children.
			// The left child is our immediate neighbor.
			// The right child is given by the offset stored in our node payload.
			uint32_t leftNodeIndex = entry.nodeIndex + 1;
			uint32_t rightNodeIndex = entry.nodeIndex  + payload;

			// Compute leaf indices for our children.
			// Since an inner node is not a leaf, we pass the leaf index on to
			// the left child.
			// For the right child, we know that the whole left subtree fits
			// exactly into the offset between ourselves and our right child.
			// Because this is a binary tree, it must have exactly one more
			// leaf node than inner nodes. However, since the current inner
			// node is counted as well, the number of inner nodes and child
			// nodes is exactly equal. Therefore, we can compute the number of
			// leaf nodes to skip by dividing offset by 2.
			uint32_t leftLeafIndex = entry.leafIndex;
			uint32_t rightLeafIndex = entry.leafIndex + payload / 2;

			// Conversely, we can compute the number of inner nodes as the
			// difference of total nodes and leaf nodes, which is used to index
			// into the bounds array, as only inner nodes store the bounds of
			// their children.
			uint32_t boundsIndex = entry.nodeIndex - entry.leafIndex;

			// Compute intersections with child bounds.
			float tminLeft, tmaxLeft, tminRight, tmaxRight;
			intersectRayAABB(ray, bounds[2 * boundsIndex + 0], tminLeft, tmaxLeft);
			intersectRayAABB(ray, bounds[2 * boundsIndex + 1], tminRight, tmaxRight);

			// Restrict to current range of interest.
			tminLeft = max(tminLeft, entry.tmin);
			tmaxLeft = min(tmaxLeft, entry.tmax);
			tminRight = max(tminRight, entry.tmin);
			tmaxRight = min(tmaxRight, entry.tmax);

			// If the left child is not hit, or the right child is hit before the left child,
			// swap them, because we will handle the left child first.
			if (tminLeft > tmaxLeft || tminRight < tminLeft) {
				swap(tminLeft, tminRight);
				swap(tmaxLeft, tmaxRight);
				swap(leftNodeIndex, rightNodeIndex);
				swap(leftLeafIndex, rightLeafIndex);
			}

			// If the right child is hit, push it on the stack.
			if (!(tminRight > tmaxRight)) {
				stack[top++] = StackEntry{tminRight, tmaxRight, rightNodeIndex, rightLeafIndex};
			}

			// If the left child is hit, continue immediately with it.
			// Otherwise we fall through and handle the next entry from the stack.
			if (!(tminLeft > tmaxLeft)) {
				entry.tmin = tminLeft;
				entry.tmax = tmaxLeft;
				entry.nodeIndex = leftNodeIndex;
				entry.leafIndex = leftLeafIndex;
				continue; // don't ascent
			}
		}

		do {
			// Return result, if stack is empty.
			if (top == 0) return hit;
			// Pop entries until one is in front of the best hit so far.
			entry = stack[--top];
		} while (entry.tmin > hitpoint.t);
	}
}

// Same as traverseBVH except it returns as soon as any hit is found.
__device__ bool traverseBVHShadow(const BVH::Node* nodes, const AABB* bounds, const Face* faces, const vec3* vertices, uint32_t maxPrimitives, const Ray& ray) {
	StackEntry stack[128];
	uint32_t top = 0; // index of first free stack entry

	StackEntry entry = {};
	entry.tmin = 0; entry.tmax = FLT_MAX;
	while (true) {
		BVH::Node node = nodes[entry.nodeIndex];
		bool isLeaf = node >> 31;
		uint32_t payload = node & 0x7fffffffu;
		__syncthreads();

		if (isLeaf) {
			uint32_t offset = entry.leafIndex * maxPrimitives;
			for (int i = 0; i < payload; i++) {
				HitPoint currentHitpoint;
				bool currentHit = intersectRayTriangle(ray, faces, vertices, offset + i, currentHitpoint);
				if (currentHit) return true;
			}

		} else {
			uint32_t leftNodeIndex = entry.nodeIndex + 1;
			uint32_t rightNodeIndex = entry.nodeIndex  + payload;
			uint32_t leftLeafIndex = entry.leafIndex;
			uint32_t rightLeafIndex = entry.leafIndex + payload / 2;
			uint32_t boundsIndex = entry.nodeIndex - entry.leafIndex;

			float tminLeft, tmaxLeft, tminRight, tmaxRight;
			intersectRayAABB(ray, bounds[2 * boundsIndex + 0], tminLeft, tmaxLeft);
			intersectRayAABB(ray, bounds[2 * boundsIndex + 1], tminRight, tmaxRight);

			tminLeft = max(tminLeft, entry.tmin);
			tmaxLeft = min(tmaxLeft, entry.tmax);
			tminRight = max(tminRight, entry.tmin);
			tmaxRight = min(tmaxRight, entry.tmax);

			if (tminLeft > tmaxLeft || tminRight < tminLeft) {
				swap(tminLeft, tminRight);
				swap(tmaxLeft, tmaxRight);
				swap(leftNodeIndex, rightNodeIndex);
				swap(leftLeafIndex, rightLeafIndex);
			}

			if (!(tminRight > tmaxRight)) {
				stack[top++] = StackEntry{tminRight, tmaxRight, rightNodeIndex, rightLeafIndex};
			}

			if (!(tminLeft > tmaxLeft)) {
				entry.tmin = tminLeft;
				entry.tmax = tmaxLeft;
				entry.nodeIndex = leftNodeIndex;
				entry.leafIndex = leftLeafIndex;
				continue; // don't ascent
			}
		}

		if (top == 0) return false;
		entry = stack[--top];
	}
}

__device__ vec3 computeColor(const Vertex& vertex, const vec3& light, bool hitShadow) {
	vec3 l = (light - vertex.position).normalizedOrZero();
	vec3 n = vertex.normal.normalizedOrZero();
	float d = max(dot(l, n), 0.0f);
	float v = 0.05f + 0.95f * d * (hitShadow ? 0.0f : 1.0f);
	v = clamp(v, 0.0f, 1.0f);
	return vec3(v, v, v);
}

__global__ void traceKernel(int x, int y, uint8_t* framebuffer, const BVH::Node* nodes, const AABB* bounds, const Face* faces, const vec3* vertices, const VertexData* vertexData, Camera cam, Light light, uint32_t w, uint32_t h, uint32_t maxPrimitives, bool flat, bool shadows) {
#ifdef __CUDACC__
	x = blockDim.x * (0+blockIdx.x) + threadIdx.x;
	y = blockDim.y * (0+blockIdx.y) + threadIdx.y;
	if (x >= w || y >= h) return;
#endif

	Ray ray = makeViewRay(x + 0.5f, y + 0.5f, w, h, cam);

	HitPoint hitpoint;
	bool hit = traverseBVH(hitpoint, nodes, bounds, faces, vertices, maxPrimitives, ray);

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

		if (flat) {
			// Calculate normal for flat shading.
			vec3 e1 = v1 - v0;
			vec3 e2 = v2 - v0;
			vertex.normal = cross(e1, e2).normalizedOrZero();
		}

		vec3 lightPos(light.x, light.y, light.z);

		bool hitShadow = false;
		if (shadows) {
			Ray shadowRay;
			shadowRay.origin = vertex.position + 1e-7 * vertex.normal.normalizedOrZero();
			shadowRay.dir = (lightPos - shadowRay.origin).normalizedOrZero();
			hitShadow = traverseBVHShadow(nodes, bounds, faces, vertices, maxPrimitives, shadowRay);
		}

		color = computeColor(vertex, lightPos, hitShadow);
	} else {
		// Compute background color gradient.
		float tc = (y + 0.5) / h;
		color = (1.0f - tc) * vec3(115.0f, 193.0f, 245.0f) / 255.0f + tc * vec3(75.0f, 151.0f, 201.0f) / 255.0f;
	}

	int64_t x1 = x, y1 = y;
	framebuffer[3 * (y1 * w + x1) + 0] = color.x * 255.0f;
	framebuffer[3 * (y1 * w + x1) + 1] = color.y * 255.0f;
	framebuffer[3 * (y1 * w + x1) + 2] = color.z * 255.0f;
}

void trace(uint8_t* framebuffer, const BVH::Node* nodes, const AABB* bounds, const Face* faces, const vec3* vertices, const VertexData* vertexData, Camera cam, Light light, uint32_t w, uint32_t h, uint32_t maxPrimitives, bool flat, bool shadows) {
#ifdef __CUDACC__
	dim3 blockDim(8, 8);
	dim3 gridDim((w + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);
	traceKernel<<<gridDim, blockDim>>>(0, 0, framebuffer, nodes, bounds, faces, vertices, vertexData, cam, light, w, h, maxPrimitives, flat, shadows);
	CUDA_CHECK_LAST_ERROR();
#else
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			traceKernel(x, y, framebuffer, nodes, bounds, faces, vertices, vertexData, cam, light, w, h, maxPrimitives, flat, shadows);
		}
	}
#endif
}

void *my_malloc(std::size_t size) {
	void* result = nullptr;
	#ifdef __CUDACC__
		CUDA_CHECK(cudaMalloc(&result, size));
	#else
		result = malloc(size);
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

#ifdef __CUDACC__
	typedef cudaEvent_t event_t;
#else
	typedef void* event_t;
#endif

void my_event_create(event_t* event) {
	#ifdef __CUDACC__
		CUDA_CHECK(cudaEventCreate(event));
	#endif
}

void my_event_destroy(event_t event) {
	#ifdef __CUDACC__
		CUDA_CHECK(cudaEventDestroy(event));
	#endif	
}

void my_event_record(event_t event) {
	#ifdef __CUDACC__
		CUDA_CHECK(cudaEventRecord(event));
	#endif
}

double my_event_elapsed_time_seconds(event_t start, event_t stop) {
	#ifdef __CUDACC__
		float millis = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&millis, start, stop));
		return static_cast<double>(millis) / 1000.0;
	#else
		return 0.0;
	#endif
}

void run(Configuration& config) {
	std::vector<std::chrono::high_resolution_clock::time_point> ts;
	ts.reserve(200);

	std::vector<double> gpu_timings;
	gpu_timings.reserve(16);

	ts.push_back(std::chrono::high_resolution_clock::now());

	std::cout << "Loading mesh..." << std::endl;
	Mesh mesh = loadMesh(config.input);
	mesh.compute_normals();
	std::cout << "Mesh: " << mesh.faces.size() << " faces; " << mesh.vertices.size() << " vertices" << std::endl;

	ts.push_back(std::chrono::high_resolution_clock::now());

	std::cout << "Computing AABBs..." << std::endl;
	std::vector<AABB> aabbs(mesh.faces.size());
	std::vector<vec3> centers(mesh.faces.size());
	for (uint32_t i = 0; i < mesh.faces.size(); ++i) {
		const Face& face = mesh.faces[i];
		const Vertex& v0 = mesh.vertices[face.idx[0]];
		const Vertex& v1 = mesh.vertices[face.idx[1]];
		const Vertex& v2 = mesh.vertices[face.idx[2]];

		AABB aabb;
		aabb.feed(v0.position);
		aabb.feed(v1.position);
		aabb.feed(v2.position);
		aabbs[i] = aabb;

		centers[i] = (v0.position + v1.position + v2.position) / 3.0f;
	}

	ts.push_back(std::chrono::high_resolution_clock::now());

	std::cout << "Building BVH..." << std::endl;
	Heuristic heuristic = parseHeuristic(config.heuristic);
	BVH bvh = constructBVH(aabbs, centers, 32, heuristic);
	std::cout << "BVH: " << bvh.nodes.size() << " nodes; " << bvh.bounds.size() << " aabbs; " << bvh.primitives.size() << " primitives; " << bvh.depth << " max depth" << std::endl;

	ts.push_back(std::chrono::high_resolution_clock::now());

	std::cout << "Rearranging data..." << std::endl;
	std::vector<Face> bvhFaces(bvh.primitives.size());
	std::vector<vec3> bvhVertices(mesh.vertices.size());
	std::vector<VertexData> bvhVertexData(mesh.vertices.size());

	for (int i = 0; i < bvh.primitives.size(); i++) {
		uint32_t sourceIndex = bvh.primitives[i];
		bvhFaces[i] = (sourceIndex == UINT32_MAX) ? Face() : mesh.faces[sourceIndex];
	}

	for (int i = 0; i < mesh.vertices.size(); i++) {
		bvhVertices[i] = mesh.vertices[i].position;
		bvhVertexData[i] = VertexData{ mesh.vertices[i].normal };
	}

	ts.push_back(std::chrono::high_resolution_clock::now());

	std::cout << "Initializing CUDA runtime..." << std::endl;
	// Dummy call to initialize the runtime to avoid polluting the upload timing.
	CUDA_CHECK(cudaDeviceSynchronize());

	event_t start, stop;
	my_event_create(&start);
	my_event_create(&stop);

	ts.push_back(std::chrono::high_resolution_clock::now());

	std::cout << "Uploading..." << std::endl;
	image_b output(config.width, config.height, 3);
	uint8_t* d_framebuffer = (uint8_t*) my_malloc(output.size());

	BVH::Node *d_nodes = (BVH::Node*) my_malloc(bvh.nodes.size() * sizeof(BVH::Node));
	my_upload(d_nodes, bvh.nodes.data(), bvh.nodes.size() * sizeof(BVH::Node));

	AABB* d_bounds = (AABB*) my_malloc(bvh.bounds.size() * sizeof(AABB));
	my_upload(d_bounds, bvh.bounds.data(), bvh.bounds.size() * sizeof(AABB));

	Face* d_faces = (Face*) my_malloc(bvhFaces.size() * sizeof(Face));
	my_upload(d_faces, bvhFaces.data(), bvhFaces.size() * sizeof(Face));

	vec3* d_vertices = (vec3*) my_malloc(bvhVertices.size() * sizeof(vec3));
	my_upload(d_vertices, bvhVertices.data(), bvhVertices.size() * sizeof(vec3));

	VertexData *d_vertexData = (VertexData*) my_malloc(bvhVertexData.size() * sizeof(VertexData));
	my_upload(d_vertexData, bvhVertexData.data(), bvhVertexData.size() * sizeof(VertexData));

	ts.push_back(std::chrono::high_resolution_clock::now());

	for (int i = 0; i < config.cameras.size(); i++) {
		std::cout << "Rendering..." << std::endl;

		my_event_record(start);
		trace(d_framebuffer, d_nodes, d_bounds, d_faces, d_vertices, d_vertexData, config.cameras[i], config.light, config.width, config.height, bvh.maxPrimitives, config.shading == FLAT, config.shadows);
		my_event_record(stop);
		my_synchronize();
		
		double gpu_time = my_event_elapsed_time_seconds(start, stop);
		gpu_timings.push_back(gpu_time);

		ts.push_back(std::chrono::high_resolution_clock::now());

		std::cout << "Downloading..." << std::endl;
		my_download(output.data(), d_framebuffer, output.size());

		ts.push_back(std::chrono::high_resolution_clock::now());

		if (!config.output.empty()) {
			std::cout << "Saving..." << std::endl;
			char buffer[256];
			snprintf(buffer, sizeof(buffer), config.output.c_str(), i);
			image_io::save(output, buffer);
		} else {
			std::cout << "No output file specified" << std::endl;
		}

		ts.push_back(std::chrono::high_resolution_clock::now());
	}

	int tj = 0;
	printf("Mesh:       %0.9fs\n", std::chrono::duration<double>(ts[tj+1] - ts[tj]).count()); tj++;
	printf("AABBs:      %0.9fs\n", std::chrono::duration<double>(ts[tj+1] - ts[tj]).count()); tj++;
	printf("BVH:        %0.9fs\n", std::chrono::duration<double>(ts[tj+1] - ts[tj]).count()); tj++;
	printf("Rearrange:  %0.9fs\n", std::chrono::duration<double>(ts[tj+1] - ts[tj]).count()); tj++;
	printf("Cuda init:  %0.9fs\n", std::chrono::duration<double>(ts[tj+1] - ts[tj]).count()); tj++;
	printf("Upload:     %0.9fs\n", std::chrono::duration<double>(ts[tj+1] - ts[tj]).count()); tj++;
	for (int i = 0; i < config.cameras.size(); i++) {
		printf("Render GPU: %0.9fs\n", gpu_timings[i]);
		printf("Render CPU: %0.9fs\n", std::chrono::duration<double>(ts[tj+1] - ts[tj]).count()); tj++;
		printf("Download:   %0.9fs\n", std::chrono::duration<double>(ts[tj+1] - ts[tj]).count()); tj++;
		if (!config.output.empty()) {
			printf("Save:       %0.9fs\n", std::chrono::duration<double>(ts[tj+1] - ts[tj]).count());
		}
		tj++;
	}

	fflush(stdout);

	my_event_destroy(start);
	my_event_destroy(stop);
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

		if (argument == "-shading") {
			if (i + 1 >= argc) {
				std::cerr << "missing value for " << argument << std::endl;
				return 1;
			}
			std::string value = argv[++i];
			if (value == "smooth") {
				config.shading = SMOOTH;
			} else if (value == "flat") {
				config.shading = FLAT;
			} else {
				std::cerr << "warning: ignoring unknown value for " << argument << ": " << value << std::endl;
			}
			continue;
		}

		if (argument == "-flat") {
			config.shading = FLAT;
			continue;
		}

		if (argument == "-smooth") {
			config.shading = SMOOTH;
			continue;
		}

		if (argument == "-shadows") {
			if (i + 1 < argc && argv[i+1][0] != '-') {
				std::string value = argv[++i];
				if (value == "true") {
					config.shadows = true;
				} else if (value == "false") {
					config.shadows = false;
				} else {
					std::cerr << "warning: ignoring unknown value for " << argument << ": " << value << std::endl;
				}
			} else {
				config.shadows = true;
			}
			continue;
		}

		if (argument == "-heuristic") {
			if (i + 1 >= argc) {
				std::cerr << "missing value for " << argument << std::endl;
				return 1;
			}
			std::string value = argv[++i];
			if (validateHeuristic(value)) {
				config.heuristic = value;
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

	if (config.cameras.empty()) {
		Camera defaultCam = {};
		config.cameras.push_back(defaultCam);
	}

	try {
		run(config);
		return 0;
	} catch (const std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}
}
