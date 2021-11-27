// Copyright (c) 2021, Max von Buelow, GRIS, Technical University of Darmstadt

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <float.h>
#include <stdint.h>
#include <stdio.h>

#include "happly.h"
#include "image.h"
#include "meminf.h"

#include "bvh.h"
#include "types.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#ifndef __CUDACC__
template <typename T>
T min(T a, T b)
{
	return a < b ? a : b;
}
template <typename T>
T max(T a, T b)
{
	return a > b ? a : b;
}

struct float4 { float x, y, z, w; };

#define __HD__
#define __D__
#else
#define __HD__ __host__ __device__
#define __D__ __device__
#endif

struct RayG {
	float org[3];
	float dir[3];
// 	inline void normalize()
// 	{
// 		float n = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
// 		for (int i = 0; i < 3; ++i) dir[i] /= n;
// 	}
};
__HD__ inline bool tri_intersect(float &t, float &uu, float &vv, const float *rayorg, const float *raydir, const float *v0, const float *v1, const float *v2)
{
	// from wikipedia moeller trumbore
	const float EPSILON = FLT_EPSILON;

	float e1x = v1[0] - v0[0], e1y = v1[1] - v0[1], e1z = v1[2] - v0[2];
	float e2x = v2[0] - v0[0], e2y = v2[1] - v0[1], e2z = v2[2] - v0[2];

	float hx = raydir[1] * e2z - raydir[2] * e2y;
	float hy = raydir[2] * e2x - raydir[0] * e2z;
	float hz = raydir[0] * e2y - raydir[1] * e2x;

	float a = e1x * hx + e1y * hy + e1z * hz;
	if (a > -EPSILON && a < EPSILON) return false; // This ray is parallel to this triangle.

	float f = 1.0 / a;
	float sx = rayorg[0] - v0[0], sy = rayorg[1] - v0[1], sz = rayorg[2] - v0[2];
	float u = f * (sx * hx + sy * hy + sz * hz);
	if (u < 0.0 || u > 1.0) return false;

	float qx = sy * e1z - sz * e1y;
	float qy = sz * e1x - sx * e1z;
	float qz = sx * e1y - sy * e1x;

	float v = f * (raydir[0] * qx + raydir[1] * qy + raydir[2] * qz);
	if (v < 0.0 || u + v > 1.0) return false;
	uu = u;
	vv = v;

	// At this stage we can compute t to find out where the intersection point is on the line.
	t = f * (e2x * qx + e2y * qy + e2z * qz);
	if (t > EPSILON) return true; // ray intersection
	else return false; // This means that there is a line intersection but not a ray intersection.
}

struct camera {
	float x, y, z;
	float mat[9];
	float fov;
};

struct FaceG {
	uint32_t idx[3];
};
struct Vtx {
	float v[3];
	__HD__ Vtx()
	{}
	__HD__ Vtx(const Vtx &vtx) : v{ vtx.v[0], vtx.v[1], vtx.v[2] }
	{}
	__HD__ Vtx(float a, float b, float c) : v{ a, b, c }
	{}
};
struct VtxExtra {
	float v[3];
	__HD__ VtxExtra()
	{}
	__HD__ VtxExtra(const VtxExtra &vtx) : v{ vtx.v[0], vtx.v[1], vtx.v[2] }
	{}
	__HD__ VtxExtra(float a, float b, float c) : v{ a, b, c }
	{}
};


__D__ void g_mkray(float *rayorg, float *raydir, int x, int y, int w, int h, float ox, float oy, float oz, const float *M, float fov = 3)
{
	float a = w / h; // assuming width > height 
	float Px = (2 * ((x + 0.5) / w) - 1) * tan(fov / 2 * M_PI / 180) * a;
	float Py = (1 - 2 * ((y + 0.5) / h)) * tan(fov / 2 * M_PI / 180);


// 	raydir[0] = Px;
// 	raydir[1] = Py;
// 	raydir[2] = -1;

	float indir[] = { Px, Py, -1 };
	for (int i = 0; i < 3; ++i) {
		float acc = 0;
		for (int j = 0; j < 3; ++j) {
			acc += M[i * 3 + j] * indir[j];
		}
		raydir[i] = acc;
	}

	rayorg[0] = ox;
	rayorg[1] = oy;
	rayorg[2] = oz;
// 	ray.normalize();
}



template <typename T>
__D__ void swap(T &a, T &b)
{
	T t;
	t = a;
	a = b;
	b = t;
}
__D__ void intersect_bounding_planes_native(float &t1, float &t2, float min, float max, int axis, const float *rayorg, const float *raydir)
{
	float dirfrac = 1.f / (axis == 0 ? raydir[0] : axis == 1 ? raydir[1] : raydir[2]);

	float ro = axis == 0 ? rayorg[0] : axis == 1 ? rayorg[1] : rayorg[2];
	t1 = (min - ro) * dirfrac;
	t2 = (max - ro) * dirfrac;
	if (t1 > t2) {
		swap(t1, t2);
	}
}

struct BoundsBVH {
	const float *bounds;
	__D__ BoundsBVH(const float *_bounds) : bounds(_bounds)
	{}
	__D__ void intersect(float &t1l, float &t2l, float &t1r, float &t2r, uint32_t idx, const float *rayorg, const float *raydir) const
	{
		const float *b = bounds + idx * 12;
		float q, w, e, r;
		t1l = FLT_MIN; t2l = FLT_MAX;
		t1r = FLT_MIN; t2r = FLT_MAX;
		for (int axis = 0; axis < 3; ++axis) {
			intersect_bounding_planes_native(q, w, b[axis], b[axis + 3], axis, rayorg, raydir); // TODO early return
			t1l = max(t1l, q);
			t2l = min(t2l, w);
			if (t1l > t2l) break;
		}
		for (int axis = 0; axis < 3; ++axis) {
			intersect_bounding_planes_native(e, r, b[axis + 6], b[axis + 9], axis, rayorg, raydir);
			t1r = max(t1r, e);
			t2r = min(t2r, r);
			if (t1r > t2r) break;
		}
	}
};

struct HitPoint {
	uint32_t idx;
	float u, v;
};
struct LeavesBVH {
	const FaceG *tris;
	const Vtx *vtx;
	int nleafesmax;
	__D__ LeavesBVH(const FaceG *_tris, const Vtx *_vtx, int nleafesmax) : tris(_tris), vtx(_vtx), nleafesmax(nleafesmax)
	{}
	__D__ uint32_t get_off(uint32_t li) const
	{
		return li * nleafesmax;
	}
	__D__ bool intersect(float &t, HitPoint *hitpoint, uint32_t idx, uint32_t nchilds, const float *rayorg, const float *raydir) const
	{
		FaceG f = tris[idx];
		Vtx a = vtx[f.idx[0]];
		Vtx b = vtx[f.idx[1]];
		Vtx c = vtx[f.idx[2]];

		float u, vv;
		if (!tri_intersect(t, u, vv, rayorg, raydir, a.v, b.v, c.v)) return false;

		hitpoint->idx = idx;
		hitpoint->u = u;
		hitpoint->v = vv;

		return true;
	}
};



struct StackEntry3 {
	float t0, t1;
	uint32_t idx, leaves;
	__D__ StackEntry3()
	{}
	__D__ StackEntry3(float _t0, float _t1, uint32_t _idx, uint32_t _leaves) : t0(_t0), t1(_t1), idx(_idx), leaves(_leaves)
	{}
};
template <typename Subtrees, typename Bounds, typename Leaves, int L>
__D__ inline bool trace3(float &t, HitPoint *hitpoint, const Subtrees &subtrees /* subtree sizes */, const Bounds &bounds, const Leaves &leaves, const float *rayorg, const float *raydir)
{
	bool hit = false;
	float t0 = 0, t1 = FLT_MAX;

	StackEntry3 stack[128];

	uint32_t ni = 0, li = 0, top = 0;
	while (1) {
		uint32_t st = subtrees[ni];
		uint32_t axis = st >> 30, left_subtree = st & 0x3fffffffu;
#ifdef __CUDACC__
		__syncthreads();
#endif

		if (axis == 3) {

			uint32_t nn = left_subtree;

			uint32_t off = leaves.get_off(li);
			for (int i = 0; i < nn; ++i) {
				float tt = FLT_MAX;
				HitPoint hitpoint_tmp;
				bool bhit = leaves.intersect(tt, &hitpoint_tmp, off + i, nn, rayorg, raydir);
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
			bounds.intersect(t0l, t1l, t0r, t1r, bi, rayorg, raydir);
			t0l = max(t0l, t0);
			t1l = min(t1l, t1);
			t0r = max(t0r, t0);
			t1r = min(t1r, t1);

			if (t0l > t1l || t0l > t0r) {
				swap(t0l, t0r);
				swap(t1l, t1r);
				swap(cl, cr);
				swap(ll, lr);
			} else {
			}
			if (!(t0r > t1r)) {
				StackEntry3 e = StackEntry3(t0r, t1r, cr, lr);
				stack[top] = e;
				++top;
			} else {
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
			StackEntry3 e = stack[top];
			ni = e.idx;
			li = e.leaves;
			t0 = e.t0;
			t1 = e.t1;
		} while (t0 > t);
	}
	return false;
}


__D__ void fragment_shader(const float *vin, const float *light, float *colout, bool hit_shadow)
{
	float x = vin[0], y = vin[1], z = vin[2];
	float nx = vin[3], ny = vin[4], nz = vin[5];

	float lx = light[0] - x, ly = light[1] - y, lz = light[2] - z;
	float ll = sqrt(lx * lx + ly * ly + lz * lz);
	if (ll != 0.f) { lx /= ll; ly /= ll; lz /= ll; }

	float dot = fabs(nx * lx + ny * ly + nz * lz);
	colout[0] = min(max(1.f * dot, 0.f), 1.f) - (hit_shadow ? 0.5 : 0);
}


#ifdef __CUDACC__
__global__
#endif
void TraceKernel(int x, int y, uint8_t *framebuf, const uint32_t *subtrees, const float *bounds, const FaceG *faces, const Vtx *vtx, const VtxExtra *ve, uint32_t w, uint32_t h, camera cam, int nleafesmax)
{
#ifdef __CUDACC__
	x = blockDim.x * (0+blockIdx.x) + threadIdx.x; y = blockDim.y * (0+blockIdx.y) + threadIdx.y;
	if (x >= w || y >= h) return;
#endif

	static const float light[] = { 50, 220, 1140 };

	float rayorg[3], raydir[3];
	g_mkray(rayorg, raydir, x, y, w, h, cam.x, cam.y, cam.z, cam.mat, cam.fov/*, 0, 200, 10000*//*, 0, 0, 4000*/);

	float t = FLT_MAX;
	HitPoint hitpoint;

	BoundsBVH bo(bounds);
	LeavesBVH lv(faces, vtx, nleafesmax);

	bool hit = trace3<const uint32_t*, BoundsBVH, LeavesBVH, 5 * 4>(t, &hitpoint, subtrees, bo, lv, rayorg, raydir);

	float res = 1;
	if (hit) {
		float u = hitpoint.u;
		float v = hitpoint.v;
		uint32_t idx = hitpoint.idx;

		FaceG f = lv.tris[idx];

		// load hit vertices completely
		Vtx v0 = lv.vtx[f.idx[0]];
		Vtx v1 = lv.vtx[f.idx[1]];
		Vtx v2 = lv.vtx[f.idx[2]];
		VtxExtra v0e = ve[f.idx[0]];
		VtxExtra v1e = ve[f.idx[1]];
		VtxExtra v2e = ve[f.idx[2]];

		// lerp
		float vertex[6];
		for (int i = 0; i < 3; ++i) {
			vertex[i] = v0.v[i] * (1.f - u - v) + v1.v[i] * u + v2.v[i] * v;
			vertex[3 + i] = v0e.v[i] * (1.f - u - v) + v1e.v[i] * u + v2e.v[i] * v;
		}
		bool hit_shadow = false;

		fragment_shader(vertex, light, &res, hit_shadow);
	}
	framebuf[y * w + x] = res * 255;
}


void trace_gpu_sah(uint8_t *framebuf, uint32_t *subtrees, float *bounds, FaceG *faces, Vtx *vtx, VtxExtra *vtxextra, uint32_t w, uint32_t h, uint32_t maxlvl, camera cam, int nleafesmax)
{
	std::cout << "Max lvl: " << maxlvl << " " << maxlvl * sizeof(StackEntry3) << std::endl;

	std::cout << "Sizes: " << sizeof(FaceG) << " " << sizeof(Vtx) << " " << sizeof(VtxExtra) << std::endl;

#ifdef __CUDACC__
	dim3 blockd(8, 8);
	dim3 gridd((w + blockd.x - 1) / blockd.x, (h + blockd.y - 1) / blockd.y);
	std::cout << "Max lvl: " << maxlvl << " " << maxlvl * sizeof(StackEntry3) << std::endl;

	TraceKernel<<<gridd, blockd>>>(0, 0, framebuf, subtrees, bounds, faces, vtx, vtxextra, w, h, cam, nleafesmax);
#else
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			TraceKernel(x, y, framebuf, subtrees, bounds, faces, vtx, vtxextra, w, h, cam, nleafesmax);
		}
	}
#endif
}


void *my_malloc(std::size_t s, int desc)
{
	void *p;
#ifdef __CUDACC__
	cudaMalloc(&p, s);
#else
	p = malloc(s);
#endif
	meminf_describe(p, desc);
	return p;
}
void my_upload(void *dst, const void *src, std::size_t s)
{
#ifdef __CUDACC__
	cudaMemcpy(dst, src, s, cudaMemcpyHostToDevice);
#else
	std::memcpy(dst, src, s);
#endif
}
void my_download(void *dst, const void *src, std::size_t s)
{
#ifdef __CUDACC__
	cudaMemcpy(dst, src, s, cudaMemcpyDeviceToHost);
#else
	std::memcpy(dst, src, s);
#endif
}


int myatoi(const std::string &s)
{
	int v = 1;
	int l = s.size();
	switch (s.back()) {
	case 'm':
	case 'M':
		v *= 1024;
	case 'k':
		v *= 1024;
		--l;
	}
	return v * std::stoi(s.substr(0, l));
}

Mesh loadMesh(const char* name) {
	Mesh mesh;

	happly::PLYData data(name);

	auto& vertex = data.getElement("vertex");
	std::vector<float> x = vertex.getProperty<float>("x");
	std::vector<float> y = vertex.getProperty<float>("y");
	std::vector<float> z = vertex.getProperty<float>("z");

	mesh.vertices.reserve(x.size());
	for (int i = 0; i < x.size(); i++) {
		mesh.vertices.emplace_back(x[i], y[i], z[i]);
	}

	std::vector<std::vector<size_t>> indicesList = data.getFaceIndices<size_t>();
	for (auto& indices : indicesList) {
		// Perform basic triangulation for faces with more than 3 vertices.
		for (int i = 1; i + 1 < indices.size(); i++) {
			mesh.faces.emplace_back(indices[0], indices[i], indices[i+1]);
		}
	}

	return mesh;
}

void trace(const char* name, int x, camera cam, Heuristic heu) {
	image_b test(x, x, 1);
	Mesh mesh = loadMesh(name);
	mesh.compute_normals();


	std::cout << "Mesh statistics: Faces: " << mesh.faces.size() << " Vertices: " << mesh.vertices.size() << std::endl;
	static const int SIZEOF_EXPPATCH = 512;

	BVHBuilder bvhb;

	int nleafesmax = 32;

	int tag = 0;
	if (1||!bvhb.restore(name, tag)) {
		std::vector<float> aabbs(mesh.faces.size() * 6);
		std::vector<float> cens(mesh.faces.size() * 3);
		for (uint32_t i = 0; i < mesh.faces.size(); ++i) {
			const Face &face = mesh.faces[i];
			const Vertex &v0 = mesh.vertices[face.idx[0]];
			const Vertex &v1 = mesh.vertices[face.idx[1]];
			const Vertex &v2 = mesh.vertices[face.idx[2]];
			
			AABB aabb;
			aabb.feed(v0.pos);
			aabb.feed(v1.pos);
			aabb.feed(v2.pos);

			for (int j = 0; j < 3; ++j) {
				aabbs[i * 6 + j] = aabb.min[j];
				aabbs[i * 6 + j + 3] = aabb.max[j];
				cens[i * 3 + j] = (v0.pos[j] + v1.pos[j] + v2.pos[j]) / 3;
			}
		}

		bvhb.construct(cens.data(), aabbs.data(), mesh.faces.size(), nleafesmax, heu);

		bvhb.backup(name, tag);
	} else {
		std::cout << "Found a BVH backup!" << std::endl;
	}

	std::cout << "Got " << bvhb.num_nodes() << " nodes; bounds: " << bvhb.bounds.size() / 4 << " sum: " << bvhb.bounds.size() / 4 << std::endl;

	uint8_t *framebuf = (uint8_t*)my_malloc(test.width() * test.height(), 0);
	uint32_t *d_subtrees = (uint32_t*)my_malloc(bvhb.subtrees.size() * 4, 1);
// 	std::cout << "x" << std::endl;
	my_upload(d_subtrees, (const char*)bvhb.subtrees.data(), bvhb.subtrees.size() * 4 /* TODO: here was a *3 */);
// 	std::cout << "y" << std::endl;
	std::vector<Face> trispermuted(bvhb.leaf_nodes.size());
	float *d_bounds = (float*)my_malloc(bvhb.bounds.size() * 4 * 6, 2);
	my_upload(d_bounds, (const char*)bvhb.bounds.data(), bvhb.bounds.size() * 4 * 6);

	FaceG *d_tris = (FaceG*)my_malloc(trispermuted.size() * 4 * 3, 3);
	Vtx *d_vtx = (Vtx*)my_malloc(mesh.vertices.size() * sizeof(Vtx), 4);
	VtxExtra *d_vtxextra = (VtxExtra*)my_malloc(mesh.vertices.size() * sizeof(VtxExtra), 5);


	std::vector<Vtx> vtx(mesh.vertices.size());
	std::vector<VtxExtra> vtxextra(mesh.vertices.size());

	std::cout << bvhb.leaf_nodes.size() << " " << mesh.faces.size() << " " << bvhb.leaf_nodes.size() * sizeof(Face) << std::endl;
	for (int i = 0; i < bvhb.leaf_nodes.size(); ++i) {
		uint32_t f = bvhb.leaf_nodes[i];
		if (f == -1u) trispermuted[i] = Face(0, 0, 0);
		else trispermuted[i] = mesh.faces[bvhb.leaf_nodes[i]];
// 		std::cout << trispermuted[i].idx[0] << " " << trispermuted[i].idx[1] << " " << trispermuted[i].idx[2] << std::endl;
	}
	for (int i = 0; i < mesh.vertices.size(); ++i) {
		vtx[i] = Vtx(mesh.vertices[i].pos[0], mesh.vertices[i].pos[1], mesh.vertices[i].pos[2]);
		vtxextra[i] = VtxExtra{ mesh.vertices[i].pos[3], mesh.vertices[i].pos[4], mesh.vertices[i].pos[5] };
	}

	my_upload(d_tris, (const char*)trispermuted.data(), trispermuted.size() * 4 * 3);
	my_upload(d_vtx, (const char*)vtx.data(), vtx.size() * sizeof(Vtx));
	my_upload(d_vtxextra, (const char*)vtxextra.data(), vtxextra.size() * sizeof(VtxExtra));

	std::cout << "Starting renderer" << std::endl;

	trace_gpu_sah(framebuf, d_subtrees, d_bounds, d_tris, d_vtx, d_vtxextra, test.width(), test.height(), bvhb.maxlvl, cam, nleafesmax);

	std::cout << "Download" << std::endl;
	my_download((char*)test.data(), framebuf, test.width() * test.height());

	std::cout << "Original mesh size: " << mesh.faces.size() << std::endl;
	std::cout << "Leaf triangles: " << bvhb.leaf_nodes.size() << std::endl;
	image_io::save(test, "test.png");
}

int main(int argc, const char **argv)
{
	camera cam{ std::atof(argv[3]), std::atof(argv[4]), std::atof(argv[5]), {
		std::atof(argv[6]), std::atof(argv[7]), std::atof(argv[8]),
		std::atof(argv[9]), std::atof(argv[10]), std::atof(argv[11]),
		std::atof(argv[12]), std::atof(argv[13]), std::atof(argv[14])
	}, std::atof(argv[15]) };

	int x = myatoi(argv[2]);
	const char *name = argv[1];
	trace(name, x, cam, argv[16] == std::string("sah") ? SAH : MEDIAN);
}


