#pragma once

typedef __attribute__((packed)) struct {
    uint32_t cta_id_x;
    uint32_t cta_id_y;
    uint32_t cta_id_z;
    uint32_t warp_id;
    uint32_t opcode_id;
    uint64_t addrs[32];
	uint32_t preds;
// 	uint32_t ctaid;
// 	char isexit;
} mem_access_t;

struct __attribute__((packed)) memop {
	enum MemorySpace {
		LOCAL,             // local memory operation
		GENERIC,           // generic memory operation
		GLOBAL,            // global memory operation
		SHARED,            // shared memory operation
// 		CONSTANT,          // constant memory operation
		GLOBAL_TO_SHARED,  // read from global memory then write to shared memory
		SURFACE,   // surface memory operation
		TEXTURE,   // texture memory operation
	};
	enum Flags { NONCOHERENT = 1 };
	char ld, st;
	MemorySpace ms;
	uint8_t flags;
	int s; // 1 2 4 8 16
};
