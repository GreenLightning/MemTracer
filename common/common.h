#include <stdint.h>

struct file_header_t {
	uint32_t magic;
	uint32_t version;

	// Version 4: file contains: <file_header> <trace_header> <data>; header_size = sizeof(file_header_t) + sizeof(trace_header_t);
	// Version 5: file contains: <file_header> <data> <trace_header>; header_size = sizeof(trace_header_t);
	uint64_t header_size;
};

struct trace_header_t {
	uint8_t  hash[16];

	uint64_t mem_access_size;
	uint64_t mem_access_count;
	uint64_t mem_access_offset;

	uint64_t launch_info_size;
	uint64_t launch_info_count;
	uint64_t launch_info_offset;

	uint64_t mem_region_size;
	uint64_t mem_region_count;
	uint64_t mem_region_offset;

	uint64_t addr_info_size;
	uint64_t addr_info_count;
	uint64_t addr_info_offset;

	uint64_t strings_offset;
	uint64_t strings_size;

	uint64_t mem_contents_offset;
	uint64_t mem_contents_size;
};

struct mem_access_t {
	uint64_t instr_addr;
	uint64_t grid_launch_id;
	int32_t block_idx_z;
	int32_t block_idx_y;
	int32_t block_idx_x;
	int32_t local_warp_id;
	uint64_t addrs[32];
};

struct launch_info_t {
	uint64_t grid_launch_id;
	int32_t grid_dim_x;
	int32_t grid_dim_y;
	int32_t grid_dim_z;
	int32_t block_dim_x;
	int32_t block_dim_y;
	int32_t block_dim_z;
};

struct mem_region_t {
	uint64_t grid_launch_id;
	uint64_t mem_region_id;
	uint64_t start;
	uint64_t size;
	uint64_t contents_offset;
};

struct addr_info_t {
	uint64_t addr;
	int32_t  opcode_string_index;
};
