/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>

struct header_t {
	uint32_t magic;
	uint32_t version;
	uint64_t header_size;

	uint8_t  hash[16];

	uint64_t mem_access_size;
	uint64_t mem_access_count;
	uint64_t mem_access_offset;

	uint64_t mem_region_size;
	uint64_t mem_region_count;
	uint64_t mem_region_offset;

	uint64_t addr_info_size;
	uint64_t addr_info_count;
	uint64_t addr_info_offset;

	uint64_t opcode_offset;
	uint64_t opcode_size;
};

struct mem_access_t {
	uint64_t instr_addr;
	uint64_t grid_launch_id;
	int32_t cta_id_x;
	int32_t cta_id_y;
	int32_t cta_id_z;
	int32_t warp_id;
	uint64_t addrs[32];
};

struct mem_region_t {
	uint64_t grid_launch_id;
	uint64_t start;
	uint64_t size;
	int32_t  description;
};

struct addr_info_t {
	uint64_t addr;
	uint64_t opcode_offset;
};
