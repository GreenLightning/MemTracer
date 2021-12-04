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
#include <stdio.h>

#include "utils/channel.hpp"
#include "utils/utils.h"

#include "common.h"

//
// Original code shipped with NVBit for comparison purposes:
//
// extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id,
//                                                        uint64_t addr,
//                                                        uint64_t grid_launch_id,
//                                                        uint64_t pchannel_dev) {
//     /* if thread is predicated off, return */
//     if (!pred) {
//         return;
//     }

//     int active_mask = __ballot_sync(__activemask(), 1);
//     const int laneid = get_laneid();
//     const int first_laneid = __ffs(active_mask) - 1;

//     mem_access_t ma;

//     /* collect memory address information from other threads */
//     for (int i = 0; i < 32; i++) {
//         ma.addrs[i] = __shfl_sync(active_mask, addr, i);
//     }

//     int4 cta = get_ctaid();
//     ma.grid_launch_id = grid_launch_id;
//     ma.cta_id_x = cta.x;
//     ma.cta_id_y = cta.y;
//     ma.cta_id_z = cta.z;
//     ma.warp_id = get_warpid();
//     ma.opcode_id = opcode_id;

//     /* first active lane pushes information on the channel */
//     if (first_laneid == laneid) {
//         ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
//         channel_dev->push(&ma, sizeof(mem_access_t));
//     }
// }
//


extern "C" __device__ __noinline__ void instrument_mem(int32_t pred, uint64_t instr_addr, uint64_t addr, uint64_t grid_launch_id, uint64_t pchannel_dev) {
	// In SASS and PTX all instructions can be conditionally executed depending on a predicate register (1 bit per thread).
	// We have to check the predicate value, so that we do not record memory accesses for threads that are predicated off
	// and do not actually execute the instruction.
	// See: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#predicated-execution
	// See: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#register-state-space
	if (!pred) return;

	// In the case of incidental divergence with independent thread scheduling (on Volta and higher),
	// this code will push multiple memory access records onto the channel.
	// See: High-level blog post
	//      https://developer.nvidia.com/blog/inside-volta/
	// See: Low-level blog post
	//      https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
	// See: Forum post
	//      https://coderedirect.com/questions/280642/activemask-vs-ballot-sync

	// Note that the use of __ballot_sync() in the original code has no effect,
	// because predicate is always non-zero and all threads in the mask are active,
	// since __activemask() was used as the mask for __ballot_sync(),
	// therefore the result is just __activemask() itself.
	const int mask = __activemask();
	const int leader = __ffs(mask) - 1;
	const int lane = get_laneid();

	// Compute the warp id inside the current block, so that each memory
	// access can be mapped back to a logical thread.
	const int local_thread_id = (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + threadIdx.x;
	const int local_warp_id = local_thread_id / 32;

	mem_access_t ma;

	// Collect memory address information from other threads.
	for (int i = 0; i < 32; i++) {
		ma.addrs[i] = __shfl_sync(mask, addr, i);
	}

	ma.instr_addr = instr_addr;
	ma.grid_launch_id = grid_launch_id;
	ma.block_idx_z = blockIdx.z;
	ma.block_idx_y = blockIdx.y;
	ma.block_idx_x = blockIdx.x;
	ma.local_warp_id = local_warp_id;

	// Push information on the channel from the first active lane.
	if (lane == leader) {
		ChannelDev* channel_dev = (ChannelDev*) pchannel_dev;
		channel_dev->push(&ma, sizeof(mem_access_t));
	}
}
