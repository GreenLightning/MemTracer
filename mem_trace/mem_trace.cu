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

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <map>
#include <sstream>
#include <string>
#include <unordered_set>

// Every tool needs to include this once.
#include "nvbit_tool.h"

#include "nvbit.h"
#include "utils/channel.hpp"
#include "meminf_data.h"
#include "common.h"

#define CHANNEL_SIZE (1l << 20)

struct context_state_t {
	ChannelDev* channel_dev;
	ChannelHost channel_host;
};

// mutex protects all global data of this tool.
pthread_mutex_t mutex;
// skip_callback_flag is used to prevent instrumenting kernel calls from the tool itself.
bool skip_callback_flag = false;

std::unordered_map<CUcontext, context_state_t*> ctx_state_map;
std::unordered_set<CUfunction> already_instrumented;
std::map<uint64_t, std::string> addr_to_opcode_map;
uint64_t grid_launch_id = 0;
int next_channel_id = 0;

// Global control variables for this tool.
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

void nvbit_at_init() {
	setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
	GET_VAR_INT(
		instr_begin_interval, "INSTR_BEGIN", 0,
		"Beginning of the instruction interval where to apply instrumentation");
	GET_VAR_INT(
		instr_end_interval, "INSTR_END", UINT32_MAX,
		"End of the instruction interval where to apply instrumentation");
	GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
	std::string pad(100, '-');
	printf("%s\n", pad.c_str());

	// Configure mutex to be recursive.
	pthread_mutexattr_t attr;
	pthread_mutexattr_init(&attr);
	pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
	pthread_mutex_init(&mutex, &attr);
}

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
	assert(ctx_state_map.find(ctx) != ctx_state_map.end());
	context_state_t* ctx_state = ctx_state_map[ctx];

	// Related functions are the functions that can be called by the kernel.
	std::vector<CUfunction> functions_to_instrument = nvbit_get_related_functions(ctx, func);
	functions_to_instrument.push_back(func);

	for (auto f : functions_to_instrument) {
		bool inserted_successfully = already_instrumented.insert(f).second;
		if (!inserted_successfully) continue;

		const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
		uint64_t func_addr = nvbit_get_func_addr(f);

		if (verbose) {
			printf("MEMTRACE: CTX %p, Inspecting CUfunction %p name %s at address 0x%lx\n",
				ctx, f, nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f));
		}

		uint32_t cnt = 0;
		for (auto instr : instrs) {
			if (cnt < instr_begin_interval || cnt >= instr_end_interval ||
				instr->getMemorySpace() == InstrType::MemorySpace::NONE ||
				instr->getMemorySpace() == InstrType::MemorySpace::CONSTANT) {
				cnt++;
				continue;
			}

			if (verbose) instr->printDecoded();

			uint32_t offset = instr->getOffset();
			uint64_t instr_addr = func_addr + static_cast<uint64_t>(offset);
			addr_to_opcode_map[instr_addr] = std::string(instr->getOpcode());

			int mref_idx = 0;
			for (int i = 0; i < instr->getNumOperands(); i++) {
				const InstrType::operand_t* op = instr->getOperand(i);
				if (op->type == InstrType::OperandType::MREF) {
					nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
					nvbit_add_call_arg_guard_pred_val(instr);
					nvbit_add_call_arg_const_val64(instr, instr_addr);
					nvbit_add_call_arg_mref_addr64(instr, mref_idx);
					// grid_launch_id is a dynamic argument, that is set using nvbit_set_at_launch below.
					nvbit_add_call_arg_launch_val64(instr, 0);
					nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
					mref_idx++;
				}
			}

			cnt++;
		}
	}
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char* name, void* params, CUresult* pStatus) {
	pthread_mutex_lock(&mutex);
	if (skip_callback_flag) {
		pthread_mutex_unlock(&mutex);
		return;
	}
	skip_callback_flag = true;

	assert(ctx_state_map.find(ctx) != ctx_state_map.end());
	context_state_t* ctx_state = ctx_state_map[ctx];

	if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
		cbid == API_CUDA_cuLaunchKernel) {
		cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;

		// Make sure GPU is idle.
		cudaDeviceSynchronize();
		assert(cudaGetLastError() == cudaSuccess);

		if (!is_exit) {
			instrument_function_if_needed(ctx, p->f);

			int nregs = 0;
			CUDA_SAFECALL(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f));

			int shmem_static_nbytes = 0;
			CUDA_SAFECALL(cuFuncGetAttribute(&shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, p->f));

			const char* func_name = nvbit_get_func_name(ctx, p->f);
			uint64_t func_addr = nvbit_get_func_addr(p->f);

			nvbit_set_at_launch(ctx, p->f, &grid_launch_id, sizeof(uint64_t));
			grid_launch_id++;

			nvbit_enable_instrumented(ctx, p->f, true);

			printf(
				"MEMTRACE: CTX 0x%016lx - LAUNCH - Kernel 0x%016lx - Kernel "
				"name %s - grid launch id %ld - grid size %d,%d,%d - block "
				"size %d,%d,%d - nregs %d - shmem %d - cuda stream id %ld\n",
				(uint64_t)ctx, func_addr, func_name, grid_launch_id, p->gridDimX,
				p->gridDimY, p->gridDimZ, p->blockDimX, p->blockDimY,
				p->blockDimZ, nregs, shmem_static_nbytes + p->sharedMemBytes,
				(uint64_t)p->hStream
			);

			// fill allocation sizes from the driver API
			const char *names[] = { "framebuf", "nodes", "aabbs", "faces", "vtxpos", "vtxnrm" };
			for (auto &x : meminfs) {
				size_t size;
				if (cuPointerGetAttribute(&size, CU_POINTER_ATTRIBUTE_RANGE_SIZE, x.first) != CUDA_SUCCESS) {
					std::cerr << "Invalid meminf pointer" << std::endl;
					std::exit(1);
				}
				x.second.size = size;
			}
			for (auto x : meminfs) {
				std::cout << x.first << " +" << x.second.size << ": " << names[x.second.desc] << std::endl;
			}
		}
	}

	skip_callback_flag = false;
	pthread_mutex_unlock(&mutex);
}

void* recv_thread_func(void* args) {
	CUcontext ctx = (CUcontext)args;

	pthread_mutex_lock(&mutex);
	assert(ctx_state_map.find(ctx) != ctx_state_map.end());
	context_state_t* ctx_state = ctx_state_map[ctx];
	ChannelHost* ch_host = &ctx_state->channel_host;
	pthread_mutex_unlock(&mutex);

	char* recv_buffer = (char*)malloc(CHANNEL_SIZE);
	char* print_buffer = (char*)malloc(4 * 1024);

	bool done = false;
	while (!done) {
		uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
		uint32_t num_processed_bytes = 0;
		while (num_processed_bytes < num_recv_bytes) {
			mem_access_t* ma = (mem_access_t*) &recv_buffer[num_processed_bytes];

			// Check for signal from flush_channel that this was the last message.
			if (ma->cta_id_x == -1) {
				done = true;
				break;
			}

			int length = 0;
			length += sprintf(print_buffer+length, "MEMTRACE: ");
			length += sprintf(print_buffer+length,
				"CTX 0x%016lx - grid_launch_id %ju - %s 0x%016lx - CTA %d,%d,%d - warp %d - ",
				(uint64_t) ctx, ma->grid_launch_id, addr_to_opcode_map[ma->instr_addr].c_str(), ma->instr_addr,
				ma->cta_id_x, ma->cta_id_y, ma->cta_id_z, ma->warp_id
			);
			for (int i = 0; i < 32; i++) {
				length += sprintf(print_buffer+length, "0x%016lx ", ma->addrs[i]);
			}
			length += sprintf(print_buffer+length, "\n");
			fwrite(print_buffer, length, 1, stdout);

			num_processed_bytes += sizeof(mem_access_t);
		}
	}

	free(recv_buffer);
	free(print_buffer);
	return NULL;
}

__global__ void flush_channel(ChannelDev* ch_dev) {
	mem_access_t ma = {};
	ma.cta_id_x = -1;
	ch_dev->push(&ma, sizeof(mem_access_t));
	ch_dev->flush();
}

void nvbit_at_ctx_init(CUcontext ctx) {
	pthread_mutex_lock(&mutex);

	if (verbose) printf("MEMTRACE: STARTING CONTEXT %p\n", ctx);
	
	context_state_t* ctx_state = (context_state_t*) malloc(sizeof(context_state_t));
	assert(ctx_state_map.find(ctx) == ctx_state_map.end());
	ctx_state_map[ctx] = ctx_state;
	
	cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
	ctx_state->channel_host.init(next_channel_id, CHANNEL_SIZE, ctx_state->channel_dev, recv_thread_func, ctx);
	nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
	next_channel_id++;

	pthread_mutex_unlock(&mutex);
}

void nvbit_at_ctx_term(CUcontext ctx) {
	pthread_mutex_lock(&mutex);
	skip_callback_flag = true;

	if (verbose) printf("MEMTRACE: TERMINATING CONTEXT %p\n", ctx);

	assert(ctx_state_map.find(ctx) != ctx_state_map.end());
	context_state_t* ctx_state = ctx_state_map[ctx];
	flush_channel<<<1, 1>>>(ctx_state->channel_dev);

	// Make sure flush of channel is complete.
	cudaDeviceSynchronize();
	assert(cudaGetLastError() == cudaSuccess);

	ctx_state->channel_host.destroy(false);
	cudaFree(ctx_state->channel_dev);
	ctx_state_map.erase(ctx);
	free(ctx_state);

	skip_callback_flag = false;
	pthread_mutex_unlock(&mutex);
}
