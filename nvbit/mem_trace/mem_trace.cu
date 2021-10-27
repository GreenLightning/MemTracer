/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2021, Max von Buelow, GRIS, Technical University of Darmstadt
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *	notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *	notice, this list of conditions and the following disclaimer in the
 *	documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *	contributors may be used to endorse or promote products derived
 *	from this software without specific prior written permission.
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
#include <string>
#include <map>
#include <unordered_set>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include "vec.h"
#include "meminf_storage.h"

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "hiacc.h"



/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;
std::map<int, memop> memops;

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

// 	std::string logfn;
// 	GET_VAR_STR(logfn, "OUTFILE", "Filename of the output trace");
// 	if (logfn.empty()) {
// 		std::cerr << "ERROR: No log file specified." << std::endl;
// 		std::exit(1);
// 	}
// 	os.open(logfn.c_str());
}
/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;


memop parse_op(Instr &instr)
{
	memop res;
	res.ld = instr.isLoad();
	res.st = instr.isStore();
	memop::MemorySpace map[128];
	map[(int)InstrType::MemorySpace::LOCAL] = memop::LOCAL;
	map[(int)InstrType::MemorySpace::GENERIC] = memop::GENERIC;
	map[(int)InstrType::MemorySpace::GLOBAL] = memop::GLOBAL;
	map[(int)InstrType::MemorySpace::SHARED] = memop::SHARED;
// 	map[(int)InstrType::MemorySpace::CONSTANT] = memop::CONSTANT;
	map[(int)InstrType::MemorySpace::GLOBAL_TO_SHARED] = memop::GLOBAL_TO_SHARED;
	map[(int)InstrType::MemorySpace::SURFACE] = memop::SURFACE;
	map[(int)InstrType::MemorySpace::TEXTURE] = memop::TEXTURE;
	res.ms = map[(int)instr.getMemorySpace()];   
	res.s = instr.getSize();
	res.flags = 0;
	if (res.ms == memop::GLOBAL) {
		const char *op = instr.getOpcode();
		const char *search = "CONSTANT";
		int state = 0;
		for (int i = 0; i == 0 || op[i - 1]; ++i) {
			if (state != -1) {
				if ((op[i] == '.' || op[i] == '\0') && state == 8) res.flags |= memop::NONCOHERENT;
				if (op[i] == search[state]) {
					++state;
				} else {
					state = -1;
				}
			} else {
				if (op[i] == '.') state = 0;
			}
		}
	}
	return res;
}

class Instr2 {
  public:

	/* Reserved variable used internally by NVBit */
	const void* reserved;
};

uint32_t nctax, nctay, nctaz;
std::size_t taboff;
uint64_t min_a = std::numeric_limits<uint64_t>::max();
uint64_t max_a = std::numeric_limits<uint64_t>::min();
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
	/* Get related functions of the kernel (device function that can be
	 * called by the kernel) */
	std::vector<CUfunction> related_functions =
		nvbit_get_related_functions(ctx, func);

	/* add kernel itself to the related function vector */
	related_functions.push_back(func);

	/* iterate on function */
	for (auto f : related_functions) {
		/* "recording" function was instrumented, if set insertion failed
		 * we have already encountered this function */
		if (!already_instrumented.insert(f).second) {
			continue;
		}
		const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
		if (verbose) {
			printf("Inspecting function %s at address 0x%lx\n",
				   nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f));
		}

		uint32_t cnt = 0;
		/* iterate on all the static instructions in the function */
		for (auto instr : instrs) {
// 			std::cout << i2.reserved << std::endl;
// 			if (std::string(instr->getOpcode()) == "EXIT") {
//				 instr->printDecoded();
// 				nvbit_insert_call(instr, "instrument_exit", IPOINT_BEFORE);
// 				nvbit_add_call_arg_pred_val(instr);
// 				nvbit_add_call_arg_const_val64(instr,
// 												(uint64_t)&channel_dev);
// // 			std::cout << instr->getOpcode() << std::endl;
// 			}

			if (cnt < instr_begin_interval || cnt >= instr_end_interval ||
				instr->getMemorySpace() == InstrType::MemorySpace::NONE ||
				instr->getMemorySpace() == InstrType::MemorySpace::CONSTANT) {
				cnt++;
				continue;
			}

			if (opcode_to_id_map.find(instr->getOpcode()) ==
				opcode_to_id_map.end()) {
				uint32_t opcode_id = opcode_to_id_map.size();
				opcode_to_id_map[instr->getOpcode()] = opcode_id;
				const char *opstr = instr->getOpcode();
				uint32_t opstrlen = std::strlen(opstr);
				id_to_opcode_map[opcode_id] = std::string(opstr);
// 				os.write((const char*)&opcode_id, 4);
// 				os.write(opstr, opstrlen + 1);
				memop mop = parse_op(*instr);
				memops[opcode_id] = mop;
// 				os.write((const char*)&mop, sizeof(memop));
			}

			int opcode_id = opcode_to_id_map[instr->getOpcode()];
			int mref_idx = 0;
			/* iterate on the operands */
			for (int i = 0; i < instr->getNumOperands(); i++) {
				/* get the operand "i" */
				const InstrType::operand_t* op = instr->getOperand(i);

				if (op->type == InstrType::OperandType::MREF) {
					/* insert call to the instrumentation function with its
					 * arguments */
					nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
				instr->printDecoded();
					/* predicate value */
					nvbit_add_call_arg_guard_pred_val(instr);
					/* opcode id */
					nvbit_add_call_arg_const_val32(instr, opcode_id);
					/* memory reference 64 bit address */
					nvbit_add_call_arg_mref_addr64(instr, mref_idx);
					/* add pointer to channel_dev*/
					nvbit_add_call_arg_const_val64(instr,
												   (uint64_t)&channel_dev);
					mref_idx++;
				}
			}
			cnt++;
		}
//		 std::exit(1);
	}
	memop mop;
	mop.ld = mop.st = 0;
// 	os.write((const char*)&mop, sizeof(memop));

// 	std::cout << nctax<<nctay<<nctaz <<std::endl;
}

__global__ void flush_channel() {
	/* push memory access with negative cta id to communicate the kernel is
	 * completed */
	mem_access_t ma;
	ma.cta_id_x = -1;
	channel_dev.push(&ma, sizeof(mem_access_t));

	/* flush channel */
	channel_dev.flush();
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
						 const char *name, void *params, CUresult *pStatus) {
	if (skip_flag) return;

	if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
		cbid == API_CUDA_cuLaunchKernel) {
		cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

		if (!is_exit) {
			int nregs;
			CUDA_SAFECALL(
				cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f));

			int shmem_static_nbytes;
			CUDA_SAFECALL(
				cuFuncGetAttribute(&shmem_static_nbytes,
								   CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, p->f));

			int lmem_static_nbytes;
			CUDA_SAFECALL(
				cuFuncGetAttribute(&lmem_static_nbytes,
								   CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, p->f));

			instrument_function_if_needed(ctx, p->f);

			nvbit_enable_instrumented(ctx, p->f, true);

			nctax = p->gridDimX;
			nctay = p->gridDimY;
			nctaz = p->gridDimZ;


// 			uint64_t lmem_base = nvbit_get_local_mem_base_addr(ctx);
// 			std::cout << "LMEM: " << lmem_base << " +" << lmem_static_nbytes << " per thread" << std::endl;
// 			uint32_t _vals[] = { nctax, nctay, nctaz, p->blockDimX, p->blockDimY, p->blockDimZ, nregs, shmem_static_nbytes + p->sharedMemBytes, lmem_static_nbytes };


			printf(
				"Kernel %s - grid size %d,%d,%d - block size %d,%d,%d - nregs "
				"%d - shmem %d - cuda stream id %ld\n",
				nvbit_get_func_name(ctx, p->f), p->gridDimX, p->gridDimY,
				p->gridDimZ, p->blockDimX, p->blockDimY, p->blockDimZ, nregs,
				shmem_static_nbytes + p->sharedMemBytes, (uint64_t)p->hStream);
			recv_thread_receiving = true;

		} else {
			/* make sure current kernel is completed */
			cudaDeviceSynchronize();
			assert(cudaGetLastError() == cudaSuccess);

			/* make sure we prevent re-entry on the nvbit_callback when issuing
			 * the flush_channel kernel */
			skip_flag = true;

			/* issue flush of channel so we are sure all the memory accesses
			 * have been pushed */
			flush_channel<<<1, 1>>>();
			cudaDeviceSynchronize();
			assert(cudaGetLastError() == cudaSuccess);

			/* unset the skip flag */
			skip_flag = false;

			/* wait here until the receiving thread has not finished with the
			 * current kernel */
			while (recv_thread_receiving) {
				pthread_yield();
			}
		}
	}
}
#define GETPRED(m, i) ((m >> (31 - i)) & 1)

struct Alloc {
	CUdeviceptr size;
	int order;

	int sizehash_minor;
	std::vector<uint32_t> memaccs;

	uint32_t sh()
	{
		return (uint32_t)size << 8 | sizehash_minor;
	}
};



void *recv_thread_fun(void *) {
	char *recv_buffer = (char *)malloc(CHANNEL_SIZE);
	uint64_t idx = 0;
	int allocidx = 0;
	std::map<CUdeviceptr, Alloc> allocs;
	std::map<int, int> sizehashminormap;
	uint64_t local_min = std::numeric_limits<uint64_t>::max(), local_max = 0;


// 	std::list<mem_access_t> buffer;
// 	bool found_allocs = false;
// 	uint64_t face_ptr, vtx_ptr;
// 	std::map<int, std::vector<uint32_t>> buffers;
	std::map<uint32_t, int> facelist;

	while (recv_thread_started) {
// 		std::cout << "START" << std::endl;
// 		idx = 0;
		uint32_t num_recv_bytes = 0;
		if (recv_thread_receiving &&
			(num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) >
				0) {
			uint32_t num_processed_bytes = 0;
			while (num_processed_bytes < num_recv_bytes) {
				mem_access_t *ma =
					(mem_access_t *)&recv_buffer[num_processed_bytes];

				/* when we get this cta_id_x it means the kernel has completed
				 */
				if (ma->cta_id_x == -1) {
					recv_thread_receiving = false;
					break;
				}
//				 if (ma->isexit) {
// 					std::cout << "Exit @" << ma->cta_id_x << " " << ma->cta_id_y << " " << ma->cta_id_z << std::endl;
// 				}

				uint32_t ncta = ma->cta_id_x + ma->cta_id_y * nctax + ma->cta_id_z * nctax * nctay;
				if (ma->opcode_id != 0xfffffffeu) {
					for (int i = 0; i < 32; ++i) {
						if (!GETPRED(ma->preds, i)) continue;
						min_a = std::min(min_a, (uint64_t)ma->addrs[i]);
						max_a = std::max(max_a, (uint64_t)ma->addrs[i]);

	// 					memseg seg = segmentation(ma->addrs[i]);
	// 					allocs[seg.p] = seg.s;
						if (memops[ma->opcode_id].ms == memop::LOCAL) {
							local_max = std::max(ma->addrs[i] + memops[ma->opcode_id].s, local_max);
							local_min = std::min(ma->addrs[i], local_min);
						} else {
	// 						std::cout <<"ub" << std::endl;
							auto cache = allocs.empty() ? allocs.end() : std::prev(allocs.upper_bound(ma->addrs[i]));
	// 						std::cout <<"if" << std::endl;
							if (cache == allocs.end() || ma->addrs[i] < cache->first || ma->addrs[i] + memops[ma->opcode_id].s > cache->first + cache->second.size) {
	// 						std::cout <<"fail" << std::endl;
								CUdeviceptr p, s;
								if (cuPointerGetAttribute(&p, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, ma->addrs[i]) == CUDA_SUCCESS && cuPointerGetAttribute(&s, CU_POINTER_ATTRIBUTE_RANGE_SIZE, ma->addrs[i]) == CUDA_SUCCESS) {
	// 						std::cout <<"emp " << p << " " << s << std::endl;
									int shm = sizehashminormap.emplace((int)memops[ma->opcode_id].s, 0).first->second++;
									cache = allocs.emplace(p, Alloc{ s, allocidx++, shm }).first;

								} else {
									std::cerr << "SEG FAULT" << std::endl;
								}
							} else if (cache != allocs.end()) {
							}
// 							if (memops[ma->opcode_id].ld) {
// 								// WARNING: ONLY CACHE LOADS!
								cache->second.memaccs.push_back(ma->addrs[i] - cache->first);
// 							}
						}
					}
				}
				num_processed_bytes += sizeof(mem_access_t);
			}
		}
	}

	free(recv_buffer);

	// TODO: hier koennte man die memory accesses verarbeiten; geht aber praktisch auch ohne buffer oben inline.

	std::map<std::string, int> descs;
	descs["FRAMEBUF"] = 0;
	descs["BVH"] = 1;
	descs["AABBS"] = 2;
	descs["VTX_ATTRIB"] = 3;
	descs["FACES"] = 4;
	descs["VTX_POS"] = 5;
	int stride[] = { 1, 4, 24, 12, 12, 12 };

	for (auto e : allocs) {
		uint64_t p = e.first, s = e.second.size;
		int id = descs[meminfs[p]];
		std::cout << p << " +" << s << " @" << e.second.order << " " << meminfs[p] << " (" << e.second.memaccs.size() << " accesses)" << std::endl;
		for (int i = 0; i < e.second.memaccs.size(); ++i) {
			uint64_t ma = e.second.memaccs[i];
			uint64_t elm = ma / stride[id], off = ma % stride[id];
			if (off /* or/and size not eq stride */) continue; // HACK partial access; assume that elements are not accessed partially and only care for off==0
			std::cout << "Access element " << elm << std::endl;
		}
	}

	return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx)
{
	recv_thread_started = true;
	channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
	pthread_create(&recv_thread, NULL, recv_thread_fun, NULL);
}

void nvbit_at_ctx_term(CUcontext ctx)
{
	if (recv_thread_started) {
		recv_thread_started = false;
		pthread_join(recv_thread, NULL);
	}
}
