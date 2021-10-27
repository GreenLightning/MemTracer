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


bool getHeatMapColor(float value, float *red, float *green, float *blue)
{
  const int NUM_COLORS = 6;
  static float color[NUM_COLORS][3] = { {0,1,0}, {1,1,0}, {1,0,0}, {0.5,0,0}, {0.5,0,0.5},{0,0,0.5} };
    // A static array of 4 colors:  (blue,   green,  yellow,  red) using {r,g,b} for each.
  
  int idx1;        // |-- Our desired color will be between these two indexes in "color".
  int idx2;        // |
  float fractBetween = 0;  // Fraction between "idx1" and "idx2" where our value is.
  
  if(value <= 0)      {  idx1 = idx2 = 0;            }    // accounts for an input <=0
  else if(value >= 1)  {  idx1 = idx2 = NUM_COLORS-1; }    // accounts for an input >=0
  else
  {
    value = value * (NUM_COLORS-1);        // Will multiply value by 3.
    idx1  = floor(value);                  // Our desired color will be after this index.
    idx2  = idx1+1;                        // ... and before this index (inclusive).
    fractBetween = value - float(idx1);    // Distance between the two indexes (0-1).
  }
    
  *red   = (color[idx2][0] - color[idx1][0])*fractBetween + color[idx1][0];
  *green = (color[idx2][1] - color[idx1][1])*fractBetween + color[idx1][1];
  *blue  = (color[idx2][2] - color[idx1][2])*fractBetween + color[idx1][2];
}

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
std::ofstream os;
int fidx, vidx;
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

	std::string logfn;
	GET_VAR_STR(logfn, "OUTFILE", "Filename of the output trace");
	if (logfn.empty()) {
		std::cerr << "ERROR: No log file specified." << std::endl;
		std::exit(1);
	}
	GET_VAR_INT(fidx, "FHASH", -1, "Face Buffer hash");
	GET_VAR_INT(vidx, "VHASH", -1, "Vertex buffer hash");
	if ((fidx == -1) != (vidx == -1)) {
		std::cerr << "ERROR: Face and vertex buffer hash must be specified together." << std::endl;
		std::exit(1);
	}
	os.open(logfn.c_str());
}
/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

// struct __attribute__((packed)) memop {
// 	enum MemorySpace {
// 		LOCAL,             // local memory operation
// 		GENERIC,           // generic memory operation
// 		GLOBAL,            // global memory operation
// 		SHARED,            // shared memory operation
// // 		CONSTANT,          // constant memory operation
// 		GLOBAL_TO_SHARED,  // read from global memory then write to shared memory
// 		SURFACE,   // surface memory operation
// 		TEXTURE,   // texture memory operation
// 	};
// 	enum Flags { NONCOHERENT = 1 };
// 	char ld, st;
// 	MemorySpace ms;
// 	uint8_t flags;
// 	int s; // 1 2 4 8 16
// };

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
// 
//     LDG.E.S8.SYS     LDG.E.U8.SYS
//     LDG.E.S16.SYS     LDG.E.U16.SYS
//     LDG.E.SYS
//     LDG.E.64.SYS
//     LDG.E.128.SYS
// 
// LDG.E.S8.CONSTANT.SYS LDG.E.U8.CONSTANT.SYS
// LDG.E.S16.CONSTANT.SYS LDG.E.U16.CONSTANT.SYS
// LDG.E.CONSTANT.SYS
// LDG.E.64.CONSTANT.SYS
// LDG.E.128.CONSTANT.SYS
// 
// STG.E.S8.SYS STG.E.U8.SYS
// STG.E.S16.SYS STG.E.U16.SYS
// STG.E.SYS
//     STG.E.64.SYS
//     STG.E.128.SYS
// 
// // LOCAL MEMORY
// LDL.S8 LDL.U8
// LDL.S16 LDL.U16
// LDL
// LDL.64
// LDL.128

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
//                 instr->printDecoded();
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
// 			Instr2 i2 = *(Instr2*)&instr;
// 			std::cout << "INSTR " << instr->getOpcode() << std::endl;
// 			for (int i = 0; i < 256/4; ++i) {
// 				for (int j = 0; j < 4; ++j) {
// 					std::cout << std::dec << (unsigned int)((const unsigned char*)i2.reserved)[i*4+j] << " ";
// 				}
// 					std::cout << std::dec << "[" << (unsigned int)((const uint32_t*)i2.reserved)[i] << "] ";
// 			}
// 			std::cout << std::endl;
//             if (verbose) {
//             }

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
//         std::exit(1);
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


			uint64_t lmem_base = nvbit_get_local_mem_base_addr(ctx);
			std::cout << "LMEM: " << lmem_base << " +" << lmem_static_nbytes << " per thread" << std::endl;
			uint32_t _vals[] = { nctax, nctay, nctaz, p->blockDimX, p->blockDimY, p->blockDimZ, nregs, shmem_static_nbytes + p->sharedMemBytes, lmem_static_nbytes };
// 			os.write((const char*)_vals, 9 * 4);
// 			os.write((const char*)&lmem_base, 8);
// 			std::cout << "REMEMBER " << os.tellp() << " " <<(std::size_t)nctax * nctay * nctaz << std::endl;
// 			std::cout << "MEM INFS: " << std::endl;
// 			uint32_t mis = meminfs.size();
// 			os.write((const char*)&mis, 4);
// 			for (auto mi : meminfs) {
// 				mi.second.ptr = mi.first;
// 				mi.second.print(std::cout);
// 				mi.second.size = 0;
// 				CUdeviceptr s;
// 				if (cuPointerGetAttribute(&s, CU_POINTER_ATTRIBUTE_RANGE_SIZE, mi.second.ptr) == CUDA_SUCCESS) {
// 					mi.second.size = s;
// 				}
// 				os.write((const char*)&mi.second, sizeof(Meminf));
// 				if (mi.second.size) {
// 					std::vector<char> dst(mi.second.size);
// 					cudaMemcpy(dst.data(), (void*)mi.second.ptr, mi.second.size, cudaMemcpyDeviceToHost);
// 					os.write(dst.data(), dst.size());
// 				}
// 			}
// 			taboff = os.tellp();
// 			for (std::size_t i = 0; i < (std::size_t)nctax * nctay * nctaz + 3 /* addr bounds + ptr to segments */; ++i) os.write("\0\0\0\0\0\0\0\0", 8);

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
// struct __attribute__((packed)) MemAcc {
// 	uint32_t x, y, z, warp, op/*, sm*/;
// 	uint64_t a[32];
// };
// struct memseg {
// 	CUdeviceptr p, s;
// };
// memseg segmentation(CUdeviceptr ptr)
// {
// 	memseg s;
// 	cuPointerGetAttribute(&s.p, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, ptr);
// 	cuPointerGetAttribute(&s.s, CU_POINTER_ATTRIBUTE_RANGE_SIZE, ptr);
// 	return s;
// }
#define GETPRED(m, i) ((m >> (31 - i)) & 1)

struct Alloc {
	CUdeviceptr size;
	int order;
	std::size_t ldacc, stacc;

	int sizehash_minor;
	std::vector<uint32_t> load_offsets;

	uint32_t sh()
	{
		return (uint32_t)size << 8 | sizehash_minor;
	}
};

struct VtxAcc {
	uint32_t x, y, z;
	uint32_t v;
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
//                 if (ma->isexit) {
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
									cache = allocs.emplace(p, Alloc{ s, allocidx++, memops[ma->opcode_id].ld ? (std::size_t)memops[ma->opcode_id].s : 0, memops[ma->opcode_id].st ? (std::size_t)memops[ma->opcode_id].s : 0, shm }).first;

								} else {
									std::cerr << "SEG FAULT" << std::endl;
								}
							} else if (cache != allocs.end()) {
								if (memops[ma->opcode_id].ld) cache->second.ldacc += memops[ma->opcode_id].s;
								if (memops[ma->opcode_id].st) cache->second.stacc += memops[ma->opcode_id].s;
							}
							if (memops[ma->opcode_id].ld) {
								cache->second.load_offsets.push_back(ma->addrs[i] - cache->first);
							}
						}
					}
				}
                num_processed_bytes += sizeof(mem_access_t);
            }
        }
    }
//     uint64_t local_s = local_max - local_min;
	bool has_local = local_min != std::numeric_limits<uint64_t>::max();
    std::cout << "Local memory: " << local_min << " +" << local_max - local_min << std::endl;
    free(recv_buffer);
// 			std::cout << "REMEMBERED " << taboff << " " << last.size() << std::endl;
	mem_access_t ma;
	ma.opcode_id = std::numeric_limits<uint32_t>::max();
// 	os.write((const char*)&ma, sizeof(mem_access_t));

	// write segmentation to the end
	uint64_t segptr = os.tellp();
	uint32_t l = allocs.size();
// 	os.write((const char*)&local_min, 8);
// 	os.write((const char*)&local_max, 8);
// 	os.write((const char*)&idx, 8);
// 	os.write((const char*)&l, 4);
	uint64_t maxa = 0;
	std::vector<uint64_t> allocvec(l);
	for (auto e : allocs) {
		allocvec[e.second.order] = e.first;
		uint64_t p = e.first, s = e.second.size;
// 		os.write((const char*)&p, 8);
// 		os.write((const char*)&s, 8);
		maxa = std::max(p+s,maxa);
		std::cout << p << " +" << s << " @" << e.second.order << " freq ld: " << e.second.ldacc << " freq st: " << e.second.stacc << std::endl;
	}

	if (vidx == -1 && 0) {
		for (int i = 0; i < l; ++i) {
			uint64_t pi = allocvec[i];
// 			int fi = allocs[pi].stacc;
			if (allocs[pi].stacc >= 16 * 16 && !allocs[pi].ldacc) {
				std::cout << "OUTSIZE " << allocs[pi].size << std::endl;
			}
		}
	} else {
		uint64_t pv = -1ull, pf = -1ull;
		uint64_t curf = 0;
		uint64_t imagedim = -1ull;
		for (int i = 0; i < l; ++i) {
			uint64_t pi = allocvec[i];
// 			int fi = allocs[pi].stacc;
			if (allocs[pi].stacc >= 16 * 16 && !allocs[pi].ldacc) {
				imagedim = allocs[pi].size;
// 				std::cout << "OUTSIZE " << allocs[pi].size << std::endl;
			}

			for (int j = 0; j < l; ++j) {
				if (i == j) continue;
				uint64_t pi = allocvec[i], pj = allocvec[j];
				uint64_t fi = allocs[pi].ldacc, fj = allocs[pj].ldacc;
				if (fi && fj && fi % fj == 0 && fi != fj && fi + fj > curf) {
					curf = fi + fj;
					pf = fi < fj ? pi : pj;
					pv = fi < fj ? pj : pi;
					uint32_t facei = fi < fj ? allocs[pi].sh() : allocs[pj].sh();
					uint32_t vtxi = fi < fj ? allocs[pj].sh() : allocs[pi].sh();
					std::cout << "Face hash: " << facei << "; Vertex hash: " << vtxi << " (" << fi + fj << ")" << std::endl;
				}
			}
		}
		std::cout << "image dim " << imagedim << std::endl;

// 		for (auto e : allocs) {
// 			if (e.second.sh() == vidx) pv = e.first;
// 			else if (e.second.sh() == fidx) pf = e.first;
// 		}
		if (pv == -1ull || pf == -1ull || imagedim == -1ull) {
			std::cerr << "Invalid hashes" << std::endl;
			std::exit(1);
		}
		uint32_t facestride = 12;
		for (int i = 0; i < allocs[pf].load_offsets.size(); ++i) {
			++facelist.emplace(allocs[pf].load_offsets[i] / facestride, 0).first->second;
		}
		std::vector<uint8_t> vertices(allocs[pv].size);
		std::vector<uint32_t> faces(allocs[pf].size / 4);
		cudaMemcpy(vertices.data(), (void*)pv, allocs[pv].size, cudaMemcpyDeviceToHost);
		cudaMemcpy(faces.data(), (void*)pf, allocs[pf].size, cudaMemcpyDeviceToHost);
		std::vector<uint32_t> facespacked;
		facespacked.reserve(faces.size());
		std::vector<uint32_t> perm, invperm(faces.size());
		perm.reserve(faces.size());
		uint32_t nv = 0;
		for (uint32_t f = 0; f < faces.size() / 3; ++f) {
			uint32_t a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
			if (a == 0 && b == 0 && c == 0) continue;
			nv = std::max(nv, a + 1);
			nv = std::max(nv, b + 1);
			nv = std::max(nv, c + 1);
			invperm[f] = perm.size();
			perm.push_back(f);
			facespacked.push_back(a);
			facespacked.push_back(b);
			facespacked.push_back(c);
		}
		uint32_t nf = facespacked.size() / 3;
		uint32_t stride = allocs[pv].size / nv;
		std::cout << "x" << stride << " " << allocs[pv].size << " " << nv << std::endl;
		if (allocs[pv].size % stride) {
			std::cerr << "Cannot deduce sizeof(Vertex); estimated stride: " << stride << std::endl;
			std::exit(1);
		}
		std::ofstream mesh("mesh.ply");
		mesh << "ply\nformat binary_little_endian 1.0\nelement vertex " << nf * 3 << "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face " << nf << "\nproperty list uchar int vertex_indices\nend_header\n";
		int flistmax = 0;
// 		for (auto x : facelist) flistmax = std::max(flistmax, x.second);
		flistmax = 50;

		vec3 aabb_min(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
		vec3 aabb_max(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());
		for (int i = 0; i < nv; ++i) {
			vec3 v((const float*)(vertices.data() + i * stride));
			aabb_min = min3(v, aabb_min);
			aabb_max = max3(v, aabb_max);
		}
		float dmax = max3i(abs3(aabb_max - aabb_min));
// 		float norm = std::sqrt(dot2(d));
// 		float vol = d.x * d.y * d.z;
// 		float a0 = d.x * d.y;
// 		float a1 = d.x * d.z;
// 		float a2 = d.y * d.z;
// 		float vol = std::max(std::max(a0, a1), a2);
// 		std::cout << "VOL: "<< vol << std::endl;std::exit(1);
		std::cout << "mesh build" << std::endl;
		for (int i = 0; i < nf; ++i) {
// 			std::cout << i << " " << perm[i] << std::endl;
			uint32_t a = facespacked[i * 3], b = facespacked[i * 3 + 1], c = facespacked[i * 3 + 2];
// 			std::cout << "SF"<<std::endl;
			vec3 va((const float*)(vertices.data() + a * stride)), vb((const float*)(vertices.data() + b * stride)), vc((const float*)(vertices.data() + c * stride));
// 			std::cout << "SF2"<<std::endl;
			va = va / dmax;
			vb = vb / dmax;
			vc = vc / dmax;
			vec3 sa = va - vc, sb = vb - vc;
			float A = std::sqrt(dot2(cross(sa, sb))) / 2.f;
// 			std::cout << "SF21"<<std::endl;
// 			std::cout << A / (vol / 1000) << std::endl;
			float intensity = A < std::numeric_limits<float>::epsilon() ? 0 : (float)facelist[perm[i]] / (A * imagedim * 100);
// 			std::cout << "SF22"<<std::endl;
// 			std::cout << intensity << std::endl;
			float rgb[3];
			getHeatMapColor(std::min(1.f, intensity), rgb, rgb + 1, rgb + 2);
			uint8_t color[] = { rgb[0] * 255, rgb[1] * 255, rgb[2] * 255 };
// 			std::cout << "SF3"<<std::endl;
			mesh.write((const char*)&va, 12);
			mesh.write((const char*)color, 3);
			mesh.write((const char*)&vb, 12);
			mesh.write((const char*)color, 3);
			mesh.write((const char*)&vc, 12);
			mesh.write((const char*)color, 3);
// 			std::cout << "SF4"<<std::endl;
		}
// 		std::cout << flistmax << std::endl;
		for (int i = 0; i < nf; ++i) {
			uint32_t fi[] = { i * 3, i * 3 + 1, i * 3 + 2 };
			mesh.write("\3", 1);
			mesh.write((const char*)fi, 12);
		}
		mesh.close();
		
		nv = allocs[pv].size / stride;
		std::cout << "Faces: " << nf << "; Vertices: " << nv << std::endl;
		os.write((const char*)&nv, 4);
		os.write((const char*)&stride, 4);
		os.write((const char*)&nf, 4);
		os.write((const char*)&facestride, 4);
		os.write((const char*)vertices.data(), vertices.size());
		os.write((const char*)facespacked.data(), facespacked.size() * 4);
		uint32_t bs = allocs[pf].load_offsets.size();
		os.write((const char*)&bs, 4);
		for (int i = 0; i < allocs[pf].load_offsets.size(); ++i) {
			os.write((const char*)&invperm[allocs[pf].load_offsets[i] / facestride], 4);
		}
// 		os.write((const char*)buffer.data(), buffer.size() * 4);

		int sum = 0;
		for (auto x : facelist) {
			sum += x.second;
// 			std::cout << x.first << " " << x.second << std::endl;
		}
		double xx = 0;
		for (auto x : facelist) {
			xx += (double)x.second/* / sum*/;
		}
		std::cout << xx /*/ facelist.size()*/ << std::endl;
	}
// 	if (has_local) {
// 		os.write((const char*)&local_min, 8);
// 		os.write((const char*)&local_s, 8);
// 	}
	std::cout << maxa << " " << max_a << std::endl;

// 	os.seekp(taboff);
// 	os.write((const char*)&segptr, 8);
// 	os.write((const char*)&min_a, 8);
// 	os.write((const char*)&max_a, 8);
// 	os.write((const char*)last.data(), last.size() * 8);
    return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {

// 			uint64_t lmem_base = nvbit_get_local_mem_base_addr(ctx);
// 			std::cout << "LMEM: " << lmem_base << " +" << 0 << " per thread" << std::endl;
    recv_thread_started = true;
    channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
    pthread_create(&recv_thread, NULL, recv_thread_fun, NULL);
}

void nvbit_at_ctx_term(CUcontext ctx) {

// 			uint64_t lmem_base = nvbit_get_local_mem_base_addr(ctx);
// 			std::cout << "LMEM: " << lmem_base << " +" << 1 << " per thread" << std::endl;
    if (recv_thread_started) {
        recv_thread_started = false;
        pthread_join(recv_thread, NULL);
    }
}
