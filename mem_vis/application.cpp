#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>

#include <defer.hpp>
#include <mio.hpp>

#include "meow_hash_x64_aesni.h"

#include "common.h"
#include "gl_utils.hpp"

#include "application.hpp"

struct TraceInstructionKey {
	uint64_t grid_launch_id;
	uint64_t instr_addr;

	friend bool operator<(const TraceInstructionKey& lhs, const TraceInstructionKey& rhs) {
		if (lhs.grid_launch_id != rhs.grid_launch_id) return lhs.grid_launch_id < rhs.grid_launch_id;
		return lhs.instr_addr < rhs.instr_addr;
	}

	friend bool operator==(const TraceInstructionKey& lhs, const TraceInstructionKey& rhs) {
		return lhs.grid_launch_id == rhs.grid_launch_id && lhs.instr_addr == rhs.instr_addr;
	}
};

template<>
struct std::hash<TraceInstructionKey> {
	std::size_t operator()(TraceInstructionKey const& key) const noexcept {
		return 7919 * key.grid_launch_id + key.instr_addr;
	}
};

struct TraceInstruction {
	uint64_t    index;
	uint64_t    grid_launch_id;
	uint64_t    instr_addr;
	std::string opcode;
	uint64_t    count = 0;
	uint64_t    min = UINT64_MAX;
	uint64_t    max = 0;
	uint64_t    mem_region_id = UINT64_MAX;
};

struct TraceRegion {
	uint64_t grid_launch_id;
	uint64_t mem_region_id;
	uint64_t start;
	uint64_t size;
	uint64_t contents_offset;

	uint64_t object_size;
	uint64_t object_count;
};

uint64_t calculate_address(TraceRegion* region, int64_t index, uint64_t object_size = 0) {
	if (!object_size) object_size = region->object_size;
	return region->start + index * object_size;
}

int64_t calculate_index(TraceRegion* region, uint64_t address, uint64_t object_size = 0) {
	if (!object_size) object_size = region->object_size;
	if (address == 0) return -1;
	return (address - region->start) / object_size;
}

class Trace {
public:
	std::string filename;
	mio::mmap_source mmap;
	header_t header = {};
	std::vector<std::string> strings;
	std::vector<TraceInstruction> instructions;
	std::vector<TraceRegion> regions;
	std::vector<mem_access_t> accesses;

private:
	std::unordered_map<TraceInstructionKey, TraceInstruction> instructionsByKey;
	std::vector<uint64_t> firstAccessIndexByLaunchID;

public:
	~Trace() {
		mmap.unmap();
	}

	void renderGuiInWindow();

	uint64_t begin_index(uint64_t grid_launch_id) {
		return firstAccessIndexByLaunchID[grid_launch_id];
	}

	uint64_t end_index(uint64_t grid_launch_id) {
		return firstAccessIndexByLaunchID[grid_launch_id+1];
	}

	TraceInstruction* find_instr(uint64_t grid_launch_id, uint64_t instr_addr) {
		auto it = this->instructionsByKey.find(TraceInstructionKey{grid_launch_id, instr_addr});
		return it == this->instructionsByKey.end() ? nullptr : &it->second;
	}

	TraceRegion* find_region(uint64_t grid_launch_id, uint64_t mem_region_id) {
		for (int i = 0; i < regions.size(); i++) {
			TraceRegion* region = &regions[i];
			if (region->grid_launch_id == grid_launch_id && region->mem_region_id == mem_region_id) {
				return region;
			}
		}
		return nullptr;
	}

	void* find_mem_contents(TraceRegion* region) {
		if (!header.mem_contents_offset) return nullptr;
		return &mmap[header.mem_contents_offset + region->contents_offset];
	}

	std::string load(std::string filename) {
		auto t0 = std::chrono::high_resolution_clock::now();

		this->filename = filename;

		std::error_code err_code;
		mmap.map(filename, err_code);
		if (err_code) return err_code.message();

		if (mmap.size() == 0) return "empty file";
		if (mmap.size() < 32) return "file too short";

		memcpy(&header, mmap.data(), 32);

		if (header.magic != (('T' << 0) | ('R' << 8) | ('A' << 16) | ('C' << 24))) {
			return "invalid file (magic)";
		}

		if (header.version != 4) {
			char buffer[256];
			snprintf(buffer, sizeof(buffer), "file version (%d) is not supported", header.version);
			return buffer;
		}

		if (header.header_size < 120 || header.header_size > sizeof(header_t)) {
			return "invalid file (header size)";
		}

		memcpy(&header, mmap.data(), header.header_size);

		auto t1 = std::chrono::high_resolution_clock::now();

		// VALIDATE

		// Save hash and clear memory for hash computation.
		meow_u128 expectedHash;
		memcpy(&expectedHash, &header.hash, 128 / 8);
		memset(&header.hash, 0, 128 / 8);

		meow_state hashState;
		MeowBegin(&hashState, MeowDefaultSeed);
		MeowAbsorb(&hashState, mmap.size() - header.header_size, &mmap[header.header_size]);
		MeowAbsorb(&hashState, header.header_size, &header);
		meow_u128 actualHash = MeowEnd(&hashState, nullptr);

		// Restore hash.
		memcpy(&header.hash, &expectedHash, 128 / 8);

		int match = MeowHashesAreEqual(actualHash, expectedHash);
		if (!match) return "invalid file (hash)";

		if (header.strings_offset < header.header_size ||
			header.strings_offset >= mmap.size() ||
			header.strings_size >= mmap.size() ||
			header.strings_offset + header.strings_size > mmap.size() ||
			(header.strings_size != 0 && mmap[header.strings_offset + header.strings_size - 1] != 0)
			) return "invalid file (strings section)";

			uint64_t string_start = header.strings_offset;
		for (uint64_t i = 0; i < header.strings_size; i++) {
			if (mmap[header.strings_offset + i] == 0) {
				strings.push_back(std::string(&mmap[string_start], &mmap[header.strings_offset + i]));
				string_start = header.strings_offset + i + 1;
			}
		}

		// TODO: Validate other header fields.

		auto t2 = std::chrono::high_resolution_clock::now();

		// NOTE: Sorting
		//
		// We analyze the memory accesses one warp at a time, so that we only
		// have to store intermediate analysis state for a single warp. To make
		// this efficient the memory accesses are sorted by grid_launch_id,
		// block_idx_z, block_idx_y, block_idx_x, local_warp_id. In this way,
		// each analysis can perform a single linear scan over all the memory
		// accesses and see each warp in order.
		//
		// We perform a counting sort, which is vastly more efficient than a
		// comparison-based sort. In fact, we have to copy the memory accesses
		// at least once anyway, because we do not want to modify the memory
		// mapped file, so the copy is unavoidable. And by merging the counting
		// with the trace instruction collection below, we avoid having to read
		// the entire memory again, just to count the accesses. Therefore we
		// basically get the sorting for free.
		//
		// Counting sort is also a stable sort, so the order of memory accesses
		// within a warp is preserved without any special considerations.
		//
		// For sorting, the block index and the local warp id are combined into
		// a single identifier called the global warp index. Because each launch
		// might have a different number of warps, we use a two-stage approach
		// keeping the launch ID and global warp index seperate.

		// WarpIndexFactors contains the multipliers to compute the global warp
		// index for memory accesses in a specific grid.
		struct WarpIndexFactors {
			int32_t z;
			int32_t y;
			int32_t x;
			int32_t local_warp_id;
		};

		std::vector<WarpIndexFactors> warpIndexFactorsByLaunchID;
		warpIndexFactorsByLaunchID.resize(header.launch_info_count);

		// accessInfo is the main data structure for the counting sort.
		// At first it contains the number of memory accesses per warp.
		// Later, these counts are converted into indices by a prefix scan.
		std::vector<std::vector<uint64_t>> accessInfoByGlobalWarpIndexByLaunchID;
		accessInfoByGlobalWarpIndexByLaunchID.resize(header.launch_info_count);

		for (uint64_t launchID = 0; launchID < header.launch_info_count; launchID++) {
			launch_info_t* info = (launch_info_t*) &mmap[header.launch_info_offset + launchID * header.launch_info_size];

			// Compute number of warps per block.
			int32_t grid_dim_warp = (info->block_dim_x * info->block_dim_y * info->block_dim_z + 31) / 32;

			warpIndexFactorsByLaunchID[launchID].z = grid_dim_warp * info->grid_dim_x * info->grid_dim_y;
			warpIndexFactorsByLaunchID[launchID].y = grid_dim_warp * info->grid_dim_x;
			warpIndexFactorsByLaunchID[launchID].x = grid_dim_warp;
			warpIndexFactorsByLaunchID[launchID].local_warp_id = 1;

			accessInfoByGlobalWarpIndexByLaunchID[launchID].resize(info->grid_dim_z * info->grid_dim_y * info->grid_dim_x * grid_dim_warp);
		}

		// TRACE INSTRUCTIONS

		for (int64_t i = 0; i < static_cast<int64_t>(header.mem_access_count); i++) {
			mem_access_t* ma = (mem_access_t*) &mmap[header.mem_access_offset + i * header.mem_access_size];

			// Count number of memory accesses per warp.
			WarpIndexFactors& factors = warpIndexFactorsByLaunchID[ma->grid_launch_id];
			int32_t globalWarpIndex = factors.z * ma->block_idx_z + factors.y * ma->block_idx_y + factors.x * ma->block_idx_x + factors.local_warp_id * ma->local_warp_id;
			accessInfoByGlobalWarpIndexByLaunchID[ma->grid_launch_id][globalWarpIndex]++;

			auto key = TraceInstructionKey{ma->grid_launch_id, ma->instr_addr};
			auto count = instructionsByKey.count(key);
			TraceInstruction& instr = instructionsByKey[key];
			if (count == 0) {
				instr.grid_launch_id = ma->grid_launch_id;
				instr.instr_addr = ma->instr_addr;
				for (int i = 0; i < header.addr_info_count; i++) {
					addr_info_t* info = (addr_info_t*) &mmap[header.addr_info_offset + i * header.addr_info_size];
					if (info->addr == instr.instr_addr) {
						instr.opcode = strings[info->opcode_string_index];
						break;
					}
				}
			}
			for (int i = 0; i < 32; i++) {
				uint64_t addr = ma->addrs[i];
				if (addr != 0) {
					instr.count++;
					if (addr < instr.min) instr.min = addr;
					if (addr > instr.max) instr.max = addr;
				}
			}
		}

		for (auto& pair : instructionsByKey) {
			TraceInstruction& instr = pair.second;
			for (int i = 0; i < header.mem_region_count; i++) {
				mem_region_t* region = (mem_region_t*) &mmap[header.mem_region_offset + i * header.mem_region_size];
				if (region->grid_launch_id == instr.grid_launch_id && region->start <= instr.min && instr.max < region->start + region->size) {
					instr.mem_region_id = region->mem_region_id;
					break;
				}
			}
		}

		for (auto& pair : instructionsByKey) {
			instructions.push_back(pair.second);
		}

		std::sort(instructions.begin(), instructions.end(), [] (const TraceInstruction& a, const TraceInstruction& b) {
			if (a.grid_launch_id != b.grid_launch_id) return a.grid_launch_id < b.grid_launch_id;
			return a.instr_addr < b.instr_addr;
		});

		for (uint64_t i = 0; i < instructions.size(); i++) {
			TraceInstruction& instr = instructions[i];
			instr.index = i;
			auto key = TraceInstructionKey{instr.grid_launch_id, instr.instr_addr};
			instructionsByKey[key].index = i;
		}

		auto t3 = std::chrono::high_resolution_clock::now();

		// TRACE REGIONS

		regions.reserve(header.mem_region_count);
		for (int i = 0; i < header.mem_region_count; i++) {
			mem_region_t* mem_region = (mem_region_t*) &mmap[header.mem_region_offset + i * header.mem_region_size];

			TraceRegion trace_region = {};
			trace_region.grid_launch_id = mem_region->grid_launch_id;
			trace_region.mem_region_id = mem_region->mem_region_id;
			trace_region.start = mem_region->start;
			trace_region.size = mem_region->size;
			trace_region.contents_offset = mem_region->contents_offset;
			regions.push_back(trace_region);
		}

		auto t4 = std::chrono::high_resolution_clock::now();

		// COPY

		// Compute prefix sum of access counts.
		uint64_t index = 0;
		firstAccessIndexByLaunchID.resize(header.launch_info_count + 1);
		for (uint64_t launchID = 0; launchID < header.launch_info_count; launchID++) {
			for (uint64_t globalWarpIndex = 0; globalWarpIndex < accessInfoByGlobalWarpIndexByLaunchID[launchID].size(); globalWarpIndex++) {
				uint64_t temp = accessInfoByGlobalWarpIndexByLaunchID[launchID][globalWarpIndex];
				accessInfoByGlobalWarpIndexByLaunchID[launchID][globalWarpIndex] = index;
				index += temp;
			}
			firstAccessIndexByLaunchID[launchID+1] = index;
		}

		// Copy memory accesses into sorted order.
		accesses.resize(header.mem_access_count);
		for (int64_t i = 0; i < static_cast<int64_t>(header.mem_access_count); i++) {
			mem_access_t* ma = (mem_access_t*) &mmap[header.mem_access_offset + i * header.mem_access_size];
			WarpIndexFactors& factors = warpIndexFactorsByLaunchID[ma->grid_launch_id];
			int32_t globalWarpIndex = factors.z * ma->block_idx_z + factors.y * ma->block_idx_y + factors.x * ma->block_idx_x + factors.local_warp_id * ma->local_warp_id;
			uint64_t index = accessInfoByGlobalWarpIndexByLaunchID[ma->grid_launch_id][globalWarpIndex]++;
			accesses[index] = *ma;
		}

		auto t5 = std::chrono::high_resolution_clock::now();

		printf("init:         %.9fs\n", std::chrono::duration<double>(t1 - t0).count());
		printf("validate:     %.9fs\n", std::chrono::duration<double>(t2 - t1).count());
		printf("instructions: %.9fs\n", std::chrono::duration<double>(t3 - t2).count());
		printf("regions:      %.9fs\n", std::chrono::duration<double>(t4 - t3).count());
		printf("copy:         %.9fs\n", std::chrono::duration<double>(t5 - t4).count());

		return "";
	}
};

struct OffsetSizeAnalysis {
	struct RegionInfo {
		std::map<int64_t, uint64_t> counts;
	};

	uint64_t grid_launch_id;
	std::vector<RegionInfo> regionInfos;

	void run(Trace* trace);
	void renderGui(const char* title, bool* open);
};

void OffsetSizeAnalysis::run(Trace* trace) {
	int32_t last_block_idx_z = -1;
	int32_t last_block_idx_y = -1;
	int32_t last_block_idx_x = -1;
	int32_t last_local_warp_id = -1;

	regionInfos.resize(trace->header.mem_region_count);

	std::vector<uint64_t> last_addresses;
	last_addresses.resize(trace->header.mem_region_count * 32);

	for (uint64_t i = trace->begin_index(this->grid_launch_id), n = trace->end_index(this->grid_launch_id); i < n; i++) {
		mem_access_t* ma = &trace->accesses[i];

		if (ma->block_idx_z != last_block_idx_z || ma->block_idx_y != last_block_idx_y || ma->block_idx_x != last_block_idx_x || ma->local_warp_id != last_local_warp_id) {
			last_block_idx_z = ma->block_idx_z;
			last_block_idx_y = ma->block_idx_y;
			last_block_idx_x = ma->block_idx_x;
			last_local_warp_id = ma->local_warp_id;

			for (int j = 0; j < last_addresses.size(); j++) {
				last_addresses[j] = 0;
			}
		}

		TraceInstruction* instr = trace->find_instr(ma->grid_launch_id, ma->instr_addr);
		if (instr->mem_region_id == UINT64_MAX) continue;

		RegionInfo* info = &regionInfos[instr->mem_region_id];

		uint64_t* last_addrs = &last_addresses[instr->mem_region_id * 32];
		for (int j = 0; j < 32; j++) {
			if (ma->addrs[j] != 0) {
				if (last_addrs[j]) {
					int64_t offset = static_cast<int64_t>(ma->addrs[j] - last_addrs[j]);
					info->counts[offset]++;
				}
				last_addrs[j] = ma->addrs[j];
			}
		}
	}
}

struct InstructionBasedSizeAnalysis {
	struct InstrInfo {
		uint64_t instr_index;
		uint64_t instr_addr;
		uint64_t mem_region_id;
		uint64_t estimate;
	};

	struct RegionInfo {
		uint64_t mem_region_id;
		uint64_t min, max;
	};

	uint64_t grid_launch_id;
	std::vector<InstrInfo> instrInfos;
	std::vector<RegionInfo> regionInfos;

	void run(Trace* trace);
	void renderGui(const char* title, bool* open);
};

uint64_t gcd(uint64_t a, uint64_t b) {
	if (b == 0) return a;
	return gcd(b, a % b);
}

void InstructionBasedSizeAnalysis::run(Trace* trace) {
	int32_t last_block_idx_z = -1;
	int32_t last_block_idx_y = -1;
	int32_t last_block_idx_x = -1;
	int32_t last_local_warp_id = -1;

	std::vector<uint64_t> estimates;
	estimates.resize(trace->instructions.size());
	std::vector<uint64_t> last_addresses;
	last_addresses.resize(trace->instructions.size() * 32);

	for (uint64_t i = trace->begin_index(this->grid_launch_id), n = trace->end_index(this->grid_launch_id); i < n; i++) {
		mem_access_t* ma = &trace->accesses[i];

		if (ma->block_idx_z != last_block_idx_z || ma->block_idx_y != last_block_idx_y || ma->block_idx_x != last_block_idx_x || ma->local_warp_id != last_local_warp_id) {
			last_block_idx_z = ma->block_idx_z;
			last_block_idx_y = ma->block_idx_y;
			last_block_idx_x = ma->block_idx_x;
			last_local_warp_id = ma->local_warp_id;

			for (int j = 0; j < last_addresses.size(); j++) {
				last_addresses[j] = 0;
			}
		}

		TraceInstruction* instr = trace->find_instr(ma->grid_launch_id, ma->instr_addr);
		if (instr->mem_region_id == UINT64_MAX) continue;

		uint64_t* last_addrs = &last_addresses[instr->index * 32];
		for (int j = 0; j < 32; j++) {
			if (last_addrs[j] != 0 && ma->addrs[j] != 0) {
				uint64_t diff = (ma->addrs[j] > last_addrs[j]) ? (ma->addrs[j] - last_addrs[j]) : (last_addrs[j] - ma->addrs[j]);
				estimates[instr->index] = gcd(estimates[instr->index], diff);
			}
			last_addrs[j] = ma->addrs[j];
		}
	}

	for (int i = 0; i < estimates.size(); i++) {
		TraceInstruction& instr = trace->instructions[i];
		if (estimates[i] != 0) {
			InstrInfo info = {};
			info.instr_index = instr.index;
			info.instr_addr = instr.instr_addr;
			info.mem_region_id = instr.mem_region_id;
			info.estimate = estimates[i];
			instrInfos.push_back(info);
		}
	}

	for (TraceRegion& region : trace->regions) {
		uint64_t min = 0, max = 0;
		for (InstrInfo& info : instrInfos) {
			if (info.mem_region_id != region.mem_region_id) continue;
			if (min == 0) {
				min = info.estimate;
				max = info.estimate;
				continue;
			}
			if (info.estimate < min) min = info.estimate;
			if (info.estimate > max) max = info.estimate;
		}
		if (min == 0) continue;
		RegionInfo info = {};
		info.mem_region_id = region.mem_region_id;
		info.min = min;
		info.max = max;
		regionInfos.push_back(info);
	}
}

struct LinearAccessAnalysis {
	enum Flags : uint8_t {
		ACCESSED = 0x01,
		LINEAR   = 0x02,
	};

	TraceRegion* region;
	std::vector<uint8_t> flags;

	void run(Trace* trace);
	void renderGui(const char* title, bool* open);
};

void LinearAccessAnalysis::run(Trace* trace) {
	int32_t last_block_idx_z = -1;
	int32_t last_block_idx_y = -1;
	int32_t last_block_idx_x = -1;
	int32_t last_local_warp_id = -1;
	int64_t last_access[32];

	this->flags.resize(this->region->object_count);

	for (uint64_t i = trace->begin_index(this->region->grid_launch_id), n = trace->end_index(this->region->grid_launch_id); i < n; i++) {
		mem_access_t* ma = &trace->accesses[i];

		if (ma->block_idx_z != last_block_idx_z || ma->block_idx_y != last_block_idx_y || ma->block_idx_x != last_block_idx_x || ma->local_warp_id != last_local_warp_id) {
			last_block_idx_z = ma->block_idx_z;
			last_block_idx_y = ma->block_idx_y;
			last_block_idx_x = ma->block_idx_x;
			last_local_warp_id = ma->local_warp_id;

			for (int i = 0; i < 32; i++) {
				last_access[i] = -1;
			}
		}

		TraceInstruction* instr = trace->find_instr(ma->grid_launch_id, ma->instr_addr);
		if (instr->mem_region_id != this->region->mem_region_id) continue;

		for (int i = 0; i < 32; i++) {
			int64_t access = calculate_index(this->region, ma->addrs[i]);
			if (access >= 0) {
				this->flags[access] |= ACCESSED;
				if (last_access[i] >= 0 && access == last_access[i] + 1) {
					this->flags[access] |= LINEAR;
				}
			}
			last_access[i] = access;
		}
	}
}

struct ConsecutiveAccessAnalysis {
	TraceRegion* region;
	std::vector<uint64_t> matrix;

	void run(Trace* trace);
	void renderGui(const char* title, bool* open);
};

void ConsecutiveAccessAnalysis::run(Trace* trace) {
	int32_t last_block_idx_z = -1;
	int32_t last_block_idx_y = -1;
	int32_t last_block_idx_x = -1;
	int32_t last_local_warp_id = -1;
	int64_t last_access[32];

	matrix.resize(this->region->object_count * this->region->object_count);

	for (uint64_t i = trace->begin_index(this->region->grid_launch_id), n = trace->end_index(this->region->grid_launch_id); i < n; i++) {
		mem_access_t* ma = &trace->accesses[i];

		if (ma->block_idx_z != last_block_idx_z || ma->block_idx_y != last_block_idx_y || ma->block_idx_x != last_block_idx_x || ma->local_warp_id != last_local_warp_id) {
			last_block_idx_z = ma->block_idx_z;
			last_block_idx_y = ma->block_idx_y;
			last_block_idx_x = ma->block_idx_x;
			last_local_warp_id = ma->local_warp_id;

			for (int i = 0; i < 32; i++) {
				last_access[i] = -1;
			}
		}

		TraceInstruction* instr = trace->find_instr(ma->grid_launch_id, ma->instr_addr);
		if (instr->mem_region_id != this->region->mem_region_id) continue;

		for (int i = 0; i < 32; i++) {
			int64_t access = calculate_index(this->region, ma->addrs[i]);
			if (last_access[i] >= 0 && access >= 0) {
				this->matrix[last_access[i] * this->region->object_count + access]++;
			}
			last_access[i] = access;
		}
	}
}

struct StackAnalysis {
	TraceRegion* region;
	std::vector<uint64_t> matrix;

	void run(Trace* trace);
	void renderGui(const char* title, bool* open);
};

void StackAnalysis::run(Trace* trace) {
	int32_t last_block_idx_z = -1;
	int32_t last_block_idx_y = -1;
	int32_t last_block_idx_x = -1;
	int32_t last_local_warp_id = -1;
	int32_t instr_indices[32];
	int64_t last_access[32];
	std::unordered_map<uint64_t, int64_t> stack_maps[32];

	matrix.resize(this->region->object_count * this->region->object_count);

	for (uint64_t i = trace->begin_index(this->region->grid_launch_id), n = trace->end_index(this->region->grid_launch_id); i < n; i++) {
		mem_access_t* ma = &trace->accesses[i];

		if (ma->block_idx_z != last_block_idx_z || ma->block_idx_y != last_block_idx_y || ma->block_idx_x != last_block_idx_x || ma->local_warp_id != last_local_warp_id) {
			last_block_idx_z = ma->block_idx_z;
			last_block_idx_y = ma->block_idx_y;
			last_block_idx_x = ma->block_idx_x;
			last_local_warp_id = ma->local_warp_id;

			for (int i = 0; i < 32; i++) {
				instr_indices[i] = 0;
				last_access[i] = -1;
				stack_maps[i].clear();
			}
		}

		defer {
			for (int i = 0; i < 32; i++) {
				if (ma->addrs[i] != 0) {
					instr_indices[i]++;
				}
			}
		};

		TraceInstruction* instr = trace->find_instr(ma->grid_launch_id, ma->instr_addr);

		if (instr->opcode == "STL.128") {
			for (int i = 0; i < 32; i++) {
				uint64_t stack_addr = ma->addrs[i];
				if (!stack_addr) continue;
				stack_maps[i][stack_addr] = last_access[i];
			}
			continue;
		}

		if (instr->opcode == "LDL.128") {
			for (int i = 0; i < 32; i++) {
				uint64_t stack_addr = ma->addrs[i];
				if (!stack_addr) continue;
				auto it = stack_maps[i].find(stack_addr);
				if (it == stack_maps[i].end()) {
					last_access[i] = -1;
				} else {
					last_access[i] = it->second;
					stack_maps[i].erase(it);
				}
			}
			continue;
		}

		if (instr->mem_region_id != this->region->mem_region_id) continue;

		for (int i = 0; i < 32; i++) {
			int64_t access = calculate_index(this->region, ma->addrs[i]);
			if (last_access[i] >= 0 && access >= 0) {
				this->matrix[last_access[i] * this->region->object_count + access]++;
			}
			last_access[i] = access;
		}
	}
}

struct RegionLinkAnalysis {
	TraceRegion* region_a;
	TraceRegion* region_b;
	std::map<int64_t, std::set<int64_t>> links;

	void run(Trace* trace);
	void renderGui(const char* title, bool* open);
};

void RegionLinkAnalysis::run(Trace* trace) {
	uint64_t grid_launch_id = this->region_a->grid_launch_id;

	int32_t last_block_idx_z = -1;
	int32_t last_block_idx_y = -1;
	int32_t last_block_idx_x = -1;
	int32_t last_local_warp_id = -1;
	int64_t last_access[32];

	for (uint64_t i = trace->begin_index(grid_launch_id), n = trace->end_index(grid_launch_id); i < n; i++) {
		mem_access_t* ma = &trace->accesses[i];

		if (ma->block_idx_z != last_block_idx_z || ma->block_idx_y != last_block_idx_y || ma->block_idx_x != last_block_idx_x || ma->local_warp_id != last_local_warp_id) {
			last_block_idx_z = ma->block_idx_z;
			last_block_idx_y = ma->block_idx_y;
			last_block_idx_x = ma->block_idx_x;
			last_local_warp_id = ma->local_warp_id;

			for (int i = 0; i < 32; i++) {
				last_access[i] = -1;
			}
		}

		TraceInstruction* instr = trace->find_instr(ma->grid_launch_id, ma->instr_addr);
		if (instr->mem_region_id == this->region_a->mem_region_id) {
			for (int i = 0; i < 32; i++) {
				last_access[i] = calculate_index(this->region_a, ma->addrs[i]);
			}
		} else if (instr->mem_region_id == this->region_b->mem_region_id) {
			for (int i = 0; i < 32; i++) {
				int64_t access = calculate_index(this->region_b, ma->addrs[i]);
				if (last_access[i] >= 0 && access >= 0) {
					this->links[last_access[i]].insert(access);
				}
				last_access[i] = -1;
			}
		} else if (instr->mem_region_id != UINT64_MAX) {
			for (int i = 0; i < 32; i++) {
				last_access[i] = -1;
			}
		}
	}
}

struct CaaDistributionAnalysis {
	constexpr static int64_t num_buckets = 101;

	ConsecutiveAccessAnalysis* caa;
	int64_t buckets[num_buckets];

	void run() {
		std::vector<uint64_t> counts;
		for (int64_t parent_index = 0; parent_index < static_cast<int64_t>(caa->region->object_count); parent_index++) {
			counts.clear();

			for (int64_t child_index = 0; child_index < static_cast<int64_t>(caa->region->object_count); child_index++) {
				auto count = caa->matrix[parent_index * caa->region->object_count + child_index];
				if (count != 0) counts.emplace_back(count);
			}

			if (counts.empty()) continue;

			uint64_t max = counts[0];
			for (auto count : counts) {
				if (count > max) max = count;
			}
			for (auto count : counts) {
				if (count == max) continue;
				float value = static_cast<float>(count) / static_cast<float>(max);
				int64_t bucket = static_cast<int64_t>(value * (num_buckets-1));
				buckets[bucket]++;
			}
		}
	}

	void renderGui(const char* title, bool* open);
};

struct AnalysisSet {
	OffsetSizeAnalysis osa;
	InstructionBasedSizeAnalysis ibsa;
	ConsecutiveAccessAnalysis caa;
	StackAnalysis sa;
	RegionLinkAnalysis index_rla;
	RegionLinkAnalysis bounds_rla;
	LinearAccessAnalysis nodes_laa;
	LinearAccessAnalysis index_laa;
	CaaDistributionAnalysis caada;

	void init(Trace* trace, uint64_t grid_launch_id) {
		osa.grid_launch_id = grid_launch_id;
		osa.run(trace);

		ibsa.grid_launch_id = grid_launch_id;
		ibsa.run(trace);

		// Set object sizes.
		TraceRegion* node_region = trace->find_region(grid_launch_id, 1);
		node_region->object_size = 4;
		node_region->object_count = node_region->size / node_region->object_size;

		TraceRegion* bounds_region = trace->find_region(grid_launch_id, 2);
		bounds_region->object_size = 6 * 4;
		bounds_region->object_count = bounds_region->size / bounds_region->object_size;

		TraceRegion* index_region = trace->find_region(grid_launch_id, 3);
		index_region->object_size = 3 * 4;
		index_region->object_count = index_region->size / index_region->object_size;

		caa.region = node_region;
		caa.run(trace);

		sa.region = node_region;
		sa.run(trace);

		index_rla.region_a = node_region;
		index_rla.region_b = index_region;
		index_rla.run(trace);

		bounds_rla.region_a = node_region;
		bounds_rla.region_b = bounds_region;
		bounds_rla.run(trace);

		nodes_laa.region = node_region;
		nodes_laa.run(trace);

		index_laa.region = index_region;
		index_laa.run(trace);

		caada.caa = &caa;
		caada.run();
	}
};

struct Node {
	enum Type {
		UNKNOWN,
		PARENT,
		LEAF
	};

	Type type;
	uint64_t address;

	union {
		struct {
			Node* left;
			Node* right;
		} parent_data;

		struct {
			uint64_t face_address;
			uint64_t face_count;
		} leaf_data;
	};
};

struct Tree {
	std::vector<Node> nodes;
};

struct TreeStats {
	int32_t unknowns    = 0;
	int32_t parents     = 0;
	int32_t leafs       = 0;
	int32_t total       = 0;
	int32_t connections = 0;
};

TreeStats countTree(Tree* tree) {
	TreeStats stats;
	for (auto& node : tree->nodes) {
		switch (node.type) {
			case Node::UNKNOWN:
			stats.unknowns++;
			break;

			case Node::PARENT:
			stats.parents++;
			if (node.parent_data.left) stats.connections++;
			if (node.parent_data.right) stats.connections++;
			break;

			case Node::LEAF:
			stats.leafs++;
			break;
		}
	}
	stats.total = stats.unknowns + stats.parents + stats.leafs;
	return stats;
}

struct TreeResults {
	int32_t parents_typed_correctly = 0;
	int32_t leafs_typed_correctly   = 0;
	int32_t parents_fully_correct   = 0;
	int32_t leafs_fully_correct     = 0;
	int32_t connections_correct     = 0;
};

TreeResults rateTree(Tree* tree, Tree* reference) {
	TreeResults results;

	std::unordered_map<uint64_t, Node*> reference_node_by_address;
	for (auto& node : reference->nodes) {
		reference_node_by_address[node.address] = &node;
	}

	for (int i = 0; i < tree->nodes.size(); i++) {
		Node* node = &tree->nodes[i];

		auto it = reference_node_by_address.find(node->address);
		if (it == reference_node_by_address.end()) continue;
		Node* ref = it->second;

		switch (node->type) {
			case Node::PARENT:
			if (ref->type == Node::PARENT) {
				results.parents_typed_correctly++;
				uint64_t nl = node->parent_data.left  ? node->parent_data.left ->address : 0;
				uint64_t nr = node->parent_data.right ? node->parent_data.right->address : 0;
				uint64_t rl = ref->parent_data.left ->address;
				uint64_t rr = ref->parent_data.right->address;
				if ((nl == rl && nr == rr) || (nl == rr && nr == rl)) {
					results.parents_fully_correct++;
					results.connections_correct += 2;
				} else if (nl == rl || nl == rr || nr == rl || nr == rr) {
					results.connections_correct++;
				}
			}
			break;

			case Node::LEAF:
			if (ref->type == Node::LEAF) {
				results.leafs_typed_correctly++;
				if (node->leaf_data.face_address == ref->leaf_data.face_address && node->leaf_data.face_count == ref->leaf_data.face_count) {
					results.leafs_fully_correct++;
				}
			}
			break;
		}
	}

	return results;
}

// Rebuilds the tree using only nodes that are reachable from the root node.
Tree pruneTree(Tree* source) {
	Tree dest;
	dest.nodes.reserve(source->nodes.size());

	if (source->nodes.empty()) return dest;

	// The stack contains double pointers, because the child pointers need to
	// be patched to point to the new nodes. After the while loop, root
	// will have been patched to point to the root node of dest.
	std::vector<Node**> stack;
	Node* root = &source->nodes[0];
	stack.push_back(&root);
	while (!stack.empty()) {
		Node** p = stack.back();
		stack.pop_back();

		dest.nodes.push_back(**p);
		Node* node = &dest.nodes.back();
		*p = node;

		if (node->type == Node::PARENT) {
			if (node->parent_data.left)  stack.push_back(&node->parent_data.left);
			if (node->parent_data.right) stack.push_back(&node->parent_data.right);
		}
	}

	return dest;
}

Tree mergeTrees(Tree* left, Tree* right) {
	Tree dest;

	// Very bad upper bound.
	// Can be avoided by building node_by_address after all the nodes have been inserted.
	dest.nodes.reserve(left->nodes.size() + right->nodes.size());

	std::unordered_map<uint64_t, Node*> node_by_address;
	std::unordered_map<uint64_t, Node*> left_node_by_address;
	std::unordered_map<uint64_t, Node*> right_node_by_address;

	for (auto& node : left->nodes) {
		left_node_by_address[node.address] = &node;
		if (!node_by_address.count(node.address)) {
			dest.nodes.push_back(Node{});
			Node* p = &dest.nodes.back();
			p->address = node.address;
			node_by_address[node.address] = p;
		}
	}

	for (auto& node : right->nodes) {
		right_node_by_address[node.address] = &node;
		if (!node_by_address.count(node.address)) {
			dest.nodes.push_back(Node{});
			Node* p = &dest.nodes.back();
			p->address = node.address;
			node_by_address[node.address] = p;
		}
	}

	int type_conflict = 0;
	int parent_conflict = 0;
	int leaf_conflict = 0;

	for (auto& node : dest.nodes) {
		auto left_it = left_node_by_address.find(node.address);
		Node* l = left_it == left_node_by_address.end() ? nullptr : left_it->second;
		auto right_it = right_node_by_address.find(node.address);
		Node* r = right_it == right_node_by_address.end() ? nullptr : right_it->second;

		defer {
			// Patch pointers to always point into dest, duh.
			if (node.type == Node::PARENT) {
				if (node.parent_data.left) node.parent_data.left = node_by_address[node.parent_data.left->address];
				if (node.parent_data.right) node.parent_data.right = node_by_address[node.parent_data.right->address];
			}
		};

		if (l == nullptr) {
			node = *r;
			continue;
		}
		if (r == nullptr) {
			node = *l;
			continue;
		}

		if (l->type == Node::UNKNOWN) {
			node = *r;
			continue;
		}
		if (r->type == Node::UNKNOWN) {
			node = *l;
			continue;
		}
		if (l->type != r->type) {
			type_conflict++;
			node.type = Node::UNKNOWN;
			continue;
		}

		node.type = l->type;
		switch (node.type) {
			case Node::PARENT: {
				uint64_t children[4];
				int count = 0;
				if (l->parent_data.left) {
					bool found = false;
					for (int i = 0; i < count; i++) {
						if (l->parent_data.left->address == children[i]) {
							found = true;
							break;
						}
					}
					if (!found) {
						children[count++] = l->parent_data.left->address;
					}
				}
				if (l->parent_data.right) {
					bool found = false;
					for (int i = 0; i < count; i++) {
						if (l->parent_data.right->address == children[i]) {
							found = true;
							break;
						}
					}
					if (!found) {
						children[count++] = l->parent_data.right->address;
					}
				}
				if (r->parent_data.left) {
					bool found = false;
					for (int i = 0; i < count; i++) {
						if (r->parent_data.left->address == children[i]) {
							found = true;
							break;
						}
					}
					if (!found) {
						children[count++] = r->parent_data.left->address;
					}
				}
				if (r->parent_data.right) {
					bool found = false;
					for (int i = 0; i < count; i++) {
						if (r->parent_data.right->address == children[i]) {
							found = true;
							break;
						}
					}
					if (!found) {
						children[count++] = r->parent_data.right->address;
					}
				}
				if (count > 2) {
					node.type = Node::UNKNOWN;
					parent_conflict++;
				} else {
					node.parent_data.left = nullptr;
					node.parent_data.right = nullptr;
					if (count > 0) node.parent_data.left = node_by_address[children[0]];
					if (count > 1) node.parent_data.right = node_by_address[children[1]];
				}
			} break;

			case Node::LEAF: {
				// if (l->leaf_data.face_address != r->leaf_data.face_address) {
				// 	leaf_conflict++;
				// 	node.type = Node::UNKNOWN;
				// } else {
				// 	node.leaf_data.face_address = l->leaf_data.face_address;
				// 	node.leaf_data.face_count = l->leaf_data.face_count < r->leaf_data.face_count ? l->leaf_data.face_count : r->leaf_data.face_count;
				// }
				if (l->leaf_data.face_address != r->leaf_data.face_address || l->leaf_data.face_count != r->leaf_data.face_count) {
					leaf_conflict++;
					node.type = Node::UNKNOWN;
				} else {
					node.leaf_data = l->leaf_data;
				}
			} break;
		}
	}


	printf("type_conflict %d\n", type_conflict);
	printf("parent_conflict %d\n", parent_conflict);
	printf("leaf_conflict %d\n", leaf_conflict);
	return dest;
}

Tree buildReferenceTree(Trace* trace) {
	Tree tree;

	// Check if trace contains memory contents.
	if (!trace->header.mem_contents_offset) return tree;

	TraceRegion* node_region = trace->find_region(0, 1);
	TraceRegion* index_region = trace->find_region(0, 3);

	uint32_t* node_data = (uint32_t*) trace->find_mem_contents(node_region);
	int node_count = static_cast<int>(node_region->size / 4);
	int leaf_count = (node_count + 1) / 2;
	int triangles_per_leaf = static_cast<int>(index_region->size / (3 * 4) / leaf_count);

	int leaf_index = 0;
	tree.nodes.resize(node_count);
	for (int i = 0; i < node_count; i++) {
		Node* node = &tree.nodes[i];

		uint32_t data = node_data[i];
		uint32_t payload = data & 0x7fffffffu;
		bool is_leaf = data >> 31;

		node->type = is_leaf ? Node::LEAF : Node::PARENT;
		node->address = node_region->start + i * 4;

		if (is_leaf) {
			node->leaf_data.face_address = index_region->start + leaf_index * triangles_per_leaf * 3 * 4;
			node->leaf_data.face_count = payload;
			leaf_index++;
		} else {
			node->parent_data.left = &tree.nodes[i + 1];
			node->parent_data.right = &tree.nodes[i + payload];
		}
	}

	return tree;
}

template <typename T>
Tree reconstructTree(AnalysisSet* analysis, T* node_analysis) {
	Tree tree;
	std::unordered_map<int64_t, Node*> nodeByIndex;

	tree.nodes.reserve(analysis->nodes_laa.region->object_count);
	for (int64_t index = 0; index < static_cast<int64_t>(analysis->nodes_laa.region->object_count); index++) {
		if (analysis->nodes_laa.flags[index] & LinearAccessAnalysis::ACCESSED) {
			tree.nodes.push_back(Node{});
			Node* node = &tree.nodes.back();
			node->address = calculate_address(analysis->nodes_laa.region, index);
			nodeByIndex[index] = node;
		}
	}

	// Reconstruct leaf nodes.
	{
		const int64_t index_count = static_cast<int64_t>(analysis->index_laa.region->object_count);

		// startsNewLeaf contains true if another leaf starts at this index.
		// This is used to prevent overlap between leafes.
		std::vector<bool> startsNewLeaf;
		startsNewLeaf.resize(index_count);

		for (auto& pair : analysis->index_rla.links) {
			if (pair.second.size() == 1) {
				auto to_index = *pair.second.begin();
				startsNewLeaf[to_index] = true;
			}
		}

		for (auto& pair : analysis->index_rla.links) {
			if (pair.second.size() == 1) {
				auto from_index = pair.first;
				auto to_index = *pair.second.begin();

				Node* node = nodeByIndex[from_index];
				node->type = Node::LEAF;

				int64_t end_index = to_index + 1;
				while (end_index < index_count && (analysis->index_laa.flags[end_index] & LinearAccessAnalysis::LINEAR) && !startsNewLeaf[end_index]) end_index++;

				node->leaf_data.face_address = calculate_address(analysis->index_rla.region_b, to_index);
				node->leaf_data.face_count = end_index - to_index;
			}
		}
	}

	std::unordered_set<Node*> used;
	if (!tree.nodes.empty()) used.insert(&tree.nodes[0]);

	// Each node may only be used once in the tree construction (i.e. have at most one parent)
	// to prevent forming a graph. We sort by the total number of accesses, which is a proxy
	// for our confidence in the relation, to form connections we are more confident of first.

	std::vector<std::pair<int64_t, uint64_t>> totals; // (parent_index, total)
	for (int64_t parent_index = 0; parent_index < static_cast<int64_t>(node_analysis->region->object_count); parent_index++) {
		uint64_t total = 0;
		for (int64_t child_index = 0; child_index < static_cast<int64_t>(node_analysis->region->object_count); child_index++) {
			total += node_analysis->matrix[parent_index * node_analysis->region->object_count + child_index];
		}
		if (total) {
			totals.emplace_back(parent_index, total);
		}
	}

	std::sort(totals.begin(), totals.end(), [] (const std::pair<int64_t, uint64_t>& a, const std::pair<int64_t, uint64_t>& b) {
		return a.second > b.second;
	});

	std::vector<std::pair<int64_t, uint64_t>> children; // (child_index, count)
	for (const auto& pair : totals) {
		int64_t parent_index = pair.first;

		children.clear();

		for (int64_t child_index = 0; child_index < static_cast<int64_t>(node_analysis->region->object_count); child_index++) {
			auto count = node_analysis->matrix[parent_index * node_analysis->region->object_count + child_index];
			if (count != 0) {
				children.emplace_back(child_index, count);
			}
		}

		if (children.empty()) continue;

		Node* parent = nodeByIndex[parent_index];
		if (parent->type != Node::UNKNOWN) continue;

		std::sort(children.begin(), children.end(), [] (const std::pair<int64_t, uint64_t>& a, const std::pair<int64_t, uint64_t>& b) {
			return a.second > b.second;
		});

		uint64_t cutoff = children[0].second / 4;
		uint64_t size = 1;
		while (size < children.size() && children[size].second >= cutoff) size++;
		children.resize(size);

		parent->type = Node::PARENT;
		Node** target = &parent->parent_data.left;
		for (const auto& pair : children) {
			Node* child = nodeByIndex[pair.first];
			if (used.count(child)) continue;
			used.insert(child);
			*target = child;
			if (target == &parent->parent_data.left) {
				target = &parent->parent_data.right;
			} else {
				break;
			}
		}
	}

	return tree;
}

struct Workspace {
	std::unique_ptr<Trace> trace;
	std::vector<AnalysisSet> analysis;
	std::vector<Tree> preciseReconstructions;
	Tree reference;
	Tree reconstruction;
	Tree prunedReconstruction;
	Tree preciseReconstruction;
	Tree prunedPreciseReconstruction;
	Tree fullReconstruction;
};

std::unique_ptr<Workspace> buildWorkspace(std::unique_ptr<Trace> trace) {
	auto ws = std::make_unique<Workspace>();
	ws->trace = std::move(trace);

	auto t0 = std::chrono::high_resolution_clock::now();

	ws->analysis.resize(ws->trace->header.launch_info_count);
	for (uint64_t i = 0; i < ws->trace->header.launch_info_count; i++) {
		ws->analysis[i].init(ws->trace.get(), i);
	}

	auto t1 = std::chrono::high_resolution_clock::now();

	AnalysisSet* as = &ws->analysis[0];
	ws->reference = buildReferenceTree(ws->trace.get());
	ws->reconstruction = reconstructTree(as, &as->caa);
	ws->prunedReconstruction = pruneTree(&ws->reconstruction);
	ws->preciseReconstruction = reconstructTree(as, &as->sa);
	ws->prunedPreciseReconstruction = pruneTree(&ws->preciseReconstruction);

	ws->fullReconstruction = ws->preciseReconstruction;
	ws->preciseReconstructions.push_back(ws->preciseReconstruction);
	for (uint64_t i = 1; i < ws->trace->header.launch_info_count; i++) {
		AnalysisSet* as = &ws->analysis[i];
		Tree partial = reconstructTree(as, &as->sa);
		ws->fullReconstruction = mergeTrees(&ws->fullReconstruction, &partial);
		ws->preciseReconstructions.push_back(std::move(partial));
	}
	ws->fullReconstruction = pruneTree(&ws->fullReconstruction);

	auto t2 = std::chrono::high_resolution_clock::now();

	printf("analysis:       %.9fs\n", std::chrono::duration<double>(t1 - t0).count());
	printf("reconstruction: %.9fs\n", std::chrono::duration<double>(t2 - t1).count());

	TreeStats ref = countTree(&ws->reference);
	{
		TreeStats s;
		
		s = countTree(&ws->reference);
		printf("Reference:           U%05d P%05d L%05d T%05d=%3.0f%% C%05d=%3.0f%%\n", s.unknowns, s.parents, s.leafs, s.total, 100.0f * s.total / ref.total, s.connections, 100.0f * s.connections / ref.connections);

		s = countTree(&ws->reconstruction);
		printf("Reconstructed Nodes: U%05d P%05d L%05d T%05d=%3.0f%% C%05d=%3.0f%%\n", s.unknowns, s.parents, s.leafs, s.total, 100.0f * s.total / ref.total, s.connections, 100.0f * s.connections / ref.connections);
		
		s = countTree(&ws->prunedReconstruction);
		printf("Reconstructed Tree:  U%05d P%05d L%05d T%05d=%3.0f%% C%05d=%3.0f%%\n", s.unknowns, s.parents, s.leafs, s.total, 100.0f * s.total / ref.total, s.connections, 100.0f * s.connections / ref.connections);

		s = countTree(&ws->preciseReconstruction);
		printf("Precise Nodes:       U%05d P%05d L%05d T%05d=%3.0f%% C%05d=%3.0f%%\n", s.unknowns, s.parents, s.leafs, s.total, 100.0f * s.total / ref.total, s.connections, 100.0f * s.connections / ref.connections);
		
		s = countTree(&ws->prunedPreciseReconstruction);
		printf("Precise Tree:        U%05d P%05d L%05d T%05d=%3.0f%% C%05d=%3.0f%%\n", s.unknowns, s.parents, s.leafs, s.total, 100.0f * s.total / ref.total, s.connections, 100.0f * s.connections / ref.connections);

		s = countTree(&ws->fullReconstruction);
		printf("Full Tree:           U%05d P%05d L%05d T%05d=%3.0f%% C%05d=%3.0f%%\n", s.unknowns, s.parents, s.leafs, s.total, 100.0f * s.total / ref.total, s.connections, 100.0f * s.connections / ref.connections);
	}

	{
		TreeResults r;

		r = rateTree(&ws->reconstruction, &ws->reference);
		printf("Reconstructed Nodes: PT%05d P+%05d=%3.0f%% LT%05d L+%05d=%3.0f%% C%05d=%3.0f%%\n", r.parents_typed_correctly, r.parents_fully_correct, 100.0f * r.parents_fully_correct / ref.parents, r.leafs_typed_correctly, r.leafs_fully_correct, 100.0f * r.leafs_fully_correct / ref.leafs, r.connections_correct, 100.0f * r.connections_correct / ref.connections);
		
		r = rateTree(&ws->prunedReconstruction, &ws->reference);
		printf("Reconstructed Tree:  PT%05d P+%05d=%3.0f%% LT%05d L+%05d=%3.0f%% C%05d=%3.0f%%\n", r.parents_typed_correctly, r.parents_fully_correct, 100.0f * r.parents_fully_correct / ref.parents, r.leafs_typed_correctly, r.leafs_fully_correct, 100.0f * r.leafs_fully_correct / ref.leafs, r.connections_correct, 100.0f * r.connections_correct / ref.connections);

		r = rateTree(&ws->preciseReconstruction, &ws->reference);
		printf("Precise Nodes:       PT%05d P+%05d=%3.0f%% LT%05d L+%05d=%3.0f%% C%05d=%3.0f%%\n", r.parents_typed_correctly, r.parents_fully_correct, 100.0f * r.parents_fully_correct / ref.parents, r.leafs_typed_correctly, r.leafs_fully_correct, 100.0f * r.leafs_fully_correct / ref.leafs, r.connections_correct, 100.0f * r.connections_correct / ref.connections);
		
		r = rateTree(&ws->prunedPreciseReconstruction, &ws->reference);
		printf("Precise Tree:        PT%05d P+%05d=%3.0f%% LT%05d L+%05d=%3.0f%% C%05d=%3.0f%%\n", r.parents_typed_correctly, r.parents_fully_correct, 100.0f * r.parents_fully_correct / ref.parents, r.leafs_typed_correctly, r.leafs_fully_correct, 100.0f * r.leafs_fully_correct / ref.leafs, r.connections_correct, 100.0f * r.connections_correct / ref.connections);

		r = rateTree(&ws->fullReconstruction, &ws->reference);
		printf("Full Tree:           PT%05d P+%05d=%3.0f%% LT%05d L+%05d=%3.0f%% C%05d=%3.0f%%\n", r.parents_typed_correctly, r.parents_fully_correct, 100.0f * r.parents_fully_correct / ref.parents, r.leafs_typed_correctly, r.leafs_fully_correct, 100.0f * r.leafs_fully_correct / ref.leafs, r.connections_correct, 100.0f * r.connections_correct / ref.connections);
	}

	return ws;
}

struct Selection {
	uint64_t launch_id     = UINT64_MAX;
	uint64_t instr_addr    = UINT64_MAX;
	uint64_t mem_region_id = UINT64_MAX;
};

#include "app_grid.cpp"
#include "app_vis.cpp"

struct Application {
	int width, height;
	
	bool osa = true;
	bool ibsa = false;
	bool caa = false;
	bool sa = false;
	bool index_rla = false;
	bool bounds_rla = false;
	bool index_laa = false;
	bool caada = false;
	bool demo = false;

	Selection selected;

	std::unique_ptr<Workspace> workspace;
	Grid grid;
	Visualizer vis;
	std::string status;
};

namespace {
	Application app;
}

void appInitialize(GLFWwindow* window, int argc, char* argv[]) {
	if (argc > 2) {
		fprintf(stderr, "too many arguments\n");
		fprintf(stderr, "usage: <trace-filename>\n");
		exit(1);
	}

	if (argc == 2) {
		auto trace = std::make_unique<Trace>();
		std::string error = trace->load(argv[1]);
		if (!error.empty()) {
			fprintf(stderr, "failed to load %s: %s\n", trace->filename.c_str(), error.c_str());
			exit(1);
		}
		app.workspace = buildWorkspace(std::move(trace));
	}
}

void appTerminate(GLFWwindow* window) {
}

void appKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

void appMouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {}

void appCursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {}

void appDropCallback(GLFWwindow* window, int count, const char** paths) {
	if (count >= 1) {
		auto trace = std::make_unique<Trace>();
		std::string error = trace->load(paths[count-1]);
		if (!error.empty()) {
			app.status = "Failed to load " + trace->filename + ": " + error;
			return;
		}
		app.workspace = buildWorkspace(std::move(trace));
		app.status = "Loaded " + app.workspace->trace->filename;
	}
}

void appSetSize(GLFWwindow* window, int width, int height) {
	app.width = width;
	app.height = height;
	glViewport(0, 0, width, height);
}

void appRender(GLFWwindow* window, float delta) {
	// UPDATE

	// RENDERING

	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	GL_CHECK();
}

void Trace::renderGuiInWindow() {
	ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

	ImGui::Text("Filename: %s", this->filename.c_str());

	meow_u128 hash;
	memcpy(&hash, &this->header.hash, 128 / 8);
	ImGui::Text("Hash: %08X-%08X-%08X-%08X", MeowU32From(hash, 3), MeowU32From(hash, 2), MeowU32From(hash, 1), MeowU32From(hash, 0));

	float infosHeight = min((this->header.launch_info_count + 2) * ImGui::GetFrameHeight(), 200);
	if (ImGui::BeginTable("Launch Infos", 3, flags, ImVec2(0.0f, infosHeight))) {
		ImGui::TableSetupScrollFreeze(0, 1);
		ImGui::TableSetupColumn("Launch ID", ImGuiTableColumnFlags_None);
		ImGui::TableSetupColumn("Grid Size", ImGuiTableColumnFlags_None);
		ImGui::TableSetupColumn("Block Size", ImGuiTableColumnFlags_None);
		ImGui::TableHeadersRow();

		for (uint64_t i = 0; i < this->header.launch_info_count; i++) {
			launch_info_t* info = (launch_info_t*) &this->mmap[this->header.launch_info_offset + i * this->header.launch_info_size];
			ImGui::TableNextRow();
			ImGui::TableNextColumn();

			char label[32];
			snprintf(label, sizeof(label), "%llu", info->grid_launch_id);
			ImGuiSelectableFlags selectable_flags = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap;
			if (ImGui::Selectable(label, app.selected.launch_id == info->grid_launch_id, selectable_flags)) {
				app.selected.launch_id = info->grid_launch_id;
			}

			ImGui::TableNextColumn();
			ImGui::Text("%d,%d,%d", info->grid_dim_x, info->grid_dim_y, info->grid_dim_z);
			ImGui::TableNextColumn();
			ImGui::Text("%d,%d,%d", info->block_dim_x, info->block_dim_y, info->block_dim_z);
		}
		ImGui::EndTable();
	}

	if (ImGui::BeginTable("Memory Regions", 7, flags, ImVec2(0.0f, 200.0f))) {
		ImGui::TableSetupScrollFreeze(0, 1);
		ImGui::TableSetupColumn("Region ID", ImGuiTableColumnFlags_None);
		ImGui::TableSetupColumn("Start", ImGuiTableColumnFlags_None);
		ImGui::TableSetupColumn("End", ImGuiTableColumnFlags_None);
		ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_None);
		ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_None);
		ImGui::TableSetupColumn("Object Size", ImGuiTableColumnFlags_None);
		ImGui::TableSetupColumn("Object Count", ImGuiTableColumnFlags_None);
		ImGui::TableHeadersRow();

		for (uint64_t i = 0; i < this->regions.size(); i++) {
			TraceRegion* region = &this->regions[i];
			if (region->grid_launch_id != app.selected.launch_id) continue;
			
			ImGui::TableNextRow();
			ImGui::TableNextColumn();

			char label[32];
			snprintf(label, sizeof(label), "%llu", region->mem_region_id);
			ImGuiSelectableFlags selectable_flags = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap;
			if (ImGui::Selectable(label, app.selected.mem_region_id == region->mem_region_id, selectable_flags)) {
				app.selected.instr_addr = UINT64_MAX;
				app.selected.mem_region_id = region->mem_region_id;
			}

			ImGui::TableNextColumn();
			ImGui::Text("0x%016lx", region->start);
			ImGui::TableNextColumn();
			ImGui::Text("0x%016lx", region->start + region->size);
			ImGui::TableNextColumn();
			ImGui::Text("0x%016lx", region->size);
			ImGui::TableNextColumn();
			ImGui::Text("%ld", region->size);
			ImGui::TableNextColumn();
			ImGui::Text("%ld", region->object_size);
			ImGui::TableNextColumn();
			ImGui::Text("%ld", region->object_count);
		}
		ImGui::EndTable();
	}

	float instructionsHeight = max(ImGui::GetContentRegionAvail().y, 500);
	if (ImGui::BeginTable("Instructions", 7, flags, ImVec2(0.0f, instructionsHeight))) {
		ImGui::TableSetupScrollFreeze(0, 1);
		ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_None);
		ImGui::TableSetupColumn("IP", ImGuiTableColumnFlags_None);
		ImGui::TableSetupColumn("Opcode", ImGuiTableColumnFlags_None);
		ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_None);
		ImGui::TableSetupColumn("Min", ImGuiTableColumnFlags_None);
		ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_None);
		ImGui::TableSetupColumn("Region", ImGuiTableColumnFlags_None);

		ImGui::TableHeadersRow();

		for (uint64_t i = 0; i < this->instructions.size(); i++) {
			TraceInstruction& instr = this->instructions[i];
			if (instr.grid_launch_id != app.selected.launch_id) continue;

			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			char label[32];
			snprintf(label, sizeof(label), "%llu", i);
			ImGuiSelectableFlags selectable_flags = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap;
			if (ImGui::Selectable(label, app.selected.instr_addr == instr.instr_addr, selectable_flags)) {
				app.selected.instr_addr = instr.instr_addr;
				app.selected.mem_region_id = UINT64_MAX;
			}

			ImGui::TableNextColumn();
			ImGui::Text("0x%016lx", instr.instr_addr);
			ImGui::TableNextColumn();
			ImGui::Text("%s", instr.opcode.c_str());
			ImGui::TableNextColumn();
			ImGui::Text("%d", instr.count);
			ImGui::TableNextColumn();
			ImGui::Text("0x%016lx", instr.min);
			ImGui::TableNextColumn();
			ImGui::Text("0x%016lx", instr.max);
			ImGui::TableNextColumn();
			if (instr.mem_region_id != UINT64_MAX) {
				ImGui::Text("%d", instr.mem_region_id);
			}
		}
		ImGui::EndTable();
	}
}

void OffsetSizeAnalysis::renderGui(const char* title, bool* open) {
	if (!*open) return;

	ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

	ImGui::SetNextWindowSize(ImVec2{700, 400}, ImGuiCond_FirstUseEver);

	if (ImGui::Begin(title, open)) {
		int i = 0;
		for (auto& info : this->regionInfos) {
			ImGui::Text("Region %d", i++);
			float regionsHeight = max(ImGui::GetContentRegionAvail().y, 500);
			if (ImGui::BeginTable("Region", 2, flags, ImVec2(0.0f, regionsHeight))) {
				ImGui::TableSetupScrollFreeze(0, 1);
				ImGui::TableSetupColumn("Offset", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_None);
				ImGui::TableHeadersRow();

				for (auto& pair : info.counts) {
					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::Text("%lld", pair.first);
					ImGui::TableNextColumn();
					ImGui::Text("%llu", pair.second);
				}
				ImGui::EndTable();
			}
		}
	}
	ImGui::End();
}

void InstructionBasedSizeAnalysis::renderGui(const char* title, bool* open) {
	if (!*open) return;

	ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

	ImGui::SetNextWindowSize(ImVec2{700, 400}, ImGuiCond_FirstUseEver);

	if (ImGui::Begin(title, open)) {
		float estimatesHeight = max(ImGui::GetContentRegionAvail().y, 500);
		if (ImGui::BeginTable("Estimates", 3, flags, ImVec2(0.0f, estimatesHeight))) {
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableSetupColumn("Inst. Index", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("Region", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_None);
			ImGui::TableHeadersRow();

			for (auto& info : this->instrInfos) {
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				ImGui::Text("%llu", info.instr_index);
				ImGui::TableNextColumn();
				ImGui::Text("%llu", info.mem_region_id);
				ImGui::TableNextColumn();
				ImGui::Text("%llu", info.estimate);
			}
			ImGui::EndTable();
		}

		float regionsHeight = max(ImGui::GetContentRegionAvail().y, 500);
		if (ImGui::BeginTable("Regions", 3, flags, ImVec2(0.0f, regionsHeight))) {
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableSetupColumn("Region", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("Min", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_None);
			ImGui::TableHeadersRow();

			for (auto& info : this->regionInfos) {
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				ImGui::Text("%llu", info.mem_region_id);
				ImGui::TableNextColumn();
				ImGui::Text("%llu", info.min);
				ImGui::TableNextColumn();
				ImGui::Text("%llu", info.max);
			}
			ImGui::EndTable();
		}
	}
	ImGui::End();
}

void LinearAccessAnalysis::renderGui(const char* title, bool* open) {
	if (!*open) return;

	ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

	ImGui::SetNextWindowSize(ImVec2{700, 400}, ImGuiCond_FirstUseEver);

	if (ImGui::Begin(title, open)) {
		ImGui::Text("Region: %llu", this->region->mem_region_id);
		ImGui::Text("Objects: %llu * %llu bytes", this->region->object_count, this->region->object_size);

		if (ImGui::BeginTable("Stats", 3, flags)) {
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("Accessed", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("Linear", ImGuiTableColumnFlags_None);
			ImGui::TableHeadersRow();

			ImGuiListClipper clipper;
			clipper.Begin(static_cast<int>(this->flags.size()));
			while (clipper.Step()) {
				for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
					uint8_t flags = this->flags[row];
					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::Text("%d", row);
					ImGui::TableNextColumn();
					ImGui::Text("%s", (flags & ACCESSED) ? "x" : "");
					ImGui::TableNextColumn();
					ImGui::Text("%s", (flags & LINEAR) ? "x" : "");
				}
			}
			ImGui::EndTable();
		}
	}
	ImGui::End();
}

void ConsecutiveAccessAnalysis::renderGui(const char* title, bool* open) {
	if (!*open) return;

	ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

	ImGui::SetNextWindowSize(ImVec2{700, 400}, ImGuiCond_FirstUseEver);

	if (ImGui::Begin(title, open)) {
		static int index = 0;
		ImGui::InputInt("index", &index);
		if (index >= static_cast<int64_t>(this->region->object_count)) index = static_cast<int>(this->region->object_count) - 1;
		if (index < 0) index = 0;

		if (ImGui::BeginTable("Successors", 2, flags)) {
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_None);
			ImGui::TableHeadersRow();

			for (int i = 0; i < this->region->object_count; i++) {
				auto count = this->matrix[index * this->region->object_count + i];
				if (count != 0) {
					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::Text("%d", i);
					ImGui::TableNextColumn();
					ImGui::Text("%d", count);
				}
			}
			ImGui::EndTable();
		}
	}
	ImGui::End();
}

void StackAnalysis::renderGui(const char* title, bool* open) {
	if (!*open) return;

	ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

	ImGui::SetNextWindowSize(ImVec2{700, 400}, ImGuiCond_FirstUseEver);

	if (ImGui::Begin(title, open)) {
		static int index = 0;
		ImGui::InputInt("index", &index);
		if (index >= static_cast<int64_t>(this->region->object_count)) index = static_cast<int>(this->region->object_count) - 1;
		if (index < 0) index = 0;

		if (ImGui::BeginTable("Successors", 2, flags)) {
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_None);
			ImGui::TableHeadersRow();

			for (int i = 0; i < this->region->object_count; i++) {
				auto count = this->matrix[index * this->region->object_count + i];
				if (count != 0) {
					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::Text("%d", i);
					ImGui::TableNextColumn();
					ImGui::Text("%d", count);
				}
			}
			ImGui::EndTable();
		}
	}
	ImGui::End();
}

void RegionLinkAnalysis::renderGui(const char* title, bool* open) {
	if (!*open) return;

	ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

	ImGui::SetNextWindowSize(ImVec2{700, 400}, ImGuiCond_FirstUseEver);

	if (ImGui::Begin(title, open)) {
		if (ImGui::BeginTable("Links", 2, flags)) {
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableSetupColumn("Index A", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("Index B", ImGuiTableColumnFlags_None);
			ImGui::TableHeadersRow();

			for (auto& pair : this->links) {
				char buffer[256];
				int pos = 0;

				for (auto target : pair.second) {
					const char* format = (pos == 0) ? "%d" : ", %d";
					int written = snprintf(buffer+pos, sizeof(buffer)-pos, format, target);
					pos += (written < 0) ? 0 : written;
					if (written < 0 || pos >= sizeof(buffer)) break;
				}

				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				ImGui::Text("%d", pair.first);
				ImGui::TableNextColumn();
				ImGui::Text("%s", buffer);
			}

			ImGui::EndTable();
		}
	}
	ImGui::End();
}

void CaaDistributionAnalysis::renderGui(const char* title, bool* open) {
	if (!*open) return;

	ImGui::SetNextWindowSize(ImVec2{700, 400}, ImGuiCond_FirstUseEver);

	if (ImGui::Begin(title, open)) {
		if (ImPlot::BeginPlot("Distribution")) {
			ImPlot::PlotBars("My Bar Plot", buckets, num_buckets);
			ImPlot::EndPlot();
		}
	}
	ImGui::End();
}

bool BeginMainStatusBar() {
	ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar;
	bool open = ImGui::BeginViewportSideBar("##MainStatusBar", ImGui::GetMainViewport(), ImGuiDir_Down, ImGui::GetFrameHeight(), flags);
	if (open) {
		ImGui::BeginMenuBar();
	} else {
		ImGui::End();
	}
	return open;
}

void EndMainStatusBar() {
	ImGui::EndMenuBar();
	ImGui::End();
}

void appRenderGui(GLFWwindow* window, float delta) {
	// Render gui.

	if (app.workspace && app.selected.launch_id >= app.workspace->trace->header.launch_info_count) {
		app.selected.launch_id = 0;
	}

	if (ImGui::BeginMainMenuBar()) {
		if (ImGui::BeginMenu("View")) {
			if (ImGui::MenuItem("Instruction Based Size Analysis", "", app.ibsa, true)) {
				app.ibsa = !app.ibsa;
			}
			if (ImGui::MenuItem("Consecutive Access Analysis", "", app.caa, true)) {
				app.caa = !app.caa;
			}
			if (ImGui::MenuItem("Stack Analysis", "", app.sa, true)) {
				app.sa = !app.sa;
			}
			if (ImGui::MenuItem("Region Link Analysis - Nodes - Indices", "", app.index_rla, true)) {
				app.index_rla = !app.index_rla;
			}
			if (ImGui::MenuItem("Region Link Analysis - Nodes - Bounds", "", app.bounds_rla, true)) {
				app.bounds_rla = !app.bounds_rla;
			}
			if (ImGui::MenuItem("Linear Access Analysis - Indices", "", app.index_laa, true)) {
				app.index_laa = !app.index_laa;
			}
			if (ImGui::MenuItem("CAA Distribution Analysis", "", app.caada, true)) {
				app.caada = !app.caada;
			}
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("Help")) {
			if (ImGui::MenuItem("ImGUI Demo", "", app.demo, true)) {
				app.demo = !app.demo;
			}
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	if (BeginMainStatusBar()) {
		ImGui::Text(app.status.c_str());
		EndMainStatusBar();
	}

	ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

	ImGui::SetNextWindowSize(ImVec2{700, 400}, ImGuiCond_FirstUseEver);

	if (ImGui::Begin("Trace")) {
		if (app.workspace) {
			app.workspace->trace->renderGuiInWindow();
		} else {
			ImGui::Text("Drag and drop a file over this window to open it.");
		}
	}
	ImGui::End();

	app.grid.renderGui(app.workspace.get(), app.selected);
	app.vis.renderGui(app.workspace.get(), app.selected);

	if (app.workspace) {
		AnalysisSet& as = app.workspace->analysis[app.selected.launch_id];
		as.osa.renderGui("Offset Size Analysis", &app.osa);
		as.ibsa.renderGui("Instruction Based Size Analysis", &app.ibsa);
		as.caa.renderGui("Consecutive Access Analysis", &app.caa);
		as.sa.renderGui("Stack Analysis", &app.sa);
		as.index_rla.renderGui("Region Link Analysis - Nodes - Indices", &app.index_rla);
		as.bounds_rla.renderGui("Region Link Analysis - Nodes - Bounds", &app.bounds_rla);
		as.index_laa.renderGui("Linear Access Analysis", &app.index_laa);
		as.caada.renderGui("CAA Distribution Analysis", &app.caada);
	}

	if (app.demo) {
		ImGui::ShowDemoWindow(&app.demo);
	}
}
