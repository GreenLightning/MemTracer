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

#include <defer.hpp>
#include <mio.hpp>

#include "meow_hash_x64_aesni.h"

#include "common.h"
#include "gl_utils.hpp"

#include "application.hpp"

struct ConsecutiveAccessAnalysis {
	uint64_t grid_launch_id;
	uint64_t mem_region_id;
	mem_region_t* region;
	uint64_t object_size;
	uint64_t object_count;
	std::vector<uint64_t> matrix;
};

struct RegionLinkAnalysis {
	uint64_t grid_launch_id;
	uint64_t region_id_a;
	uint64_t region_id_b;
	mem_region_t* region_a;
	mem_region_t* region_b;
	uint64_t object_size_a;
	uint64_t object_size_b;
	uint64_t object_count_a;
	uint64_t object_count_b;
	std::map<int64_t, std::set<int64_t>> links;
};

struct TraceInstruction {
	uint64_t    instr_addr;
	std::string opcode;
	uint64_t    count = 0;
	uint64_t    min = UINT64_MAX;
	uint64_t    max = 0;
	uint64_t    mem_region_id = UINT64_MAX;
};

struct Trace {
	std::string filename;
	mio::mmap_source mmap;
	header_t header = {};
	std::vector<std::string> strings;
	std::unordered_map<uint64_t, TraceInstruction> instructionsByAddr;
	std::vector<TraceInstruction> instructions;
	ConsecutiveAccessAnalysis caa;
	RegionLinkAnalysis rla;

	~Trace() {
		mmap.unmap();
	}

	mem_region_t* find_mem_region(uint64_t grid_launch_id, uint64_t mem_region_id) {
		for (int i = 0; i < header.mem_region_count; i++) {
			mem_region_t* region = (mem_region_t*) &mmap[header.mem_region_offset + i * header.mem_region_size];
			if (region->grid_launch_id == grid_launch_id && region->mem_region_id == mem_region_id) {
				return region;
			}
		}
		return nullptr;
	}

	std::string load(std::string filename) {
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

		// TRACE INSTRUCTIONS

		for (int i = 0; i < header.mem_access_count; i++) {
			mem_access_t* ma = (mem_access_t*) &mmap[header.mem_access_offset + i * header.mem_access_size];
			auto count = instructionsByAddr.count(ma->instr_addr);
			TraceInstruction& instr = instructionsByAddr[ma->instr_addr];
			if (count == 0) {
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

		for (auto& pair : instructionsByAddr) {
			TraceInstruction& instr = pair.second;
			for (int i = 0; i < header.mem_region_count; i++) {
				mem_region_t* region = (mem_region_t*) &mmap[header.mem_region_offset + i * header.mem_region_size];
				// TODO: This ignores the grid_launch_id.
				if (region->start <= instr.min && instr.max < region->start + region->size) {
					instr.mem_region_id = region->mem_region_id;
					break;
				}
			}
		}

		for (auto& pair : instructionsByAddr) {
			instructions.push_back(pair.second);
		}

		std::sort(instructions.begin(), instructions.end(), [] (const TraceInstruction& a, const TraceInstruction& b) {
			return a.instr_addr < b.instr_addr;
		});

		// SORTING

		auto t0 = std::chrono::high_resolution_clock::now();

		std::vector<mem_access_t> accesses;
		accesses.reserve(header.mem_access_count);
		for (int i = 0; i < header.mem_access_count; i++) {
			mem_access_t* ma = (mem_access_t*) &mmap[header.mem_access_offset + i * header.mem_access_size];
			accesses.push_back(*ma);
		}

		auto t1 = std::chrono::high_resolution_clock::now();

		std::stable_sort(accesses.begin(), accesses.end(), [] (const mem_access_t& a, const mem_access_t& b) {
			if (a.grid_launch_id != b.grid_launch_id) return a.grid_launch_id < b.grid_launch_id;
			if (a.block_idx_z != b.block_idx_z) return a.block_idx_z < b.block_idx_z;
			if (a.block_idx_y != b.block_idx_y) return a.block_idx_y < b.block_idx_y;
			if (a.block_idx_x != b.block_idx_x) return a.block_idx_x < b.block_idx_x;
			if (a.local_warp_id != b.local_warp_id) return a.local_warp_id < b.local_warp_id;
			return false;
		});
		
		auto t2 = std::chrono::high_resolution_clock::now();

		printf("copy: %.9fs\n", std::chrono::duration<double>(t1 - t0).count());
		printf("sort: %.9fs\n", std::chrono::duration<double>(t2 - t1).count());

		// ANALYSIS

		{
			caa.grid_launch_id = 0;
			caa.mem_region_id = 1;
			caa.region = find_mem_region(caa.grid_launch_id, caa.mem_region_id);
			caa.object_size = 4;
			caa.object_count = caa.region->size / caa.object_size;
			caa.matrix.reserve(caa.object_count * caa.object_count);

			int32_t last_block_idx_z = -1;
			int32_t last_block_idx_y = -1;
			int32_t last_block_idx_x = -1;
			int32_t last_local_warp_id = -1;
			int64_t last_access[32];

			int i = 0;
			while (i < accesses.size() && accesses[i].grid_launch_id != caa.grid_launch_id) i++;
			for (; i < accesses.size(); i++) {
				mem_access_t* ma = &accesses[i];
				if (ma->grid_launch_id != caa.grid_launch_id) break;

				if (ma->block_idx_z != last_block_idx_z || ma->block_idx_y != last_block_idx_y || ma->block_idx_x != last_block_idx_x || ma->local_warp_id != last_local_warp_id) {
					last_block_idx_z = ma->block_idx_z;
					last_block_idx_y = ma->block_idx_y;
					last_block_idx_x = ma->block_idx_x;
					last_local_warp_id = ma->local_warp_id;

					for (int i = 0; i < 32; i++) {
						last_access[i] = -1;
					}
				}

				TraceInstruction& instr = instructionsByAddr[ma->instr_addr];
				if (instr.mem_region_id != caa.mem_region_id) continue;

				for (int i = 0; i < 32; i++) {
					int64_t access = (ma->addrs[i] == 0) ? -1 : (ma->addrs[i] - caa.region->start) / caa.object_size;
					if (last_access[i] >= 0 && access >= 0) {
						caa.matrix[last_access[i] * caa.object_count + access]++;
					}
					last_access[i] = access;
				}
			}
		}

		{
			rla.grid_launch_id = 0;
			rla.region_id_a = 1;
			rla.region_id_b = 3;
			rla.region_a = find_mem_region(rla.grid_launch_id, rla.region_id_a);
			rla.region_b = find_mem_region(rla.grid_launch_id, rla.region_id_b);
			rla.object_size_a = 4;
			rla.object_size_b = 12;
			rla.object_count_a = rla.region_a->size / rla.object_size_a;
			rla.object_count_b = rla.region_b->size / rla.object_size_b;

			int32_t last_block_idx_z = -1;
			int32_t last_block_idx_y = -1;
			int32_t last_block_idx_x = -1;
			int32_t last_local_warp_id = -1;
			int64_t last_access[32];

			int i = 0;
			while (i < accesses.size() && accesses[i].grid_launch_id != rla.grid_launch_id) i++;
			for (; i < accesses.size(); i++) {
				mem_access_t* ma = &accesses[i];
				if (ma->grid_launch_id != rla.grid_launch_id) break;

				if (ma->block_idx_z != last_block_idx_z || ma->block_idx_y != last_block_idx_y || ma->block_idx_x != last_block_idx_x || ma->local_warp_id != last_local_warp_id) {
					last_block_idx_z = ma->block_idx_z;
					last_block_idx_y = ma->block_idx_y;
					last_block_idx_x = ma->block_idx_x;
					last_local_warp_id = ma->local_warp_id;

					for (int i = 0; i < 32; i++) {
						last_access[i] = -1;
					}
				}

				TraceInstruction& instr = instructionsByAddr[ma->instr_addr];
				if (instr.mem_region_id == rla.region_id_a) {
					for (int i = 0; i < 32; i++) {
						last_access[i] = (ma->addrs[i] == 0) ? -1 : (ma->addrs[i] - rla.region_a->start) / rla.object_size_a;
					}
				} else if (instr.mem_region_id == rla.region_id_b) {
					for (int i = 0; i < 32; i++) {
						int64_t access = (ma->addrs[i] == 0) ? -1 : (ma->addrs[i] - rla.region_b->start) / rla.object_size_b;
						if (last_access[i] >= 0 && access >= 0) {
							rla.links[last_access[i]].insert(access);
						}
						last_access[i] = -1;
					}
				} else if (instr.mem_region_id != UINT64_MAX) {
					for (int i = 0; i < 32; i++) {
						last_access[i] = -1;
					}
				}
			}
		}

		return "";
	}
};

struct GridInstruction {
	uint64_t    instr_addr;
	std::string opcode;
	uint64_t    addr;
	uint64_t    mem_region_id;
	std::string info;
};

struct Grid {
	const int scale = 5;

	int targetX = 107, targetY = 99;

	Trace* cached_trace = nullptr;
	uint64_t cached_grid_launch_id = UINT64_MAX;
	int cached_target_x = -1, cached_target_y = -1;

	std::vector<GridInstruction> instructions;
	std::vector<int32_t> counts;
	int32_t maxCount;
	std::vector<uint8_t> image;
	int imageWidth, imageHeight;

	GLuint texture = 0;

	void update(Trace* trace, uint64_t grid_launch_id) {
		if (cached_trace == trace && cached_grid_launch_id == grid_launch_id && cached_target_x == targetX && cached_target_y == targetY) return;

		defer {
			cached_trace = trace;
			cached_grid_launch_id = grid_launch_id;
			cached_target_x = targetX;
			cached_target_y = targetY;
		};

		launch_info_t info = {};
		if (trace) {
			for (int i = 0; i < trace->header.launch_info_count; i++) {
				launch_info_t* current = (launch_info_t*) &trace->mmap[trace->header.launch_info_offset + i * trace->header.launch_info_size];
				if (current->grid_launch_id == grid_launch_id) {
					info = *current;
					break;
				}
			}
		}

		instructions.clear();
		if (trace) {
			int targetBlockX = targetX / info.block_dim_x;
			int targetBlockY = targetY / info.block_dim_y;
			int targetThreadX = targetX % info.block_dim_x;
			int targetThreadY = targetY % info.block_dim_y;
			int targetThreadIndex = targetThreadY * info.block_dim_x + targetThreadX;
			int targetWarpIndex = targetThreadIndex / 32;
			int targetAccessIndex = targetThreadIndex % 32;

			for (int i = 0; i < trace->header.mem_access_count; i++) {
				mem_access_t* ma = (mem_access_t*) &trace->mmap[trace->header.mem_access_offset + i * trace->header.mem_access_size];
				if (ma->grid_launch_id != grid_launch_id || ma->block_idx_x != targetBlockX || ma->block_idx_y != targetBlockY || ma->block_idx_z != 0 || ma->local_warp_id != targetWarpIndex || ma->addrs[targetAccessIndex] == 0) continue;
				
				GridInstruction instr = {};
				instr.instr_addr = ma->instr_addr;
				for (int i = 0; i < trace->header.addr_info_count; i++) {
					addr_info_t* info = (addr_info_t*) &trace->mmap[trace->header.addr_info_offset + i * trace->header.addr_info_size];
					if (info->addr == instr.instr_addr) {
						instr.opcode = trace->strings[info->opcode_string_index];
						break;
					}
				}
				instr.addr = ma->addrs[targetAccessIndex];
				instr.mem_region_id = UINT64_MAX;
				for (int i = 0; i < trace->header.mem_region_count; i++) {
					mem_region_t* region = (mem_region_t*) &trace->mmap[trace->header.mem_region_offset + i * trace->header.mem_region_size];
					if (region->grid_launch_id != grid_launch_id) continue;
					if (region->start <= instr.addr && instr.addr < region->start + region->size) {
						instr.mem_region_id = region->mem_region_id;

						uint64_t offset = instr.addr - region->start;
						char buffer[256];
						switch (instr.mem_region_id) {
							case 1: {
								uint64_t nodeIndex = offset / 4;
								snprintf(buffer, sizeof(buffer), "nodes[%d]", nodeIndex);
								break;
							}
							case 2: {
								uint64_t floatIndex = offset / 4;
								uint64_t aabbIndex = floatIndex / 6;
								uint64_t subIndex = floatIndex % 6;
								if (subIndex < 3) {
									snprintf(buffer, sizeof(buffer), "bounds[%d].min[%d]", aabbIndex, subIndex);
								} else {
									snprintf(buffer, sizeof(buffer), "bounds[%d].max[%d]", aabbIndex, subIndex - 3);
								}
								break;
							}
							case 3: {
								uint64_t intIndex = offset / 4;
								uint64_t faceIndex = intIndex / 3;
								uint64_t subIndex = intIndex % 3;
								snprintf(buffer, sizeof(buffer), "faces[%d].indices[%d]", faceIndex, subIndex);
								break;
							}
							default: {
								snprintf(buffer, sizeof(buffer), "%d bytes", offset);
								break;
							}
						}
						instr.info = std::string(buffer);
						break;
					}
				}
				instructions.push_back(instr);
			}
		}

		int heightInThreads = info.grid_dim_y * info.block_dim_y;
		int widthInThreads = info.grid_dim_x * info.block_dim_x;

		if (trace != cached_trace || grid_launch_id != cached_grid_launch_id) {
			counts.resize(heightInThreads * widthInThreads);
			maxCount = 0;
			if (trace) {
				for (int i = 0; i < trace->header.mem_access_count; i++) {
					mem_access_t* ma = (mem_access_t*) &trace->mmap[trace->header.mem_access_offset + i * trace->header.mem_access_size];
					if (ma->grid_launch_id != grid_launch_id || ma->block_idx_z != 0) continue;

					for (int i = 0; i < 32; i++) {
						uint64_t addr = ma->addrs[i];
						if (addr != 0) {
							int threadIndex = ma->local_warp_id * 32 + i;
							int threadX = ma->block_idx_x * info.block_dim_x + threadIndex % info.block_dim_x;
							int threadY = ma->block_idx_y * info.block_dim_y + threadIndex / info.block_dim_x;
							int count = counts[threadY * widthInThreads + threadX] + 1;
							counts[threadY * widthInThreads + threadX] = count;
							if (count > maxCount) maxCount = count;
						}
					}
				}
			}
		}

		imageHeight = scale * heightInThreads;
		imageWidth = scale * widthInThreads;
		int pitch = imageWidth * 3;
		image.resize(imageHeight * pitch);

		for (int yt = 0; yt < heightInThreads; yt++) {
			for (int xt = 0; xt < widthInThreads; xt++) {
				int c = counts[yt * widthInThreads + xt] * 255 / maxCount;
				uint32_t color = (c << 16) | ((255 - c) << 8) | c; // 0xd9cd2e;
				if (yt == targetY && xt == targetX) color = 0xffff00;

				for (int yp = 0; yp < scale; yp++) {
					for (int xp = 0; xp < scale; xp++) {
						int y = yt * scale + yp;
						int x = xt * scale + xp;
						image[y * pitch + x * 3 + 0] = (color >> 16);
						image[y * pitch + x * 3 + 1] = (color >>  8);
						image[y * pitch + x * 3 + 2] = (color >>  0);
					}
				}
			}
		}

		for (int yt = 0; yt < heightInThreads; yt++) {
			int y = yt * scale;
			for (int x = 0; x < imageWidth; x++) {
				image[y * pitch + x * 3 + 0] = 0x80;
				image[y * pitch + x * 3 + 1] = 0x80;
				image[y * pitch + x * 3 + 2] = 0x80;
			}
		}

		for (int xt = 0; xt < widthInThreads; xt++) {
			int x = xt * scale;
			for (int y = 0; y < imageHeight; y++) {
				image[y * pitch + x * 3 + 0] = 0x80;
				image[y * pitch + x * 3 + 1] = 0x80;
				image[y * pitch + x * 3 + 2] = 0x80;
			}
		}

		for (int yt = 0; yt < heightInThreads; yt += info.block_dim_y) {
			int y = yt * scale;
			for (int x = 0; x < imageWidth; x++) {
				image[y * pitch + x * 3 + 0] = 0x00;
				image[y * pitch + x * 3 + 1] = 0x00;
				image[y * pitch + x * 3 + 2] = 0x00;
			}
		}

		for (int xt = 0; xt < widthInThreads; xt += info.block_dim_x) {
			int x = xt * scale;
			for (int y = 0; y < imageHeight; y++) {
				image[y * pitch + x * 3 + 0] = 0x00;
				image[y * pitch + x * 3 + 1] = 0x00;
				image[y * pitch + x * 3 + 2] = 0x00;
			}
		}

		if (!texture) {
			glGenTextures(1, &texture);
			glBindTexture(GL_TEXTURE_2D, texture);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glBindTexture(GL_TEXTURE_2D, 0);
		}

		glBindTexture(GL_TEXTURE_2D, texture);
		glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageWidth, imageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, image.data());
		glBindTexture(GL_TEXTURE_2D, 0);
	}
};

struct Application {
	int width, height;
	bool demo = false;
	uint64_t selected_instr_addr    = UINT64_MAX;
	uint64_t selected_mem_region_id = UINT64_MAX;

	std::unique_ptr<Trace> trace;
	Grid grid;
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
		app.trace = std::move(trace);
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
		if (error.empty()) {
			app.trace = std::move(trace);
			app.status = "";
		} else {
			app.status = "Failed to load " + trace->filename + ": " + error;			
		}
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

	if (ImGui::BeginMainMenuBar()) {
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

	if (ImGui::Begin("Trace")) {
		if (app.trace) {
			auto& trace = app.trace;

			ImGui::Text("Filename: %s", trace->filename.c_str());

			meow_u128 hash;
			memcpy(&hash, &trace->header.hash, 128 / 8);
			ImGui::Text("Hash: %08X-%08X-%08X-%08X", MeowU32From(hash, 3), MeowU32From(hash, 2), MeowU32From(hash, 1), MeowU32From(hash, 0));

			ImGui::Text("Instructions");

			float infosHeight = min((trace->header.launch_info_count + 2) * ImGui::GetFrameHeight(), 200);
			if (ImGui::BeginTable("Launch Infos", 3, flags, ImVec2(0.0f, infosHeight))) {
				ImGui::TableSetupScrollFreeze(0, 1);
				ImGui::TableSetupColumn("Launch ID", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Grid Size", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Block Size", ImGuiTableColumnFlags_None);
				ImGui::TableHeadersRow();

				ImGuiListClipper clipper;
				clipper.Begin(trace->header.launch_info_count);
				while (clipper.Step()) {
					for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
						launch_info_t* info = (launch_info_t*) &trace->mmap[trace->header.launch_info_offset + row * trace->header.launch_info_size];
						ImGui::TableNextRow();
						ImGui::TableNextColumn();
						ImGui::Text("%d", info->grid_launch_id);
						ImGui::TableNextColumn();
						ImGui::Text("%d,%d,%d", info->grid_dim_x, info->grid_dim_y, info->grid_dim_z);
						ImGui::TableNextColumn();
						ImGui::Text("%d,%d,%d", info->block_dim_x, info->block_dim_y, info->block_dim_z);
					}
				}
				ImGui::EndTable();
			}

			float regionsHeight = min((trace->header.mem_region_count + 2) * ImGui::GetFrameHeight(), 200);
			if (ImGui::BeginTable("Memory Regions", 6, flags, ImVec2(0.0f, regionsHeight))) {
				ImGui::TableSetupScrollFreeze(0, 1);
				ImGui::TableSetupColumn("Launch ID", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Region ID", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Start", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("End", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_None);
				ImGui::TableHeadersRow();

				ImGuiListClipper clipper;
				clipper.Begin(trace->header.mem_region_count);
				while (clipper.Step()) {
					for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
						mem_region_t* region = (mem_region_t*) &trace->mmap[trace->header.mem_region_offset + row * trace->header.mem_region_size];
						
						ImGui::TableNextRow();
						ImGui::TableNextColumn();
						ImGui::Text("%d", region->grid_launch_id);
						ImGui::TableNextColumn();

						char label[32];
						snprintf(label, sizeof(label), "%d", region->mem_region_id);
						ImGuiSelectableFlags selectable_flags = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap;
						if (ImGui::Selectable(label, app.selected_mem_region_id == region->mem_region_id, selectable_flags)) {
							app.selected_instr_addr = UINT64_MAX;
							app.selected_mem_region_id = region->mem_region_id;
						}

						ImGui::TableNextColumn();
						ImGui::Text("0x%016lx", region->start);
						ImGui::TableNextColumn();
						ImGui::Text("0x%016lx", region->start + region->size);
						ImGui::TableNextColumn();
						ImGui::Text("0x%016lx", region->size);
						ImGui::TableNextColumn();
						ImGui::Text("%ld", region->size);
					}
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

				ImGuiListClipper clipper;
				clipper.Begin(trace->instructions.size());
				while (clipper.Step()) {
					for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
						TraceInstruction& instr = trace->instructions[row];

						ImGui::TableNextRow();
						ImGui::TableNextColumn();
						char label[32];
						snprintf(label, sizeof(label), "%d", row);
						ImGuiSelectableFlags selectable_flags = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap;
						if (ImGui::Selectable(label, app.selected_instr_addr == instr.instr_addr, selectable_flags)) {
							app.selected_instr_addr = instr.instr_addr;
							app.selected_mem_region_id = UINT64_MAX;
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
				}
				ImGui::EndTable();
			}

		} else {
			ImGui::Text("Drag and drop a file over this window to open it.");
		}
	}
	ImGui::End();

	if (ImGui::Begin("Grid")) {
		ImGui::InputInt("x", &app.grid.targetX);
		ImGui::InputInt("y", &app.grid.targetY);

		ImVec2 mousePos = ImGui::GetMousePos();
		ImVec2 cursorPos = ImGui::GetCursorScreenPos();
		int x = mousePos.x - cursorPos.x;
		int y = mousePos.y - cursorPos.y;

		ImRect bb(cursorPos, ImVec2(cursorPos.x + app.grid.imageWidth, cursorPos.y + app.grid.imageHeight));
		ImGuiID id = ImGui::GetID("Texture");
		if (ImGui::ButtonBehavior(bb, id, nullptr, nullptr, ImGuiButtonFlags_PressedOnClickRelease)) {
			app.grid.targetX = x / app.grid.scale;
			app.grid.targetY = y / app.grid.scale;
		}

		app.grid.update(app.trace.get(), 0);

		ImGui::Image((void*)(intptr_t)app.grid.texture, ImVec2(app.grid.imageWidth, app.grid.imageHeight));

		if (x >= 0 && x < app.grid.imageWidth && y >= 0 && y < app.grid.imageHeight) {
			ImGui::Text("%d, %d", x, y);
		} else {
			ImGui::Text("-");
		}
	}
	ImGui::End();

	if (ImGui::Begin("Grid Instructions")) {
		ImGui::Text("Count: %d", app.grid.instructions.size());

		float instructionsHeight = max(ImGui::GetContentRegionAvail().y, 500);
		if (ImGui::BeginTable("Instructions", 6, flags, ImVec2(0.0f, instructionsHeight))) {
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("IP", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("Opcode", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("Address", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("Region", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("Offset", ImGuiTableColumnFlags_None);
			ImGui::TableHeadersRow();

			ImGuiListClipper clipper;
			clipper.Begin(app.grid.instructions.size());
			while (clipper.Step()) {
				for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
					GridInstruction& instr = app.grid.instructions[row];

					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					char label[32];
					snprintf(label, sizeof(label), "%d", row);
					ImGuiSelectableFlags selectable_flags = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap;
					bool selected = app.selected_instr_addr == instr.instr_addr ||
						(app.selected_instr_addr == UINT64_MAX && app.selected_mem_region_id != UINT64_MAX && app.selected_mem_region_id == instr.mem_region_id);
					if (ImGui::Selectable(label, selected, selectable_flags)) {
						app.selected_instr_addr = instr.instr_addr;
						app.selected_mem_region_id = instr.mem_region_id;
					}

					ImGui::TableNextColumn();
					if (row > 0 && instr.instr_addr <= app.grid.instructions[row-1].instr_addr) {
						ImGui::TextColored(ImVec4{1, 0.5, 0.5, 1}, "0x%016lx", instr.instr_addr);
					} else {
						ImGui::Text("0x%016lx", instr.instr_addr);
					}
				
					ImGui::TableNextColumn();
					ImGui::Text("%s", instr.opcode.c_str());
					ImGui::TableNextColumn();
					ImGui::Text("0x%016lx", instr.addr);
					ImGui::TableNextColumn();
					if (instr.mem_region_id != UINT64_MAX) {
						ImGui::Text("%d", instr.mem_region_id);
					}
					ImGui::TableNextColumn();
					ImGui::Text("%s", instr.info.c_str());
				}
			}
			ImGui::EndTable();
		}
	}
	ImGui::End();

	if (app.trace) {
		ConsecutiveAccessAnalysis* caa = &app.trace->caa;
		if (ImGui::Begin("Consecutive Access Analysis")) {
			static int index = 0;
			ImGui::InputInt("index", &index);

			if (ImGui::BeginTable("Successors", 2, flags)) {
				ImGui::TableSetupScrollFreeze(0, 1);
				ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_None);
				ImGui::TableHeadersRow();

				for (int i = 0; i < caa->object_count; i++) {
					auto count = caa->matrix[index * caa->object_count + i];
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

		RegionLinkAnalysis* rla = &app.trace->rla;
		if (ImGui::Begin("Region Link Analysis")) {
			if (ImGui::BeginTable("Links", 2, flags)) {
				ImGui::TableSetupScrollFreeze(0, 1);
				ImGui::TableSetupColumn("Index A", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Index B", ImGuiTableColumnFlags_None);
				ImGui::TableHeadersRow();

				for (auto& pair : rla->links) {
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

	if (app.demo) {
		ImGui::ShowDemoWindow(&app.demo);
	}
}
