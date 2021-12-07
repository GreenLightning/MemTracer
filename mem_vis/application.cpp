#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>

#include <defer.hpp>
#include <mio.hpp>

#include "meow_hash_x64_aesni.h"

#include "common.h"
#include "gl_utils.hpp"

#include "application.hpp"

struct TraceInstruction {
	uint64_t    instr_addr;
	const char* opcode;
	uint64_t    min = UINT64_MAX;
	uint64_t    max = 0;
};

struct Trace {
	std::string filename;
	mio::mmap_source mmap;
	header_t header = {};
	std::map<uint64_t, TraceInstruction> instructionsByAddr;
	std::vector<TraceInstruction> instructions;

	~Trace() {
		mmap.unmap();
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

		if (header.version != 2) {
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

		// TODO: Validate other header fields.

		for (int i = 0; i < header.mem_access_count; i++) {
			mem_access_t* ma = (mem_access_t*) &mmap[header.mem_access_offset + i * header.mem_access_size];
			auto count = instructionsByAddr.count(ma->instr_addr);
			TraceInstruction& instr = instructionsByAddr[ma->instr_addr];
			if (count == 0) {
				instr.instr_addr = ma->instr_addr;
				instr.opcode = nullptr;
				for (int i = 0; i < header.addr_info_count; i++) {
					addr_info_t* info = (addr_info_t*) &mmap[header.addr_info_offset + i * header.addr_info_size];
					if (info->addr == instr.instr_addr) {
						instr.opcode = (const char*) &mmap[header.opcode_offset + info->opcode_offset];
						break;
					}
				}
			}
			for (int i = 0; i < 32; i++) {
				uint64_t addr = ma->addrs[i];
				if (addr != 0) {
					if (addr < instr.min) instr.min = addr;
					if (addr > instr.max) instr.max = addr;
				}
			}
		}

		for (auto pair : instructionsByAddr) {
			instructions.push_back(pair.second);
		}

		return "";
	}
};

struct GridInstruction {
	uint64_t    instr_addr;
	const char* opcode;
	uint64_t    addr;
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
				if (ma->grid_launch_id != grid_launch_id || ma->block_idx_x != targetBlockX || ma->block_idx_y != targetBlockY || ma->block_idx_z != 0 || ma->local_warp_id != targetWarpIndex) continue;
				
				GridInstruction instr;
				instr.instr_addr = ma->instr_addr;
				instr.opcode = nullptr;
				for (int i = 0; i < trace->header.addr_info_count; i++) {
					addr_info_t* info = (addr_info_t*) &trace->mmap[trace->header.addr_info_offset + i * trace->header.addr_info_size];
					if (info->addr == instr.instr_addr) {
						instr.opcode = (const char*) &trace->mmap[trace->header.opcode_offset + info->opcode_offset];
						break;
					}
				}
				instr.addr = ma->addrs[targetAccessIndex];
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
				ImGui::TableSetupColumn("Start", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("End", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_None);
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
						ImGui::Text("0x%016lx", region->start);
						ImGui::TableNextColumn();
						ImGui::Text("0x%016lx", region->start + region->size);
						ImGui::TableNextColumn();
						ImGui::Text("0x%016lx", region->size);
						ImGui::TableNextColumn();
						ImGui::Text("%ld", region->size);
						ImGui::TableNextColumn();
						ImGui::Text("%d", region->description);
					}
				}
				ImGui::EndTable();
			}

			float instructionsHeight = max(ImGui::GetContentRegionAvail().y, 500);
			if (ImGui::BeginTable("Instructions", 5, flags, ImVec2(0.0f, instructionsHeight))) {
				ImGui::TableSetupScrollFreeze(0, 1);
				ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("IP", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Opcode", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Min", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_None);
				ImGui::TableHeadersRow();

				ImGuiListClipper clipper;
				clipper.Begin(trace->instructions.size());
				while (clipper.Step()) {
					for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
						TraceInstruction& instr = trace->instructions[row];
						ImGui::TableNextRow();
						ImGui::TableNextColumn();
						ImGui::Text("%d", row);
						ImGui::TableNextColumn();
						ImGui::Text("0x%016lx", instr.instr_addr);
						ImGui::TableNextColumn();
						ImGui::Text("%s", instr.opcode);
						ImGui::TableNextColumn();
						ImGui::Text("0x%016lx", instr.min);
						ImGui::TableNextColumn();
						ImGui::Text("0x%016lx", instr.max);
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

		if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
			if (x >= 0 && x < app.grid.imageWidth && y >= 0 && y < app.grid.imageHeight) {
				app.grid.targetX = x / app.grid.scale;
				app.grid.targetY = y / app.grid.scale;
			}
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
		if (ImGui::BeginTable("Instructions", 4, flags, ImVec2(0.0f, instructionsHeight))) {
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("IP", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("Opcode", ImGuiTableColumnFlags_None);
			ImGui::TableSetupColumn("Address", ImGuiTableColumnFlags_None);
			ImGui::TableHeadersRow();

			ImGuiListClipper clipper;
			clipper.Begin(app.grid.instructions.size());
			while (clipper.Step()) {
				for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
					GridInstruction& instr = app.grid.instructions[row];
					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::Text("%d", row);
					ImGui::TableNextColumn();
					ImGui::Text("0x%016lx", instr.instr_addr);
					ImGui::TableNextColumn();
					ImGui::Text("%s", instr.opcode);
					ImGui::TableNextColumn();
					ImGui::Text("0x%016lx", instr.addr);
				}
			}
			ImGui::EndTable();
		}
	}
	ImGui::End();

	if (app.demo) {
		ImGui::ShowDemoWindow(&app.demo);
	}
}
