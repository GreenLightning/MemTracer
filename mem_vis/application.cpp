#include <cstdio>
#include <cstdlib>
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

struct Instruction {
	uint64_t    addr;
	const char* opcode;
	uint64_t    start;
	uint64_t    count;
};

struct Trace {
	std::string filename;
	mio::mmap_source mmap;
	header_t header = {};
	std::vector<Instruction> instructions;

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

		if (header.version != 1) {
			return "file version is not supported";
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

		Instruction* last = nullptr;
		for (int i = 0; i < header.mem_access_count; i++) {
			mem_access_t* ma = (mem_access_t*) &mmap[header.mem_access_offset + i * header.mem_access_size];
			if (ma->cta_id_x != 1 || ma->cta_id_y != 1 || ma->cta_id_z != 0) continue;
			if (last == nullptr || ma->instr_addr != last->addr) {
				Instruction instr;
				instr.addr = ma->instr_addr;
				instr.opcode = nullptr;
				instr.start = i;
				instr.count = 1;

				for (int i = 0; i < header.addr_info_count; i++) {
					addr_info_t* info = (addr_info_t*) &mmap[header.addr_info_offset + i * header.addr_info_size];
					if (info->addr == instr.addr) {
						instr.opcode = (const char*) &mmap[header.opcode_offset + info->opcode_offset];
						break;
					}
				}

				instructions.push_back(instr);
				last = &instructions[instructions.size()-1];
			} else {
				last->count++;
			}
		}

		return "";
	}

};

struct Application {
	int width, height;
	bool demo = false;

	std::unique_ptr<Trace> trace;
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

	if (ImGui::Begin("Trace")) {
		if (app.trace) {

			ImGui::Text("Filename: %s", app.trace->filename.c_str());

			meow_u128 hash;
			memcpy(&hash, &app.trace->header.hash, 128 / 8);
			ImGui::Text("Hash: %08X-%08X-%08X-%08X", MeowU32From(hash, 3), MeowU32From(hash, 2), MeowU32From(hash, 1), MeowU32From(hash, 0));

			ImGui::Text("Instructions");

			ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

			if (ImGui::BeginTable("Instructions", 4, flags, ImVec2(0.0f, ImGui::GetContentRegionAvail().y))) {
				ImGui::TableSetupScrollFreeze(0, 1);
				ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("IP", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Opcode", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_None);
				ImGui::TableHeadersRow();

				ImGuiListClipper clipper;
				clipper.Begin(app.trace->instructions.size());
				while (clipper.Step()) {
					for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
						Instruction& instr = app.trace->instructions[row];
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::Text("%d", row);
						ImGui::TableSetColumnIndex(1);
						ImGui::Text("0x%016lx", instr.addr);
						ImGui::TableSetColumnIndex(2);
						ImGui::Text("%s", instr.opcode);
						ImGui::TableSetColumnIndex(3);
						ImGui::Text("%d", instr.count);
					}
				}
				ImGui::EndTable();
			}

		} else {
			ImGui::Text("Drag and drop a file over this window to open it.");
		}
	}
	ImGui::End();

	if (app.demo) {
		ImGui::ShowDemoWindow(&app.demo);
	}
}
