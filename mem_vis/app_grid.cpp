struct GridInstruction {
	uint64_t    instr_addr;
	std::string opcode;
	uint64_t    addr;
	uint64_t    mem_region_id;
	int64_t     intra_region_offset;
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

			std::unordered_map<uint64_t, uint64_t> last_addresses;

			for (uint64_t i = trace->begin_index(grid_launch_id), n = trace->end_index(grid_launch_id); i < n; i++) {
				mem_access_t* ma = &trace->accesses[i];
				if (ma->block_idx_x != targetBlockX || ma->block_idx_y != targetBlockY || ma->block_idx_z != 0 || ma->local_warp_id != targetWarpIndex || ma->addrs[targetAccessIndex] == 0) continue;
				
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

						uint64_t last_address = last_addresses[instr.mem_region_id];
						if (last_address) instr.intra_region_offset = int64_t(instr.addr - last_address);
						last_addresses[instr.mem_region_id] = instr.addr;

						uint64_t offset = instr.addr - region->start;
						char buffer[256];
						switch (instr.mem_region_id) {
							case 1: {
								uint64_t nodeIndex = offset / 4;
								snprintf(buffer, sizeof(buffer), "nodes[%llu]", nodeIndex);
								break;
							}
							case 2: {
								uint64_t floatIndex = offset / 4;
								uint64_t aabbIndex = floatIndex / 6;
								uint64_t subIndex = floatIndex % 6;
								if (subIndex < 3) {
									snprintf(buffer, sizeof(buffer), "bounds[%llu].min[%llu]", aabbIndex, subIndex);
								} else {
									snprintf(buffer, sizeof(buffer), "bounds[%llu].max[%llu]", aabbIndex, subIndex - 3);
								}
								break;
							}
							case 3: {
								uint64_t intIndex = offset / 4;
								uint64_t faceIndex = intIndex / 3;
								uint64_t subIndex = intIndex % 3;
								snprintf(buffer, sizeof(buffer), "faces[%llu].indices[%llu]", faceIndex, subIndex);
								break;
							}
							default: {
								snprintf(buffer, sizeof(buffer), "%llu bytes", offset);
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
			counts.resize(0);
			counts.resize(heightInThreads * widthInThreads);
			maxCount = 0;
			if (trace) {
				for (uint64_t i = trace->begin_index(grid_launch_id), n = trace->end_index(grid_launch_id); i < n; i++) {
					mem_access_t* ma = &trace->accesses[i];
					if (ma->block_idx_z != 0) continue;

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
	
	void renderGui(Workspace* workspace, Selection& selected) {
		ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

		Trace* trace = workspace ? workspace->trace.get() : nullptr;

		ImGui::SetNextWindowSize(ImVec2{700, 400}, ImGuiCond_FirstUseEver);

		if (ImGui::Begin("Grid")) {
			ImGui::InputInt("x", &this->targetX);
			ImGui::InputInt("y", &this->targetY);

			ImVec2 mousePos = ImGui::GetMousePos();
			ImVec2 cursorPos = ImGui::GetCursorScreenPos();
			float x = mousePos.x - cursorPos.x;
			float y = mousePos.y - cursorPos.y;

			ImRect bb(cursorPos, ImVec2(cursorPos.x + this->imageWidth, cursorPos.y + this->imageHeight));
			ImGuiID id = ImGui::GetID("Texture");
			if (ImGui::ButtonBehavior(bb, id, nullptr, nullptr, ImGuiButtonFlags_PressedOnClickRelease)) {
				this->targetX = static_cast<int>(x / this->scale);
				this->targetY = static_cast<int>(y / this->scale);
			}

			this->update(trace, selected.launch_id);

			ImGui::Image((void*)(intptr_t)this->texture, ImVec2(static_cast<float>(this->imageWidth), static_cast<float>(this->imageHeight)));

			if (x >= 0 && x < this->imageWidth && y >= 0 && y < this->imageHeight) {
				ImGui::Text("%d, %d", x, y);
			} else {
				ImGui::Text("-");
			}
		}
		ImGui::End();

		ImGui::SetNextWindowSize(ImVec2{700, 400}, ImGuiCond_FirstUseEver);

		if (ImGui::Begin("Grid Instructions")) {
			ImGui::Text("Count: %d", this->instructions.size());

			float instructionsHeight = std::max(ImGui::GetContentRegionAvail().y, 500.0f);
			if (ImGui::BeginTable("Instructions", 7, flags, ImVec2(0.0f, instructionsHeight))) {
				ImGui::TableSetupScrollFreeze(0, 1);
				ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("IP", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Opcode", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Address", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Region", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Offset", ImGuiTableColumnFlags_None);
				ImGui::TableSetupColumn("Info", ImGuiTableColumnFlags_None);
				ImGui::TableHeadersRow();

				ImGuiListClipper clipper;
				clipper.Begin(static_cast<int>(this->instructions.size()));
				while (clipper.Step()) {
					for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
						GridInstruction& instr = this->instructions[row];

						ImGui::TableNextRow();
						ImGui::TableNextColumn();
						char label[32];
						snprintf(label, sizeof(label), "%d", row);
						ImGuiSelectableFlags selectable_flags = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap;
						bool isSelected = (selected.instr_addr == instr.instr_addr) ||
							(selected.instr_addr == UINT64_MAX && selected.mem_region_id != UINT64_MAX && selected.mem_region_id == instr.mem_region_id);
						if (ImGui::Selectable(label, isSelected, selectable_flags)) {
							selected.instr_addr = instr.instr_addr;
							selected.mem_region_id = instr.mem_region_id;
						}

						ImGui::TableNextColumn();
						if (row > 0 && instr.instr_addr <= this->instructions[row-1].instr_addr) {
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
							ImGui::Text("%lld", instr.mem_region_id);
						}
						ImGui::TableNextColumn();
						if (instr.mem_region_id != UINT64_MAX) {
							ImGui::Text("%lld", instr.intra_region_offset);
						}
						ImGui::TableNextColumn();
						ImGui::Text("%s", instr.info.c_str());
					}
				}
				ImGui::EndTable();
			}
		}
		ImGui::End();
	}
};
