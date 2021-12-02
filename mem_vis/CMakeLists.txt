cmake_minimum_required(VERSION 3.10)

add_executable(mem_vis
	"main.cpp"
	"gl_utils.cpp"
	"application.cpp"

	"lib/imgui/imgui.cpp"
	"lib/imgui/imgui_demo.cpp"
	"lib/imgui/imgui_widgets.cpp"
	"lib/imgui/imgui_draw.cpp"
	"lib/imgui/imgui_tables.cpp"
	"lib/imgui/imgui_impl_glfw.cpp"
	"lib/imgui/imgui_impl_opengl3.cpp"
)

target_include_directories(mem_vis PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_include_directories(mem_vis PRIVATE "lib/imgui")

# Set Working Directory in Visual Studio.
set_property(TARGET mem_vis PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# OpenGL.
find_package(OpenGL REQUIRED)
target_include_directories(mem_vis PRIVATE ${OpenGL_INCLUDE_DIR})
target_link_libraries(mem_vis ${OPENGL_LIBRARIES})

# GLFW and GLEW.
set(USE_PREBUILT_LIBRARIES_DEFAULT OFF)
if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	set(USE_PREBUILT_LIBRARIES_DEFAULT ON)
endif()
option(USE_PREBUILT_LIBRARIES "Use a pre-built version of GLFW and GLEW" ${USE_PREBUILT_LIBRARIES_DEFAULT})

if(USE_PREBUILT_LIBRARIES)

	target_include_directories(mem_vis PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/lib/include")
	target_compile_definitions(mem_vis PRIVATE GLEW_STATIC)

	if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
		set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MT")
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
		target_link_libraries(mem_vis 
			debug "${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/glfw3d.lib"
			optimized "${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/glfw3.lib"
		)
		target_link_libraries(mem_vis 
			debug "${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/glew32sd.lib"
			optimized "${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/glew32s.lib"
		)
	else()
		message(SEND_ERROR "Pre-built libraries only available for Windows.")
	endif()

else(USE_PREBUILT_LIBRARIES)

	find_package(glfw3 REQUIRED)
	target_include_directories(mem_vis PRIVATE ${GLFW_INCLUDE_DIR})
	target_link_libraries(mem_vis glfw)

	find_package(GLEW REQUIRED)
	target_include_directories(mem_vis PRIVATE ${GLEW_INCLUDE_DIRS})
	target_link_libraries(mem_vis ${GLEW_LIBRARIES})

endif(USE_PREBUILT_LIBRARIES)
