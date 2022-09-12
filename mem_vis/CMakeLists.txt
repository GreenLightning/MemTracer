cmake_minimum_required(VERSION 3.10)

# Statically link with the MSVC runtime library.
# This setting is automatically propagated into all subdirectories.
# NOTE: This effectively overrides USE_MSVC_RUNTIME_LIBRARY_DLL from GLFW,
# so we set it to OFF as well.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(USE_MSVC_RUNTIME_LIBRARY_DLL OFF CACHE BOOL "" FORCE)

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

	"lib/implot/implot.cpp"
	"lib/implot/implot_demo.cpp"
	"lib/implot/implot_items.cpp"
)

target_include_directories(mem_vis PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_include_directories(mem_vis PRIVATE "include")
target_include_directories(mem_vis PRIVATE "lib/imgui")
target_include_directories(mem_vis PRIVATE "lib/implot")

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	# Set Working Directory in Visual Studio.
	set_property(TARGET mem_vis PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
	target_compile_definitions(mem_vis PRIVATE "_CRT_SECURE_NO_WARNINGS")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1 -maes")
	target_link_libraries(mem_vis "dl")
endif()

# OpenGL.
cmake_policy(SET CMP0072 NEW)
find_package(OpenGL REQUIRED)
target_include_directories(mem_vis PRIVATE ${OpenGL_INCLUDE_DIR})
target_link_libraries(mem_vis ${OPENGL_LIBRARIES})

# GLAD.
add_subdirectory("lib/glad")
target_link_libraries(mem_vis glad)

# GLFW.
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
add_subdirectory("lib/glfw")
target_link_libraries(mem_vis glfw)
