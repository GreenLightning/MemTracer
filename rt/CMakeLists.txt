cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

option(MEMINF_ENABLED "enable meminf library" ON)

enable_language(CUDA)
add_subdirectory(image)

add_executable(rt bvh.cc config.cc mesh.cc trace.cu)
target_compile_definitions(rt PUBLIC _USE_MATH_DEFINES) # for M_PI on Windows
target_include_directories(rt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/lib/include)
target_link_libraries(rt image)
if(MEMINF_ENABLED)
	target_compile_definitions(rt PUBLIC MEMINF_ENABLED)
	target_link_libraries(rt meminf)
endif()
