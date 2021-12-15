cmake_minimum_required(VERSION 3.10)

project(MemTracer)

option(BUILD_MEM_VIS "build mem_vis program" OFF)

add_subdirectory(${CMAKE_SOURCE_DIR}/mem_trace)

if(BUILD_MEM_VIS)
	add_subdirectory(${CMAKE_SOURCE_DIR}/mem_vis)
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/rt)
