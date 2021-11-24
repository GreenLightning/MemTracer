cmake_minimum_required(VERSION 3.10)

project(MemTracer)

add_subdirectory(${CMAKE_SOURCE_DIR}/meminf)
add_subdirectory(${CMAKE_SOURCE_DIR}/mem_trace)
add_subdirectory(${CMAKE_SOURCE_DIR}/rt)
