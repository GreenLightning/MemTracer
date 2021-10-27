cmake_minimum_required(VERSION 3.10)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

enable_language(CUDA)

project(NVBITMemTrace)

find_package(NVBit REQUIRED)

add_library(mem_trace MODULE mem_trace.cu inject_funcs.cu meminf.cc)

set_source_files_properties(inject_funcs.cu PROPERTIES COMPILE_FLAGS " -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions -arch=sm_35")
set_source_files_properties(mem_trace.cu PROPERTIES COMPILE_FLAGS " -dc -arch=sm_35")

target_include_directories(mem_trace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(mem_trace NVBit cuda)
