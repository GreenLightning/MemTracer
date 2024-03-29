cmake_minimum_required(VERSION 3.10)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

enable_language(CUDA)

find_package(NVBit REQUIRED)

add_library(mem_trace_inject_funcs OBJECT inject_funcs.cu)
add_library(mem_trace MODULE mem_trace.cu)

set_property(TARGET mem_trace PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(inject_funcs.cu PROPERTIES COMPILE_FLAGS " -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions -arch=sm_52")

# For meow hash.
set_source_files_properties(mem_trace.cu PROPERTIES COMPILE_FLAGS "-Xcompiler \"-mavx2 -maes\"")

target_link_libraries(mem_trace_inject_funcs NVBit)
target_link_libraries(mem_trace NVBit cuda $<TARGET_OBJECTS:mem_trace_inject_funcs>)
target_include_directories(mem_trace_inject_funcs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_include_directories(mem_trace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
