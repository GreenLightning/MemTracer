cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

enable_language(CUDA)

add_executable(obj_size main.cu)
target_include_directories(obj_size PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
