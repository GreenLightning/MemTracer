cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
enable_language(CUDA)
add_subdirectory(image)

add_executable(rt trace.cu bvh.cc reader/ply.cc)
target_include_directories(rt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rt image meminf)
