cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CcTrace CUDA CXX)
add_subdirectory(image)

add_subdirectory(${CMAKE_SOURCE_DIR}/nvbit/mem_trace)

add_library(meminf SHARED meminf_default.cc)
add_executable(rt trace.cu bvh.cc reader/ply.cc)
target_include_directories(rt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(rt PUBLIC cxx_std_11)
target_link_libraries(rt image meminf)
