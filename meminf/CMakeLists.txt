cmake_minimum_required(VERSION 3.10)

add_library(meminf SHARED meminf_default.cc)
add_library(meminf_impl SHARED meminf_impl.cc)
target_include_directories(meminf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(meminf_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
