cmake_minimum_required(VERSION 3.28)
project(CUDA LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_library(precompiled_headers OBJECT pch.cu)
set_target_properties(precompiled_headers PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(CUDA main.cu)

set(PCH_FILE pch.cuh)
add_library(pch STATIC ${PCH_FILE} pch.cu)
target_include_directories(pch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(CUDA PRIVATE pch)
target_precompile_headers(CUDA PRIVATE ${PCH_FILE})
