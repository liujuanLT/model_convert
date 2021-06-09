cmake_minimum_required(VERSION 2.6)

project(trtcali)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
set(CMAKE_BUILD_TYPE Debug)
find_package(CUDA)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/local/TensorRT/include)
link_directories(/usr/local/TensorRT/lib)

cuda_add_library(trtcaliplugin SHARED ${PROJECT_SOURCE_DIR}/trtcali.cu)
target_link_libraries(trtcaliplugin nvinfer cudart)
# opencv
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(trtcali ${PROJECT_SOURCE_DIR}/trtcali.cpp)
target_link_libraries(trtcali nvinfer)
target_link_libraries(trtcali cudart)
target_link_libraries(trtcali trtcaliplugin)
target_link_libraries(trtcali ${OpenCV_LIBS})

add_definitions(-O2 -pthread)